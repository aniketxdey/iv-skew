#!/usr/bin/env python3
"""
Download end-of-day option-chain data from Dolthub's open mirror of U.S.
listed options (`post-no-preference/options`) and convert it into the
CBOE-wide schema this project's pipeline expects.

Usage
-----
    python models/fetch_dolt_options.py \\
        --tickers NVDA AAPL TSLA MSFT META GOOGL AVGO AMD \\
        --start   2022-01-01 \\
        --end     2022-12-31 \\
        --out     models/data

The script fetches one (ticker, date) at a time via the Dolthub SQL REST
API (free, no auth required), joins with Yahoo Finance underlying closes,
pivots call/put rows into the CBOE-wide format, and writes one CSV per
ticker.  Partial progress is checkpointed so the script is restartable.

Dependencies: requests, yfinance (both already in requirements.txt).
"""
import argparse
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DOLT_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"


def _dolt_query(sql, max_retries=3, timeout=90):
    """Hit the Dolthub REST API with simple retry/backoff."""
    url = f"{DOLT_BASE}?q={urllib.parse.quote(sql)}"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                time.sleep(1 + attempt)
                continue
            data = r.json()
            if data.get('query_execution_status') == 'Success':
                return data.get('rows', [])
            # context deadline etc. -> retry
            time.sleep(2 + attempt)
        except Exception:
            time.sleep(2 + attempt)
    return None


def _fetch_one_day(ticker, date_str):
    sql = (
        "SELECT date, expiration, strike, call_put, bid, ask, vol, "
        "       delta, gamma, theta, vega, rho "
        "FROM option_chain "
        f"WHERE act_symbol='{ticker}' AND date='{date_str}'"
    )
    return date_str, _dolt_query(sql)


def fetch_ticker_window(ticker, start, end, verbose=True, workers=8):
    """Fetch option_chain rows for a ticker across [start, end] in parallel."""
    ticker = ticker.upper()
    # Only query weekdays (Mon-Fri); markets are closed on weekends.
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)

    all_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one_day, ticker, d): d for d in days}
        for fut in as_completed(futs):
            d, rows = fut.result()
            done += 1
            if rows is None:
                if verbose:
                    print(f"  [{ticker}] {d}  FAILED (skipping)", flush=True)
            else:
                all_rows.extend(rows)
                if verbose and done % 10 == 0:
                    print(f"  [{ticker}] {done}/{len(days)} days done "
                          f"({len(all_rows):,} contracts so far)", flush=True)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # Coerce dtypes
    df['date']       = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    for col in ['strike', 'bid', 'ask', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def to_cboe_wide(df_long, underlying_close):
    """Pivot long call/put rows into CBOE-wide (one row per date x expiry x strike)."""
    if df_long.empty:
        return pd.DataFrame()
    df_long['call_put'] = df_long['call_put'].str.lower()
    calls = df_long[df_long['call_put'] == 'call'].drop(columns=['call_put'])
    puts  = df_long[df_long['call_put'] == 'put' ].drop(columns=['call_put'])
    calls = calls.rename(columns={
        'bid': 'C_BID', 'ask': 'C_ASK',
        'vol': 'C_IV', 'delta': 'C_DELTA', 'gamma': 'C_GAMMA',
        'theta': 'C_THETA', 'vega': 'C_VEGA', 'rho': 'C_RHO',
    })
    puts = puts.rename(columns={
        'bid': 'P_BID', 'ask': 'P_ASK',
        'vol': 'P_IV', 'delta': 'P_DELTA', 'gamma': 'P_GAMMA',
        'theta': 'P_THETA', 'vega': 'P_VEGA', 'rho': 'P_RHO',
    })
    key = ['date', 'expiration', 'strike']
    wide = calls.merge(puts, on=key, how='inner')
    # Join underlying close
    u = underlying_close.rename(columns={'Date': 'date', 'Close': 'UNDERLYING_LAST'})
    u['date'] = pd.to_datetime(u['date']).dt.tz_localize(None)
    wide = wide.merge(u, on='date', how='left')
    # Derived columns the ETL expects
    wide['QUOTE_DATE']  = wide['date']
    wide['EXPIRE_DATE'] = wide['expiration']
    wide['STRIKE']      = wide['strike']
    wide['DTE'] = (wide['expiration'] - wide['date']).dt.days
    wide['STRIKE_DISTANCE']     = (wide['STRIKE'] - wide['UNDERLYING_LAST']).abs()
    wide['STRIKE_DISTANCE_PCT'] = wide['STRIKE_DISTANCE'] / wide['UNDERLYING_LAST']
    # Fill controls not present in Dolt (volume missing); downstream handles NaN
    wide['P_VOLUME'] = np.nan
    wide['C_VOLUME'] = np.nan
    # Reorder to a stable column set
    cols = ['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'DTE',
            'STRIKE', 'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT',
            'C_BID', 'C_ASK', 'C_IV', 'C_DELTA', 'C_GAMMA',
            'C_THETA', 'C_VEGA', 'C_RHO', 'C_VOLUME',
            'P_BID', 'P_ASK', 'P_IV', 'P_DELTA', 'P_GAMMA',
            'P_THETA', 'P_VEGA', 'P_RHO', 'P_VOLUME']
    return wide[cols]


def download_underlying(ticker, start, end):
    """yfinance daily close, tz-naive."""
    end_plus = (end + timedelta(days=1)).strftime('%Y-%m-%d')
    hist = yf.Ticker(ticker).history(start=start.strftime('%Y-%m-%d'),
                                     end=end_plus, auto_adjust=False)
    hist = hist[['Close']].reset_index()
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    return hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tickers', nargs='+',
                   default=['NVDA', 'AAPL', 'TSLA', 'MSFT', 'META',
                            'GOOGL', 'AVGO', 'AMD'])
    p.add_argument('--start', default='2022-01-01')
    p.add_argument('--end',   default='2022-12-31')
    p.add_argument('--out',   default='models/data')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--day-workers', type=int, default=8,
                   help='concurrent Dolthub requests per ticker')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end   = datetime.strptime(args.end,   '%Y-%m-%d')

    for ticker in args.tickers:
        ticker = ticker.upper()
        out_csv = os.path.join(args.out,
                               f"{ticker.lower()}_{start:%Y}_{end:%Y}.csv")
        if os.path.exists(out_csv) and not args.overwrite:
            print(f"[{ticker}] already exists at {out_csv}, skipping "
                  f"(use --overwrite to replace)", flush=True)
            continue

        t0 = time.time()
        print(f"[{ticker}] fetching option chains {args.start} -> {args.end}", flush=True)
        df_long = fetch_ticker_window(ticker, start, end, workers=args.day_workers)
        if df_long.empty:
            print(f"[{ticker}] NO ROWS returned, skipping", flush=True)
            continue

        print(f"[{ticker}] fetched {len(df_long):,} long rows across "
              f"{df_long['date'].nunique()} dates in {time.time()-t0:.0f}s; "
              f"pulling underlying", flush=True)
        try:
            undl = download_underlying(ticker, start, end)
        except Exception as e:
            print(f"[{ticker}] underlying fetch failed: {e}", flush=True)
            continue

        wide = to_cboe_wide(df_long, undl)
        if wide.empty:
            print(f"[{ticker}] pivot produced empty frame", flush=True)
            continue

        wide.to_csv(out_csv, index=False)
        print(f"[{ticker}] wrote {len(wide):,} rows -> {out_csv} "
              f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
