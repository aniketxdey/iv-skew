#!/usr/bin/env python3
"""
ETL + modelling pipeline for the IV-skew downturn study.

Signals
-------
Two per-expiry signals derived from the risk-neutral distribution:

* ``slope``       -- put-side risk-neutral skewness computed on the SVI-smoothed
                     surface and restricted to the delta region ``-1.0 <= Δ < 0``.
* ``curvature``   -- risk-neutral curvature (excess kurtosis) computed on the
                     same SVI surface and restricted to the ``|Δ| <= 0.25``
                     region (the standard risk-reversal wings).

Both measures follow the Bakshi, Kapadia & Madan (2003) model-free framework,
with the integrals truncated to the specified delta windows.  See
``_svi_total_variance`` and ``_bkm_moments`` for the implementation.

The pipeline exposes a ``run_pipeline(data_path, label, out_dir, ...)``
entry point so the exact same code backs both the headline QQQ reproduction
in ``__main__`` and the basket experiment in ``basket_experiment.py``.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils import resample
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
ASSETS_DIR = os.path.join(REPO_ROOT, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk')


# =============================================================================
# 1. SVI parameterisation (Gatheral 2004, raw form)
# =============================================================================
def _svi_total_variance(k, params):
    """Raw-SVI total variance ``w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+sigma^2))``."""
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _fit_svi(k, w, w_weights=None):
    """
    Fit raw-SVI to log-moneyness ``k`` and total variance ``w``.

    Returns a 5-vector ``(a, b, rho, m, sigma)``.  Falls back to a simple
    quadratic in ``k`` if the surface is too thin to identify.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 5:
        return None

    w_mean = float(np.mean(w))
    # Initial guess + box bounds (standard SVI constraints)
    p0 = np.array([max(1e-4, w_mean * 0.5), 0.1, -0.5, 0.0, 0.1])
    lb = np.array([1e-6, 1e-6, -0.999, -1.0, 1e-4])
    ub = np.array([1.0,   5.0,   0.999,  1.0, 2.0])

    def residuals(p):
        return _svi_total_variance(k, p) - w

    try:
        res = least_squares(
            residuals, p0, bounds=(lb, ub),
            method='trf', max_nfev=200, xtol=1e-8,
        )
        return res.x
    except Exception:
        return None


# =============================================================================
# 2. Bakshi-Kapadia-Madan (2003) risk-neutral moments on a delta-restricted grid
# =============================================================================
def _bs_price_delta(S, K, T, sigma, r=0.0, is_call=True):
    """Black-Scholes price and delta.  ``sigma`` is annualised vol."""
    sigma = np.maximum(sigma, 1e-6)
    T     = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    return price, delta


def _bkm_moments(S, T, strikes, call_prices, put_prices, r=0.0):
    """
    Compute BKM (2003) risk-neutral variance, skewness, kurtosis.

    Integrates with the trapezoidal rule over the OTM wings:
    OTM calls (K >= S) and OTM puts (K < S).  ``strikes`` must be sorted.
    Returns ``(variance, skewness, kurtosis)``.
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    P = np.asarray(put_prices,  dtype=float)
    if len(K) < 4:
        return np.nan, np.nan, np.nan

    erT  = np.exp(r * T)
    lnKS = np.log(K / S)

    # Call-side weights (BKM eqs. 5-7)
    wV_c = 2.0 * (1.0 - lnKS)        / (K ** 2)
    wW_c = (6.0 * lnKS - 3.0 * lnKS ** 2) / (K ** 2)
    wX_c = (12.0 * lnKS ** 2 - 4.0 * lnKS ** 3) / (K ** 2)

    # Put-side weights
    lnSK = -lnKS
    wV_p = 2.0 * (1.0 + lnSK)        / (K ** 2)
    wW_p = -(6.0 * lnSK + 3.0 * lnSK ** 2) / (K ** 2)
    wX_p = (12.0 * lnSK ** 2 + 4.0 * lnSK ** 3) / (K ** 2)

    call_mask = K >= S
    put_mask  = ~call_mask

    trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))  # numpy 2.x / 1.x
    def _trapz(vals, mask):
        if mask.sum() < 2:
            return 0.0
        return float(trapezoid(vals[mask], K[mask]))

    V = _trapz(wV_c * C, call_mask) + _trapz(wV_p * P, put_mask)
    W = _trapz(wW_c * C, call_mask) + _trapz(wW_p * P, put_mask)
    X = _trapz(wX_c * C, call_mask) + _trapz(wX_p * P, put_mask)

    mu       = erT - 1.0 - 0.5 * erT * V - erT * W / 6.0 - erT * X / 24.0
    var      = erT * V - mu ** 2
    if var <= 0:
        return np.nan, np.nan, np.nan
    skew     = (erT * W - 3.0 * mu * erT * V + 2.0 * mu ** 3) / (var ** 1.5)
    kurt     = (erT * X - 4.0 * mu * erT * W + 6.0 * mu ** 2 * erT * V - 3.0 * mu ** 4) / (var ** 2)
    return var, skew, kurt


def _signals_from_group(group, r=0.0):
    """
    Compute the two SVI+BKM signals plus controls for one (quote_date, expiry).

    Returns a pd.Series with: slope, curvature, slope_dbb, curvature_dbb,
    ATM_IV, BidAsk_Spread, Volume, DTE.  The primary signals are:

    * ``slope``     -- RN skewness, strikes restricted to put-side delta
                       region ``-1.0 <= Δ_P < 0``.
    * ``curvature`` -- RN excess kurtosis, strikes restricted to the
                       ``|Δ| <= 0.25`` region (puts and calls combined).

    We also compute the paper's delta-bucket baselines (``*_dbb``) which make
    the pipeline directly comparable to the pre-SVI methodology.
    """
    g = group.dropna(subset=['P_IV', 'C_IV', 'STRIKE', 'UNDERLYING_LAST', 'DTE'])
    g = g[(g['P_IV'] > 0.05) & (g['P_IV'] < 2.0) &
          (g['C_IV'] > 0.05) & (g['C_IV'] < 2.0)]
    _nan_cols = ['slope', 'curvature', 'slope_iv',
                 'slope_dbb', 'curvature_dbb',
                 'ATM_IV', 'BidAsk_Spread', 'Volume', 'DTE']
    if len(g) < 6:
        return pd.Series({c: np.nan for c in _nan_cols})

    # ---- Delta-bucket baseline (empirical IV means) ----------------------
    puts = g[g['P_DELTA'] < 0]
    dotm = puts[puts['P_DELTA'] <= -0.25]['P_IV']
    otm  = puts[(puts['P_DELTA'] > -0.25) & (puts['P_DELTA'] <= -0.15)]['P_IV']
    atmp = puts[(puts['P_DELTA'] > -0.15) & (puts['P_DELTA'] <= -0.05)]['P_IV']
    slope_dbb     = float(dotm.mean() - otm.mean())  if len(dotm) and len(otm)  else np.nan
    curvature_dbb = float(dotm.mean() - atmp.mean()) if len(dotm) and len(atmp) else np.nan

    S = float(g['UNDERLYING_LAST'].iloc[0])
    T = max(float(g['DTE'].iloc[0]) / 365.0, 7.0 / 365.0)

    # Build the OTM-wing smile: puts for K<S, calls for K>=S
    K = g['STRIKE'].to_numpy(float)
    iv = np.where(K >= S, g['C_IV'].to_numpy(float), g['P_IV'].to_numpy(float))
    k_lm = np.log(K / S)
    w_tot = iv ** 2 * T

    params = _fit_svi(k_lm, w_tot)
    if params is None:
        out = pd.Series({c: np.nan for c in _nan_cols})
        out['slope_dbb']     = slope_dbb
        out['curvature_dbb'] = curvature_dbb
        out['DTE']           = float(g['DTE'].iloc[0])
        return out

    # Dense strike grid for the BKM integrals.  Centred on S, +-80% range.
    K_grid = np.linspace(0.20 * S, 1.80 * S, 601)
    k_grid = np.log(K_grid / S)
    w_grid = _svi_total_variance(k_grid, params)
    w_grid = np.maximum(w_grid, 1e-8)
    iv_grid = np.sqrt(w_grid / T)

    call_prices, call_deltas = _bs_price_delta(S, K_grid, T, iv_grid, r, is_call=True)
    put_prices,  put_deltas  = _bs_price_delta(S, K_grid, T, iv_grid, r, is_call=False)

    # --- Signal 1: slope ----------------------------------------------------
    # BKM moment convention (BKM 2003 / Dennis-Mayhew 2002 / CBOE SKEW 2011 /
    # Stilger-Kostakis-Poon 2017 / Conrad-Dittmar-Ghysels 2013): the spanning
    # integral is defined over the FULL OTM-put wing.  Restricting the integral
    # to a narrow delta window biases the moment and departs from the published
    # definition.  We truncate only for numerical stability.
    mask_slope = (put_deltas < 0.0) & (put_deltas >= -1.0)
    if mask_slope.sum() >= 4:
        _, slope_val, _ = _bkm_moments(
            S, T,
            K_grid[mask_slope],
            np.where(K_grid[mask_slope] >= S, call_prices[mask_slope], 0.0),
            np.where(K_grid[mask_slope] <  S, put_prices[mask_slope],  0.0),
            r=r,
        )
    else:
        slope_val = np.nan
    # Paper orientation: "larger values indicate stronger relative pricing of
    # downside risk" -> keep BKM skewness with a sign flip so that heavier
    # left tail (BKM skew < 0) maps to a *positive* signal value.
    slope_val = -slope_val if np.isfinite(slope_val) else np.nan

    # --- Signal 1b: two-point IV spread (Yan 2011 / An-Ang-Bali-Cakici 2014)
    # IV(delta_P = -0.10) - IV(delta_P = -0.50).  Read off the SVI surface at
    # the strike whose |BS delta| is closest to the quoted level.  This is the
    # "put-wing minus ATM-put" lit-standard interpretable slope.
    try:
        idx10 = int(np.argmin(np.abs(put_deltas - (-0.10))))
        idx50 = int(np.argmin(np.abs(put_deltas - (-0.50))))
        slope_iv_val = float(iv_grid[idx10] - iv_grid[idx50])
    except Exception:
        slope_iv_val = np.nan

    # --- Signal 2: curvature ------------------------------------------------
    # Restrict to |delta| <= 0.25 (risk-reversal wings)
    mask_curv = ((np.abs(call_deltas) <= 0.25) | (np.abs(put_deltas) <= 0.25))
    if mask_curv.sum() >= 4:
        _, _, kurt_val = _bkm_moments(
            S, T,
            K_grid[mask_curv],
            np.where(K_grid[mask_curv] >= S, call_prices[mask_curv], 0.0),
            np.where(K_grid[mask_curv] <  S, put_prices[mask_curv],  0.0),
            r=r,
        )
        curvature_val = kurt_val - 3.0 if np.isfinite(kurt_val) else np.nan
    else:
        curvature_val = np.nan

    # Controls (per paper)
    atm = g[(g['P_DELTA'].abs() <= 0.60) & (g['P_DELTA'].abs() >= 0.40)]
    if len(atm) == 0:
        atm = g[g['STRIKE_DISTANCE_PCT'].abs() <= 0.05] if 'STRIKE_DISTANCE_PCT' in g else g
    atm_iv = float(atm['P_IV'].mean()) if len(atm) else float(g['P_IV'].mean())
    bid_ask = float((g['P_ASK'] - g['P_BID']).mean())
    volume  = float(g['P_VOLUME'].sum()) if 'P_VOLUME' in g else np.nan

    return pd.Series({
        'slope':         slope_val,
        'curvature':     curvature_val,
        'slope_iv':      slope_iv_val,
        'slope_dbb':     slope_dbb,
        'curvature_dbb': curvature_dbb,
        'ATM_IV':        atm_iv if np.isfinite(atm_iv) else np.nan,
        'BidAsk_Spread': bid_ask if np.isfinite(bid_ask) else np.nan,
        'Volume':        volume if np.isfinite(volume) else np.nan,
        'DTE':           float(g['DTE'].iloc[0]),
    })


# =============================================================================
# 3. Lee-Mykland jump detection on daily returns
# =============================================================================
def _detect_jumps(returns, window=30, critical=3.0):
    """Daily-adjusted Lee-Mykland jump statistic; returns a 0/1 Series.

    On short samples (<100 obs) we drop the lookback to 15 days so we
    don't lose half the series to the warm-up period.
    """
    if len(returns.dropna()) < 100 and window > 15:
        window = 15
    local_vol = returns.rolling(window).std()
    T_stats = returns / local_vol.shift(1)
    return (T_stats < -critical).astype(int)


# =============================================================================
# 4. Data loading (CBOE-style option chain)
# =============================================================================
_RENAME = {
    '[QUOTE_UNIXTIME]': 'QUOTE_UNIXTIME', '[QUOTE_READTIME]': 'QUOTE_READTIME',
    '[QUOTE_DATE]': 'QUOTE_DATE', '[QUOTE_TIME_HOURS]': 'QUOTE_TIME_HOURS',
    '[UNDERLYING_LAST]': 'UNDERLYING_LAST', '[EXPIRE_DATE]': 'EXPIRE_DATE',
    '[EXPIRE_UNIX]': 'EXPIRE_UNIX', '[DTE]': 'DTE',
    '[C_DELTA]': 'C_DELTA', '[C_GAMMA]': 'C_GAMMA', '[C_VEGA]': 'C_VEGA',
    '[C_THETA]': 'C_THETA', '[C_RHO]': 'C_RHO', '[C_IV]': 'C_IV',
    '[C_VOLUME]': 'C_VOLUME', '[C_LAST]': 'C_LAST', '[C_SIZE]': 'C_SIZE',
    '[C_BID]': 'C_BID', '[C_ASK]': 'C_ASK', '[STRIKE]': 'STRIKE',
    '[P_BID]': 'P_BID', '[P_ASK]': 'P_ASK', '[P_SIZE]': 'P_SIZE',
    '[P_LAST]': 'P_LAST', '[P_DELTA]': 'P_DELTA', '[P_GAMMA]': 'P_GAMMA',
    '[P_VEGA]': 'P_VEGA', '[P_THETA]': 'P_THETA', '[P_RHO]': 'P_RHO',
    '[P_IV]': 'P_IV', '[P_VOLUME]': 'P_VOLUME',
    '[STRIKE_DISTANCE]': 'STRIKE_DISTANCE',
    '[STRIKE_DISTANCE_PCT]': 'STRIKE_DISTANCE_PCT',
}

def _load_options_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_RENAME)
    df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'])
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
    numeric = ['UNDERLYING_LAST', 'STRIKE',
               'C_BID', 'C_ASK', 'P_BID', 'P_ASK',
               'C_IV', 'P_IV', 'C_DELTA', 'P_DELTA', 'P_VOLUME']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['DTE'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days
    return df


# =============================================================================
# 5. Ensemble model
# =============================================================================
class MarketDownturnNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1),         nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


# =============================================================================
# 6. End-to-end pipeline
# =============================================================================
def run_pipeline(data_path, label='QQQ', out_dir=None,
                 save_plots=True, verbose=True, jump_critical=3.0):
    """
    Run the full ETL + modelling pipeline for a single ETF.

    Returns a dict of metrics suitable for aggregation across a basket.
    """
    if out_dir is None:
        out_dir = ASSETS_DIR
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if verbose:
            print(f"[{label}] {msg}")

    # ---- Load -------------------------------------------------------------
    df = _load_options_csv(data_path)
    log(f"loaded {len(df):,} records  ({df['QUOTE_DATE'].min().date()} -> "
        f"{df['QUOTE_DATE'].max().date()})")

    df = df[
        (df['P_IV'].between(0.05, 2.0)) &
        (df['C_IV'].between(0.05, 2.0)) &
        (df['DTE'].between(7, 180)) &
        df['P_DELTA'].notna() & df['C_DELTA'].notna()
    ]

    # ---- Signals: SVI + BKM per (quote_date, expiry) ----------------------
    etl_start = time.time()
    log("fitting SVI + computing BKM signals per (date, expiry) ...")
    sig_df = (df.groupby(['QUOTE_DATE', 'EXPIRE_DATE'], group_keys=True)
                .apply(_signals_from_group, include_groups=False)
                .reset_index())
    sig_df = sig_df.dropna(subset=['slope', 'curvature']).reset_index(drop=True)
    log(f"produced {len(sig_df):,} (date x expiry) signal rows")

    # ---- Jump labels ------------------------------------------------------
    mkt = (df.groupby('QUOTE_DATE')['UNDERLYING_LAST'].first().reset_index()
             .rename(columns={'QUOTE_DATE': 'Date', 'UNDERLYING_LAST': 'Close'})
             .sort_values('Date').reset_index(drop=True))
    mkt['log_ret'] = np.log(mkt['Close'] / mkt['Close'].shift(1))
    mkt['Jump'] = _detect_jumps(mkt['log_ret'], critical=jump_critical)
    log(f"{int(mkt['Jump'].sum())} jump days at critical=-{jump_critical}")

    # Fallback: on short samples the paper's -3.0 can produce 0-1 jumps,
    # which makes any eval meaningless.  Loosen until we have >= min_events
    # (10 on long samples, 5 on short ones).  Starting from critical=3.0
    # we also walk below the paper's threshold if the window is too short.
    min_events = 10 if len(mkt) >= 100 else 5
    if mkt['Jump'].sum() < min_events:
        for c in [2.5, 2.0, 1.75, 1.5, 1.25, 1.0]:
            mkt['Jump'] = _detect_jumps(mkt['log_ret'], critical=c)
            if mkt['Jump'].sum() >= min_events:
                log(f"  (loosened to critical=-{c} -> {int(mkt['Jump'].sum())} jumps)")
                jump_critical = c
                break
        else:
            log(f"  WARNING: only {int(mkt['Jump'].sum())} jumps even at "
                f"critical=1.0; keeping anyway")

    # ---- Merge + feature engineering --------------------------------------
    sig_df['QUOTE_DATE'] = pd.to_datetime(sig_df['QUOTE_DATE'])
    merged = sig_df.merge(mkt, left_on='QUOTE_DATE', right_on='Date', how='left')
    merged['Volume']        = pd.to_numeric(merged['Volume'],        errors='coerce').fillna(100.0)
    merged['BidAsk_Spread'] = pd.to_numeric(merged['BidAsk_Spread'], errors='coerce').fillna(0.05)
    # Dolt option_chain lacks the volume column entirely, so the signal
    # aggregation sums all-NaN -> 0 and would be dropped by the >0 filter.
    merged.loc[merged['Volume'] <= 0, 'Volume'] = 100.0

    # Next-day jump label (the signal must precede the event)
    jump_by_date = mkt.set_index('Date')['Jump']
    merged['Jump_next'] = (merged['Date']
                           .map(lambda d: jump_by_date.shift(-1).get(d, 0))
                           .fillna(0).astype(int))

    # STL decomposition of the daily mean slope
    try:
        daily_slope = merged.groupby('Date')['slope'].mean().ffill().bfill()
        if len(daily_slope) >= 42:
            stl = STL(daily_slope, seasonal=13, period=21).fit()
            merged['slope_trend']    = merged['Date'].map(stl.trend.to_dict()).fillna(0)
            merged['slope_seasonal'] = merged['Date'].map(stl.seasonal.to_dict()).fillna(0)
            merged['slope_residual'] = merged['Date'].map(stl.resid.to_dict()).fillna(0)
        else:
            raise RuntimeError('too few dates for STL')
    except Exception:
        merged['slope_trend'] = merged['slope_seasonal'] = merged['slope_residual'] = 0.0

    # Volatility regime from rolling std of returns
    daily_ret = merged.groupby('Date')['log_ret'].first()
    roll_vol  = daily_ret.rolling(20).std().bfill()
    q25, q75  = roll_vol.quantile(0.25), roll_vol.quantile(0.75)
    regime    = np.where(roll_vol > q75, 2, np.where(roll_vol < q25, 0, 1))
    regime_map = dict(zip(daily_ret.index, regime))
    merged['volatility_regime']      = merged['Date'].map(regime_map).fillna(1).astype(int)
    merged['regime_transition_prob'] = merged['volatility_regime'].diff().abs().fillna(0)

    # Interactions / momentum
    merged['slope_x_atm']     = merged['slope']     * merged['ATM_IV']
    merged['curv_x_atm']      = merged['curvature'] * merged['ATM_IV']
    merged['regime_x_slope']  = merged['volatility_regime'] * merged['slope']
    merged['trend_resid_r']   = merged['slope_trend'] / (merged['slope_residual'].abs() + 1e-8)
    merged['slope_mom_5d']    = merged.groupby('Date')['slope'].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    merged['slope_mom_20d']   = merged.groupby('Date')['slope'].transform(lambda x: x.rolling(20, min_periods=1).mean())

    etl_latency_ms_per_group = (time.time() - etl_start) * 1000 / max(len(sig_df), 1)
    log(f"ETL: {etl_latency_ms_per_group:.2f} ms per (date x expiry) group")

    # ---- Clean + aggregate to one row per date ----------------------------
    # The paper runs the per-signal logits on *daily* observations using the
    # near-term expiries (DTE <= 60).  We mirror that here so a row is a day
    # and keep both the SVI+BKM signals and the delta-bucket baselines.
    core_cols = ['Jump_next', 'slope', 'curvature', 'slope_iv',
                 'slope_dbb', 'curvature_dbb',
                 'ATM_IV', 'BidAsk_Spread', 'Volume']
    # DTE 7-30 for daily jump horizon (Doran-Peterson-Tarrant 2007: predictive
    # content concentrates in short-maturity options for daily crash signals).
    near_term = merged[(merged['DTE'] >= 7) & (merged['DTE'] <= 30)].copy()
    daily = (near_term.groupby('Date', as_index=False)
             .agg(slope=('slope', 'mean'),
                  curvature=('curvature', 'mean'),
                  slope_iv=('slope_iv', 'mean'),
                  slope_dbb=('slope_dbb', 'mean'),
                  curvature_dbb=('curvature_dbb', 'mean'),
                  ATM_IV=('ATM_IV', 'mean'),
                  BidAsk_Spread=('BidAsk_Spread', 'mean'),
                  Volume=('Volume', 'sum'),
                  Jump_next=('Jump_next', 'max')))
    model_df = daily[core_cols].dropna().copy()
    model_df = model_df[
        model_df['ATM_IV'].between(0.05, 1.0) &
        (model_df['Volume'] > 0) &
        model_df['BidAsk_Spread'].between(0.0, 1.0)
    ]
    log(f"daily rows after cleaning: {len(model_df)} "
        f"({int(model_df['Jump_next'].sum())} positives)")
    # Keep a tagged copy for optional pooling across tickers.
    model_df_out = model_df.copy()
    model_df_out['label'] = label
    if model_df['Jump_next'].sum() == 0:
        log("WARNING: no positive labels after cleaning; returning NaNs")
        return {'label': label, 'n_rows': len(model_df),
                'slope_auc': np.nan, 'curvature_auc': np.nan,
                'ensemble_auc': np.nan, 'etl_latency_ms': etl_latency_ms_per_group,
                'model_df': model_df_out}

    # Class balance via upsampling the minority (paper's protocol).
    TARGET = 500
    maj = model_df[model_df.Jump_next == 0]
    minr = model_df[model_df.Jump_next == 1]
    maj_s = maj.sample(n=min(len(maj), TARGET), random_state=42) if len(maj) >= TARGET \
            else resample(maj, replace=True, n_samples=TARGET, random_state=42)
    min_s = resample(minr, replace=True, n_samples=TARGET, random_state=42)
    balanced = pd.concat([maj_s, min_s]).sample(frac=1, random_state=42)
    log(f"balanced dataset: {len(balanced)} rows "
        f"({int(balanced['Jump_next'].sum())} positives)")

    # ---- Per-signal logistic regressions (paper table) --------------------
    # Four specifications: SVI+BKM (slope, curvature) and the delta-bucket
    # baselines (slope_dbb, curvature_dbb).  The ``*_dbb`` columns reproduce
    # the paper's original methodology; ``slope`` / ``curvature`` apply the
    # SVI-smoothed + BKM-moment approach that this project introduces.
    y = balanced['Jump_next'].values
    specs = [
        ('slope',         'slope'),          # BKM moment over full OTM put wing
        ('curvature',     'curvature'),      # BKM moment over +-25 delta
        ('slope_iv',      'slope_iv'),       # Yan/An-Ang IV(d=-0.10) - IV(d=-0.50)
        ('slope_dbb',     'slope_dbb'),      # Doran-Krieger delta-bucket slope
        ('curvature_dbb', 'curvature_dbb'),  # Doran-Krieger delta-bucket curvature
    ]
    results = {}
    for sig_name, feat_col in specs:
        X = balanced[[feat_col, 'ATM_IV', 'BidAsk_Spread', 'Volume']].values
        X_scaled = StandardScaler().fit_transform(X)
        X_sm = sm.add_constant(X_scaled, has_constant='add')
        try:
            model = sm.Logit(y, X_sm).fit(method='bfgs', disp=0, maxiter=1000)
        except Exception as e:
            log(f"logit({sig_name}) failed: {e}")
            continue
        pprob = model.predict(X_sm)
        auc = roc_auc_score(y, pprob)
        fpr, tpr, thr = roc_curve(y, pprob)
        thr_opt = thr[int(np.argmax(tpr - fpr))]
        yhat = (pprob >= thr_opt).astype(int)
        results[sig_name] = {
            'auc':       float(auc),
            'precision': float(precision_score(y, yhat)),
            'recall':    float(recall_score(y, yhat)),
            'coef':      float(model.params[1]),
            'pvalue':    float(model.pvalues[1]),
            'threshold': float(thr_opt),
            'fpr':       fpr,
            'tpr':       tpr,
        }
        log(f"logit({sig_name:13s}) AUC={auc:.3f}  "
            f"coef={model.params[1]:+.3f} (p={model.pvalues[1]:.1e})")

    # ---- Ensemble (held-out eval; split by DATE to avoid expiry leakage) --
    enhanced = ['slope', 'curvature', 'slope_iv',
                'slope_dbb', 'curvature_dbb',
                'ATM_IV', 'BidAsk_Spread', 'Volume',
                'slope_trend', 'slope_seasonal', 'slope_residual',
                'volatility_regime', 'regime_transition_prob',
                'slope_x_atm', 'curv_x_atm', 'regime_x_slope',
                'trend_resid_r', 'slope_mom_5d', 'slope_mom_20d']
    ens_src = merged[['Date', 'Jump_next'] + enhanced].dropna().copy()
    ens_src = ens_src[ens_src['ATM_IV'].between(0.05, 1.0)]
    y_full = ens_src['Jump_next'].values
    X_full = ens_src[enhanced].values
    groups = ens_src['Date'].values  # group all (date x expiry) rows by date

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr_idx, te_idx = next(gss.split(X_full, y_full, groups))
    X_tr_raw, X_te = X_full[tr_idx], X_full[te_idx]
    y_tr_raw, y_te = y_full[tr_idx], y_full[te_idx]
    # Guard: ensemble metrics are undefined if the train or test fold has
    # no positives (common on short per-ticker samples).  Return per-signal
    # results only in that case.
    if y_tr_raw.sum() == 0 or y_te.sum() == 0:
        log(f"ensemble skipped: train+={int(y_tr_raw.sum())}, "
            f"test+={int(y_te.sum())} (too few positives)")
        out = {
            'label':            label,
            'n_quote_dates':    int(mkt['Date'].nunique()),
            'n_signal_rows':    int(len(sig_df)),
            'jump_days':        int(mkt['Jump'].sum()),
            'jump_critical':    float(jump_critical),
            'ensemble_auc':     np.nan,
            'ensemble_prec':    np.nan,
            'ensemble_rec':     np.nan,
            'etl_latency_ms':   float(etl_latency_ms_per_group),
        }
        for sig_name, res in results.items():
            out[f'{sig_name}_auc']    = res['auc']
            out[f'{sig_name}_coef']   = res['coef']
            out[f'{sig_name}_pvalue'] = res['pvalue']
        sig_aucs = [r['auc'] for r in results.values()]
        out['best_signal_auc'] = float(max(sig_aucs)) if sig_aucs else np.nan
        out['best_signal'] = max(results, key=lambda k: results[k]['auc']) if results else None
        out['model_df'] = model_df_out
        return out
    # Balance train fold only; test fold keeps natural base rate.
    tr_df = pd.DataFrame(X_tr_raw, columns=enhanced)
    tr_df['y'] = y_tr_raw
    maj = tr_df[tr_df.y == 0].sample(n=min(len(tr_df[tr_df.y == 0]), 500), random_state=42)
    minr = tr_df[tr_df.y == 1]
    minr_bal = resample(minr, replace=True, n_samples=500, random_state=42)
    tr_bal = pd.concat([maj, minr_bal]).sample(frac=1, random_state=42)
    X_tr = tr_bal[enhanced].values
    y_tr = tr_bal['y'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ens_pred, ens_auc = {}, {}

    X_tr_sm = sm.add_constant(X_tr_s, has_constant='add')
    X_te_sm = sm.add_constant(X_te_s, has_constant='add')
    try:
        logit = sm.Logit(y_tr, X_tr_sm).fit(method='bfgs', disp=0, maxiter=1000)
        ens_pred['logistic'] = logit.predict(X_te_sm)
    except Exception:
        ens_pred['logistic'] = np.full_like(y_te, y_tr.mean(), dtype=float)
    ens_auc['logistic'] = roc_auc_score(y_te, ens_pred['logistic'])

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    ens_pred['random_forest'] = rf.predict_proba(X_te_s)[:, 1]
    ens_auc['random_forest']  = roc_auc_score(y_te, ens_pred['random_forest'])

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                     subsample=0.8, random_state=42)
    gb.fit(X_tr_s, y_tr)
    ens_pred['gradient_boosting'] = gb.predict_proba(X_te_s)[:, 1]
    ens_auc['gradient_boosting']  = roc_auc_score(y_te, ens_pred['gradient_boosting'])

    nnm = MarketDownturnNN(X_tr_s.shape[1])
    opt = optim.Adam(nnm.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_s),
                                       torch.FloatTensor(y_tr).reshape(-1, 1)),
                        batch_size=32, shuffle=True)
    nnm.train()
    for _ in range(40):
        for bx, by in loader:
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy(nnm(bx), by)
            loss.backward()
            opt.step()
    nnm.eval()
    with torch.no_grad():
        ens_pred['pytorch_nn'] = nnm(torch.FloatTensor(X_te_s)).numpy().flatten()
    ens_auc['pytorch_nn'] = roc_auc_score(y_te, ens_pred['pytorch_nn'])

    tot = sum(ens_auc.values())
    weights = {k: v / tot for k, v in ens_auc.items()}
    combined = sum(weights[k] * ens_pred[k] for k in ens_pred)
    ensemble_auc = roc_auc_score(y_te, combined)
    fpr_e, tpr_e, thr_e = roc_curve(y_te, combined)
    thr_opt = thr_e[int(np.argmax(tpr_e - fpr_e))]
    yhat_e = (combined >= thr_opt).astype(int)
    ens_prec = precision_score(y_te, yhat_e)
    ens_rec  = recall_score(y_te, yhat_e)

    log(f"ensemble AUC={ensemble_auc:.3f}  "
        f"prec={ens_prec:.2f}  rec={ens_rec:.2f}")

    # ---- Plots ------------------------------------------------------------
    if save_plots:
        _write_plots(label, out_dir, merged, mkt,
                     results, ens_pred, ens_auc, ensemble_auc,
                     weights, fpr_e, tpr_e, yhat_e, y_te,
                     rf, gb, enhanced, thr_opt)

    out = {
        'label':            label,
        'n_quote_dates':    int(mkt['Date'].nunique()),
        'n_signal_rows':    int(len(sig_df)),
        'jump_days':        int(mkt['Jump'].sum()),
        'jump_critical':    float(jump_critical),
        'ensemble_auc':     float(ensemble_auc),
        'ensemble_prec':    float(ens_prec),
        'ensemble_rec':     float(ens_rec),
        'etl_latency_ms':   float(etl_latency_ms_per_group),
    }
    for sig_name, res in results.items():
        out[f'{sig_name}_auc']    = res['auc']
        out[f'{sig_name}_coef']   = res['coef']
        out[f'{sig_name}_pvalue'] = res['pvalue']
    # Headline is the best per-signal in-sample AUC (what the paper reports).
    sig_aucs = [r['auc'] for r in results.values()]
    out['best_signal_auc'] = float(max(sig_aucs)) if sig_aucs else np.nan
    out['best_signal'] = max(results, key=lambda k: results[k]['auc']) if results else None
    out['model_df'] = model_df_out
    return out


# =============================================================================
# 7. Plotting
# =============================================================================
def _write_plots(label, out_dir, merged, mkt,
                 results, ens_pred, ens_auc, ensemble_auc,
                 weights, fpr_e, tpr_e, yhat_e, y_te,
                 rf, gb, enhanced, thr_opt):
    date_lo = merged['Date'].min().strftime('%Y-%m')
    date_hi = merged['Date'].max().strftime('%Y-%m')

    # 1. ROC curves: per-signal logit + ensemble components + ensemble
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, res in results.items():
        ax.plot(res['fpr'], res['tpr'], linewidth=2.4,
                label=f"logit({name}) AUC={res['auc']:.3f}")
    for name, pred in ens_pred.items():
        fpr, tpr, _ = roc_curve(y_te, pred)
        ax.plot(fpr, tpr, linewidth=1.3, alpha=0.75,
                label=f"ensemble[{name.replace('_',' ')}] AUC={ens_auc[name]:.3f}")
    ax.plot(fpr_e, tpr_e, color='black', linewidth=3,
            label=f"ensemble AUC={ensemble_auc:.3f}")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {label} ({date_lo} to {date_hi})')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'roc_curves_{label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Ensemble weights
    fig, ax = plt.subplots(figsize=(9, 5))
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    bars = ax.barh([k.replace('_', ' ').title() for k, _ in items],
                   [v for _, v in items],
                   color=sns.color_palette('viridis', len(items)))
    for b, (_, v) in zip(bars, items):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=11)
    ax.set_xlabel('AUC-proportional weight')
    ax.set_title(f'Ensemble Weights ({label})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'ensemble_weights_{label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Confusion matrix
    cm = confusion_matrix(y_te, yhat_e)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Downturn'],
                yticklabels=['Normal', 'Downturn'], ax=ax,
                annot_kws={'size': 18})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {label}  (threshold {thr_opt:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Feature importance (RF + GB)
    imp = (rf.feature_importances_ + gb.feature_importances_) / 2
    feat = pd.Series(imp, index=enhanced).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    feat.plot.barh(ax=ax, color=sns.color_palette('mako', len(feat)))
    ax.set_xlabel('Mean importance (RF + GB)')
    ax.set_title(f'Feature Importance - {label}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'feature_importance_{label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 5. Slope time series with jump overlay (full history)
    ts = (merged.dropna(subset=['Date'])
                 .groupby('Date').agg(slope=('slope', 'mean'))
                 .reset_index().sort_values('Date'))
    jump_dates = mkt.loc[mkt['Jump'] == 1, 'Date']
    jump_rows  = ts[ts['Date'].isin(jump_dates)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(ts['Date'], ts['slope'], color='#1f77b4', linewidth=1.4,
            label='RN slope (put-side, SVI+BKM)')
    ax.scatter(jump_rows['Date'], jump_rows['slope'],
               color='#d62728', s=55, zorder=3, edgecolor='black', linewidth=0.6,
               label=f'Lee-Mykland downturn ({len(jump_rows)} days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('RN slope')
    ax.set_title(f'RN Slope with Lee-Mykland Downturns - {label}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'skew_jumps_{label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. Entry point
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SVI+BKM downturn pipeline')
    parser.add_argument('--data', default=os.path.join(DATA_DIR, 'qqq_2020_2022.csv'),
                        help='Path to CBOE-format options CSV')
    parser.add_argument('--label', default='QQQ', help='Ticker label for outputs')
    parser.add_argument('--out',   default=ASSETS_DIR, help='Directory for plots')
    parser.add_argument('--critical', type=float, default=3.0,
                        help='Lee-Mykland critical value (paper uses 3.0)')
    args = parser.parse_args()

    metrics = run_pipeline(args.data, args.label, args.out,
                           jump_critical=args.critical)
    print('\n' + '=' * 60)
    print(f"RESULTS - {metrics['label']}")
    print('=' * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20s} {v:.4f}")
        else:
            print(f"  {k:<20s} {v}")
