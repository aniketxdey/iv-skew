#!/usr/bin/env python3
"""
Basket experiment: run ``run_pipeline`` on every CBOE-format CSV under
``models/data/`` and report the cross-ticker summary.

The paper's own Discussion flags that its results are "centered on one ETF
and one volatile period".  This script extends the methodology to a basket
of big-tech / AI equity names to test whether the risk-neutral slope and
curvature signals generalise beyond QQQ.

Default basket: NVDA, AAPL, TSLA, MSFT, META, GOOGL, AVGO, AMD plus the
reference QQQ series (if present).  Add more tickers by placing additional
``<ticker>_YYYY_YYYY.csv`` files in ``models/data/`` — the schema should
match CBOE's "options-dx" wide format.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# Local import -- ``run_pipeline`` lives next door
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etl import run_pipeline, ASSETS_DIR  # noqa: E402

sns.set_theme(style='whitegrid', context='talk')


def _label_from_path(path):
    """Infer the ticker label from a filename like 'nvda_2022_2022.csv'."""
    name = Path(path).stem.lower()
    m = re.match(r'([a-z]+)[_-]?\d*', name)
    return (m.group(1) if m else name).upper()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='models/data',
                   help='directory to glob for options CSVs')
    p.add_argument('--pattern',  default='*_options.csv,*_20*.csv',
                   help='comma-separated glob patterns inside --data-dir')
    p.add_argument('--include',  nargs='*', default=None,
                   help='only include files whose inferred label is in this list')
    p.add_argument('--exclude',  nargs='*', default=['BACKTESTING', 'US_ETFS_DATA'],
                   help='skip files whose inferred label is in this list')
    p.add_argument('--critical', type=float, default=3.0,
                   help='Lee-Mykland critical value')
    p.add_argument('--out',  default=ASSETS_DIR, help='directory for plots')
    args = p.parse_args()

    files = []
    for pat in args.pattern.split(','):
        files.extend(glob.glob(os.path.join(args.data_dir, pat.strip())))
    files = sorted(set(files))
    if not files:
        print(f"No CSVs matched {args.pattern} under {args.data_dir}")
        return 1

    rows = []
    daily_frames = []
    for path in files:
        label = _label_from_path(path)
        if args.exclude and label in [x.upper() for x in args.exclude]:
            continue
        if args.include and label not in [x.upper() for x in args.include]:
            continue
        print(f"\n==== {label}  ({path}) ====")
        try:
            metrics = run_pipeline(path, label=label, out_dir=args.out,
                                   save_plots=True, jump_critical=args.critical,
                                   verbose=True)
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
            continue
        if 'model_df' in metrics and metrics['model_df'] is not None:
            daily_frames.append(metrics.pop('model_df'))
        rows.append(metrics)

    if not rows:
        print("No tickers completed successfully.")
        return 1

    summary = pd.DataFrame(rows)
    sig_auc_cols = [c for c in summary.columns if c.endswith('_auc')
                    and c not in ('ensemble_auc', 'best_signal_auc')]
    print("\n" + "=" * 72)
    print("BASKET SUMMARY")
    print("=" * 72)
    print(summary[['label', 'n_quote_dates', 'jump_days', 'jump_critical',
                    *sig_auc_cols, 'ensemble_auc', 'best_signal_auc',
                    'etl_latency_ms']].to_string(index=False, float_format='%.3f'))

    # Average AUC across the basket (paper's call-to-action)
    print("\n" + "-" * 72)
    print("MEAN IN-SAMPLE AUC ACROSS THE BASKET")
    print("-" * 72)
    for c in sig_auc_cols + ['ensemble_auc', 'best_signal_auc']:
        vals = pd.to_numeric(summary[c], errors='coerce').dropna()
        if len(vals):
            print(f"  {c:<22s}  mean={vals.mean():.3f}  "
                  f"median={vals.median():.3f}  "
                  f"n={len(vals)}")

    # Persist
    out_csv = os.path.join(args.out, 'basket_summary.csv')
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary table -> {out_csv}")

    # Plot: per-ticker in-sample AUC for each signal family
    _plot_basket(summary, sig_auc_cols, args.out)

    # Pooled basket logistic -- gives us real statistical power on the
    # tech-sector hypothesis.  We stack every ticker's daily frame
    # (excluding QQQ when a non-QQQ basket is present) and run one
    # balanced logistic per signal with ticker fixed effects.
    tech_frames = [df for df in daily_frames
                   if df['label'].iloc[0].upper() != 'QQQ']
    if len(tech_frames) >= 2:
        pooled_results = _pooled_logistic(tech_frames)
        print("\n" + "=" * 72)
        print(f"POOLED TECH-BASKET LOGISTIC ({len(tech_frames)} tickers, "
              f"{sum(len(d) for d in tech_frames):,} daily rows)")
        print("=" * 72)
        for sig, r in pooled_results.items():
            print(f"  {sig:<22s}  AUC={r['auc']:.3f}  coef={r['coef']:+.3f}  "
                  f"p={r['pvalue']:.2e}  n+={r['n_pos']}/{r['n_total']}")

        # Save pooled summary
        pooled_df = pd.DataFrame(pooled_results).T
        pooled_df.index.name = 'signal'
        pooled_df.to_csv(os.path.join(args.out, 'basket_pooled.csv'))
        _plot_pooled(pooled_results, args.out)
    else:
        print("\n(Pooled basket skipped: need >=2 non-QQQ tickers with jumps.)")

    return 0


def _pooled_logistic(frames):
    """Stack per-ticker daily frames, run one balanced logit per signal."""
    pooled = pd.concat(frames, ignore_index=True).dropna(subset=['Jump_next'])
    pooled['Jump_next'] = pooled['Jump_next'].astype(int)
    n_total = len(pooled)
    n_pos   = int(pooled['Jump_next'].sum())

    # Ticker fixed effects
    dummies = pd.get_dummies(pooled['label'], prefix='tkr', drop_first=True)
    controls = ['ATM_IV', 'BidAsk_Spread']  # Volume constant on Dolt -> drop
    specs = ['slope', 'curvature', 'slope_iv', 'slope_dbb', 'curvature_dbb']

    results = {}
    for sig in specs:
        rows = pooled[[sig, 'Jump_next'] + controls].dropna().copy()
        rows = rows.join(dummies, how='left')
        if rows['Jump_next'].sum() == 0:
            results[sig] = {'auc': np.nan, 'coef': np.nan, 'pvalue': np.nan,
                            'n_pos': 0, 'n_total': len(rows)}
            continue
        maj = rows[rows.Jump_next == 0]
        minr = rows[rows.Jump_next == 1]
        target = max(500, len(maj))
        maj_s = (maj.sample(n=min(len(maj), target), random_state=42)
                 if len(maj) >= target
                 else resample(maj, replace=True, n_samples=target, random_state=42))
        min_s = resample(minr, replace=True, n_samples=target, random_state=42)
        bal   = pd.concat([maj_s, min_s]).sample(frac=1, random_state=42)
        feat_cols = [sig] + controls + list(dummies.columns)
        X = sm.add_constant(bal[feat_cols].astype(float).values, has_constant='add')
        y = bal['Jump_next'].values
        try:
            m = sm.Logit(y, X).fit(method='bfgs', disp=0, maxiter=1000)
            pred = m.predict(X)
            auc  = roc_auc_score(y, pred)
            coef = m.params[1]       # position 0 = const, 1 = signal
            pv   = m.pvalues[1]
        except Exception as e:
            print(f"  [pooled logit on {sig}] failed: {e}")
            auc = coef = pv = np.nan
        results[sig] = {'auc': float(auc), 'coef': float(coef),
                        'pvalue': float(pv), 'n_pos': int(bal['Jump_next'].sum()),
                        'n_total': len(bal)}
    return results


def _plot_pooled(pooled_results, out_dir):
    sigs = list(pooled_results.keys())
    aucs = [pooled_results[s]['auc'] for s in sigs]
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette('viridis', n_colors=len(sigs))
    bars = ax.bar(sigs, aucs, color=palette)
    for b, a in zip(bars, aucs):
        if np.isfinite(a):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{a:.3f}', ha='center', va='bottom', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Pooled in-sample AUC')
    ax.set_title('Big-tech / AI basket -- pooled logistic (ticker fixed effects)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'basket_pooled.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved pooled plot -> {out_dir}/basket_pooled.png")


def _plot_basket(summary, sig_cols, out_dir):
    plot_df = (summary.melt(id_vars=['label'], value_vars=sig_cols,
                            var_name='signal', value_name='auc')
                      .dropna())
    plot_df['signal'] = plot_df['signal'].str.replace('_auc', '', regex=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x='label', y='auc', hue='signal',
                ax=ax, palette='viridis')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('In-sample AUC')
    ax.set_title('IV-skew signals across tech-basket (per-signal logistic, in-sample)')
    ax.legend(loc='lower right', ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'basket_auc_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot -> {out_dir}/basket_auc_comparison.png")


if __name__ == '__main__':
    sys.exit(main())
