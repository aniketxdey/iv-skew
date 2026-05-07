# Detecting Downturns in Equity ETFs with Implied Volatility Analysis

A HFT pipeline to detect next-day market downturns using delta-segmented implied volatility features. Backtested on QQQ options from Q1 2020 to Q2 2024, the current implementation centers on an ensemble classifier built from logistic regression, random forest, gradient boosting, and a small PyTorch neural network.

> Original research & [paper](./paper.pdf) produced at Dartmouth Economics Department with Prof. John Welborn & advisory from Prof. Victor Van Erp.

<img width="1896" height="771" alt="image" src="https://github.com/user-attachments/assets/914afd49-c1a6-425e-9cdf-8b72d0078b93" />


---

## Abstract

Implied volatility (IV) skew captures investors’ perceptions of crash risk by reflecting higher im-plied volatilities for deep out-of-the-money (DOTM) puts compared to options closer to at-the-money
(ATM). This project rigorously evaluates the predictive strength of IV skew metrics during the turbulent 2020–2024 period, which spans the COVID-19 crash, the 2020–2021 recovery, and the 2022 tech correction. Rather than relying on broad skew measures, we segment put options by precise delta thresholds to isolate hedging demand for extreme downturns from general volatility expectations. We evaluate two skew metrics: a slope (DOTM–OTM IV difference) and a curvature (DOTM–ATM IV difference) measure. Both are statistically significant predictors of next-day downturns, and the ensemble classifier built on top of them achieves AUC = 0.91, Precision = 89%, Recall = 72%, with end-to-end ETL latency under 50 ms.

---

## Methodology

### 1. Data ingestion
- End-of-day QQQ option quotes and Greeks, Q1 2020 – Q2 2024 (~1.77M records).
- Underlying QQQ price series used to label downturn days.
- Filters applied: `0.05 < IV < 2.0`, `7 ≤ DTE ≤ 180`, non-null deltas.

### 2. Skew & control features (per quote-date × expiry)
Put options are bucketed by delta:

| Bucket | Delta range |
|---|---|
| Deep OTM (DOTM) | `Δ ≤ −0.25` |
| OTM             | `−0.25 < Δ ≤ −0.15` |
| ATM             | `−0.15 < Δ ≤ −0.05` |

Two skew metrics are computed from bucket-mean IVs:

- **Slope:** $\Delta s_{Pdo,o} = \overline{IV}_{\text{DOTM}} - \overline{IV}_{\text{OTM}}$
- **Curvature:** $\Delta s_{Pdo,a} = \overline{IV}_{\text{DOTM}} - \overline{IV}_{\text{ATM}}$

Per-group controls: ATM IV (near-ATM puts), mean put bid-ask spread, total put volume, and DTE.

### 3. Jump target (next-day downturn proxy)
- Daily log returns from underlying close.
- Lee–Mykland-style T-stat on a 30-day rolling vol; a day is labeled a jump if `T < −3` (99% confidence).

### 4. ETL & feature engineering
- Merge skew features with the market series on quote date.
- **Delta-neutralization** of skew vs. contemporaneous returns via 30-day rolling beta.
- **STL decomposition** of the skew time series into trend / seasonal / residual components.
- **Volatility regimes** from rolling-std terciles (low / medium / high) plus a regime-transition flag.
- **Interactions & momentum:** `skew × ATM_IV`, `curvature × ATM_IV`, `regime × skew`, trend/residual ratio, 5-day and 20-day skew momentum.

### 5. Ensemble model
- Base learners: logistic regression, random forest, gradient boosting, PyTorch NN (4-layer MLP with batch norm and dropout).
- 70/30 stratified train/validation split; features standardized with `StandardScaler`.
- Ensemble weights are proportional to each learner's validation AUC (vectorized).
- Decision threshold chosen by maximizing `TPR − FPR` on the validation ROC curve.

### 6. Outputs
- Per-model AUCs and ensemble weights.
- Ensemble AUC, precision, recall.
- Final downturn probability and decision for the most recent date.
- End-to-end ETL latency (ms) for the custom data batch.

---

## How to Use

### Reproduce the research
The full research workflow (data exploration, skew construction, logistic regression tables, ROC curves, and figures used in the paper) lives in `models/final_research.ipynb`. To reproduce:

```bash
git clone https://github.com/aniketxdey/iv-skew.git
cd iv-skew
pip install -r requirements.txt
jupyter notebook models/final_research.ipynb
```

Run the cells top-to-bottom. The notebook expects the QQQ options dataset at `data/qqq_2020_2022.csv`.

### Deploy the pipeline
`models/hft_pipeline.py` is the production-style script: it runs the full ETL, trains the ensemble, and prints a downturn probability and decision for the most recent date.

```bash
python models/hft_pipeline.py
```

To adapt the pipeline to a custom ticker or dataset, edit the following in `models/hft_pipeline.py`:

1. **Dataset path & schema** — update the `pd.read_csv(...)` call and the `rename_dict` to match your data's column names (the defaults are for the CBOE-style QQQ export).
2. **Delta buckets** — adjust the thresholds in `calculate_skew_measures` (`-0.25`, `-0.15`, `-0.05`) if your underlying has a different skew shape (e.g. single names vs. indices).
3. **DTE filter** — change the `(df['DTE'] >= 7) & (df['DTE'] <= 180)` filter to match the maturity range you care about.
4. **Jump detection** — tune the `window` and `critical` arguments in `detect_jumps(...)` to change the lookback and significance threshold (default `T < -3.0`, ~99% confidence).
5. **Class balancing** — update `target_size`, `majority_target`, and `minority_target` to match the size and base rate of your resampled set.
6. **Model hyperparameters** — random forest / gradient boosting / NN hyperparameters are all defined inline in the ensemble section and can be swapped without affecting the ETL.

Everything downstream (STL decomposition, regime features, delta-neutralization, ensemble weighting, threshold selection) is ticker-agnostic and will work as long as the column names and skew buckets are set correctly.

---

## References

1. Lee, S. S., & Mykland, P. A. (2008). *Jumps in financial markets: A new nonparametric test and jump dynamics.*
2. Doran, J., & Krieger, K. (2010). *Implications of implied volatility smirk for stock returns.*
3. Bali, T. G., & Hovakimian, A. (2009). *Volatility spreads and expected stock returns.*
4. Cremers, M., & Weinbaum, D. (2010). *Deviations from put-call parity and stock return predictability.*
5. Xing, Y., Zhang, X., & Zhao, R. (2010). *What does the individual option volatility smirk tell us about future equity returns?*
6. Bollen, N. P. B., & Whaley, R. E. (2004). *Does net buying pressure affect the shape of implied volatility functions?*
7. Pan, J., & Poteshman, A. M. (2006). *The information in option volume for future stock prices.*
