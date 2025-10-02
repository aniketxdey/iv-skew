# Implied Volatility Skew Analysis to Detect Market Downturns

A production-style pipeline to detect next-day market downturns using delta-segmented implied-volatility (IV) skew features from QQQ options. Current implementation centers on an ensemble classifier built from logistic regression, random forest, gradient boosting, and a small PyTorch neural network.

**Performance on QQQ, 2020–2022:** AUC 0.91, Precision 0.89, Recall 0.72. full ETL latency is typically <50 ms on the provided sample for end-to-end pipeline.


##  hft_pipeline.py

### 1) Skew & Controls (per quote-date × expiry)

Put deltas are bucketed:
* Deep OTM (Δ ≤ −0.25), OTM (−0.25 < Δ ≤ −0.15), ATM (−0.15 < Δ ≤ −0.05)

Skew measures:
* **Slope:**  
  $\Delta s_{Pdo,o} = \overline{IV}\cdot{\text{deep OTM put}} - \overline{IV}\cdot{\text{OTM put}}$
* **Curvature:**  
  $\Delta s_{Pdo,a} = \overline{IV}\cdot{\text{deep OTM put}} - \overline{IV}\cdot{\text{ATM put}}$

Additional controls (per group):
* ATM IV (proxied by near-ATM puts)
* Mean put bid-ask spread
* Total put volume
* DTE

### 2) Jump Target (next-day downturn proxy)
* Construct daily log returns from underlying close
* Jump detection: Lee–Mykland-style T-stat on rolling 30-day vol; label jump if (T < -3)

### 3) ETL & Feature Engineering
* Merge skew features with market series
* Delta-neutralization of skew vs. contemporaneous returns (30-day rolling beta)
* STL decomposition of the skew time series (trend/seasonal/residual)
* Volatility regimes via rolling std. terciles (low/medium/high) and transition flag
* Interactions and momentum:
  * $(\text{skew} \times \text{ATM}\_IV)$
  * $(\text{curvature} \times \text{ATM}\_IV)$
  * regime × skew, trend/residual ratio
  * 5-day and 20-day skew momentum

### 4) Ensemble Model 
* Learners: logistic regression, random forest, gradient boosting, PyTorch NN
* Train/validation split stratified 70/30
* Model weights proportional to each model’s validation AUC via vectorized operations
* Decision threshold chosen by maximizing $( \text{TPR} - \text{FPR} )$ on the validation ROC

### 5) Outputs
* Per-model AUCs and ensemble weights
* Ensemble AUC, Precision, Recall
* Final downturn probability and decision for the most recent date
* ETL latency in milliseconds tied for end-to-end sample on custom data batch

---

## Quick Start

```bash
git clone https://github.com/yourusername/iv-skew.git
cd iv-skew
pip install -r requirements.txt
python "models/hft_pipeline.py"
```

---

## Literature

1. Lee, S. S., & Mykland, P. A. (2008). Jumps in financial markets: A new nonparametric test.
2. Doran, J., & Krieger, K. (2010). Implications of implied volatility smirk for stock returns.
3. Bali, T. G., & Hovakimian, A. (2009). Volatility spreads and expected stock returns.

