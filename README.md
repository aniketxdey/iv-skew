# Detecting Downturns in Equity ETFs with Deep Learning & Implied Volatility Skew
> Original research & [paper](./paper.pdf) produced at Dartmouth Economics Department with Professor John Welborn

<img width="1896" height="771" alt="image" src="https://github.com/user-attachments/assets/914afd49-c1a6-425e-9cdf-8b72d0078b93" />


---

## Introduction

Implied volatility skew reflects how option prices vary across strikes and can reveal the market price of downside protection. This study asks whether delta-segmented, moment-based measures of risk-neutral skew are associated with next-day downside jumps in technology-sector ETFs during the volatile 2020–2024 period. Risk-neutral moments are calculated from an SVI-parameterized volatility surface following Bakshi, Kapadia, and Madan (2003). Delta restrictions then focus the measures on selected regions of the surface: a put-side slope signal near the 10-delta wing and a curvature signal spanning the 25-delta put and call regions. We label downside jumps using a daily adaptation of the Lee–Mykland statistic and evaluate each signal in a separate logistic regression with market controls. Both signals have positive, statistically significant coefficients, and the curvature model attains a peak AUC of 0.91 in prediction accuracy for the labeled downturn events. 

---

## Methodology

### 1. Data ingestion
The dataset contains end-of-day QQQ option quotes and Greeks from Q1 2020 through Q2 2024, totaling approximately 1.77 million records. We pair the options data with QQQ closing prices to construct the downside-jump label. The sample retains quotes with `0.05 < IV < 2.0`, 7 to 180 days to expiration, and non-null deltas.

### 2. Skew & control features (per quote-date × expiry)
We smooth the implied volatility surface with an SVI parameterization and compute risk-neutral moments following Bakshi, Kapadia, and Madan (2003). Applying delta restrictions to the moment-based skew combines information from the risk-neutral distribution with the concentration of option pricing in selected regions of the surface.

| Signal | Delta region | Interpretation |
|---|---|---|
| Slope | Put side, from ATM to the 10-delta wing (`−0.10 ≤ Δ < 0`) | Measures risk-neutral skew variation across the left-side put region. |
| Curvature | 25-delta put and call region (`−0.25 ≤ Δ ≤ 0.25`) | Measures risk-neutral skew curvature across the conventional risk-reversal region. |

The slope and curvature signals are moment-based measures computed from risk-neutral prices within these delta regions. The regressions control for at-the-money implied volatility, mean bid–ask spread, and option volume.

### 3. Jump target (next-day downturn proxy)
We calculate daily log returns from the underlying closing prices and standardize them using a 30-day rolling volatility estimate. A downside event is labeled when the resulting Lee–Mykland-style statistic falls below `−3`. This daily implementation serves as the study’s jump target.

### 4. ETL & feature engineering
We align each date’s option-derived signals and controls with the market series by quote date, then pair them with the next-day jump label. This timing makes the skew measures precede the event they are used to explain.

### 5. Ensemble model
- Base learners: logistic regression, random forest, gradient boosting, PyTorch NN (4-layer MLP with batch norm and dropout).
- 70/30 stratified train/validation split; features standardized with `StandardScaler`.
- Ensemble weights are proportional to each learner's validation AUC (vectorized).
- Decision threshold chosen by maximizing `TPR − FPR` on the validation ROC curve.

### 6. Outputs
The paper reports each signal’s coefficient and statistical significance, along with model fit statistics and ROC performance. Both specifications attain an AUC of 0.91 in the reported evaluation.


## Deployment
The full research workflow (data exploration, skew construction, logistic regression tables, ROC curves, and figures used in the paper) lives in `models/final_research.ipynb`. To reproduce:

```bash
git clone https://github.com/aniketxdey/iv-skew.git
cd iv-skew
pip install -r requirements.txt
jupyter notebook models/final_research.ipynb
```

Run the cells top-to-bottom. The notebook expects the QQQ options dataset at `data/qqq_2020_2022.csv`, however, everything downstream (STL decomposition, regime features, delta-neutralization, ensemble weighting, threshold selection) is ticker-agnostic and will work as long as the column names and skew buckets are set correctly.

---

## References

1. Lee, S. S., & Mykland, P. A. (2008). *Jumps in financial markets: A new nonparametric test and jump dynamics.*
2. Doran, J., & Krieger, K. (2010). *Implications of implied volatility smirk for stock returns.*
3. Bali, T. G., & Hovakimian, A. (2009). *Volatility spreads and expected stock returns.*
4. Cremers, M., & Weinbaum, D. (2010). *Deviations from put-call parity and stock return predictability.*
5. Xing, Y., Zhang, X., & Zhao, R. (2010). *What does the individual option volatility smirk tell us about future equity returns?*
6. Bollen, N. P. B., & Whaley, R. E. (2004). *Does net buying pressure affect the shape of implied volatility functions?*
7. Pan, J., & Poteshman, A. M. (2006). *The information in option volume for future stock prices.*
