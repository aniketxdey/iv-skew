# Detecting Market Downturns From Implied Volatility Skew

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

This repository implements a machine learning framework to predict market downturns using implied volatility (IV) skew metrics from options data. The methodology combines delta-based volatility skew decomposition with logistic regression, achieving **91% AUC** in forecasting next-day market declines.

![ROC Curve](https://via.placeholder.com/600x400.png?text=ROC+Curve+(AUC+0.91)) <!-- Replace with actual plot from notebook -->

---

## Table of Contents
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Data](#data)
- [Methodology](#methodology)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Key Findings
- **IV Skew Predictiveness**: Both slope (DOTM-OTM IV) and curvature (DOTM-ATM IV) metrics significantly predict downturns:
  - Slope coefficient: **3.195** (p < 0.001)
  - Curvature coefficient: **3.226** (p < 0.001)
- **Model Performance**: Achieved **0.91 AUC** with 85% true positive rate at 10% false positive rate
- **Volume Paradox**: Higher trading volume unexpectedly correlated with lower crash probability (-2.26 coefficient)

---

## Installation
```bash
git clone https://github.com/yourusername/iv-skew-prediction.git
cd iv-skew-prediction
pip install -r requirements.txt
```

---

## Data
### Dataset
- **QQQ Options Data**: Daily put/call options for Nasdaq-100 ETF (Jan 2020 - Jun 2024)
- Includes: IV, Greeks, bid/ask prices, volume, strike details
- [Sample Data Structure](data/qqq_2020_2022.csv)

### Preprocessing
```python
# Load and clean data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/qqq_2020_2022.csv')
df.columns = df.columns.str.strip()
df['DTE'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days

# Convert numeric fields
numeric_cols = ['C_BID', 'C_IV', 'P_BID', 'P_IV']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

---

## Methodology
### 1. Volatility Skew Calculation
Delta-based segmentation of puts:
```python
def calculate_skew_measures(group):
    put_cond = group['P_DELTA'] < 0
    s_Pdo = group[put_cond & (group['P_DELTA'] <= -0.25)]['P_IV'].mean()
    s_Po = group[put_cond & (group['P_DELTA'] > -0.25) & (group['P_DELTA'] <= -0.15)]['P_IV'].mean()
    return pd.Series({
        'Δs_Pdo_o': s_Pdo - s_Po,  # Slope
        'Δs_Pdo_a': s_Pdo - s_Pa   # Curvature
    })
```

### 2. Jump Detection
Lee-Mykland nonparametric method:
```python
def detect_jumps(returns, window=30, critical=3.0):
    local_vol = returns.rolling(window).std()
    T_stats = returns / local_vol.shift(1)
    return (T_stats < -critical).astype(int)
```

### 3. Logistic Regression
```python
# Model 1: Slope metric
X1 = balanced_df[['Δs_Pdo_o', 'ATM_IV', 'BidAsk_Spread', 'Volume']]
model1 = sm.Logit(y1, X1_scaled).fit()

# Model 2: Curvature metric
X2 = balanced_df[['Δs_Pdo_a', 'ATM_IV', 'BidAsk_Spread', 'Volume']]
model2 = sm.Logit(y2, X2_scaled).fit()
```

---

## Results
### Regression Output
| Metric       | Coefficient | P-value | VIF  |
|--------------|-------------|---------|------|
| **Δs_Pdo_o** | 3.195       | <0.001  | 2.18 |
| Volume       | -2.168      | <0.001  | 2.07 |

![Coefficient Plot](https://via.placeholder.com/600x300.png?text=Coefficient+Magnitudes) <!-- Replace with actual plot -->

### Performance Metrics
- **AUC**: 0.91
- **Recall**: 72%
- **Precision**: 89%

---

## Usage
1. Mount Google Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Run the pipeline:
```python
# Full analysis notebook:
final_IV_regression.ipynb
```

---

## References
1. Lee & Mykland (2008) - Jump detection framework
2. Doran & Krieger (2010) - Delta-based skew metrics
3. Bali & Hovakimian (2009) - Crash risk pricing
4. Full citation list in [REFERENCES.md](REFERENCES.md)

---

**Note**: Replace placeholder images with actual plots from the notebook. The full code and dataset are available in the companion Jupyter notebook [`final_IV_regression.ipynb`](final_IV_regression.ipynb).
```
