# House Sale Price Prediction — Linear Regression

Predicting house sale prices with a linear regression baseline, then fixing its
known weakness (underprediction of high-value homes) with a log-transformed
target — a classic, measurable improvement.

**Dataset:** Kaggle [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
(train.csv, 1,460 homes, 6 features)

**Features:** `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`

## Results (test set, 80/20 split)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (raw target) | $39,711 | $25,320 | 0.794 |
| **Log-transformed target** | **$31,874** | **$20,750** | **0.868** |

| Metric | Change |
|---|---|
| RMSE | **−19.7%** |
| MAE | −18.0% |
| R² | +7.3 pts |
| RMSE on homes > $400k (n=7) | **−36%** |

### Why the log transform works

`SalePrice` is right-skewed — most homes cluster low, a few cost far more.
Linear regression minimizes squared error in *dollar* terms, so it bends the
fit toward the expensive outliers and underpredicts them. Training on
`np.log1p(SalePrice)` makes the model minimize error in *log-dollar* terms
(proportional error), which treats a $20k miss on a $100k home like a $200k
miss on a $1M home. Predictions are back-transformed with `np.expm1()`.

The result: the same model family, one principled change, and the known weak
spot (homes > $400k) improves by ~36% RMSE.

## Project structure

```
├── predict.py        # full pipeline: load → split → fit → compare → plot
├── data/train.csv    # Kaggle House Prices train set
├── figures/          # actual-vs-predicted plots (before/after)
└── requirements.txt
```

## How to run

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python predict.py
```
