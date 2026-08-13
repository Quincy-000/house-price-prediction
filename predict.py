"""House Sale Price Prediction — Linear Regression baseline vs log-transformed target.

Pipeline:
  1. Load Kaggle House Prices train.csv (1,460 homes)
  2. Select 6 features, drop missing values
  3. Train/test split (80/20, fixed seed)
  4. Baseline: LinearRegression on raw SalePrice
  5. Improved: LinearRegression on log1p(SalePrice), predictions back-transformed with expm1
  6. Compare RMSE / MAE / R^2 and save actual-vs-predicted plots

Run:  ./venv/bin/python predict.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "train.csv")
FIG = os.path.join(BASE, "figures")

FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
TARGET = "SalePrice"

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[FEATURES + [TARGET]].dropna()
    print(f"Loaded {len(df)} homes (after dropping missing values)")
    return df


def evaluate(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"  {name:<28} RMSE ${rmse:>9,.0f}   MAE ${mae:>8,.0f}   R^2 {r2:.3f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def fit_predict(X_train, y_train, X_test, y_test, log_target=False):
    """Fit LinearRegression (optionally on log-transformed target) and return
    predictions in ORIGINAL price units, plus the fitted model/scaler."""
    scaler = StandardScaler().fit(X_train)
    X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)

    y_fit = np.log1p(y_train) if log_target else y_train
    model = LinearRegression().fit(X_tr, y_fit)

    preds = model.predict(X_te)
    if log_target:
        preds = np.expm1(preds)  # back-transform to dollars
    return preds, model, scaler


def plot_actual_vs_pred(y_test, preds, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, preds, s=14, alpha=0.55, color="#1f77b4")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", lw=1.2, label="perfect prediction")
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted sale price ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Split: {len(X_train)} train / {len(X_test)} test\n")

    print("Results (test set):")
    print("  " + "-" * 62)

    # Baseline — raw target
    preds_base, _, _ = fit_predict(X_train, y_train, X_test, y_test, log_target=False)
    baseline = evaluate(y_test, preds_base, "Baseline (raw target)")

    # Improved — log-transformed target
    preds_log, _, _ = fit_predict(X_train, y_train, X_test, y_test, log_target=True)
    improved = evaluate(y_test, preds_log, "Log-transform (log1p)")

    print("\nSummary:")
    print(f"  RMSE  {baseline['rmse']:,.0f} -> {improved['rmse']:,.0f}  ({(improved['rmse']/baseline['rmse']-1)*100:+.1f}%)")
    print(f"  MAE   {baseline['mae']:,.0f} -> {improved['mae']:,.0f}  ({(improved['mae']/baseline['mae']-1)*100:+.1f}%)")
    print(f"  R^2   {baseline['r2']:.3f} -> {improved['r2']:.3f}  ({(improved['r2']-baseline['r2'])*100:+.2f} pts)")

    # High-value homes (>$400k) — the known weak spot
    mask = y_test > 400_000
    if mask.sum():
        print(f"\nHigh-value homes (>$400k, n={mask.sum()}):")
        for name, preds in [("Baseline", preds_base), ("Log-transform", preds_log)]:
            err = np.sqrt(mean_squared_error(y_test[mask], preds[mask]))
            print(f"  {name:<14} RMSE ${err:>10,.0f}")

    plot_actual_vs_pred(y_test, preds_base, "Baseline: linear regression on raw target", os.path.join(FIG, "baseline.png"))
    plot_actual_vs_pred(y_test, preds_log, "Improved: linear regression on log-transformed target", os.path.join(FIG, "log_transform.png"))


if __name__ == "__main__":
    main()
