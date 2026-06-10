"""
=============================================================
 TASK 5: Predictive Model – Stock Price Direction Classifier
 Dataset : Stock Market Data (AAPL, MSFT, GOOGL, AMZN, TSLA)
 Period  : 2020-01-01 to 2024-12-31
 Model   : Random Forest Classifier
 Goal    : Predict whether tomorrow's close will be UP or DOWN
           using technical features derived from price & volume
 By      : Decodelabs Data Science Internship
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        ConfusionMatrixDisplay,
                                        roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────────────
TARGET_TICKER = "AAPL"          # Change to any ticker in the dataset
TEST_RATIO    = 0.20            # 20% held out for final test
RANDOM_STATE  = 42

# ── 1. LOAD DATA ──────────────────────────────────────────
print("=" * 62)
print("  TASK 5 — STOCK DIRECTION PREDICTOR")
print(f"  Ticker: {TARGET_TICKER}   |   Method: Random Forest Classifier")
print("=" * 62)

df_all = pd.read_csv("stock_data.csv", parse_dates=["Date"])
df = df_all[df_all["Ticker"] == TARGET_TICKER].copy().sort_values("Date").reset_index(drop=True)

print(f"\n[1] Data loaded: {len(df):,} trading days for {TARGET_TICKER}")
print(f"    Period: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────
# Technical indicators commonly used in stock prediction

# Price-based features
df["Return_1d"]   = df["Close"].pct_change(1)           # 1-day return
df["Return_5d"]   = df["Close"].pct_change(5)           # 5-day return
df["Return_20d"]  = df["Close"].pct_change(20)          # 20-day return

# Moving averages
df["MA_5"]        = df["Close"].rolling(5).mean()
df["MA_20"]       = df["Close"].rolling(20).mean()
df["MA_50"]       = df["Close"].rolling(50).mean()
df["MA_ratio_5_20"]  = df["MA_5"] / df["MA_20"]         # Golden/death cross signal
df["MA_ratio_20_50"] = df["MA_20"] / df["MA_50"]

# Volatility
df["Volatility_5d"]  = df["Return_1d"].rolling(5).std()
df["Volatility_20d"] = df["Return_1d"].rolling(20).std()

# Momentum (RSI approximation)
delta    = df["Close"].diff()
gain     = delta.clip(lower=0).rolling(14).mean()
loss     = (-delta.clip(upper=0)).rolling(14).mean()
rs       = gain / (loss + 1e-9)
df["RSI_14"] = 100 - (100 / (1 + rs))

# Bollinger Band position  (0 = lower band, 1 = upper band)
rolling_mean = df["Close"].rolling(20).mean()
rolling_std  = df["Close"].rolling(20).std()
df["BB_pos"] = (df["Close"] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-9)

# MACD
ema_12       = df["Close"].ewm(span=12, adjust=False).mean()
ema_26       = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = ema_12 - ema_26
df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

# Volume features
df["Volume_change"]  = df["Volume"].pct_change()
df["Volume_MA_5"]    = df["Volume"].rolling(5).mean()
df["Volume_ratio"]   = df["Volume"] / df["Volume_MA_5"]

# Price range (intraday)
df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]

# ── 3. TARGET VARIABLE ────────────────────────────────────
# Predict tomorrow's direction: 1 = price goes UP, 0 = price goes DOWN
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# ── 4. CLEAN UP ───────────────────────────────────────────
FEATURES = [
    "Return_1d", "Return_5d", "Return_20d",
    "MA_ratio_5_20", "MA_ratio_20_50",
    "Volatility_5d", "Volatility_20d",
    "RSI_14", "BB_pos",
    "MACD", "MACD_sig", "MACD_hist",
    "Volume_change", "Volume_ratio",
    "HL_ratio"
]

df_model = df[FEATURES + ["Target", "Date"]].dropna().reset_index(drop=True)
X = df_model[FEATURES].values
y = df_model["Target"].values
dates = df_model["Date"].values

print(f"\n[2] Features engineered: {len(FEATURES)}")
print(f"    Usable rows after NaN drop: {len(df_model):,}")
print(f"    Class balance — UP: {y.sum():,} ({y.mean()*100:.1f}%)  "
      f"DOWN: {(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")

# ── 5. TRAIN / TEST SPLIT (time-aware) ────────────────────
split_idx = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test      = dates[split_idx:]

print(f"\n[3] Train/Test split (chronological)")
print(f"    Train: {split_idx:,} rows   |   Test: {len(X_test):,} rows")

# Scale features (important for Logistic Regression)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. MODELS ─────────────────────────────────────────────
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=6,
                                                  min_samples_leaf=20, random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}

print("\n[4] CROSS-VALIDATION RESULTS (5-fold time-series split)")
tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    X_cv = X_train_sc if name == "Logistic Regression" else X_train
    cv_scores = cross_val_score(model, X_cv, y_train, cv=tscv, scoring="accuracy")
    print(f"    {name:<22} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. TRAIN BEST MODEL (Random Forest) ───────────────────
rf = models["Random Forest"]
rf.fit(X_train, y_train)
y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

print("\n[5] FINAL TEST SET RESULTS — Random Forest")
print(f"    AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"],
                             digits=4))

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("[6] TOP 10 FEATURE IMPORTANCES")
print(importances.head(10).round(4).to_string())

# ── 9. VISUALIZATIONS ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Task 5 – {TARGET_TICKER} Direction Prediction (Random Forest)",
             fontsize=13, fontweight="bold")

# (a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontweight="bold")

# (b) Feature Importance (top 10)
importances.head(10).sort_values().plot(kind="barh", ax=axes[1], color="#2196F3")
axes[1].set_title("Top 10 Feature Importances", fontweight="bold")
axes[1].set_xlabel("Importance Score")

# (c) Predicted Probability vs Actual (last 120 test days)
n_plot = 120
axes[2].plot(range(n_plot), y_pred_prob[-n_plot:], color="#9C27B0", linewidth=1.2,
             label="Predicted Prob (UP)")
axes[2].scatter(range(n_plot), y_test[-n_plot:], c=["#4CAF50" if v else "#FF5722" for v in y_test[-n_plot:]],
                alpha=0.5, s=15, zorder=3, label="Actual (1=UP, 0=DOWN)")
axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
axes[2].set_title(f"Predicted Probability (last {n_plot} test days)", fontweight="bold")
axes[2].set_xlabel("Trading Day")
axes[2].set_ylabel("P(UP)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("task5_model_results.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n✅  Saved: task5_model_results.png")

# ── 10. SIMULATED TRADING STRATEGY ────────────────────────
print("\n[7] SIMULATED TRADING STRATEGY (Buy when model predicts UP)")
df_trade = pd.DataFrame({
    "Date":       dates_test,
    "Close":      df.loc[df["Date"].isin(dates_test), "Close"].values[:len(dates_test)],
    "Prediction": y_pred,
    "Actual":     y_test,
})
df_trade["Market_Return"]   = df_trade["Close"].pct_change().fillna(0)
df_trade["Strategy_Return"] = df_trade["Market_Return"] * df_trade["Prediction"].shift(1).fillna(0)

cumulative_market   = (1 + df_trade["Market_Return"]).cumprod().iloc[-1] - 1
cumulative_strategy = (1 + df_trade["Strategy_Return"]).cumprod().iloc[-1] - 1

print(f"    Buy-and-Hold Return : {cumulative_market*100:+.2f}%")
print(f"    Model Strategy Return: {cumulative_strategy*100:+.2f}%")

print("\n" + "=" * 62)
print("  TASK 5 COMPLETE")
print("=" * 62)
