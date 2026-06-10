"""
=============================================================
 TASK 3: Exploratory Data Analysis (EDA)
 Dataset : Stock Market Data (AAPL, MSFT, GOOGL, AMZN, TSLA)
 Period  : 2020-01-01 to 2024-12-31
 By      : Decodelabs Data Science Internship
=============================================================
"""

import pandas as pd
import numpy as np

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("stock_data.csv", parse_dates=["Date"])
print("=" * 60)
print("  STOCK MARKET – EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 2. BASIC STRUCTURE ────────────────────────────────────
print("\n[1] DATASET SHAPE")
print(f"    Rows: {df.shape[0]:,}   |   Columns: {df.shape[1]}")

print("\n[2] COLUMN NAMES & DATA TYPES")
print(df.dtypes.to_string())

print("\n[3] FIRST 5 ROWS")
print(df.head().to_string(index=False))

print("\n[4] LAST 5 ROWS")
print(df.tail().to_string(index=False))

# ── 3. MISSING VALUES ─────────────────────────────────────
print("\n[5] MISSING VALUES PER COLUMN")
missing = df.isnull().sum()
pct     = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"Missing": missing, "% Missing": pct}).to_string())

# ── 4. DESCRIPTIVE STATISTICS ─────────────────────────────
print("\n[6] DESCRIPTIVE STATISTICS (all tickers combined)")
print(df[["Open", "High", "Low", "Close", "Volume"]].describe().round(2).to_string())

# ── 5. PER-TICKER STATISTICS ──────────────────────────────
print("\n[7] AVERAGE CLOSING PRICE PER TICKER")
avg_close = df.groupby("Ticker")["Close"].mean().round(2).sort_values(ascending=False)
print(avg_close.to_string())

print("\n[8] PRICE RANGE (Max − Min Close) PER TICKER")
price_range = df.groupby("Ticker")["Close"].agg(lambda x: round(x.max() - x.min(), 2))
print(price_range.sort_values(ascending=False).to_string())

# ── 6. DAILY RETURNS ──────────────────────────────────────
df = df.sort_values(["Ticker", "Date"])
df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change() * 100  # in %

print("\n[9] DAILY RETURN STATISTICS (%) PER TICKER")
ret_stats = df.groupby("Ticker")["Daily_Return"].agg(
    Mean=lambda x: x.mean().round(4),
    Std=lambda x: x.std().round(4),
    Min=lambda x: x.min().round(4),
    Max=lambda x: x.max().round(4),
)
print(ret_stats.to_string())

# ── 7. VOLATILITY (30-day rolling std) ────────────────────
df["Volatility_30d"] = df.groupby("Ticker")["Daily_Return"].transform(
    lambda x: x.rolling(30).std()
)
avg_vol = df.groupby("Ticker")["Volatility_30d"].mean().round(4).sort_values(ascending=False)
print("\n[10] AVERAGE 30-DAY ROLLING VOLATILITY (%) PER TICKER")
print(avg_vol.to_string())

# ── 8. OUTLIER DETECTION (IQR method on Daily Return) ─────
print("\n[11] OUTLIER COUNT IN DAILY RETURNS (IQR method)")
for ticker, grp in df.groupby("Ticker"):
    returns = grp["Daily_Return"].dropna()
    Q1, Q3 = returns.quantile(0.25), returns.quantile(0.75)
    IQR     = Q3 - Q1
    outliers = returns[(returns < Q1 - 1.5 * IQR) | (returns > Q3 + 1.5 * IQR)]
    print(f"    {ticker}: {len(outliers)} outlier days "
          f"(worst: {outliers.min():.2f}% / best: {outliers.max():.2f}%)")

# ── 9. CORRELATION MATRIX OF CLOSE PRICES ─────────────────
pivot = df.pivot(index="Date", columns="Ticker", values="Close")
corr  = pivot.corr().round(4)
print("\n[12] CORRELATION MATRIX – CLOSING PRICES")
print(corr.to_string())

# ── 10. YEARLY SUMMARY ────────────────────────────────────
df["Year"] = df["Date"].dt.year
yearly_close = df.groupby(["Year", "Ticker"])["Close"].mean().unstack().round(2)
print("\n[13] YEARLY AVERAGE CLOSE PRICE BY TICKER")
print(yearly_close.to_string())

# ── 11. BEST & WORST TRADING DAYS ─────────────────────────
df_clean = df.dropna(subset=["Daily_Return"])
best  = df_clean.loc[df_clean["Daily_Return"].idxmax()]
worst = df_clean.loc[df_clean["Daily_Return"].idxmin()]
print("\n[14] BEST SINGLE TRADING DAY")
print(f"    {best['Ticker']} on {best['Date'].date()}  →  +{best['Daily_Return']:.2f}%")
print("\n[15] WORST SINGLE TRADING DAY")
print(f"    {worst['Ticker']} on {worst['Date'].date()}  →  {worst['Daily_Return']:.2f}%")

print("\n" + "=" * 60)
print("  EDA COMPLETE — all insights printed above")
print("=" * 60)
