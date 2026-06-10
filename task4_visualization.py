"""
=============================================================
 TASK 4: Data Visualization
 Dataset : Stock Market Data (AAPL, MSFT, GOOGL, AMZN, TSLA)
 Period  : 2020-01-01 to 2024-12-31
 By      : Decodelabs Data Science Internship
=============================================================
 Charts produced (saved as PNG):
   1. closing_prices.png        – Line chart: close prices over time
   2. daily_returns_dist.png    – Histogram: daily return distribution
   3. volatility_30d.png        – Rolling 30-day volatility
   4. correlation_heatmap.png   – Correlation matrix heatmap
   5. volume_bar.png            – Average monthly trading volume
   6. yearly_growth.png         – Year-over-year % growth per stock
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "#f9f9f9",
    "axes.facecolor":    "#f9f9f9",
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
})
COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# ── LOAD ──────────────────────────────────────────────────
df = pd.read_csv("stock_data.csv", parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"])
df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change() * 100
df["Volatility_30d"] = df.groupby("Ticker")["Daily_Return"].transform(
    lambda x: x.rolling(30).std()
)

pivot_close = df.pivot(index="Date", columns="Ticker", values="Close")

# ─────────────────────────────────────────────────────────
# CHART 1 – Closing Prices Over Time
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle("Stock Closing Prices (2020–2024)", fontsize=16, fontweight="bold", y=1.01)

for ax, ticker, color in zip(axes, TICKERS, COLORS):
    data = df[df["Ticker"] == ticker]
    ax.plot(data["Date"], data["Close"], color=color, linewidth=1.2, label=ticker)
    ax.fill_between(data["Date"], data["Close"], alpha=0.08, color=color)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

axes[-1].set_xlabel("Date", fontsize=10)
plt.tight_layout()
plt.savefig("closing_prices.png", bbox_inches="tight")
plt.close()
print("✅  Saved: closing_prices.png")

# ─────────────────────────────────────────────────────────
# CHART 2 – Daily Return Distribution
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
fig.suptitle("Daily Return Distribution per Stock", fontsize=14, fontweight="bold")

for ax, ticker, color in zip(axes, TICKERS, COLORS):
    returns = df[df["Ticker"] == ticker]["Daily_Return"].dropna()
    ax.hist(returns, bins=60, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(returns.mean(), color="black", linewidth=1.2, linestyle="--", label=f"Mean: {returns.mean():.2f}%")
    ax.set_title(ticker, fontsize=11, fontweight="bold")
    ax.set_xlabel("Daily Return (%)", fontsize=8)
    ax.legend(fontsize=7)

axes[0].set_ylabel("Frequency", fontsize=9)
plt.tight_layout()
plt.savefig("daily_returns_dist.png", bbox_inches="tight")
plt.close()
print("✅  Saved: daily_returns_dist.png")

# ─────────────────────────────────────────────────────────
# CHART 3 – 30-Day Rolling Volatility
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for ticker, color in zip(TICKERS, COLORS):
    data = df[df["Ticker"] == ticker].dropna(subset=["Volatility_30d"])
    ax.plot(data["Date"], data["Volatility_30d"], label=ticker, color=color, linewidth=1.3)

ax.set_title("30-Day Rolling Volatility (%) per Stock", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (Std Dev of Daily Return %)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("volatility_30d.png", bbox_inches="tight")
plt.close()
print("✅  Saved: volatility_30d.png")

# ─────────────────────────────────────────────────────────
# CHART 4 – Correlation Heatmap
# ─────────────────────────────────────────────────────────
pivot_ret = df.pivot(index="Date", columns="Ticker", values="Daily_Return")
corr = pivot_ret.corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.5,
    ax=ax, annot_kws={"size": 10}
)
ax.set_title("Correlation of Daily Returns Between Stocks", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("✅  Saved: correlation_heatmap.png")

# ─────────────────────────────────────────────────────────
# CHART 5 – Average Monthly Trading Volume
# ─────────────────────────────────────────────────────────
df["YearMonth"] = df["Date"].dt.to_period("M")
monthly_vol = (
    df.groupby(["YearMonth", "Ticker"])["Volume"]
    .mean()
    .reset_index()
)
monthly_vol["YearMonth"] = monthly_vol["YearMonth"].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
for ticker, color in zip(TICKERS, COLORS):
    data = monthly_vol[monthly_vol["Ticker"] == ticker]
    ax.plot(data["YearMonth"], data["Volume"] / 1e6, label=ticker, color=color, linewidth=1.3)

ax.set_title("Average Monthly Trading Volume (Millions of Shares)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Volume (M shares)")
ax.legend(fontsize=9)
xticks = [i for i in range(0, len(monthly_vol["YearMonth"].unique()), 6)]
ax.set_xticks(xticks)
ax.set_xticklabels(monthly_vol["YearMonth"].unique()[xticks], rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("volume_bar.png", bbox_inches="tight")
plt.close()
print("✅  Saved: volume_bar.png")

# ─────────────────────────────────────────────────────────
# CHART 6 – Year-over-Year % Growth
# ─────────────────────────────────────────────────────────
df["Year"] = df["Date"].dt.year
yearly_first = df.groupby(["Year", "Ticker"])["Close"].first().unstack()
yearly_growth = yearly_first.pct_change() * 100  # % change from prior year
yearly_growth = yearly_growth.dropna()

x = np.arange(len(yearly_growth.index))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 5))
for i, (ticker, color) in enumerate(zip(TICKERS, COLORS)):
    ax.bar(x + i * width, yearly_growth[ticker], width, label=ticker, color=color, alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Year-over-Year Stock Price Growth (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Growth (%)")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(yearly_growth.index)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
plt.tight_layout()
plt.savefig("yearly_growth.png", bbox_inches="tight")
plt.close()
print("✅  Saved: yearly_growth.png")

print("\n🎉 All 6 charts saved successfully!")
