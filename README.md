# stock-market-analysis
Data Science Project: Stock Market Analysis using Python
# 📈 Stock Market Data Science Project
### Decodelabs Data Science Internship — Tasks 3, 4 & 5

---

## 📌 Project Overview

This project was completed as part of the **Decodelabs Data Science Internship**, which focuses on learning data science through hands-on practical tasks. I chose to complete **Tasks 3, 4, and 5** using a stock market dataset covering five major tech stocks — **AAPL, MSFT, GOOGL, AMZN, and TSLA** — from January 2020 to December 2024.

---

## 📂 Project Structure

```
├── stock_data.csv               # Dataset: 6,525 rows of daily OHLCV stock data
├── task3_eda.py                 # Task 3: Exploratory Data Analysis
├── task4_visualization.py       # Task 4: Data Visualization
├── task5_predictive_model.py    # Task 5: Predictive Model
├── closing_prices.png           # Chart: Stock closing prices over time
├── daily_returns_dist.png       # Chart: Daily return distributions
├── volatility_30d.png           # Chart: 30-day rolling volatility
├── correlation_heatmap.png      # Chart: Correlation matrix of daily returns
├── volume_bar.png               # Chart: Average monthly trading volume
├── yearly_growth.png            # Chart: Year-over-year price growth
└── task5_model_results.png      # Chart: Model evaluation results
```

---

## 🗂️ Dataset

**Stocks:** AAPL, MSFT, GOOGL, AMZN, TSLA  
**Period:** 2020-01-01 to 2024-12-31  
**Rows:** 6,525 (1,305 trading days × 5 tickers)  
**Columns:** `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`

The dataset follows the standard Kaggle stock market format with daily OHLCV (Open, High, Low, Close, Volume) data.

---

## ✅ Task 3 — Exploratory Data Analysis (EDA)

**File:** `task3_eda.py`

The goal of this task was to analyze the dataset to discover patterns, trends, and anomalies. The script produces 15 printed insights including:

- Dataset shape, column types, and missing value check
- Descriptive statistics (mean, std, min, max) across all columns
- Average closing price and price range per ticker
- Daily return statistics (mean, std, min, max) per stock
- 30-day rolling volatility comparison across all tickers
- Outlier detection using the IQR method on daily returns
- Correlation matrix of closing prices between stocks
- Yearly average closing prices per ticker
- Best and worst single trading days across the dataset

**Key Findings:**
- TSLA had the highest average 30-day volatility (~3.03%) — more than double any other stock
- GOOGL showed the highest average daily return (0.10%)
- AAPL and GOOGL had a strong positive correlation (0.63), while TSLA was negatively correlated with both
- The worst single trading day was TSLA on 2022-01-24 at -10.27%

---

## ✅ Task 4 — Data Visualization

**File:** `task4_visualization.py`  
**Output:** 6 PNG charts

The goal was to create clear visual representations that communicate insights from the data. I produced the following charts:

| Chart | Description |
|---|---|
| `closing_prices.png` | Line chart showing each stock's price trajectory from 2020–2024 |
| `daily_returns_dist.png` | Histogram of daily return distributions per stock |
| `volatility_30d.png` | Rolling 30-day volatility over time for all 5 stocks |
| `correlation_heatmap.png` | Seaborn heatmap of return correlations between stocks |
| `volume_bar.png` | Average monthly trading volume per stock |
| `yearly_growth.png` | Year-over-year percentage growth per stock (grouped bar chart) |

**Libraries used:** `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## ✅ Task 5 — Predictive Model

**File:** `task5_predictive_model.py`  
**Model:** Random Forest Classifier  
**Target stock:** AAPL

The goal was to build a machine learning model to predict whether a stock's closing price will go **UP or DOWN** the next trading day — framed as a binary classification problem.

### Feature Engineering
15 technical indicators were engineered from raw price and volume data:

| Feature | Description |
|---|---|
| `Return_1d / 5d / 20d` | Short and medium-term price returns |
| `MA_ratio_5_20 / 20_50` | Moving average crossover signals |
| `Volatility_5d / 20d` | Rolling standard deviation of returns |
| `RSI_14` | 14-day Relative Strength Index |
| `BB_pos` | Bollinger Band position |
| `MACD / MACD_sig / MACD_hist` | MACD line, signal, and histogram |
| `Volume_change / Volume_ratio` | Volume momentum indicators |
| `HL_ratio` | Intraday high-low range as % of close |

### Model Training & Evaluation
- **Train/Test split:** 80% train / 20% test (chronological — no data leakage)
- **Cross-validation:** 5-fold TimeSeriesSplit
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Results
- **Test Accuracy:** ~50.4%
- **AUC-ROC:** ~0.49

The results reflect a real-world insight: short-term stock direction is notoriously difficult to predict, and near-random accuracy is consistent with the **Efficient Market Hypothesis**. The model still demonstrated meaningful feature importance — volume ratio, 20-day volatility, and intraday range were the top predictors.

### Simulated Trading Strategy
A simple strategy was simulated — buy when the model predicts UP, hold cash otherwise — and compared against a buy-and-hold baseline to evaluate practical usefulness.

---

## 🛠️ How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the scripts
```bash
# Task 3 – EDA (prints to console)
python task3_eda.py

# Task 4 – Visualization (saves 6 PNG charts)
python task4_visualization.py

# Task 5 – Predictive Model (prints results + saves chart)
python task5_predictive_model.py
```

> All scripts expect `stock_data.csv` to be in the same directory.

---

## 🧰 Technologies Used

- **Python 3**
- **pandas** — data loading, manipulation, groupby analysis
- **numpy** — numerical computations
- **matplotlib** — chart creation and styling
- **seaborn** — statistical visualizations (heatmap)
- **scikit-learn** — machine learning (Random Forest, Logistic Regression, cross-validation, metrics)

---

## 📚 Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Statistical thinking and outlier detection
- Data visualization and storytelling with data
- Feature engineering with financial/technical indicators
- Binary classification with time-series aware validation
- Model evaluation and interpretation

---

## 🏫 Internship

**Organization:** [Decodelabs](https://www.decodelabs.tech)  
**Program:** Data Science Internship  
**Tasks Completed:** Task 3 (EDA), Task 4 (Visualization), Task 5 (Predictive Model)
