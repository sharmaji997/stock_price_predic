# 📈 Stock Buy / Sell / Hold Signal Predictor

A machine learning web app that predicts a **Buy / Hold / Sell** rating
for any stock ticker, using real historical price data (via **yfinance**),
technical-indicator feature engineering, a **scikit-learn Random Forest
Classifier**, and a **Streamlit** UI with interactive Plotly charts.

Built with the same tech stack as the reference
[car-price-prediction](https://github.com/sharmaji997/car-price-prediction)
project (Python + pandas/numpy + scikit-learn RandomForest + Streamlit),
adapted from price *regression* to Buy/Hold/Sell *classification*.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## 🧠 Overview

Given a stock ticker, the app pulls recent price history, computes
technical indicators (moving averages, RSI, MACD, Bollinger Bands,
momentum, volatility, volume trends), and feeds them into a trained
Random Forest model that predicts whether the stock's next-5-trading-day
outlook looks like a **Buy**, **Hold**, or **Sell**, along with a
confidence score, an interactive candlestick chart, and a multi-stock
watchlist scanner.

## 📁 Project Structure

```
stock-signal-predictor/
├── app.py                 # Streamlit application (3 tabs: analysis, watchlist, model insights)
├── train_model.py         # Downloads data, engineers features, trains & saves the model
├── features.py            # Shared technical indicator + labeling logic (train & app both use this)
├── config.py               # Ticker universe, thresholds, lookback settings
├── requirements.txt
├── model/                  # created by train_model.py
│   ├── signal_model.pkl        # trained RandomForestClassifier
│   ├── scaler.pkl               # StandardScaler for numeric features
│   ├── feature_columns.pkl      # exact feature order expected by the model
│   └── metrics.pkl              # accuracy / classification report / confusion matrix
└── README.md
```

## 📊 Data & Labels

Data comes live from Yahoo Finance via `yfinance` — no static CSV needed,
so the model always trains on up-to-date history. For each trading day,
we look `HORIZON_DAYS` (default 5) days into the future:

| Future 5-day return       | Label  |
| -------------------------- | ------ |
| > +2%                      | Buy    |
| between -2% and +2%        | Hold   |
| < -2%                      | Sell   |

## ⚙️ How It Works

1. **Data collection** — `train_model.py` downloads ~3 years of OHLCV
   history for ~25 stocks (mix of US and NSE-listed) via `yfinance`.
2. **Feature engineering** (`features.py`) — computes SMA/EMA ratios,
   RSI(14), MACD + signal + histogram, Bollinger %B and band width,
   10-day momentum and volatility, and volume-change/volume-ratio —
   14 features total, all normalized so the model generalizes across
   stocks at very different price levels.
3. **Labeling** — each row is labeled Buy/Hold/Sell based on the
   *actual* forward 5-day return (see table above).
4. **Training** — a `RandomForestClassifier` (300 trees, max depth 10,
   class-balanced) is trained on an 80/20 stratified split.
5. **App** (`app.py`) — loads the saved artifacts and exposes:
   - **Stock Analysis tab** — candlestick chart with SMA/Bollinger
     overlays, MACD and RSI subplots, volume chart, and the model's
     live Buy/Hold/Sell prediction with confidence.
   - **Watchlist Suggestions tab** — scans a list of tickers (default
     watchlist or your own comma-separated list) and ranks them by
     signal, so you get "many other suggestions" at a glance.
   - **Model Insights tab** — test accuracy, classification report,
     confusion matrix, and a feature-importance chart.

## 🚀 Run Locally

```bash
# 1. Set up
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the model (downloads data — needs internet access)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **"New app"**, pick the repo/branch, set main file to `app.py`.
4. Add a step (or GitHub Action) to run `train_model.py` before deploy,
   or commit the `model/*.pkl` files directly (they're small, well
   under GitHub's file-size limits — no Git LFS needed).

## 📌 Notes for Your Class Presentation

- Explain *why* ratios/normalized indicators (e.g. `close/SMA - 1`)
  are used instead of raw prices — it lets one model work across
  stocks with wildly different price levels (e.g. a ₹200 stock vs a
  ₹4,000 stock).
- You can genuinely discuss the labeling design choice: it's a
  **classification** problem (Buy/Hold/Sell) built on top of a
  **regression** quantity (forward return), thresholded into classes —
  good material for a Q&A.
- Easy extensions for extra credit: try `GradientBoostingClassifier` or
  `XGBoost`, tune the Buy/Sell thresholds or horizon in `config.py`,
  add more indicators (e.g. ATR, OBV), or add a backtest that shows
  cumulative returns if you'd followed the model's signals historically.
- **Disclaimer to include in your presentation**: this is a supervised
  learning project on historical patterns, not investment advice —
  markets are influenced by far more than technical indicators.

## 🛠️ Tech Stack

- **Python 3.9+**
- **yfinance** — live historical stock data
- **scikit-learn** — Random Forest Classifier, preprocessing
- **pandas / numpy** — data handling & feature engineering
- **Streamlit** — web UI
- **Plotly** — interactive candlestick/indicator charts

## 📄 License

MIT — feel free to use this for your coursework.
