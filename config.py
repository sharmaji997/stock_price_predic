"""
config.py
---------
Central place to edit the stock universe. TRAIN_TICKERS is the set of
symbols used to build the training dataset (more tickers + more history
= a more robust model). WATCHLIST is what the app scans by default in
"Suggestions" mode. Edit both freely for your class demo.

Works with any yfinance-supported symbol. Indian NSE stocks need a
".NS" suffix (e.g. "RELIANCE.NS", "TCS.NS").
"""

TRAIN_TICKERS = [
    # US large-caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "V", "WMT", "DIS", "NFLX", "KO", "PEP", "XOM", "PFE",
    # Indian large-caps (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "HINDUNILVR.NS",
]

WATCHLIST = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS",
]

HISTORY_PERIOD = "3y"      # how much history to pull for training
HORIZON_DAYS = 5           # look this many trading days ahead for the label
BUY_THRESHOLD = 0.02       # future 5-day return > +2%  -> Buy
SELL_THRESHOLD = -0.02     # future 5-day return < -2%  -> Sell
