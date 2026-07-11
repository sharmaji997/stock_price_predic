"""
features.py
------------
Shared technical-indicator feature engineering.
Used by BOTH train_model.py (to build the training set) and app.py
(to build the feature row for the live prediction), so training and
inference always see identically-computed features.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (classic Wilder formulation)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands: upper, lower, and %B (position within the band)."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, percent_b.fillna(0.5)


# The exact ordered list of columns the model is trained/predicted on.
# Keeping this in one place guarantees train/inference consistency.
FEATURE_COLUMNS = [
    "sma_10_ratio",
    "sma_50_ratio",
    "ema_10_ratio",
    "ema_50_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_percent_b",
    "bb_width",
    "momentum_10",
    "volatility_10",
    "volume_change",
    "volume_ratio",
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an OHLCV dataframe (columns: Open, High, Low, Close, Volume),
    return a new dataframe with engineered, model-ready features appended.
    Ratios/normalized values are used instead of raw prices so the model
    generalizes across stocks of very different price levels.
    """
    out = df.copy()
    close = out["Close"]
    volume = out["Volume"]

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    out["sma_10_ratio"] = close / sma_10 - 1
    out["sma_50_ratio"] = close / sma_50 - 1
    out["ema_10_ratio"] = close / ema_10 - 1
    out["ema_50_ratio"] = close / ema_50 - 1

    out["rsi_14"] = compute_rsi(close, 14)

    macd_line, signal_line, hist = compute_macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    bb_upper, bb_lower, bb_pct_b = compute_bollinger(close)
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_percent_b"] = bb_pct_b
    out["bb_width"] = (bb_upper - bb_lower) / close

    out["momentum_10"] = close.pct_change(10)
    out["volatility_10"] = close.pct_change().rolling(10).std()

    # volume can legitimately be 0 on thin-trading days (common around
    # holidays on some exchanges) -- guard against divide-by-zero before
    # computing pct_change, or it silently produces +inf.
    safe_volume = volume.replace(0, np.nan)
    out["volume_change"] = safe_volume.pct_change()
    vol_avg_20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_avg_20.replace(0, np.nan)

    # keep helper columns (sma_10, sma_50 etc.) too, useful for charting
    out["sma_10"] = sma_10
    out["sma_50"] = sma_50
    out["ema_10"] = ema_10
    out["ema_50"] = ema_50

    # belt-and-suspenders: any other divide-by-zero edge case (e.g. a
    # near-zero SMA/close) becomes NaN instead of inf, so downstream
    # dropna() calls catch it instead of crashing scikit-learn.
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    return out


def make_labels(df: pd.DataFrame, horizon: int = 5, buy_thresh: float = 0.02,
                 sell_thresh: float = -0.02) -> pd.Series:
    """
    Label each row by looking `horizon` trading days into the future:
      2 = Buy   (future return > buy_thresh)
      1 = Hold  (in between)
      0 = Sell  (future return < sell_thresh)
    This is only used at TRAINING time (needs future data available).
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    labels = pd.Series(1, index=df.index)  # default Hold
    labels[future_return > buy_thresh] = 2
    labels[future_return < sell_thresh] = 0
    labels[future_return.isna()] = np.nan
    return labels


LABEL_NAMES = {0: "Sell", 1: "Hold", 2: "Buy"}
