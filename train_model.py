"""
train_model.py
---------------
Downloads historical OHLCV data for a basket of stocks (config.TRAIN_TICKERS),
engineers technical-indicator features (features.py), labels each row
Buy/Hold/Sell based on the actual forward return, and trains a
RandomForestClassifier -- the same modelling approach (scikit-learn +
Random Forest) as the reference car-price project, just a Classifier
here instead of a Regressor since we're predicting a category
(Buy/Hold/Sell) instead of a number.

Run this once before launching the Streamlit app:
    python train_model.py

Saves to model/:
    signal_model.pkl     trained RandomForestClassifier
    scaler.pkl            StandardScaler fit on the training features
    feature_columns.pkl   exact ordered feature list the model expects
    metrics.pkl            accuracy / classification report / confusion matrix
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (BUY_THRESHOLD, HISTORY_PERIOD, HORIZON_DAYS,
                     SELL_THRESHOLD, TRAIN_TICKERS)
from features import FEATURE_COLUMNS, add_technical_indicators, make_labels

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)


def build_dataset() -> pd.DataFrame:
    """Download + featurize + label every ticker, then concatenate."""
    frames = []
    for ticker in TRAIN_TICKERS:
        print(f"Downloading {ticker} ...")
        try:
            df = yf.download(ticker, period=HISTORY_PERIOD, progress=False,
                              auto_adjust=True)
        except Exception as exc:
            print(f"  skipped {ticker}: {exc}")
            continue

        if df is None or df.empty or len(df) < 100:
            print(f"  skipped {ticker}: not enough data")
            continue

        # yfinance sometimes returns MultiIndex columns for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = add_technical_indicators(df)
        df["label"] = make_labels(df, HORIZON_DAYS, BUY_THRESHOLD, SELL_THRESHOLD)
        df["ticker"] = ticker
        frames.append(df)

    full = pd.concat(frames, axis=0)
    full = full.dropna(subset=FEATURE_COLUMNS + ["label"])
    return full


def main():
    dataset = build_dataset()
    print(f"\nTotal training rows: {len(dataset)}")
    print("Label distribution:\n", dataset["label"].value_counts())

    X = dataset[FEATURE_COLUMNS].values
    y = dataset["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.3f}")
    print(report)

    feature_importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
    }

    with open(MODEL_DIR / "signal_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
