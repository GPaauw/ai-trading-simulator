"""LightGBM Training Script — draait via GitHub Actions.

Downloads 2 jaar historische data, berekent features, en traint:
1. Een classifier (5 klassen: strong_sell → strong_buy)
2. Een regressor (verwachte return %)

Output: model artifacts in backend/ml/models/
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

# Voeg backend toe aan path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.config import (
    FEATURE_COLUMNS, SIGNAL_CLASSES, LGBM_PARAMS,
    HISTORY_YEARS, LABEL_HORIZON_SHORT, LABEL_HORIZON_LONG,
    CV_SPLITS, LABEL_THRESHOLDS, ML_MODELS_DIR,
    SIGNAL_MODEL_PATH, RETURN_MODEL_PATH,
)
from services.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_data(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Download historische OHLCV data via yfinance."""
    import yfinance as yf

    logger.info("Downloading data voor %d symbolen...", len(symbols))
    data = {}

    # Batch download
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            df = yf.download(batch, period=period, group_by="ticker", progress=False, threads=True)
            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df.copy()
                    else:
                        sym_df = df[sym].copy()
                    sym_df = sym_df.dropna()
                    sym_df.columns = [c.lower() for c in sym_df.columns]
                    if len(sym_df) >= 100:
                        data[sym] = sym_df
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Batch download mislukt: %s", exc)

    logger.info("Data ontvangen voor %d symbolen.", len(data))
    return data


def create_labels(close_prices: pd.Series, horizon: int = 5) -> pd.Series:
    """Creëer labels op basis van toekomstige prijsverandering."""
    future_return = close_prices.pct_change(periods=horizon).shift(-horizon)

    labels = pd.Series(index=close_prices.index, dtype=int)
    labels[future_return <= LABEL_THRESHOLDS["strong_sell"]] = 0  # strong_sell
    labels[(future_return > LABEL_THRESHOLDS["strong_sell"]) & (future_return <= LABEL_THRESHOLDS["sell"])] = 1  # sell
    labels[(future_return > LABEL_THRESHOLDS["sell"]) & (future_return <= LABEL_THRESHOLDS["hold"])] = 2  # hold
    labels[(future_return > LABEL_THRESHOLDS["hold"]) & (future_return <= LABEL_THRESHOLDS["buy"])] = 3  # buy
    labels[future_return > LABEL_THRESHOLDS["buy"]] = 4  # strong_buy

    return labels, future_return


def build_dataset(
    histories: dict[str, pd.DataFrame],
    feature_engine: FeatureEngine,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Bouw training dataset: features + labels voor alle symbolen over tijd."""
    all_features = []
    all_labels = []
    all_returns = []

    for symbol, hist in histories.items():
        if len(hist) < 100:
            continue

        # Bereken features voor elk tijdstip (rolling window)
        for end_idx in range(60, len(hist) - horizon):
            window = hist.iloc[:end_idx + 1]
            features = feature_engine.compute_features(window)
            if features is None:
                continue

            # Label: toekomstige return
            current_close = float(hist.iloc[end_idx]["close"])
            future_close = float(hist.iloc[end_idx + horizon]["close"])
            future_return = (future_close - current_close) / current_close

            # Discretiseer naar klasse
            if future_return <= LABEL_THRESHOLDS["strong_sell"]:
                label = 0
            elif future_return <= LABEL_THRESHOLDS["sell"]:
                label = 1
            elif future_return <= LABEL_THRESHOLDS["hold"]:
                label = 2
            elif future_return <= LABEL_THRESHOLDS["buy"]:
                label = 3
            else:
                label = 4

            all_features.append(features)
            all_labels.append(label)
            all_returns.append(future_return)

    if not all_features:
        raise ValueError("Geen trainingsdata gegenereerd")

    X = pd.DataFrame(all_features)[FEATURE_COLUMNS]
    y = pd.Series(all_labels, name="label")
    returns = pd.Series(all_returns, name="return")

    logger.info("Dataset: %d samples, %d features", len(X), len(FEATURE_COLUMNS))
    logger.info("Label verdeling: %s", dict(y.value_counts().sort_index()))

    return X, y, returns


def train_classifier(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Train LightGBM classifier met TimeSeriesSplit cross-validatie."""
    logger.info("Training classifier...")

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        logger.info("  Fold %d: accuracy=%.4f", fold + 1, acc)

    logger.info("Gemiddelde CV accuracy: %.4f (±%.4f)", np.mean(scores), np.std(scores))

    # Finale training op alle data
    final_model = lgb.LGBMClassifier(**{k: v for k, v in LGBM_PARAMS.items() if k != "early_stopping_rounds"})
    final_model.fit(X, y)

    return final_model


def train_regressor(X: pd.DataFrame, returns: pd.Series) -> lgb.LGBMRegressor:
    """Train LightGBM regressor voor verwachte return voorspelling."""
    logger.info("Training return regressor...")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "n_estimators": 300,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X, returns)

    preds = model.predict(X)
    rmse = np.sqrt(np.mean((preds - returns.values) ** 2))
    logger.info("Return model RMSE: %.6f", rmse)

    return model


def get_training_symbols() -> list[str]:
    """Haal een lijst van symbolen om op te trainen."""
    # Top 50 S&P 500 + populaire crypto + commodities
    stocks = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "WMT", "MCD", "CSCO",
        "TMO", "ACN", "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "UNP",
        "RTX", "LOW", "HON", "ORCL", "AMGN", "COP", "BMY", "QCOM", "INTC",
        "AMD", "NFLX", "SBUX", "INTU", "CAT",
    ]
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
              "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD"]
    commodities = ["GC=F", "SI=F", "CL=F", "NG=F"]

    return stocks + crypto + commodities


def main():
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    symbols = get_training_symbols()
    histories = download_data(symbols, period="2y")

    if len(histories) < 10:
        logger.error("Te weinig data ontvangen (%d symbolen). Training afgebroken.", len(histories))
        sys.exit(1)

    fe = FeatureEngine()

    # Dataset bouwen met 5-daagse horizon (swing trading)
    logger.info("=== Building dataset (5d horizon) ===")
    X, y, returns = build_dataset(histories, fe, horizon=5)

    # Train classifier
    classifier = train_classifier(X, y)

    # Train return regressor
    regressor = train_regressor(X, returns)

    # Opslaan
    joblib.dump(classifier, SIGNAL_MODEL_PATH)
    logger.info("Classifier opgeslagen: %s", SIGNAL_MODEL_PATH)

    joblib.dump(regressor, RETURN_MODEL_PATH)
    logger.info("Regressor opgeslagen: %s", RETURN_MODEL_PATH)

    # Feature importance rapport
    importance = classifier.feature_importance(importance_type="gain")
    total = sum(importance)
    fi = {FEATURE_COLUMNS[i]: round(float(importance[i]) / total, 4) for i in range(len(FEATURE_COLUMNS))}
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    logger.info("Feature importance (top 10):")
    for feat, imp in list(fi_sorted.items())[:10]:
        logger.info("  %s: %.4f", feat, imp)

    # Sla rapport op
    report = {
        "trained_at": datetime.now().isoformat(),
        "symbols_count": len(histories),
        "samples_count": len(X),
        "feature_importance": fi_sorted,
    }
    report_path = os.path.join(ML_MODELS_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training rapport: %s", report_path)

    logger.info("=== Training voltooid ===")


if __name__ == "__main__":
    main()
