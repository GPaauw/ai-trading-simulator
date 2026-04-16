"""Ensemble Training Script — traint het 3-window LightGBM ensemble.

Workflow:
1. Download historische OHLCV data (yfinance)
2. Bereken v2 features (FeatureEngine)
3. Genereer labels met meerdere horizons (3, 5, 10 dagen)
4. Selecteer optimale horizon per asset-type (beste S/N ratio)
5. Train 3 classifiers (3m, 1y, 2y lookback windows)
6. Train 1 regressor (verwachte return)
7. SMOTE voor class imbalance (optioneel)
8. ChampionChallenger: vergelijk met bestaand model
9. SHAP feature importance + drift baseline opslaan

Kan handmatig of via scheduler (wekelijks) worden uitgevoerd.

Gebruik:
    python -m ml.training.train_ensemble
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Voeg backend toe aan path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.config import (
    FEATURE_COLUMNS_V2,
    SIGNAL_CLASSES,
    NUM_CLASSES,
    CV_SPLITS,
    LABEL_THRESHOLDS,
    ML_MODELS_DIR,
)
from ml.ensemble_lgbm import (
    EnsembleLightGBM,
    ModelMetrics,
    _WINDOWS,
    _LABEL_HORIZONS,
    _ENSEMBLE_DIR,
)
from services.feature_engine import FeatureEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Hyperparameters per window ──────────────────────────────────────
_LGBM_PARAMS: Dict[str, Dict[str, Any]] = {
    "3m": {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.08,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "n_estimators": 300,
        "class_weight": "balanced",
        "verbose": -1,
        "random_state": 42,
    },
    "1y": {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 500,
        "class_weight": "balanced",
        "verbose": -1,
        "random_state": 42,
    },
    "2y": {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 800,
        "class_weight": "balanced",
        "verbose": -1,
        "random_state": 42,
    },
}


# ── Data Download ───────────────────────────────────────────────────

def download_data(symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Download historische OHLCV data via yfinance."""
    import yfinance as yf

    logger.info("Downloading data voor %d symbolen (period=%s)...", len(symbols), period)
    data: Dict[str, pd.DataFrame] = {}

    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            df = yf.download(
                batch, period=period, group_by="ticker",
                progress=False, threads=True,
            )
            for sym in batch:
                try:
                    sym_df = df[sym].copy() if len(batch) > 1 else df.copy()
                    sym_df = sym_df.dropna()
                    sym_df.columns = [c.lower() for c in sym_df.columns]
                    if len(sym_df) >= 100:
                        data[sym] = sym_df
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Batch download mislukt: %s", exc)

    logger.info("Data ontvangen voor %d/%d symbolen.", len(data), len(symbols))
    return data


# ── Label Generatie ─────────────────────────────────────────────────

def select_best_horizon(
    close: pd.Series,
    horizons: List[int],
) -> int:
    """Selecteer de horizon met de beste signaal/ruis-verhouding.

    Berekent voor elke horizon de ratio van gemiddeld absoluut rendement
    tot standaarddeviatie van het rendement. Hogere ratio = beter signaal.

    Args:
        close: Close-prijzen reeks.
        horizons: Lijst van horizons om te evalueren.

    Returns:
        Optimale horizon in dagen.
    """
    best_horizon = horizons[1]  # default: middelste
    best_snr = 0.0

    for h in horizons:
        returns = close.pct_change(periods=h).dropna()
        if len(returns) < 50:
            continue

        mean_abs = float(returns.abs().mean())
        std = float(returns.std())

        if std > 1e-9:
            snr = mean_abs / std
            if snr > best_snr:
                best_snr = snr
                best_horizon = h

    return best_horizon


def discretize_return(future_return: float) -> int:
    """Converteer een toekomstig rendement naar een klasse-label (0-4)."""
    if future_return <= LABEL_THRESHOLDS["strong_sell"]:
        return 0
    elif future_return <= LABEL_THRESHOLDS["sell"]:
        return 1
    elif future_return <= LABEL_THRESHOLDS["hold"]:
        return 2
    elif future_return <= LABEL_THRESHOLDS["buy"]:
        return 3
    else:
        return 4


def build_dataset(
    histories: Dict[str, pd.DataFrame],
    feature_engine: FeatureEngine,
    window_days: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Bouw training dataset voor een specifiek lookback-window.

    Args:
        histories: mapping symbol → OHLCV DataFrame.
        feature_engine: FeatureEngine instantie.
        window_days: hoeveel recente trading-dagen te gebruiken.

    Returns:
        (X features, y labels, returns serie).
    """
    all_features: List[Dict[str, float]] = []
    all_labels: List[int] = []
    all_returns: List[float] = []

    for symbol, hist in histories.items():
        # Beperk tot window
        # ASSUMPTION: data is dagelijks en gesorteerd op datum
        if len(hist) > window_days:
            hist_window = hist.iloc[-window_days:]
        else:
            hist_window = hist

        if len(hist_window) < 80:
            continue

        # Selecteer optimale horizon voor dit symbool
        horizon = select_best_horizon(
            hist_window["close"],
            _LABEL_HORIZONS,
        )

        # Bereken features voor elk tijdstip
        for end_idx in range(60, len(hist_window) - horizon):
            window = hist_window.iloc[:end_idx + 1]
            features = feature_engine.compute_features(window, symbol=symbol, v2=True)
            if features is None:
                continue

            # Label
            current_close = float(hist_window.iloc[end_idx]["close"])
            future_close = float(hist_window.iloc[end_idx + horizon]["close"])
            if current_close <= 0:
                continue
            future_return = (future_close - current_close) / current_close
            label = discretize_return(future_return)

            all_features.append(features)
            all_labels.append(label)
            all_returns.append(future_return)

    if not all_features:
        raise ValueError(f"Geen trainingsdata voor window={window_days}d")

    X = pd.DataFrame(all_features).reindex(columns=FEATURE_COLUMNS_V2).fillna(0)
    y = pd.Series(all_labels, name="label")
    returns = pd.Series(all_returns, name="return")

    logger.info(
        "Dataset (window=%dd): %d samples, %d features. Labels: %s",
        window_days, len(X), len(FEATURE_COLUMNS_V2),
        dict(y.value_counts().sort_index()),
    )

    return X, y, returns


# ── SMOTE ───────────────────────────────────────────────────────────

def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Pas SMOTE toe voor class imbalance correctie.

    Alleen als imbalanced-learn beschikbaar is; anders terugvallen op
    ongebalanceerde data (class_weight='balanced' helpt al).
    """
    try:
        from imblearn.over_sampling import SMOTE

        min_class_count = int(y.value_counts().min())
        if min_class_count < 6:
            logger.info("SMOTE overgeslagen: kleinste klasse heeft %d samples (< 6).", min_class_count)
            return X, y

        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        logger.info(
            "SMOTE: %d → %d samples. Nieuwe verdeling: %s",
            len(X), len(X_resampled),
            dict(pd.Series(y_resampled).value_counts().sort_index()),
        )
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name="label")

    except ImportError:
        logger.info("imbalanced-learn niet geïnstalleerd; SMOTE overgeslagen.")
        return X, y


# ── Model Training ──────────────────────────────────────────────────

def train_window_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    window_name: str,
    use_smote: bool = True,
) -> Tuple[Any, ModelMetrics]:
    """Train een LightGBM classifier voor een specifiek window.

    Args:
        X: Feature matrix.
        y: Labels (0-4).
        window_name: "3m", "1y", of "2y".
        use_smote: SMOTE toepassen voor class imbalance.

    Returns:
        (trained model, ModelMetrics).
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit

    params = _LGBM_PARAMS[window_name].copy()
    logger.info("Training classifier [%s]: %d samples...", window_name, len(X))

    # FIXED: issue-1 — SMOTE wordt nu ALLEEN binnen elke fold toegepast, niet vooraf op volledige dataset

    # TimeSeriesSplit cross-validatie
    tscv = TimeSeriesSplit(n_splits=min(CV_SPLITS, max(2, len(X) // 200)))
    best_model = None
    best_accuracy = 0.0
    f1_scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Gebruik originele (niet-SMOTE) data voor splits, SMOTE alleen op train
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # SMOTE op train fold
        if use_smote:
            X_train_fold, y_train_fold = apply_smote(X_train_fold, y_train_fold)

        n_estimators = params.pop("n_estimators", 500)
        model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        params["n_estimators"] = n_estimators  # restore

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        f1_scores.append(f1)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

        logger.info(
            "  Fold %d [%s]: accuracy=%.4f, f1_macro=%.4f",
            fold + 1, window_name, acc, f1,
        )

    # Finale training op alle data (met SMOTE — veilig want geen validatieset)
    if use_smote:  # FIXED: issue-1 — SMOTE pas hier op volledige dataset, na CV
        X_train_full, y_train_full = apply_smote(X, y)
    else:
        X_train_full, y_train_full = X, y
    n_estimators = params.pop("n_estimators", 500)
    final_params = {k: v for k, v in params.items()}
    final_model = lgb.LGBMClassifier(n_estimators=n_estimators, **final_params)
    final_model.fit(X_train_full, y_train_full)

    # Per-class F1
    all_preds = final_model.predict(X)
    report = classification_report(
        y, all_preds, target_names=SIGNAL_CLASSES,
        output_dict=True, zero_division=0,
    )

    per_class_f1 = {
        cls: round(report[cls]["f1-score"], 4)
        for cls in SIGNAL_CLASSES
        if cls in report
    }

    metrics = ModelMetrics(
        window=window_name,
        accuracy=round(best_accuracy, 4),
        f1_macro=round(float(np.mean(f1_scores)), 4),
        per_class_f1=per_class_f1,
        samples=len(X),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Classifier [%s] klaar: best_acc=%.4f, f1_macro=%.4f",
        window_name, best_accuracy, float(np.mean(f1_scores)),
    )

    return final_model, metrics


def train_regressor(X: pd.DataFrame, returns: pd.Series) -> Any:
    """Train LightGBM regressor voor verwachte return voorspelling."""
    import lightgbm as lgb

    logger.info("Training return regressor: %d samples...", len(X))

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        n_estimators=400,
        verbose=-1,
        random_state=42,
    )
    model.fit(X, returns)

    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((preds - returns.values) ** 2)))
    logger.info("Return regressor RMSE: %.6f", rmse)

    return model


# ── Symbolen ────────────────────────────────────────────────────────

def get_training_symbols() -> List[str]:
    """Haal een lijst van symbolen om op te trainen."""
    stocks = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
        "AMD", "CRM", "NFLX", "ADBE", "ORCL", "QCOM", "AMAT",
        "JPM", "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD",
        "MU", "LRCX", "PANW", "NOW", "ABNB", "COIN", "PLTR",
    ]
    crypto = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    ]
    return stocks + crypto


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    """Hoofd-training pipeline voor het ensemble."""
    os.makedirs(_ENSEMBLE_DIR, exist_ok=True)

    # 1. Download data
    symbols = get_training_symbols()
    histories = download_data(symbols, period="2y")

    if len(histories) < 10:
        logger.error("Te weinig data (%d symbolen). Training afgebroken.", len(histories))
        sys.exit(1)

    # 2. Feature engine
    fe = FeatureEngine()

    # 3. Train per window
    classifiers: Dict[str, Any] = {}
    all_metrics: Dict[str, ModelMetrics] = {}
    all_X: Optional[pd.DataFrame] = None
    all_y: Optional[pd.Series] = None
    all_returns: Optional[pd.Series] = None

    for window_name, window_days in _WINDOWS.items():
        try:
            X, y, returns = build_dataset(histories, fe, window_days)
            model, metrics = train_window_classifier(X, y, window_name)
            classifiers[window_name] = model
            all_metrics[window_name] = metrics

            # Bewaar langste dataset voor regressor
            if all_X is None or len(X) > len(all_X):
                all_X = X
                all_y = y
                all_returns = returns

        except Exception as exc:
            logger.error("Training mislukt voor window %s: %s", window_name, exc)

    if len(classifiers) < 2:
        logger.error("Minder dan 2 classifiers getraind. Afgebroken.")
        sys.exit(1)

    # 4. Train regressor op langste dataset
    regressor = train_regressor(all_X, all_returns)

    # 5. Champion-Challenger check
    ensemble = EnsembleLightGBM()
    existing_loaded = ensemble.load()

    if existing_loaded:
        should_replace = ensemble.champion_challenger(classifiers, all_metrics)
        if not should_replace:
            logger.info("Bestaand model behouden. Training klaar.")
            return
        logger.info("Challenger vervangt champion.")

    # 6. Training distributie berekenen (voor drift detection)
    training_dist = np.zeros(NUM_CLASSES)
    for label in all_y:
        training_dist[int(label)] += 1
    training_dist /= max(training_dist.sum(), 1e-9)

    # 7. Feature statistieken
    feature_means = all_X.mean()
    feature_stds = all_X.std()

    # 8. Opslaan
    ensemble.save_models(
        classifiers=classifiers,
        regressor=regressor,
        training_dist=training_dist,
        feature_means=feature_means,
        feature_stds=feature_stds,
        metrics=all_metrics,
    )

    # 9. Rapport
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "symbols_count": len(histories),
        "windows": {
            w: {
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "samples": m.samples,
                "per_class_f1": m.per_class_f1,
            }
            for w, m in all_metrics.items()
        },
        "training_distribution": {
            SIGNAL_CLASSES[i]: round(float(training_dist[i]), 4)
            for i in range(NUM_CLASSES)
        },
        "total_features": len(FEATURE_COLUMNS_V2),
    }

    report_path = os.path.join(_ENSEMBLE_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Ensemble training voltooid. Rapport: %s", report_path)

    # Feature importance samenvatting
    fi = ensemble.get_feature_importance()
    if fi:
        logger.info("Top 10 features:")
        for feat, imp in list(fi.items())[:10]:
            logger.info("  %s: %.4f", feat, imp)


if __name__ == "__main__":
    main()
