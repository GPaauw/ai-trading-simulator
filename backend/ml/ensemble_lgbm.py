"""Ensemble LightGBM: multi-window ensemble van 3 LightGBM modellen.

Vervangt de enkele LightGBM classifier met een ensemble van 3 modellen
getraind op verschillende lookback-windows (3 maanden, 1 jaar, 2 jaar).
Combineert via soft voting op de finale signaal-probabiliteiten.

Key features:
  - Soft voting ensemble (gewogen gemiddelde van predict_proba)
  - SHAP integratie voor feature importance en drift detection
  - ChampionChallenger: wekelijks hertraining, vervangt alleen als validatie verbetert
  - Class imbalance fix: class_weight='balanced' + optioneel SMOTE
  - Verbeterde labeling: meerdere horizons (3, 5, 10 dagen) met beste S/N ratio

Gebruik:
    ensemble = EnsembleLightGBM()
    ensemble.load()  # laad bestaande modellen
    signals = ensemble.predict(features_df)
    drift = ensemble.check_drift(features_df)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.config import (
    FEATURE_COLUMNS_V2,
    SIGNAL_CLASSES,
    NUM_CLASSES,
    ML_MODELS_DIR,
)

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
_ENSEMBLE_DIR: str = os.path.join(ML_MODELS_DIR, "ensemble_v2")

# Lookback-windows in trading-dagen
_WINDOWS: Dict[str, int] = {
    "3m": 63,       # ~3 maanden
    "1y": 252,      # ~1 jaar
    "2y": 504,      # ~2 jaar
}

# Ensemble gewichten (langere window = minder gewicht voor recency bias)
_WINDOW_WEIGHTS: Dict[str, float] = {
    "3m": 0.45,
    "1y": 0.35,
    "2y": 0.20,
}

# Label horizons om te evalueren
_LABEL_HORIZONS: List[int] = [3, 5, 10]

# SHAP drift detection drempel
_DRIFT_KL_THRESHOLD: float = 0.15

# Champion-challenger minimum verbetering
_MIN_IMPROVEMENT: float = 0.005  # 0.5% accuracy verbetering vereist


@dataclass
class EnsemblePrediction:
    """Resultaat van ensemble-voorspelling voor één asset."""

    symbol: str
    action: str                          # "buy", "sell", "hold"
    action_detail: str                   # "strong_buy", "buy", etc.
    confidence: float                    # max probabiliteit
    expected_return_pct: float           # verwacht rendement (%)
    risk_score: float                    # ∈ [0, 1]
    horizon_days: int                    # voorspellingshorizon
    ranking_score: float                 # ∈ [0, 100]
    probabilities: Dict[str, float]      # per-class kansen
    per_window_predictions: Dict[str, str] = field(default_factory=dict)
    shap_top_features: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class DriftReport:
    """Resultaat van feature-drift detectie."""

    is_drifted: bool
    kl_divergence: float                 # KL-divergentie van predictie-distributie
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    recommendation: str = ""


@dataclass
class ModelMetrics:
    """Metrics van een getraind ensemble-model."""

    window: str
    accuracy: float
    f1_macro: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    samples: int = 0
    trained_at: str = ""


class EnsembleLightGBM:
    """Ensemble van 3 LightGBM-modellen met SHAP en drift detection.

    Architectuur:
    - 3 classifiers: getraind op 3m, 1y, 2y windows
    - 1 regressor: getraind op volledige dataset (return voorspelling)
    - Soft voting: gewogen gemiddelde van predict_proba
    - SHAP: feature importance per predictie + drift monitoring
    """

    def __init__(self) -> None:
        self._classifiers: Dict[str, Any] = {}  # window → LGBMClassifier
        self._regressor: Optional[Any] = None
        self._model_loaded: bool = False
        self._model_version: Optional[str] = None
        self._training_distribution: Optional[np.ndarray] = None  # voor drift detection
        self._feature_means: Optional[pd.Series] = None
        self._feature_stds: Optional[pd.Series] = None
        self._shap_explainer: Optional[Any] = None
        self._metrics: Dict[str, ModelMetrics] = {}

    # ── Public API ──────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        """Geeft aan of minstens één ensemble-model geladen is."""
        return self._model_loaded

    def get_model_version(self) -> Optional[str]:
        """Retourneer versie-identificatie van het geladen ensemble."""
        return self._model_version

    def load(self) -> bool:
        """Laad bestaande ensemble-modellen van disk.

        Returns True als minstens 2 van de 3 window-modellen geladen zijn.
        """
        try:
            import joblib
        except ImportError:
            logger.warning("joblib niet geïnstalleerd; ensemble laden overgeslagen.")
            return False

        loaded_count = 0
        for window_name in _WINDOWS:
            model_path = os.path.join(_ENSEMBLE_DIR, f"classifier_{window_name}.joblib")
            try:
                if os.path.exists(model_path):
                    self._classifiers[window_name] = joblib.load(model_path)
                    loaded_count += 1
                    logger.info("Ensemble classifier %s geladen.", window_name)
            except Exception as exc:
                logger.warning("Ensemble classifier %s laden mislukt: %s", window_name, exc)

        # Regressor
        regressor_path = os.path.join(_ENSEMBLE_DIR, "regressor.joblib")
        try:
            if os.path.exists(regressor_path):
                self._regressor = joblib.load(regressor_path)
                logger.info("Ensemble regressor geladen.")
        except Exception as exc:
            logger.warning("Ensemble regressor laden mislukt: %s", exc)

        # Training distributie (voor drift detection)
        dist_path = os.path.join(_ENSEMBLE_DIR, "training_distribution.npy")
        try:
            if os.path.exists(dist_path):
                self._training_distribution = np.load(dist_path)
        except Exception:
            pass

        # Feature statistieken (voor drift detection)
        stats_path = os.path.join(_ENSEMBLE_DIR, "feature_stats.json")
        try:
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                self._feature_means = pd.Series(stats.get("means", {}))
                self._feature_stds = pd.Series(stats.get("stds", {}))
        except Exception:
            pass

        # Versie
        version_path = os.path.join(_ENSEMBLE_DIR, "version.txt")
        try:
            if os.path.exists(version_path):
                with open(version_path, "r") as f:
                    self._model_version = f.read().strip()
        except Exception:
            pass

        # ASSUMPTION: minstens 2 van 3 classifiers nodig voor betrouwbaar ensemble
        self._model_loaded = loaded_count >= 2
        if self._model_loaded:
            logger.info("Ensemble geladen: %d/%d classifiers, version=%s",
                        loaded_count, len(_WINDOWS), self._model_version)
        else:
            logger.warning("Ensemble niet compleet: %d/%d classifiers.", loaded_count, len(_WINDOWS))

        return self._model_loaded

    def predict(self, features_df: pd.DataFrame) -> List[EnsemblePrediction]:
        """Voorspel signalen via ensemble soft voting.

        Args:
            features_df: DataFrame met symbolen als index, FEATURE_COLUMNS_V2 als kolommen.

        Returns:
            Gesorteerde lijst van EnsemblePrediction (hoog → laag ranking score).
        """
        if features_df.empty or not self._model_loaded:
            return []

        X = features_df.reindex(columns=FEATURE_COLUMNS_V2).fillna(0)

        # ── Soft voting: gewogen gemiddelde van predict_proba ───────
        ensemble_probas = np.zeros((len(X), NUM_CLASSES))
        total_weight = 0.0

        per_window_preds: Dict[str, List[str]] = {w: [] for w in _WINDOWS}

        for window_name, weight in _WINDOW_WEIGHTS.items():
            if window_name not in self._classifiers:
                continue
            try:
                probas = self._classifiers[window_name].predict_proba(X)
                ensemble_probas += weight * probas
                total_weight += weight

                # Per-window voorspelling voor logging
                for i in range(len(X)):
                    class_idx = int(np.argmax(probas[i]))
                    per_window_preds[window_name].append(SIGNAL_CLASSES[class_idx])

            except Exception as exc:
                logger.warning("Ensemble predict mislukt voor window %s: %s", window_name, exc)

        if total_weight < 1e-9:
            return []

        ensemble_probas /= total_weight

        # ── Regressie: verwachte return ────────────────────────────
        if self._regressor is not None:
            try:
                expected_returns = self._regressor.predict(X)
            except Exception:
                expected_returns = np.zeros(len(X))
        else:
            expected_returns = np.zeros(len(X))

        # ── SHAP feature importance (eerste 10 samples) ────────────
        shap_values = self._compute_shap(X.iloc[:min(10, len(X))])

        # ── Bouw resultaten ────────────────────────────────────────
        results: List[EnsemblePrediction] = []

        for i, symbol in enumerate(features_df.index):
            proba = ensemble_probas[i]
            class_idx = int(np.argmax(proba))
            action_raw = SIGNAL_CLASSES[class_idx]
            confidence = float(proba[class_idx])

            # Map naar buy/sell/hold
            if action_raw in ("strong_buy", "buy"):
                action = "buy"
            elif action_raw in ("strong_sell", "sell"):
                action = "sell"
            else:
                action = "hold"

            # Risico-score
            vol = float(X.iloc[i].get("volatility_20d", 0.02))
            entropy = float(-np.sum(proba * np.log(proba + 1e-10)))
            risk_score = min(1.0, vol * 10 + entropy * 0.3)

            # Horizon
            horizon = 3 if vol > 0.03 or action_raw.startswith("strong") else 5

            # Ranking score
            buy_prob = float(proba[3] + proba[4])
            sell_prob = float(proba[0] + proba[1])
            ranking_score = (
                buy_prob * 100 if action == "buy"
                else (1 - sell_prob) * 100 if action == "sell"
                else 50.0
            )

            # Per-window predictions
            window_preds = {}
            for w in _WINDOWS:
                if w in per_window_preds and i < len(per_window_preds[w]):
                    window_preds[w] = per_window_preds[w][i]

            # SHAP top features
            top_feats: List[Tuple[str, float]] = []
            if shap_values is not None and i < len(shap_values):
                feat_importance = np.abs(shap_values[i])
                top_idx = np.argsort(feat_importance)[-5:][::-1]
                top_feats = [
                    (FEATURE_COLUMNS_V2[j], round(float(shap_values[i][j]), 4))
                    for j in top_idx
                    if j < len(FEATURE_COLUMNS_V2)
                ]

            results.append(EnsemblePrediction(
                symbol=symbol,
                action=action,
                action_detail=action_raw,
                confidence=round(confidence, 4),
                expected_return_pct=round(float(expected_returns[i]) * 100, 2),
                risk_score=round(risk_score, 4),
                horizon_days=horizon,
                ranking_score=round(ranking_score, 2),
                probabilities={
                    c: round(float(proba[j]), 4)
                    for j, c in enumerate(SIGNAL_CLASSES)
                },
                per_window_predictions=window_preds,
                shap_top_features=top_feats,
            ))

        return sorted(results, key=lambda x: x.ranking_score, reverse=True)

    def check_drift(self, features_df: pd.DataFrame) -> DriftReport:
        """Detecteer feature en prediction drift t.o.v. trainingsdistributie.

        Gebruikt twee methoden:
        1. KL-divergentie op predictie-distributie
        2. Z-score analyse op individuele features

        Args:
            features_df: Huidige feature DataFrame.

        Returns:
            DriftReport met aanbeveling.
        """
        now = datetime.now(timezone.utc).isoformat()

        if not self._model_loaded or features_df.empty:
            return DriftReport(
                is_drifted=False, kl_divergence=0.0,
                timestamp=now, recommendation="Geen model geladen.",
            )

        X = features_df.reindex(columns=FEATURE_COLUMNS_V2).fillna(0)

        # 1. Prediction distribution drift (KL-divergentie)
        kl_div = self._compute_prediction_kl(X)

        # 2. Feature drift (per-feature Z-score)
        feature_drift = self._compute_feature_drift(X)

        is_drifted = kl_div > _DRIFT_KL_THRESHOLD
        drifted_features = {
            k: v for k, v in feature_drift.items() if abs(v) > 3.0
        }

        if is_drifted and drifted_features:
            recommendation = (
                f"Hertraining aanbevolen: KL={kl_div:.4f} > {_DRIFT_KL_THRESHOLD}. "
                f"Gedrifte features: {list(drifted_features.keys())[:5]}"
            )
        elif is_drifted:
            recommendation = f"Hertraining aanbevolen: KL={kl_div:.4f} > {_DRIFT_KL_THRESHOLD}."
        elif drifted_features:
            recommendation = f"Feature drift gedetecteerd maar predictie stabiel. Monitoren."
        else:
            recommendation = "Model performt binnen verwachte parameters."

        return DriftReport(
            is_drifted=is_drifted,
            kl_divergence=round(kl_div, 6),
            feature_drift_scores={k: round(v, 4) for k, v in feature_drift.items()},
            timestamp=now,
            recommendation=recommendation,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Retourneer geaggregeerde feature importance over alle ensemble-modellen."""
        if not self._model_loaded:
            return {}

        aggregated = np.zeros(len(FEATURE_COLUMNS_V2))
        count = 0

        for window_name, model in self._classifiers.items():
            try:
                imp = model.feature_importances_
                if len(imp) == len(FEATURE_COLUMNS_V2):
                    aggregated += imp
                    count += 1
            except Exception:
                continue

        if count == 0:
            return {}

        aggregated /= count
        total = max(float(aggregated.sum()), 1e-9)

        importance = {
            FEATURE_COLUMNS_V2[i]: round(float(aggregated[i]) / total, 4)
            for i in range(len(FEATURE_COLUMNS_V2))
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_metrics(self) -> Dict[str, Any]:
        """Retourneer trainingsmetrics per window."""
        return {
            window: {
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "per_class_f1": m.per_class_f1,
                "samples": m.samples,
                "trained_at": m.trained_at,
            }
            for window, m in self._metrics.items()
        }

    # ── SHAP integratie ─────────────────────────────────────────────

    def _compute_shap(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Bereken SHAP-waarden voor de gegeven samples.

        Lazy-init van de SHAP explainer bij eerste aanroep.
        """
        try:
            import shap
        except ImportError:
            return None

        if self._shap_explainer is None:
            # Gebruik het eerste beschikbare model voor SHAP
            model = next(iter(self._classifiers.values()), None)
            if model is None:
                return None
            try:
                self._shap_explainer = shap.TreeExplainer(model)
            except Exception as exc:
                logger.debug("SHAP explainer initialisatie mislukt: %s", exc)
                return None

        try:
            shap_values = self._shap_explainer.shap_values(X)
            # Voor multi-class: neem de absolute max over alle klassen
            if isinstance(shap_values, list):
                # shap_values is een lijst per klasse
                stacked = np.stack(shap_values, axis=-1)
                # Neem de klasse met hoogste absolute SHAP per sample/feature
                max_class_idx = np.argmax(np.abs(stacked), axis=-1)
                result = np.take_along_axis(
                    stacked, max_class_idx[..., np.newaxis], axis=-1
                ).squeeze(-1)
                return result
            return shap_values
        except Exception as exc:
            logger.debug("SHAP berekening mislukt: %s", exc)
            return None

    # ── Drift detection helpers ─────────────────────────────────────

    def _compute_prediction_kl(self, X: pd.DataFrame) -> float:
        """Bereken KL-divergentie tussen huidige en training predictie-distributie."""
        if self._training_distribution is None:
            return 0.0

        # Bereken huidige predictie-distributie
        current_dist = np.zeros(NUM_CLASSES)
        total_weight = 0.0

        for window_name, weight in _WINDOW_WEIGHTS.items():
            if window_name not in self._classifiers:
                continue
            try:
                preds = self._classifiers[window_name].predict(X)
                for pred in preds:
                    current_dist[int(pred)] += weight
                total_weight += weight * len(X)
            except Exception:
                continue

        if total_weight < 1e-9:
            return 0.0

        current_dist /= max(current_dist.sum(), 1e-9)
        train_dist = self._training_distribution

        # KL-divergentie met smoothing
        eps = 1e-10
        current_dist = np.clip(current_dist, eps, 1.0)
        train_dist = np.clip(train_dist, eps, 1.0)

        # Renormaliseer na clipping
        current_dist /= current_dist.sum()
        train_dist /= train_dist.sum()

        kl = float(np.sum(current_dist * np.log(current_dist / train_dist)))
        return max(0.0, kl)

    def _compute_feature_drift(self, X: pd.DataFrame) -> Dict[str, float]:
        """Bereken per-feature Z-score drift t.o.v. training statistieken."""
        if self._feature_means is None or self._feature_stds is None:
            return {}

        drift_scores: Dict[str, float] = {}
        current_means = X.mean()

        for col in FEATURE_COLUMNS_V2:
            if col in self._feature_means.index and col in self._feature_stds.index:
                train_mean = self._feature_means[col]
                train_std = self._feature_stds[col]
                if train_std > 1e-9:
                    z_score = (current_means.get(col, 0.0) - train_mean) / train_std
                    drift_scores[col] = float(z_score)

        return drift_scores

    # ── Serialisatie helpers ────────────────────────────────────────

    def save_models(
        self,
        classifiers: Dict[str, Any],
        regressor: Any,
        training_dist: np.ndarray,
        feature_means: pd.Series,
        feature_stds: pd.Series,
        metrics: Dict[str, ModelMetrics],
    ) -> None:
        """Sla ensemble-modellen en metadata op naar disk."""
        try:
            import joblib
        except ImportError:
            logger.error("joblib niet geïnstalleerd; kan modellen niet opslaan.")
            return

        os.makedirs(_ENSEMBLE_DIR, exist_ok=True)

        for window_name, model in classifiers.items():
            path = os.path.join(_ENSEMBLE_DIR, f"classifier_{window_name}.joblib")
            joblib.dump(model, path)
            logger.info("Ensemble classifier %s opgeslagen: %s", window_name, path)

        if regressor is not None:
            path = os.path.join(_ENSEMBLE_DIR, "regressor.joblib")
            joblib.dump(regressor, path)

        # Training distributie
        np.save(os.path.join(_ENSEMBLE_DIR, "training_distribution.npy"), training_dist)

        # Feature statistieken
        stats = {
            "means": feature_means.to_dict(),
            "stds": feature_stds.to_dict(),
        }
        with open(os.path.join(_ENSEMBLE_DIR, "feature_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Versie
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(_ENSEMBLE_DIR, "version.txt"), "w") as f:
            f.write(version)

        # Metrics
        metrics_dict = {
            w: {
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "per_class_f1": m.per_class_f1,
                "samples": m.samples,
                "trained_at": m.trained_at,
            }
            for w, m in metrics.items()
        }
        with open(os.path.join(_ENSEMBLE_DIR, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=2)

        self._classifiers = classifiers
        self._regressor = regressor
        self._training_distribution = training_dist
        self._feature_means = feature_means
        self._feature_stds = feature_stds
        self._model_version = version
        self._model_loaded = len(classifiers) >= 2
        self._metrics = metrics

        logger.info("Ensemble v2 opgeslagen: version=%s", version)

    def champion_challenger(
        self,
        challenger_classifiers: Dict[str, Any],
        challenger_metrics: Dict[str, ModelMetrics],
    ) -> bool:
        """Vergelijk challenger-modellen met de huidige champion.

        Vervangt alleen als de challenger een verbeterde gemiddelde accuracy biedt.

        Args:
            challenger_classifiers: nieuw getrainde modellen per window.
            challenger_metrics: metrics van de nieuwe modellen.

        Returns:
            True als de challenger gepromoveerd is tot champion.
        """
        if not self._metrics:
            logger.info("Geen bestaand champion — challenger wordt direct champion.")
            return True

        # Vergelijk gemiddelde accuracy
        champion_acc = np.mean([m.accuracy for m in self._metrics.values()])
        challenger_acc = np.mean([m.accuracy for m in challenger_metrics.values()])

        improvement = challenger_acc - champion_acc

        if improvement >= _MIN_IMPROVEMENT:
            logger.info(
                "Challenger wint: acc %.4f → %.4f (verbetering: +%.4f)",
                champion_acc, challenger_acc, improvement,
            )
            return True
        else:
            logger.info(
                "Champion behouden: acc %.4f vs challenger %.4f (verbetering: %.4f < %.4f)",
                champion_acc, challenger_acc, improvement, _MIN_IMPROVEMENT,
            )
            return False
