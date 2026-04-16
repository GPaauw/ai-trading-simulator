"""Regime Classifier: detecteert marktregime (bull / bear / sideways / high_vol).

Traint een LightGBM-classifier op afgeleide macro- en prijsfeatures om het
huidige marktregime te classificeren. Het regime conditioneert de volledige
downstream pipeline (feature weging, signaaldrempels, RL reward shaping).

Regime-labels worden achteraf berekend op historische data via een heuristiek
en vervolgens geleerd door het model zodat real-time classificatie mogelijk is
zonder forward-looking bias.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
REGIME_CLASSES: List[str] = ["bear", "sideways", "bull", "high_vol"]
NUM_REGIMES: int = len(REGIME_CLASSES)

# Heuristiek-drempels voor label-generatie
_BULL_RETURN_THRESHOLD: float = 0.02       # 20-dag return > 2%
_BEAR_RETURN_THRESHOLD: float = -0.02      # 20-dag return < -2%
_HIGH_VOL_PERCENTILE: float = 80.0         # volatiliteit boven 80e percentiel
_MIN_HISTORY_BARS: int = 60                # minimaal 60 bars voor classificatie

# Model-pad
_REGIME_MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models")
_REGIME_MODEL_PATH: str = os.path.join(_REGIME_MODEL_DIR, "regime_classifier.joblib")


@dataclass
class RegimeResult:
    """Resultaat van een regime-classificatie."""

    regime: str                          # "bull", "bear", "sideways", "high_vol"
    probabilities: Dict[str, float]      # kans per regime-klasse
    confidence: float                    # max(probabilities)
    features_used: Dict[str, float] = field(default_factory=dict)

    def to_onehot(self) -> Dict[str, int]:
        """Retourneert 4 binaire kolommen voor downstream feature engineering."""
        return {
            f"regime_{cls}": int(self.regime == cls)
            for cls in REGIME_CLASSES
        }


class RegimeClassifier:
    """Classificeert het huidige marktregime op basis van prijsgebaseerde features.

    Ondersteunt twee modi:
    1. **Trained**: geladen LightGBM-model (snelle inference)
    2. **Heuristic fallback**: regelgebaseerde classificatie als model ontbreekt

    Features voor het regime-model (berekend uit een brede marktindex):
    - return_20d:      20-dag cumulatief rendement
    - return_60d:      60-dag cumulatief rendement
    - volatility_20d:  20-dag rolling standaarddeviatie van dagrendementen
    - volatility_ratio: volatility_20d / volatility_60d
    - adx_14:          Average Directional Index (trendsterkte)
    - rsi_14:          Relative Strength Index
    - sma_cross:       SMA20 / SMA50 ratio (trendrichting)
    - volume_trend:    5-dag gem. volume / 20-dag gem. volume
    """

    # Feature-namen die het regime-model verwacht
    REGIME_FEATURES: List[str] = [
        "return_20d",
        "return_60d",
        "volatility_20d",
        "volatility_ratio",
        "adx_14",
        "rsi_14",
        "sma_cross",
        "volume_trend",
    ]

    def __init__(self) -> None:
        self._model = None
        self._model_loaded: bool = False
        self._load_model()

    # ── Public API ──────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        """Geeft aan of een getraind model beschikbaar is."""
        return self._model_loaded

    def classify(self, ohlcv: pd.DataFrame) -> RegimeResult:
        """Classificeer het marktregime op basis van OHLCV-data.

        Args:
            ohlcv: DataFrame met kolommen [open, high, low, close, volume].
                   Minimaal ``_MIN_HISTORY_BARS`` rijen.

        Returns:
            RegimeResult met regime-label, kansen en gebruikte features.
        """
        if ohlcv is None or len(ohlcv) < _MIN_HISTORY_BARS:
            logger.debug("Onvoldoende data (%s bars); default regime = sideways.",
                         len(ohlcv) if ohlcv is not None else 0)
            return RegimeResult(
                regime="sideways",
                probabilities={cls: 0.25 for cls in REGIME_CLASSES},
                confidence=0.25,
            )

        features = self._compute_regime_features(ohlcv)

        if self._model_loaded:
            return self._classify_ml(features)
        return self._classify_heuristic(features)

    def classify_batch(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, RegimeResult]:
        """Classificeer regime voor meerdere assets tegelijk.

        Args:
            ohlcv_dict: mapping van symbol → OHLCV DataFrame.

        Returns:
            mapping van symbol → RegimeResult.
        """
        results: Dict[str, RegimeResult] = {}
        for symbol, ohlcv in ohlcv_dict.items():
            try:
                results[symbol] = self.classify(ohlcv)
            except Exception as exc:
                logger.warning("Regime classificatie mislukt voor %s: %s", symbol, exc)
                results[symbol] = RegimeResult(
                    regime="sideways",
                    probabilities={cls: 0.25 for cls in REGIME_CLASSES},
                    confidence=0.25,
                )
        return results

    def train(self, ohlcv: pd.DataFrame, min_samples: int = 200) -> Dict[str, float]:
        """Train het regime-model op historische OHLCV-data.

        Labels worden gegenereerd via de regelgebaseerde heuristiek.
        Het getrainde model vervangt daarna de heuristiek voor inference.

        Args:
            ohlcv: Lange OHLCV-reeks (minimaal ``min_samples`` bars).
            min_samples: Minimum aantal datapunten voor training.

        Returns:
            Dict met trainingsmetrics (accuracy, per-class f1).
        """
        try:
            import joblib
            import lightgbm as lgb
            from sklearn.metrics import classification_report
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError as exc:
            logger.error("Benodigde packages voor training ontbreken: %s", exc)
            return {"error": str(exc)}

        if len(ohlcv) < min_samples:
            return {"error": f"Onvoldoende data: {len(ohlcv)} < {min_samples}"}

        # ── Features en labels berekenen voor elke bar ──────────────
        feature_rows: List[Dict[str, float]] = []
        labels: List[int] = []

        for i in range(_MIN_HISTORY_BARS, len(ohlcv)):
            window = ohlcv.iloc[:i + 1]
            feats = self._compute_regime_features(window)
            label = self._heuristic_label(feats)
            feature_rows.append(feats)
            labels.append(label)

        X = pd.DataFrame(feature_rows, columns=self.REGIME_FEATURES)
        y = np.array(labels)

        # ── Train met TimeSeriesSplit ───────────────────────────────
        tscv = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_accuracy: float = 0.0

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=NUM_REGIMES,
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                class_weight="balanced",
                verbose=-1,
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            accuracy = float(model.score(X_val, y_val))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        if best_model is None:
            return {"error": "Training gefaald — geen model geproduceerd."}

        # ── Opslaan ─────────────────────────────────────────────────
        os.makedirs(_REGIME_MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, _REGIME_MODEL_PATH)

        self._model = best_model
        self._model_loaded = True

        # ── Rapportage ──────────────────────────────────────────────
        y_pred = best_model.predict(X)
        report = classification_report(
            y, y_pred,
            target_names=REGIME_CLASSES,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "accuracy": best_accuracy,
            "samples": len(y),
        }
        for cls in REGIME_CLASSES:
            if cls in report:
                metrics[f"f1_{cls}"] = round(report[cls]["f1-score"], 4)
        logger.info("Regime classifier getraind: accuracy=%.3f, samples=%d", best_accuracy, len(y))
        return metrics

    # ── Interne methoden ────────────────────────────────────────────

    def _load_model(self) -> None:
        """Probeer een eerder getraind model te laden."""
        try:
            import joblib
            if os.path.exists(_REGIME_MODEL_PATH):
                self._model = joblib.load(_REGIME_MODEL_PATH)
                self._model_loaded = True
                logger.info("Regime classifier model geladen van %s.", _REGIME_MODEL_PATH)
        except Exception as exc:
            logger.warning("Kon regime model niet laden: %s. Heuristiek wordt gebruikt.", exc)
            self._model_loaded = False

    def _compute_regime_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Bereken de 8 regime-features uit een OHLCV DataFrame.

        Verwacht kolommen: open, high, low, close, volume.
        """
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float)

        daily_returns = close.pct_change().dropna()

        # return_20d
        return_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0.0

        # return_60d
        return_60d = float(close.iloc[-1] / close.iloc[-min(61, len(close))] - 1)

        # volatility_20d
        vol_20 = float(daily_returns.tail(20).std()) if len(daily_returns) >= 20 else 0.01

        # volatility_ratio  (short / long vol)
        vol_60 = float(daily_returns.tail(60).std()) if len(daily_returns) >= 60 else vol_20
        volatility_ratio = vol_20 / vol_60 if vol_60 > 1e-9 else 1.0

        # ADX 14 (vereenvoudigde berekening)
        adx_14 = self._compute_adx(high, low, close, period=14)

        # RSI 14
        rsi_14 = self._compute_rsi(close, period=14)

        # SMA cross ratio
        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(min(50, len(close))).mean())
        sma_cross = sma20 / sma50 if sma50 > 1e-9 else 1.0

        # Volume trend
        vol_5 = float(volume.tail(5).mean())
        vol_20_ma = float(volume.tail(20).mean())
        volume_trend = vol_5 / vol_20_ma if vol_20_ma > 1e-9 else 1.0

        return {
            "return_20d": return_20d,
            "return_60d": return_60d,
            "volatility_20d": vol_20,
            "volatility_ratio": volatility_ratio,
            "adx_14": adx_14,
            "rsi_14": rsi_14,
            "sma_cross": sma_cross,
            "volume_trend": volume_trend,
        }

    def _classify_ml(self, features: Dict[str, float]) -> RegimeResult:
        """Classificeer regime met het getrainde LightGBM-model."""
        X = pd.DataFrame([features], columns=self.REGIME_FEATURES)
        probas = self._model.predict_proba(X)[0]
        class_idx = int(np.argmax(probas))

        return RegimeResult(
            regime=REGIME_CLASSES[class_idx],
            probabilities={
                REGIME_CLASSES[i]: round(float(probas[i]), 4)
                for i in range(NUM_REGIMES)
            },
            confidence=round(float(probas[class_idx]), 4),
            features_used=features,
        )

    def _classify_heuristic(self, features: Dict[str, float]) -> RegimeResult:
        """Regelgebaseerde fallback wanneer geen model beschikbaar is.

        Logica:
        1. Hoge volatiliteit → high_vol
        2. Return > drempel → bull
        3. Return < drempel → bear
        4. Anders → sideways
        """
        vol = features["volatility_20d"]
        ret = features["return_20d"]
        vol_ratio = features["volatility_ratio"]

        # Hoge volatiliteit check: ratio > 1.3 OF absolute vol hoog
        # ASSUMPTION: volatility_ratio > 1.3 duidt op snel stijgende vol
        if vol_ratio > 1.3 and vol > 0.02:
            regime = "high_vol"
        elif ret > _BULL_RETURN_THRESHOLD:
            regime = "bull"
        elif ret < _BEAR_RETURN_THRESHOLD:
            regime = "bear"
        else:
            regime = "sideways"

        # Zachte probabiliteiten genereren op basis van features
        probas = self._heuristic_probabilities(features, regime)

        return RegimeResult(
            regime=regime,
            probabilities=probas,
            confidence=probas[regime],
            features_used=features,
        )

    def _heuristic_label(self, features: Dict[str, float]) -> int:
        """Geeft een integer label (index in REGIME_CLASSES) op basis van heuristiek."""
        result = self._classify_heuristic(features)
        return REGIME_CLASSES.index(result.regime)

    def _heuristic_probabilities(
        self,
        features: Dict[str, float],
        primary_regime: str,
    ) -> Dict[str, float]:
        """Genereer zachte kansen voor de heuristiek-modus.

        Geeft 60% aan het primaire regime, rest verdeeld op basis van feature-afstand.
        """
        base_prob = 0.6
        remaining = 1.0 - base_prob
        n_other = NUM_REGIMES - 1

        probas: Dict[str, float] = {}
        for cls in REGIME_CLASSES:
            if cls == primary_regime:
                probas[cls] = round(base_prob, 4)
            else:
                probas[cls] = round(remaining / n_other, 4)
        return probas

    # ── Hulpfuncties voor technische indicatoren ────────────────────

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> float:
        """Bereken RSI over de gegeven periode."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()

        last_gain = avg_gain.iloc[-1]
        last_loss = avg_loss.iloc[-1]
        if last_loss < 1e-9:
            return 100.0
        rs = last_gain / last_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _compute_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """Vereenvoudigde ADX-berekening (Average Directional Index).

        Geeft trendsterkte terug: < 20 = zwak, 20-40 = sterk, > 40 = zeer sterk.
        """
        if len(close) < period + 1:
            return 20.0  # neutraal default

        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Alleen opwaartse/neerwaartse directional movement
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100.0
        adx = dx.rolling(period).mean()

        last_adx = adx.iloc[-1]
        return float(last_adx) if np.isfinite(last_adx) else 20.0
