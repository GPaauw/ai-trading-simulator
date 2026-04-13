"""Signal Predictor: LightGBM-gebaseerde signaalvoorspelling.

Laadt een voorgetraind LightGBM model en voorspelt buy/sell/hold signalen
op basis van feature vectors uit de FeatureEngine. Valt terug op
regelgebaseerde logica als het model nog niet getraind is.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.config import (
    FEATURE_COLUMNS,
    SIGNAL_CLASSES,
    SIGNAL_MODEL_PATH,
    RETURN_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Voorspelt trading-signalen met LightGBM of regelgebaseerde fallback."""

    def __init__(self) -> None:
        self._classifier = None
        self._regressor = None
        self._model_loaded = False
        self._model_version: Optional[str] = None
        self._load_models()

    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def get_model_version(self) -> Optional[str]:
        return self._model_version

    def predict(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Voorspel signalen voor alle instrumenten in de feature DataFrame.

        Args:
            features_df: DataFrame met symbolen als index, FEATURE_COLUMNS als kolommen.

        Returns:
            Lijst van dicts met per instrument:
            - symbol, action, confidence, expected_return_pct, risk_score, horizon
        """
        if features_df.empty:
            return []

        if self._model_loaded:
            return self._predict_ml(features_df)
        return self._predict_rules(features_df)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance van het geladen model."""
        if not self._model_loaded or self._classifier is None:
            return {}
        try:
            importance = self._classifier.feature_importances_
            total = max(float(sum(importance)), 1e-9)
            return {
                FEATURE_COLUMNS[i]: round(float(importance[i]) / total, 4)
                for i in range(min(len(FEATURE_COLUMNS), len(importance)))
            }
        except Exception:
            return {}

    # ── ML voorspelling ─────────────────────────────────────────────

    def _predict_ml(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Voorspel met geladen LightGBM modellen."""
        results = []

        # Zorg voor juiste kolommen en vul ontbrekende aan met 0
        X = features_df.reindex(columns=FEATURE_COLUMNS).fillna(0)

        # Classificatie: multi-class probabilities
        probas = self._classifier.predict_proba(X)  # shape: (n_samples, n_classes)

        # Regressie: verwachte return
        if self._regressor is not None:
            expected_returns = self._regressor.predict(X)
        else:
            expected_returns = np.zeros(len(X))

        for i, symbol in enumerate(features_df.index):
            proba = probas[i]
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

            expected_return = float(expected_returns[i])

            # Risico-score: gebaseerd op volatiliteit feature + model onzekerheid
            vol = float(features_df.loc[symbol, "volatility_20d"]) if "volatility_20d" in features_df.columns else 0.02
            entropy = float(-np.sum(proba * np.log(proba + 1e-10)))
            risk_score = min(1.0, vol * 10 + entropy * 0.3)

            # Horizon: hoge volatiliteit → korter
            horizon = 1 if vol > 0.03 or action_raw.startswith("strong") else 5

            # Ranking score (0-100)
            buy_prob = float(proba[3] + proba[4])  # buy + strong_buy
            sell_prob = float(proba[0] + proba[1])  # strong_sell + sell
            ranking_score = buy_prob * 100 if action == "buy" else (1 - sell_prob) * 100 if action == "sell" else 50.0

            results.append({
                "symbol": symbol,
                "action": action,
                "action_detail": action_raw,
                "confidence": round(confidence, 4),
                "expected_return_pct": round(expected_return * 100, 2),
                "risk_score": round(risk_score, 4),
                "horizon_days": horizon,
                "ranking_score": round(ranking_score, 2),
                "probabilities": {c: round(float(proba[j]), 4) for j, c in enumerate(SIGNAL_CLASSES)},
            })

        return sorted(results, key=lambda x: x["ranking_score"], reverse=True)

    # ── Regelgebaseerde fallback ────────────────────────────────────

    def _predict_rules(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Fallback: regelgebaseerde signalen op basis van technische indicatoren."""
        results = []

        for symbol in features_df.index:
            row = features_df.loc[symbol]
            score = 0.0
            reasons = []

            # RSI
            rsi = row.get("rsi_14", 50)
            if rsi < 30:
                score += 2.0
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                score -= 2.0
                reasons.append(f"RSI overbought ({rsi:.0f})")

            # MACD
            macd_hist = row.get("macd_hist", 0)
            if macd_hist > 0:
                score += 1.0
                reasons.append("MACD bullish")
            elif macd_hist < 0:
                score -= 1.0
                reasons.append("MACD bearish")

            # SMA crossover
            sma5 = row.get("sma_5", 0)
            sma20 = row.get("sma_20", 0)
            if sma5 > 0 and sma20 < 0:  # prijs boven SMA5 en onder SMA20 (genormaliseerd)
                score += 0.5
            elif sma5 < 0 and sma20 > 0:
                score -= 0.5

            # Bollinger Bands
            bb_lower = row.get("bb_lower_dist", 0)
            if bb_lower < 0.01:
                score += 1.5
                reasons.append("Near lower BB")or None

            # Volume bevestiging
            vol_ratio = row.get("volume_ratio", 1)
            if vol_ratio > 1.5:
                score *= 1.2
                reasons.append("High volume")

            # Momentum
            roc = row.get("roc_10", 0)
            score += roc * 10  # schaal ~0.01 returns naar scores

            # Bepaal actie
            if score >= 2.0:
                action = "buy"
            elif score <= -2.0:
                action = "sell"
            else:
                action = "hold"

            confidence = min(0.95, max(0.3, 0.5 + abs(score) * 0.05))
            vol = row.get("volatility_20d", 0.02)
            risk = min(1.0, vol * 10)
            expected_return = score * 0.5

            results.append({
                "symbol": symbol,
                "action": action,
                "action_detail": action,
                "confidence": round(confidence, 4),
                "expected_return_pct": round(expected_return, 2),
                "risk_score": round(risk, 4),
                "horizon_days": 1 if abs(score) > 3 else 5,
                "ranking_score": round(max(0, min(100, 50 + score * 10)), 2),
                "probabilities": {},
                "reason": "; ".join(reasons) if reasons else "Geen sterk signaal",
            })

        return sorted(results, key=lambda x: x["ranking_score"], reverse=True)

    # ── Model laden ─────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Probeer voorgetrainde modellen te laden."""
        try:
            if os.path.exists(SIGNAL_MODEL_PATH):
                import joblib
                self._classifier = joblib.load(SIGNAL_MODEL_PATH)
                logger.info("LightGBM classifier geladen: %s", SIGNAL_MODEL_PATH)

                if os.path.exists(RETURN_MODEL_PATH):
                    self._regressor = joblib.load(RETURN_MODEL_PATH)
                    logger.info("LightGBM regressor geladen: %s", RETURN_MODEL_PATH)

                self._model_loaded = True
                mtime = os.path.getmtime(SIGNAL_MODEL_PATH)
                self._model_version = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            else:
                logger.info("Geen ML-model gevonden op %s — gebruik regelgebaseerde fallback.", SIGNAL_MODEL_PATH)
        except Exception as exc:
            logger.warning("Kon ML-model niet laden: %s — gebruik fallback.", exc)
            self._model_loaded = False
