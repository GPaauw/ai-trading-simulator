"""Feature Engine: berekent technische indicatoren als feature vectors voor ML-modellen."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Cache
_cache: Dict[str, Any] = {}
_CACHE_TTL = 300  # 5 minuten


class FeatureEngine:
    """Berekent ~30 technische features uit OHLCV-data voor elk instrument."""

    def compute_features(self, history: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Bereken features uit een DataFrame met kolommen: open, high, low, close, volume.

        Returns een dict met feature-naam → waarde, of None bij onvoldoende data.
        """
        if history is None or len(history) < 50:
            return None

        try:
            close = history["close"].astype(float)
            high = history["high"].astype(float)
            low = history["low"].astype(float)
            volume = history["volume"].astype(float)

            features: Dict[str, float] = {}

            # ── Prijs-returns ───────────────────────────────────────
            features["return_1d"] = self._safe_return(close, 1)
            features["return_5d"] = self._safe_return(close, 5)
            features["return_20d"] = self._safe_return(close, 20)
            features["log_return_1d"] = float(np.log(close.iloc[-1] / close.iloc[-2])) if close.iloc[-2] > 0 else 0.0
            features["volatility_20d"] = float(close.pct_change().tail(20).std()) if len(close) >= 20 else 0.0

            # ── RSI ─────────────────────────────────────────────────
            features["rsi_14"] = self._rsi(close, 14)

            # ── MACD ────────────────────────────────────────────────
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            features["macd_hist"] = float(macd_hist.iloc[-1])
            features["macd_signal_diff"] = float(macd_line.iloc[-1] - signal_line.iloc[-1])

            # ── SMA ─────────────────────────────────────────────────
            sma5 = close.rolling(5).mean()
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(min(len(close), 20)).mean()
            features["sma_5"] = self._norm_price(close, sma5)
            features["sma_20"] = self._norm_price(close, sma20)
            features["sma_50"] = self._norm_price(close, sma50)

            # ── EMA ─────────────────────────────────────────────────
            features["ema_12"] = self._norm_price(close, ema12)
            features["ema_26"] = self._norm_price(close, ema26)
            features["ema_diff"] = float((ema12.iloc[-1] - ema26.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

            # ── Bollinger Bands ─────────────────────────────────────
            bb_mid = sma20
            bb_std = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            price = close.iloc[-1]
            features["bb_upper_dist"] = float((bb_upper.iloc[-1] - price) / price) if price > 0 else 0.0
            features["bb_lower_dist"] = float((price - bb_lower.iloc[-1]) / price) if price > 0 else 0.0
            features["bb_width"] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / price) if price > 0 else 0.0

            # ── ATR ─────────────────────────────────────────────────
            atr = self._atr(high, low, close, 14)
            features["atr_14"] = atr
            features["atr_pct"] = atr / price if price > 0 else 0.0

            # ── Volume ──────────────────────────────────────────────
            vol_ma = volume.rolling(20).mean()
            features["volume_ratio"] = float(volume.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
            features["volume_ma_ratio"] = float(volume.tail(5).mean() / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
            features["obv_slope"] = self._obv_slope(close, volume)

            # ── Momentum ────────────────────────────────────────────
            features["roc_10"] = self._safe_return(close, 10)
            features["stoch_rsi"] = self._stoch_rsi(close, 14)
            features["williams_r"] = self._williams_r(high, low, close, 14)

            # ── Prijsrelaties ───────────────────────────────────────
            features["price_vs_sma20"] = float((price - sma20.iloc[-1]) / sma20.iloc[-1]) if sma20.iloc[-1] > 0 else 0.0
            features["price_vs_sma50"] = float((price - sma50.iloc[-1]) / sma50.iloc[-1]) if sma50.iloc[-1] > 0 else 0.0
            features["high_low_range"] = float((high.iloc[-1] - low.iloc[-1]) / price) if price > 0 else 0.0

            # Vervang NaN/inf door 0
            for k, v in features.items():
                if not np.isfinite(v):
                    features[k] = 0.0

            return features

        except Exception as exc:
            logger.warning("Feature berekening mislukt: %s", exc)
            return None

    def compute_batch(
        self,
        instrument_histories: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Bereken features voor meerdere instrumenten.

        Returns een DataFrame met symbolen als index en FEATURE_COLUMNS als kolommen.
        """
        cache_key = f"batch_{hash(tuple(sorted(instrument_histories.keys())))}"
        cached = _cache.get(cache_key)
        if cached and time.time() - cached["ts"] < _CACHE_TTL:
            return cached["data"]

        results: Dict[str, Dict[str, float]] = {}
        for symbol, hist in instrument_histories.items():
            features = self.compute_features(hist)
            if features:
                results[symbol] = features

        if not results:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        df = pd.DataFrame.from_dict(results, orient="index")
        # Zorg dat alle verwachte kolommen bestaan
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[FEATURE_COLUMNS]

        _cache[cache_key] = {"data": df, "ts": time.time()}
        return df

    # ── Hulpfuncties ────────────────────────────────────────────────

    @staticmethod
    def _safe_return(series: pd.Series, periods: int) -> float:
        if len(series) <= periods or series.iloc[-periods - 1] == 0:
            return 0.0
        return float((series.iloc[-1] - series.iloc[-periods - 1]) / series.iloc[-periods - 1])

    @staticmethod
    def _norm_price(close: pd.Series, indicator: pd.Series) -> float:
        price = close.iloc[-1]
        val = indicator.iloc[-1]
        if price > 0 and np.isfinite(val):
            return float((price - val) / price)
        return 0.0

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _obv_slope(close: pd.Series, volume: pd.Series, period: int = 10) -> float:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * volume).cumsum()
        if len(obv) < period:
            return 0.0
        x = np.arange(period)
        y = obv.iloc[-period:].values.astype(float)
        if np.std(y) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope / (np.mean(np.abs(y)) + 1e-10))

    @staticmethod
    def _stoch_rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        denominator = rsi_max.iloc[-1] - rsi_min.iloc[-1]
        if denominator == 0:
            return 0.5
        return float((rsi.iloc[-1] - rsi_min.iloc[-1]) / denominator)

    @staticmethod
    def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        highest = high.rolling(period).max().iloc[-1]
        lowest = low.rolling(period).min().iloc[-1]
        if highest == lowest:
            return -50.0
        return float(-100 * (highest - close.iloc[-1]) / (highest - lowest))
