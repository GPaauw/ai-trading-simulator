"""Feature Engine v2: berekent technische + alternatieve features voor ML-modellen.

Uitbreiding van de originele FeatureEngine met 17-27 nieuwe features:
  - Sentiment (FinBERT via AlternativeDataCollector)
  - Reddit mention velocity
  - Insider buy/sell ratio (SEC EDGAR Form 4)
  - Institutional holdings delta (SEC EDGAR 13F)
  - VIX level + 5-dag verandering
  - Market regime one-hot (4 kolommen: bull/bear/sideways/high_vol)
  - Order book imbalance proxy
  - Earnings surprise flag
  - Extra technische: Keltner Channel, VWAP afwijking, Heikin-Ashi momentum,
    ADX trendsterkte, Chaikin Money Flow

Volledige backward compatibility: alle originele 28 features behouden
dezelfde naam en berekening.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.config import FEATURE_COLUMNS, FEATURE_COLUMNS_V2

logger = logging.getLogger(__name__)

# Cache
_cache: Dict[str, Any] = {}
_CACHE_TTL = 300  # 5 minuten


class FeatureEngine:
    """Berekent ~50 features uit OHLCV-data + alternatieve data voor elk instrument.

    V2 features worden alleen berekend als ``v2=True`` (default). Achterwaarts
    compatibel: ``compute_features()`` retourneert altijd minstens de originele 28.
    """

    def __init__(self) -> None:
        # Optionele externe data-injectie (gezet door AlternativeDataCollector / MarketScanner)
        self._alternative_data: Dict[str, Dict[str, float]] = {}
        self._regime_data: Dict[str, Dict[str, int]] = {}
        self._macro_data: Dict[str, float] = {}
        self._earnings_data: Dict[str, Dict[str, float]] = {}

    # ── Externe data injectie ───────────────────────────────────────

    def set_alternative_data(self, data: Dict[str, Dict[str, float]]) -> None:
        """Injecteer alternatieve features per symbool van AlternativeDataCollector.

        Args:
            data: mapping symbol → {sentiment_score, reddit_mention_velocity,
                  insider_buy_sell_ratio, institutional_holdings_delta}
        """
        self._alternative_data = data

    def set_regime_data(self, data: Dict[str, Dict[str, int]]) -> None:
        """Injecteer regime one-hot encoding per symbool van RegimeClassifier.

        Args:
            data: mapping symbol → {regime_bear, regime_sideways, regime_bull, regime_high_vol}
        """
        self._regime_data = data

    def set_macro_data(self, data: Dict[str, float]) -> None:
        """Injecteer macro-economische data van FRED.

        Args:
            data: {vix_level, vix_5d_change}
        """
        self._macro_data = data

    def set_earnings_data(self, data: Dict[str, Dict[str, float]]) -> None:
        """Injecteer earnings surprise data per symbool.

        Args:
            data: mapping symbol → {days_since_earnings, earnings_surprise}
        """
        self._earnings_data = data

    def compute_features(
        self,
        history: pd.DataFrame,
        symbol: str = "",
        v2: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Bereken features uit een DataFrame met kolommen: open, high, low, close, volume.

        Args:
            history: OHLCV DataFrame.
            symbol: Asset-symbool voor het ophalen van alternatieve data.
            v2: Als True, bereken ook de nieuwe v2 features.

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

            # ═══════════════════════════════════════════════════════════
            # V2 FEATURES — nieuwe features, alleen als v2=True
            # ═══════════════════════════════════════════════════════════
            if v2:
                # ── Alternatieve data (geïnjecteerd) ───────────────
                alt = self._alternative_data.get(symbol, {})
                features["sentiment_score"] = alt.get("sentiment_score", 0.5)
                features["reddit_mention_velocity"] = alt.get("reddit_mention_velocity", 0.0)
                features["insider_buy_sell_ratio"] = alt.get("insider_buy_sell_ratio", 0.5)
                features["institutional_holdings_delta"] = alt.get("institutional_holdings_delta", 0.0)

                # ── VIX (macro, geïnjecteerd) ──────────────────────
                features["vix_level"] = self._macro_data.get("vix_level", 20.0)
                features["vix_5d_change"] = self._macro_data.get("vix_5d_change", 0.0)

                # ── Market regime one-hot (geïnjecteerd) ───────────
                regime = self._regime_data.get(symbol, {})
                features["regime_bear"] = float(regime.get("regime_bear", 0))
                features["regime_sideways"] = float(regime.get("regime_sideways", 1))
                features["regime_bull"] = float(regime.get("regime_bull", 0))
                features["regime_high_vol"] = float(regime.get("regime_high_vol", 0))

                # ── Order book imbalance proxy ─────────────────────
                # Proxy: (close - low) / (high - low) — meet koopdruk
                hl_range = float(high.iloc[-1] - low.iloc[-1])
                features["order_book_imbalance"] = (
                    float((close.iloc[-1] - low.iloc[-1]) / hl_range)
                    if hl_range > 1e-9 else 0.5
                )

                # ── Earnings surprise flag ─────────────────────────
                earn = self._earnings_data.get(symbol, {})
                features["days_since_earnings"] = earn.get("days_since_earnings", 90.0)
                features["earnings_surprise"] = earn.get("earnings_surprise", 0.0)

                # ── Keltner Channel positie ────────────────────────
                kc_mid = close.ewm(span=20, adjust=False).mean()
                kc_atr = atr  # hergebruik ATR-14
                kc_upper = kc_mid.iloc[-1] + 2.0 * kc_atr
                kc_lower = kc_mid.iloc[-1] - 2.0 * kc_atr
                kc_range = kc_upper - kc_lower
                features["keltner_position"] = (
                    float((price - kc_lower) / kc_range)
                    if kc_range > 1e-9 else 0.5
                )

                # ── VWAP afwijking (proxy via volume-weighted close) ──
                if len(close) >= 20:
                    vwap = float(
                        (close.tail(20) * volume.tail(20)).sum()
                        / volume.tail(20).sum()
                    ) if volume.tail(20).sum() > 0 else price
                    features["vwap_deviation"] = float((price - vwap) / vwap) if vwap > 0 else 0.0
                else:
                    features["vwap_deviation"] = 0.0

                # ── Heikin-Ashi momentum ───────────────────────────
                ha_close = (history["open"] + high + low + close) / 4.0
                ha_diff = ha_close.diff()
                # Laatste 5 bars: hoeveel zijn bullish (positieve diff)?
                features["heikin_ashi_streak"] = float(
                    (ha_diff.tail(5) > 0).sum() / 5.0
                )

                # ── ADX trendsterkte ───────────────────────────────
                features["adx_14"] = self._adx(high, low, close, 14)

                # ── Chaikin Money Flow (20-dag) ────────────────────
                features["cmf_20"] = self._chaikin_money_flow(high, low, close, volume, 20)

                # ── Return volatiliteit ratio (korte/lange vol) ────
                vol_5 = float(close.pct_change().tail(5).std()) if len(close) >= 5 else 0.01
                vol_20 = features.get("volatility_20d", 0.01)
                features["vol_ratio_5_20"] = vol_5 / vol_20 if vol_20 > 1e-9 else 1.0

                # ── Momentum divergentie (prijs vs RSI) ────────────
                # Als prijs stijgt maar RSI daalt → bearish divergentie
                rsi_val = features.get("rsi_14", 50.0)
                ret_5d = features.get("return_5d", 0.0)
                # Simpele proxy: teken-verschil als divergentie-indicator
                features["momentum_divergence"] = (
                    1.0 if (ret_5d > 0 and rsi_val < 40) else
                    -1.0 if (ret_5d < 0 and rsi_val > 60) else
                    0.0
                )

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
        v2: bool = True,
    ) -> pd.DataFrame:
        """Bereken features voor meerdere instrumenten.

        Args:
            instrument_histories: mapping symbol → OHLCV DataFrame.
            v2: Als True, bereken ook v2 features.

        Returns een DataFrame met symbolen als index en feature kolommen.
        """
        cache_key = f"batch_{hash(tuple(sorted(instrument_histories.keys())))}_{v2}"
        cached = _cache.get(cache_key)
        if cached and time.time() - cached["ts"] < _CACHE_TTL:
            return cached["data"]

        columns = FEATURE_COLUMNS_V2 if v2 else FEATURE_COLUMNS

        results: Dict[str, Dict[str, float]] = {}
        for symbol, hist in instrument_histories.items():
            features = self.compute_features(hist, symbol=symbol, v2=v2)
            if features:
                results[symbol] = features

        if not results:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame.from_dict(results, orient="index")
        # Zorg dat alle verwachte kolommen bestaan
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[columns]

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

    # ── V2 hulpfuncties ─────────────────────────────────────────────

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """Average Directional Index — meet trendsterkte (0-100)."""
        if len(close) < period + 1:
            return 20.0

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr_s = tr.rolling(period).mean()
        plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr_s)
        minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr_s)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100.0
        adx_val = dx.rolling(period).mean()
        result = adx_val.iloc[-1]
        return float(result) if np.isfinite(result) else 20.0

    @staticmethod
    def _chaikin_money_flow(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20,
    ) -> float:
        """Chaikin Money Flow — meet accumulatie/distributie (-1 tot +1)."""
        hl_range = high - low
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
        mfm = mfm.fillna(0.0)
        # Money Flow Volume
        mfv = mfm * volume
        # CMF
        vol_sum = volume.rolling(period).sum()
        cmf = mfv.rolling(period).sum() / vol_sum.replace(0, np.nan)
        result = cmf.iloc[-1]
        return float(result) if np.isfinite(result) else 0.0
