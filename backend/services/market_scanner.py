"""Market Scanner: scoort en rankt assets dagelijks op tradability.

Combineert vier scoring-dimensies per asset:
  1. **Volatiliteit** — ATR-gebaseerd (hogere ATR = meer winst-potentieel)
  2. **Liquiditeit**  — Gemiddeld dagvolume in USD
  3. **Momentum**     — Multi-timeframe price momentum
  4. **Sentiment**    — Placeholder score (gevuld door AlternativeDataCollector in Module 2)

Output: top-5 crypto pairs (via ccxt, Binance public) en top-5 US momentum
stocks (via yfinance), inclusief het gedetecteerde marktregime per asset.

Gebruik:
    scanner = MarketScanner()
    universe = scanner.scan()            # ScanResult met top assets
    regime  = scanner.get_market_regime() # huidig breed marktregime
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.regime_classifier import RegimeClassifier, RegimeResult

logger = logging.getLogger(__name__)

# ── Configuratie ────────────────────────────────────────────────────
_TOP_N: int = 5                      # top-N per asset-klasse
_OHLCV_LOOKBACK_DAYS: int = 90       # hoeveel historie we ophalen
_MIN_BARS: int = 60                  # minimum bars voor scoring
_CACHE_TTL: int = 3600               # scan-resultaten cachen (1 uur)

# Scoring gewichten (sommeren tot 1.0)
_W_VOLATILITY: float = 0.30
_W_LIQUIDITY: float = 0.25
_W_MOMENTUM: float = 0.30
_W_SENTIMENT: float = 0.15

# Binance klines tijdsinterval
_BINANCE_INTERVAL: str = "1d"
_BINANCE_KLINES_URL: str = "https://api.binance.com/api/v3/klines"

# Crypto-paren om te scannen (Binance, geen auth nodig)
_CRYPTO_SCAN_SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "NEAR/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT",
]

# Stock-symbolen om te scannen (momentum-kandidaten uit grote indices)
_STOCK_SCAN_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "CRM", "NFLX", "ADBE", "ORCL", "QCOM", "AMAT",
    "MU", "LRCX", "KLAC", "MRVL", "PANW", "NOW", "SNPS", "CDNS",
    "ABNB", "COIN", "MSTR", "SQ", "SHOP", "PLTR", "SMCI",
]


@dataclass
class AssetScore:
    """Score-breakdown voor één asset."""

    symbol: str
    market: str                           # "crypto" of "us"
    composite_score: float                # gewogen totaalscore ∈ [0, 1]
    volatility_score: float               # genormaliseerd ∈ [0, 1]
    liquidity_score: float                # genormaliseerd ∈ [0, 1]
    momentum_score: float                 # genormaliseerd ∈ [0, 1]
    sentiment_score: float                # placeholder, default 0.5
    regime: RegimeResult = field(default_factory=lambda: RegimeResult(
        regime="sideways",
        probabilities={"bear": 0.25, "sideways": 0.25, "bull": 0.25, "high_vol": 0.25},
        confidence=0.25,
    ))
    atr_14: float = 0.0                   # ruwe ATR-waarde
    avg_daily_volume_usd: float = 0.0     # gemiddeld dagvolume in USD
    momentum_20d: float = 0.0             # 20-dag return


@dataclass
class ScanResult:
    """Resultaat van een volledige MarketScanner scan."""

    timestamp: str
    top_crypto: List[AssetScore]
    top_stocks: List[AssetScore]
    broad_regime: RegimeResult            # breed marktregime (bijv. SPY-gebaseerd)
    all_scores: List[AssetScore] = field(default_factory=list)

    @property
    def top_assets(self) -> List[AssetScore]:
        """Gecombineerde top crypto + stocks, gesorteerd op composite score."""
        combined = self.top_crypto + self.top_stocks
        return sorted(combined, key=lambda a: a.composite_score, reverse=True)

    def symbols(self) -> List[str]:
        """Alle geselecteerde symbolen als platte lijst."""
        return [a.symbol for a in self.top_assets]


class MarketScanner:
    """Scant en rankt assets dagelijks voor het trading-universum.

    Workflow:
    1. Download OHLCV-data voor crypto (ccxt/Binance) en stocks (yfinance)
    2. Bereken per asset: volatiliteit, liquiditeit, momentum
    3. Classificeer marktregime per asset via RegimeClassifier
    4. Rank assets op gewogen composite score
    5. Retourneer top-5 crypto + top-5 stocks
    """

    def __init__(self, regime_classifier: Optional[RegimeClassifier] = None) -> None:
        self._regime_clf = regime_classifier or RegimeClassifier()
        self._cache: Optional[ScanResult] = None
        self._cache_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────

    def scan(self, force_refresh: bool = False) -> ScanResult:
        """Voer een volledige marktscan uit.

        Args:
            force_refresh: Negeer cache en haal verse data op.

        Returns:
            ScanResult met gerankte assets en regime-informatie.
        """
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time) < _CACHE_TTL:
            logger.debug("MarketScanner: cached resultaat retourneren.")
            return self._cache

        logger.info("MarketScanner: start scan van %d crypto + %d stocks.",
                     len(_CRYPTO_SCAN_SYMBOLS), len(_STOCK_SCAN_SYMBOLS))

        # 1. Data ophalen
        crypto_data = self._fetch_crypto_ohlcv()
        stock_data = self._fetch_stock_ohlcv()

        # 2. Score per asset
        crypto_scores = self._score_assets(crypto_data, market="crypto")
        stock_scores = self._score_assets(stock_data, market="us")

        # 3. Rank en selecteer top-N
        crypto_ranked = sorted(crypto_scores, key=lambda a: a.composite_score, reverse=True)
        stock_ranked = sorted(stock_scores, key=lambda a: a.composite_score, reverse=True)

        top_crypto = crypto_ranked[:_TOP_N]
        top_stocks = stock_ranked[:_TOP_N]

        # 4. Breed marktregime (op basis van een brede index)
        broad_regime = self._compute_broad_regime(stock_data)

        from datetime import datetime, timezone
        result = ScanResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            top_crypto=top_crypto,
            top_stocks=top_stocks,
            broad_regime=broad_regime,
            all_scores=crypto_ranked + stock_ranked,
        )

        self._cache = result
        self._cache_time = now

        logger.info("MarketScanner: scan voltooid — top crypto: %s, top stocks: %s, regime: %s.",
                     [a.symbol for a in top_crypto],
                     [a.symbol for a in top_stocks],
                     broad_regime.regime)
        return result

    def get_market_regime(self) -> RegimeResult:
        """Retourneer het meest recente breed marktregime.

        Voert een scan uit als er nog geen resultaat gecached is.
        """
        if self._cache is None:
            self.scan()
        # ASSUMPTION: cache is nu gevuld door scan()
        assert self._cache is not None
        return self._cache.broad_regime

    def get_scannable_symbols(self) -> Dict[str, List[str]]:
        """Retourneer alle symbolen die de scanner evalueert."""
        return {
            "crypto": list(_CRYPTO_SCAN_SYMBOLS),
            "stocks": list(_STOCK_SCAN_SYMBOLS),
        }

    def set_sentiment_scores(self, scores: Dict[str, float]) -> None:
        """Injecteer sentiment-scores van AlternativeDataCollector (Module 2).

        Args:
            scores: mapping van symbol → sentiment score ∈ [0, 1].
        """
        self._external_sentiment = scores

    # ── Data ophalen ────────────────────────────────────────────────

    def _fetch_crypto_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """Haal OHLCV-data op voor crypto-paren via ccxt (Binance public API).

        Geen authenticatie vereist — gebruikt alleen publieke market data endpoints.
        """
        data: Dict[str, pd.DataFrame] = {}
        try:
            import ccxt
        except ImportError:
            logger.warning("ccxt niet geïnstalleerd; crypto-scan overgeslagen. pip install ccxt")
            return data

        exchange = ccxt.binance({"enableRateLimit": True})

        for symbol in _CRYPTO_SCAN_SYMBOLS:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe="1d",
                    limit=_OHLCV_LOOKBACK_DAYS,
                )
                if not ohlcv:
                    continue

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = df.astype(float)

                # Gebruik base symbol als key (bijv. "BTC" uit "BTC/USDT")
                base = symbol.split("/")[0]
                data[base] = df
                logger.debug("Crypto OHLCV opgehaald: %s (%d bars)", base, len(df))

            except Exception as exc:
                logger.warning("Crypto OHLCV ophalen mislukt voor %s: %s", symbol, exc)

        logger.info("Crypto data opgehaald: %d/%d symbolen.", len(data), len(_CRYPTO_SCAN_SYMBOLS))
        return data

    def _fetch_stock_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """Haal OHLCV-data op voor US stocks via yfinance."""
        data: Dict[str, pd.DataFrame] = {}
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance niet geïnstalleerd; stock-scan overgeslagen.")
            return data

        # Batch-download voor efficiëntie
        try:
            tickers_str = " ".join(_STOCK_SCAN_SYMBOLS)
            period = f"{_OHLCV_LOOKBACK_DAYS}d"

            batch = yf.download(
                tickers_str,
                period=period,
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if batch.empty:
                logger.warning("yfinance batch download retourneerde lege data.")
                return data

            for symbol in _STOCK_SCAN_SYMBOLS:
                try:
                    if len(_STOCK_SCAN_SYMBOLS) == 1:
                        df = batch.copy()
                    else:
                        df = batch[symbol].copy()

                    df.columns = [c.lower() for c in df.columns]
                    df = df.dropna(subset=["close"])

                    if len(df) >= _MIN_BARS:
                        data[symbol] = df
                        logger.debug("Stock OHLCV opgehaald: %s (%d bars)", symbol, len(df))
                except Exception:
                    continue

        except Exception as exc:
            logger.warning("yfinance batch download mislukt: %s. Probeer individueel.", exc)
            data = self._fetch_stocks_individual()

        logger.info("Stock data opgehaald: %d/%d symbolen.", len(data), len(_STOCK_SCAN_SYMBOLS))
        return data

    def _fetch_stocks_individual(self) -> Dict[str, pd.DataFrame]:
        """Fallback: haal stocks individueel op als batch faalt."""
        data: Dict[str, pd.DataFrame] = {}
        try:
            import yfinance as yf
        except ImportError:
            return data

        for symbol in _STOCK_SCAN_SYMBOLS:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{_OHLCV_LOOKBACK_DAYS}d", interval="1d")
                if df is not None and len(df) >= _MIN_BARS:
                    df.columns = [c.lower() for c in df.columns]
                    data[symbol] = df
            except Exception:
                continue
        return data

    # ── Scoring ─────────────────────────────────────────────────────

    def _score_assets(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        market: str,
    ) -> List[AssetScore]:
        """Bereken scores voor alle assets in een markt-segment.

        Stappen:
        1. Bereken ruwe metrics per asset (ATR, volume, momentum)
        2. Normaliseer met rank-based scaling ∈ [0, 1]
        3. Classificeer regime per asset
        4. Bereken gewogen composite score
        """
        if not ohlcv_dict:
            return []

        raw_metrics: List[Dict[str, Any]] = []

        for symbol, df in ohlcv_dict.items():
            if len(df) < _MIN_BARS:
                continue
            metrics = self._compute_raw_metrics(symbol, df)
            raw_metrics.append(metrics)

        if not raw_metrics:
            return []

        # Rank-based normalisatie per dimensie
        metrics_df = pd.DataFrame(raw_metrics)
        for col in ["atr_pct", "avg_volume_usd", "momentum_20d"]:
            if col in metrics_df.columns:
                ranked = metrics_df[col].rank(pct=True, na_option="bottom")
                metrics_df[f"{col}_norm"] = ranked

        # Regime classificatie (batch)
        regimes = self._regime_clf.classify_batch(ohlcv_dict)

        # Composite scores bouwen
        scores: List[AssetScore] = []
        for _, row in metrics_df.iterrows():
            symbol = row["symbol"]
            regime = regimes.get(symbol, RegimeResult(
                regime="sideways",
                probabilities={"bear": 0.25, "sideways": 0.25, "bull": 0.25, "high_vol": 0.25},
                confidence=0.25,
            ))

            vol_score = float(row.get("atr_pct_norm", 0.5))
            liq_score = float(row.get("avg_volume_usd_norm", 0.5))
            mom_score = float(row.get("momentum_20d_norm", 0.5))

            # Sentiment: extern geïnjecteerd of default 0.5
            sent_score = getattr(self, "_external_sentiment", {}).get(symbol, 0.5)

            composite = (
                _W_VOLATILITY * vol_score
                + _W_LIQUIDITY * liq_score
                + _W_MOMENTUM * mom_score
                + _W_SENTIMENT * sent_score
            )

            # Regime-bonus: extra punten voor bull + high_vol (meer tradeable)
            if regime.regime == "bull":
                composite *= 1.05
            elif regime.regime == "high_vol":
                composite *= 1.10
            elif regime.regime == "bear":
                composite *= 0.90
            # ASSUMPTION: sideways krijgt geen aanpassing

            composite = min(composite, 1.0)

            scores.append(AssetScore(
                symbol=symbol,
                market=market,
                composite_score=round(composite, 4),
                volatility_score=round(vol_score, 4),
                liquidity_score=round(liq_score, 4),
                momentum_score=round(mom_score, 4),
                sentiment_score=round(sent_score, 4),
                regime=regime,
                atr_14=float(row.get("atr_14", 0.0)),
                avg_daily_volume_usd=float(row.get("avg_volume_usd", 0.0)),
                momentum_20d=float(row.get("momentum_20d", 0.0)),
            ))

        return scores

    def _compute_raw_metrics(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Bereken ruwe scoring-metrics voor één asset.

        Returns dict met: symbol, atr_14, atr_pct, avg_volume_usd, momentum_20d.
        """
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float)

        # ATR-14 (als percentage van prijs)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
        price = float(close.iloc[-1])
        atr_pct = atr_14 / price if price > 0 else 0.0

        # Gemiddeld dagvolume in USD (20-dag)
        avg_volume_usd = float((volume * close).tail(20).mean())

        # Momentum (20-dag return)
        momentum_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0.0

        return {
            "symbol": symbol,
            "atr_14": atr_14,
            "atr_pct": atr_pct,
            "avg_volume_usd": avg_volume_usd,
            "momentum_20d": momentum_20d,
        }

    def _compute_broad_regime(
        self,
        stock_data: Dict[str, pd.DataFrame],
    ) -> RegimeResult:
        """Bepaal het brede marktregime op basis van een proxy-index.

        Gebruikt SPY als het beschikbaar is in de stock data, anders het gemiddelde
        van alle stocks. Bij gebrek aan data → yfinance fallback voor SPY.
        """
        # Probeer SPY als brede markt proxy
        if "SPY" in stock_data and len(stock_data["SPY"]) >= _MIN_BARS:
            return self._regime_clf.classify(stock_data["SPY"])

        # Fallback: haal SPY apart op
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            df = spy.history(period=f"{_OHLCV_LOOKBACK_DAYS}d", interval="1d")
            if df is not None and len(df) >= _MIN_BARS:
                df.columns = [c.lower() for c in df.columns]
                return self._regime_clf.classify(df)
        except Exception as exc:
            logger.warning("SPY ophalen mislukt voor breed regime: %s", exc)

        # Laatste fallback: gemiddelde van beschikbare stocks
        if stock_data:
            first_key = next(iter(stock_data))
            first_df = stock_data[first_key]
            if len(first_df) >= _MIN_BARS:
                return self._regime_clf.classify(first_df)

        # Geen data beschikbaar → default sideways
        logger.warning("Geen data voor breed marktregime; default = sideways.")
        return RegimeResult(
            regime="sideways",
            probabilities={"bear": 0.25, "sideways": 0.25, "bull": 0.25, "high_vol": 0.25},
            confidence=0.25,
        )
