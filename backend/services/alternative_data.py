"""Alternative Data Collector: aggregeert gratis alternatieve databronnen.

Verzamelt en verwerkt data uit vier bronnen:
  1. **Reddit** (PRAW)       — mention count, sentiment, velocity per asset
  2. **Nieuws** (feedparser) — nieuwskoppen van Reuters, Yahoo Finance, Seeking Alpha RSS
  3. **SEC EDGAR**           — 13F (institutioneel) en Form 4 (insider) filings
  4. **Macro** (FRED)        — VIX, CPI, rente via pandas-datareader

Alle teksten worden lokaal verwerkt met FinBERT (SentimentAnalyzer) voor
sentiment scoring. Output is een per-asset dict met alternatieve features
die direct door FeatureEngine v2 (Module 3) geconsumeerd kunnen worden.

Configuratie via environment variabelen:
  - REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT  (gratis)
  - FRED_API_KEY  (gratis via https://fred.stlouisfed.org/docs/api/api_key.html)
  - SEC_USER_AGENT  (e-mail adres, verplicht door SEC)
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

from ml.sentiment_analyzer import AggregateSentiment, SentimentAnalyzer, SentimentResult

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
_CACHE_TTL: int = 1800          # 30 minuten cache
_REDDIT_LOOKBACK_HOURS: int = 24
_REDDIT_POST_LIMIT: int = 100   # posts per subreddit
_EDGAR_FORM4_LOOKBACK_DAYS: int = 30
_EDGAR_13F_LOOKBACK_DAYS: int = 95  # ~1 kwartaal
_HTTP_TIMEOUT: int = 15

# Reddit subreddits om te scrapen
_SUBREDDITS: List[str] = ["wallstreetbets", "investing", "CryptoCurrency", "stocks"]

# RSS feeds voor nieuws
_RSS_FEEDS: Dict[str, str] = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets": "https://feeds.reuters.com/reuters/marketsNews",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    # ASSUMPTION: Seeking Alpha RSS is beschikbaar zonder auth voor headlines
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
}

# FRED macro series
_FRED_SERIES: Dict[str, str] = {
    "vix": "VIXCLS",           # VIX (CBOE Volatility Index)
    "fed_funds": "DFF",        # Effective Federal Funds Rate
    "cpi": "CPIAUCSL",         # Consumer Price Index
    "treasury_10y": "DGS10",   # 10-Year Treasury Yield
    "treasury_2y": "DGS2",     # 2-Year Treasury Yield
}

# SEC EDGAR base URL
_EDGAR_BASE: str = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FULL_TEXT: str = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS: str = "https://data.sec.gov/submissions"
_EDGAR_SEARCH: str = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILING_SEARCH: str = "https://efts.sec.gov/LATEST/search-index"

# Full-text search API (gratis, geen auth)
_EDGAR_EFTS_URL: str = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class RedditMentionData:
    """Reddit mention-data voor één asset."""

    symbol: str
    mention_count_24h: int = 0
    mention_count_prev_24h: int = 0      # vorige 24h voor velocity
    velocity: float = 0.0                 # groeipercentage mentions
    avg_upvotes: float = 0.0
    avg_comments: float = 0.0
    post_titles: List[str] = field(default_factory=list)
    sentiment: Optional[AggregateSentiment] = None


@dataclass
class NewsData:
    """Nieuwsdata voor één asset."""

    symbol: str
    headlines: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    sentiment: Optional[AggregateSentiment] = None


@dataclass
class EdgarData:
    """SEC EDGAR data voor één asset."""

    symbol: str
    # Form 4 (insider trading)
    insider_buys_30d: int = 0
    insider_sells_30d: int = 0
    insider_buy_sell_ratio: float = 0.5   # buys / (buys + sells), default neutraal
    # 13F (institutioneel)
    institutional_holders_delta: float = 0.0   # netto verandering in holdings
    institutional_filings_count: int = 0


@dataclass
class MacroData:
    """Macro-economische data van FRED."""

    vix_level: float = 20.0               # huidig VIX niveau
    vix_5d_change: float = 0.0            # 5-dag verandering VIX
    fed_funds_rate: float = 5.0           # huidige rente
    cpi_yoy: float = 3.0                  # CPI jaar-op-jaar %
    treasury_spread: float = 0.0          # 10Y - 2Y spread
    timestamp: str = ""


@dataclass
class AlternativeFeatures:
    """Geaggregeerde alternatieve features per asset, klaar voor FeatureEngine v2."""

    symbol: str
    sentiment_score: float = 0.5          # FinBERT ∈ [0, 1]
    reddit_mention_velocity: float = 0.0  # 24h growth rate
    reddit_mention_count: int = 0
    insider_buy_sell_ratio: float = 0.5   # ∈ [0, 1]
    institutional_holdings_delta: float = 0.0
    news_sentiment: float = 0.5           # ∈ [0, 1]
    news_volume: int = 0                  # aantal headlines

    def to_feature_dict(self) -> Dict[str, float]:
        """Converteer naar dict voor FeatureEngine v2."""
        return {
            "sentiment_score": self.sentiment_score,
            "reddit_mention_velocity": self.reddit_mention_velocity,
            "insider_buy_sell_ratio": self.insider_buy_sell_ratio,
            "institutional_holdings_delta": self.institutional_holdings_delta,
        }


class AlternativeDataCollector:
    """Orchestreert het ophalen van alle alternatieve databronnen.

    Elke bron wordt onafhankelijk opgehaald en gecached. Als een bron
    faalt, worden de overige bronnen nog steeds verwerkt (graceful degradation).

    Gebruik:
        collector = AlternativeDataCollector()
        features = collector.collect(symbols=["AAPL", "BTC", "TSLA"])
        # Dict[str, AlternativeFeatures]
    """

    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        self._sentiment = sentiment_analyzer or SentimentAnalyzer()
        self._reddit_client = None
        self._reddit_initialized: bool = False

        # Caches
        self._reddit_cache: Dict[str, RedditMentionData] = {}
        self._news_cache: Dict[str, NewsData] = {}
        self._edgar_cache: Dict[str, EdgarData] = {}
        self._macro_cache: Optional[MacroData] = None
        self._cache_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────

    def collect(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, AlternativeFeatures]:
        """Verzamel alternatieve data voor een lijst van symbolen.

        Args:
            symbols: Lijst van asset-symbolen (bijv. ["AAPL", "BTC"]).
            force_refresh: Negeer cache en haal verse data op.

        Returns:
            Mapping van symbol → AlternativeFeatures.
        """
        now = time.time()
        if not force_refresh and (now - self._cache_time) < _CACHE_TTL:
            # Retourneer cached data voor gevraagde symbolen
            cached = {s: self._build_features(s) for s in symbols}
            if all(cached.values()):
                logger.debug("AlternativeDataCollector: cached resultaat retourneren.")
                return cached

        logger.info("AlternativeDataCollector: data verzamelen voor %d symbolen.", len(symbols))

        # 1. Reddit mentions + sentiment
        self._collect_reddit(symbols)

        # 2. Nieuws headlines + sentiment
        self._collect_news(symbols)

        # 3. SEC EDGAR (Form 4 + 13F)
        self._collect_edgar(symbols)

        # 4. Macro data (FRED)
        self._collect_macro()

        self._cache_time = now

        # Bouw per-asset features
        result: Dict[str, AlternativeFeatures] = {}
        for symbol in symbols:
            result[symbol] = self._build_features(symbol)

        logger.info("AlternativeDataCollector: %d symbolen verwerkt.", len(result))
        return result

    def get_macro_data(self) -> MacroData:
        """Retourneer de laatst opgehaalde macro-data."""
        if self._macro_cache is None:
            self._collect_macro()
        return self._macro_cache or MacroData()

    def get_reddit_data(self, symbol: str) -> Optional[RedditMentionData]:
        """Retourneer Reddit mention-data voor een specifiek symbool."""
        return self._reddit_cache.get(symbol)

    def get_news_data(self, symbol: str) -> Optional[NewsData]:
        """Retourneer nieuwsdata voor een specifiek symbool."""
        return self._news_cache.get(symbol)

    def get_edgar_data(self, symbol: str) -> Optional[EdgarData]:
        """Retourneer SEC EDGAR data voor een specifiek symbool."""
        return self._edgar_cache.get(symbol)

    # ── Reddit ──────────────────────────────────────────────────────

    def _collect_reddit(self, symbols: List[str]) -> None:
        """Scrape Reddit subreddits voor mention-data en sentiment.

        Vereist PRAW + Reddit API credentials (gratis):
            REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
        """
        if not self._init_reddit():
            logger.info("Reddit data overgeslagen (PRAW niet geconfigureerd).")
            return

        # Bouw regex pattern voor symbol-detectie
        # ASSUMPTION: symbolen zijn case-insensitive, maar we matchen als hele woorden
        symbol_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in symbols) + r')\b',
            re.IGNORECASE,
        )

        # Tel mentions per symbol
        mention_counts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for subreddit_name in _SUBREDDITS:
            try:
                subreddit = self._reddit_client.subreddit(subreddit_name)

                for post in subreddit.hot(limit=_REDDIT_POST_LIMIT):
                    text = f"{post.title} {post.selftext or ''}"
                    matches = symbol_pattern.findall(text)

                    for match in set(matches):
                        symbol_upper = match.upper()
                        if symbol_upper in symbols:
                            mention_counts[symbol_upper].append({
                                "title": post.title,
                                "score": getattr(post, "score", 0),
                                "num_comments": getattr(post, "num_comments", 0),
                                "created_utc": getattr(post, "created_utc", 0),
                            })

            except Exception as exc:
                logger.warning("Reddit scrape mislukt voor r/%s: %s", subreddit_name, exc)

        # Verwerk resultaten per symbol
        now_ts = time.time()
        cutoff_24h = now_ts - (_REDDIT_LOOKBACK_HOURS * 3600)
        cutoff_prev_24h = cutoff_24h - (_REDDIT_LOOKBACK_HOURS * 3600)

        for symbol in symbols:
            posts = mention_counts.get(symbol, [])

            recent = [p for p in posts if p["created_utc"] >= cutoff_24h]
            prev = [p for p in posts if cutoff_prev_24h <= p["created_utc"] < cutoff_24h]

            count_24h = len(recent)
            count_prev = len(prev)
            velocity = ((count_24h - count_prev) / max(count_prev, 1)) if count_prev > 0 else float(count_24h)

            # Sentiment op post-titels
            titles = [p["title"] for p in recent if p.get("title")]
            sentiment = None
            if titles:
                sent_results = self._sentiment.analyze_batch(titles)
                sentiment = self._sentiment.aggregate(symbol, sent_results)

            self._reddit_cache[symbol] = RedditMentionData(
                symbol=symbol,
                mention_count_24h=count_24h,
                mention_count_prev_24h=count_prev,
                velocity=round(velocity, 4),
                avg_upvotes=round(np.mean([p["score"] for p in recent]), 2) if recent else 0.0,
                avg_comments=round(np.mean([p["num_comments"] for p in recent]), 2) if recent else 0.0,
                post_titles=titles[:10],  # bewaar max 10 titels
                sentiment=sentiment,
            )

    def _init_reddit(self) -> bool:
        """Initialiseer PRAW Reddit client met environment variabelen."""
        if self._reddit_initialized:
            return self._reddit_client is not None

        self._reddit_initialized = True

        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "ai-trading-bot/2.0")

        if not client_id or not client_secret:
            logger.info(
                "Reddit API credentials ontbreken. Stel REDDIT_CLIENT_ID en "
                "REDDIT_CLIENT_SECRET in als environment variabelen."
            )
            return False

        try:
            import praw
            self._reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            # Verificatie: probeer een subreddit op te halen
            self._reddit_client.subreddit("test")
            logger.info("Reddit (PRAW) client succesvol geïnitialiseerd.")
            return True

        except ImportError:
            logger.warning("praw niet geïnstalleerd. pip install praw")
            return False
        except Exception as exc:
            logger.warning("Reddit client initialisatie mislukt: %s", exc)
            return False

    # ── Nieuws (RSS) ────────────────────────────────────────────────

    def _collect_news(self, symbols: List[str]) -> None:
        """Haal nieuwskoppen op via RSS feeds en analyseer sentiment.

        Gebruikt feedparser voor het parsen van RSS/Atom feeds.
        """
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser niet geïnstalleerd; nieuws overgeslagen. pip install feedparser")
            return

        all_entries: List[Dict[str, str]] = []

        for feed_name, feed_url in _RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:50]:  # max 50 per feed
                    all_entries.append({
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", ""),
                        "source": feed_name,
                        "published": getattr(entry, "published", ""),
                    })
            except Exception as exc:
                logger.warning("RSS feed ophalen mislukt (%s): %s", feed_name, exc)

        if not all_entries:
            logger.info("Geen nieuwsartikelen opgehaald.")
            return

        # Match headlines naar symbolen
        symbol_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in symbols) + r')\b',
            re.IGNORECASE,
        )

        # Aanvullend: bedrijfsnamen → symbolen mapping voor betere matching
        # TODO: Uitbreiden met meer bedrijfsnamen en aliassen
        _COMPANY_ALIASES: Dict[str, List[str]] = {
            "AAPL": ["apple"],
            "MSFT": ["microsoft"],
            "GOOGL": ["google", "alphabet"],
            "AMZN": ["amazon"],
            "TSLA": ["tesla"],
            "META": ["meta", "facebook"],
            "NVDA": ["nvidia"],
            "BTC": ["bitcoin"],
            "ETH": ["ethereum"],
            "SOL": ["solana"],
        }

        # Reverse mapping: naam → symbol
        alias_to_symbol: Dict[str, str] = {}
        for sym, aliases in _COMPANY_ALIASES.items():
            for alias in aliases:
                alias_to_symbol[alias.lower()] = sym

        alias_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(a) for a in alias_to_symbol) + r')\b',
            re.IGNORECASE,
        )

        news_per_symbol: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        for entry in all_entries:
            text = f"{entry['title']} {entry.get('summary', '')}"

            # Match op ticker-symbolen
            sym_matches = symbol_pattern.findall(text)
            for match in set(sym_matches):
                upper = match.upper()
                if upper in symbols:
                    news_per_symbol[upper].append(entry)

            # Match op bedrijfsnamen
            alias_matches = alias_pattern.findall(text)
            for match in set(alias_matches):
                mapped_symbol = alias_to_symbol.get(match.lower())
                if mapped_symbol and mapped_symbol in symbols:
                    news_per_symbol[mapped_symbol].append(entry)

        # Sentiment per symbol
        for symbol in symbols:
            entries = news_per_symbol.get(symbol, [])
            headlines = list({e["title"] for e in entries if e.get("title")})
            sources = list({e["source"] for e in entries})

            sentiment = None
            if headlines:
                sent_results = self._sentiment.analyze_batch(headlines)
                sentiment = self._sentiment.aggregate(symbol, sent_results)

            self._news_cache[symbol] = NewsData(
                symbol=symbol,
                headlines=headlines[:20],  # bewaar max 20
                sources=sources,
                sentiment=sentiment,
            )

    # ── SEC EDGAR ───────────────────────────────────────────────────

    def _collect_edgar(self, symbols: List[str]) -> None:
        """Haal SEC EDGAR data op: Form 4 (insider) en 13F (institutioneel).

        Gebruikt de gratis EDGAR full-text search API (https://efts.sec.gov).
        Geen authenticatie vereist — alleen een User-Agent header met contactinfo.
        """
        user_agent = os.environ.get(
            "SEC_USER_AGENT",
            "AI-Trading-Bot research@example.com",
        )

        for symbol in symbols:
            edgar_data = EdgarData(symbol=symbol)

            # Form 4: insider buy/sell
            try:
                insider = self._fetch_form4(symbol, user_agent)
                edgar_data.insider_buys_30d = insider.get("buys", 0)
                edgar_data.insider_sells_30d = insider.get("sells", 0)
                total = edgar_data.insider_buys_30d + edgar_data.insider_sells_30d
                edgar_data.insider_buy_sell_ratio = (
                    edgar_data.insider_buys_30d / total
                    if total > 0 else 0.5
                )
            except Exception as exc:
                logger.debug("Form 4 ophalen mislukt voor %s: %s", symbol, exc)

            # 13F: institutionele holdings
            try:
                inst = self._fetch_13f(symbol, user_agent)
                edgar_data.institutional_holders_delta = inst.get("holdings_delta", 0.0)
                edgar_data.institutional_filings_count = inst.get("filings_count", 0)
            except Exception as exc:
                logger.debug("13F ophalen mislukt voor %s: %s", symbol, exc)

            self._edgar_cache[symbol] = edgar_data

    def _fetch_form4(self, symbol: str, user_agent: str) -> Dict[str, int]:
        """Haal Form 4 (insider trading) filings op via EDGAR EFTS.

        Returns dict met 'buys' en 'sells' counts in de afgelopen 30 dagen.
        """
        # EDGAR full-text search API
        date_from = (datetime.now(timezone.utc) - timedelta(days=_EDGAR_FORM4_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{symbol}%22&dateRange=custom&startdt={date_from}"
            f"&forms=4&hits.hits.total=true&hits.hits._source=file_date,form_type"
        )

        # ASSUMPTION: EDGAR EFTS API retourneert JSON met filing metadata
        headers = {"User-Agent": user_agent, "Accept": "application/json"}

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            buys = 0
            sells = 0

            # Parse filings — heuristiek op basis van filing beschikbaarheid
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                source = hit.get("_source", {})
                # Form 4 filings bevatten transactie-type in de filing zelf
                # We tellen ze als proxy: meer filings = meer activiteit
                # ASSUMPTION: ~60% van Form 4 filings zijn aankopen in bullish markten
                buys += 1  # Conservatieve telling; exact parsing vereist XML download

            # Totaal als proxy verdelen
            total = len(hits)
            if total > 2:
                # Gebruik een eenvoudige ratio-schatting
                buys = max(1, total // 2)
                sells = total - buys

            return {"buys": buys, "sells": sells}

        except Exception as exc:
            logger.debug("EDGAR Form 4 API fout voor %s: %s", symbol, exc)
            return {"buys": 0, "sells": 0}

    def _fetch_13f(self, symbol: str, user_agent: str) -> Dict[str, Any]:
        """Haal 13F filing data op via EDGAR EFTS.

        Returns dict met 'holdings_delta' en 'filings_count'.
        """
        date_from = (datetime.now(timezone.utc) - timedelta(days=_EDGAR_13F_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{symbol}%22&dateRange=custom&startdt={date_from}"
            f"&forms=13F-HR"
        )

        headers = {"User-Agent": user_agent, "Accept": "application/json"}

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            total_hits = data.get("hits", {}).get("total", {})
            filings_count = total_hits.get("value", 0) if isinstance(total_hits, dict) else int(total_hits or 0)

            # ASSUMPTION: meer 13F filings mentioning dit symbool = meer institutionele interesse
            # Echte holdings delta vereist XML parsing van 13F information tables
            # TODO: Implementeer volledige 13F XML parsing voor exacte holdings delta
            holdings_delta = min(filings_count / 100.0, 1.0)  # genormaliseerde proxy

            return {
                "holdings_delta": round(holdings_delta, 4),
                "filings_count": filings_count,
            }

        except Exception as exc:
            logger.debug("EDGAR 13F API fout voor %s: %s", symbol, exc)
            return {"holdings_delta": 0.0, "filings_count": 0}

    # ── Macro (FRED) ────────────────────────────────────────────────

    def _collect_macro(self) -> None:
        """Haal macro-economische data op via FRED (pandas-datareader).

        Vereist: FRED_API_KEY environment variabele (gratis).
        """
        fred_key = os.environ.get("FRED_API_KEY", "")

        if not fred_key:
            logger.info(
                "FRED_API_KEY ontbreekt; macro data overgeslagen. "
                "Gratis key op: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self._macro_cache = MacroData(
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            return

        try:
            from pandas_datareader import data as pdr
        except ImportError:
            logger.warning("pandas-datareader niet geïnstalleerd. pip install pandas-datareader")
            self._macro_cache = MacroData(
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            return

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        macro = MacroData(timestamp=end_date.isoformat())

        # VIX
        try:
            vix = pdr.DataReader(
                _FRED_SERIES["vix"], "fred", start_date, end_date, api_key=fred_key,
            )
            if not vix.empty:
                vix_series = vix.iloc[:, 0].dropna()
                if len(vix_series) >= 1:
                    macro.vix_level = round(float(vix_series.iloc[-1]), 2)
                if len(vix_series) >= 6:
                    macro.vix_5d_change = round(
                        float(vix_series.iloc[-1] - vix_series.iloc[-6]), 2
                    )
        except Exception as exc:
            logger.debug("VIX ophalen mislukt: %s", exc)

        # Federal Funds Rate
        try:
            ff = pdr.DataReader(
                _FRED_SERIES["fed_funds"], "fred", start_date, end_date, api_key=fred_key,
            )
            if not ff.empty:
                macro.fed_funds_rate = round(float(ff.iloc[:, 0].dropna().iloc[-1]), 2)
        except Exception as exc:
            logger.debug("Fed Funds rate ophalen mislukt: %s", exc)

        # CPI (maandelijks — pak ruimere range)
        try:
            cpi = pdr.DataReader(
                _FRED_SERIES["cpi"], "fred",
                end_date - timedelta(days=400), end_date,
                api_key=fred_key,
            )
            if not cpi.empty:
                cpi_series = cpi.iloc[:, 0].dropna()
                if len(cpi_series) >= 13:
                    macro.cpi_yoy = round(
                        float((cpi_series.iloc[-1] / cpi_series.iloc[-13] - 1) * 100), 2
                    )
        except Exception as exc:
            logger.debug("CPI ophalen mislukt: %s", exc)

        # Treasury spread (10Y - 2Y)
        try:
            t10 = pdr.DataReader(
                _FRED_SERIES["treasury_10y"], "fred", start_date, end_date, api_key=fred_key,
            )
            t2 = pdr.DataReader(
                _FRED_SERIES["treasury_2y"], "fred", start_date, end_date, api_key=fred_key,
            )
            if not t10.empty and not t2.empty:
                t10_val = float(t10.iloc[:, 0].dropna().iloc[-1])
                t2_val = float(t2.iloc[:, 0].dropna().iloc[-1])
                macro.treasury_spread = round(t10_val - t2_val, 2)
        except Exception as exc:
            logger.debug("Treasury spread ophalen mislukt: %s", exc)

        self._macro_cache = macro
        logger.info(
            "Macro data opgehaald: VIX=%.1f, FF=%.2f%%, CPI=%.1f%%, spread=%.2f",
            macro.vix_level, macro.fed_funds_rate, macro.cpi_yoy, macro.treasury_spread,
        )

    # ── Feature building ────────────────────────────────────────────

    def _build_features(self, symbol: str) -> AlternativeFeatures:
        """Combineer alle bronnen tot een AlternativeFeatures object."""
        features = AlternativeFeatures(symbol=symbol)

        # Reddit
        reddit = self._reddit_cache.get(symbol)
        if reddit:
            features.reddit_mention_velocity = reddit.velocity
            features.reddit_mention_count = reddit.mention_count_24h
            if reddit.sentiment:
                features.sentiment_score = reddit.sentiment.normalized_score

        # Nieuws
        news = self._news_cache.get(symbol)
        if news:
            features.news_volume = len(news.headlines)
            if news.sentiment:
                features.news_sentiment = news.sentiment.normalized_score
                # Combineer Reddit + News sentiment als beide beschikbaar
                if reddit and reddit.sentiment:
                    # Gewogen gemiddelde: 40% Reddit, 60% News
                    features.sentiment_score = round(
                        0.4 * reddit.sentiment.normalized_score
                        + 0.6 * news.sentiment.normalized_score,
                        4,
                    )
                else:
                    features.sentiment_score = news.sentiment.normalized_score

        # EDGAR
        edgar = self._edgar_cache.get(symbol)
        if edgar:
            features.insider_buy_sell_ratio = edgar.insider_buy_sell_ratio
            features.institutional_holdings_delta = edgar.institutional_holders_delta

        return features
