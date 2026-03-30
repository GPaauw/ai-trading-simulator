import csv
import json
import os
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen


class MarketDataService:
    """Haalt gratis marktdata op met een lichte cache om rate limits te beperken."""

    WATCHLIST: List[Dict[str, str]] = [
        {"symbol": "AAPL", "provider_symbol": "aapl.us", "market": "us"},
        {"symbol": "MSFT", "provider_symbol": "msft.us", "market": "us"},
        {"symbol": "NVDA", "provider_symbol": "nvda.us", "market": "us"},
        {"symbol": "AMZN", "provider_symbol": "amzn.us", "market": "us"},
        {"symbol": "ASML", "provider_symbol": "asml.nl", "market": "eu"},
        {"symbol": "SAP", "provider_symbol": "sap.de", "market": "eu"},
        {"symbol": "SIE", "provider_symbol": "sie.de", "market": "eu"},
        {"symbol": "BMW", "provider_symbol": "bmw.de", "market": "eu"},
        {"symbol": "BTC", "provider_symbol": "BTCUSDT", "market": "crypto"},
        {"symbol": "ETH", "provider_symbol": "ETHUSDT", "market": "crypto"},
    ]

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, object]] = {}
        self._headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ai-trading-simulator/1.0)",
            "Accept": "application/json,text/plain,*/*",
        }
        # Optional: laad een uitgebreider markt-universum uit een CSV-bestand als
        # het pad is opgegeven via de env-var `MARKET_UNIVERSE_FILE`.
        self._extended_watchlist: List[Dict[str, str]] | None = None
        universe_file = os.getenv("MARKET_UNIVERSE_FILE")
        if universe_file:
            p = Path(universe_file)
            if p.exists() and p.is_file():
                try:
                    self._extended_watchlist = self._read_watchlist_csv(p)
                except Exception:
                    # Fouten bij het inlezen negeren; fallback naar ingebouwde watchlist
                    self._extended_watchlist = None

    def get_watchlist(self) -> List[Dict[str, str]]:
        # Als er een uitgebreidere watchlist beschikbaar is, gebruik die.
        return self._extended_watchlist or self.WATCHLIST

    def _read_watchlist_csv(self, path: Path) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        with path.open(newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                symbol = (row.get('symbol') or '').strip()
                provider = (row.get('provider_symbol') or '').strip()
                market = (row.get('market') or '').strip()
                if not symbol or not provider or not market:
                    continue
                items.append({"symbol": symbol.upper(), "provider_symbol": provider, "market": market})
        return items

    def get_instrument(self, symbol: str, market: str | None = None) -> Dict[str, str]:
        normalized_symbol = symbol.upper()
        candidates = [item for item in self.WATCHLIST if item["symbol"] == normalized_symbol]
        if market:
            candidates = [item for item in candidates if item["market"] == market]
        if not candidates:
            raise ValueError(f"Instrument niet in watchlist: {symbol}")
        return candidates[0]

    def get_snapshot(self, instrument: Dict[str, str]) -> Dict[str, float]:
        key = f"snapshot:{instrument['provider_symbol']}"
        cached = self._get_cache(key)
        if cached:
            return cached  # type: ignore[return-value]

        if instrument["market"] == "crypto":
            snapshot = self._fetch_crypto_snapshot(instrument["provider_symbol"])
        else:
            snapshot = self._fetch_stooq_snapshot(instrument["provider_symbol"])

        self._set_cache(key, snapshot, ttl_seconds=45)
        return snapshot

    def get_history(self, instrument: Dict[str, str], points: int = 120) -> List[float]:
        key = f"history:{instrument['provider_symbol']}:{points}"
        cached = self._get_cache(key)
        if cached:
            return cached  # type: ignore[return-value]

        if instrument["market"] == "crypto":
            closes = self._fetch_crypto_history(instrument["provider_symbol"], points)
        else:
            closes = self._fetch_stooq_daily_history(instrument["provider_symbol"], points)

        self._set_cache(key, closes, ttl_seconds=5 * 60)
        return closes

    def _fetch_stooq_snapshot(self, provider_symbol: str) -> Dict[str, float]:
        url = f"https://stooq.com/q/l/?s={provider_symbol}&i=1"
        raw = self._get_text(url).strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 8 or parts[6] == "N/D":
            raise ValueError(f"Geen snapshot beschikbaar voor {provider_symbol}")

        return {
            "open": float(parts[3]),
            "high": float(parts[4]),
            "low": float(parts[5]),
            "price": float(parts[6]),
            "volume": float(parts[7] or 0),
        }

    def _fetch_stooq_daily_history(self, provider_symbol: str, points: int) -> List[float]:
        url = f"https://stooq.com/q/d/l/?s={provider_symbol}&i=d"
        raw = self._get_text(url)
        reader = csv.DictReader(StringIO(raw))
        closes: List[float] = []
        for row in reader:
            close_value = row.get("Close")
            if not close_value:
                continue
            try:
                closes.append(float(close_value))
            except ValueError:
                continue
        return closes[-points:]

    def _fetch_crypto_snapshot(self, provider_symbol: str) -> Dict[str, float]:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={provider_symbol}"
        data = self._get_json(url)
        return {
            "open": float(data["openPrice"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "price": float(data["lastPrice"]),
            "volume": float(data["volume"]),
        }

    def _fetch_crypto_history(self, provider_symbol: str, points: int) -> List[float]:
        limit = max(30, min(points, 500))
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={provider_symbol}&interval=15m"
            f"&limit={limit}"
        )
        data = self._get_json(url)
        closes: List[float] = []
        for row in data:
            closes.append(float(row[4]))
        return closes

    def _get_json(self, url: str):
        req = Request(url, headers=self._headers)
        with urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_text(self, url: str) -> str:
        req = Request(url, headers=self._headers)
        with urlopen(req, timeout=12) as resp:
            return resp.read().decode("utf-8")

    def _set_cache(self, key: str, value, ttl_seconds: int) -> None:
        self._cache[key] = {
            "expires_at": time.time() + ttl_seconds,
            "value": value,
        }

    def _get_cache(self, key: str):
        item = self._cache.get(key)
        if not item:
            return None
        if item["expires_at"] < time.time():
            self._cache.pop(key, None)
            return None
        return item["value"]