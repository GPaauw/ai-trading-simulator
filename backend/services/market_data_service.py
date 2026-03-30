import gc
import json
import logging
import os
import re
import time
import threading
from typing import Dict, List
from urllib.request import Request, urlopen

import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataService:
    """Haalt gratis marktdata op. Downloadt automatisch S&P 500, Nasdaq-100 en top crypto."""

    # Fallback hardcoded watchlist voor als de download faalt
    _FALLBACK_STOCKS: List[Dict[str, str]] = [
        {"symbol": "AAPL", "market": "us"},
        {"symbol": "MSFT", "market": "us"},
        {"symbol": "NVDA", "market": "us"},
        {"symbol": "AMZN", "market": "us"},
        {"symbol": "GOOGL", "market": "us"},
        {"symbol": "META", "market": "us"},
        {"symbol": "TSLA", "market": "us"},
        {"symbol": "AVGO", "market": "us"},
        {"symbol": "JPM", "market": "us"},
        {"symbol": "V", "market": "us"},
    ]

    _CRYPTO_SYMBOLS: List[Dict[str, str]] = [
        {"symbol": "BTC", "provider_symbol": "BTCUSDT", "market": "crypto"},
        {"symbol": "ETH", "provider_symbol": "ETHUSDT", "market": "crypto"},
        {"symbol": "SOL", "provider_symbol": "SOLUSDT", "market": "crypto"},
        {"symbol": "BNB", "provider_symbol": "BNBUSDT", "market": "crypto"},
        {"symbol": "XRP", "provider_symbol": "XRPUSDT", "market": "crypto"},
        {"symbol": "ADA", "provider_symbol": "ADAUSDT", "market": "crypto"},
        {"symbol": "DOGE", "provider_symbol": "DOGEUSDT", "market": "crypto"},
        {"symbol": "AVAX", "provider_symbol": "AVAXUSDT", "market": "crypto"},
        {"symbol": "DOT", "provider_symbol": "DOTUSDT", "market": "crypto"},
        {"symbol": "LINK", "provider_symbol": "LINKUSDT", "market": "crypto"},
    ]

    _COMMODITY_SYMBOLS: List[Dict[str, str]] = [
        {"symbol": "GOLD", "yf_symbol": "GC=F", "market": "commodity"},
        {"symbol": "SILVER", "yf_symbol": "SI=F", "market": "commodity"},
        {"symbol": "OIL", "yf_symbol": "CL=F", "market": "commodity"},
        {"symbol": "NATGAS", "yf_symbol": "NG=F", "market": "commodity"},
        {"symbol": "COPPER", "yf_symbol": "HG=F", "market": "commodity"},
        {"symbol": "PLATINUM", "yf_symbol": "PL=F", "market": "commodity"},
    ]

    # Batch-grootte voor yfinance downloads
    _DEFAULT_YF_BATCH_SIZE = 15

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, object]] = {}
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
        }
        self._watchlist: List[Dict[str, str]] | None = None
        self._watchlist_loaded_at: float = 0.0
        self._watchlist_ttl = 6 * 3600  # herlaad elke 6 uur
        self._stock_data_loaded: bool = False
        self._stock_data_time: float = 0.0
        self._stock_data_ttl = 5 * 60  # herlaad stock data elke 5 min
        # Prefetch scheduler state
        self._prefetch_thread = None
        self._prefetch_stop_event = threading.Event()
        self._prefetch_lock = threading.Lock()
        self._prefetch_status: Dict[str, object] = {
            "is_running": False,
            "last_run_start": None,
            "last_run_end": None,
            "last_run_elapsed": None,
            "last_run_count": 0,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def get_watchlist(self) -> List[Dict[str, str]]:
        now = time.time()
        if self._watchlist and (now - self._watchlist_loaded_at) < self._watchlist_ttl:
            return self._watchlist

        stocks = self._download_index_constituents()
        self._watchlist = stocks + self._CRYPTO_SYMBOLS + self._COMMODITY_SYMBOLS
        self._watchlist_loaded_at = now
        return self._watchlist

    def get_instrument(self, symbol: str, market: str | None = None) -> Dict[str, str]:
        normalized = symbol.upper()
        watchlist = self.get_watchlist()
        candidates = [item for item in watchlist if item["symbol"] == normalized]
        if market:
            candidates = [item for item in candidates if item["market"] == market]
        if not candidates:
            raise ValueError(f"Instrument niet in watchlist: {symbol}")
        return candidates[0]

    def get_fast_watchlist(self) -> List[Dict[str, str]]:
        """Kleine watchlist voor snelle cold-start responses."""
        return self._FALLBACK_STOCKS + self._CRYPTO_SYMBOLS + self._COMMODITY_SYMBOLS

    def has_warm_stock_cache(self) -> bool:
        """Geeft aan of er al voldoende stock-history in cache zit voor een volledige scan."""
        if not self._stock_data_loaded:
            return False
        cached_count = sum(1 for key in self._cache if key.startswith("history:yf:"))
        return cached_count >= max(10, len(self._FALLBACK_STOCKS))

    def ensure_background_prefetch_started(self, interval_seconds: int = 300) -> None:
        """Start background prefetch alleen als de stock-cache nog niet warm is."""
        if self.has_warm_stock_cache():
            return
        self.start_prefetch_scheduler(interval_seconds=interval_seconds)

    # ------------------------------------------------------------------
    # Batch prefetch: download alle aandelen in één keer
    # ------------------------------------------------------------------

    def prefetch_stock_data(self, force: bool = False, instruments: List[Dict[str, str]] | None = None) -> None:
        """Download alle stock-data in batches via yf.download(). Vult cache."""
        now = time.time()
        if instruments is None and not force and self._stock_data_loaded and (now - self._stock_data_time) < self._stock_data_ttl:
            return

        batch_size = self._get_yf_batch_size()
        use_yf_threads = self._use_yf_threads()

        watchlist = instruments or self.get_watchlist()
        non_crypto = [inst for inst in watchlist if inst["market"] != "crypto"]
        if not non_crypto:
            return

        # Mapping van yfinance-ticker naar display-symbool
        yf_to_display: Dict[str, str] = {}
        yf_syms: List[str] = []
        for inst in non_crypto:
            yf_sym = inst.get("yf_symbol", inst["symbol"])
            yf_to_display[yf_sym] = inst["symbol"]
            yf_syms.append(yf_sym)

        logger.info("Batch-download %d instrumenten via yfinance...", len(yf_syms))
        start = time.time()

        # Download in kleinere batches om memory-pieken op kleine instances te voorkomen.
        for i in range(0, len(yf_syms), batch_size):
            batch = yf_syms[i : i + batch_size]
            try:
                if len(batch) == 1:
                    data = yf.download(batch[0], period="6mo", progress=False)
                    if data.empty:
                        del data
                        gc.collect()
                        continue
                    close_col = data["Close"].dropna()
                    closes = list(close_col.values.flatten())
                    if len(closes) >= 30:
                        last_row = data.iloc[-1]
                        self._cache_stock(yf_to_display[batch[0]], closes, last_row)
                    del data
                    gc.collect()
                else:
                    data = yf.download(
                        batch, period="6mo", group_by="ticker",
                        threads=use_yf_threads, progress=False,
                    )
                    if data.empty:
                        del data
                        gc.collect()
                        continue
                    for yf_sym in batch:
                        try:
                            sym_data = data[yf_sym]
                            close_col = sym_data["Close"].dropna()
                            closes = list(close_col.values.flatten())
                            if len(closes) >= 30:
                                last_row = sym_data.dropna().iloc[-1]
                                self._cache_stock(yf_to_display[yf_sym], closes, last_row)
                        except (KeyError, IndexError):
                            continue
                    del data
                    gc.collect()
            except Exception as exc:
                logger.warning("Batch-download fout voor %s: %s", batch[:3], exc)

        elapsed = time.time() - start
        cached_count = sum(1 for k in self._cache if k.startswith("history:yf:"))
        logger.info("Data geladen: %d/%d instrumenten in %.1fs", cached_count, len(yf_syms), elapsed)
        if instruments is None:
            self._stock_data_loaded = True
            self._stock_data_time = time.time()

    def _get_yf_batch_size(self) -> int:
        raw_value = os.environ.get("PREFETCH_BATCH_SIZE", str(self._DEFAULT_YF_BATCH_SIZE)).strip()
        try:
            batch_size = int(raw_value)
        except ValueError:
            batch_size = self._DEFAULT_YF_BATCH_SIZE
        return max(1, min(batch_size, 50))

    def _use_yf_threads(self) -> bool:
        raw_value = os.environ.get("PREFETCH_YF_THREADS", "false").strip().lower()
        return raw_value in {"1", "true", "yes"}

    def _cache_stock(self, symbol: str, closes: List[float], last_row) -> None:
        """Sla stock-data op in cache."""
        # History cache (lang TTL, wijzigt pas bij volgende batch)
        self._set_cache(f"history:yf:{symbol}", closes, ttl_seconds=self._stock_data_ttl + 60)
        # Snapshot cache
        try:
            snapshot = {
                "open": float(last_row.get("Open", closes[-1])),
                "high": float(last_row.get("High", closes[-1])),
                "low": float(last_row.get("Low", closes[-1])),
                "price": float(closes[-1]),
                "volume": float(last_row.get("Volume", 0)),
            }
            self._set_cache(f"snapshot:yf:{symbol}", snapshot, ttl_seconds=self._stock_data_ttl + 60)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Snapshot & History
    # ------------------------------------------------------------------

    def get_snapshot(self, instrument: Dict[str, str]) -> Dict[str, float]:
        if instrument["market"] == "crypto":
            key = f"snapshot:{instrument['provider_symbol']}"
            cached = self._get_cache(key)
            if cached:
                return cached  # type: ignore[return-value]
            snapshot = self._fetch_crypto_snapshot(instrument["provider_symbol"])
            self._set_cache(key, snapshot, ttl_seconds=30)
            return snapshot
        else:
            key = f"snapshot:yf:{instrument['symbol']}"
            cached = self._get_cache(key)
            if cached:
                return cached  # type: ignore[return-value]
            # Individuele fallback als batch-data niet beschikbaar is
            yf_symbol = instrument.get("yf_symbol", instrument["symbol"])
            snapshot = self._fetch_stock_snapshot_individual(yf_symbol)
            self._set_cache(key, snapshot, ttl_seconds=60)
            return snapshot

    def get_history(self, instrument: Dict[str, str], points: int = 120) -> List[float]:
        if instrument["market"] == "crypto":
            key = f"history:{instrument['provider_symbol']}:{points}"
            cached = self._get_cache(key)
            if cached:
                return cached  # type: ignore[return-value]
            closes = self._fetch_crypto_history(instrument["provider_symbol"], points)
            self._set_cache(key, closes, ttl_seconds=3 * 60)
            return closes
        else:
            key = f"history:yf:{instrument['symbol']}"
            cached = self._get_cache(key)
            if cached:
                closes = cached  # type: ignore[assignment]
                return closes[-points:]
            # Individuele fallback
            yf_symbol = instrument.get("yf_symbol", instrument["symbol"])
            closes = self._fetch_stock_history_individual(yf_symbol, points)
            self._set_cache(key, closes, ttl_seconds=3 * 60)
            return closes

    def invalidate_cache(self) -> None:
        """Verwijder snapshot-cache en markeer stock data als verlopen."""
        keys_to_remove = [k for k in self._cache if k.startswith("snapshot:")]
        for k in keys_to_remove:
            self._cache.pop(k, None)
        self._stock_data_loaded = False

    # ------------------------------------------------------------------
    # Index-download: S&P 500 + Nasdaq-100
    # ------------------------------------------------------------------

    def _download_index_constituents(self) -> List[Dict[str, str]]:
        """Download S&P 500 en Nasdaq-100 tickers van Wikipedia."""
        seen: set[str] = set()
        instruments: List[Dict[str, str]] = []

        # S&P 500
        try:
            sp500 = self._fetch_wikipedia_table(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                symbol_col=0,
            )
            for symbol in sp500:
                if symbol not in seen:
                    seen.add(symbol)
                    instruments.append({"symbol": symbol, "market": "us"})
        except Exception:
            pass

        # Nasdaq-100
        try:
            ndx = self._fetch_wikipedia_table(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                symbol_col=1,
            )
            for symbol in ndx:
                if symbol not in seen:
                    seen.add(symbol)
                    instruments.append({"symbol": symbol, "market": "us"})
        except Exception:
            pass

        if not instruments:
            return self._FALLBACK_STOCKS

        return instruments

    def _fetch_wikipedia_table(self, url: str, symbol_col: int) -> List[str]:
        """Parseer de eerste HTML tabel op een Wikipedia pagina en return ticker symbols."""
        html = self._get_text(url)
        table_match = re.search(
            r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>',
            html,
            re.DOTALL,
        )
        if not table_match:
            return []

        table_html = table_match.group(1)
        symbols: List[str] = []

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
        for row in rows[1:]:  # skip header
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
            if len(cells) <= symbol_col:
                continue

            cell_text = re.sub(r"<[^>]+>", "", cells[symbol_col]).strip()
            symbol = re.sub(r"\[.*?\]", "", cell_text).strip()
            if not symbol or not re.match(r"^[A-Z]{1,5}$", symbol):
                continue
            symbols.append(symbol)

        return symbols

    # ------------------------------------------------------------------
    # Individuele stock fetchers (fallback als batch niet beschikbaar)
    # ------------------------------------------------------------------

    def _fetch_stock_snapshot_individual(self, symbol: str) -> Dict[str, float]:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = float(info.last_price)
        prev_close = float(info.previous_close) if info.previous_close else price
        day_high = float(info.day_high) if info.day_high else price
        day_low = float(info.day_low) if info.day_low else price
        day_open = float(info.open) if info.open else prev_close
        volume = float(info.last_volume) if info.last_volume else 0.0
        return {
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "price": price,
            "volume": volume,
        }

    def _fetch_stock_history_individual(self, symbol: str, points: int) -> List[float]:
        ticker = yf.Ticker(symbol)
        period = "6mo" if points <= 130 else "1y"
        hist = ticker.history(period=period)
        if hist.empty:
            return []
        closes = hist["Close"].dropna().tolist()
        return closes[-points:]

    # ------------------------------------------------------------------
    # Crypto fetchers (Binance, 5 min candles)
    # ------------------------------------------------------------------

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
            f"https://api.binance.com/api/v3/klines?symbol={provider_symbol}&interval=5m"
            f"&limit={limit}"
        )
        data = self._get_json(url)
        closes: List[float] = []
        for row in data:
            closes.append(float(row[4]))
        return closes

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str):
        req = Request(url, headers=self._headers)
        with urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_text(self, url: str) -> str:
        req = Request(url, headers=self._headers)
        with urlopen(req, timeout=12) as resp:
            return resp.read().decode("utf-8")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _set_cache(self, key: str, value: object, ttl_seconds: int) -> None:
        self._cache[key] = {
            "expires_at": time.time() + ttl_seconds,
            "value": value,
        }

    def _get_cache(self, key: str) -> object | None:
        item = self._cache.get(key)
        if not item:
            return None
        if item["expires_at"] < time.time():  # type: ignore[operator]
            self._cache.pop(key, None)
            return None
        return item["value"]

    # ------------------------------------------------------------------
    # Prefetch scheduler / status
    # ------------------------------------------------------------------

    def start_prefetch_scheduler(self, interval_seconds: int = 300) -> None:
        """Start een achtergrond-thread die elke `interval_seconds` de batch-prefetch uitvoert."""
        if self._prefetch_thread and getattr(self._prefetch_thread, "is_alive", lambda: False)():
            return
        self._prefetch_stop_event.clear()
        t = threading.Thread(target=self._prefetch_loop, args=(interval_seconds,), daemon=True)
        self._prefetch_thread = t
        t.start()

    def stop_prefetch_scheduler(self) -> None:
        """Stop de achtergrond prefetch-thread indien actief."""
        if self._prefetch_thread and getattr(self._prefetch_thread, "is_alive", lambda: False)():
            self._prefetch_stop_event.set()
            try:
                self._prefetch_thread.join(timeout=5)
            except Exception:
                pass

    def _prefetch_loop(self, interval_seconds: int) -> None:
        while not self._prefetch_stop_event.is_set():
            with self._prefetch_lock:
                self._prefetch_status["is_running"] = True
                self._prefetch_status["last_run_start"] = time.time()
                self._prefetch_status["error"] = None
            try:
                self.prefetch_stock_data(force=True)
                cached_count = sum(1 for k in self._cache if k.startswith("history:yf:"))
                with self._prefetch_lock:
                    self._prefetch_status["last_run_count"] = cached_count
            except Exception as exc:
                logger.exception("Prefetch error: %s", exc)
                with self._prefetch_lock:
                    self._prefetch_status["error"] = str(exc)
            finally:
                with self._prefetch_lock:
                    end = time.time()
                    start = self._prefetch_status.get("last_run_start") or end
                    self._prefetch_status["last_run_end"] = end
                    self._prefetch_status["last_run_elapsed"] = round(end - start, 2)
                    self._prefetch_status["is_running"] = False
            # wacht of we moeten stoppen of wachten tot volgende run
            self._prefetch_stop_event.wait(interval_seconds)

    def prefetch_once(self) -> None:
        """Voer één directe prefetch uit (blokkerend)."""
        with self._prefetch_lock:
            self._prefetch_status["is_running"] = True
            self._prefetch_status["last_run_start"] = time.time()
            self._prefetch_status["error"] = None
        try:
            self.prefetch_stock_data(force=True)
            cached_count = sum(1 for k in self._cache if k.startswith("history:yf:"))
            with self._prefetch_lock:
                self._prefetch_status["last_run_count"] = cached_count
        except Exception as exc:
            with self._prefetch_lock:
                self._prefetch_status["error"] = str(exc)
            raise
        finally:
            with self._prefetch_lock:
                end = time.time()
                start = self._prefetch_status.get("last_run_start") or end
                self._prefetch_status["last_run_end"] = end
                self._prefetch_status["last_run_elapsed"] = round(end - start, 2)
                self._prefetch_status["is_running"] = False

    def get_prefetch_status(self) -> Dict[str, object]:
        with self._prefetch_lock:
            return dict(self._prefetch_status)
