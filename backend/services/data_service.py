"""Data Service: PostgreSQL (Supabase) database abstractie.

Beheert alle persistente data: trades, holdings, portfolio snapshots,
feature snapshots (voor ML retraining) en model performance metrics.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from models.trade import TradeResult

logger = logging.getLogger(__name__)

SELL_QUANTITY_EPSILON = 1e-6
_DATABASE_URL = os.getenv("DATABASE_URL", "")


class _PgCursor:
    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> Any:
        return self._cur.fetchall()


class _PgConn:
    def __init__(self, dsn: str, retries: int = 3, backoff: float = 2.0) -> None:
        import psycopg2
        import psycopg2.extras

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self._conn = psycopg2.connect(
                    dsn, cursor_factory=psycopg2.extras.RealDictCursor, connect_timeout=10,
                )
                return
            except psycopg2.OperationalError as exc:
                last_err = exc
                logger.warning("PostgreSQL connect poging %d/%d mislukt: %s", attempt, retries, exc)
                if attempt < retries:
                    time.sleep(backoff * attempt)
        raise last_err  # type: ignore[misc]

    def execute(self, sql: str, params: tuple = ()) -> _PgCursor:
        pg_sql = sql.replace("?", "%s")
        cur = self._conn.cursor()
        cur.execute(pg_sql, params)
        return _PgCursor(cur)

    def commit(self) -> None:
        self._conn.commit()

    def __enter__(self) -> "_PgConn":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()


class DataService:
    def __init__(self) -> None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "trading.sqlite3",
        )
        self._db_path = os.getenv("TRADING_DB_PATH", default_path)
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._configured_start_balance = float(os.getenv("START_BALANCE", "5000"))
        self._pg_unavailable = False
        self._init_db()

    # ── Trades ──────────────────────────────────────────────────────

    def get_trade_history(self, limit: int = 0) -> List[TradeResult]:
        with self._connect() as conn:
            sql = """
                SELECT id, symbol, action, amount, quantity, price, profit_loss,
                       timestamp, status, expected_return_pct, risk_pct,
                       confidence, model_version
                FROM trade_history ORDER BY timestamp ASC
            """
            if limit > 0:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql).fetchall()
        return [self._trade_from_row(row) for row in rows]

    def add_trade(self, trade: TradeResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_history (
                    id, symbol, action, amount, quantity, price, profit_loss,
                    timestamp, status, expected_return_pct, risk_pct, confidence, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    symbol=excluded.symbol, action=excluded.action, amount=excluded.amount,
                    quantity=excluded.quantity, price=excluded.price, profit_loss=excluded.profit_loss,
                    timestamp=excluded.timestamp, status=excluded.status,
                    expected_return_pct=excluded.expected_return_pct, risk_pct=excluded.risk_pct,
                    confidence=excluded.confidence, model_version=excluded.model_version
                """,
                (trade.id, trade.symbol, trade.action, trade.amount, trade.quantity,
                 trade.price, trade.profit_loss, trade.timestamp, trade.status,
                 trade.expected_return_pct, trade.risk_pct, trade.confidence, trade.model_version),
            )
            conn.commit()

    # ── Holdings ────────────────────────────────────────────────────

    def get_holdings(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at "
                "FROM holdings ORDER BY symbol ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_holding(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at "
                "FROM holdings WHERE symbol = ?", (symbol.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def record_buy(self, *, symbol: str, market: str, quantity: float, amount: float, price: float) -> Dict[str, Any]:
        normalized = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()
        current_cash = self.get_cash_balance()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at "
                "FROM holdings WHERE symbol = ?", (normalized,),
            ).fetchone()

            if existing:
                total_qty = float(existing["quantity"]) + quantity
                total_inv = float(existing["invested_amount"]) + amount
                avg_price = total_inv / total_qty
                conn.execute(
                    "UPDATE holdings SET quantity=?, invested_amount=?, avg_entry_price=?, updated_at=? WHERE symbol=?",
                    (total_qty, total_inv, avg_price, now, normalized),
                )
                holding = {"symbol": normalized, "market": str(existing["market"]),
                           "quantity": total_qty, "invested_amount": total_inv,
                           "avg_entry_price": avg_price, "opened_at": str(existing["opened_at"]), "updated_at": now}
            else:
                conn.execute(
                    "INSERT INTO holdings(symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (normalized, market, quantity, amount, price, now, now),
                )
                holding = {"symbol": normalized, "market": market, "quantity": quantity,
                           "invested_amount": amount, "avg_entry_price": price, "opened_at": now, "updated_at": now}

            conn.execute(
                "INSERT INTO settings(key, value) VALUES ('cash_balance', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(current_cash - amount),),
            )
            conn.commit()
        return holding

    def record_sell(self, *, symbol: str, quantity: float, price: float) -> Dict[str, float | str]:
        normalized = symbol.upper()
        current_cash = self.get_cash_balance()
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            holding = conn.execute(
                "SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at "
                "FROM holdings WHERE symbol = ?", (normalized,),
            ).fetchone()
            if not holding:
                raise ValueError(f"Geen positie gevonden voor {normalized}")

            current_qty = float(holding["quantity"])
            if abs(quantity - current_qty) <= SELL_QUANTITY_EPSILON:
                quantity = current_qty
            if quantity <= 0 or quantity - current_qty > SELL_QUANTITY_EPSILON:
                raise ValueError("Ongeldige verkoophoeveelheid")

            avg_entry = float(holding["avg_entry_price"])
            cost_basis = avg_entry * quantity
            proceeds = price * quantity
            realized_pl = proceeds - cost_basis
            remaining = current_qty - quantity

            if remaining <= SELL_QUANTITY_EPSILON:
                conn.execute("DELETE FROM holdings WHERE symbol = ?", (normalized,))
                status = "executed: positie gesloten"
            else:
                conn.execute(
                    "UPDATE holdings SET quantity=?, invested_amount=?, updated_at=? WHERE symbol=?",
                    (remaining, avg_entry * remaining, now, normalized),
                )
                status = "executed: gedeeltelijk verkocht"

            conn.execute(
                "INSERT INTO settings(key, value) VALUES ('cash_balance', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(current_cash + proceeds),),
            )
            conn.commit()

        return {"quantity": quantity, "proceeds": proceeds, "profit_loss": realized_pl, "status": status}

    # ── Cash & Portfolio ────────────────────────────────────────────

    def get_cash_balance(self) -> float:
        return float(self._get_setting("cash_balance", str(self._configured_start_balance)))

    def get_start_balance(self) -> float:
        return float(self._get_setting("start_balance", str(self._configured_start_balance)))

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Bedrag moet positief zijn")
        current = self.get_cash_balance()
        new_balance = current + amount
        self._set_setting("cash_balance", str(new_balance))
        start = self.get_start_balance()
        self._set_setting("start_balance", str(start + amount))
        return new_balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Bedrag moet positief zijn")
        current = self.get_cash_balance()
        if amount > current:
            raise ValueError(f"Onvoldoende cash (beschikbaar: €{current:.2f})")
        new_balance = current - amount
        self._set_setting("cash_balance", str(new_balance))
        start = self.get_start_balance()
        self._set_setting("start_balance", str(max(0, start - amount)))
        return new_balance

    def get_daily_trade_count(self) -> int:
        today = date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM trade_history WHERE timestamp LIKE ?", (f"{today}%",),
            ).fetchone()
        return int(row["count"]) if row else 0

    def get_daily_realized_pnl(self) -> float:
        today = date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(profit_loss), 0) AS total FROM trade_history "
                "WHERE timestamp LIKE ? AND action = 'sell'", (f"{today}%",),
            ).fetchone()
        return float(row["total"]) if row else 0.0

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT date, balance FROM portfolio_history ORDER BY date ASC").fetchall()
        return [{"date": row["date"], "equity": round(float(row["balance"]), 2)} for row in rows]

    def record_portfolio_snapshot(self, balance: float) -> None:
        today = date.today().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO portfolio_history(date, balance) VALUES (?, ?) "
                "ON CONFLICT(date) DO UPDATE SET balance=excluded.balance", (today, balance),
            )
            conn.commit()

    # ── Feature snapshots (voor ML retraining) ──────────────────────

    def save_feature_snapshot(self, symbol: str, features_json: str, signal_action: str, confidence: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feature_snapshots(symbol, timestamp, features, signal_action, confidence) "
                "VALUES (?, ?, ?, ?, ?)", (symbol.upper(), now, features_json, signal_action, confidence),
            )
            conn.commit()

    def get_feature_snapshots(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT symbol, timestamp, features, signal_action, confidence "
                f"FROM feature_snapshots ORDER BY timestamp DESC LIMIT {int(limit)}"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Model performance tracking ──────────────────────────────────

    def save_model_performance(self, metrics: Dict[str, Any]) -> None:
        today = date.today().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_performance(date, metrics) VALUES (?, ?) "
                "ON CONFLICT(date) DO UPDATE SET metrics=excluded.metrics",
                (today, json.dumps(metrics)),
            )
            conn.commit()

    def get_model_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT date, metrics FROM model_performance ORDER BY date DESC LIMIT {int(days)}"
            ).fetchall()
        result = []
        for r in rows:
            entry = {"date": r["date"]}
            try:
                entry["metrics"] = json.loads(r["metrics"])
            except Exception:
                entry["metrics"] = {}
            result.append(entry)
        return result

    # ── Settings ────────────────────────────────────────────────────

    def _get_setting(self, key: str, default: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def _set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    # ── Database init ───────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, action TEXT NOT NULL,
                    amount REAL NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
                    profit_loss REAL NOT NULL, timestamp TEXT NOT NULL, status TEXT NOT NULL,
                    expected_return_pct REAL NOT NULL DEFAULT 0, risk_pct REAL NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0, model_version TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY, market TEXT NOT NULL,
                    quantity REAL NOT NULL, invested_amount REAL NOT NULL,
                    avg_entry_price REAL NOT NULL, opened_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS portfolio_history (date TEXT PRIMARY KEY, balance REAL NOT NULL)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
                    features TEXT NOT NULL, signal_action TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS model_performance (date TEXT PRIMARY KEY, metrics TEXT NOT NULL)")

            if not conn.execute("SELECT value FROM settings WHERE key = 'start_balance'").fetchone():
                conn.execute("INSERT INTO settings(key, value) VALUES ('start_balance', ?)",
                             (str(self._configured_start_balance),))
            if not conn.execute("SELECT value FROM settings WHERE key = 'cash_balance'").fetchone():
                conn.execute("INSERT INTO settings(key, value) VALUES ('cash_balance', ?)",
                             (str(self._configured_start_balance),))
            conn.commit()

    def _connect(self) -> Any:
        if _DATABASE_URL and not self._pg_unavailable:
            try:
                return _PgConn(_DATABASE_URL)
            except Exception as exc:
                logger.error("PostgreSQL niet bereikbaar, val terug op SQLite: %s", exc)
                self._pg_unavailable = True
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _trade_from_row(self, row: Any) -> TradeResult:
        return TradeResult(
            id=str(row["id"]), symbol=str(row["symbol"]), action=str(row["action"]),
            amount=float(row["amount"]), quantity=float(row["quantity"]), price=float(row["price"]),
            profit_loss=float(row["profit_loss"]), timestamp=str(row["timestamp"]),
            status=str(row["status"]), expected_return_pct=float(row["expected_return_pct"]),
            risk_pct=float(row["risk_pct"]),
            confidence=float(row.get("confidence", 0) or 0),
            model_version=str(row.get("model_version", "") or ""),
        )
