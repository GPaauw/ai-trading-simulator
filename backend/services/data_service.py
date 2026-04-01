import json
import os
import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from models.holding import HoldingView
from models.trade import TradeResult


SELL_QUANTITY_EPSILON = 1e-6

# DATABASE_URL instellen → PostgreSQL (Supabase). Niet instellen → lokale SQLite.
_DATABASE_URL = os.getenv("DATABASE_URL", "")


class _PgCursor:
    """Maakt psycopg2-cursor compatibel met sqlite3 (fetchone/fetchall)."""

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> Any:
        return self._cur.fetchall()


class _PgConn:
    """Maakt psycopg2-verbinding compatibel met sqlite3 conn.execute()-interface."""

    def __init__(self, dsn: str) -> None:
        import psycopg2  # noqa: PLC0415
        import psycopg2.extras  # noqa: PLC0415

        self._conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)

    def execute(self, sql: str, params: tuple = ()) -> _PgCursor:
        # Vervang SQLite '?' placeholders door PostgreSQL '%s'
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
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "trading.sqlite3",
        )
        self._db_path = os.getenv("TRADING_DB_PATH", default_path)
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._configured_start_balance = float(os.getenv("START_BALANCE", "2000"))
        self._init_db()

    def get_trade_history(self) -> List[TradeResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, action, amount, quantity, price, profit_loss, timestamp, status,
                       expected_return_pct, risk_pct
                FROM trade_history
                ORDER BY timestamp ASC
                """
            ).fetchall()
        return [self._trade_from_row(row) for row in rows]

    def add_trade(self, trade: TradeResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_history (
                    id, symbol, action, amount, quantity, price, profit_loss, timestamp, status,
                    expected_return_pct, risk_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    symbol = excluded.symbol,
                    action = excluded.action,
                    amount = excluded.amount,
                    quantity = excluded.quantity,
                    price = excluded.price,
                    profit_loss = excluded.profit_loss,
                    timestamp = excluded.timestamp,
                    status = excluded.status,
                    expected_return_pct = excluded.expected_return_pct,
                    risk_pct = excluded.risk_pct
                """,
                (
                    trade.id,
                    trade.symbol,
                    trade.action,
                    trade.amount,
                    trade.quantity,
                    trade.price,
                    trade.profit_loss,
                    trade.timestamp,
                    trade.status,
                    trade.expected_return_pct,
                    trade.risk_pct,
                ),
            )
            conn.commit()

    def get_holdings(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at
                FROM holdings
                ORDER BY symbol ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_holding(self, symbol: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at
                FROM holdings
                WHERE symbol = ?
                """,
                (symbol.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def get_cash_balance(self) -> float:
        return float(self._get_setting("cash_balance", str(self._configured_start_balance)))

    def get_last_recommendation(self, symbol: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT recommendation FROM holding_alert_states WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchone()
        return str(row["recommendation"]) if row else None

    def set_last_recommendation(self, symbol: str, recommendation: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO holding_alert_states(symbol, recommendation, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    recommendation = excluded.recommendation,
                    updated_at = excluded.updated_at
                """,
                (symbol.upper(), recommendation, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def clear_recommendation_state(self, symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM holding_alert_states WHERE symbol = ?", (symbol.upper(),))
            conn.commit()

    def record_buy(
        self,
        *,
        symbol: str,
        market: str,
        quantity: float,
        amount: float,
        price: float,
    ) -> Dict[str, Any]:
        normalized_symbol = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()
        current_cash = self.get_cash_balance()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at
                FROM holdings
                WHERE symbol = ?
                """,
                (normalized_symbol,),
            ).fetchone()

            if existing:
                total_quantity = float(existing["quantity"]) + quantity
                total_invested = float(existing["invested_amount"]) + amount
                avg_entry_price = total_invested / total_quantity
                conn.execute(
                    """
                    UPDATE holdings
                    SET quantity = ?, invested_amount = ?, avg_entry_price = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    (total_quantity, total_invested, avg_entry_price, now, normalized_symbol),
                )
                holding = {
                    "symbol": normalized_symbol,
                    "market": str(existing["market"]),
                    "quantity": total_quantity,
                    "invested_amount": total_invested,
                    "avg_entry_price": avg_entry_price,
                    "opened_at": str(existing["opened_at"]),
                    "updated_at": now,
                }
            else:
                conn.execute(
                    """
                    INSERT INTO holdings(symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (normalized_symbol, market, quantity, amount, price, now, now),
                )
                holding = {
                    "symbol": normalized_symbol,
                    "market": market,
                    "quantity": quantity,
                    "invested_amount": amount,
                    "avg_entry_price": price,
                    "opened_at": now,
                    "updated_at": now,
                }

            conn.execute(
                "INSERT INTO settings(key, value) VALUES ('cash_balance', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(current_cash - amount),),
            )
            conn.commit()

        return holding

    def record_sell(self, *, symbol: str, quantity: float, price: float) -> Dict[str, float | str]:
        normalized_symbol = symbol.upper()
        current_cash = self.get_cash_balance()
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            holding = conn.execute(
                """
                SELECT symbol, market, quantity, invested_amount, avg_entry_price, opened_at, updated_at
                FROM holdings
                WHERE symbol = ?
                """,
                (normalized_symbol,),
            ).fetchone()
            if not holding:
                raise ValueError(f"Geen positie gevonden voor {normalized_symbol}")

            current_quantity = float(holding["quantity"])
            if abs(quantity - current_quantity) <= SELL_QUANTITY_EPSILON:
                quantity = current_quantity

            if quantity <= 0 or quantity - current_quantity > SELL_QUANTITY_EPSILON:
                raise ValueError("Ongeldige verkoophoeveelheid")

            avg_entry_price = float(holding["avg_entry_price"])
            cost_basis = avg_entry_price * quantity
            proceeds = price * quantity
            realized_profit_loss = proceeds - cost_basis
            remaining_quantity = current_quantity - quantity

            if remaining_quantity <= SELL_QUANTITY_EPSILON:
                conn.execute("DELETE FROM holdings WHERE symbol = ?", (normalized_symbol,))
                conn.execute("DELETE FROM holding_alert_states WHERE symbol = ?", (normalized_symbol,))
                status = "executed: positie gesloten"
            else:
                conn.execute(
                    """
                    UPDATE holdings
                    SET quantity = ?, invested_amount = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    (remaining_quantity, avg_entry_price * remaining_quantity, now, normalized_symbol),
                )
                status = "executed: gedeeltelijk verkocht"

            conn.execute(
                "INSERT INTO settings(key, value) VALUES ('cash_balance', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(current_cash + proceeds),),
            )
            conn.commit()

        return {
            "quantity": quantity,
            "proceeds": proceeds,
            "profit_loss": realized_profit_loss,
            "status": status,
        }

    def get_daily_trade_count(self) -> int:
        today = date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM trade_history WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()
        return int(row["count"]) if row else 0

    def get_daily_realized_pnl(self) -> float:
        """Bereken de gerealiseerde winst/verlies van vandaag (alleen verkopen)."""
        today = date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(profit_loss), 0) AS total FROM trade_history WHERE timestamp LIKE ? AND action = 'sell'",
                (f"{today}%",),
            ).fetchone()
        return float(row["total"]) if row else 0.0

    def get_portfolio(self, holdings: List[HoldingView] | None = None) -> dict:
        holdings = holdings or []
        invested_value = sum(h.invested_amount for h in holdings)
        market_value = sum(h.market_value for h in holdings)
        total_equity = self.get_cash_balance() + market_value
        start_balance = float(self._get_setting("start_balance", str(self._configured_start_balance)))

        # Sla dagelijks een snapshot op
        self._record_portfolio_snapshot(total_equity)

        return {
            "start_balance": round(start_balance, 2),
            "current_balance": round(total_equity, 2),
            "available_cash": round(self.get_cash_balance(), 2),
            "invested_value": round(invested_value, 2),
            "market_value": round(market_value, 2),
            "holdings_count": len(holdings),
            "total_profit_loss": round(total_equity - start_balance, 2),
            "daily_trade_count": self.get_daily_trade_count(),
            "portfolio_history": self.get_portfolio_history(),
        }

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, balance FROM portfolio_history ORDER BY date ASC"
            ).fetchall()
        return [{"date": row["date"], "balance": round(float(row["balance"]), 2)} for row in rows]

    def _record_portfolio_snapshot(self, balance: float) -> None:
        today = date.today().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_history(date, balance)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET balance = excluded.balance
                """,
                (today, balance),
            )
            conn.commit()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    amount REAL NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected_return_pct REAL NOT NULL,
                    risk_pct REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY,
                    market TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    invested_amount REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS holding_alert_states (
                    symbol TEXT PRIMARY KEY,
                    recommendation TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    date TEXT PRIMARY KEY,
                    balance REAL NOT NULL
                )
                """
            )

            existing_start = conn.execute("SELECT value FROM settings WHERE key = 'start_balance'").fetchone()
            existing_cash = conn.execute("SELECT value FROM settings WHERE key = 'cash_balance'").fetchone()
            if not existing_start:
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES ('start_balance', ?)",
                    (str(self._configured_start_balance),),
                )
            if not existing_cash:
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES ('cash_balance', ?)",
                    (str(self._configured_start_balance),),
                )
            conn.commit()

    def _get_setting(self, key: str, default: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def _set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

    def _connect(self) -> Any:
        if _DATABASE_URL:
            return _PgConn(_DATABASE_URL)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _trade_from_row(self, row: Any) -> TradeResult:
        return TradeResult(
            id=str(row["id"]),
            symbol=str(row["symbol"]),
            action=str(row["action"]),
            amount=float(row["amount"]),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            profit_loss=float(row["profit_loss"]),
            timestamp=str(row["timestamp"]),
            status=str(row["status"]),
            expected_return_pct=float(row["expected_return_pct"]),
            risk_pct=float(row["risk_pct"]),
        )
