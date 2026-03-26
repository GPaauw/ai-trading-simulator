import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from models.holding import HoldingView
from models.trade import TradeResult


class DataService:
    def __init__(self) -> None:
        self._trade_history: List[TradeResult] = []
        self._holdings: Dict[str, Dict[str, Any]] = {}
        self._start_balance = float(os.getenv("START_BALANCE", "2000"))
        self._cash_balance = self._start_balance

    def get_trade_history(self) -> List[TradeResult]:
        return self._trade_history

    def add_trade(self, trade: TradeResult) -> None:
        self._trade_history.append(trade)

    def get_holdings(self) -> List[Dict[str, Any]]:
        return list(self._holdings.values())

    def get_holding(self, symbol: str) -> Dict[str, Any] | None:
        return self._holdings.get(symbol.upper())

    def get_cash_balance(self) -> float:
        return self._cash_balance

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
        existing = self._holdings.get(normalized_symbol)

        if existing:
            total_quantity = float(existing["quantity"]) + quantity
            total_invested = float(existing["invested_amount"]) + amount
            existing["quantity"] = total_quantity
            existing["invested_amount"] = total_invested
            existing["avg_entry_price"] = total_invested / total_quantity
            existing["updated_at"] = now
            holding = existing
        else:
            holding = {
                "symbol": normalized_symbol,
                "market": market,
                "quantity": quantity,
                "invested_amount": amount,
                "avg_entry_price": price,
                "opened_at": now,
                "updated_at": now,
            }
            self._holdings[normalized_symbol] = holding

        self._cash_balance -= amount
        return holding

    def record_sell(self, *, symbol: str, quantity: float, price: float) -> Dict[str, float | str]:
        normalized_symbol = symbol.upper()
        holding = self._holdings.get(normalized_symbol)
        if not holding:
            raise ValueError(f"Geen positie gevonden voor {normalized_symbol}")

        current_quantity = float(holding["quantity"])
        if quantity <= 0 or quantity > current_quantity:
            raise ValueError("Ongeldige verkoophoeveelheid")

        avg_entry_price = float(holding["avg_entry_price"])
        cost_basis = avg_entry_price * quantity
        proceeds = price * quantity
        realized_profit_loss = proceeds - cost_basis
        remaining_quantity = current_quantity - quantity
        now = datetime.now(timezone.utc).isoformat()

        if remaining_quantity <= 1e-8:
            self._holdings.pop(normalized_symbol, None)
            status = "executed: positie gesloten"
        else:
            holding["quantity"] = remaining_quantity
            holding["invested_amount"] = avg_entry_price * remaining_quantity
            holding["updated_at"] = now
            status = "executed: gedeeltelijk verkocht"

        self._cash_balance += proceeds
        return {
            "quantity": quantity,
            "proceeds": proceeds,
            "profit_loss": realized_profit_loss,
            "status": status,
        }

    def get_daily_trade_count(self) -> int:
        today = date.today().isoformat()
        return sum(1 for t in self._trade_history if t.timestamp.startswith(today))

    def get_portfolio(self, holdings: List[HoldingView] | None = None) -> dict:
        holdings = holdings or []
        invested_value = sum(h.invested_amount for h in holdings)
        market_value = sum(h.market_value for h in holdings)
        total_equity = self._cash_balance + market_value
        return {
            "start_balance": round(self._start_balance, 2),
            "current_balance": round(total_equity, 2),
            "available_cash": round(self._cash_balance, 2),
            "invested_value": round(invested_value, 2),
            "market_value": round(market_value, 2),
            "holdings_count": len(holdings),
            "total_profit_loss": round(total_equity - self._start_balance, 2),
            "daily_trade_count": self.get_daily_trade_count(),
        }
