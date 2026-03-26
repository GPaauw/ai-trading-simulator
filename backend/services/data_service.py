import os
from datetime import date
from typing import List

from models.trade import TradeResult


class DataService:
    def __init__(self) -> None:
        self._trade_history: List[TradeResult] = []
        self._start_balance = float(os.getenv("START_BALANCE", "2000"))
        self._balance = self._start_balance

    def get_trade_history(self) -> List[TradeResult]:
        return self._trade_history

    def add_trade(self, trade: TradeResult) -> None:
        self._trade_history.append(trade)
        self._balance += trade.profit_loss

    def get_daily_trade_count(self) -> int:
        today = date.today().isoformat()
        return sum(1 for t in self._trade_history if t.timestamp.startswith(today))

    def get_portfolio(self) -> dict:
        return {
            "start_balance": round(self._start_balance, 2),
            "current_balance": round(self._balance, 2),
            "total_profit_loss": round(self._balance - self._start_balance, 2),
            "daily_trade_count": self.get_daily_trade_count(),
        }
