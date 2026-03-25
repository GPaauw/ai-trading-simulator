import random
import uuid
from datetime import date
from typing import List

from models.signal import Signal
from models.trade import TradeResult


class DataService:
    def __init__(self) -> None:
        self._trade_history: List[TradeResult] = []
        self._symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC", "ETH"]

    def get_signals(self) -> List[Signal]:
        signals = []
        for symbol in self._symbols[:5]:
            signals.append(
                Signal(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    action=random.choice(["buy", "sell"]),
                    confidence=round(random.uniform(0.50, 0.99), 2),
                    price=round(random.uniform(100.0, 5000.0), 2),
                )
            )
        return signals

    def get_trade_history(self) -> List[TradeResult]:
        return self._trade_history

    def add_trade(self, trade: TradeResult) -> None:
        self._trade_history.append(trade)

    def get_daily_trade_count(self) -> int:
        today = date.today().isoformat()
        return sum(1 for t in self._trade_history if t.timestamp.startswith(today))
