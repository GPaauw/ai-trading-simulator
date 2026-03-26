from pydantic import BaseModel


class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" of "sell"
    amount: float
    expected_return_pct: float = 0.0
    risk_pct: float = 0.0


class TradeResult(BaseModel):
    id: str
    symbol: str
    action: str
    amount: float
    profit_loss: float
    timestamp: str
    status: str
    expected_return_pct: float = 0.0
    risk_pct: float = 0.0
