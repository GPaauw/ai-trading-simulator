from pydantic import BaseModel


class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" of "sell"
    amount: float
    market: str | None = None
    quantity: float | None = None
    price: float | None = None
    expected_return_pct: float = 0.0
    risk_pct: float = 0.0


class TradeResult(BaseModel):
    id: str
    symbol: str
    action: str
    amount: float
    quantity: float = 0.0
    price: float = 0.0
    profit_loss: float
    timestamp: str
    status: str
    expected_return_pct: float = 0.0
    risk_pct: float = 0.0
    confidence: float = 0.0
    model_version: str = ""
