from pydantic import BaseModel


class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" of "sell"
    amount: float


class TradeResult(BaseModel):
    id: str
    symbol: str
    action: str
    amount: float
    profit_loss: float
    timestamp: str
    status: str
