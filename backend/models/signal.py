from pydantic import BaseModel


class Signal(BaseModel):
    id: str
    symbol: str
    market: str  # "us", "eu" of "crypto"
    action: str  # "buy", "sell" of "hold"
    confidence: float
    price: float
    risk_pct: float
    expected_return_pct: float
    reason: str
