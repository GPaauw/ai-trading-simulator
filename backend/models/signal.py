from pydantic import BaseModel


class Signal(BaseModel):
    id: str
    symbol: str
    action: str  # "buy" of "sell"
    confidence: float
    price: float
