from pydantic import BaseModel
from typing import Dict


class Signal(BaseModel):
    symbol: str
    market: str = "us"
    action: str  # "buy", "sell", "hold"
    action_detail: str = ""
    confidence: float
    price: float = 0.0
    expected_return_pct: float = 0.0
    risk_score: float = 0.0
    horizon_days: int = 1
    ranking_score: float = 0.0
    probabilities: Dict[str, float] = {}
    reason: str = ""
