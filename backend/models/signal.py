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
    target_price: float = 0.0
    expected_profit: float = 0.0  # verwachte winst bij investering van €1000
    expected_days: int = 0  # geschatte dagen tot koersdoel
    ranking_score: float = 0.0
    rank_label: str = ""
    reason: str
    strategy: str = "daytrade"  # "daytrade" or "longterm"
