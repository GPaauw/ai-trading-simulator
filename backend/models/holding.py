from pydantic import BaseModel


class HoldingView(BaseModel):
    symbol: str
    market: str
    quantity: float
    invested_amount: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: str
    updated_at: str
