from pydantic import BaseModel


class HoldingView(BaseModel):
    symbol: str
    market: str
    quantity: float
    invested_amount: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_profit_loss: float
    unrealized_profit_loss_pct: float
    recommendation: str
    recommendation_label: str
    recommendation_reason: str
    suggested_action: str
    suggested_sell_fraction: float
    target_price: float
    stop_loss_price: float
    confidence: float
    expected_return_pct: float
    risk_pct: float
    position_score: float
    opened_at: str
    updated_at: str