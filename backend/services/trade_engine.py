import random
import uuid
from datetime import datetime, timezone

from models.trade import TradeRequest, TradeResult
from services.data_service import DataService


def execute_trade(
    request: TradeRequest, limits: dict, data_service: DataService
) -> TradeResult:
    """
    Voert een paper trade uit op basis van het gegenereerde advies.
    Gebruikt expected_return_pct en risk_pct als input voor P/L-simulatie.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Limiet: max trades per dag
    if data_service.get_daily_trade_count() >= limits["max_trades_per_day"]:
        return TradeResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            action=request.action,
            amount=request.amount,
            profit_loss=0.0,
            timestamp=now,
            status="rejected: dagelijks maximum bereikt",
            expected_return_pct=request.expected_return_pct,
            risk_pct=request.risk_pct,
        )

    # Limiet: max positiegrootte
    if request.amount > limits["max_position"]:
        return TradeResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            action=request.action,
            amount=request.amount,
            profit_loss=0.0,
            timestamp=now,
            status="rejected: positie te groot",
            expected_return_pct=request.expected_return_pct,
            risk_pct=request.risk_pct,
        )

    # Simuleer gerealiseerd rendement rond het verwachte rendement.
    expected = max(0.0, request.expected_return_pct)
    risk_pct = max(0.1, request.risk_pct)
    edge_pct = expected
    realized_pct = (edge_pct * random.uniform(0.5, 1.35)) - (risk_pct * random.uniform(0.15, 0.75))

    # Kleine kans op afwijkende uitkomst om marktruis te simuleren.
    if random.random() < 0.18:
        realized_pct = -abs(realized_pct) * random.uniform(0.4, 1.1)

    profit_loss = round(request.amount * (realized_pct / 100.0), 2)

    result = TradeResult(
        id=str(uuid.uuid4()),
        symbol=request.symbol,
        action=request.action,
        amount=request.amount,
        profit_loss=profit_loss,
        timestamp=now,
        status="executed",
        expected_return_pct=request.expected_return_pct,
        risk_pct=request.risk_pct,
    )
    data_service.add_trade(result)
    return result
