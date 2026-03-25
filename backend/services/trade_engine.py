import random
import uuid
from datetime import datetime, timezone

from models.trade import TradeRequest, TradeResult
from services.data_service import DataService


def execute_trade(
    request: TradeRequest, limits: dict, data_service: DataService
) -> TradeResult:
    """
    Voert een gesimuleerde trade uit en controleert de ingestelde limieten.
    Genereert een willekeurige winst/verlies (dummy).
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
        )

    # Gesimuleerde winst/verlies op basis van max_risk
    max_loss = request.amount * limits["max_risk"]
    profit_loss = round(random.uniform(-max_loss, max_loss * 3.0), 2)

    result = TradeResult(
        id=str(uuid.uuid4()),
        symbol=request.symbol,
        action=request.action,
        amount=request.amount,
        profit_loss=profit_loss,
        timestamp=now,
        status="executed",
    )
    data_service.add_trade(result)
    return result
