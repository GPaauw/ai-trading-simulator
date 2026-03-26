import uuid
from datetime import datetime, timezone

from models.signal import Signal
from models.trade import TradeRequest, TradeResult
from services.data_service import DataService
from services.market_data_service import MarketDataService


def execute_trade(
    request: TradeRequest,
    limits: dict,
    data_service: DataService,
    market_data_service: MarketDataService,
    current_signal: Signal | None = None,
) -> TradeResult:
    now = datetime.now(timezone.utc).isoformat()

    if data_service.get_daily_trade_count() >= limits["max_trades_per_day"]:
        return _reject_trade(request, now, "rejected: dagelijks maximum bereikt")

    try:
        instrument = market_data_service.get_instrument(request.symbol, request.market)
    except ValueError as exc:
        return _reject_trade(request, now, f"rejected: {exc}")

    live_price = request.price or (current_signal.price if current_signal else None)
    if live_price is None:
        try:
            live_price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            return _reject_trade(request, now, "rejected: live prijs niet beschikbaar")

    expected_return_pct = request.expected_return_pct or (current_signal.expected_return_pct if current_signal else 0.0)
    risk_pct = request.risk_pct or (current_signal.risk_pct if current_signal else 1.0)

    if request.action == "buy":
        quantity, amount = _resolve_quantity_and_amount(request.amount, request.quantity, live_price)
        if quantity <= 0 or amount <= 0:
            return _reject_trade(request, now, "rejected: ongeldig bedrag of quantity", price=live_price)
        if amount > limits["max_position"]:
            return _reject_trade(request, now, "rejected: positie te groot", quantity=quantity, price=live_price)
        if amount > data_service.get_cash_balance():
            return _reject_trade(request, now, "rejected: onvoldoende cash", quantity=quantity, price=live_price)

        data_service.record_buy(
            symbol=request.symbol,
            market=instrument["market"],
            quantity=quantity,
            amount=amount,
            price=live_price,
        )
        result = TradeResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol.upper(),
            action=request.action,
            amount=round(amount, 2),
            quantity=round(quantity, 6),
            price=round(live_price, 4),
            profit_loss=0.0,
            timestamp=now,
            status="executed: positie toegevoegd",
            expected_return_pct=expected_return_pct,
            risk_pct=risk_pct,
        )
        data_service.add_trade(result)
        return result

    if request.action == "sell":
        holding = data_service.get_holding(request.symbol)
        if not holding:
            return _reject_trade(request, now, "rejected: geen positie om te verkopen", price=live_price)

        sell_quantity = request.quantity if request.quantity else float(holding["quantity"])
        try:
            sale = data_service.record_sell(symbol=request.symbol, quantity=sell_quantity, price=live_price)
        except ValueError as exc:
            return _reject_trade(request, now, f"rejected: {exc}", quantity=sell_quantity, price=live_price)

        result = TradeResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol.upper(),
            action=request.action,
            amount=round(float(sale["proceeds"]), 2),
            quantity=round(float(sale["quantity"]), 6),
            price=round(live_price, 4),
            profit_loss=round(float(sale["profit_loss"]), 2),
            timestamp=now,
            status=str(sale["status"]),
            expected_return_pct=expected_return_pct,
            risk_pct=risk_pct,
        )
        data_service.add_trade(result)
        return result

    return _reject_trade(
        request,
        now,
        "rejected: unsupported actie",
        price=live_price,
    )


def _resolve_quantity_and_amount(amount: float, quantity: float | None, price: float) -> tuple[float, float]:
    if quantity is not None and quantity > 0:
        resolved_quantity = quantity
        resolved_amount = amount if amount > 0 else quantity * price
        return resolved_quantity, resolved_amount
    if amount <= 0 or price <= 0:
        return 0.0, 0.0
    return amount / price, amount


def _reject_trade(
    request: TradeRequest,
    timestamp: str,
    status: str,
    *,
    quantity: float = 0.0,
    price: float = 0.0,
) -> TradeResult:
    return TradeResult(
        id=str(uuid.uuid4()),
        symbol=request.symbol.upper(),
        action=request.action,
        amount=round(request.amount, 2),
        quantity=round(quantity, 6),
        price=round(price, 4),
        profit_loss=0.0,
        timestamp=timestamp,
        status=status,
        expected_return_pct=request.expected_return_pct,
        risk_pct=request.risk_pct,
    )
