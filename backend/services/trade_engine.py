"""Trade Engine: voert trades uit met transactiekosten en slippage simulatie."""

import uuid
from datetime import datetime, timezone

from ml.config import TRANSACTION_COST_PCT, SLIPPAGE_PCT
from models.trade import TradeRequest, TradeResult
from services.data_service import DataService
from services.market_data_service import MarketDataService


def execute_trade(
    request: TradeRequest,
    limits: dict,
    data_service: DataService,
    market_data_service: MarketDataService,
    confidence: float = 0.0,
    model_version: str = "",
) -> TradeResult:
    now = datetime.now(timezone.utc).isoformat()

    if data_service.get_daily_trade_count() >= limits["max_trades_per_day"]:
        return _reject(request, now, "rejected: dagelijks maximum bereikt")

    try:
        instrument = market_data_service.get_instrument(request.symbol, request.market)
    except ValueError as exc:
        return _reject(request, now, f"rejected: {exc}")

    live_price = request.price
    if live_price is None:
        try:
            live_price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            return _reject(request, now, "rejected: live prijs niet beschikbaar")

    # Slippage simulatie
    if request.action == "buy":
        live_price *= (1 + SLIPPAGE_PCT)
    elif request.action == "sell":
        live_price *= (1 - SLIPPAGE_PCT)

    if request.action == "buy":
        quantity, amount = _resolve_qty(request.amount, request.quantity, live_price)
        # Transactiekosten
        cost = amount * TRANSACTION_COST_PCT
        total_cost = amount + cost

        if quantity <= 0 or amount <= 0:
            return _reject(request, now, "rejected: ongeldig bedrag", price=live_price)
        if total_cost > data_service.get_cash_balance():
            return _reject(request, now, "rejected: onvoldoende cash", quantity=quantity, price=live_price)

        data_service.record_buy(
            symbol=request.symbol, market=instrument["market"],
            quantity=quantity, amount=total_cost, price=live_price,
        )
        result = TradeResult(
            id=str(uuid.uuid4()), symbol=request.symbol.upper(), action="buy",
            amount=round(total_cost, 2), quantity=round(quantity, 6), price=round(live_price, 4),
            profit_loss=round(-cost, 2), timestamp=now, status="executed: positie toegevoegd",
            expected_return_pct=request.expected_return_pct, risk_pct=request.risk_pct,
            confidence=confidence, model_version=model_version,
        )
        data_service.add_trade(result)
        return result

    if request.action == "sell":
        holding = data_service.get_holding(request.symbol)
        if not holding:
            return _reject(request, now, "rejected: geen positie", price=live_price)

        sell_qty = request.quantity if request.quantity else float(holding["quantity"])
        try:
            sale = data_service.record_sell(symbol=request.symbol, quantity=sell_qty, price=live_price)
        except ValueError as exc:
            return _reject(request, now, f"rejected: {exc}", quantity=sell_qty, price=live_price)

        proceeds = float(sale["proceeds"])
        cost = proceeds * TRANSACTION_COST_PCT
        net_pl = float(sale["profit_loss"]) - cost

        result = TradeResult(
            id=str(uuid.uuid4()), symbol=request.symbol.upper(), action="sell",
            amount=round(proceeds - cost, 2), quantity=round(float(sale["quantity"]), 6),
            price=round(live_price, 4), profit_loss=round(net_pl, 2),
            timestamp=now, status=str(sale["status"]),
            expected_return_pct=request.expected_return_pct, risk_pct=request.risk_pct,
            confidence=confidence, model_version=model_version,
        )
        data_service.add_trade(result)
        return result

    return _reject(request, now, "rejected: onbekende actie", price=live_price)


def _resolve_qty(amount: float, quantity: float | None, price: float) -> tuple[float, float]:
    if quantity is not None and quantity > 0:
        return quantity, amount if amount > 0 else quantity * price
    if amount <= 0 or price <= 0:
        return 0.0, 0.0
    return amount / price, amount


def _reject(request: TradeRequest, ts: str, status: str, *, quantity: float = 0, price: float = 0) -> TradeResult:
    return TradeResult(
        id=str(uuid.uuid4()), symbol=request.symbol.upper(), action=request.action,
        amount=round(request.amount, 2), quantity=round(quantity, 6), price=round(price, 4),
        profit_loss=0.0, timestamp=ts, status=status,
        expected_return_pct=request.expected_return_pct, risk_pct=request.risk_pct,
    )
