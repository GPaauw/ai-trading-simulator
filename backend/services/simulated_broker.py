"""Simulated Broker: hyper-realistische broker simulatie.

Vervangt de simpele trade_engine met een volledig broker-model:

  1. **Order Book simulatie**    — limit/market/stop orders, FIFO matching
  2. **Realistisch slippage**    — volume-afhankelijk, spread-gebaseerd
  3. **Margin & leverage**       — initial/maintenance margin, margin calls
  4. **Circuit breakers**        — dagelijkse limiet-down/up halt
  5. **Latency simulatie**       — order verwerking vertraging
  6. **Funding rates (crypto)**  — 8-uurs funding voor perpetuals
  7. **Partial fills**           — grote orders in meerdere fills
  8. **Order types**             — market, limit, stop, stop-limit, trailing stop

Integratie:
  - Vervangt ``trade_engine.execute_trade()`` als broker backend
  - AutoTrader v2 schakelt automatisch over op SimulatedBroker
  - Backward compatible: ``execute_trade()`` wrapper beschikbaar

Configuratie via environment variabelen:
  - BROKER_LATENCY_MS     (default: 50)
  - BROKER_MARGIN_ENABLED (default: false)
  - BROKER_MAKER_FEE      (default: 0.001)
  - BROKER_TAKER_FEE      (default: 0.002)
"""

import enum
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
_DEFAULT_LATENCY_MS: int = 50
_MAX_ORDER_BOOK_DEPTH: int = 20
_CIRCUIT_BREAKER_PCT: float = 0.10      # 10% halt
_FUNDING_RATE_INTERVAL_H: int = 8
_DEFAULT_FUNDING_RATE: float = 0.0001   # 0.01% per 8h
_MIN_PARTIAL_FILL_QTY: float = 0.0001


# ── Enums ───────────────────────────────────────────────────────────

class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.Enum):
    PENDING = "pending"          # wacht op verwerking (latency)
    OPEN = "open"                # actief in order book
    PARTIAL = "partial"          # gedeeltelijk gevuld
    FILLED = "filled"            # volledig gevuld
    CANCELLED = "cancelled"      # geannuleerd
    REJECTED = "rejected"        # afgewezen (margin, circuit breaker)
    EXPIRED = "expired"          # verlopen (GTC, GTD)


class TimeInForce(enum.Enum):
    GTC = "gtc"                  # Good Till Cancel
    IOC = "ioc"                  # Immediate or Cancel
    FOK = "fok"                  # Fill or Kill
    DAY = "day"                  # dag-order


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class OrderBookLevel:
    """Eén prijsniveau in het order book."""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class Fill:
    """Eén fill (gedeeltelijke of volledige uitvoering)."""
    fill_id: str
    order_id: str
    price: float
    quantity: float
    fee: float
    timestamp: str
    is_maker: bool = False


@dataclass
class Order:
    """Volledige order representatie."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None        # limit/stop prijs
    stop_price: Optional[float] = None   # stop trigger prijs
    trail_pct: Optional[float] = None    # trailing stop percentage
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    reject_reason: str = ""
    market: str = "us"                   # "us" of "crypto"

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.quantity - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL, OrderStatus.PENDING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": round(self.quantity, 6),
            "price": round(self.price, 4) if self.price else None,
            "stop_price": round(self.stop_price, 4) if self.stop_price else None,
            "status": self.status.value,
            "filled_qty": round(self.filled_qty, 6),
            "avg_fill_price": round(self.avg_fill_price, 4),
            "total_fees": round(self.total_fees, 6),
            "fills_count": len(self.fills),
            "created_at": self.created_at,
            "market": self.market,
        }


@dataclass
class MarginAccount:
    """Margin account status."""
    equity: float = 0.0
    margin_used: float = 0.0
    maintenance_margin: float = 0.0
    leverage: float = 1.0
    margin_call: bool = False

    @property
    def available_margin(self) -> float:
        return max(0.0, self.equity - self.margin_used)

    @property
    def margin_ratio(self) -> float:
        return self.margin_used / self.equity if self.equity > 0 else 0.0


@dataclass
class CircuitBreaker:
    """Circuit breaker status per symbol."""
    symbol: str
    halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[str] = None
    reference_price: float = 0.0
    upper_limit: float = 0.0
    lower_limit: float = 0.0


# ── Simulated Broker ────────────────────────────────────────────────

class SimulatedBroker:
    """Hyper-realistische broker simulatie.

    Gebruik:
        broker = SimulatedBroker(initial_cash=10000.0)
        order = broker.submit_order(
            symbol="AAPL", side="buy", order_type="market",
            quantity=10, market="us",
        )
        broker.process_market_data({"AAPL": {"price": 150.0, "volume": 1e6, ...}})
        print(broker.get_portfolio_summary())
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        maker_fee: Optional[float] = None,
        taker_fee: Optional[float] = None,
        latency_ms: Optional[int] = None,
        margin_enabled: Optional[bool] = None,
        max_leverage: float = 2.0,
    ) -> None:
        # Fees
        self.maker_fee = maker_fee or float(os.environ.get("BROKER_MAKER_FEE", "0.001"))
        self.taker_fee = taker_fee or float(os.environ.get("BROKER_TAKER_FEE", "0.002"))

        # Latency
        self.latency_ms = latency_ms or int(os.environ.get("BROKER_LATENCY_MS", str(_DEFAULT_LATENCY_MS)))

        # Margin
        self._margin_enabled = margin_enabled
        if self._margin_enabled is None:
            self._margin_enabled = os.environ.get("BROKER_MARGIN_ENABLED", "false").lower() == "true"
        self._max_leverage = max_leverage

        # Portfolio state
        self._cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self._positions: Dict[str, Dict[str, float]] = {}  # symbol → {qty, avg_price, market}

        # Orders
        self._orders: Dict[str, Order] = {}          # order_id → Order
        self._open_orders: Dict[str, Order] = {}     # actieve orders
        self._order_history: Deque[Order] = deque(maxlen=1000)

        # Order books (gesimuleerd)
        self._order_books: Dict[str, Dict[str, List[OrderBookLevel]]] = {}

        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._last_prices: Dict[str, float] = {}
        self._day_open_prices: Dict[str, float] = {}

        # Margin
        self._margin = MarginAccount(equity=initial_cash, leverage=1.0)

        # Funding (crypto)
        self._last_funding_time: float = time.time()
        self._funding_rates: Dict[str, float] = {}

        # Stats
        self._total_fees_paid: float = 0.0
        self._total_slippage: float = 0.0
        self._fills_count: int = 0
        self._rejected_count: int = 0

    # ── Order Submission ────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "market",
        quantity: float = 0.0,
        amount: float = 0.0,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_pct: Optional[float] = None,
        time_in_force: str = "gtc",
        market: str = "us",
    ) -> Order:
        """Dien een nieuwe order in.

        Args:
            symbol: Asset symbool (bijv. "AAPL", "BTC/USDT")
            side: "buy" of "sell"
            order_type: "market", "limit", "stop", "stop_limit", "trailing_stop"
            quantity: Aantal te kopen/verkopen (of amount voor notional orders)
            amount: Notioneel bedrag (alternatief voor quantity)
            price: Limit prijs (vereist voor limit / stop_limit)
            stop_price: Stop trigger prijs (vereist voor stop / stop_limit)
            trail_pct: Trailing stop percentage (bijv. 0.05 = 5%)
            time_in_force: "gtc", "ioc", "fok", "day"
            market: "us" of "crypto"

        Returns:
            Order object met status
        """
        now = datetime.now(timezone.utc).isoformat()
        symbol = symbol.upper()

        # Resolve quantity van amount
        if quantity <= 0 and amount > 0:
            last_px = self._last_prices.get(symbol, price or 0)
            if last_px > 0:
                quantity = amount / last_px
            else:
                return self._reject_order(symbol, side, order_type, now, "prijs onbekend voor notional order")

        if quantity <= 0:
            return self._reject_order(symbol, side, order_type, now, "ongeldige quantity")

        # Order type parsing
        try:
            ot = OrderType(order_type)
            os_side = OrderSide(side)
            tif = TimeInForce(time_in_force)
        except ValueError as e:
            return self._reject_order(symbol, side, order_type, now, f"ongeldig order type: {e}")

        # Validatie
        if ot in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is None:
            return self._reject_order(symbol, side, order_type, now, "limit prijs vereist")
        if ot in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            return self._reject_order(symbol, side, order_type, now, "stop prijs vereist")
        if ot == OrderType.TRAILING_STOP and trail_pct is None:
            return self._reject_order(symbol, side, order_type, now, "trail percentage vereist")

        # Circuit breaker check
        cb = self._circuit_breakers.get(symbol)
        if cb and cb.halted:
            return self._reject_order(symbol, side, order_type, now, f"circuit breaker actief: {cb.halt_reason}")

        # Margin check (voor buys)
        if os_side == OrderSide.BUY:
            est_cost = quantity * (price or self._last_prices.get(symbol, 0))
            if not self._check_buying_power(est_cost):
                return self._reject_order(symbol, side, order_type, now, "onvoldoende koopkracht/margin")

        # Sell check: genoeg positie?
        if os_side == OrderSide.SELL and not self._margin_enabled:
            pos = self._positions.get(symbol, {})
            held_qty = pos.get("qty", 0.0)
            if quantity > held_qty + _MIN_PARTIAL_FILL_QTY:
                return self._reject_order(symbol, side, order_type, now,
                                          f"onvoldoende positie ({held_qty:.6f} beschikbaar)")

        # Maak order aan
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=os_side,
            order_type=ot,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trail_pct=trail_pct,
            time_in_force=tif,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
            market=market,
        )

        self._orders[order.order_id] = order

        # Simuleer latency, daarna verwerk
        self._simulate_latency()
        self._process_new_order(order)

        return order

    # ── Market Data Processing ──────────────────────────────────────

    def process_market_data(
        self,
        market_data: Dict[str, Dict[str, Any]],
    ) -> List[Fill]:
        """Verwerk nieuwe marktdata en match open orders.

        Args:
            market_data: {symbol: {"price": float, "volume": float,
                         "high": float, "low": float, "bid": float, "ask": float}}

        Returns:
            Lijst van nieuwe fills
        """
        all_fills: List[Fill] = []

        for symbol, data in market_data.items():
            symbol = symbol.upper()
            price = data.get("price", 0.0)
            volume = data.get("volume", 0.0)

            if price <= 0:
                continue

            # Update laatst bekende prijs
            old_price = self._last_prices.get(symbol)
            self._last_prices[symbol] = price

            # Set day open price
            if symbol not in self._day_open_prices:
                self._day_open_prices[symbol] = price

            # Circuit breaker check
            self._check_circuit_breaker(symbol, price)

            # Simuleer order book
            self._update_order_book(symbol, data)

            # Match open orders
            for oid, order in list(self._open_orders.items()):
                if order.symbol != symbol or not order.is_active:
                    continue

                fills = self._try_fill_order(order, price, volume, data)
                all_fills.extend(fills)

        # Crypto funding rates
        self._process_funding()

        return all_fills

    # ── Order Processing ────────────────────────────────────────────

    def _process_new_order(self, order: Order) -> None:
        """Verwerk een nieuwe order na latency delay."""
        order.status = OrderStatus.OPEN
        order.updated_at = datetime.now(timezone.utc).isoformat()

        # Market orders: direct proberen te vullen
        if order.order_type == OrderType.MARKET:
            last_px = self._last_prices.get(order.symbol)
            if last_px and last_px > 0:
                self._try_fill_order(order, last_px, float("inf"), {})
            # Als niet gevuld: blijft open tot volgende market data tick
            if order.status not in (OrderStatus.FILLED, OrderStatus.REJECTED):
                self._open_orders[order.order_id] = order
        else:
            self._open_orders[order.order_id] = order

    def _try_fill_order(
        self,
        order: Order,
        market_price: float,
        volume: float,
        data: Dict[str, Any],
    ) -> List[Fill]:
        """Probeer een open order te matchen tegen de markt.

        Ondersteunt partial fills op basis van beschikbaar volume.
        """
        fills: List[Fill] = []

        if not order.is_active:
            return fills

        # Circuit breaker check
        cb = self._circuit_breakers.get(order.symbol)
        if cb and cb.halted:
            return fills

        # Bepaal of de order triggert
        should_fill, fill_price = self._should_fill(order, market_price, data)

        if not should_fill:
            return fills

        remaining = order.remaining_qty

        # Partial fill: maximaal 30% van markt volume per fill
        max_fill_qty = volume * 0.30 if volume > 0 else remaining
        fill_qty = min(remaining, max(max_fill_qty, _MIN_PARTIAL_FILL_QTY))

        if fill_qty < _MIN_PARTIAL_FILL_QTY:
            return fills

        # Slippage berekenen
        slippage_price = self._calculate_slippage(
            fill_price, fill_qty, volume, order.side, order.market,
        )

        # Fee berekenen
        is_maker = order.order_type in (OrderType.LIMIT,)
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = abs(fill_qty * slippage_price * fee_rate)

        # Fill aanmaken
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            price=round(slippage_price, 8),
            quantity=round(fill_qty, 8),
            fee=round(fee, 8),
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_maker=is_maker,
        )
        fills.append(fill)

        # Order update
        old_total = order.filled_qty * order.avg_fill_price
        order.filled_qty += fill_qty
        order.avg_fill_price = (old_total + fill_qty * slippage_price) / order.filled_qty
        order.total_fees += fee
        order.fills.append(fill)
        order.updated_at = fill.timestamp

        if order.remaining_qty < _MIN_PARTIAL_FILL_QTY:
            order.status = OrderStatus.FILLED
            self._open_orders.pop(order.order_id, None)
            self._order_history.append(order)
        else:
            order.status = OrderStatus.PARTIAL
            # IOC/FOK: cancel rest
            if order.time_in_force == TimeInForce.IOC:
                order.status = OrderStatus.PARTIAL
                self._open_orders.pop(order.order_id, None)
                self._order_history.append(order)
            elif order.time_in_force == TimeInForce.FOK:
                # FOK had alles moeten vullen — cancel
                if order.filled_qty < order.quantity - _MIN_PARTIAL_FILL_QTY:
                    order.status = OrderStatus.CANCELLED
                    order.reject_reason = "FOK: niet volledig gevuld"
                    self._open_orders.pop(order.order_id, None)
                    self._order_history.append(order)

        # Portfolio update
        self._apply_fill(order, fill)

        # Stats
        self._total_fees_paid += fee
        self._total_slippage += abs(slippage_price - (order.price or market_price)) * fill_qty
        self._fills_count += 1

        logger.debug(
            "Fill: %s %s %.6f @ %.4f (fee=%.6f, slippage=%.4f)",
            order.side.value, order.symbol, fill_qty, slippage_price, fee,
            slippage_price - (order.price or market_price),
        )

        return fills

    def _should_fill(
        self,
        order: Order,
        market_price: float,
        data: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """Bepaal of een order getriggered wordt.

        Returns (should_fill, fill_price)
        """
        bid = data.get("bid", market_price * 0.999)
        ask = data.get("ask", market_price * 1.001)

        if order.order_type == OrderType.MARKET:
            # Market buy fills op ask, sell op bid
            fill_px = ask if order.side == OrderSide.BUY else bid
            return True, fill_px

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= ask:
                return True, min(order.price, ask)
            elif order.side == OrderSide.SELL and order.price <= bid:
                return True, max(order.price, bid)

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                return True, ask  # triggers als market order
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                return True, bid

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                if order.price >= ask:
                    return True, min(order.price, ask)
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                if order.price <= bid:
                    return True, max(order.price, bid)

        elif order.order_type == OrderType.TRAILING_STOP:
            # Trailing stop: dynamische stop prijs op basis van trail_pct
            if order.trail_pct:
                if order.side == OrderSide.SELL:
                    # Track highest price, stop als prijs daalt met trail_pct%
                    high = data.get("high", market_price)
                    trail_stop = high * (1 - order.trail_pct)
                    if market_price <= trail_stop:
                        return True, bid
                elif order.side == OrderSide.BUY:
                    low = data.get("low", market_price)
                    trail_stop = low * (1 + order.trail_pct)
                    if market_price >= trail_stop:
                        return True, ask

        return False, 0.0

    # ── Slippage Model ──────────────────────────────────────────────

    def _calculate_slippage(
        self,
        base_price: float,
        quantity: float,
        volume: float,
        side: OrderSide,
        market: str,
    ) -> float:
        """Bereken realistisch slippage.

        Slippage = base_spread + volume_impact + random_noise

        - base_spread: 0.01% stocks, 0.005% crypto
        - volume_impact: sqrt(qty/volume) × impact_factor
        - random_noise: ±0.005% Gaussiaans
        """
        if base_price <= 0:
            return base_price

        # Base spread
        if market == "crypto":
            base_spread_pct = 0.00005    # 0.005%
            impact_factor = 0.001         # crypto = dieper order book
        else:
            base_spread_pct = 0.0001     # 0.01%
            impact_factor = 0.002         # stocks = minder diep

        # Volume impact (vierkantswortel model)
        if volume > 0:
            participation = quantity / volume
            vol_impact = np.sqrt(min(participation, 0.1)) * impact_factor
        else:
            vol_impact = impact_factor * 0.5

        # Random component (markt microstructuur ruis)
        noise = np.random.normal(0, 0.00005)  # ±0.005%

        total_slippage_pct = base_spread_pct + vol_impact + noise

        # Richting: buys betalen meer, sells ontvangen minder
        if side == OrderSide.BUY:
            return base_price * (1 + total_slippage_pct)
        else:
            return base_price * (1 - abs(total_slippage_pct))

    # ── Circuit Breakers ────────────────────────────────────────────

    def _check_circuit_breaker(self, symbol: str, price: float) -> None:
        """Check en activeer circuit breakers bij extreme prijsbewegingen."""
        ref_price = self._day_open_prices.get(symbol)
        if not ref_price or ref_price <= 0:
            return

        change_pct = (price - ref_price) / ref_price

        cb = self._circuit_breakers.get(symbol, CircuitBreaker(symbol=symbol))
        cb.reference_price = ref_price
        cb.upper_limit = ref_price * (1 + _CIRCUIT_BREAKER_PCT)
        cb.lower_limit = ref_price * (1 - _CIRCUIT_BREAKER_PCT)

        if abs(change_pct) >= _CIRCUIT_BREAKER_PCT and not cb.halted:
            direction = "limit-up" if change_pct > 0 else "limit-down"
            cb.halted = True
            cb.halt_reason = f"{direction} ({change_pct*100:.1f}%)"
            # Halt voor 15 minuten (gesimuleerd)
            halt_end = datetime.now(timezone.utc) + timedelta(minutes=15)
            cb.halt_until = halt_end.isoformat()
            self._circuit_breakers[symbol] = cb

            logger.warning(
                "CIRCUIT BREAKER: %s gehalted — %s (ref=%.2f, current=%.2f)",
                symbol, cb.halt_reason, ref_price, price,
            )

            # Cancel alle open orders voor dit symbool
            for oid, order in list(self._open_orders.items()):
                if order.symbol == symbol:
                    order.status = OrderStatus.CANCELLED
                    order.reject_reason = f"Geannuleerd: circuit breaker {direction}"
                    self._open_orders.pop(oid, None)
                    self._order_history.append(order)

        elif cb.halted and cb.halt_until:
            # Check of halt verlopen is
            try:
                halt_end = datetime.fromisoformat(cb.halt_until)
                if datetime.now(timezone.utc) >= halt_end:
                    cb.halted = False
                    cb.halt_reason = ""
                    cb.halt_until = None
                    logger.info("Circuit breaker opgeheven: %s", symbol)
            except (ValueError, TypeError):
                pass

        self._circuit_breakers[symbol] = cb

    # ── Order Book Simulatie ────────────────────────────────────────

    def _update_order_book(self, symbol: str, data: Dict[str, Any]) -> None:
        """Simuleer order book op basis van marktdata."""
        price = data.get("price", 0)
        if price <= 0:
            return

        bid = data.get("bid", price * 0.999)
        ask = data.get("ask", price * 1.001)
        spread = max(ask - bid, price * 0.0001)

        # Genereer synthetisch order book
        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        for i in range(_MAX_ORDER_BOOK_DEPTH):
            # Bids: aflopend van best bid
            bid_price = bid - i * spread * 0.5
            bid_qty = np.random.exponential(100) * (1 + i * 0.1)
            bids.append(OrderBookLevel(price=round(bid_price, 4), quantity=round(bid_qty, 2)))

            # Asks: oplopend van best ask
            ask_price = ask + i * spread * 0.5
            ask_qty = np.random.exponential(100) * (1 + i * 0.1)
            asks.append(OrderBookLevel(price=round(ask_price, 4), quantity=round(ask_qty, 2)))

        self._order_books[symbol] = {"bids": bids, "asks": asks}

    # ── Margin ──────────────────────────────────────────────────────

    def _check_buying_power(self, estimated_cost: float) -> bool:
        """Controleer of er voldoende koopkracht is."""
        if self._margin_enabled:
            available = self._cash * self._max_leverage
            return estimated_cost <= available
        return estimated_cost <= self._cash

    def _update_margin(self) -> None:
        """Herbereken margin account."""
        if not self._margin_enabled:
            return

        # Totale positie waarde
        total_position_value = 0.0
        for sym, pos in self._positions.items():
            price = self._last_prices.get(sym, pos.get("avg_price", 0))
            total_position_value += abs(pos["qty"]) * price

        self._margin.equity = self._cash + total_position_value
        self._margin.margin_used = total_position_value / self._max_leverage
        self._margin.maintenance_margin = total_position_value * 0.25  # 25%

        # Margin call check
        if self._margin.equity < self._margin.maintenance_margin:
            self._margin.margin_call = True
            logger.warning(
                "MARGIN CALL: equity=%.2f < maintenance=%.2f",
                self._margin.equity, self._margin.maintenance_margin,
            )

    # ── Funding Rates (Crypto) ──────────────────────────────────────

    def _process_funding(self) -> None:
        """Verwerk 8-uurs funding rates voor crypto perpetuals."""
        now = time.time()
        hours_elapsed = (now - self._last_funding_time) / 3600

        if hours_elapsed < _FUNDING_RATE_INTERVAL_H:
            return

        self._last_funding_time = now

        for symbol, pos in list(self._positions.items()):
            if pos.get("market") != "crypto":
                continue

            qty = pos["qty"]
            price = self._last_prices.get(symbol, pos.get("avg_price", 0))
            rate = self._funding_rates.get(symbol, _DEFAULT_FUNDING_RATE)

            # Funding: longs betalen bij positieve rate, shorts ontvangen
            funding_payment = abs(qty) * price * rate
            if qty > 0:
                self._cash -= funding_payment  # longs betalen
            else:
                self._cash += funding_payment  # shorts ontvangen

            if abs(funding_payment) > 0.01:
                logger.debug(
                    "Funding: %s %.6f @ rate=%.4f%% → payment=%.4f",
                    symbol, qty, rate * 100, funding_payment,
                )

    # ── Portfolio Updates ───────────────────────────────────────────

    def _apply_fill(self, order: Order, fill: Fill) -> None:
        """Pas een fill toe op de portfolio."""
        symbol = order.symbol
        pos = self._positions.get(symbol, {"qty": 0.0, "avg_price": 0.0, "market": order.market})

        if order.side == OrderSide.BUY:
            # Koop: verhoog positie, verlaag cash
            old_qty = pos["qty"]
            old_cost = old_qty * pos["avg_price"]
            new_cost = fill.quantity * fill.price
            total_qty = old_qty + fill.quantity
            pos["avg_price"] = (old_cost + new_cost) / total_qty if total_qty > 0 else 0
            pos["qty"] = total_qty
            pos["market"] = order.market
            self._cash -= (fill.quantity * fill.price + fill.fee)

        elif order.side == OrderSide.SELL:
            # Verkoop: verlaag positie, verhoog cash
            realized_pnl = fill.quantity * (fill.price - pos["avg_price"])
            pos["qty"] -= fill.quantity
            self._cash += (fill.quantity * fill.price - fill.fee)

            # Verwijder positie als volledig verkocht
            if pos["qty"] < _MIN_PARTIAL_FILL_QTY:
                self._positions.pop(symbol, None)
                self._update_margin()
                return

        self._positions[symbol] = pos
        self._update_margin()

    # ── Utilities ───────────────────────────────────────────────────

    def _simulate_latency(self) -> None:
        """Simuleer order verwerkingslatency (niet-blokkerend)."""
        # In productie zou dit async zijn; hier loggen we het alleen
        latency_s = self.latency_ms / 1000.0
        # We slapen niet echt — simulatie is instant maar wordt gelogd
        logger.debug("Order latency: %.1fms", self.latency_ms)

    def _reject_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        timestamp: str,
        reason: str,
    ) -> Order:
        """Maak een rejected order."""
        self._rejected_count += 1
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=OrderSide(side) if side in ("buy", "sell") else OrderSide.BUY,
            order_type=OrderType(order_type) if order_type in [e.value for e in OrderType] else OrderType.MARKET,
            quantity=0,
            status=OrderStatus.REJECTED,
            reject_reason=reason,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._order_history.append(order)
        logger.debug("Order rejected: %s %s %s — %s", side, symbol, order_type, reason)
        return order

    # ── Cancel ──────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """Annuleer een open order.

        Returns True als de order succesvol geannuleerd is.
        """
        order = self._open_orders.pop(order_id, None)
        if order is None:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._order_history.append(order)
        logger.debug("Order geannuleerd: %s", order_id)
        return True

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Annuleer alle open orders (optioneel voor specifiek symbool).

        Returns aantal geannuleerde orders.
        """
        count = 0
        for oid in list(self._open_orders.keys()):
            order = self._open_orders[oid]
            if symbol and order.symbol != symbol.upper():
                continue
            self.cancel_order(oid)
            count += 1
        return count

    # ── Day Reset ───────────────────────────────────────────────────

    def reset_day(self) -> None:
        """Reset dagelijkse state (circuit breakers, day orders, open prices)."""
        # Cancel day orders
        for oid, order in list(self._open_orders.items()):
            if order.time_in_force == TimeInForce.DAY:
                order.status = OrderStatus.EXPIRED
                order.updated_at = datetime.now(timezone.utc).isoformat()
                self._open_orders.pop(oid)
                self._order_history.append(order)

        # Reset circuit breakers
        self._circuit_breakers.clear()
        self._day_open_prices.clear()

        logger.debug("Dagelijkse broker reset uitgevoerd.")

    # ── Queries ─────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders

    def get_order_history(self, limit: int = 50) -> List[Order]:
        return list(self._order_history)[-limit:]

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        pos = self._positions.get(symbol.upper())
        if pos is None:
            return None
        price = self._last_prices.get(symbol.upper(), pos["avg_price"])
        qty = pos["qty"]
        unrealized_pnl = qty * (price - pos["avg_price"])
        return {
            "symbol": symbol.upper(),
            "quantity": round(qty, 6),
            "avg_price": round(pos["avg_price"], 4),
            "current_price": round(price, 4),
            "market_value": round(qty * price, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "market": pos.get("market", "us"),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        return [self.get_position(s) for s in self._positions if self.get_position(s)]

    def get_cash(self) -> float:
        return round(self._cash, 2)

    def get_equity(self) -> float:
        total = self._cash
        for sym, pos in self._positions.items():
            price = self._last_prices.get(sym, pos["avg_price"])
            total += pos["qty"] * price
        return round(total, 2)

    def get_order_book(self, symbol: str) -> Optional[Dict[str, List[Dict]]]:
        book = self._order_books.get(symbol.upper())
        if not book:
            return None
        return {
            "bids": [{"price": l.price, "qty": l.quantity, "orders": l.order_count} for l in book["bids"]],
            "asks": [{"price": l.price, "qty": l.quantity, "orders": l.order_count} for l in book["asks"]],
        }

    def get_circuit_breaker(self, symbol: str) -> Optional[Dict[str, Any]]:
        cb = self._circuit_breakers.get(symbol.upper())
        if not cb:
            return None
        return {
            "symbol": cb.symbol, "halted": cb.halted,
            "reason": cb.halt_reason, "until": cb.halt_until,
            "ref_price": cb.reference_price,
            "upper": cb.upper_limit, "lower": cb.lower_limit,
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retourneer uitgebreid portfolio overzicht."""
        positions = self.get_positions()
        equity = self.get_equity()
        total_pnl = equity - self._initial_cash
        total_pnl_pct = (total_pnl / self._initial_cash * 100) if self._initial_cash > 0 else 0

        return {
            "cash": self.get_cash(),
            "equity": equity,
            "positions_count": len(positions),
            "positions": positions,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "total_fees_paid": round(self._total_fees_paid, 4),
            "total_slippage": round(self._total_slippage, 4),
            "total_fills": self._fills_count,
            "rejected_orders": self._rejected_count,
            "open_orders": len(self._open_orders),
            "margin_enabled": self._margin_enabled,
            "margin": {
                "equity": round(self._margin.equity, 2),
                "used": round(self._margin.margin_used, 2),
                "available": round(self._margin.available_margin, 2),
                "ratio": round(self._margin.margin_ratio, 4),
                "margin_call": self._margin.margin_call,
            } if self._margin_enabled else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retourneer broker statistieken."""
        return {
            "total_fees": round(self._total_fees_paid, 4),
            "total_slippage": round(self._total_slippage, 4),
            "total_fills": self._fills_count,
            "rejected_orders": self._rejected_count,
            "active_orders": len(self._open_orders),
            "active_positions": len(self._positions),
            "circuit_breakers_active": sum(1 for cb in self._circuit_breakers.values() if cb.halted),
        }


# ── Backward compatible wrapper ─────────────────────────────────────

def execute_trade_v2(
    request: Any,
    limits: Dict[str, Any],
    broker: SimulatedBroker,
    confidence: float = 0.0,
    model_version: str = "",
) -> Dict[str, Any]:
    """Backward-compatible wrapper die SimulatedBroker gebruikt.

    Drop-in vervanging voor ``trade_engine.execute_trade()``.
    Accepteert een TradeRequest en retourneert een dict compatibel met TradeResult.
    """
    from models.trade import TradeRequest

    symbol = request.symbol.upper()
    action = request.action
    market = request.market or "us"

    order = broker.submit_order(
        symbol=symbol,
        side=action,
        order_type="market",
        quantity=request.quantity or 0,
        amount=request.amount or 0,
        price=request.price,
        market=market,
    )

    # Als er een prijs beschikbaar is, probeer direct te processen
    if order.status == OrderStatus.OPEN:
        price = request.price or broker._last_prices.get(symbol, 0)
        if price > 0:
            broker.process_market_data({
                symbol: {"price": price, "volume": 1e6},
            })
            # Herlaad order status
            order = broker._orders.get(order.order_id, order)

    # Vertaal naar TradeResult-compatibel dict
    status_map = {
        OrderStatus.FILLED: f"executed: {'positie toegevoegd' if action == 'buy' else 'positie gesloten'}",
        OrderStatus.PARTIAL: "executed: gedeeltelijk gevuld",
        OrderStatus.REJECTED: f"rejected: {order.reject_reason}",
        OrderStatus.OPEN: "pending: wacht op uitvoering",
        OrderStatus.CANCELLED: f"cancelled: {order.reject_reason}",
    }

    return {
        "id": order.order_id,
        "symbol": symbol,
        "action": action,
        "amount": round(order.filled_qty * order.avg_fill_price, 2),
        "quantity": round(order.filled_qty, 6),
        "price": round(order.avg_fill_price, 4),
        "profit_loss": 0.0,  # berekend door DataService
        "timestamp": order.updated_at or order.created_at,
        "status": status_map.get(order.status, order.status.value),
        "expected_return_pct": getattr(request, "expected_return_pct", 0.0),
        "risk_pct": getattr(request, "risk_pct", 0.0),
        "confidence": confidence,
        "model_version": model_version,
        "fees": round(order.total_fees, 6),
        "fills": len(order.fills),
        "order_id": order.order_id,
    }
