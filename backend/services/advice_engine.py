import statistics
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from models.holding import HoldingView
from models.signal import Signal
from services.market_data_service import MarketDataService


class AdviceEngine:
    def __init__(self, market_data: MarketDataService) -> None:
        self._market_data = market_data

    def build_signals(self) -> list[Signal]:
        signals: list[Signal] = []
        for instrument in self._market_data.get_watchlist():
            try:
                snapshot = self._market_data.get_snapshot(instrument)
                closes = self._market_data.get_history(instrument, points=120)
                if len(closes) < 25:
                    continue
                signals.append(self._to_signal(instrument, snapshot["price"], closes))
            except Exception:
                # Bij externe API-fouten slaan we het instrument over.
                continue

        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def build_signal_map(self) -> dict[str, Signal]:
        return {signal.symbol: signal for signal in self.build_signals()}

    def build_ranked_buy_signals(self) -> list[Signal]:
        return sorted(
            [signal for signal in self.build_signals() if signal.action == "buy"],
            key=lambda signal: (-signal.ranking_score, -signal.confidence, -signal.expected_return_pct),
        )

    def build_holding_view(
        self,
        holding: dict[str, object],
        signal: Signal | None,
        current_price: float,
    ) -> HoldingView:
        quantity = float(holding["quantity"])
        invested_amount = float(holding["invested_amount"])
        avg_entry_price = float(holding["avg_entry_price"])
        market_value = quantity * current_price
        unrealized_profit_loss = market_value - invested_amount
        unrealized_profit_loss_pct = (
            ((current_price / avg_entry_price) - 1.0) * 100 if avg_entry_price else 0.0
        )

        risk_pct = signal.risk_pct if signal else 1.2
        expected_return_pct = signal.expected_return_pct if signal else 2.0
        confidence = signal.confidence if signal else 0.55
        recommendation_label = "Houd nog vast"
        suggested_action = "hold"
        suggested_sell_fraction = 0.0

        stop_loss_price = avg_entry_price * (1.0 - (risk_pct / 100.0))
        target_price = avg_entry_price * (1.0 + (expected_return_pct / 100.0))
        stretch_target_price = avg_entry_price * (1.0 + ((expected_return_pct * 1.35) / 100.0))
        position_score = _clamp((confidence * 100) + unrealized_profit_loss_pct - (risk_pct * 4.5), 0.0, 100.0)

        recommendation = "hold"
        reason = "Positie monitoren; nog geen verkooptrigger geraakt."
        if current_price <= stop_loss_price:
            recommendation = "sell_now"
            recommendation_label = "Verkoop nu"
            suggested_action = "sell"
            suggested_sell_fraction = 1.0
            reason = "Stop-loss geraakt op basis van actuele marktprijs."
        elif signal and signal.action == "sell" and signal.confidence >= 0.64:
            recommendation = "sell_now"
            recommendation_label = "Verkoop nu"
            suggested_action = "sell"
            suggested_sell_fraction = 1.0
            reason = "Actueel model geeft een sell-signaal met voldoende zekerheid."
        elif current_price >= stretch_target_price:
            recommendation = "sell_now"
            recommendation_label = "Verkoop nu"
            suggested_action = "sell"
            suggested_sell_fraction = 1.0
            reason = "Sterk koersdoel ruim overschreden; winst veiligstellen is verstandig."
        elif current_price >= target_price and signal and signal.action == "buy" and signal.confidence >= 0.7:
            recommendation = "take_partial_profit"
            recommendation_label = "Neem deels winst"
            suggested_action = "sell_partial"
            suggested_sell_fraction = 0.5
            reason = "Koersdoel bereikt, maar momentum blijft positief; gedeeltelijk winst nemen is logisch."
        elif unrealized_profit_loss_pct >= max(expected_return_pct, 4.0):
            recommendation = "take_partial_profit"
            recommendation_label = "Neem deels winst"
            suggested_action = "sell_partial"
            suggested_sell_fraction = 0.35
            reason = "De positie staat duidelijk in de plus; gedeeltelijk winst nemen verlaagt risico."
        elif signal and signal.action == "buy":
            reason = "Trend blijft positief; positie kan worden aangehouden."

        return HoldingView(
            symbol=str(holding["symbol"]),
            market=str(holding["market"]),
            quantity=round(quantity, 6),
            invested_amount=round(invested_amount, 2),
            avg_entry_price=round(avg_entry_price, 4),
            current_price=round(current_price, 4),
            market_value=round(market_value, 2),
            unrealized_profit_loss=round(unrealized_profit_loss, 2),
            unrealized_profit_loss_pct=round(unrealized_profit_loss_pct, 3),
            recommendation=recommendation,
            recommendation_label=recommendation_label,
            recommendation_reason=reason,
            suggested_action=suggested_action,
            suggested_sell_fraction=round(suggested_sell_fraction, 3),
            target_price=round(target_price, 4),
            stop_loss_price=round(stop_loss_price, 4),
            confidence=round(confidence, 4),
            expected_return_pct=round(expected_return_pct, 3),
            risk_pct=round(risk_pct, 3),
            position_score=round(position_score, 3),
            opened_at=str(holding["opened_at"]),
            updated_at=str(holding["updated_at"]),
        )

    def _to_signal(self, instrument: dict[str, str], price: float, closes: list[float]) -> Signal:
        sma_short = _sma(closes, 5)
        sma_long = _sma(closes, 20)
        rsi = _rsi(closes, 14)
        momentum_pct = ((closes[-1] / closes[-20]) - 1.0) * 100
        volatility_pct = _volatility_pct(closes, 20)

        score = 0.0
        if sma_short > sma_long:
            score += 1.0
        else:
            score -= 1.0

        if rsi < 35:
            score += 1.0
        elif rsi > 65:
            score -= 1.0

        if momentum_pct > 0.5:
            score += 1.0
        elif momentum_pct < -0.5:
            score -= 1.0

        expected_move_pct = _clamp((score * 0.45) + (momentum_pct * 0.35), -4.0, 4.0)

        market_open = _is_market_open(instrument["market"])
        if expected_move_pct > 0.30:
            action = "buy"
        elif expected_move_pct < -0.30:
            action = "sell"
        else:
            action = "hold"

        if not market_open and instrument["market"] != "crypto":
            action = "hold"

        confidence = _clamp(0.5 + (abs(score) * 0.08) + (min(abs(momentum_pct), 3.0) * 0.04), 0.52, 0.95)

        market_risk_bias = 0.25 if instrument["market"] == "crypto" else 0.0
        risk_pct = _clamp(1.4 - (volatility_pct * 0.06) + market_risk_bias, 0.4, 2.5)

        expected_trade_return_pct = abs(expected_move_pct)
        ranking_score = _clamp(
            (confidence * 100)
            + (expected_trade_return_pct * 8.0)
            + max(momentum_pct, 0.0) * 2.5
            - (risk_pct * 5.0),
            0.0,
            100.0,
        )
        rank_label = ""
        if action == "buy":
            if ranking_score >= 85:
                rank_label = "Beste kansen vandaag"
            elif ranking_score >= 74:
                rank_label = "Sterke setup"
            else:
                rank_label = "Interessante kans"

        reason = (
            f"SMA5/SMA20={sma_short:.2f}/{sma_long:.2f}, RSI={rsi:.1f}, "
            f"momentum={momentum_pct:.2f}%, vol={volatility_pct:.2f}%"
        )

        return Signal(
            id=str(uuid.uuid4()),
            symbol=instrument["symbol"],
            market=instrument["market"],
            action=action,
            confidence=round(confidence, 4),
            price=round(price, 4),
            risk_pct=round(risk_pct, 3),
            expected_return_pct=round(expected_trade_return_pct, 3),
            ranking_score=round(ranking_score, 3),
            rank_label=rank_label,
            reason=reason,
        )


def _is_market_open(market: str) -> bool:
    if market == "crypto":
        return True

    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return 8 <= now.hour < 16


def _sma(values: list[float], period: int) -> float:
    period_values = values[-period:]
    return sum(period_values) / len(period_values)


def _rsi(values: list[float], period: int) -> float:
    if len(values) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _volatility_pct(values: list[float], period: int) -> float:
    if len(values) < period + 1:
        return 1.0
    returns = []
    recent = values[-(period + 1):]
    for i in range(1, len(recent)):
        prev = recent[i - 1]
        curr = recent[i]
        if prev <= 0:
            continue
        returns.append((curr / prev) - 1.0)

    if len(returns) < 2:
        return 1.0
    return statistics.stdev(returns) * 100


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))