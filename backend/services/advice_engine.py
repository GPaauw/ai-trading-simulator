import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict
from zoneinfo import ZoneInfo

from models.holding import HoldingView
from models.signal import Signal
from services.market_data_service import MarketDataService

# Maximaal aantal parallelle API-verzoeken
_MAX_WORKERS = 25
# Signaal-cache TTL in seconden (5 minuten)
_SIGNAL_CACHE_TTL = 5 * 60


class AdviceEngine:
    def __init__(self, market_data: MarketDataService) -> None:
        self._market_data = market_data
        # Learningparameters die door de LearningAgent worden bijgesteld
        self._params: Dict[str, float] = {
            "confidence_bias": 0.0,
            "risk_multiplier": 1.0,
            "momentum_weight": 0.35,
            "sma_weight": 1.0,
            "rsi_weight": 1.0,
            "macd_weight": 0.8,
            "bb_weight": 0.7,
            "volume_weight": 0.5,
        }
        # Signaal-cache
        self._cached_signals: list[Signal] = []
        self._cache_time: float = 0.0

    def apply_learned_params(self, params: Dict[str, float]) -> None:
        for key in params:
            if key in self._params:
                self._params[key] = params[key]

    def invalidate_signal_cache(self) -> None:
        self._cache_time = 0.0

    def build_signals(self, force_refresh: bool = False) -> list[Signal]:
        now = time.time()
        if not force_refresh and self._cached_signals and (now - self._cache_time) < _SIGNAL_CACHE_TTL:
            return self._cached_signals

        # Pre-fetch alle stock data in batches (sneller dan individueel)
        self._market_data.prefetch_stock_data(force=force_refresh)

        watchlist = self._market_data.get_watchlist()
        signals: list[Signal] = []

        def _process(instrument: dict[str, str]) -> Signal | None:
            try:
                snapshot = self._market_data.get_snapshot(instrument)
                closes = self._market_data.get_history(instrument, points=120)
                if len(closes) < 30:
                    return None
                return self._to_signal(instrument, snapshot, closes)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_process, inst): inst for inst in watchlist}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    signals.append(result)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        self._cached_signals = signals
        self._cache_time = time.time()
        return signals

    def build_signal_map(self) -> dict[str, Signal]:
        return {signal.symbol: signal for signal in self.build_signals()}

    def build_ranked_buy_signals(self, force_refresh: bool = False) -> list[Signal]:
        return sorted(
            [signal for signal in self.build_signals(force_refresh=force_refresh) if signal.action == "buy"],
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
        reason = "Positie monitoren; dag nog niet voorbij."

        # Day-trading: verkoop ALTIJD aan einde van de dag
        end_of_day = _is_end_of_day(str(holding["market"]))
        if end_of_day:
            recommendation = "sell_now"
            recommendation_label = "Verkoop nu (einde dag)"
            suggested_action = "sell"
            suggested_sell_fraction = 1.0
            reason = "Einde handelsdag: sluit alle posities af (day-trading strategie)."
        elif current_price <= stop_loss_price:
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

    # ------------------------------------------------------------------
    # Signaalberekening met SMA, RSI, MACD, Bollinger, Volume
    # ------------------------------------------------------------------

    def _to_signal(self, instrument: dict[str, str], snapshot: dict[str, float], closes: list[float]) -> Signal:
        price = snapshot["price"]
        volume = snapshot.get("volume", 0.0)

        sma5 = _sma(closes, 5)
        sma20 = _sma(closes, 20)
        rsi = _rsi(closes, 14)
        momentum_pct = ((closes[-1] / closes[-20]) - 1.0) * 100
        volatility_pct = _volatility_pct(closes, 20)

        # MACD (12, 26, 9)
        macd_line, signal_line, macd_histogram = _macd(closes)

        # Bollinger Bands (20, 2)
        bb_upper, bb_middle, bb_lower = _bollinger_bands(closes, 20, 2.0)

        # Volume-trend: vergelijk huidige volume met gemiddeld
        avg_volume = _avg_volume_from_closes(closes, 20)  # benadering via prijsbeweging
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 1.0

        # ------ Score berekenen ------
        p = self._params
        score = 0.0

        # SMA crossover
        if sma5 > sma20:
            score += 1.0 * p["sma_weight"]
        else:
            score -= 1.0 * p["sma_weight"]

        # RSI
        if rsi < 30:
            score += 1.5 * p["rsi_weight"]  # sterk oversold
        elif rsi < 40:
            score += 0.8 * p["rsi_weight"]
        elif rsi > 70:
            score -= 1.5 * p["rsi_weight"]  # sterk overbought
        elif rsi > 60:
            score -= 0.5 * p["rsi_weight"]

        # Momentum
        if momentum_pct > 1.0:
            score += 1.0 * p["momentum_weight"]
        elif momentum_pct > 0.3:
            score += 0.5 * p["momentum_weight"]
        elif momentum_pct < -1.0:
            score -= 1.0 * p["momentum_weight"]
        elif momentum_pct < -0.3:
            score -= 0.5 * p["momentum_weight"]

        # MACD
        if macd_histogram > 0 and macd_line > signal_line:
            score += 1.0 * p["macd_weight"]
        elif macd_histogram < 0 and macd_line < signal_line:
            score -= 1.0 * p["macd_weight"]

        # Bollinger Bands
        if price <= bb_lower:
            score += 1.2 * p["bb_weight"]  # prijs bij lower band = koopkans
        elif price >= bb_upper:
            score -= 1.2 * p["bb_weight"]  # prijs bij upper band = verkoop-signaal

        # Volume bevestiging
        if volume_ratio > 1.5 and score > 0:
            score += 0.5 * p["volume_weight"]  # hoog volume bevestigt trend
        elif volume_ratio > 1.5 and score < 0:
            score -= 0.5 * p["volume_weight"]

        # ------ Actie bepalen (day-trading: intraday focus) ------
        expected_move_pct = _clamp(
            (score * 0.38) + (momentum_pct * p["momentum_weight"]),
            -2.0,
            2.0,
        )

        market_open = _is_market_open(instrument["market"])
        if expected_move_pct > 0.15:
            action = "buy"
        elif expected_move_pct < -0.15:
            action = "sell"
        else:
            action = "hold"

        if not market_open and instrument["market"] != "crypto":
            action = "hold"

        # ------ Confidence & risico ------
        confidence = _clamp(
            0.5
            + p["confidence_bias"]
            + (abs(score) * 0.06)
            + (min(abs(momentum_pct), 4.0) * 0.035)
            + (0.03 if macd_histogram > 0 and action == "buy" else 0.0)
            + (0.02 if price <= bb_lower and action == "buy" else 0.0),
            0.50,
            0.95,
        )

        market_risk_bias = 0.30 if instrument["market"] == "crypto" else 0.0
        # Day-trading: strakke stop-loss (max 1%)
        risk_pct = _clamp(
            (0.6 - (volatility_pct * 0.03) + market_risk_bias) * p["risk_multiplier"],
            0.2,
            1.0,
        )

        # Day-trading: verwachte winst is intraday (0.2-2%)
        expected_trade_return_pct = _clamp(abs(expected_move_pct), 0.0, 2.0)

        # ------ Ranking score (0-100, genuanceerd) ------
        ranking_score = _clamp(
            (confidence - 0.5) * 80          # 0-36 punten (hoe hoger confidence, hoe beter)
            + expected_trade_return_pct * 5.0 # 0-25 punten (verwachte winst)
            + max(momentum_pct, 0.0) * 1.5   # 0-6 punten (positief momentum)
            + (3.0 if macd_histogram > 0 and action == "buy" else 0.0)
            + (2.5 if price <= bb_lower and action == "buy" else 0.0)
            + (2.0 if volume_ratio > 1.5 and action == "buy" else 0.0)
            + 30.0                            # basislijn
            - (risk_pct * 3.0),               # risico-aftrek
            0.0,
            100.0,
        )

        rank_label = ""
        if action == "buy":
            if ranking_score >= 85:
                rank_label = "Topkans"
            elif ranking_score >= 78:
                rank_label = "Sterke setup"
            elif ranking_score >= 70:
                rank_label = "Goede kans"
            else:
                rank_label = "Interessante kans"

        # ------ Koersdoel & tijdshorizon (day-trading: altijd 1 dag) ------
        target_price = round(price * (1 + expected_trade_return_pct / 100), 4)
        expected_days = 1  # day-trading: koop ochtend, verkoop einde dag
        expected_profit = round(1000 * expected_trade_return_pct / 100, 2)
        time_label = "vandaag (daytrade)"

        # Reden met alle indicators
        reason = (
            f"SMA5/20={sma5:.2f}/{sma20:.2f}, RSI={rsi:.1f}, "
            f"MACD={macd_histogram:+.3f}, "
            f"BB={bb_lower:.2f}/{bb_upper:.2f}, "
            f"mom={momentum_pct:+.2f}%, vol={volatility_pct:.2f}% | "
            f"doel €{target_price:.2f} in {time_label}"
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
            target_price=target_price,
            expected_profit=expected_profit,
            expected_days=expected_days,
            ranking_score=round(ranking_score, 3),
            rank_label=rank_label,
            reason=reason,
        )


# ==================================================================
# Technische analyse functies
# ==================================================================

def _is_market_open(market: str) -> bool:
    if market == "crypto":
        return True
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    # US markt: 15:30-22:00 NL tijd
    return (now.hour == 15 and now.minute >= 30) or (16 <= now.hour < 22)


def _is_end_of_day(market: str) -> bool:
    """Check of het bijna einde handelsdag is."""
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    if market == "crypto":
        # Crypto: sluit elke 'dag' om 23:00 NL tijd
        return now.hour >= 22
    # US markt sluit om 22:00 NL tijd
    return now.hour >= 21 and now.minute >= 30


def _sma(values: list[float], period: int) -> float:
    period_values = values[-period:]
    return sum(period_values) / len(period_values)


def _ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average over de hele reeks."""
    if len(values) < period:
        return values[:]
    multiplier = 2.0 / (period + 1)
    ema_values = [_sma(values[:period], period)]
    for i in range(period, len(values)):
        ema_values.append((values[i] - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _macd(values: list[float]) -> tuple[float, float, float]:
    """MACD (12, 26, 9). Return (macd_line, signal_line, histogram)."""
    if len(values) < 35:
        return 0.0, 0.0, 0.0
    ema12 = _ema(values, 12)
    ema26 = _ema(values, 26)
    # Align op de kortste
    min_len = min(len(ema12), len(ema26))
    macd_line_series = [ema12[-(min_len - i)] - ema26[-(min_len - i)] for i in range(min_len)]
    if len(macd_line_series) < 9:
        val = macd_line_series[-1] if macd_line_series else 0.0
        return val, 0.0, val
    signal_series = _ema(macd_line_series, 9)
    macd_val = macd_line_series[-1]
    signal_val = signal_series[-1]
    return macd_val, signal_val, macd_val - signal_val


def _bollinger_bands(values: list[float], period: int, num_std: float) -> tuple[float, float, float]:
    """Return (upper, middle, lower) Bollinger Band."""
    if len(values) < period:
        mid = values[-1] if values else 0.0
        return mid, mid, mid
    window = values[-period:]
    middle = sum(window) / period
    std = statistics.stdev(window) if len(window) > 1 else 0.0
    return middle + num_std * std, middle, middle - num_std * std


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


def _avg_volume_from_closes(closes: list[float], period: int) -> float:
    """Schat gemiddelde beweging als proxy voor volume wanneer we geen volume data hebben."""
    if len(closes) < period + 1:
        return 1.0
    moves = []
    recent = closes[-(period + 1):]
    for i in range(1, len(recent)):
        moves.append(abs(recent[i] - recent[i - 1]))
    return sum(moves) / len(moves) if moves else 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
