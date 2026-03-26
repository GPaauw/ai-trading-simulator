from typing import Dict, List

from models.signal import Signal
from models.trade import TradeResult


class LearningAgent:
    """Lichte leerlaag die risico- en confidence-suggesties afleidt uit recente resultaten."""

    def __init__(self) -> None:
        self._scores: Dict[str, float] = {"buy": 0.6, "sell": 0.6}
        self._learning_rate: float = 0.04
        self._total_iterations: int = 0

    def learn(self, history: List[TradeResult], latest_signals: List[Signal]) -> Dict[str, object]:
        if not history:
            return self._build_result(0, 0.0, 0.0, latest_signals)

        analyzed = 0
        wins = 0
        total_pl = 0.0
        for trade in history:
            if trade.status != "executed":
                continue

            reward = 1.0 if trade.profit_loss > 0 else -1.0
            if trade.profit_loss > 0:
                wins += 1

            total_pl += trade.profit_loss
            action = trade.action
            if action in self._scores:
                self._scores[action] += self._learning_rate * reward
                # Houd waarde tussen 0 en 1
                self._scores[action] = max(0.0, min(1.0, self._scores[action]))
            analyzed += 1

        self._total_iterations += 1
        win_rate = (wins / analyzed) if analyzed else 0.0
        avg_pl = (total_pl / analyzed) if analyzed else 0.0
        return self._build_result(analyzed, win_rate, avg_pl, latest_signals)

    def _build_result(
        self,
        analyzed: int,
        win_rate: float,
        avg_pl: float,
        latest_signals: List[Signal],
    ) -> Dict[str, object]:
        avg_signal_risk = (
            sum(s.risk_pct for s in latest_signals) / len(latest_signals)
            if latest_signals
            else 1.0
        )
        suggested_risk = max(0.4, min(2.0, avg_signal_risk * (0.8 + win_rate)))

        return {
            "buy_confidence": round(self._scores["buy"], 4),
            "sell_confidence": round(self._scores["sell"], 4),
            "learning_rate": self._learning_rate,
            "trades_analyzed": analyzed,
            "win_rate": round(win_rate, 4),
            "avg_profit_loss": round(avg_pl, 4),
            "suggested_risk_pct": round(suggested_risk, 3),
            "total_learning_iterations": self._total_iterations,
        }
