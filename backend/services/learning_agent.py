from typing import Dict, List

from models.trade import TradeResult


class LearningAgent:
    """
    Eenvoudige reinforcement-learning simulatie.
    Houdt een score bij per actie en past confidence aan op basis van winst/verlies.
    Uitbreidbaar naar complexere modellen.
    """

    def __init__(self) -> None:
        self._scores: Dict[str, float] = {"buy": 0.5, "sell": 0.5}
        self._learning_rate: float = 0.05
        self._total_iterations: int = 0

    def learn(self, history: List[TradeResult]) -> Dict[str, object]:
        if not history:
            return self._build_result(0)

        analyzed = 0
        for trade in history:
            if trade.status != "executed":
                continue
            reward = 1.0 if trade.profit_loss > 0 else -1.0
            action = trade.action
            if action in self._scores:
                self._scores[action] += self._learning_rate * reward
                # Houd waarde tussen 0 en 1
                self._scores[action] = max(0.0, min(1.0, self._scores[action]))
            analyzed += 1

        self._total_iterations += 1
        return self._build_result(analyzed)

    def _build_result(self, analyzed: int) -> Dict[str, object]:
        return {
            "buy_confidence": round(self._scores["buy"], 4),
            "sell_confidence": round(self._scores["sell"], 4),
            "learning_rate": self._learning_rate,
            "trades_analyzed": analyzed,
            "total_learning_iterations": self._total_iterations,
        }
