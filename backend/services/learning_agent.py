import json
from typing import Any, Dict, List

from models.signal import Signal
from models.trade import TradeResult


class LearningAgent:
    """Leerlaag die risico- en confidence-suggesties bijstelt op basis van resultaten.

    Analyseert fouten (verliesgevende trades) en past signaalparameters dynamisch aan
    zodat dezelfde fouten minder vaak herhaald worden.
    """

    def __init__(self) -> None:
        self._learning_rate: float = 0.04
        self._total_iterations: int = 0

        # Parameters die doorgegeven worden aan de AdviceEngine
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

        # Foutstatistieken
        self._error_log: List[Dict[str, object]] = []
        self._max_drawdown_pct: float = 0.0
        self._consecutive_losses: int = 0
        self._peak_balance: float = 0.0

    def learn(self, history: List[TradeResult], latest_signals: List[Signal]) -> Dict[str, object]:
        if not history:
            return self._build_result(0, 0.0, 0.0, latest_signals)

        # Analyse van gesloten trades (sells)
        sells = [t for t in history if t.status.startswith("executed") and t.action == "sell"]
        if not sells:
            return self._build_result(0, 0.0, 0.0, latest_signals)

        wins = 0
        losses = 0
        total_pl = 0.0
        total_win_pl = 0.0
        total_loss_pl = 0.0
        biggest_loss = 0.0
        biggest_win = 0.0
        losing_streaks: List[int] = []
        current_streak = 0

        for trade in sells:
            total_pl += trade.profit_loss
            if trade.profit_loss > 0:
                wins += 1
                total_win_pl += trade.profit_loss
                biggest_win = max(biggest_win, trade.profit_loss)
                if current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 0
            else:
                losses += 1
                total_loss_pl += trade.profit_loss
                biggest_loss = min(biggest_loss, trade.profit_loss)
                current_streak += 1

                # Log de fout voor analyse
                self._error_log.append({
                    "symbol": trade.symbol,
                    "loss": trade.profit_loss,
                    "expected_return_pct": trade.expected_return_pct,
                    "risk_pct": trade.risk_pct,
                })

        if current_streak > 0:
            losing_streaks.append(current_streak)

        analyzed = len(sells)
        win_rate = wins / analyzed if analyzed else 0.0
        avg_pl = total_pl / analyzed if analyzed else 0.0

        # ------ Parameter-tuning op basis van fouten ------
        self._tune_parameters(win_rate, avg_pl, biggest_loss, losing_streaks, sells)

        self._total_iterations += 1
        self._consecutive_losses = current_streak

        return self._build_result(analyzed, win_rate, avg_pl, latest_signals, extra={
            "wins": wins,
            "losses": losses,
            "biggest_win": round(biggest_win, 2),
            "biggest_loss": round(biggest_loss, 2),
            "avg_win": round(total_win_pl / wins, 2) if wins else 0.0,
            "avg_loss": round(total_loss_pl / losses, 2) if losses else 0.0,
            "max_losing_streak": max(losing_streaks) if losing_streaks else 0,
            "consecutive_losses": self._consecutive_losses,
            "risk_level": self._risk_level_label(),
            "error_patterns": self._analyze_error_patterns(),
        })

    def get_engine_params(self) -> Dict[str, float]:
        """Return de geleerde parameters die naar AdviceEngine gaan."""
        return dict(self._params)

    def save_params(self, data_service: Any) -> None:
        """Sla geleerde parameters permanent op in de database (overleeft herstart)."""
        data_service._set_setting("learning_params", json.dumps(self._params))
        data_service._set_setting("learning_iterations", str(self._total_iterations))

    def load_params(self, data_service: Any) -> None:
        """Laad eerder opgeslagen parameters uit de database."""
        raw = data_service._get_setting("learning_params", "")
        if raw:
            try:
                loaded = json.loads(raw)
                for k in self._params:
                    if k in loaded:
                        self._params[k] = float(loaded[k])
            except Exception:
                pass
        iterations_str = data_service._get_setting("learning_iterations", "0")
        try:
            self._total_iterations = int(iterations_str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Parameter-tuning
    # ------------------------------------------------------------------

    def _tune_parameters(
        self,
        win_rate: float,
        avg_pl: float,
        biggest_loss: float,
        losing_streaks: List[int],
        sells: List[TradeResult],
    ) -> None:
        lr = self._learning_rate

        # 1. Als winrate laag is → verhoog confidence-drempel (meer selectief)
        if win_rate < 0.45:
            self._params["confidence_bias"] = _clamp(
                self._params["confidence_bias"] + lr * 0.5, -0.1, 0.15
            )
        elif win_rate > 0.65:
            self._params["confidence_bias"] = _clamp(
                self._params["confidence_bias"] - lr * 0.3, -0.1, 0.15
            )

        # 2. Als gemiddeld verlies te groot → verhoog risico-multiplier (strikter)
        if avg_pl < -5.0:
            self._params["risk_multiplier"] = _clamp(
                self._params["risk_multiplier"] + lr * 2, 0.5, 2.0
            )
        elif avg_pl > 5.0:
            self._params["risk_multiplier"] = _clamp(
                self._params["risk_multiplier"] - lr, 0.5, 2.0
            )

        # 3. Analyseer welke indicatoren fout zitten
        # Als veel verliezen bij hoog momentum → verlaag momentum-gewicht
        high_mom_losses = [t for t in sells if t.profit_loss < 0 and t.expected_return_pct > 2.0]
        if len(high_mom_losses) > len(sells) * 0.3:
            self._params["momentum_weight"] = _clamp(
                self._params["momentum_weight"] - lr, 0.1, 0.6
            )

        # 4. Als grote losing streaks → alles wat conservatiever
        max_streak = max(losing_streaks) if losing_streaks else 0
        if max_streak >= 4:
            self._params["confidence_bias"] = _clamp(
                self._params["confidence_bias"] + lr, -0.1, 0.15
            )
            self._params["risk_multiplier"] = _clamp(
                self._params["risk_multiplier"] + lr, 0.5, 2.0
            )

    # ------------------------------------------------------------------
    # Foutpatroon-analyse
    # ------------------------------------------------------------------

    def _analyze_error_patterns(self) -> List[str]:
        """Geeft kort leesbare patronen terug uit de foutlog."""
        if not self._error_log:
            return ["Nog geen fouten om te analyseren."]

        patterns: List[str] = []
        symbol_losses: Dict[str, float] = {}
        for err in self._error_log:
            sym = str(err["symbol"])
            symbol_losses[sym] = symbol_losses.get(sym, 0.0) + float(err["loss"])

        # Welke symbolen veroorzaken het meeste verlies?
        worst = sorted(symbol_losses.items(), key=lambda x: x[1])[:3]
        if worst and worst[0][1] < -5:
            symbols_str = ", ".join(f"{s} (€{v:.0f})" for s, v in worst)
            patterns.append(f"Meeste verlies bij: {symbols_str}")

        # Herhalende fouten op hoog-risico trades?
        high_risk = [e for e in self._error_log if float(e["risk_pct"]) > 1.8]
        if len(high_risk) > 3:
            patterns.append(f"Al {len(high_risk)}x verlies op hoog-risico trades; risico-drempel is verhoogd.")

        if not patterns:
            patterns.append("Geen duidelijke foutpatronen gevonden.")

        return patterns

    def _risk_level_label(self) -> str:
        rm = self._params["risk_multiplier"]
        cb = self._params["confidence_bias"]
        if rm > 1.3 or cb > 0.08:
            return "conservatief (leert van fouten)"
        elif rm < 0.7:
            return "agressief"
        return "normaal"

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        analyzed: int,
        win_rate: float,
        avg_pl: float,
        latest_signals: List[Signal],
        extra: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        avg_signal_risk = (
            sum(s.risk_pct for s in latest_signals) / len(latest_signals)
            if latest_signals
            else 1.0
        )
        suggested_risk = max(0.4, min(2.5, avg_signal_risk * self._params["risk_multiplier"]))

        result: Dict[str, object] = {
            "trades_analyzed": analyzed,
            "win_rate": round(win_rate, 4),
            "avg_profit_loss": round(avg_pl, 4),
            "suggested_risk_pct": round(suggested_risk, 3),
            "total_learning_iterations": self._total_iterations,
            "engine_params": {k: round(v, 4) for k, v in self._params.items()},
        }
        if extra:
            result.update(extra)
        return result


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
