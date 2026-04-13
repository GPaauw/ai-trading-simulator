"""Portfolio Manager: RL-agent (PPO) voor portfolio-allocatie beslissingen.

Laadt een voorgetraind Stable-Baselines3 PPO model en beslist:
- Welke signalen te kopen/verkopen
- Hoeveel te alloceren per positie
- Wanneer winst te nemen of stop-loss te activeren

Valt terug op regelgebaseerde logica als het RL model niet geladen kan worden.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ml.config import (
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
    MAX_POSITION_PCT,
    RL_MODEL_PATH,
    TOP_N_SIGNALS,
)

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Beslist over portfolio-allocatie via RL of regelgebaseerde fallback."""

    def __init__(self) -> None:
        self._model = None
        self._model_loaded = False
        self._load_model()

    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def decide(
        self,
        signals: List[Dict[str, Any]],
        holdings: List[Dict[str, Any]],
        cash: float,
        total_equity: float,
        start_balance: float,
        daily_pnl: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Genereer trading-beslissingen.

        Args:
            signals: Gesorteerde signalen van SignalPredictor.
            holdings: Huidige posities uit DataService.
            cash: Beschikbare cash.
            total_equity: Totale portfoliowaarde.
            start_balance: Startkapitaal.
            daily_pnl: Gerealiseerde P/L vandaag.

        Returns:
            Lijst van beslissingen: {symbol, action, fraction, amount, reason}
        """
        # ── Risico checks ───────────────────────────────────────────
        drawdown_pct = (start_balance - total_equity) / start_balance if start_balance > 0 else 0
        if drawdown_pct > MAX_DRAWDOWN_PCT:
            logger.warning("Max drawdown bereikt (%.1f%%). Trading gestopt.", drawdown_pct * 100)
            return []

        if daily_pnl < 0 and abs(daily_pnl) > total_equity * MAX_DAILY_LOSS_PCT:
            logger.warning("Dagelijks verlies limiet bereikt (€%.2f). Geen nieuwe trades.", daily_pnl)
            return []

        if self._model_loaded:
            return self._decide_rl(signals, holdings, cash, total_equity)
        return self._decide_rules(signals, holdings, cash, total_equity)

    # ── RL beslissing ───────────────────────────────────────────────

    def _decide_rl(
        self,
        signals: List[Dict[str, Any]],
        holdings: List[Dict[str, Any]],
        cash: float,
        total_equity: float,
    ) -> List[Dict[str, Any]]:
        """Gebruik het PPO model voor beslissingen."""
        obs = self._build_observation(signals, holdings, cash, total_equity)
        action, _ = self._model.predict(obs, deterministic=True)

        decisions = []
        top_signals = signals[:TOP_N_SIGNALS]
        held_symbols = {str(h["symbol"]).upper() for h in holdings}

        for i, sig in enumerate(top_signals):
            if i >= len(action):
                break
            alloc = float(action[i])
            symbol = sig["symbol"]

            if alloc > 0.05 and sig["action"] == "buy" and symbol not in held_symbols:
                # Koop
                amount = min(cash * alloc, total_equity * MAX_POSITION_PCT)
                if amount >= 10:
                    decisions.append({
                        "symbol": symbol,
                        "action": "buy",
                        "fraction": round(alloc, 4),
                        "amount": round(amount, 2),
                        "reason": f"RL: alloc={alloc:.2f}, conf={sig['confidence']:.2f}, exp_ret={sig['expected_return_pct']:.1f}%",
                    })

            elif alloc < -0.05 and symbol in held_symbols:
                # Verkoop
                decisions.append({
                    "symbol": symbol,
                    "action": "sell",
                    "fraction": min(1.0, abs(alloc)),
                    "amount": 0,
                    "reason": f"RL: alloc={alloc:.2f}, signaal={sig['action']}",
                })

        return decisions

    def _build_observation(
        self,
        signals: List[Dict[str, Any]],
        holdings: List[Dict[str, Any]],
        cash: float,
        total_equity: float,
    ) -> np.ndarray:
        """Bouw de observatie-vector voor het RL model."""
        obs = []

        # Portfolio state
        cash_ratio = cash / total_equity if total_equity > 0 else 1.0
        obs.append(cash_ratio)
        obs.append(1.0 - cash_ratio)  # invested ratio

        # Top-N signalen
        for i in range(TOP_N_SIGNALS):
            if i < len(signals):
                sig = signals[i]
                obs.append(sig.get("confidence", 0.5))
                obs.append(sig.get("expected_return_pct", 0) / 100.0)
                obs.append(sig.get("risk_score", 0.5))
            else:
                obs.extend([0.5, 0.0, 0.5])

        # Risk metrics (vereenvoudigd — worden berekend in auto_trader)
        obs.extend([0.0, 0.0, 0.0, 0.5])  # vol, drawdown, sharpe, win_rate

        # Tijd features
        from datetime import datetime
        now = datetime.now()
        obs.append(min(now.weekday(), 4) / 4.0)  # dag van de week
        obs.append(now.hour / 23.0)
        obs.append(0.5)  # market phase placeholder

        return np.array(obs, dtype=np.float32)

    # ── Regelgebaseerde fallback ────────────────────────────────────

    def _decide_rules(
        self,
        signals: List[Dict[str, Any]],
        holdings: List[Dict[str, Any]],
        cash: float,
        total_equity: float,
    ) -> List[Dict[str, Any]]:
        """Regelgebaseerde beslissingen als RL model niet beschikbaar is."""
        decisions = []
        held_symbols = {str(h["symbol"]).upper() for h in holdings}

        # ── VERKOOP: posities met winst >5% of verlies >3% ──────────
        for holding in holdings:
            symbol = str(holding["symbol"])
            avg_price = float(holding["avg_entry_price"])
            invested = float(holding["invested_amount"])
            if avg_price <= 0:
                continue

            # Zoek huidige prijs in signalen
            current_price = None
            for sig in signals:
                if sig["symbol"] == symbol:
                    # Signaal is sell → extra reden om te verkopen
                    if sig["action"] == "sell":
                        decisions.append({
                            "symbol": symbol,
                            "action": "sell",
                            "fraction": 1.0,
                            "amount": 0,
                            "reason": f"ML sell signaal (conf={sig['confidence']:.2f})",
                        })
                        break
            else:
                # Gebruik invested / qty als proxy
                pass

        # ── KOOP: top buy-signalen ──────────────────────────────────
        if cash >= 50:
            buy_signals = [s for s in signals if s["action"] == "buy" and s["symbol"] not in held_symbols]
            for sig in buy_signals[:3]:
                # Alloceer proportioneel aan confidence, max MAX_POSITION_PCT
                alloc = min(0.25, sig["confidence"] * 0.3)
                amount = min(cash * alloc, total_equity * MAX_POSITION_PCT)
                if amount >= 10:
                    decisions.append({
                        "symbol": sig["symbol"],
                        "action": "buy",
                        "fraction": round(alloc, 4),
                        "amount": round(amount, 2),
                        "reason": f"Top signaal: conf={sig['confidence']:.2f}, exp_ret={sig['expected_return_pct']:.1f}%",
                    })
                    cash -= amount

        return decisions

    # ── Model laden ─────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            if os.path.exists(RL_MODEL_PATH):
                from stable_baselines3 import PPO
                self._model = PPO.load(RL_MODEL_PATH)
                self._model_loaded = True
                logger.info("PPO RL model geladen: %s", RL_MODEL_PATH)
            else:
                logger.info("Geen RL model gevonden op %s — gebruik regelgebaseerde fallback.", RL_MODEL_PATH)
        except Exception as exc:
            logger.warning("Kon RL model niet laden: %s — gebruik fallback.", exc)
            self._model_loaded = False
