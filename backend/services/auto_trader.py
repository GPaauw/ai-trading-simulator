"""Autonome auto-trader: handelt hypothetisch om het AI-model te laten leren.

Start met €2000 virtueel kapitaal. Draait als achtergrondtaak elke INTERVAL
minuten. Het AI-model (Groq/Llama 3.3 70B) beslist welke trades uitgevoerd worden.
Na elke ronde evalueert het model zijn eigen resultaten en past strategie aan.

De auto-trader gebruikt dezelfde DataService en MarketDataService als de rest
van de app, zodat portfolio en trade-historie consistent zijn.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models.trade import TradeRequest
from services.advice_engine import AdviceEngine
from services.data_service import DataService
from services.groq_client import GroqClient
from services.learning_agent import LearningAgent
from services.market_data_service import MarketDataService
from services.trade_engine import execute_trade

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 5
_MAX_TRADES_PER_CYCLE = 5
_MAX_CYCLE_HISTORY = 50


class AutoTrader:
    """Autonome trading-loop die op de achtergrond draait."""

    def __init__(
        self,
        data_service: DataService,
        market_data_service: MarketDataService,
        advice_engine: AdviceEngine,
        learning_agent: LearningAgent,
        limits: Dict[str, Any],
    ) -> None:
        self._data = data_service
        self._market = market_data_service
        self._advice = advice_engine
        self._learning = learning_agent
        self._limits = limits
        self._groq = GroqClient()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval_minutes = _DEFAULT_INTERVAL_MINUTES

        # Statistieken
        self._cycles_completed = 0
        self._total_trades_executed = 0
        self._last_cycle_time: Optional[str] = None
        self._last_cycle_decisions: List[Dict] = []
        self._last_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._cycle_summaries: List[Dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, interval_minutes: int = _DEFAULT_INTERVAL_MINUTES) -> bool:
        """Start de autonome trading-loop in een achtergrondthread."""
        if self._running:
            return False
        self._interval_minutes = max(1, interval_minutes)
        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-trader")
        self._thread.start()
        ai_mode = "Groq/Llama 3.3 70B" if self._groq.is_available() else "technisch (geen Groq)"
        logger.info("Auto-trader gestart (interval=%dm, modus=%s).", self._interval_minutes, ai_mode)
        return True

    def stop(self) -> bool:
        """Stop de autonome trading-loop."""
        if not self._running:
            return False
        self._running = False
        logger.info("Auto-trader gestopt na %d cycli.", self._cycles_completed)
        return True

    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Retourneer huidige status voor het dashboard."""
        cash = self._data.get_cash_balance()
        holdings = self._data.get_holdings()
        total_market_value = 0.0
        for h in holdings:
            try:
                instrument = self._market.get_instrument(str(h["symbol"]), str(h["market"]))
                price = float(self._market.get_snapshot(instrument)["price"])
                total_market_value += price * float(h["quantity"])
            except Exception:
                total_market_value += float(h["invested_amount"])

        total_equity = cash + total_market_value
        start_balance = float(self._data._get_setting("start_balance", "2000"))
        total_pnl = total_equity - start_balance
        pnl_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0

        return {
            "running": self._running,
            "started_at": self._started_at,
            "interval_minutes": self._interval_minutes,
            "cycles_completed": self._cycles_completed,
            "total_trades_executed": self._total_trades_executed,
            "last_cycle_time": self._last_cycle_time,
            "last_decisions": self._last_cycle_decisions,
            "last_error": self._last_error,
            "groq_available": self._groq.is_available(),
            "groq_rate_limited": self._groq.is_rate_limited(),
            "ai_mode": "groq" if self._groq.is_available() else "technical",
            "cycle_summaries": self._cycle_summaries[-10:],
            # Portfolio snapshot
            "cash": round(cash, 2),
            "market_value": round(total_market_value, 2),
            "total_equity": round(total_equity, 2),
            "start_balance": round(start_balance, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(pnl_pct, 2),
            "open_positions": len(holdings),
        }

    # ------------------------------------------------------------------
    # Hoofdloop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Achtergrondloop die elke interval een cycle draait."""
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Auto-trader cycle fout: %s", exc, exc_info=True)

            # Wacht interval (in stappen van 10s zodat stop snel reageert)
            wait_seconds = self._interval_minutes * 60
            elapsed = 0
            while elapsed < wait_seconds and self._running:
                time.sleep(min(10, wait_seconds - elapsed))
                elapsed += 10

    def _run_cycle(self) -> None:
        """Eén complete trading-cyclus: scan → AI beslissing → uitvoeren → leren."""
        now = datetime.now(timezone.utc).isoformat()
        self._last_cycle_time = now
        self._last_error = None

        logger.info("Auto-trader cyclus %d gestart.", self._cycles_completed + 1)

        # 1. Verzamel marktdata en signalen
        signals = self._get_signals()
        if not signals:
            logger.info("Geen signalen beschikbaar; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # 2. Haal huidige portfolio state
        holdings = self._data.get_holdings()
        cash = self._data.get_cash_balance()

        # 3. Bouw context voor AI
        trade_summary = self._build_trade_summary()
        perf_summary = self._build_performance_summary()

        # 4. Vraag AI om beslissingen
        if self._groq.is_available() and not self._groq.is_rate_limited():
            signal_dicts = [s.model_dump() for s in signals]
            decisions = self._groq.decide_trades(
                signals=signal_dicts,
                holdings=holdings,
                cash=cash,
                trade_history_summary=trade_summary,
                performance_summary=perf_summary,
            )
        else:
            decisions = self._decide_without_ai(signals, holdings, cash)

        self._last_cycle_decisions = decisions
        logger.info("AI gaf %d beslissingen.", len(decisions))

        # 5. Voer beslissingen uit
        executed = 0
        for decision in decisions[:_MAX_TRADES_PER_CYCLE]:
            success = self._execute_decision(decision, cash, holdings)
            if success:
                executed += 1
                # Herlaad cash na elke trade
                cash = self._data.get_cash_balance()
                holdings = self._data.get_holdings()

        self._total_trades_executed += executed

        # 6. Leer van resultaten
        self._learn_from_results()

        self._cycles_completed += 1
        logger.info(
            "Auto-trader cyclus %d voltooid: %d/%d trades uitgevoerd.",
            self._cycles_completed, executed, len(decisions),
        )
        cycle_summary = {
            "cycle": self._cycles_completed,
            "timestamp": now,
            "decisions_count": len(decisions),
            "executed_count": executed,
            "decisions": decisions[:5],
            "cash_after": round(self._data.get_cash_balance(), 2),
            "ai_mode": "groq" if self._groq.is_available() else "technical",
        }
        self._cycle_summaries.append(cycle_summary)
        if len(self._cycle_summaries) > _MAX_CYCLE_HISTORY:
            self._cycle_summaries = self._cycle_summaries[-_MAX_CYCLE_HISTORY:]

    # ------------------------------------------------------------------
    # Signalen ophalen
    # ------------------------------------------------------------------

    def _get_signals(self) -> list:
        """Haal technische signalen op via de advice engine."""
        try:
            if self._market.has_warm_stock_cache():
                return self._advice.build_ranked_buy_signals()
            else:
                self._market.ensure_background_prefetch_started()
                return self._advice.build_ranked_buy_signals(
                    watchlist=self._market.get_fast_watchlist(),
                )
        except Exception as exc:
            logger.warning("Kon signalen niet ophalen: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Trade uitvoering
    # ------------------------------------------------------------------

    def _execute_decision(
        self,
        decision: Dict,
        cash: float,
        holdings: List[Dict],
    ) -> bool:
        """Voer één AI-beslissing uit via de bestaande trade engine."""
        symbol = decision["symbol"]
        action = decision["action"]
        fraction = decision["fraction"]
        reason = decision.get("reason", "AI auto-trade")

        try:
            if action == "buy":
                amount = cash * fraction
                if amount < 1.0:
                    logger.info("Skip buy %s: bedrag te klein (€%.2f)", symbol, amount)
                    return False

                # Bepaal markt
                market = self._detect_market(symbol)
                request = TradeRequest(
                    symbol=symbol,
                    action="buy",
                    amount=amount,
                    market=market,
                )
                signal_map = self._advice.build_signal_map()
                result = execute_trade(
                    request, self._limits, self._data, self._market,
                    signal_map.get(symbol),
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info(
                        "AUTO BUY %s: %.4f stuks @ €%.2f (€%.2f) — %s",
                        symbol, result.quantity, result.price, result.amount, reason,
                    )
                else:
                    logger.info("AUTO BUY %s afgewezen: %s", symbol, result.status)
                return success

            elif action == "sell":
                holding = self._data.get_holding(symbol)
                if not holding:
                    logger.info("Skip sell %s: geen positie.", symbol)
                    return False

                sell_quantity = float(holding["quantity"]) * fraction
                if sell_quantity <= 0:
                    return False

                request = TradeRequest(
                    symbol=symbol,
                    action="sell",
                    amount=0.0,
                    market=str(holding["market"]),
                    quantity=sell_quantity,
                )
                result = execute_trade(
                    request, self._limits, self._data, self._market,
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info(
                        "AUTO SELL %s: %.4f stuks @ €%.2f, P/L €%.2f — %s",
                        symbol, result.quantity, result.price, result.profit_loss, reason,
                    )
                else:
                    logger.info("AUTO SELL %s afgewezen: %s", symbol, result.status)
                return success

        except Exception as exc:
            logger.warning("Auto-trade %s %s mislukt: %s", action, symbol, exc)
            return False

        return False

    def _decide_without_ai(self, signals: list, holdings: List[Dict], cash: float) -> List[Dict]:
        """Eenvoudige rule-based beslissingen als Groq niet beschikbaar is."""
        decisions: List[Dict] = []

        # Verkoop posities met >5% winst of <-3% verlies (stop-loss)
        for holding in holdings:
            symbol = str(holding["symbol"])
            market = str(holding["market"])
            try:
                instrument = self._market.get_instrument(symbol, market)
                price = float(self._market.get_snapshot(instrument)["price"])
                avg_price = float(holding["avg_entry_price"])
                if avg_price <= 0:
                    continue
                pnl_pct = (price - avg_price) / avg_price * 100
                if pnl_pct >= 5.0:
                    decisions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "fraction": 1.0,
                        "reason": f"Winst genomen: +{pnl_pct:.1f}%",
                    })
                elif pnl_pct <= -3.0:
                    decisions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "fraction": 1.0,
                        "reason": f"Stop-loss geraakt: {pnl_pct:.1f}%",
                    })
            except Exception:
                continue

        # Koop top-signalen als er genoeg cash is
        if cash >= 50:
            held_symbols = {str(h["symbol"]).upper() for h in holdings}
            buy_signals = sorted(
                [s for s in signals if hasattr(s, "action") and s.action == "buy"],
                key=lambda s: getattr(s, "ranking_score", 0),
                reverse=True,
            )
            for signal in buy_signals[:2]:
                symbol = getattr(signal, "symbol", "").upper()
                if not symbol or symbol in held_symbols:
                    continue
                decisions.append({
                    "symbol": symbol,
                    "action": "buy",
                    "fraction": 0.25,
                    "reason": f"Top technisch signaal: {getattr(signal, 'reason', 'n/a')}",
                })

        return decisions

    def _detect_market(self, symbol: str) -> str:
        """Probeer het juiste markttype te detecteren voor een symbool."""
        try:
            instrument = self._market.get_instrument(symbol)
            return str(instrument.get("market", "us"))
        except Exception:
            # Heuristiek
            symbol_upper = symbol.upper()
            if symbol_upper.endswith("USD") or symbol_upper in (
                "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
                "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
            ):
                return "crypto"
            if symbol_upper in ("GC=F", "SI=F", "CL=F", "NG=F", "PL=F"):
                return "commodity"
            return "us"

    # ------------------------------------------------------------------
    # Leren
    # ------------------------------------------------------------------

    def _learn_from_results(self) -> None:
        """Laat de learning agent leren van de hele trade-historie."""
        try:
            history = self._data.get_trade_history()
            signals = self._advice.build_signals()
            self._learning.learn(history, signals)
            self._advice.apply_learned_params(self._learning.get_engine_params())
        except Exception as exc:
            logger.warning("Learning-stap mislukt: %s", exc)

    def _build_trade_summary(self) -> str:
        """Bouw een korte samenvatting van recente trades voor de AI prompt."""
        history = self._data.get_trade_history()
        if not history:
            return "Nog geen trades uitgevoerd."

        recent = history[-20:]
        sells = [t for t in recent if t.action == "sell" and t.status.startswith("executed")]
        buys = [t for t in recent if t.action == "buy" and t.status.startswith("executed")]

        total_pl = sum(t.profit_loss for t in sells)
        winners = sum(1 for t in sells if t.profit_loss > 0)
        losers = sum(1 for t in sells if t.profit_loss < 0)

        parts = [
            f"Laatste {len(recent)} trades: {len(buys)} kopen, {len(sells)} verkopen.",
            f"Resultaat verkopen: {winners} winst, {losers} verlies, netto €{total_pl:.2f}.",
        ]

        # Symbolen met verlies
        loss_syms: Dict[str, float] = {}
        for t in sells:
            if t.profit_loss < 0:
                loss_syms[t.symbol] = loss_syms.get(t.symbol, 0) + t.profit_loss
        if loss_syms:
            worst = sorted(loss_syms.items(), key=lambda x: x[1])[:3]
            parts.append(f"Verliezers: {', '.join(f'{s} (€{v:.2f})' for s, v in worst)}.")

        # Symbolen met winst
        win_syms: Dict[str, float] = {}
        for t in sells:
            if t.profit_loss > 0:
                win_syms[t.symbol] = win_syms.get(t.symbol, 0) + t.profit_loss
        if win_syms:
            best = sorted(win_syms.items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"Winnaars: {', '.join(f'{s} (+€{v:.2f})' for s, v in best)}.")

        return " ".join(parts)

    def _build_performance_summary(self) -> str:
        """Bouw een snapshot van de huidige portfolio performance."""
        cash = self._data.get_cash_balance()
        holdings = self._data.get_holdings()
        start_balance = float(self._data._get_setting("start_balance", "2000"))

        total_invested = sum(float(h.get("invested_amount", 0)) for h in holdings)
        total_equity = cash + total_invested  # vereenvoudigd (invested ≈ market value)
        pnl = total_equity - start_balance
        pnl_pct = (pnl / start_balance * 100) if start_balance > 0 else 0

        return (
            f"Startkapitaal: €{start_balance:.2f} | "
            f"Cash: €{cash:.2f} | "
            f"Geïnvesteerd: €{total_invested:.2f} | "
            f"Totaal: €{total_equity:.2f} | "
            f"P/L: €{pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"Open posities: {len(holdings)} | "
            f"Cyclus: {self._cycles_completed + 1}"
        )
