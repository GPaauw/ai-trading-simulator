"""Auto Trader: ML-gestuurde autonome trading loop.

Orchestreert de volledige pipeline per cyclus:
1. Marktdata ophalen (MarketDataService)
2. Features berekenen (FeatureEngine)
3. Signalen voorspellen (SignalPredictor)
4. Portfolio beslissingen (PortfolioManager)
5. Trades uitvoeren (TradeEngine)
6. Feature snapshots opslaan voor retraining
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from models.trade import TradeRequest
from services.data_service import DataService
from services.feature_engine import FeatureEngine
from services.market_data_service import MarketDataService
from services.portfolio_manager import PortfolioManager
from services.signal_predictor import SignalPredictor
from services.trade_engine import execute_trade

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 5
_MAX_TRADES_PER_CYCLE = 5
_MAX_CYCLE_HISTORY = 50


class AutoTrader:
    def __init__(
        self,
        data_service: DataService,
        market_data_service: MarketDataService,
        feature_engine: FeatureEngine,
        signal_predictor: SignalPredictor,
        portfolio_manager: PortfolioManager,
        limits: Dict[str, Any],
    ) -> None:
        self._data = data_service
        self._market = market_data_service
        self._features = feature_engine
        self._predictor = signal_predictor
        self._portfolio_mgr = portfolio_manager
        self._limits = limits

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval_minutes = _DEFAULT_INTERVAL_MINUTES

        self._cycles_completed = 0
        self._total_trades_executed = 0
        self._last_cycle_time: Optional[str] = None
        self._last_cycle_decisions: List[Dict] = []
        self._last_signals: List[Dict] = []
        self._last_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._cycle_summaries: List[Dict] = []

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self, interval_minutes: int = _DEFAULT_INTERVAL_MINUTES) -> bool:
        if self._running:
            return False
        self._interval_minutes = max(1, interval_minutes)
        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-trader")
        self._thread.start()
        ml_mode = "ML" if self._predictor.is_model_loaded() else "rules-fallback"
        rl_mode = "RL" if self._portfolio_mgr.is_model_loaded() else "rules-fallback"
        logger.info("Auto-trader gestart (interval=%dm, signals=%s, portfolio=%s).",
                     self._interval_minutes, ml_mode, rl_mode)
        return True

    def stop(self) -> bool:
        if not self._running:
            return False
        self._running = False
        logger.info("Auto-trader gestopt na %d cycli.", self._cycles_completed)
        return True

    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "started_at": self._started_at,
            "interval_minutes": self._interval_minutes,
            "cycles_completed": self._cycles_completed,
            "total_trades_executed": self._total_trades_executed,
            "last_cycle_time": self._last_cycle_time,
            "last_decisions": self._last_cycle_decisions,
            "last_error": self._last_error,
            "ml_signal_model": self._predictor.is_model_loaded(),
            "ml_signal_version": self._predictor.get_model_version(),
            "rl_portfolio_model": self._portfolio_mgr.is_model_loaded(),
            "cycle_summaries": self._cycle_summaries[-10:],
        }

    def get_last_signals(self) -> List[Dict]:
        return self._last_signals

    # ── Hoofdloop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Auto-trader cycle fout: %s", exc, exc_info=True)

            wait_seconds = self._interval_minutes * 60
            elapsed = 0
            while elapsed < wait_seconds and self._running:
                time.sleep(min(10, wait_seconds - elapsed))
                elapsed += 10

    def _run_cycle(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._last_cycle_time = now
        self._last_error = None

        logger.info("Cyclus %d gestart.", self._cycles_completed + 1)

        # 1. Marktdata → histories ophalen
        instrument_histories = self._get_instrument_histories()
        if not instrument_histories:
            logger.info("Geen marktdata beschikbaar; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # 2. Features berekenen
        features_df = self._features.compute_batch(instrument_histories)
        if features_df.empty:
            logger.info("Geen features berekend; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # 3. Signalen voorspellen
        signals = self._predictor.predict(features_df)
        self._last_signals = signals

        # Voeg live prijzen toe aan signalen
        for sig in signals:
            try:
                instrument = self._market.get_instrument(sig["symbol"])
                sig["price"] = float(self._market.get_snapshot(instrument)["price"])
                sig["market"] = instrument.get("market", "us")
            except Exception:
                sig["price"] = 0.0
                sig["market"] = "us"

        # 4. Portfolio beslissingen
        holdings = self._data.get_holdings()
        cash = self._data.get_cash_balance()
        start_balance = self._data.get_start_balance()
        total_invested = sum(float(h["invested_amount"]) for h in holdings)
        total_equity = cash + total_invested
        daily_pnl = self._data.get_daily_realized_pnl()

        decisions = self._portfolio_mgr.decide(
            signals=signals, holdings=holdings, cash=cash,
            total_equity=total_equity, start_balance=start_balance, daily_pnl=daily_pnl,
        )
        self._last_cycle_decisions = decisions
        logger.info("Portfolio manager gaf %d beslissingen.", len(decisions))

        # 5. Trades uitvoeren
        executed = 0
        model_version = self._predictor.get_model_version() or "rules"
        for decision in decisions[:_MAX_TRADES_PER_CYCLE]:
            success = self._execute_decision(decision, model_version)
            if success:
                executed += 1

        self._total_trades_executed += executed

        # 6. Feature snapshots opslaan voor retraining
        self._save_feature_snapshots(signals, features_df)

        # 7. Portfolio snapshot
        cash = self._data.get_cash_balance()
        holdings = self._data.get_holdings()
        market_value = self._compute_market_value(holdings)
        self._data.record_portfolio_snapshot(cash + market_value)

        # 8. Model performance metrics
        self._update_performance_metrics()

        self._cycles_completed += 1
        logger.info("Cyclus %d voltooid: %d/%d trades.", self._cycles_completed, executed, len(decisions))

        self._cycle_summaries.append({
            "cycle": self._cycles_completed,
            "timestamp": now,
            "decisions_count": len(decisions),
            "executed_count": executed,
            "signals_count": len(signals),
            "cash_after": round(self._data.get_cash_balance(), 2),
        })
        if len(self._cycle_summaries) > _MAX_CYCLE_HISTORY:
            self._cycle_summaries = self._cycle_summaries[-_MAX_CYCLE_HISTORY:]

    # ── Marktdata ophalen ───────────────────────────────────────────

    def _get_instrument_histories(self) -> Dict[str, pd.DataFrame]:
        """Haal OHLCV histories op voor alle instrumenten in de watchlist."""
        histories: Dict[str, pd.DataFrame] = {}
        try:
            watchlist = self._market.get_watchlist()
            if not watchlist:
                watchlist = self._market.get_fast_watchlist()

            for instrument in watchlist:
                symbol = instrument.get("symbol", "")
                try:
                    hist = self._market.get_history(instrument)
                    if hist is not None and len(hist) >= 50:
                        # Converteer naar DataFrame als het een dict/list is
                        if isinstance(hist, pd.DataFrame):
                            histories[symbol] = hist
                        elif isinstance(hist, list) and hist:
                            histories[symbol] = pd.DataFrame(hist)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Kon watchlist niet ophalen: %s", exc)
        return histories

    # ── Trade uitvoering ────────────────────────────────────────────

    def _execute_decision(self, decision: Dict, model_version: str) -> bool:
        symbol = decision["symbol"]
        action = decision["action"]
        amount = decision.get("amount", 0)
        fraction = decision.get("fraction", 0.25)

        try:
            if action == "buy":
                if amount < 1.0:
                    return False
                market = self._detect_market(symbol)
                request = TradeRequest(symbol=symbol, action="buy", amount=amount, market=market)
                result = execute_trade(
                    request, self._limits, self._data, self._market,
                    confidence=decision.get("confidence", 0), model_version=model_version,
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info("BUY %s: %.4f @ €%.2f (€%.2f) — %s",
                                symbol, result.quantity, result.price, result.amount, decision.get("reason", ""))
                return success

            elif action == "sell":
                holding = self._data.get_holding(symbol)
                if not holding:
                    return False
                sell_qty = float(holding["quantity"]) * fraction
                if sell_qty <= 0:
                    return False
                request = TradeRequest(
                    symbol=symbol, action="sell", amount=0.0,
                    market=str(holding["market"]), quantity=sell_qty,
                )
                result = execute_trade(
                    request, self._limits, self._data, self._market,
                    confidence=decision.get("confidence", 0), model_version=model_version,
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info("SELL %s: %.4f @ €%.2f, P/L €%.2f — %s",
                                symbol, result.quantity, result.price, result.profit_loss, decision.get("reason", ""))
                return success

        except Exception as exc:
            logger.warning("Trade %s %s mislukt: %s", action, symbol, exc)
            return False
        return False

    def _detect_market(self, symbol: str) -> str:
        try:
            instrument = self._market.get_instrument(symbol)
            return str(instrument.get("market", "us"))
        except Exception:
            s = symbol.upper()
            if s.endswith("USD") or s in ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
                                          "ADA-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"):
                return "crypto"
            if s in ("GC=F", "SI=F", "CL=F", "NG=F", "PL=F"):
                return "commodity"
            return "us"

    # ── Feature snapshots ───────────────────────────────────────────

    def _save_feature_snapshots(self, signals: List[Dict], features_df: pd.DataFrame) -> None:
        """Sla feature snapshots op voor top signalen (voor retraining)."""
        try:
            for sig in signals[:20]:
                symbol = sig["symbol"]
                if symbol in features_df.index:
                    features = features_df.loc[symbol].to_dict()
                    self._data.save_feature_snapshot(
                        symbol=symbol,
                        features_json=json.dumps({k: round(float(v), 6) for k, v in features.items()}),
                        signal_action=sig["action"],
                        confidence=sig.get("confidence", 0),
                    )
        except Exception as exc:
            logger.warning("Feature snapshot opslaan mislukt: %s", exc)

    # ── Performance metrics ─────────────────────────────────────────

    def _update_performance_metrics(self) -> None:
        try:
            history = self._data.get_trade_history()
            sells = [t for t in history if t.action == "sell" and t.status.startswith("executed")]
            if not sells:
                return

            wins = sum(1 for t in sells if t.profit_loss > 0)
            total_pl = sum(t.profit_loss for t in sells)
            win_rate = wins / len(sells) if sells else 0

            self._data.save_model_performance({
                "win_rate": round(win_rate, 4),
                "total_trades": len(sells),
                "total_pnl": round(total_pl, 2),
                "avg_pnl": round(total_pl / len(sells), 2) if sells else 0,
                "signal_model_loaded": self._predictor.is_model_loaded(),
                "rl_model_loaded": self._portfolio_mgr.is_model_loaded(),
            })
        except Exception as exc:
            logger.warning("Performance metrics opslaan mislukt: %s", exc)

    # ── Helpers ─────────────────────────────────────────────────────

    def _compute_market_value(self, holdings: List[Dict]) -> float:
        total = 0.0
        for h in holdings:
            try:
                instrument = self._market.get_instrument(str(h["symbol"]), str(h["market"]))
                price = float(self._market.get_snapshot(instrument)["price"])
                total += price * float(h["quantity"])
            except Exception:
                total += float(h["invested_amount"])
        return total
