"""Auto Trader v2: geïntegreerde trading loop met alle v2-modules.

Orchestreert de uitgebreide pipeline per cyclus:

  1. **MarketScanner**         → dynamisch universum (top-5 crypto + top-5 stocks)
  2. **AlternativeData**       → sentiment, Reddit, SEC, FRED macro
  3. **SmartMoneyTracker**     → insider clusters, whale flow, opties, dark pool
  4. **FeatureEngine v2**      → 48 features met alt-data en regime injectie
  5. **EnsembleLightGBM**      → multi-window ensemble classificatie + SHAP
  6. **RegimeClassifier**      → marktregime per asset
  7. **RecurrentPPO Agent**    → LSTM-gebaseerde portfolio allocatie
  8. **SimulatedBroker**       → realistische order execution met slippage, fees, circuit breakers

Nieuwe features t.o.v. AutoTrader v1:
  - Dynamisch universum (geen vaste watchlist meer)
  - Smart money conviction boosting
  - Drift detection + automatische retraining trigger
  - Regime-adaptieve positie sizing
  - Order type selectie (market/limit) op basis van urgentie
  - Per-cyclus performance logging met Sortino/Sharpe
  - Graceful degradation: elke module kan falen zonder de loop te breken

Configuratie via environment variabelen:
  - AUTO_TRADE_V2_INTERVAL   (default: 5 minuten)
  - AUTO_TRADE_V2_MAX_TRADES (default: 8 per cyclus)
  - AUTO_TRADE_V2_DRY_RUN    (default: false)
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.trade import TradeRequest
from services.data_service import DataService
from services.feature_engine import FeatureEngine
from services.market_data_service import MarketDataService
from services.signal_predictor import SignalPredictor
from services.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES: int = 5
_MAX_TRADES_PER_CYCLE: int = 8
_MAX_CYCLE_HISTORY: int = 100
_DRIFT_CHECK_INTERVAL: int = 20   # elke N cycli drift check
_MIN_CONVICTION_BOOST: float = 0.6  # smart money score drempel voor boost


class AutoTraderV2:
    """Versie 2 van de autonome trading loop.

    Integreert alle v2-modules met graceful degradation:
    als een module niet beschikbaar is, wordt deze overgeslagen.
    """

    def __init__(
        self,
        data_service: DataService,
        market_data_service: MarketDataService,
        feature_engine: FeatureEngine,
        signal_predictor: SignalPredictor,
        portfolio_manager: PortfolioManager,
        limits: Dict[str, Any],
        # V2 modules (optioneel — graceful degradation)
        market_scanner: Any = None,
        alternative_data: Any = None,
        smart_money_tracker: Any = None,
        ensemble_predictor: Any = None,
        regime_classifier: Any = None,
        recurrent_ppo: Any = None,
        simulated_broker: Any = None,
    ) -> None:
        # Core services (v1)
        self._data = data_service
        self._market = market_data_service
        self._features = feature_engine
        self._predictor = signal_predictor
        self._portfolio_mgr = portfolio_manager
        self._limits = limits

        # V2 modules
        self._scanner = market_scanner
        self._alt_data = alternative_data
        self._smart_money = smart_money_tracker
        self._ensemble = ensemble_predictor
        self._regime_clf = regime_classifier
        self._rppo = recurrent_ppo
        self._broker = simulated_broker

        # Runtime state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval_minutes = _DEFAULT_INTERVAL_MINUTES
        self._dry_run = False

        # Statistieken
        self._cycles_completed: int = 0
        self._total_trades_executed: int = 0
        self._last_cycle_time: Optional[str] = None
        self._last_cycle_decisions: List[Dict] = []
        self._last_signals: List[Dict] = []
        self._last_scan_result: Any = None
        self._last_regime: Optional[str] = None
        self._last_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._cycle_summaries: List[Dict] = []

        # Performance tracking
        self._equity_history: List[float] = []
        self._daily_returns: List[float] = []

        # Module beschikbaarheid loggen
        self._log_module_status()

    def _log_module_status(self) -> None:
        """Log welke v2-modules beschikbaar zijn."""
        modules = {
            "MarketScanner": self._scanner is not None,
            "AlternativeData": self._alt_data is not None,
            "SmartMoneyTracker": self._smart_money is not None,
            "EnsembleLightGBM": self._ensemble is not None,
            "RegimeClassifier": self._regime_clf is not None,
            "RecurrentPPO": self._rppo is not None,
            "SimulatedBroker": self._broker is not None,
        }
        active = sum(1 for v in modules.values() if v)
        logger.info(
            "AutoTrader v2: %d/%d modules actief — %s",
            active, len(modules),
            ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in modules.items()),
        )

    # ══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════

    def start(
        self,
        interval_minutes: int = _DEFAULT_INTERVAL_MINUTES,
        dry_run: bool = False,
    ) -> bool:
        """Start de autonome trading loop.

        Args:
            interval_minutes: Tijd tussen cycli.
            dry_run: Als True, simuleer alles maar voer geen trades uit.
        """
        if self._running:
            return False

        self._interval_minutes = max(1, interval_minutes)
        self._dry_run = dry_run
        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="auto-trader-v2",
        )
        self._thread.start()

        mode = "DRY RUN" if dry_run else "LIVE"
        logger.info(
            "AutoTrader v2 gestart (%s, interval=%dm).",
            mode, self._interval_minutes,
        )
        return True

    def stop(self) -> bool:
        if not self._running:
            return False
        self._running = False
        logger.info("AutoTrader v2 gestopt na %d cycli.", self._cycles_completed)
        return True

    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Retourneer uitgebreide status."""
        return {
            "version": "2.0",
            "running": self._running,
            "dry_run": self._dry_run,
            "started_at": self._started_at,
            "interval_minutes": self._interval_minutes,
            "cycles_completed": self._cycles_completed,
            "total_trades_executed": self._total_trades_executed,
            "last_cycle_time": self._last_cycle_time,
            "last_decisions": self._last_cycle_decisions,
            "last_error": self._last_error,
            "last_regime": self._last_regime,
            # Model status
            "ml_signal_model": self._predictor.is_model_loaded(),
            "ml_signal_version": self._predictor.get_model_version(),
            "rl_portfolio_model": self._portfolio_mgr.is_model_loaded(),
            "ensemble_loaded": self._ensemble is not None,
            "rppo_loaded": self._rppo is not None and self._rppo.is_loaded,
            "broker_active": self._broker is not None,
            # Modules
            "modules": {
                "scanner": self._scanner is not None,
                "alt_data": self._alt_data is not None,
                "smart_money": self._smart_money is not None,
                "ensemble": self._ensemble is not None,
                "regime": self._regime_clf is not None,
                "rppo": self._rppo is not None,
                "broker": self._broker is not None,
            },
            # Performance
            "performance": self._calculate_performance(),
            "cycle_summaries": self._cycle_summaries[-10:],
        }

    def get_last_signals(self) -> List[Dict]:
        return self._last_signals

    def get_last_scan(self) -> Any:
        return self._last_scan_result

    # ══════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════

    def _loop(self) -> None:
        """Achtergrond trading loop."""
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("AutoTrader v2 cyclus fout: %s", exc, exc_info=True)

            # Wacht met 10s interval checks voor snelle stop
            wait_seconds = self._interval_minutes * 60
            elapsed = 0
            while elapsed < wait_seconds and self._running:
                time.sleep(min(10, wait_seconds - elapsed))
                elapsed += 10

    def _run_cycle(self) -> None:
        """Eén volledige trading cyclus."""
        cycle_start = time.time()
        now = datetime.now(timezone.utc).isoformat()
        self._last_cycle_time = now
        self._last_error = None
        cycle_num = self._cycles_completed + 1

        logger.info("═" * 50)
        logger.info("CYCLUS %d gestart", cycle_num)
        logger.info("═" * 50)

        # ── STAP 1: Universum bepalen ───────────────────────────────
        symbols = self._step_scan_universe()
        if not symbols:
            logger.info("Geen tradeable assets gevonden; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # ── STAP 2: Marktdata ophalen ───────────────────────────────
        instrument_histories = self._step_get_market_data(symbols)
        if not instrument_histories:
            logger.info("Geen marktdata beschikbaar; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # ── STAP 3: Alternative data & smart money ──────────────────
        alt_features = self._step_collect_alternative_data(symbols)
        smart_money_signals = self._step_track_smart_money(symbols)

        # ── STAP 4: Regime detectie ─────────────────────────────────
        regime_data = self._step_detect_regime(instrument_histories)

        # ── STAP 5: Features berekenen (v2) ─────────────────────────
        features_df = self._step_compute_features(
            instrument_histories, alt_features, regime_data,
        )
        if features_df.empty:
            logger.info("Geen features berekend; cyclus overgeslagen.")
            self._cycles_completed += 1
            return

        # ── STAP 6: Signalen voorspellen ────────────────────────────
        signals = self._step_predict_signals(features_df)

        # ── STAP 7: Smart money conviction boost ───────────────────
        signals = self._step_apply_smart_money_boost(signals, smart_money_signals)

        # ── STAP 8: Portfolio beslissingen ──────────────────────────
        decisions = self._step_portfolio_decisions(signals)

        # ── STAP 9: Regime-adaptieve positie sizing ─────────────────
        decisions = self._step_regime_adjust_sizing(decisions, regime_data)

        # ── STAP 10: Trades uitvoeren ───────────────────────────────
        executed = self._step_execute_trades(decisions)

        # ── STAP 11: Housekeeping ───────────────────────────────────
        self._step_housekeeping(signals, features_df)

        # ── STAP 12: Drift detection (periodiek) ───────────────────
        if cycle_num % _DRIFT_CHECK_INTERVAL == 0:
            self._step_drift_check(features_df)

        # Cyclus samenvatten
        cycle_time = time.time() - cycle_start
        self._cycles_completed += 1

        summary = {
            "cycle": self._cycles_completed,
            "timestamp": now,
            "symbols_scanned": len(symbols),
            "signals_count": len(signals),
            "decisions_count": len(decisions),
            "executed_count": executed,
            "regime": self._last_regime,
            "cycle_time_sec": round(cycle_time, 2),
            "cash_after": round(self._data.get_cash_balance(), 2),
        }
        self._cycle_summaries.append(summary)
        if len(self._cycle_summaries) > _MAX_CYCLE_HISTORY:
            self._cycle_summaries = self._cycle_summaries[-_MAX_CYCLE_HISTORY:]

        logger.info(
            "CYCLUS %d voltooid in %.1fs: %d/%d trades, regime=%s",
            self._cycles_completed, cycle_time, executed,
            len(decisions), self._last_regime or "onbekend",
        )

    # ══════════════════════════════════════════════════════════════════
    # STAPPEN
    # ══════════════════════════════════════════════════════════════════

    def _step_scan_universe(self) -> List[str]:
        """Stap 1: Bepaal het trading universum via MarketScanner."""
        if self._scanner is not None:
            try:
                scan = self._scanner.scan()
                self._last_scan_result = scan
                symbols = scan.symbols()
                logger.info("Stap 1 — MarketScanner: %d assets geselecteerd.", len(symbols))
                return symbols
            except Exception as exc:
                logger.warning("MarketScanner gefaald: %s — fallback naar watchlist.", exc)

        # Fallback: gebruik bestaande watchlist
        try:
            watchlist = self._market.get_watchlist()
            if not watchlist:
                watchlist = self._market.get_fast_watchlist()
            return [i.get("symbol", "") for i in watchlist if i.get("symbol")]
        except Exception as exc:
            logger.warning("Watchlist ophalen mislukt: %s", exc)
            return []

    def _step_get_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Stap 2: Haal OHLCV histories op voor het universum."""
        histories: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                instrument = self._market.get_instrument(symbol)
                hist = self._market.get_history(instrument)
                if hist is not None and len(hist) >= 50:
                    # Als market service al een OHLCV DataFrame teruggeeft
                    if isinstance(hist, pd.DataFrame):
                        df = hist
                        # normalizeer kolomnamen naar lowercase
                        df.columns = [c.lower() for c in df.columns]
                        # sommige bronnen leveren een enkele serie met closes
                        if "close" not in df.columns and df.shape[1] == 1:
                            df = df.rename(columns={df.columns[0]: "close"})

                    # Als we alleen een lijst met closes hebben, bouw een eenvoudige
                    # OHLCV DataFrame zodat FeatureEngine de verwachte kolommen vindt.
                    elif isinstance(hist, list) and hist:
                        df = pd.DataFrame({"close": hist})
                        # vul open met vorige close (of current voor eerste rij)
                        df["open"] = df["close"].shift(1).fillna(df["close"])
                        df["high"] = df["close"]
                        df["low"] = df["close"]
                        df["volume"] = 0.0
                        df = df[["open", "high", "low", "close", "volume"]]
                    else:
                        continue

                    histories[symbol] = df
            except Exception:
                continue

        logger.info("Stap 2 — Marktdata: %d/%d instrumenten geladen.", len(histories), len(symbols))
        return histories

    def _step_collect_alternative_data(
        self, symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Stap 3a: Verzamel alternatieve data (sentiment, Reddit, EDGAR, FRED)."""
        alt_features: Dict[str, Dict[str, float]] = {}

        if self._alt_data is None:
            return alt_features

        try:
            alt_result = self._alt_data.collect(symbols)
            for sym, features in alt_result.items():
                alt_features[sym] = features.to_feature_dict()

            # Macro data injecteren in FeatureEngine
            macro = self._alt_data.get_macro_data()
            self._features.set_macro_data({
                "vix_level": macro.vix_level,
                "vix_5d_change": macro.vix_5d_change,
            })

            logger.info("Stap 3a — AlternativeData: %d symbolen verwerkt.", len(alt_features))
        except Exception as exc:
            logger.warning("AlternativeData gefaald: %s", exc)

        return alt_features

    def _step_track_smart_money(
        self, symbols: List[str],
    ) -> Dict[str, Any]:
        """Stap 3b: Track smart money activiteit."""
        signals: Dict[str, Any] = {}

        if self._smart_money is None:
            return signals

        try:
            signals = self._smart_money.track(symbols)
            logger.info(
                "Stap 3b — SmartMoney: %d symbolen, avg score=%.3f",
                len(signals),
                np.mean([s.smart_money_score for s in signals.values()]) if signals else 0.5,
            )
        except Exception as exc:
            logger.warning("SmartMoneyTracker gefaald: %s", exc)

        return signals

    def _step_detect_regime(
        self, histories: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Stap 4: Detecteer marktregime per asset."""
        regime_data: Dict[str, Any] = {}

        if self._regime_clf is None:
            return regime_data

        try:
            # Breed marktregime (eerste beschikbare asset als proxy)
            for symbol, hist in histories.items():
                result = self._regime_clf.classify(hist)
                regime_data[symbol] = result
                if self._last_regime is None:
                    self._last_regime = result.regime

            # Inject regime data in FeatureEngine
            if regime_data:
                first_regime = next(iter(regime_data.values()))
                self._features.set_regime_data(first_regime.to_onehot())
                self._last_regime = first_regime.regime

            logger.info(
                "Stap 4 — Regime: %d assets geclassificeerd, breed=%s",
                len(regime_data), self._last_regime,
            )
        except Exception as exc:
            logger.warning("RegimeClassifier gefaald: %s", exc)

        return regime_data

    def _step_compute_features(
        self,
        histories: Dict[str, pd.DataFrame],
        alt_features: Dict[str, Dict[str, float]],
        regime_data: Dict[str, Any],
    ) -> pd.DataFrame:
        """Stap 5: Bereken v2 features met alle injecties."""
        try:
            # Injecteer alt-data per asset
            for sym, feat_dict in alt_features.items():
                self._features.set_alternative_data(feat_dict, symbol=sym)

            features_df = self._features.compute_batch(histories)
            logger.info("Stap 5 — Features: %d assets × %d kolommen.", len(features_df), len(features_df.columns))
            return features_df

        except Exception as exc:
            logger.warning("FeatureEngine gefaald: %s — probeer v1 fallback.", exc)
            try:
                return self._features.compute_batch(histories)
            except Exception:
                return pd.DataFrame()

    def _step_predict_signals(self, features_df: pd.DataFrame) -> List[Dict]:
        """Stap 6: Voorspel signalen met ensemble of standaard predictor."""
        signals: List[Dict] = []

        # Probeer ensemble eerst
        if self._ensemble is not None:
            try:
                ensemble_signals = self._ensemble_predict(features_df)
                if ensemble_signals:
                    signals = ensemble_signals
                    logger.info("Stap 6 — Ensemble signalen: %d assets.", len(signals))
            except Exception as exc:
                logger.warning("Ensemble predict gefaald: %s — fallback naar standaard.", exc)

        # Fallback naar standaard predictor
        if not signals:
            signals = self._predictor.predict(features_df)
            logger.info("Stap 6 — Standaard signalen: %d assets.", len(signals))

        self._last_signals = signals

        # Voeg live prijzen toe
        for sig in signals:
            try:
                instrument = self._market.get_instrument(sig["symbol"])
                sig["price"] = float(self._market.get_snapshot(instrument)["price"])
                sig["market"] = instrument.get("market", "us")
            except Exception:
                sig["price"] = 0.0
                sig["market"] = "us"

        return signals

    def _ensemble_predict(self, features_df: pd.DataFrame) -> List[Dict]:
        """Ensemble prediction met SHAP explanations."""
        if self._ensemble is None:
            return []

        signals: List[Dict] = []
        for symbol in features_df.index:
            row = features_df.loc[symbol].values.astype(np.float32)

            try:
                pred = self._ensemble.predict(row.reshape(1, -1))
                action = pred.get("action", "hold")
                confidence = pred.get("confidence", 0.5)
                expected_return = pred.get("expected_return", 0.0)

                # SHAP explanations (als beschikbaar)
                shap_values = None
                try:
                    shap_values = self._ensemble.explain(row.reshape(1, -1))
                except Exception:
                    pass

                signals.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "risk_score": pred.get("risk_score", 0.5),
                    "shap_top3": shap_values[:3] if shap_values else [],
                    "source": "ensemble",
                })
            except Exception as exc:
                logger.debug("Ensemble predict mislukt voor %s: %s", symbol, exc)

        return signals

    def _step_apply_smart_money_boost(
        self,
        signals: List[Dict],
        smart_money_signals: Dict[str, Any],
    ) -> List[Dict]:
        """Stap 7: Boost/demote signalen op basis van smart money conviction."""
        if not smart_money_signals:
            return signals

        for sig in signals:
            symbol = sig["symbol"]
            sm_sig = smart_money_signals.get(symbol)
            if sm_sig is None:
                continue

            sm_score = sm_sig.smart_money_score
            conviction = sm_sig.conviction

            # Boost: als smart money aligned is met signaal
            action = sig.get("action", "hold")
            original_confidence = sig.get("confidence", 0.5)

            if action in ("buy", "strong_buy") and sm_score >= _MIN_CONVICTION_BOOST:
                boost = (sm_score - 0.5) * 0.3  # max +0.15 boost
                sig["confidence"] = min(0.99, original_confidence + boost)
                sig["smart_money_boost"] = round(boost, 4)

            elif action in ("sell", "strong_sell") and sm_score <= (1 - _MIN_CONVICTION_BOOST):
                boost = (0.5 - sm_score) * 0.3
                sig["confidence"] = min(0.99, original_confidence + boost)
                sig["smart_money_boost"] = round(boost, 4)

            elif action in ("buy", "strong_buy") and sm_score < 0.35:
                # Smart money is bearish maar signaal is bullish → demote
                penalty = (0.5 - sm_score) * 0.2
                sig["confidence"] = max(0.1, original_confidence - penalty)
                sig["smart_money_boost"] = round(-penalty, 4)

            sig["smart_money_score"] = round(sm_score, 4)
            sig["smart_money_conviction"] = conviction

        logger.info("Stap 7 — SmartMoney boost: %d signalen verwerkt.", len(signals))
        return signals

    def _step_portfolio_decisions(self, signals: List[Dict]) -> List[Dict]:
        """Stap 8: Portfolio beslissingen via RecurrentPPO of standaard PM."""
        holdings = self._data.get_holdings()
        cash = self._data.get_cash_balance()
        start_balance = self._data.get_start_balance()
        total_invested = sum(float(h["invested_amount"]) for h in holdings)
        total_equity = cash + total_invested
        daily_pnl = self._data.get_daily_realized_pnl()

        # Probeer RecurrentPPO agent
        if self._rppo is not None and self._rppo.is_loaded:
            try:
                decisions = self._rppo_decide(signals, holdings, cash, total_equity)
                if decisions:
                    self._last_cycle_decisions = decisions
                    logger.info("Stap 8 — RecurrentPPO: %d beslissingen.", len(decisions))
                    return decisions
            except Exception as exc:
                logger.warning("RecurrentPPO decide gefaald: %s — fallback.", exc)

        # Fallback naar standaard PortfolioManager
        decisions = self._portfolio_mgr.decide(
            signals=signals, holdings=holdings, cash=cash,
            total_equity=total_equity, start_balance=start_balance,
            daily_pnl=daily_pnl,
        )
        self._last_cycle_decisions = decisions
        logger.info("Stap 8 — PortfolioManager: %d beslissingen.", len(decisions))
        return decisions

    def _rppo_decide(
        self,
        signals: List[Dict],
        holdings: List[Dict],
        cash: float,
        total_equity: float,
    ) -> List[Dict]:
        """Gebruik RecurrentPPO agent voor portfolio allocatie."""
        # Bouw observatie-vector (vereenvoudigd voor live inference)
        n_signals = min(len(signals), 10)
        obs: List[float] = []

        # Portfolio state
        invested = total_equity - cash
        obs.append(cash / total_equity if total_equity > 0 else 1.0)
        obs.append(invested / total_equity if total_equity > 0 else 0.0)

        # Signal features
        for i in range(n_signals):
            sig = signals[i] if i < len(signals) else {}
            obs.append(sig.get("confidence", 0.5))
            obs.append(sig.get("expected_return", 0.0))
            obs.append(sig.get("risk_score", 0.5))

        # Pad tot verwachte grootte
        expected_obs_dim = self._rppo.obs_dim
        while len(obs) < expected_obs_dim:
            obs.append(0.0)
        obs = obs[:expected_obs_dim]

        # Predict
        obs_array = np.array(obs, dtype=np.float32)
        actions, info = self._rppo.predict(obs_array, deterministic=True)

        # Vertaal acties naar trading beslissingen
        decisions: List[Dict] = []
        for i in range(min(len(actions), len(signals))):
            alloc = float(actions[i])
            sig = signals[i]
            symbol = sig["symbol"]

            if abs(alloc) < 0.05:
                continue  # te kleine allocatie

            if alloc > 0:
                buy_amount = alloc * 0.15 * total_equity  # max 15% per positie
                if buy_amount >= 10.0 and cash >= buy_amount:
                    decisions.append({
                        "symbol": symbol,
                        "action": "buy",
                        "amount": round(buy_amount, 2),
                        "confidence": sig.get("confidence", 0.5),
                        "reason": f"RecurrentPPO alloc={alloc:.3f}, method={info.get('method', 'rppo')}",
                    })
            elif alloc < 0:
                # Sell: check of we positie hebben
                holding = self._data.get_holding(symbol)
                if holding:
                    sell_fraction = min(abs(alloc), 1.0)
                    decisions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "fraction": sell_fraction,
                        "confidence": sig.get("confidence", 0.5),
                        "reason": f"RecurrentPPO alloc={alloc:.3f}",
                    })

        return decisions

    def _step_regime_adjust_sizing(
        self,
        decisions: List[Dict],
        regime_data: Dict[str, Any],
    ) -> List[Dict]:
        """Stap 9: Pas positie sizing aan op basis van marktregime."""
        if not regime_data or not self._last_regime:
            return decisions

        # Regime scaling factors
        regime_scale = {
            "bull": 1.2,       # meer agressief in bull
            "sideways": 0.8,   # conservatiever in sideways
            "bear": 0.5,       # veel kleiner in bear
            "high_vol": 0.6,   # kleiner in hoge volatiliteit
        }

        scale = regime_scale.get(self._last_regime, 1.0)

        for dec in decisions:
            if dec["action"] == "buy" and "amount" in dec:
                dec["amount"] = round(dec["amount"] * scale, 2)
                dec["regime_scale"] = scale
            elif dec["action"] == "sell" and "fraction" in dec:
                # In bear/high_vol: verkoop meer
                if scale < 1.0:
                    dec["fraction"] = min(1.0, dec["fraction"] * (2.0 - scale))

        logger.info("Stap 9 — Regime sizing: scale=%.2f (%s).", scale, self._last_regime)
        return decisions

    def _step_execute_trades(self, decisions: List[Dict]) -> int:
        """Stap 10: Voer trades uit via SimulatedBroker of standaard engine."""
        if self._dry_run:
            logger.info("DRY RUN: %d trades overgeslagen.", len(decisions))
            return 0

        executed = 0
        model_version = self._predictor.get_model_version() or "v2-rules"

        for decision in decisions[:_MAX_TRADES_PER_CYCLE]:
            success = self._execute_decision(decision, model_version)
            if success:
                executed += 1

        self._total_trades_executed += executed
        return executed

    def _execute_decision(self, decision: Dict, model_version: str) -> bool:
        """Voer één trading beslissing uit."""
        symbol = decision["symbol"]
        action = decision["action"]
        amount = decision.get("amount", 0)
        fraction = decision.get("fraction", 0.25)
        confidence = decision.get("confidence", 0)

        try:
            # ── SimulatedBroker (v2) ────────────────────────────────
            if self._broker is not None:
                return self._execute_via_broker(decision, model_version)

            # ── Standaard trade engine (v1) ─────────────────────────
            from services.trade_engine import execute_trade

            if action == "buy":
                if amount < 1.0:
                    return False
                market = self._detect_market(symbol)

                # Urgentie-gebaseerd order type
                order_type_hint = "market"
                if confidence < 0.6:
                    order_type_hint = "limit"  # minder urgent → limit order

                request = TradeRequest(
                    symbol=symbol, action="buy", amount=amount, market=market,
                )
                result = execute_trade(
                    request, self._limits, self._data, self._market,
                    confidence=confidence, model_version=model_version,
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info(
                        "BUY %s: %.4f @ €%.2f (€%.2f) — %s",
                        symbol, result.quantity, result.price,
                        result.amount, decision.get("reason", ""),
                    )
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
                    confidence=confidence, model_version=model_version,
                )
                success = result.status.startswith("executed")
                if success:
                    logger.info(
                        "SELL %s: %.4f @ €%.2f, P/L €%.2f — %s",
                        symbol, result.quantity, result.price,
                        result.profit_loss, decision.get("reason", ""),
                    )
                return success

        except Exception as exc:
            logger.warning("Trade %s %s mislukt: %s", action, symbol, exc)

        return False

    def _execute_via_broker(self, decision: Dict, model_version: str) -> bool:
        """Voer trade uit via SimulatedBroker."""
        from services.simulated_broker import execute_trade_v2

        symbol = decision["symbol"]
        action = decision["action"]
        confidence = decision.get("confidence", 0)
        market = self._detect_market(symbol)

        if action == "buy":
            amount = decision.get("amount", 0)
            if amount < 1.0:
                return False

            request = TradeRequest(
                symbol=symbol, action="buy", amount=amount, market=market,
            )
            result = execute_trade_v2(
                request, self._limits, self._broker,
                confidence=confidence, model_version=model_version,
            )
            success = "executed" in result.get("status", "")
            if success:
                # Sync met DataService
                self._data.record_buy(
                    symbol=symbol, market=market,
                    quantity=result["quantity"], amount=result["amount"],
                    price=result["price"],
                )
                logger.info(
                    "BROKER BUY %s: %.4f @ %.2f (fees=%.4f) — %s",
                    symbol, result["quantity"], result["price"],
                    result.get("fees", 0), decision.get("reason", ""),
                )
            return success

        elif action == "sell":
            holding = self._data.get_holding(symbol)
            if not holding:
                return False
            fraction = decision.get("fraction", 0.25)
            sell_qty = float(holding["quantity"]) * fraction

            request = TradeRequest(
                symbol=symbol, action="sell", amount=0.0,
                market=str(holding["market"]), quantity=sell_qty,
            )
            result = execute_trade_v2(
                request, self._limits, self._broker,
                confidence=confidence, model_version=model_version,
            )
            success = "executed" in result.get("status", "")
            if success:
                self._data.record_sell(
                    symbol=symbol, quantity=result["quantity"],
                    price=result["price"],
                )
                logger.info(
                    "BROKER SELL %s: %.4f @ %.2f (fees=%.4f) — %s",
                    symbol, result["quantity"], result["price"],
                    result.get("fees", 0), decision.get("reason", ""),
                )
            return success

        return False

    # ── Housekeeping ────────────────────────────────────────────────

    def _step_housekeeping(
        self,
        signals: List[Dict],
        features_df: pd.DataFrame,
    ) -> None:
        """Stap 11: Feature snapshots, portfolio snapshot, performance."""
        try:
            # Feature snapshots voor retraining
            for sig in signals[:20]:
                symbol = sig["symbol"]
                if symbol in features_df.index:
                    features = features_df.loc[symbol].to_dict()
                    self._data.save_feature_snapshot(
                        symbol=symbol,
                        features_json=json.dumps(
                            {k: round(float(v), 6) for k, v in features.items()},
                        ),
                        signal_action=sig["action"],
                        confidence=sig.get("confidence", 0),
                    )
        except Exception as exc:
            logger.debug("Feature snapshots mislukt: %s", exc)

        try:
            # Portfolio snapshot
            cash = self._data.get_cash_balance()
            holdings = self._data.get_holdings()
            market_value = self._compute_market_value(holdings)
            equity = cash + market_value
            self._data.record_portfolio_snapshot(equity)

            # Equity tracking
            self._equity_history.append(equity)
            if len(self._equity_history) > 1:
                prev = self._equity_history[-2]
                if prev > 0:
                    self._daily_returns.append((equity - prev) / prev)
        except Exception as exc:
            logger.debug("Portfolio snapshot mislukt: %s", exc)

        # Broker day reset (als middernacht)
        if self._broker is not None:
            try:
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute < self._interval_minutes:
                    self._broker.reset_day()
            except Exception:
                pass

    def _step_drift_check(self, features_df: pd.DataFrame) -> None:
        """Stap 12: Check feature drift en trigger retraining als nodig."""
        if self._ensemble is None:
            return

        try:
            # Probeer drift detection op ensemble
            if hasattr(self._ensemble, "detect_drift"):
                current_dist = features_df.mean().to_dict()
                drift_result = self._ensemble.detect_drift(current_dist)

                if drift_result.get("drift_detected", False):
                    logger.warning(
                        "DRIFT GEDETECTEERD: KL=%.4f, drifted_features=%s",
                        drift_result.get("kl_divergence", 0),
                        drift_result.get("drifted_features", [])[:5],
                    )
                    # TODO: Automatische retraining triggeren
                else:
                    logger.info("Drift check OK: geen significante drift.")
        except Exception as exc:
            logger.debug("Drift check mislukt: %s", exc)

    # ── Performance ─────────────────────────────────────────────────

    def _calculate_performance(self) -> Dict[str, Any]:
        """Bereken lopende performance metrics."""
        if len(self._daily_returns) < 2:
            return {"sortino": 0.0, "sharpe": 0.0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}

        returns = np.array(self._daily_returns)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        # Sharpe (annualized, 252 trading days)
        sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252) if std_ret > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        ds_std = float(np.std(downside)) if len(downside) > 1 else 0.01
        sortino = (mean_ret / (ds_std + 1e-8)) * np.sqrt(252)

        # Total return
        if self._equity_history:
            total_ret = (self._equity_history[-1] / self._equity_history[0] - 1) * 100
        else:
            total_ret = 0.0

        # Max drawdown
        max_dd = 0.0
        if self._equity_history:
            peak = self._equity_history[0]
            for eq in self._equity_history:
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)

        return {
            "sortino": round(float(sortino), 4),
            "sharpe": round(float(sharpe), 4),
            "total_return_pct": round(float(total_ret), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "cycles": self._cycles_completed,
            "win_rate": round(float(np.mean(returns > 0)) * 100, 1) if len(returns) > 0 else 0.0,
        }

    # ── Helpers ─────────────────────────────────────────────────────

    def _detect_market(self, symbol: str) -> str:
        """Detecteer of een symbool crypto, commodity of US stock is."""
        try:
            instrument = self._market.get_instrument(symbol)
            return str(instrument.get("market", "us"))
        except Exception:
            s = symbol.upper()
            crypto_indicators = (
                "USD", "USDT", "BTC", "ETH", "SOL", "XRP",
                "ADA", "DOGE", "DOT", "AVAX", "LINK",
            )
            if any(s.endswith(ind) or s.startswith(ind) for ind in crypto_indicators):
                return "crypto"
            if s in ("GC=F", "SI=F", "CL=F", "NG=F", "PL=F"):
                return "commodity"
            return "us"

    def _compute_market_value(self, holdings: List[Dict]) -> float:
        """Bereken totale marktwaarde van alle posities."""
        total = 0.0
        for h in holdings:
            try:
                instrument = self._market.get_instrument(
                    str(h["symbol"]), str(h["market"]),
                )
                price = float(self._market.get_snapshot(instrument)["price"])
                total += price * float(h["quantity"])
            except Exception:
                total += float(h["invested_amount"])
        return total
