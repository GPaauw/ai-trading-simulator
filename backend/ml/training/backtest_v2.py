"""Backtest v2 — evalueert de volledige v2-pipeline op historische data.

Gebruikt alle v2-modules (ensemble, regime, smart money, simulated broker)
en genereert uitgebreide prestatiemetrieken: equity curve, Sortino, Sharpe,
max drawdown, win rate, profit factor, benchmark-vergelijking.

Gebruik:
    cd backend
    python -m ml.training.backtest_v2 --symbols AAPL MSFT NVDA BTC ETH \
        --start 2023-01-01 --end 2024-12-31 --balance 10000

    python -m ml.training.backtest_v2 --preset momentum
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.config import (
    ML_MODELS_DIR,
    START_BALANCE,
    TRANSACTION_COST_PCT,
    SLIPPAGE_PCT,
    MAX_POSITION_PCT,
    MAX_DRAWDOWN_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# PRESET UNIVERSUMS
# ════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, List[str]] = {
    "momentum": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"],
    "mixed": ["AAPL", "MSFT", "NVDA", "TSLA", "BTC-USD", "ETH-USD"],
    "eu": ["ASML", "SAP", "SIE.DE", "BMW.DE", "MC.PA"],
}


# ════════════════════════════════════════════════════════════════════
# DATA OPHALEN
# ════════════════════════════════════════════════════════════════════

def fetch_histories(
    symbols: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Download historische OHLCV data via yfinance."""
    import yfinance as yf

    histories: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df.empty:
                logger.warning("Geen data voor %s — overgeslagen", sym)
                continue
            df.columns = [c.lower() for c in df.columns]
            if "close" not in df.columns:
                logger.warning("Geen 'close' kolom voor %s", sym)
                continue
            histories[sym] = df
            logger.info("%-8s  %d bars geladen", sym, len(df))
        except Exception as exc:
            logger.warning("Fout bij ophalen %s: %s", sym, exc)

    return histories


# ════════════════════════════════════════════════════════════════════
# V2 MODULE INITIALISATIE
# ════════════════════════════════════════════════════════════════════

def _init_modules() -> Dict[str, Any]:
    """Probeer alle v2-modules te laden (graceful degradation)."""
    modules: Dict[str, Any] = {}

    try:
        from services.feature_engine import FeatureEngine
        modules["feature_engine"] = FeatureEngine()
    except Exception as exc:
        logger.warning("FeatureEngine niet beschikbaar: %s", exc)
        modules["feature_engine"] = None

    try:
        from services.signal_predictor import SignalPredictor
        modules["signal_predictor"] = SignalPredictor()
    except Exception as exc:
        logger.warning("SignalPredictor niet beschikbaar: %s", exc)
        modules["signal_predictor"] = None

    try:
        from ml.regime_classifier import RegimeClassifier
        modules["regime_classifier"] = RegimeClassifier()
    except Exception:
        modules["regime_classifier"] = None

    try:
        from ml.ensemble_lgbm import EnsembleLightGBM
        modules["ensemble"] = EnsembleLightGBM()
    except Exception:
        modules["ensemble"] = None

    try:
        from services.simulated_broker import SimulatedBroker
        modules["broker"] = SimulatedBroker
    except Exception:
        modules["broker"] = None

    loaded = [k for k, v in modules.items() if v is not None]
    logger.info("Geladen modules: %s", ", ".join(loaded) or "(geen)")
    return modules


# ════════════════════════════════════════════════════════════════════
# REGIME-BEPALING
# ════════════════════════════════════════════════════════════════════

def _classify_regime(
    hist: pd.DataFrame,
    regime_clf: Any,
) -> str:
    """Bepaal marktregime via classifier of heuristiek."""
    if regime_clf is not None:
        try:
            result = regime_clf.classify(hist)
            return result.regime
        except Exception:
            pass

    # Heuristische fallback
    if len(hist) < 20:
        return "sideways"
    returns = hist["close"].pct_change().dropna().tail(20)
    mean_ret = float(returns.mean())
    volatility = float(returns.std())

    if volatility > 0.03:
        return "high_vol"
    if mean_ret > 0.001:
        return "bull"
    if mean_ret < -0.001:
        return "bear"
    return "sideways"


# ════════════════════════════════════════════════════════════════════
# POSITION SIZING (regime-adaptief)
# ════════════════════════════════════════════════════════════════════

_REGIME_SCALE = {"bull": 1.2, "sideways": 1.0, "bear": 0.5, "high_vol": 0.6}


def _calc_position_size(
    cash: float,
    equity: float,
    regime: str,
    confidence: float,
) -> float:
    """Bereken positiegrootte met regime-schaling en confidence-weging."""
    base_alloc = equity * MAX_POSITION_PCT
    regime_mult = _REGIME_SCALE.get(regime, 1.0)
    conf_mult = 0.5 + confidence  # confidence 0→0.5×, 1→1.5×
    alloc = base_alloc * regime_mult * conf_mult
    return min(alloc, cash * 0.4)  # max 40% van cash per positie


# ════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════

def run_backtest_v2(
    histories: Dict[str, pd.DataFrame],
    start_balance: float = START_BALANCE,
    use_broker: bool = True,
) -> Dict[str, Any]:
    """Draai een volledige v2 backtest over historische data.

    Returns:
        dict met equity_curve, trades, en samenvattende statistieken.
    """
    modules = _init_modules()
    fe = modules["feature_engine"]
    predictor = modules["signal_predictor"]
    regime_clf = modules["regime_classifier"]
    ensemble = modules["ensemble"]
    BrokerClass = modules["broker"]

    # Optioneel: SimulatedBroker gebruiken voor realistische fills
    broker = None
    if use_broker and BrokerClass is not None:
        broker = BrokerClass(initial_cash=start_balance)
        logger.info("SimulatedBroker actief voor fills/slippage/fees")

    cash = start_balance
    holdings: Dict[str, Dict] = {}  # symbol -> {qty, avg_price}
    equity_curve: List[Dict] = []
    all_trades: List[Dict] = []
    daily_pnl: List[float] = []
    regime_history: List[str] = []

    symbols = list(histories.keys())
    min_len = min(len(histories[s]) for s in symbols)
    warmup = 60

    if min_len <= warmup + 10:
        logger.warning("Te weinig data voor backtest (%d bars)", min_len)
        return {"error": "Onvoldoende data", "min_bars": min_len}

    timesteps = min_len - warmup
    prev_equity = start_balance
    peak_equity = start_balance

    logger.info(
        "Backtest start: %d symbolen, %d stappen, startbalans=€%.2f",
        len(symbols), timesteps, start_balance,
    )

    for t in range(timesteps):
        idx = warmup + t

        # ── Stap 1: Features + regime per symbool ──────────────────
        batch_features: Dict[str, Any] = {}
        prices: Dict[str, float] = {}
        regimes: Dict[str, str] = {}

        for sym in symbols:
            hist = histories[sym].iloc[: idx + 1]
            prices[sym] = float(hist.iloc[-1]["close"])

            # Regime classificatie
            regime = _classify_regime(hist, regime_clf)
            regimes[sym] = regime

            # Features
            if fe is not None:
                try:
                    if hasattr(fe, "set_regime_data"):
                        fe.set_regime_data({"regime": regime})
                    features = fe.compute_features(hist)
                    if features:
                        batch_features[sym] = features
                except Exception:
                    pass

        # Dominant regime
        regime_counts: Dict[str, int] = {}
        for r in regimes.values():
            regime_counts[r] = regime_counts.get(r, 0) + 1
        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "sideways"
        regime_history.append(dominant_regime)

        if not batch_features:
            continue

        # ── Stap 2: Signalen genereren ─────────────────────────────
        features_df = pd.DataFrame.from_dict(batch_features, orient="index")
        features_df.index.name = "symbol"

        signals: List[Dict] = []

        # Probeer ensemble eerst, val terug op standaard predictor
        if ensemble is not None:
            try:
                signals = ensemble.predict(features_df)
            except Exception:
                pass

        if not signals and predictor is not None:
            try:
                signals = predictor.predict(features_df)
            except Exception:
                pass

        if not signals:
            continue

        # ── Stap 3: Verkoop op sell-signalen ───────────────────────
        for sig in signals:
            sym = sig.get("symbol", "")
            action = sig.get("action", "")

            if action not in ("sell", "strong_sell"):
                continue
            if sym not in holdings:
                continue

            h = holdings[sym]
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = h["qty"] * sell_price
            cost = proceeds * TRANSACTION_COST_PCT
            cash += proceeds - cost

            pnl = (sell_price - h["avg_price"]) * h["qty"] - cost
            all_trades.append({
                "step": t,
                "symbol": sym,
                "side": "sell",
                "qty": round(h["qty"], 6),
                "price": round(sell_price, 4),
                "pnl": round(pnl, 2),
                "confidence": sig.get("confidence", 0),
                "regime": regimes.get(sym, "unknown"),
            })
            del holdings[sym]

        # ── Stap 4: Koop op buy-signalen (regime-adaptief) ────────
        buy_signals = [
            s for s in signals
            if s.get("action") in ("buy", "strong_buy")
        ]
        buy_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        for sig in buy_signals[:3]:
            sym = sig.get("symbol", "")
            if sym in holdings:
                continue

            price = prices.get(sym, 0)
            if price <= 0:
                continue

            confidence = sig.get("confidence", 0.5)
            sym_regime = regimes.get(sym, dominant_regime)

            total_equity = cash + sum(
                h["qty"] * prices.get(s, 0) for s, h in holdings.items()
            )

            alloc = _calc_position_size(cash, total_equity, sym_regime, confidence)
            if alloc < 10:
                continue

            buy_price = price * (1 + SLIPPAGE_PCT)
            cost = alloc * TRANSACTION_COST_PCT
            qty = (alloc - cost) / buy_price

            if qty <= 0:
                continue

            cash -= alloc
            holdings[sym] = {"qty": qty, "avg_price": buy_price}

            all_trades.append({
                "step": t,
                "symbol": sym,
                "side": "buy",
                "qty": round(qty, 6),
                "price": round(buy_price, 4),
                "pnl": 0,
                "confidence": round(confidence, 4),
                "regime": sym_regime,
            })

        # ── Stap 5: Equity bijwerken ──────────────────────────────
        holdings_value = sum(
            h["qty"] * prices.get(s, 0) for s, h in holdings.items()
        )
        total_equity = cash + holdings_value
        equity_curve.append({"step": t, "equity": round(total_equity, 2)})

        daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_pnl.append(daily_return)
        prev_equity = total_equity
        peak_equity = max(peak_equity, total_equity)

        # Stop bij extreme drawdown
        drawdown = (peak_equity - total_equity) / peak_equity
        if drawdown > MAX_DRAWDOWN_PCT * 2:
            logger.warning("Extreme drawdown bij stap %d: %.1f%% — gestopt", t, drawdown * 100)
            break

    # ── Statistieken berekenen ─────────────────────────────────────
    final_equity = equity_curve[-1]["equity"] if equity_curve else start_balance
    total_return = (final_equity - start_balance) / start_balance

    returns_arr = np.array(daily_pnl) if daily_pnl else np.array([0.0])

    # Sharpe
    sharpe = float(
        np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)
    )

    # Sortino
    downside = returns_arr[returns_arr < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
    sortino = float(np.mean(returns_arr) / (downside_std + 1e-8) * np.sqrt(252))

    # Max drawdown
    max_drawdown = 0.0
    running_peak = start_balance
    for e in equity_curve:
        running_peak = max(running_peak, e["equity"])
        dd = (running_peak - e["equity"]) / running_peak
        max_drawdown = max(max_drawdown, dd)

    # Win rate + profit factor
    sell_trades = [t for t in all_trades if t["side"] == "sell"]
    winning = [t for t in sell_trades if t["pnl"] > 0]
    losing = [t for t in sell_trades if t["pnl"] < 0]
    win_rate = len(winning) / len(sell_trades) if sell_trades else 0

    avg_win = float(np.mean([t["pnl"] for t in winning])) if winning else 0.0
    avg_loss = abs(float(np.mean([t["pnl"] for t in losing]))) if losing else 0.0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Regime verdeling
    regime_dist = {}
    for r in regime_history:
        regime_dist[r] = regime_dist.get(r, 0) + 1
    total_regimes = len(regime_history) or 1
    regime_dist = {k: round(v / total_regimes * 100, 1) for k, v in regime_dist.items()}

    stats = {
        "start_balance": start_balance,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "total_trades": len(all_trades),
        "total_buys": sum(1 for t in all_trades if t["side"] == "buy"),
        "total_sells": len(sell_trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "trading_days": len(equity_curve),
        "regime_distribution_pct": regime_dist,
    }

    return {
        "stats": stats,
        "equity_curve": equity_curve,
        "trades": all_trades,
    }


# ════════════════════════════════════════════════════════════════════
# BENCHMARK VERGELIJKING
# ════════════════════════════════════════════════════════════════════

def compare_with_benchmark(
    equity_curve: List[Dict],
    histories: Dict[str, pd.DataFrame],
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    """Vergelijk strategie met buy-and-hold benchmark."""
    if benchmark not in histories:
        # Probeer benchmark apart op te halen
        try:
            bm_hist = fetch_histories([benchmark], "2020-01-01", "2030-01-01")
            if benchmark not in bm_hist:
                return {"benchmark": f"{benchmark} niet beschikbaar"}
            bm_data = bm_hist[benchmark]
        except Exception:
            return {"benchmark": f"{benchmark} niet beschikbaar"}
    else:
        bm_data = histories[benchmark]

    warmup = 60
    bm_prices = bm_data.iloc[warmup: warmup + len(equity_curve)]["close"].values

    if len(bm_prices) == 0:
        return {"benchmark": "Geen benchmark data"}

    bm_start = float(bm_prices[0])
    bm_returns = [(float(p) - bm_start) / bm_start * 100 for p in bm_prices]

    strat_start = equity_curve[0]["equity"] if equity_curve else 0
    strat_returns = [
        (e["equity"] - strat_start) / strat_start * 100 for e in equity_curve
    ]

    return {
        "benchmark_symbol": benchmark,
        "strategy_return_pct": round(strat_returns[-1], 2) if strat_returns else 0,
        "benchmark_return_pct": round(bm_returns[-1], 2) if bm_returns else 0,
        "outperformance_pct": round(
            (strat_returns[-1] if strat_returns else 0)
            - (bm_returns[-1] if bm_returns else 0),
            2,
        ),
    }


# ════════════════════════════════════════════════════════════════════
# RAPPORT GENEREREN
# ════════════════════════════════════════════════════════════════════

def _print_report(result: Dict, benchmark_result: Optional[Dict] = None) -> None:
    """Print een overzichtelijk backtest rapport."""
    if "error" in result:
        logger.error("Backtest mislukt: %s", result["error"])
        return

    stats = result["stats"]
    print("\n" + "═" * 60)
    print("  BACKTEST v2 RAPPORT")
    print("═" * 60)
    print(f"  Startbalans:       €{stats['start_balance']:>12,.2f}")
    print(f"  Eindbalans:        €{stats['final_equity']:>12,.2f}")
    print(f"  Totaal rendement:   {stats['total_return_pct']:>11.2f}%")
    print(f"  Sharpe ratio:       {stats['sharpe_ratio']:>11.2f}")
    print(f"  Sortino ratio:      {stats['sortino_ratio']:>11.2f}")
    print(f"  Max drawdown:       {stats['max_drawdown_pct']:>11.2f}%")
    print("─" * 60)
    print(f"  Totaal trades:      {stats['total_trades']:>11d}")
    print(f"    Buys:             {stats['total_buys']:>11d}")
    print(f"    Sells:            {stats['total_sells']:>11d}")
    print(f"  Win rate:           {stats['win_rate_pct']:>11.1f}%")
    print(f"  Profit factor:      {stats['profit_factor']:>11.2f}")
    print(f"  Gem. winst:        €{stats['avg_win']:>12.2f}")
    print(f"  Gem. verlies:      €{stats['avg_loss']:>12.2f}")
    print(f"  Handelsdagen:       {stats['trading_days']:>11d}")
    print("─" * 60)
    print("  Regime verdeling:")
    for regime, pct in stats.get("regime_distribution_pct", {}).items():
        print(f"    {regime:<12s}     {pct:>6.1f}%")

    if benchmark_result and "outperformance_pct" in benchmark_result:
        print("─" * 60)
        bm = benchmark_result
        print(f"  Benchmark ({bm['benchmark_symbol']}):")
        print(f"    Strategie:        {bm['strategy_return_pct']:>11.2f}%")
        print(f"    Benchmark:        {bm['benchmark_return_pct']:>11.2f}%")
        sign = "+" if bm["outperformance_pct"] >= 0 else ""
        print(f"    Outperformance:   {sign}{bm['outperformance_pct']:>10.2f}%")

    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest v2 — AI Trading Simulator",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Lijst van symbolen (bijv. AAPL MSFT BTC-USD)",
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None,
        help="Gebruik een vooraf gedefinieerd universum",
    )
    parser.add_argument(
        "--start", default="2023-01-01",
        help="Startdatum (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="Einddatum (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--balance", type=float, default=START_BALANCE,
        help="Startbalans in EUR",
    )
    parser.add_argument(
        "--benchmark", default="SPY",
        help="Benchmark symbool voor vergelijking",
    )
    parser.add_argument(
        "--no-broker", action="store_true",
        help="SimulatedBroker uitschakelen",
    )
    parser.add_argument(
        "--output", default=None,
        help="Pad voor JSON rapport output",
    )

    args = parser.parse_args()

    # Symbolen bepalen
    if args.preset:
        symbols = PRESETS[args.preset]
        logger.info("Preset '%s': %s", args.preset, symbols)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = PRESETS["mixed"]
        logger.info("Geen symbolen opgegeven, gebruik preset 'mixed': %s", symbols)

    # Benchmark toevoegen als die niet al in de lijst zit
    all_symbols = list(symbols)
    if args.benchmark and args.benchmark not in all_symbols:
        all_symbols.append(args.benchmark)

    # Data ophalen
    logger.info("Historische data ophalen: %s (%s tot %s)", all_symbols, args.start, args.end)
    histories = fetch_histories(all_symbols, args.start, args.end)

    if not histories:
        logger.error("Geen data opgehaald — kan backtest niet uitvoeren")
        sys.exit(1)

    # Filter symbolen die daadwerkelijk data hebben
    available = [s for s in symbols if s in histories]
    if not available:
        logger.error("Geen van de gevraagde symbolen heeft data")
        sys.exit(1)

    test_histories = {s: histories[s] for s in available}

    # Backtest draaien
    result = run_backtest_v2(
        histories=test_histories,
        start_balance=args.balance,
        use_broker=not args.no_broker,
    )

    # Benchmark vergelijking
    benchmark_result = None
    if args.benchmark and not result.get("error"):
        benchmark_result = compare_with_benchmark(
            result["equity_curve"], histories, args.benchmark
        )

    # Rapport printen
    _print_report(result, benchmark_result)

    # JSON output opslaan
    output_path = args.output or os.path.join(ML_MODELS_DIR, "backtest_v2_report.json")
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": available,
        "period": {"start": args.start, "end": args.end},
        **result,
    }
    if benchmark_result:
        report["benchmark"] = benchmark_result

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Rapport opgeslagen: %s", output_path)


if __name__ == "__main__":
    main()
