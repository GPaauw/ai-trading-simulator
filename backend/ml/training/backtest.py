"""Backtest Engine — evalueert het volledige ML pipeline op historische data.

Simuleert de gehele trading pipeline (features → signalen → beslissingen → trades)
en genereert prestatiemetrieken: equity curve, Sharpe ratio, max drawdown, win rate.
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.config import (
    ML_MODELS_DIR, START_BALANCE, TRANSACTION_COST_PCT, SLIPPAGE_PCT,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_backtest(
    histories: dict[str, pd.DataFrame],
    start_balance: float = START_BALANCE,
    start_date: str | None = None,
) -> dict:
    """Draai een volledige backtest over historische data.

    Returns:
        dict met equity_curve, trades, en samenvattende statistieken.
    """
    from services.feature_engine import FeatureEngine
    from services.signal_predictor import SignalPredictor

    fe = FeatureEngine()
    predictor = SignalPredictor()

    cash = start_balance
    holdings: dict[str, dict] = {}  # symbol -> {qty, avg_price}
    equity_curve = []
    all_trades = []
    daily_pnl = []

    # Bepaal tijdas: gebruik het kortste symbool, skip eerste 60 bars
    symbols = list(histories.keys())
    min_len = min(len(histories[s]) for s in symbols)
    warmup = 60

    if min_len <= warmup + 10:
        logger.warning("Te weinig data voor backtest")
        return {"error": "Insufficient data"}

    timesteps = min_len - warmup

    prev_equity = start_balance
    peak_equity = start_balance

    for t in range(timesteps):
        idx = warmup + t

        # 1) Bereken features voor elk symbool
        batch_features = {}
        prices = {}
        for sym in symbols:
            hist = histories[sym].iloc[:idx + 1]
            features = fe.compute_features(hist)
            if features:
                batch_features[sym] = features
            prices[sym] = float(hist.iloc[-1]["close"])

        if not batch_features:
            continue

        # Features DataFrame voor predictor
        features_df = pd.DataFrame.from_dict(batch_features, orient="index")
        features_df.index.name = "symbol"

        # 2) Genereer signalen
        signals = predictor.predict(features_df)

        # 3) Verwerk signalen (vereenvoudigde portfolio logica)
        # Verkoop op sell-signalen
        for sig in signals:
            sym = sig["symbol"]
            action = sig["action"]

            if action in ("sell", "strong_sell") and sym in holdings:
                h = holdings[sym]
                price = prices.get(sym, 0)
                if price <= 0:
                    continue
                # Verkoop met slippage en kosten
                sell_price = price * (1 - SLIPPAGE_PCT)
                proceeds = h["qty"] * sell_price
                cost = proceeds * TRANSACTION_COST_PCT
                cash += proceeds - cost

                pnl = (sell_price - h["avg_price"]) * h["qty"] - cost
                all_trades.append({
                    "step": t,
                    "symbol": sym,
                    "side": "sell",
                    "qty": h["qty"],
                    "price": round(sell_price, 4),
                    "pnl": round(pnl, 2),
                    "confidence": sig.get("confidence", 0),
                })
                del holdings[sym]

        # Koop op buy-signalen (top 3 op confidence)
        buy_signals = [s for s in signals if s["action"] in ("buy", "strong_buy")]
        buy_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        for sig in buy_signals[:3]:
            sym = sig["symbol"]
            if sym in holdings:
                continue  # al in portfolio

            price = prices.get(sym, 0)
            if price <= 0:
                continue

            # Bereken positiegrootte
            total_equity = cash + sum(
                h["qty"] * prices.get(s, 0) for s, h in holdings.items()
            )
            max_alloc = total_equity * MAX_POSITION_PCT
            alloc = min(max_alloc, cash * 0.3)  # max 30% van cash per positie

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
                "confidence": sig.get("confidence", 0),
            })

        # 4) Bereken equity
        holdings_value = sum(
            h["qty"] * prices.get(s, 0) for s, h in holdings.items()
        )
        total_equity = cash + holdings_value
        equity_curve.append({"step": t, "equity": round(total_equity, 2)})

        # Daily PnL
        daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_pnl.append(daily_return)
        prev_equity = total_equity

        # Peak tracking
        peak_equity = max(peak_equity, total_equity)

        # Stop bij max drawdown
        drawdown = (peak_equity - total_equity) / peak_equity
        if drawdown > MAX_DRAWDOWN_PCT * 1.5:  # geef wat ruimte in backtest
            logger.warning("Max drawdown bereikt bij stap %d: %.1f%%", t, drawdown * 100)
            break

    # 5) Bereken statistieken
    final_equity = equity_curve[-1]["equity"] if equity_curve else start_balance
    total_return = (final_equity - start_balance) / start_balance
    peak_equity = max(e["equity"] for e in equity_curve) if equity_curve else start_balance
    max_drawdown = 0
    running_peak = start_balance
    for e in equity_curve:
        running_peak = max(running_peak, e["equity"])
        dd = (running_peak - e["equity"]) / running_peak
        max_drawdown = max(max_drawdown, dd)

    returns_arr = np.array(daily_pnl) if daily_pnl else np.array([0])
    sharpe = float(np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252))

    winning_trades = [t for t in all_trades if t["pnl"] > 0 and t["side"] == "sell"]
    losing_trades = [t for t in all_trades if t["pnl"] < 0 and t["side"] == "sell"]
    total_sell_trades = len(winning_trades) + len(losing_trades)
    win_rate = len(winning_trades) / total_sell_trades if total_sell_trades > 0 else 0

    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t["pnl"] for t in losing_trades])) if losing_trades else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    stats = {
        "start_balance": start_balance,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(all_trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "trading_days": len(equity_curve),
    }

    return {
        "stats": stats,
        "equity_curve": equity_curve,
        "trades": all_trades,
    }


def compare_with_benchmark(equity_curve: list, histories: dict) -> dict:
    """Vergelijk strategie met buy-and-hold S&P 500 (SPY)."""
    if "SPY" not in histories:
        return {"benchmark": "SPY not available"}

    spy = histories["SPY"]
    warmup = 60
    spy_prices = spy.iloc[warmup:warmup + len(equity_curve)]["close"].values

    if len(spy_prices) == 0:
        return {"benchmark": "No SPY data"}

    spy_start = float(spy_prices[0])
    spy_returns = [(float(p) - spy_start) / spy_start * 100 for p in spy_prices]

    strategy_start = equity_curve[0]["equity"] if equity_curve else 0
    strategy_returns = [
        (e["equity"] - strategy_start) / strategy_start * 100 for e in equity_curve
    ]

    return {
        "strategy_final_return_pct": round(strategy_returns[-1], 2) if strategy_returns else 0,
        "spy_final_return_pct": round(spy_returns[-1], 2) if spy_returns else 0,
        "outperformance_pct": round(
            (strategy_returns[-1] if strategy_returns else 0) -
            (spy_returns[-1] if spy_returns else 0), 2
        ),
    }


def main():
    from ml.training.train_signals import download_data, get_training_symbols

    logger.info("=== Backtest gestart ===")

    symbols = get_training_symbols()[:20]
    # Zorg dat SPY erbij zit voor benchmark
    if "SPY" not in symbols:
        symbols.append("SPY")

    histories = download_data(symbols, period="2y")

    if len(histories) < 5:
        logger.error("Te weinig data. Backtest afgebroken.")
        sys.exit(1)

    logger.info("Backtest met %d symbolen, ~%d datapunten",
                len(histories), min(len(v) for v in histories.values()))

    result = run_backtest(histories)

    if "error" in result:
        logger.error("Backtest mislukt: %s", result["error"])
        sys.exit(1)

    stats = result["stats"]
    logger.info("=== Backtest Resultaten ===")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)

    # Benchmark vergelijking
    benchmark = compare_with_benchmark(result["equity_curve"], histories)
    logger.info("=== Benchmark Vergelijking ===")
    for key, value in benchmark.items():
        logger.info("  %s: %s", key, value)

    # Sla rapport op
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    report = {
        "backtest_at": datetime.now().isoformat(),
        "stats": stats,
        "benchmark": benchmark,
    }
    report_path = os.path.join(ML_MODELS_DIR, "backtest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Rapport opgeslagen: %s", report_path)

    logger.info("=== Backtest voltooid ===")


if __name__ == "__main__":
    main()
