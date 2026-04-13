"""AI Trading Simulator API v3.0 — ML-gestuurde autonome trading bot.

Geen externe AI APIs meer (Groq/Gemini verwijderd).
Gebruikt LightGBM voor signaalvoorspelling en PPO RL voor portfolio management.
"""

from typing import Any, Dict, List
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from services.data_service import DataService
from services.feature_engine import FeatureEngine
from services.market_data_service import MarketDataService
from services.signal_predictor import SignalPredictor
from services.portfolio_manager import PortfolioManager
from services.auto_trader import AutoTrader

app = FastAPI(title="AI Trading Bot", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

LIMITS: Dict[str, Any] = {
    "max_position": float("inf"),
    "max_risk": 0.01,
    "max_trades_per_day": 50,
}

# Singletons
data_service: Any = None
feature_engine: Any = None
market_data_service: Any = None
signal_predictor: Any = None
portfolio_manager: Any = None
auto_trader: Any = None


@app.on_event("startup")
def startup_event() -> None:
    global data_service, feature_engine, market_data_service
    global signal_predictor, portfolio_manager, auto_trader

    data_service = DataService()
    feature_engine = FeatureEngine()
    market_data_service = MarketDataService()
    signal_predictor = SignalPredictor()
    portfolio_manager = PortfolioManager()

    auto_trader = AutoTrader(
        data_service=data_service,
        market_data_service=market_data_service,
        feature_engine=feature_engine,
        signal_predictor=signal_predictor,
        portfolio_manager=portfolio_manager,
        limits=LIMITS,
    )

    try:
        market_data_service.start_prefetch_scheduler(interval_seconds=300)
    except Exception as exc:
        logging.getLogger(__name__).warning("Prefetch scheduler start mislukt: %s", exc)

    try:
        interval = int(os.environ.get("AUTO_TRADE_INTERVAL", "5"))
        auto_trader.start(interval_minutes=interval)
        logging.getLogger(__name__).info("Auto-trader gestart (interval=%dm).", interval)
    except Exception as exc:
        logging.getLogger(__name__).warning("Auto-trader start mislukt: %s", exc)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _to_amsterdam(iso_str: str | None) -> str:
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M")
    except Exception:
        return iso_str


def _compute_market_value(holdings: List[Dict]) -> float:
    total = 0.0
    for h in holdings:
        try:
            instrument = market_data_service.get_instrument(str(h["symbol"]), str(h["market"]))
            price = float(market_data_service.get_snapshot(instrument)["price"])
            total += price * float(h["quantity"])
        except Exception:
            total += float(h["invested_amount"])
    return total


# ════════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════════

@app.post("/login")
def login(creds: dict) -> Dict[str, str]:
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password or not validate_credentials(username, password):
        raise HTTPException(status_code=401, detail="Ongeldige inloggegevens")
    return {"token": create_token()}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ════════════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════════════

@app.get("/status", dependencies=[Depends(verify_token)])
def get_status() -> Dict[str, Any]:
    status = auto_trader.get_status()
    ams_now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M:%S")
    status["server_time"] = ams_now
    status["started_at"] = _to_amsterdam(status.get("started_at"))
    status["last_cycle_time"] = _to_amsterdam(status.get("last_cycle_time"))
    # Alias fields for frontend compatibility
    status["auto_trader_running"] = status.get("running", False)
    status["model_loaded"] = status.get("ml_signal_model", False)
    status["instruments_count"] = len(market_data_service.get_watchlist()) if hasattr(market_data_service, 'get_watchlist') else 0
    return status


# ════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ════════════════════════════════════════════════════════════════════

@app.get("/portfolio/summary", dependencies=[Depends(verify_token)])
def portfolio_summary() -> Dict[str, Any]:
    holdings = data_service.get_holdings()
    cash = data_service.get_cash_balance()
    start_balance = data_service.get_start_balance()

    total_invested = 0.0
    total_market_value = 0.0
    total_unrealized_pnl = 0.0
    positions = []

    for h in holdings:
        try:
            instrument = market_data_service.get_instrument(str(h["symbol"]), str(h["market"]))
            price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            price = float(h["avg_entry_price"])

        qty = float(h["quantity"])
        invested = float(h["invested_amount"])
        value = price * qty
        pnl = value - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
        total_invested += invested
        total_market_value += value
        total_unrealized_pnl += pnl

        positions.append({
            "symbol": h["symbol"], "market": h["market"],
            "quantity": round(qty, 6),
            "avg_entry_price": round(float(h["avg_entry_price"]), 4),
            "avg_price": round(float(h["avg_entry_price"]), 4),
            "current_price": round(price, 4),
            "invested_amount": round(invested, 2),
            "current_value": round(value, 2),
            "market_value": round(value, 2),
            "unrealized_pnl": round(pnl, 2), "unrealized_pnl_pct": round(pnl_pct, 2),
            "opened_at": _to_amsterdam(h.get("opened_at")),
        })

    total_equity = cash + total_market_value
    total_pnl = total_equity - start_balance
    total_pnl_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0.0

    # Realized P/L
    history = data_service.get_trade_history()
    realized_pl = sum(t.profit_loss for t in history if t.action == "sell" and t.status.startswith("executed"))
    total_buys = sum(1 for t in history if t.action == "buy")
    total_sells = sum(1 for t in history if t.action == "sell")

    return {
        "currency": "EUR",
        "start_balance": round(start_balance, 2),
        "available_cash": round(cash, 2),
        "cash": round(cash, 2),
        "total_invested": round(total_invested, 2),
        "total_market_value": round(total_market_value, 2),
        "holdings_value": round(total_market_value, 2),
        "total_equity": round(total_equity, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "unrealized_pnl": round(total_unrealized_pnl, 2),
        "realized_pnl": round(realized_pl, 2),
        "day_pnl": round(total_unrealized_pnl, 2),
        "num_holdings": len(positions),
        "total_buys": total_buys,
        "total_sells": total_sells,
        "open_positions": positions,
    }


@app.get("/portfolio/holdings", dependencies=[Depends(verify_token)])
def portfolio_holdings() -> List[Dict[str, Any]]:
    return portfolio_summary()["open_positions"]


@app.get("/portfolio/history", dependencies=[Depends(verify_token)])
def portfolio_history() -> List[Dict[str, Any]]:
    return data_service.get_portfolio_history()


class AmountRequest(BaseModel):
    amount: float


@app.post("/portfolio/deposit", dependencies=[Depends(verify_token)])
def portfolio_deposit(req: AmountRequest) -> Dict[str, Any]:
    try:
        new_balance = data_service.deposit(req.amount)
        return {"status": "ok", "message": f"€{req.amount:.2f} gestort", "new_balance": round(new_balance, 2), "new_cash_balance": round(new_balance, 2)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/portfolio/withdraw", dependencies=[Depends(verify_token)])
def portfolio_withdraw(req: AmountRequest) -> Dict[str, Any]:
    try:
        new_balance = data_service.withdraw(req.amount)
        return {"status": "ok", "message": f"€{req.amount:.2f} opgenomen", "new_balance": round(new_balance, 2), "new_cash_balance": round(new_balance, 2)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ════════════════════════════════════════════════════════════════════
# TRADES
# ════════════════════════════════════════════════════════════════════

@app.get("/trades/history", dependencies=[Depends(verify_token)])
def trades_history() -> List[Dict[str, Any]]:
    history = data_service.get_trade_history()
    return [
        {
            "id": t.id, "symbol": t.symbol, "action": t.action, "side": t.action,
            "amount": t.amount, "total": t.amount, "quantity": t.quantity, "price": t.price,
            "profit_loss": t.profit_loss, "pnl": t.profit_loss, "timestamp": _to_amsterdam(t.timestamp),
            "status": t.status, "confidence": t.confidence, "model_version": t.model_version,
        }
        for t in reversed(history)
    ]


# ════════════════════════════════════════════════════════════════════
# AI INSIGHTS
# ════════════════════════════════════════════════════════════════════

@app.get("/ai/signals", dependencies=[Depends(verify_token)])
def ai_signals() -> List[Dict[str, Any]]:
    signals = auto_trader.get_last_signals()
    return signals[:50]


@app.get("/ai/insights", dependencies=[Depends(verify_token)])
def ai_insights() -> Dict[str, Any]:
    history = data_service.get_trade_history()
    sells = [t for t in history if t.action == "sell" and t.status.startswith("executed")]

    wins = sum(1 for t in sells if t.profit_loss > 0)
    total_pl = sum(t.profit_loss for t in sells)
    win_rate = wins / len(sells) if sells else 0

    # Biggest win/loss
    biggest_win = max((t.profit_loss for t in sells), default=0)
    biggest_loss = min((t.profit_loss for t in sells), default=0)

    # Feature importance
    feature_importance = signal_predictor.get_feature_importance()

    # Model performance over time
    perf_history = data_service.get_model_performance(30)

    # Recent trade analysis
    recent_sells = sells[-20:] if sells else []
    recent_win_rate = sum(1 for t in recent_sells if t.profit_loss > 0) / len(recent_sells) if recent_sells else 0

    return {
        "total_trades_analyzed": len(sells),
        "win_rate": round(win_rate, 4),
        "recent_win_rate": round(recent_win_rate, 4),
        "total_pnl": round(total_pl, 2),
        "avg_pnl_per_trade": round(total_pl / len(sells), 2) if sells else 0,
        "biggest_win": round(biggest_win, 2),
        "biggest_loss": round(biggest_loss, 2),
        "signal_model_loaded": signal_predictor.is_model_loaded(),
        "signal_model_version": signal_predictor.get_model_version(),
        "rl_model_loaded": portfolio_manager.is_model_loaded(),
        "feature_importance": feature_importance,
        "performance_history": perf_history,
    }


# ════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════

class SettingsUpdate(BaseModel):
    interval_minutes: int | None = None
    auto_trade: bool | None = None


@app.post("/settings/update", dependencies=[Depends(verify_token)])
def update_settings(req: SettingsUpdate) -> Dict[str, Any]:
    result = {}

    if req.auto_trade is not None:
        if req.auto_trade and not auto_trader.is_running():
            interval = req.interval_minutes or int(os.environ.get("AUTO_TRADE_INTERVAL", "5"))
            auto_trader.start(interval_minutes=interval)
            result["auto_trade"] = "started"
        elif not req.auto_trade and auto_trader.is_running():
            auto_trader.stop()
            result["auto_trade"] = "stopped"

    if req.interval_minutes is not None and auto_trader.is_running():
        auto_trader.stop()
        auto_trader.start(interval_minutes=req.interval_minutes)
        result["interval_minutes"] = req.interval_minutes

    return {"status": "ok", **result}


# ════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY — auto-trader/summary endpoint
# ════════════════════════════════════════════════════════════════════

@app.get("/auto-trader/summary", dependencies=[Depends(verify_token)])
def auto_trader_summary() -> Dict[str, Any]:
    """Backward-compatible endpoint met volledige portfolio data."""
    status = get_status()
    summary = portfolio_summary()
    trades = trades_history()

    return {
        **status,
        **summary,
        "trade_history": trades,
    }
