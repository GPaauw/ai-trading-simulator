from typing import Any, Dict, List
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from services.advice_engine import AdviceEngine
from services.data_service import DataService
from services.learning_agent import LearningAgent
from services.market_data_service import MarketDataService
from services.auto_trader import AutoTrader

app = FastAPI(title="AI Trading Simulator API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Limieten (day-trading)
LIMITS: Dict[str, Any] = {
    "max_position": float("inf"),
    "max_risk": 0.01,
    "max_trades_per_day": 50,
}

# Singletons — worden geïnitialiseerd in startup_event zodat de DB-verbinding
# pas wordt gemaakt nadat FastAPI volledig is opgestart (netwerk beschikbaar op Render).
data_service: Any = None
learning_agent: Any = None
market_data_service: Any = None
advice_engine: Any = None
auto_trader: Any = None


@app.on_event("startup")
def startup_event() -> None:
    global data_service, learning_agent, market_data_service, advice_engine, auto_trader

    data_service = DataService()
    learning_agent = LearningAgent()
    market_data_service = MarketDataService()
    advice_engine = AdviceEngine(market_data_service)
    auto_trader = AutoTrader(
        data_service=data_service,
        market_data_service=market_data_service,
        advice_engine=advice_engine,
        learning_agent=learning_agent,
        limits=LIMITS,
    )

    try:
        market_data_service.start_prefetch_scheduler(interval_seconds=300)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to start prefetch scheduler: %s", exc)

    # Auto-trader altijd starten — werkt zonder Groq via technische fallback
    try:
        interval = int(os.environ.get("AUTO_TRADE_INTERVAL", "5"))
        auto_trader.start(interval_minutes=interval)
        logging.getLogger(__name__).info("Auto-trader gestart met interval=%dm.", interval)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to start auto-trader: %s", exc)

    # Laad eerder geleerde parameters uit de database
    try:
        learning_agent.load_params(data_service)
        advice_engine.apply_learned_params(learning_agent.get_engine_params())
        logging.getLogger(__name__).info("Geleerde parameters geladen uit database.")
    except Exception as exc:
        logging.getLogger(__name__).warning("Kon leerparameters niet laden: %s", exc)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════
# AUTO-TRADER ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

def _to_amsterdam(iso_str: str | None) -> str:
    """Converteer een ISO-timestamp naar Amsterdam-tijd (dd-MM-yyyy HH:mm)."""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M")
    except Exception:
        return iso_str


@app.get("/auto-trader/summary", dependencies=[Depends(verify_token)])
def auto_trader_summary() -> Dict[str, Any]:
    """Gecombineerde samenvatting: status + alle trades + open posities met live P/L."""
    status = auto_trader.get_status()

    # Amsterdam-tijd voor status-timestamps
    ams_now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M:%S")
    status["server_time"] = ams_now
    status["started_at"] = _to_amsterdam(status.get("started_at"))
    status["last_cycle_time"] = _to_amsterdam(status.get("last_cycle_time"))

    # Alle trades (nieuwste eerst)
    history = data_service.get_trade_history()
    total_buys = 0
    total_sells = 0
    realized_pl = 0.0
    all_trades: List[Dict[str, Any]] = []
    for t in reversed(history):
        if t.action == "buy":
            total_buys += 1
        elif t.action == "sell":
            total_sells += 1
            realized_pl += t.profit_loss
        all_trades.append({
            "id": t.id,
            "symbol": t.symbol,
            "action": t.action,
            "amount": t.amount,
            "quantity": t.quantity,
            "price": t.price,
            "profit_loss": t.profit_loss,
            "timestamp": _to_amsterdam(t.timestamp),
            "status": t.status,
        })

    # Open posities met live P/L
    holdings = data_service.get_holdings()
    open_positions: List[Dict[str, Any]] = []
    total_invested = 0.0
    total_market_value = 0.0
    total_unrealized_pnl = 0.0
    for h in holdings:
        try:
            instrument = market_data_service.get_instrument(str(h["symbol"]), str(h["market"]))
            price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            price = float(h["avg_entry_price"])
        quantity = float(h["quantity"])
        invested = float(h["invested_amount"])
        value = price * quantity
        pnl = value - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
        total_invested += invested
        total_market_value += value
        total_unrealized_pnl += pnl
        open_positions.append({
            "symbol": h["symbol"],
            "market": h["market"],
            "quantity": round(quantity, 6),
            "avg_entry_price": round(float(h["avg_entry_price"]), 4),
            "current_price": round(price, 4),
            "invested_amount": round(invested, 2),
            "current_value": round(value, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "opened_at": _to_amsterdam(h.get("opened_at")),
        })

    # Portfolio-samenvatting
    cash = data_service.get_cash_balance()
    start_balance = float(data_service._get_setting("start_balance", "2000"))
    total_equity = cash + total_market_value
    total_pnl = total_equity - start_balance
    total_pnl_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0.0

    status["currency"] = "EUR"
    status["start_balance"] = round(start_balance, 2)
    status["available_cash"] = round(cash, 2)
    status["total_invested"] = round(total_invested, 2)
    status["total_market_value"] = round(total_market_value, 2)
    status["total_equity"] = round(total_equity, 2)
    status["total_pnl"] = round(total_pnl, 2)
    status["total_pnl_pct"] = round(total_pnl_pct, 2)
    status["unrealized_pnl"] = round(total_unrealized_pnl, 2)
    status["realized_pnl"] = round(realized_pl, 2)
    status["total_buys"] = total_buys
    status["total_sells"] = total_sells
    status["trade_history"] = all_trades
    status["open_positions"] = open_positions
    return status


@app.post("/login")
def login(creds: dict) -> Dict[str, str]:
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password or not validate_credentials(username, password):
        raise HTTPException(status_code=401, detail="Ongeldige inloggegevens")
    token = create_token()
    return {"token": token}
