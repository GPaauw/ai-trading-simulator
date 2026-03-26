from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from models.signal import Signal
from models.trade import TradeRequest, TradeResult
from services.alert_service import AlertService
from services.advice_engine import AdviceEngine
from services.data_service import DataService
from services.learning_agent import LearningAgent
from services.market_data_service import MarketDataService
from services.trade_engine import execute_trade as run_trade

app = FastAPI(title="AI Trading Simulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Singletons (in-memory state per worker)
data_service = DataService()
learning_agent = LearningAgent()
market_data_service = MarketDataService()
advice_engine = AdviceEngine(market_data_service)
alert_service = AlertService()

# Limieten (aanpasbaar)
LIMITS = {
    "max_position": 10_000.0,
    "max_risk": 0.02,
    "max_trades_per_day": 10,
}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/signals", response_model=List[Signal])
def get_signals() -> List[Signal]:
    """Publiek endpoint: actuele marktadviezen op basis van gratis live data."""
    return advice_engine.build_signals()


@app.get("/advice", response_model=List[Signal])
def get_advice() -> List[Signal]:
    return advice_engine.build_signals()


@app.get("/watchlist")
def get_watchlist() -> List[Dict[str, str]]:
    return market_data_service.get_watchlist()


@app.post("/trade", dependencies=[Depends(verify_token)], response_model=TradeResult)
def trade_endpoint(trade_request: TradeRequest) -> TradeResult:
    if trade_request.action == "hold":
        raise HTTPException(status_code=400, detail="Hold-signalen kunnen niet als trade worden uitgevoerd")
    return run_trade(trade_request, LIMITS, data_service)


@app.get("/history", dependencies=[Depends(verify_token)], response_model=List[TradeResult])
def get_history() -> List[TradeResult]:
    return data_service.get_trade_history()


@app.get("/portfolio", dependencies=[Depends(verify_token)])
def get_portfolio() -> Dict[str, Any]:
    return data_service.get_portfolio()


@app.post("/learn", dependencies=[Depends(verify_token)])
def learn() -> Dict[str, Any]:
    history = data_service.get_trade_history()
    latest_signals = advice_engine.build_signals()
    parameters = learning_agent.learn(history, latest_signals)
    return {"parameters": parameters}


@app.post("/alerts/realtime", dependencies=[Depends(verify_token)])
def send_realtime_alerts() -> Dict[str, Any]:
    signals = advice_engine.build_signals()
    try:
        return alert_service.send_realtime_alerts(signals)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/alerts/summary", dependencies=[Depends(verify_token)])
def send_daily_summary() -> Dict[str, Any]:
    signals = advice_engine.build_signals()
    try:
        return alert_service.send_daily_summary(signals)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/login")
def login(creds: dict) -> Dict[str, str]:
    """Eenvoudige login endpoint. Verwacht JSON: {"username": "..", "password": ".."}.
    Retourneert een Bearer token dat later gebruikt wordt in Authorization header.
    """
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password or not validate_credentials(username, password):
        raise HTTPException(status_code=401, detail="Ongeldige inloggegevens")
    token = create_token()
    return {"token": token}
