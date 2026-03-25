from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from models.signal import Signal
from models.trade import TradeRequest, TradeResult
from services.data_service import DataService
from services.learning_agent import LearningAgent
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

# Limieten (aanpasbaar)
LIMITS = {
    "max_position": 10_000.0,
    "max_risk": 0.02,
    "max_trades_per_day": 20,
}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/signals", response_model=List[Signal])
def get_signals() -> List[Signal]:
    """Publiek endpoint: iedereen kan huidige dummy-signalen ophalen.
    Trades, history en learn blijven beveiligd.
    """
    return data_service.get_signals()


@app.post("/trade", dependencies=[Depends(verify_token)], response_model=TradeResult)
def trade_endpoint(trade_request: TradeRequest) -> TradeResult:
    return run_trade(trade_request, LIMITS, data_service)


@app.get("/history", dependencies=[Depends(verify_token)], response_model=List[TradeResult])
def get_history() -> List[TradeResult]:
    return data_service.get_trade_history()


@app.post("/learn", dependencies=[Depends(verify_token)])
def learn() -> Dict[str, Any]:
    history = data_service.get_trade_history()
    parameters = learning_agent.learn(history)
    return {"parameters": parameters}


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
