from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from models.holding import HoldingView
from models.signal import Signal
from models.trade import TradeRequest, TradeResult
from services.alert_service import AlertService
from services.advice_engine import AdviceEngine
from services.data_service import DataService
from services.learning_agent import LearningAgent
from services.market_data_service import MarketDataService
from services.trade_engine import execute_trade as run_trade

app = FastAPI(title="AI Trading Simulator API", version="2.0.0")

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
alert_service = AlertService(data_service)

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
    """Publiek endpoint: retourneert de 10 beste koopadviezen.

    Invalideert snapshot-cache zodat verse data wordt opgehaald bij elke refresh.
    """
    market_data_service.invalidate_cache()
    advice_engine.invalidate_signal_cache()
    buy_signals = advice_engine.build_ranked_buy_signals(force_refresh=True)
    return buy_signals[:10]


@app.get("/advice", response_model=List[Signal])
def get_advice() -> List[Signal]:
    return _build_buy_advice()


@app.get("/watchlist")
def get_watchlist() -> List[Dict[str, str]]:
    return market_data_service.get_watchlist()


@app.post("/trade", dependencies=[Depends(verify_token)], response_model=TradeResult)
def trade_endpoint(trade_request: TradeRequest) -> TradeResult:
    if trade_request.action == "hold":
        raise HTTPException(status_code=400, detail="Hold-signalen kunnen niet als trade worden uitgevoerd")
    signal_map = advice_engine.build_signal_map()
    return run_trade(
        trade_request,
        LIMITS,
        data_service,
        market_data_service,
        signal_map.get(trade_request.symbol.upper()),
    )


@app.get("/history", dependencies=[Depends(verify_token)], response_model=List[TradeResult])
def get_history() -> List[TradeResult]:
    return data_service.get_trade_history()


@app.get("/portfolio", dependencies=[Depends(verify_token)])
def get_portfolio() -> Dict[str, Any]:
    holdings = _build_holdings_view()
    portfolio = data_service.get_portfolio(holdings)
    portfolio["holdings"] = holdings
    return portfolio


@app.get("/holdings", dependencies=[Depends(verify_token)], response_model=List[HoldingView])
def get_holdings() -> List[HoldingView]:
    return _build_holdings_view()


@app.get("/sell-advice", dependencies=[Depends(verify_token)], response_model=List[HoldingView])
def get_sell_advice() -> List[HoldingView]:
    return _build_sell_advice()


@app.post("/learn", dependencies=[Depends(verify_token)])
def learn() -> Dict[str, Any]:
    """Leer van trade-geschiedenis en pas signaalparameters aan."""
    history = data_service.get_trade_history()
    latest_signals = advice_engine.build_signals()
    parameters = learning_agent.learn(history, latest_signals)

    # Pas geleerde parameters toe op de advice engine
    advice_engine.apply_learned_params(learning_agent.get_engine_params())

    return {"parameters": parameters}


@app.post("/alerts/realtime", dependencies=[Depends(verify_token)])
def send_realtime_alerts() -> Dict[str, Any]:
    signals = _build_buy_advice()
    sell_advice = _build_sell_advice()
    try:
        return alert_service.send_realtime_alerts(signals, sell_advice)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/alerts/summary", dependencies=[Depends(verify_token)])
def send_daily_summary() -> Dict[str, Any]:
    signals = _build_buy_advice()
    sell_advice = _build_sell_advice()
    try:
        return alert_service.send_daily_summary(signals, sell_advice)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/login")
def login(creds: dict) -> Dict[str, str]:
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password or not validate_credentials(username, password):
        raise HTTPException(status_code=401, detail="Ongeldige inloggegevens")
    token = create_token()
    return {"token": token}


def _build_holdings_view() -> List[HoldingView]:
    signal_map = advice_engine.build_signal_map()
    views: List[HoldingView] = []
    for holding in data_service.get_holdings():
        try:
            instrument = market_data_service.get_instrument(
                str(holding["symbol"]),
                str(holding["market"]),
            )
            current_price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            current_price = float(holding["avg_entry_price"])

        views.append(
            advice_engine.build_holding_view(
                holding,
                signal_map.get(str(holding["symbol"])),
                current_price,
            )
        )

    return sorted(views, key=lambda item: item.unrealized_profit_loss_pct, reverse=True)


def _build_buy_advice() -> List[Signal]:
    buy_signals = advice_engine.build_ranked_buy_signals()
    return sorted(
        buy_signals,
        key=lambda signal: (0 if signal.market == "us" or signal.market == "eu" else 1, -signal.ranking_score),
    )


def _build_sell_advice() -> List[HoldingView]:
    holdings = _build_holdings_view()
    return sorted(
        holdings,
        key=lambda holding: (
            0 if holding.recommendation == "sell_now" else 1 if holding.recommendation == "take_partial_profit" else 2,
            -holding.position_score,
            -abs(holding.unrealized_profit_loss_pct),
        ),
    )
