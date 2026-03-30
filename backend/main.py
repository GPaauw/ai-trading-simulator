from typing import Any, Dict, List
import logging
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth.security import validate_credentials, verify_token
from auth.session import create_token
from models.holding import HoldingView
from models.signal import Signal
from models.trade import TradeRequest, TradeResult
from services.ai_analysis_service import AiAnalysisService
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
ai_analysis_service = AiAnalysisService()


@app.on_event("startup")
def startup_event() -> None:
    try:
        prefetch_env = os.environ.get("PREFETCH_ON_STARTUP", "false").lower()
        if prefetch_env in ("1", "true", "yes"):
            market_data_service.start_prefetch_scheduler(interval_seconds=300)
        else:
            logging.getLogger(__name__).info(
                "Prefetch scheduler not started (PREFETCH_ON_STARTUP=%s)", prefetch_env
            )
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to start prefetch scheduler: %s", exc)

# Limieten (day-trading)
LIMITS = {
    "max_position": float("inf"),
    "max_risk": 0.01,
    "max_trades_per_day": 50,
}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/signals", response_model=List[Signal])
def get_signals() -> List[Signal]:
    """Publiek endpoint: retourneert de 10 beste koopadviezen.

    Gebruikt de bestaande TTL-cache zodat niet elke page refresh een volledige markt-scan triggert.
    """
    if market_data_service.has_warm_stock_cache():
        buy_signals = advice_engine.build_ranked_buy_signals()
    else:
        market_data_service.ensure_background_prefetch_started()
        buy_signals = advice_engine.build_ranked_buy_signals(
            watchlist=market_data_service.get_fast_watchlist(),
        )
    return buy_signals[:10]


@app.get("/signals/longterm", response_model=List[Signal])
def get_longterm_signals() -> List[Signal]:
    """Top 10 langetermijn koop-signalen (multi-day horizon)."""
    longterm = advice_engine.build_ranked_longterm_signals()
    return longterm[:10]


@app.get("/signals/premarket", response_model=List[Signal])
def get_premarket_signals() -> List[Signal]:
    """Pre-market scan: top 10 daytrade-picks voor vandaag (markturen genegeerd)."""
    return advice_engine.build_premarket_signals()


@app.get("/signals/ai")
def get_ai_signals() -> List[Dict[str, Any]]:
    """Top signalen verrijkt met AI-analyse via Groq (fallback: technische analyse)."""
    # Gebruik snelle watchlist voor snelle response
    buy_signals = advice_engine.build_ranked_buy_signals(
        watchlist=market_data_service.get_fast_watchlist(),
    )
    if not buy_signals:
        return []
    signal_dicts = [s.model_dump() for s in buy_signals[:10]]
    # Haal trade-historie op voor AI learning
    trade_history = data_service.get_trade_history()
    return ai_analysis_service.analyze_signals(signal_dicts, trade_history=trade_history)


@app.get("/advice")
def get_advice() -> Any:
    """Alle koopadviezen, verrijkt met AI-analyse als GROQ_API_KEY beschikbaar is."""
    return _build_buy_advice()


@app.get("/watchlist")
def get_watchlist() -> List[Dict[str, str]]:
    return market_data_service.get_watchlist()


@app.get("/prefetch/status")
def get_prefetch_status() -> Dict[str, object]:
    """Returneert status van achtergrond-prefetch (last run, running, count, error)."""
    try:
        return market_data_service.get_prefetch_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/prefetch/now", dependencies=[Depends(verify_token)])
def post_prefetch_now() -> Dict[str, object]:
    """Forceren dat de server één prefetch uitvoert (blokkerend)."""
    try:
        market_data_service.prefetch_once()
        cached = sum(1 for k in market_data_service._cache if k.startswith("history:yf:"))
        return {"started": True, "cached_count": cached}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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

    # Dagelijkse P/L
    daily_realized = data_service.get_daily_realized_pnl()
    daily_unrealized = sum(h.unrealized_profit_loss for h in holdings)
    portfolio["daily_realized_pnl"] = round(daily_realized, 2)
    portfolio["daily_unrealized_pnl"] = round(daily_unrealized, 2)
    portfolio["daily_total_pnl"] = round(daily_realized + daily_unrealized, 2)
    return portfolio


@app.get("/holdings", dependencies=[Depends(verify_token)], response_model=List[HoldingView])
def get_holdings() -> List[HoldingView]:
    return _build_holdings_view()


@app.get("/sell-advice", dependencies=[Depends(verify_token)], response_model=List[HoldingView])
def get_sell_advice() -> List[HoldingView]:
    return _build_sell_advice()


@app.post("/sell-all", dependencies=[Depends(verify_token)])
def sell_all_positions() -> Dict[str, Any]:
    """Day-trading: verkoop alle open posities aan einde van de dag."""
    holdings = data_service.get_holdings()
    if not holdings:
        return {"sold": 0, "total_profit_loss": 0.0, "results": []}

    results: List[Dict[str, Any]] = []
    total_pl = 0.0
    for holding in holdings:
        symbol = str(holding["symbol"])
        market = str(holding["market"])
        quantity = float(holding["quantity"])
        try:
            instrument = market_data_service.get_instrument(symbol, market)
            live_price = float(market_data_service.get_snapshot(instrument)["price"])
        except Exception:
            live_price = float(holding["avg_entry_price"])

        sell_request = TradeRequest(
            symbol=symbol,
            market=market,
            action="sell",
            amount=0.0,
            quantity=quantity,
        )
        result = run_trade(sell_request, LIMITS, data_service, market_data_service)
        results.append({
            "symbol": symbol,
            "quantity": round(quantity, 6),
            "price": round(live_price, 4),
            "profit_loss": result.profit_loss,
            "status": result.status,
        })
        total_pl += result.profit_loss

    return {
        "sold": len(results),
        "total_profit_loss": round(total_pl, 2),
        "results": results,
    }


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


def _build_buy_advice() -> List:
    if market_data_service.has_warm_stock_cache():
        buy_signals = advice_engine.build_ranked_buy_signals()
    else:
        market_data_service.ensure_background_prefetch_started()
        buy_signals = advice_engine.build_ranked_buy_signals(
            watchlist=market_data_service.get_fast_watchlist(),
        )
    sorted_signals = sorted(
        buy_signals,
        key=lambda signal: (0 if signal.market == "us" or signal.market == "eu" else 1, -signal.ranking_score),
    )

    # Verrijk alle signalen met AI-analyse (incl. learning van trade-historie)
    if ai_analysis_service.is_available():
        signal_dicts = [s.model_dump() for s in sorted_signals]
        trade_history = data_service.get_trade_history()
        ai_results = ai_analysis_service.analyze_signals(signal_dicts, trade_history=trade_history)
        if ai_results:
            return ai_results

    return sorted_signals


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
