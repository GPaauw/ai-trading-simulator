from typing import Any, Dict, List
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Query
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
from services.auto_trader import AutoTrader
from services.trade_engine import execute_trade as run_trade

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
alert_service: Any = None
ai_analysis_service: Any = None
auto_trader: Any = None


@app.on_event("startup")
def startup_event() -> None:
    global data_service, learning_agent, market_data_service, advice_engine, \
           alert_service, ai_analysis_service, auto_trader

    data_service = DataService()
    learning_agent = LearningAgent()
    market_data_service = MarketDataService()
    advice_engine = AdviceEngine(market_data_service)
    alert_service = AlertService(data_service)
    ai_analysis_service = AiAnalysisService()
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

@app.get("/auto-trader/status")
def auto_trader_status() -> Dict[str, Any]:
    """Status van de autonome trading-bot (publiek leesbaar)."""
    return auto_trader.get_status()


@app.get("/auto-trader/summary")
def auto_trader_summary() -> Dict[str, Any]:
    """Gecombineerde samenvatting: status + recente trades + open posities met P/L."""
    status = auto_trader.get_status()

    # Recente trades (laatste 30, nieuwste eerst)
    history = data_service.get_trade_history()
    recent_trades = [
        {
            "id": t.id,
            "symbol": t.symbol,
            "action": t.action,
            "amount": t.amount,
            "quantity": t.quantity,
            "price": t.price,
            "profit_loss": t.profit_loss,
            "timestamp": t.timestamp,
            "status": t.status,
        }
        for t in reversed(history[-30:])
    ]

    # Open posities met live P/L
    holdings = data_service.get_holdings()
    open_positions: List[Dict[str, Any]] = []
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
        })

    status["recent_trades"] = recent_trades
    status["open_positions"] = open_positions
    return status


@app.post("/auto-trader/start", dependencies=[Depends(verify_token)])
def auto_trader_start(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Start de autonome trading-bot. Body: {\"interval_minutes\": 5}"""
    interval = 5
    if payload and "interval_minutes" in payload:
        interval = int(payload["interval_minutes"])
    started = auto_trader.start(interval_minutes=interval)
    if not started:
        raise HTTPException(status_code=409, detail="Auto-trader draait al.")
    return auto_trader.get_status()


@app.post("/auto-trader/stop", dependencies=[Depends(verify_token)])
def auto_trader_stop() -> Dict[str, Any]:
    """Stop de autonome trading-bot."""
    stopped = auto_trader.stop()
    if not stopped:
        raise HTTPException(status_code=409, detail="Auto-trader draait niet.")
    return auto_trader.get_status()


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
    """Top signalen verrijkt met AI-analyse via Gemini (fallback: technische analyse)."""
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
    """Alle koopadviezen, verrijkt met AI-analyse als GEMINI_API_KEY beschikbaar is."""
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


@app.get("/search")
def search_symbols(q: str = Query(..., min_length=1, max_length=10)) -> List[Dict[str, Any]]:
    """Zoek aandelen/crypto/grondstoffen op symbool."""
    try:
        return market_data_service.search_symbol(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze-custom", dependencies=[Depends(verify_token)])
def analyze_custom_symbols(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AI-analyse voor door de gebruiker opgegeven symbolen.

    Body: {"symbols": [{"symbol": "AAPL", "market": "us", "buy_price": 150.0, "quantity": 10}]}
    """
    symbols = payload.get("symbols", [])
    if not symbols or not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="Geef een lijst van symbolen op.")

    # Bouw signaal-achtige dicts met live marktdata
    signal_dicts: List[Dict[str, Any]] = []
    for entry in symbols[:15]:
        symbol = str(entry.get("symbol", "")).upper()
        market = str(entry.get("market", "us"))
        buy_price = float(entry.get("buy_price", 0))
        quantity = float(entry.get("quantity", 0))
        if not symbol:
            continue

        # Haal live data op
        try:
            details = market_data_service.get_symbol_details(symbol, market)
            live_price = float(details["price"])
        except Exception:
            live_price = buy_price if buy_price > 0 else 0

        pnl_pct = ((live_price - buy_price) / buy_price * 100) if buy_price > 0 else 0

        signal_dicts.append({
            "symbol": symbol,
            "market": market,
            "price": live_price,
            "action": "hold",
            "confidence": 0.5,
            "risk_pct": 2.0,
            "expected_return_pct": 0.0,
            "target_price": 0.0,
            "reason": f"Handmatig toegevoegd | Aankoop: €{buy_price:.2f} | "
                      f"Nu: €{live_price:.2f} | P/L: {pnl_pct:+.2f}%"
                      + (f" | {quantity:.4f} stuks" if quantity > 0 else ""),
        })

    if not signal_dicts:
        return []

    trade_history = data_service.get_trade_history()
    return ai_analysis_service.analyze_signals(signal_dicts, trade_history=trade_history)


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
    learning_agent.save_params(data_service)

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
