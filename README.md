# AI Trading Simulator v2

Autonome AI-gestuurde trading bot voor high-volatility, high-liquidity markten (crypto + US momentum stocks).

## Architectuur

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutoTrader v2 (orchestratie)                │
│  12-staps cyclus met graceful degradation per module            │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│ Market   │ Alt-Data │ Smart    │ Feature  │ Ensemble │Recurrent│
│ Scanner  │ Collector│ Money    │ Engine   │ LightGBM │  PPO    │
│          │          │ Tracker  │ (48 feat)│ (3-win)  │ (LSTM)  │
├──────────┴──────────┴──────────┼──────────┴──────────┴─────────┤
│ Regime Classifier   Sentiment │ Simulated Broker               │
│ (bull/bear/sideways/high_vol) │ (order book, slippage, fees)   │
└───────────────────────────────┴────────────────────────────────┘
```

### V2 Modules

| Module | Bestand | Functie |
|--------|---------|---------|
| **MarketScanner** | `services/market_scanner.py` | Dynamisch universum: top-5 crypto (ccxt/Binance) + top-5 stocks (yfinance) op momentum, volume, volatiliteit |
| **RegimeClassifier** | `ml/regime_classifier.py` | LightGBM classificatie → bull/bear/sideways/high_vol met heuristische fallback |
| **SentimentAnalyzer** | `ml/sentiment_analyzer.py` | Lokaal FinBERT (ProsusAI/finbert) sentiment scoring [-1, +1] |
| **AlternativeData** | `services/alternative_data.py` | Reddit, News RSS, SEC EDGAR, FRED macro data → feature injectie |
| **SmartMoneyTracker** | `services/smart_money_tracker.py` | Insider clusters, institutional flow, whale activity, options P/C, dark pool → conviction score |
| **FeatureEngine v2** | `services/feature_engine.py` | 48 technische + alternatieve features met regime/alt-data injectie |
| **EnsembleLightGBM** | `ml/ensemble_lgbm.py` | 3-window ensemble (3m/1y/2y), SHAP uitleg, KL-drift detectie |
| **RecurrentPPO** | `ml/recurrent_ppo.py` | LSTM-gebaseerde RL agent (sb3-contrib) met curriculum training |
| **SimulatedBroker** | `services/simulated_broker.py` | Order book, 5 order types, volume-dependent slippage, margin, circuit breakers |
| **AutoTrader v2** | `services/auto_trader_v2.py` | 12-staps trading loop met regime-adaptieve sizing en smart money boosting |

### 12-staps Trading Cyclus

1. MarketScanner → dynamisch universum
2. Marktdata ophalen (OHLCV)
3. AlternativeData + SmartMoneyTracker (parallel)
4. RegimeClassifier per asset
5. FeatureEngine v2 (48 features met injecties)
6. EnsembleLightGBM (of standaard SignalPredictor)
7. Smart money conviction boost/demote
8. RecurrentPPO (of standaard PortfolioManager)
9. Regime-adaptieve position sizing (bull=1.2×, bear=0.5×, high_vol=0.6×)
10. Trade execution via SimulatedBroker
11. Housekeeping (snapshots, equity, broker day reset)
12. Drift detection (elke 20 cycli)

## Data bronnen

| Bron | Type | Markt |
|------|------|-------|
| Binance (ccxt) | OHLCV, volume | Crypto |
| yfinance | OHLCV, opties | US/EU stocks |
| Stooq | Historische OHLCV | US/EU stocks |
| SEC EDGAR | Form 4 insiders, 13F filings | US stocks |
| FRED | VIX, CPI, rentes | Macro |
| Blockchain.com | BTC whale transacties | Crypto |
| Etherscan | ETH whale transacties | Crypto |
| News RSS | Reuters, Yahoo, Seeking Alpha | Alle markten |
| FINRA ATS | Dark pool volumes | US stocks |

## Configuratie

### Verplicht
- `APP_USERNAME` — login gebruikersnaam
- `APP_PASSWORD` — login wachtwoord

### Optioneel
| Variabele | Default | Beschrijving |
|-----------|---------|--------------|
| `START_BALANCE` | `2000` | Startkapitaal in EUR |
| `AUTO_TRADE_V2_INTERVAL` | `5` | Trading cyclus interval (minuten) |
| `AUTO_TRADE_V2_DRY_RUN` | `false` | Dry run modus (geen echte trades) |
| `AUTO_TRADE_V2_MAX_TRADES` | `8` | Max trades per cyclus |
| `TRADING_DB_PATH` | `backend/data/trading.sqlite3` | SQLite pad |
| `TOKEN_SECRET` | auto-generated | JWT token secret |

### E-mail alerts
- `SMTP_HOST`, `SMTP_PORT` (default `587`), `SMTP_USERNAME`, `SMTP_PASSWORD`
- `ALERT_FROM_EMAIL`, `ALERT_TO_EMAIL`

## Lokaal starten

### Backend

```bash
cd backend
pip install -r requirements.txt
APP_USERNAME=admin APP_PASSWORD=secret uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Backtest draaien

```bash
cd backend

# Standaard (mixed preset: AAPL, MSFT, NVDA, TSLA, BTC, ETH)
python -m ml.training.backtest_v2

# Met preset
python -m ml.training.backtest_v2 --preset momentum --start 2023-01-01 --end 2024-12-31

# Custom symbolen
python -m ml.training.backtest_v2 --symbols AAPL TSLA BTC-USD ETH-USD --balance 25000

# Presets: momentum | crypto | mixed | eu
```

## Projectstructuur

```
backend/
├── main.py                          # FastAPI app, v2-integratie
├── requirements.txt
├── auth/
│   ├── security.py                  # Credentials validatie
│   └── session.py                   # JWT token management
├── models/
│   ├── holding.py, signal.py, trade.py
├── services/
│   ├── data_service.py              # SQLite CRUD, portfolio state
│   ├── market_data_service.py       # Stooq/Binance data layer
│   ├── feature_engine.py            # 48-feature engine (v2)
│   ├── signal_predictor.py          # LightGBM signaal classificatie
│   ├── portfolio_manager.py         # PPO RL portfolio manager
│   ├── trade_engine.py              # Trade execution (v1)
│   ├── auto_trader.py               # Auto-trader v1 (legacy)
│   ├── auto_trader_v2.py            # Auto-trader v2 (12-staps loop)
│   ├── market_scanner.py            # Dynamisch universum scanner
│   ├── alternative_data.py          # Alt-data collector
│   ├── smart_money_tracker.py       # Smart money signalen
│   └── simulated_broker.py          # Realistische broker simulatie
├── ml/
│   ├── config.py                    # ML configuratie + feature columns
│   ├── regime_classifier.py         # Marktregime classificatie
│   ├── sentiment_analyzer.py        # FinBERT NLP sentiment
│   ├── ensemble_lgbm.py             # Multi-window ensemble
│   ├── recurrent_ppo.py             # LSTM RL agent
│   ├── trading_env_v2.py            # Gymnasium trading environment
│   ├── models/                      # Opgeslagen modellen (.joblib)
│   └── training/
│       ├── train_signals.py         # Signaalmodel training
│       ├── train_ensemble.py        # Ensemble training (SMOTE + CV)
│       ├── train_recurrent_ppo.py   # 2-fase curriculum training
│       ├── train_portfolio.py       # PPO portfolio training
│       ├── backtest.py              # Backtest v1
│       └── backtest_v2.py           # Backtest v2 (regime-adaptief)
frontend/
├── src/
│   ├── App.tsx, main.tsx, api.ts
│   ├── components/EquityChart.tsx
│   └── pages/
│       ├── Dashboard.tsx, Overview.tsx
│       ├── Positions.tsx, Trades.tsx
│       ├── AIInsight.tsx, Management.tsx
│       └── LoginPage.tsx
```

## API endpoints

| Methode | Pad | Auth | Beschrijving |
|---------|-----|------|-------------|
| `GET` | `/health` | Nee | Health check |
| `POST` | `/login` | Nee | Inloggen → JWT token |
| `GET` | `/status` | Ja | Bot status + v2 module status |
| `GET` | `/portfolio/summary` | Ja | Portfolio overzicht |
| `GET` | `/portfolio/holdings` | Ja | Open posities |
| `GET` | `/portfolio/history` | Ja | Portfolio historie |
| `POST` | `/portfolio/deposit` | Ja | Geld storten |
| `POST` | `/portfolio/withdraw` | Ja | Geld opnemen |
| `GET` | `/trades/history` | Ja | Trade geschiedenis |
| `GET` | `/ai/signals` | Ja | Laatste signalen |
| `GET` | `/ai/insights` | Ja | AI analyse + feature importance |
| `POST` | `/settings/update` | Ja | Interval/auto-trade wijzigen |
| `GET` | `/auto-trader/summary` | Ja | Legacy compatibiliteit |

## Deployment

### Backend op Render
- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Zet minimaal `APP_USERNAME` en `APP_PASSWORD`

### Frontend op GitHub Pages
Gebruik `.github/workflows/deploy_frontend_pages.yml` en zet GitHub Pages op bron `gh-pages` / root.

## Graceful Degradation

Elke v2-module is optioneel. Als een module niet kan laden (ontbrekende dependency, API fout, etc.) wordt deze automatisch overgeslagen. De bot valt dan terug op de v1 equivalenten:

| V2 module | Fallback |
|-----------|----------|
| MarketScanner | Vaste watchlist uit MarketDataService |
| EnsembleLightGBM | Standaard SignalPredictor |
| RecurrentPPO | Standaard PortfolioManager (PPO) |
| SimulatedBroker | Standaard TradeEngine |
| RegimeClassifier | Heuristische regime-detectie |
| SentimentAnalyzer | Wordt overgeslagen |
| AlternativeData | Wordt overgeslagen |
| SmartMoneyTracker | Wordt overgeslagen |
