# AI Market Advisor (US, EU en Crypto)

Deze app geeft marktadvies in plaats van random trades:
- FastAPI-backend met live gratis data
- Flutter Web-frontend met adviesdashboard
- Markten: US stocks, EU stocks en crypto
- Marktadvies toont alleen koopkansen, opgesplitst in aandelen en crypto
- Koopadvies krijgt een echte score en labels zoals `Beste kansen vandaag`
- Registratie van echte aankopen met live holdings-overzicht
- Verkoopadvies toont alleen je eigen bezittingen met `houden` of `verkopen`
- Verkoopadvies ondersteunt triggers zoals `Verkoop nu`, `Neem deels winst` en `Houd nog vast`
- Positie-alerts kunnen mailen zodra een advies omslaat naar verkopen of winst nemen
- Holdings, tradehistorie en cash-balans worden persistent opgeslagen in SQLite
- E-mailalerts (realtime + dagoverzicht)

## Data bronnen (gratis)

- US/EU stocks: Stooq publieke endpoints
- Crypto: Binance publieke endpoints

Let op: gratis bronnen zijn prima voor MVP/paper trading, maar minder geschikt voor low-latency productiehandel.

## Watchlist (10)

- US: `AAPL`, `MSFT`, `NVDA`, `AMZN`
- EU: `ASML`, `SAP`, `SIE`, `BMW`
- Crypto: `BTC`, `ETH`

## Login en basisinstellingen

Verplicht:
- `APP_USERNAME`
- `APP_PASSWORD`

Optioneel:
- `START_BALANCE` (default: `2000`)
- `TRADING_DB_PATH` (default: `backend/data/trading.sqlite3`)

## E-mail alerts configuratie

Voor realtime alerts en dagelijkse samenvatting:
- `SMTP_HOST`
- `SMTP_PORT` (default `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `ALERT_FROM_EMAIL`
- `ALERT_TO_EMAIL` (default `gerritpaauw005@gmail.com`)

## Lokaal starten

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
APP_USERNAME APP_PASSWORD START_BALANCE=2000 uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
flutter pub get
flutter run -d edge
```

## Hoe je de app gebruikt

1. Log in met je account.
2. Bekijk `Koopadvies`: alleen koopkansen, opgesplitst in aandelen en crypto, met score en ranking.
3. Klik bij een `BUY`-advies op `Ik kocht dit` en vul in wat je echt hebt gekocht.
4. Bekijk `Verkoopadvies voor jouw posities`: alleen assets die je bezit, met `Verkoop nu`, `Neem deels winst` of `Houd nog vast`.
5. Gebruik `Verkoop` of `Neem winst` bij een positie; de app vult bij gedeeltelijke winstname alvast een logisch deel van je positie in.
6. Volg je resultaat in `Portfolio` en `Transactiegeschiedenis`.
7. Klik op `Leer van markt + trades` om risico- en confidence-inzichten te updaten.
8. Klik op `Stuur realtime alerts` of `Stuur dag samenvatting` voor e-mailnotificaties, inclusief statuswissels in je posities.

## API endpoints

Publiek:
- `GET /health`
- `GET /signals`
- `GET /advice` (alleen koopadvies)
- `GET /watchlist`

Auth vereist:
- `POST /login`
- `POST /trade`
- `GET /history`
- `GET /holdings`
- `GET /sell-advice`
- `GET /portfolio`
- `POST /learn`
- `POST /alerts/realtime`
- `POST /alerts/summary`

## Deployment

### Backend op Render

- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Zet minimaal `APP_USERNAME` en `APP_PASSWORD`

### Frontend op GitHub Pages

Gebruik [.github/workflows/deploy_frontend_pages.yml](.github/workflows/deploy_frontend_pages.yml) en zet GitHub Pages op bron `gh-pages` / root.

Als de frontend met een publieke backend moet praten, zet dan in de frontend-config de juiste backend-URL of voeg die later als build-time configuratie toe.

Benodigd om de Pages-versie echt werkend te krijgen:
- GitHub Actions moet aan staan voor de repository.
- In GitHub Pages: kies `Deploy from a branch`, branch `gh-pages`, folder `/ (root)`.
- De workflow bouwt met de juiste base-href voor deze repo, zodat assets laden onder `/ai-trading-simulator/`.
- De backend moet publiek bereikbaar zijn via HTTPS. De frontend gebruikt standaard `https://ai-trading-simulator.onrender.com`.
- Op Render moeten minimaal `APP_USERNAME` en `APP_PASSWORD` ingesteld zijn.
- Voor e-mailalerts zijn extra variabelen nodig: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `ALERT_FROM_EMAIL`, `ALERT_TO_EMAIL`.
- Aanbevolen extra backend variabelen: `TOKEN_SECRET` voor stateless auth en `START_BALANCE=2000`.
