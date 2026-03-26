# AI Market Advisor (US, EU en Crypto)

Deze app geeft marktadvies in plaats van random trades:
- FastAPI-backend met live gratis data
- Flutter Web-frontend met adviesdashboard
- Markten: US stocks, EU stocks en crypto
- Advies per instrument: `buy/sell/hold`, risico%, verwachte winst%
- Paper trading met startkapitaal en daglimiet
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
APP_USERNAME=admin APP_PASSWORD=trading123 START_BALANCE=2000 uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
flutter pub get
flutter run -d chrome
```

## Hoe je de app gebruikt

1. Log in met je account.
2. Bekijk `Marktadvies`: per asset zie je actie, risico% en verwachte winst%.
3. Klik op `Paper trade` bij een buy/sell advies om de strategie te simuleren.
4. Volg je resultaat in `Portfolio` en `Paper Trade Geschiedenis`.
5. Klik op `Leer van markt + trades` om risico- en confidence-inzichten te updaten.
6. Klik op `Stuur realtime alerts` of `Stuur dag samenvatting` voor e-mailnotificaties.

## API endpoints

Publiek:
- `GET /health`
- `GET /signals`
- `GET /advice`
- `GET /watchlist`

Auth vereist:
- `POST /login`
- `POST /trade`
- `GET /history`
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

Gebruik `.github/workflows/deploy.yml` en zet repository variable:
- `BACKEND_URL` = publieke backend-URL