# AI Trading Simulator

Eenvoudige trading-simulator met:
- FastAPI-backend
- Flutter Web-frontend
- login met gebruikersnaam en wachtwoord
- dummy signals, trades, history en learning

## Standaard login

- Gebruikersnaam: `admin`
- Wachtwoord: `trading123`

Je kunt deze later aanpassen via environment variables:
- `APP_USERNAME`
- `APP_PASSWORD`

## Lokaal starten

### 1. Backend starten

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

De backend draait dan op `http://127.0.0.1:8000`.

### 2. Frontend starten

Vereist: Flutter SDK met web support.

```bash
cd frontend
flutter pub get
flutter run -d chrome
```

De frontend is al ingesteld om lokaal naar `http://127.0.0.1:8000` te praten.

## Gebruiken

1. Open de app in de browser.
2. Log in met `admin` / `trading123`.
3. Bekijk de signalen op het dashboard.
4. Klik op `Execute` om een dummy trade uit te voeren.
5. Klik op `Leer van Trades` om de learning-agent bij te werken.
6. Bekijk onderaan de trade history.

## Deployment

### Backend op Render

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Environment variables:
	- `APP_USERNAME`
	- `APP_PASSWORD`

### Frontend op GitHub Pages

De workflow staat in `.github/workflows/deploy.yml`.
Voor productie hoef je `frontend/lib/api_client.dart` niet handmatig te wijzigen.
Zet in GitHub onder `Settings > Secrets and variables > Actions > Variables` een repository variable:

- `BACKEND_URL` = je publieke backend-URL

Daarna bouwt de workflow de frontend automatisch met die backend-URL.