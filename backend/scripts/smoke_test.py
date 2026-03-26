"""Kleine smoke-test voor de nieuwe market-advisor flow.

Run:
  APP_USERNAME=admin APP_PASSWORD=trading123 python scripts/smoke_test.py
"""

import os
import tempfile

from fastapi.testclient import TestClient

db_path = os.path.join(tempfile.gettempdir(), "ai_trading_smoke.sqlite3")
if os.path.exists(db_path):
        os.remove(db_path)
os.environ.setdefault("TRADING_DB_PATH", db_path)

from main import app
from services.data_service import DataService


def main() -> None:
    client = TestClient(app)

    signals_resp = client.get("/signals")
    assert signals_resp.status_code == 200, signals_resp.text
    signals = signals_resp.json()
    assert signals, "Geen signalen ontvangen"

    buy_advice_resp = client.get("/advice")
    assert buy_advice_resp.status_code == 200, buy_advice_resp.text
    buy_advice = buy_advice_resp.json()
    assert all(item["action"] == "buy" for item in buy_advice), buy_advice_resp.text

    login_resp = client.post("/login", json={"username": "admin", "password": "trading123"})
    assert login_resp.status_code == 200, login_resp.text
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    candidate = buy_advice[0] if buy_advice else signals[0]

    trade_resp = client.post(
        "/trade",
        headers=headers,
        json={
            "symbol": candidate["symbol"],
            "action": "buy",
            "amount": 200,
            "market": candidate["market"],
            "price": candidate["price"],
            "expected_return_pct": candidate["expected_return_pct"],
            "risk_pct": candidate["risk_pct"],
        },
    )
    assert trade_resp.status_code == 200, trade_resp.text
    assert trade_resp.json()["status"].startswith("executed"), trade_resp.text

    sell_advice_resp = client.get("/sell-advice", headers=headers)
    assert sell_advice_resp.status_code == 200, sell_advice_resp.text
    assert len(sell_advice_resp.json()) == 1, sell_advice_resp.text

    holdings_resp = client.get("/holdings", headers=headers)
    assert holdings_resp.status_code == 200, holdings_resp.text
    holdings = holdings_resp.json()
    assert holdings, "Geen holdings gevonden na buy"
    assert holdings[0]["recommendation"] in {"hold", "sell_now", "take_partial_profit"}
    rounded_quantity = holdings[0]["quantity"]

    reloaded_service = DataService()
    persisted_holdings = reloaded_service.get_holdings()
    assert persisted_holdings, "Holdings zijn niet persistent opgeslagen"

    sell_resp = client.post(
        "/trade",
        headers=headers,
        json={
            "symbol": candidate["symbol"],
            "action": "sell",
            "amount": 0,
            "quantity": rounded_quantity,
            "market": candidate["market"],
            "price": candidate["price"],
        },
    )
    assert sell_resp.status_code == 200, sell_resp.text
    assert sell_resp.json()["status"].startswith("executed"), sell_resp.text

    holdings_after_sell_resp = client.get("/holdings", headers=headers)
    assert holdings_after_sell_resp.status_code == 200, holdings_after_sell_resp.text
    assert not holdings_after_sell_resp.json(), "Holding niet gesloten na sell"

    sell_advice_after_sell_resp = client.get("/sell-advice", headers=headers)
    assert sell_advice_after_sell_resp.status_code == 200, sell_advice_after_sell_resp.text
    assert not sell_advice_after_sell_resp.json(), "Sell advice moet leeg zijn zonder holdings"

    reloaded_after_sell = DataService()
    assert not reloaded_after_sell.get_holdings(), "Gesloten holding staat nog in SQLite"

    portfolio_resp = client.get("/portfolio", headers=headers)
    assert portfolio_resp.status_code == 200, portfolio_resp.text

    learn_resp = client.post("/learn", headers=headers)
    assert learn_resp.status_code == 200, learn_resp.text

    print("Smoke test OK")


if __name__ == "__main__":
    main()
