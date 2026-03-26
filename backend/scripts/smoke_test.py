"""Kleine smoke-test voor de nieuwe market-advisor flow.

Run:
  APP_USERNAME=admin APP_PASSWORD=trading123 python scripts/smoke_test.py
"""

from fastapi.testclient import TestClient

from main import app


def main() -> None:
    client = TestClient(app)

    signals_resp = client.get("/advice")
    assert signals_resp.status_code == 200, signals_resp.text
    signals = signals_resp.json()
    assert signals, "Geen adviezen ontvangen"

    login_resp = client.post("/login", json={"username": "admin", "password": "trading123"})
    assert login_resp.status_code == 200, login_resp.text
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    non_hold = next((s for s in signals if s["action"] != "hold"), None)
    assert non_hold is not None, "Alle signalen zijn hold"

    trade_resp = client.post(
        "/trade",
        headers=headers,
        json={
            "symbol": signals[0]["symbol"],
            "action": "buy",
            "amount": 200,
            "market": signals[0]["market"],
            "price": signals[0]["price"],
            "expected_return_pct": signals[0]["expected_return_pct"],
            "risk_pct": signals[0]["risk_pct"],
        },
    )
    assert trade_resp.status_code == 200, trade_resp.text
    assert trade_resp.json()["status"].startswith("executed"), trade_resp.text

    holdings_resp = client.get("/holdings", headers=headers)
    assert holdings_resp.status_code == 200, holdings_resp.text
    holdings = holdings_resp.json()
    assert holdings, "Geen holdings gevonden na buy"

    sell_resp = client.post(
        "/trade",
        headers=headers,
        json={
            "symbol": signals[0]["symbol"],
            "action": "sell",
            "amount": 0,
            "market": signals[0]["market"],
            "price": signals[0]["price"],
        },
    )
    assert sell_resp.status_code == 200, sell_resp.text
    assert sell_resp.json()["status"].startswith("executed"), sell_resp.text

    holdings_after_sell_resp = client.get("/holdings", headers=headers)
    assert holdings_after_sell_resp.status_code == 200, holdings_after_sell_resp.text
    assert not holdings_after_sell_resp.json(), "Holding niet gesloten na sell"

    portfolio_resp = client.get("/portfolio", headers=headers)
    assert portfolio_resp.status_code == 200, portfolio_resp.text

    learn_resp = client.post("/learn", headers=headers)
    assert learn_resp.status_code == 200, learn_resp.text

    print("Smoke test OK")


if __name__ == "__main__":
    main()
