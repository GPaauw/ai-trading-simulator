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
            "symbol": non_hold["symbol"],
            "action": non_hold["action"],
            "amount": 200,
            "expected_return_pct": non_hold["expected_return_pct"],
            "risk_pct": non_hold["risk_pct"],
        },
    )
    assert trade_resp.status_code == 200, trade_resp.text

    portfolio_resp = client.get("/portfolio", headers=headers)
    assert portfolio_resp.status_code == 200, portfolio_resp.text

    learn_resp = client.post("/learn", headers=headers)
    assert learn_resp.status_code == 200, learn_resp.text

    print("Smoke test OK")


if __name__ == "__main__":
    main()
