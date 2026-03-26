import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from models.holding import HoldingView
from models.signal import Signal
from services.data_service import DataService


class AlertService:
    def __init__(self, data_service: DataService) -> None:
        self._data_service = data_service
        self._last_realtime_sent: datetime | None = None
        self._last_summary_sent: datetime | None = None

    def send_realtime_alerts(self, signals: list[Signal], sell_advice: list[HoldingView]) -> dict:
        now = datetime.now(timezone.utc)
        if self._last_realtime_sent and now - self._last_realtime_sent < timedelta(minutes=15):
            return {"sent": False, "reason": "cooldown: wacht 15 minuten"}

        candidates = [s for s in signals if s.confidence >= 0.68][:8]
        transitions = self._collect_position_transitions(sell_advice)
        if not candidates and not transitions:
            return {"sent": False, "reason": "geen sterke signalen"}

        lines = ["Realtime marktalerts", ""]
        if candidates:
            lines.append("Beste koopkansen vandaag")
            for s in candidates:
                lines.append(
                    f"#{candidates.index(s) + 1} {s.symbol} ({s.market}) | {s.rank_label} | prijs {s.price:.2f} | "
                    f"verwacht {s.expected_return_pct:.2f}% | risico {s.risk_pct:.2f}% | "
                    f"score {s.ranking_score:.1f}"
                )
            lines.append("")

        if transitions:
            lines.append("Veranderd verkoopadvies voor jouw posities")
            for holding in transitions:
                lines.append(
                    f"{holding.symbol} ({holding.market}) | {holding.recommendation_label.upper()} | "
                    f"nu {holding.current_price:.2f} | P/L {holding.unrealized_profit_loss_pct:.2f}% | "
                    f"{holding.recommendation_reason}"
                )
        body = "\n".join(lines)

        self._send_email("AI Trading Alerts - Realtime", body)
        self._sync_recommendation_states(sell_advice)
        self._last_realtime_sent = now
        return {"sent": True, "buy_count": len(candidates), "transition_count": len(transitions)}

    def send_daily_summary(self, signals: list[Signal], sell_advice: list[HoldingView]) -> dict:
        now = datetime.now(timezone.utc)
        if self._last_summary_sent and now.date() == self._last_summary_sent.date():
            return {"sent": False, "reason": "dagelijkse samenvatting is al verzonden"}

        top = sorted(signals, key=lambda s: (s.confidence, s.expected_return_pct), reverse=True)[:10]
        if not top and not sell_advice:
            return {"sent": False, "reason": "geen signalen beschikbaar"}

        lines = ["Dagelijkse marktsamenvatting", ""]
        if top:
            lines.append("Beste koopkansen vandaag")
            for s in top:
                lines.append(
                    f"{s.symbol} ({s.market}) | {s.rank_label} | score {s.ranking_score:.1f} | "
                    f"verwacht {s.expected_return_pct:.2f}% | risico {s.risk_pct:.2f}%"
                )
            lines.append("")

        if sell_advice:
            lines.append("Verkoopadvies op je huidige posities")
            for holding in sell_advice[:10]:
                lines.append(
                    f"{holding.symbol} ({holding.market}) | {holding.recommendation_label.upper()} | "
                    f"P/L {holding.unrealized_profit_loss_pct:.2f}% | {holding.recommendation_reason}"
                )
        body = "\n".join(lines)

        self._send_email("AI Trading Alerts - Dagelijkse samenvatting", body)
        self._sync_recommendation_states(sell_advice)
        self._last_summary_sent = now
        return {"sent": True, "buy_count": len(top), "sell_count": len(sell_advice[:10])}

    def _collect_position_transitions(self, sell_advice: list[HoldingView]) -> list[HoldingView]:
        transitions: list[HoldingView] = []
        for holding in sell_advice:
            previous = self._data_service.get_last_recommendation(holding.symbol)
            current = holding.recommendation
            if current in {"sell_now", "take_partial_profit"} and previous != current:
                transitions.append(holding)
        return transitions

    def _sync_recommendation_states(self, sell_advice: list[HoldingView]) -> None:
        current_symbols = {holding.symbol for holding in sell_advice}
        for holding in sell_advice:
            self._data_service.set_last_recommendation(holding.symbol, holding.recommendation)

        for holding in self._data_service.get_holdings():
            symbol = str(holding["symbol"])
            if symbol not in current_symbols:
                self._data_service.clear_recommendation_state(symbol)

    def _send_email(self, subject: str, body: str) -> None:
        host = os.getenv("SMTP_HOST")
        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("ALERT_FROM_EMAIL") or username
        to_email = os.getenv("ALERT_TO_EMAIL", "gerritpaauw005@gmail.com")

        if not host or not username or not password or not from_email or not to_email:
            raise RuntimeError(
                "SMTP niet geconfigureerd. Zet SMTP_HOST, SMTP_PORT, SMTP_USERNAME, "
                "SMTP_PASSWORD, ALERT_FROM_EMAIL en ALERT_TO_EMAIL."
            )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.sendmail(from_email, [to_email], msg.as_string())