import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from models.signal import Signal


class AlertService:
    def __init__(self) -> None:
        self._last_realtime_sent: datetime | None = None
        self._last_summary_sent: datetime | None = None

    def send_realtime_alerts(self, signals: list[Signal]) -> dict:
        now = datetime.now(timezone.utc)
        if self._last_realtime_sent and now - self._last_realtime_sent < timedelta(minutes=15):
            return {"sent": False, "reason": "cooldown: wacht 15 minuten"}

        candidates = [s for s in signals if s.action != "hold" and s.confidence >= 0.68]
        if not candidates:
            return {"sent": False, "reason": "geen sterke signalen"}

        lines = ["Realtime marktalerts", ""]
        for s in candidates[:8]:
            lines.append(
                f"{s.symbol} ({s.market}) | {s.action.upper()} | prijs {s.price:.2f} | "
                f"verwacht {s.expected_return_pct:.2f}% | risico {s.risk_pct:.2f}% | "
                f"zekerheid {s.confidence * 100:.1f}%"
            )
        body = "\n".join(lines)

        self._send_email("AI Trading Alerts - Realtime", body)
        self._last_realtime_sent = now
        return {"sent": True, "count": len(candidates[:8])}

    def send_daily_summary(self, signals: list[Signal]) -> dict:
        now = datetime.now(timezone.utc)
        if self._last_summary_sent and now.date() == self._last_summary_sent.date():
            return {"sent": False, "reason": "dagelijkse samenvatting is al verzonden"}

        top = sorted(signals, key=lambda s: (s.confidence, s.expected_return_pct), reverse=True)[:10]
        if not top:
            return {"sent": False, "reason": "geen signalen beschikbaar"}

        lines = ["Dagelijkse marktsamenvatting", ""]
        for s in top:
            lines.append(
                f"{s.symbol} ({s.market}) | {s.action.upper()} | verwacht {s.expected_return_pct:.2f}% | "
                f"risico {s.risk_pct:.2f}% | zekerheid {s.confidence * 100:.1f}%"
            )
        body = "\n".join(lines)

        self._send_email("AI Trading Alerts - Dagelijkse samenvatting", body)
        self._last_summary_sent = now
        return {"sent": True, "count": len(top)}

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