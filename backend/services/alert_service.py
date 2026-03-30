import json
import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from urllib.request import Request, urlopen

from models.holding import HoldingView
from models.signal import Signal
from services.data_service import DataService


class AlertService:
    """Stuurt alerts via Telegram (gratis) of SMTP als fallback."""

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

        lines = ["🔔 *Realtime marktalerts*", ""]
        if candidates:
            lines.append("📈 *Beste koopkansen:*")
            for i, s in enumerate(candidates, 1):
                lines.append(
                    f"#{i} *{s.symbol}* ({s.market}) | {s.rank_label}\n"
                    f"   Prijs: €{s.price:.2f} → Doel: €{s.target_price:.2f}\n"
                    f"   Verwacht: +{s.expected_return_pct:.2f}% (~€{s.expected_profit:.0f}/€1000)\n"
                    f"   Tijdshorizon: ~{s.expected_days}d | Score: {s.ranking_score:.1f}"
                )
            lines.append("")

        if transitions:
            lines.append("⚠️ *Veranderd verkoopadvies:*")
            for holding in transitions:
                lines.append(
                    f"*{holding.symbol}* ({holding.market}) | {holding.recommendation_label.upper()}\n"
                    f"   Nu €{holding.current_price:.2f} | P/L {holding.unrealized_profit_loss_pct:+.2f}%\n"
                    f"   {holding.recommendation_reason}"
                )
        body = "\n".join(lines)

        self._send_message("AI Trading - Realtime", body)
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

        lines = ["📊 *Dagelijkse marktsamenvatting*", ""]
        if top:
            lines.append("📈 *Top 10 koopkansen:*")
            for i, s in enumerate(top, 1):
                lines.append(
                    f"#{i} *{s.symbol}* ({s.market}) | Score: {s.ranking_score:.1f}\n"
                    f"   Prijs: €{s.price:.2f} → Doel: €{s.target_price:.2f} in ~{s.expected_days}d\n"
                    f"   Verwacht: +{s.expected_return_pct:.2f}% | Risico: {s.risk_pct:.2f}%"
                )
            lines.append("")

        if sell_advice:
            lines.append("🔴 *Verkoopadvies:*")
            for holding in sell_advice[:10]:
                lines.append(
                    f"*{holding.symbol}* ({holding.market}) | {holding.recommendation_label.upper()}\n"
                    f"   P/L {holding.unrealized_profit_loss_pct:+.2f}% | {holding.recommendation_reason}"
                )
        body = "\n".join(lines)

        self._send_message("AI Trading - Dagelijks", body)
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

    # ------------------------------------------------------------------
    # Message dispatch: Telegram (primair, gratis) of SMTP (fallback)
    # ------------------------------------------------------------------

    def _send_message(self, subject: str, body: str) -> None:
        """Probeert eerst Telegram, valt terug naar SMTP."""
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if telegram_token and telegram_chat_id:
            self._send_telegram(telegram_token, telegram_chat_id, body)
            return

        # Fallback naar SMTP
        self._send_email(subject, body)

    def _send_telegram(self, token: str, chat_id: str, text: str) -> None:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if not result.get("ok"):
                raise RuntimeError(f"Telegram API fout: {result}")

    def _send_email(self, subject: str, body: str) -> None:
        host = os.getenv("SMTP_HOST")
        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("ALERT_FROM_EMAIL") or username
        to_email = os.getenv("ALERT_TO_EMAIL")

        if not host or not username or not password or not from_email or not to_email:
            raise RuntimeError(
                "Geen notificatie-kanaal geconfigureerd. "
                "Stel TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in (gratis), "
                "of SMTP_HOST + SMTP_USERNAME + SMTP_PASSWORD + ALERT_TO_EMAIL."
            )

        # Strip markdown voor email
        plain_body = body.replace("*", "").replace("_", "")
        msg = MIMEText(plain_body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.sendmail(from_email, [to_email], msg.as_string())
