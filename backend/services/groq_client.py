"""Groq LLM client voor de autonome trading-loop.

Gebruikt Llama 3.3 70B via Groq's gratis API (6.000 RPD).
Enkel bedoeld voor de auto-trader; de gebruikers-UI blijft Gemini gebruiken.
"""

import json
import logging
import os
import time
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_REQUEST_TIMEOUT = 30
_RATE_LIMIT_COOLDOWN = 60


class GroqClient:
    def __init__(self) -> None:
        self._rate_limited_until: float = 0.0

    def _get_api_key(self) -> str | None:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        return key if key else None

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def is_rate_limited(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    def query(self, system_prompt: str, user_prompt: str) -> str | None:
        """Stuur een prompt naar Groq en retourneer het antwoord als string.

        Retourneert None als de API niet beschikbaar is of een fout optreedt.
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.warning("GROQ_API_KEY ontbreekt; auto-trader AI niet beschikbaar.")
            return None

        if self.is_rate_limited():
            remaining = int(self._rate_limited_until - time.monotonic())
            logger.info("Groq rate-limit actief; nog %ds wachten.", remaining)
            return None

        payload = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 2000,
        }).encode("utf-8")

        req = Request(
            _GROQ_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ai-trading-simulator/2.0",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                self._rate_limited_until = time.monotonic() + _RATE_LIMIT_COOLDOWN
                logger.warning("Groq 429; gepauzeerd voor %ds.", _RATE_LIMIT_COOLDOWN)
            else:
                logger.error("Groq HTTP-fout %s: %s", exc.code, body[:300])
            return None
        except URLError as exc:
            logger.error("Groq netwerkfout: %s", exc)
            return None
        except Exception as exc:
            logger.error("Groq onverwachte fout: %s", exc)
            return None

    def decide_trades(
        self,
        signals: List[Dict],
        holdings: List[Dict],
        cash: float,
        trade_history_summary: str,
        performance_summary: str,
    ) -> List[Dict]:
        """Vraag het AI model om concrete trade-beslissingen.

        Retourneert een lijst van dicts met: symbol, action, fraction (0.0-1.0), reason.
        """
        system_prompt = (
            "Je bent een autonome AI trading-bot die leert van eerdere resultaten. "
            "Je doel: maximaliseer winst over 7 dagen met een startkapitaal van €2000. "
            "Je handelt in aandelen, crypto en grondstoffen. "
            "REGELS:\n"
            "- Maximaal 30% van je cash in één trade\n"
            "- Altijd een stop-loss overwegen (max 3% verlies per positie)\n"
            "- Verkoop verliezers snel, laat winnaars doorlopen\n"
            "- Leer van fouten: vermijd patronen die eerder verlies gaven\n"
            "- Antwoord ALLEEN in geldig JSON\n"
        )

        holdings_text = "Geen" if not holdings else "\n".join(
            f"- {h['symbol']} ({h.get('market','?')}): {h['quantity']:.4f} stuks, "
            f"inleg €{h.get('invested_amount', 0):.2f}, gem. prijs €{h.get('avg_entry_price', 0):.2f}"
            for h in holdings
        )

        signals_text = "\n".join(
            f"- {s.get('symbol','?')} ({s.get('market','?')}): prijs=${s.get('price',0):.2f}, "
            f"actie={s.get('action','?')}, confidence={s.get('confidence',0):.0%}, "
            f"verwacht rendement={s.get('expected_return_pct',0):.1f}%, "
            f"risico={s.get('risk_pct',0):.1f}%, reden: {s.get('reason','n/a')}"
            for s in signals[:15]
        )

        user_prompt = (
            f"BESCHIKBAAR CASH: €{cash:.2f}\n\n"
            f"HUIDIGE POSITIES:\n{holdings_text}\n\n"
            f"MARKTSIGNALEN:\n{signals_text}\n\n"
            f"EERDERE RESULTATEN:\n{trade_history_summary}\n\n"
            f"PERFORMANCE:\n{performance_summary}\n\n"
            "Geef je beslissingen als JSON:\n"
            '{\n'
            '  "decisions": [\n'
            '    {\n'
            '      "symbol": "AAPL",\n'
            '      "action": "buy",\n'
            '      "fraction": 0.2,\n'
            '      "reason": "Sterke momentum..."\n'
            '    }\n'
            '  ],\n'
            '  "strategy_note": "Korte uitleg van je strategie"\n'
            '}\n\n'
            "fraction = deel van beschikbaar cash (buy) of deel van positie (sell).\n"
            "Geef 0-5 beslissingen. Geef een lege lijst als niets aantrekkelijk is."
        )

        raw = self.query(system_prompt, user_prompt)
        if not raw:
            return []

        try:
            payload = json.loads(raw)
            decisions = payload.get("decisions", [])
            if not isinstance(decisions, list):
                return []
            # Valideer elke beslissing
            valid: List[Dict] = []
            for d in decisions:
                symbol = str(d.get("symbol", "")).upper()
                action = str(d.get("action", "")).lower()
                fraction = float(d.get("fraction", 0))
                reason = str(d.get("reason", ""))
                if symbol and action in ("buy", "sell") and 0 < fraction <= 1.0:
                    valid.append({
                        "symbol": symbol,
                        "action": action,
                        "fraction": min(fraction, 0.3) if action == "buy" else fraction,
                        "reason": reason,
                    })
            strategy = str(payload.get("strategy_note", ""))
            if strategy:
                logger.info("AI strategie: %s", strategy)
            return valid
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Kon Groq trade-beslissing niet parsen: %s", exc)
            return []
