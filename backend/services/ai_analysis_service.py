"""AI-analyse service: gebruikt Groq (gratis) voor diepere signaalanalyse."""

import json
import logging
import os
import time
from typing import Dict, List
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.1-8b-instant"
_CACHE_TTL = 5 * 60  # 5 minuten


class AiAnalysisService:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict] = {}

    def _get_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def analyze_signals(
        self,
        signals: List[Dict],
        market_context: str = "",
    ) -> List[Dict]:
        """Analyseer een lijst signalen met Groq AI en retourneer verrijkte resultaten."""
        api_key = self._get_api_key()
        if not api_key:
            logger.warning("GROQ_API_KEY niet geconfigureerd; AI-analyse overgeslagen.")
            return []

        # Cache check
        cache_key = "ai:" + ",".join(s.get("symbol", "") for s in signals[:10])
        cached = self._get_cache(cache_key)
        if cached:
            return cached  # type: ignore[return-value]

        if not signals:
            return []

        # Bouw compacte signaal-summary voor de prompt
        signal_summaries = []
        for s in signals[:10]:
            signal_summaries.append(
                f"- {s['symbol']} ({s['market']}): prijs ${s['price']:.2f}, "
                f"actie={s['action']}, confidence={s['confidence']:.0%}, "
                f"RSI/MACD/BB info: {s.get('reason', 'n/a')}"
            )
        signals_text = "\n".join(signal_summaries)

        prompt = (
            "Je bent een professionele trading-analist. Analyseer de volgende top trading-signalen "
            "en geef per signaal een korte, concrete Nederlandse analyse (max 2 zinnen) met:\n"
            "1. Waarom dit een goed koopmoment is (of niet)\n"
            "2. Belangrijkste risico\n"
            "3. Geef een AI-score van 1-100\n\n"
            f"Marktcontext: {market_context or 'Reguliere handelsdag'}\n\n"
            f"Signalen:\n{signals_text}\n\n"
            "Antwoord ALLEEN in geldig JSON formaat als een array van objecten met velden: "
            '"symbol", "ai_score" (int 1-100), "ai_analysis" (string, Nederlands, max 2 zinnen), '
            '"ai_risk" (string, Nederlands, max 1 zin).\n'
            "Geen extra tekst buiten de JSON array."
        )

        try:
            result = self._call_groq(api_key, prompt)
            parsed = self._parse_response(result, signals)
            self._set_cache(cache_key, parsed)
            return parsed
        except Exception as exc:
            logger.error("Groq AI-analyse mislukt: %s", exc)
            return []

    def _call_groq(self, api_key: str, prompt: str) -> str:
        payload = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": "Je bent een professionele Nederlandse trading-analist."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        }).encode("utf-8")

        req = Request(
            _GROQ_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str, original_signals: List[Dict]) -> List[Dict]:
        """Parse Groq JSON response en merge met originele signalen."""
        # Strip eventuele markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            ai_items = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Kon Groq-response niet parsen als JSON")
            return []

        # Bouw lookup van AI-resultaten
        ai_map: Dict[str, Dict] = {}
        for item in ai_items:
            sym = item.get("symbol", "").upper()
            if sym:
                ai_map[sym] = item

        # Merge met originele signalen
        enriched: List[Dict] = []
        for signal in original_signals[:10]:
            sym = signal.get("symbol", "").upper()
            ai = ai_map.get(sym, {})
            enriched.append({
                **signal,
                "ai_score": ai.get("ai_score", 0),
                "ai_analysis": ai.get("ai_analysis", ""),
                "ai_risk": ai.get("ai_risk", ""),
            })

        # Sorteer op AI-score (hoogste eerst)
        enriched.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        return enriched

    def _set_cache(self, key: str, value: object) -> None:
        self._cache[key] = {
            "expires_at": time.time() + _CACHE_TTL,
            "value": value,
        }

    def _get_cache(self, key: str) -> object | None:
        item = self._cache.get(key)
        if not item:
            return None
        if item["expires_at"] < time.time():
            self._cache.pop(key, None)
            return None
        return item["value"]
