"""AI-analyse service: primaire analyse-engine via Groq (gratis LLM).

Alle signalen gaan door het AI model. Het model leert van eerdere trades
(winst/verlies) en past zijn adviezen daarop aan.
Fallback: als GROQ_API_KEY ontbreekt worden technische signalen ongewijzigd doorgestuurd.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
_CACHE_TTL = 5 * 60  # 5 minuten
_REQUEST_TIMEOUT_SECONDS = 30


class AiAnalysisService:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict] = {}
        self._trade_lessons: str = ""

    def _get_api_key(self) -> str | None:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        return key if key else None

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    # ------------------------------------------------------------------
    # Leren van trade-historie
    # ------------------------------------------------------------------

    def update_trade_lessons(self, trades: list) -> str:
        """Analyseer afgelopen trades en genereer lessen voor toekomstige analyses."""
        trade_dicts: List[Dict] = []
        for t in trades:
            if hasattr(t, "model_dump"):
                trade_dicts.append(t.model_dump())
            elif isinstance(t, dict):
                trade_dicts.append(t)

        if not trade_dicts:
            self._trade_lessons = ""
            return ""

        recent = trade_dicts[-30:]
        winners = [t for t in recent if (t.get("profit_loss") or 0) > 0]
        losers = [t for t in recent if (t.get("profit_loss") or 0) < 0]
        total_pl = sum(t.get("profit_loss", 0) for t in recent)

        parts: list[str] = [
            f"Totaal trades: {len(recent)}, winnaars: {len(winners)}, "
            f"verliezers: {len(losers)}, netto P/L: €{total_pl:.2f}.",
        ]

        top_winners = sorted(winners, key=lambda t: t.get("profit_loss", 0), reverse=True)[:5]
        top_losers = sorted(losers, key=lambda t: t.get("profit_loss", 0))[:5]

        if top_winners:
            w = ", ".join(f"{t['symbol']} (+€{t['profit_loss']:.2f})" for t in top_winners)
            parts.append(f"Beste trades: {w}.")
        if top_losers:
            l = ", ".join(f"{t['symbol']} (€{t['profit_loss']:.2f})" for t in top_losers)
            parts.append(f"Slechtste trades: {l}.")

        if len(losers) > len(winners) and len(recent) >= 5:
            parts.append("WAARSCHUWING: meer verliezers dan winnaars — wees selectiever.")
        if losers:
            avg_loss = sum(t.get("profit_loss", 0) for t in losers) / len(losers)
            parts.append(f"Gemiddeld verlies per verliezer: €{avg_loss:.2f}.")
        if winners:
            avg_win = sum(t.get("profit_loss", 0) for t in winners) / len(winners)
            parts.append(f"Gemiddelde winst per winnaar: €{avg_win:.2f}.")

        # Symbolen die herhaaldelijk verlies gaven
        loss_counts: Dict[str, int] = {}
        for t in losers:
            sym = t.get("symbol", "")
            loss_counts[sym] = loss_counts.get(sym, 0) + 1
        repeat = [f"{s} ({c}x)" for s, c in loss_counts.items() if c >= 2]
        if repeat:
            parts.append(f"Herhaalde verliezers (VERMIJD): {', '.join(repeat)}.")

        self._trade_lessons = " ".join(parts)
        return self._trade_lessons

    # ------------------------------------------------------------------
    # Primaire AI-analyse (verrijkt technische signalen)
    # ------------------------------------------------------------------

    def analyze_signals(
        self,
        signals: List[Dict],
        trade_history: Optional[list] = None,
        market_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Primaire analyse: het AI model analyseert technische signalen,
        leert van trade-historie, en geeft verbeterde adviezen.

        Als GROQ_API_KEY ontbreekt: retourneer originele signalen met lege AI-velden.
        """
        # Update lessen als er trade-historie is
        if trade_history:
            self.update_trade_lessons(trade_history)

        # Filter op markt als nodig
        if market_filter:
            signals = [s for s in signals if s.get("market") == market_filter]

        if not signals:
            return []

        api_key = self._get_api_key()
        if not api_key:
            logger.warning("GROQ_API_KEY ontbreekt; technische fallback wordt gebruikt.")
            return self._build_fallback_signals(signals, "AI-key ontbreekt; technische analyse wordt gebruikt.")

        # Cache check
        cache_key = self._make_cache_key(signals, market_filter)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Bouw prompt en roep Groq aan
        prompt = self._build_prompt(signals)

        try:
            raw = self._call_groq(api_key, prompt)
            results = self._parse_response(raw, signals)
            if results:
                self._set_cache(cache_key, results)
            return results
        except Exception as exc:
            logger.error("Groq AI-analyse mislukt: %s", exc)
            return self._build_fallback_signals(signals, "AI-analyse tijdelijk niet beschikbaar; technische analyse wordt gebruikt.")

    def _build_prompt(self, signals: List[Dict]) -> str:
        """Bouw prompt met marktdata + lessen uit eerdere trades."""
        data_lines = []
        for s in signals[:15]:
            price = s.get("price", 0)
            data_lines.append(
                f"- {s.get('symbol', '?')} ({s.get('market', '?')}): "
                f"prijs=${price:.2f}, technisch_advies={s.get('action', '?')}, "
                f"confidence={s.get('confidence', 0):.0%}, "
                f"technische_analyse: {s.get('reason', 'n/a')}"
            )
        data_text = "\n".join(data_lines)

        lessons_block = ""
        if self._trade_lessons:
            lessons_block = (
                f"\n\nBELANGRIJK — Leer van mijn eerdere trades:\n"
                f"{self._trade_lessons}\n"
                "Gebruik deze informatie om betere aanbevelingen te doen. "
                "Vermijd symbolen die herhaaldelijk verlies gaven. "
                "Focus op patronen die eerder winst opleverden.\n"
            )

        return (
            "Je bent een expert trading-analist. Analyseer de volgende instrumenten "
            "en geef concrete koop/verkoop adviezen in het Nederlands.\n\n"
            "REGELS:\n"
            "- Analyseer ELKE aangeboden asset\n"
            "- Geef een AI-score van 1-100 (hoger = sterker koopsignaal)\n"
            "- Wees eerlijk: als het geen goed moment is, zeg dat\n"
            "- Geef korte concrete Nederlandse analyse (max 2 zinnen)\n"
            "- Geef het belangrijkste risico (max 1 zin)\n"
            "- Bepaal actie: 'buy', 'sell' of 'hold'\n"
            "- Geef een koersdoel (target_price) en verwacht rendement in %\n"
            f"{lessons_block}\n"
            f"MARKTDATA:\n{data_text}\n\n"
            "Antwoord ALLEEN in geldig JSON als een object met sleutel \"analyses\" en daarin een array:\n"
            '{\n'
            '  "analyses": [\n'
            '    {\n'
            '      "symbol": "AAPL",\n'
            '      "action": "buy",\n'
            '      "ai_score": 85,\n'
            '      "ai_analysis": "Sterke technische setup met...",\n'
            '      "ai_risk": "Risico op...",\n'
            '      "target_price": 155.50,\n'
            '      "expected_return_pct": 2.5,\n'
            '      "confidence": 0.82\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "Geen extra tekst buiten het JSON object. Sorteer analyses op ai_score (hoogste eerst)."
        )

    # ------------------------------------------------------------------
    # Groq API
    # ------------------------------------------------------------------

    def _call_groq(self, api_key: str, prompt: str) -> str:
        payload = json.dumps({
            "model": _MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Je bent een professionele Nederlandse trading-analist. "
                        "Je leert van eerdere fouten en geeft steeds betere adviezen. "
                        "Antwoord altijd in geldig JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 2000,
        }).encode("utf-8")

        req = Request(
            _GROQ_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ai-trading-simulator/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            logger.error("Groq HTTP-fout %s: %s", exc.code, error_body)
            raise RuntimeError(f"Groq HTTP-fout {exc.code}") from exc
        except URLError as exc:
            logger.error("Groq netwerkfout: %s", exc)
            raise RuntimeError("Groq netwerkfout") from exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, original_signals: List[Dict]) -> List[Dict]:
        """Parse Groq JSON en merge AI-analyse met originele signalen."""
        cleaned = self._extract_json_payload(raw)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Kon Groq-response niet parsen: %s", cleaned[:200])
            return self._build_fallback_signals(original_signals, "AI-response kon niet worden gelezen; technische analyse wordt gebruikt.")

        if isinstance(payload, dict):
            ai_items = payload.get("analyses") or payload.get("signals") or payload.get("results") or []
        elif isinstance(payload, list):
            ai_items = payload
        else:
            ai_items = []

        if not isinstance(ai_items, list):
            return self._build_fallback_signals(original_signals, "AI-response had ongeldig formaat; technische analyse wordt gebruikt.")

        # Lookup op symbool
        ai_map: Dict[str, Dict] = {}
        for item in ai_items:
            sym = str(item.get("symbol", "")).upper()
            if sym:
                ai_map[sym] = item

        results: List[Dict] = []
        for signal in original_signals:
            sym = signal.get("symbol", "").upper()
            ai = ai_map.get(sym, {})
            ai_score = self._coerce_score(ai.get("ai_score", 0), signal)

            merged = {**signal}
            merged["ai_score"] = ai_score
            merged["ai_analysis"] = str(ai.get("ai_analysis", signal.get("reason", "")))
            merged["ai_risk"] = str(ai.get("ai_risk", "Normaal marktrisico; controleer positieomvang en stop-loss."))

            # Als AI een andere actie aanbeveelt, neem AI-advies over
            ai_action = ai.get("action")
            if ai_action in ("buy", "sell", "hold"):
                merged["action"] = ai_action
            if ai.get("confidence") is not None:
                merged["confidence"] = float(ai["confidence"])
            if ai.get("target_price") is not None:
                merged["target_price"] = float(ai["target_price"])
            if ai.get("expected_return_pct") is not None:
                merged["expected_return_pct"] = float(ai["expected_return_pct"])

            merged["rank_label"] = self._score_to_label(ai_score)
            results.append(merged)

        results.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        return results

    @staticmethod
    def _extract_json_payload(raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        if cleaned.startswith("{") or cleaned.startswith("["):
            return cleaned

        first_object = cleaned.find("{")
        last_object = cleaned.rfind("}")
        if first_object != -1 and last_object != -1 and last_object > first_object:
            return cleaned[first_object:last_object + 1]

        first_array = cleaned.find("[")
        last_array = cleaned.rfind("]")
        if first_array != -1 and last_array != -1 and last_array > first_array:
            return cleaned[first_array:last_array + 1]

        return cleaned

    def _build_fallback_signals(self, signals: List[Dict], note: str) -> List[Dict]:
        out: List[Dict] = []
        for signal in signals:
            merged = {**signal}
            ai_score = self._coerce_score(signal.get("ranking_score") or signal.get("confidence"), signal)
            merged["ai_score"] = ai_score
            merged["ai_analysis"] = f"Technische fallback: {signal.get('reason', 'Geen extra analyse beschikbaar.')}"
            merged["ai_risk"] = note
            if not merged.get("rank_label"):
                merged["rank_label"] = self._score_to_label(ai_score)
            out.append(merged)
        out.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        return out

    @staticmethod
    def _coerce_score(raw_score: object, signal: Dict) -> int:
        try:
            score = int(float(raw_score))
        except (TypeError, ValueError):
            ranking_score = signal.get("ranking_score")
            confidence = signal.get("confidence")
            if ranking_score is not None:
                try:
                    score = int(round(float(ranking_score)))
                except (TypeError, ValueError):
                    score = 0
            elif confidence is not None:
                try:
                    score = int(round(float(confidence) * 100))
                except (TypeError, ValueError):
                    score = 0
            else:
                score = 0
        return max(1, min(score, 99))

    @staticmethod
    def _score_to_label(score: int) -> str:
        if score >= 85:
            return "AI Topkans"
        elif score >= 70:
            return "AI Sterke setup"
        elif score >= 55:
            return "AI Goede kans"
        elif score >= 40:
            return "AI Neutraal"
        elif score > 0:
            return "AI Zwak"
        return ""

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _make_cache_key(self, signals: List[Dict], market_filter: Optional[str]) -> str:
        syms = ",".join(s.get("symbol", "") for s in signals[:15])
        return f"ai:{market_filter or 'all'}:{syms}"

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

    def invalidate_cache(self) -> None:
        self._cache.clear()
