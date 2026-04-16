"""Sentiment Analyzer: lokale FinBERT-gebaseerde sentimentanalyse.

Verwerkt financiële teksten (nieuwskoppen, Reddit posts, SEC filings) en
classificeert ze als positive / negative / neutral met een confidence score.

Gebruik ProsusAI/finbert (HuggingFace) — draait volledig lokaal, geen API
keys nodig. Model wordt lazy geladen bij eerste aanroep om opstarttijd
te minimaliseren.

Gebruik:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Apple beats earnings expectations")
    # SentimentResult(label='positive', score=0.92, scores={...})
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
_MODEL_NAME: str = "ProsusAI/finbert"
_MAX_LENGTH: int = 512          # FinBERT max token length
_BATCH_SIZE: int = 16           # inference batch grootte
_LABELS: List[str] = ["positive", "negative", "neutral"]


@dataclass
class SentimentResult:
    """Resultaat van een sentimentanalyse op één tekst."""

    label: str                           # "positive", "negative", "neutral"
    score: float                         # confidence van het primaire label ∈ [0, 1]
    scores: Dict[str, float]             # alle drie kansen
    text_snippet: str = ""               # eerste 100 chars van de input

    @property
    def numeric_score(self) -> float:
        """Converteer naar een enkel getal ∈ [-1, 1].

        +1 = maximaal positief, -1 = maximaal negatief, 0 = neutraal.
        """
        return self.scores.get("positive", 0.0) - self.scores.get("negative", 0.0)


@dataclass
class AggregateSentiment:
    """Geaggregeerd sentiment over meerdere teksten voor één asset."""

    symbol: str
    mean_score: float                    # gemiddelde numeric_score ∈ [-1, 1]
    positive_ratio: float                # fractie positieve teksten
    negative_ratio: float                # fractie negatieve teksten
    neutral_ratio: float                 # fractie neutrale teksten
    num_texts: int                       # totaal geanalyseerde teksten
    confidence: float                    # gemiddelde confidence
    normalized_score: float = 0.5        # genormaliseerd naar [0, 1] voor features

    def __post_init__(self) -> None:
        # Normaliseer [-1, 1] → [0, 1]
        self.normalized_score = round((self.mean_score + 1.0) / 2.0, 4)


class SentimentAnalyzer:
    """Lokale financiële sentimentanalyse via ProsusAI/finbert.

    Het model wordt lazy geladen — pas bij de eerste ``analyze()`` of
    ``analyze_batch()`` aanroep. Dit voorkomt lange opstarttijden als
    sentiment niet direct nodig is.

    Vereisten:
        pip install transformers torch
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_loaded: bool = False
        self._load_attempted: bool = False

    # ── Public API ──────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        """Geeft aan of het FinBERT model succesvol geladen is."""
        return self._model_loaded

    def analyze(self, text: str) -> SentimentResult:
        """Analyseer sentiment van een enkele tekst.

        Args:
            text: Financiële tekst (nieuwskop, Reddit titel, etc.).

        Returns:
            SentimentResult met label, score en per-class kansen.
        """
        self._ensure_model()

        if not text or not text.strip():
            return self._neutral_result("")

        if not self._model_loaded:
            return self._neutral_result(text)

        try:
            # Truncate naar max lengte
            truncated = text[:_MAX_LENGTH * 4]  # ruwe char-limiet pre-tokenisatie
            outputs = self._pipeline(truncated, truncation=True, max_length=_MAX_LENGTH)

            return self._parse_pipeline_output(outputs, text)

        except Exception as exc:
            logger.warning("Sentiment analyse mislukt: %s", exc)
            return self._neutral_result(text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyseer sentiment van meerdere teksten in batch.

        Args:
            texts: Lijst van financiële teksten.

        Returns:
            Lijst van SentimentResult, één per input tekst.
        """
        self._ensure_model()

        if not texts:
            return []

        if not self._model_loaded:
            return [self._neutral_result(t) for t in texts]

        results: List[SentimentResult] = []

        # Verwerk in batches
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            cleaned = [t[:_MAX_LENGTH * 4] if t else "" for t in batch]

            try:
                outputs = self._pipeline(
                    cleaned,
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    batch_size=_BATCH_SIZE,
                )
                for j, output in enumerate(outputs):
                    original_text = batch[j] if j < len(batch) else ""
                    results.append(self._parse_pipeline_output(output, original_text))
            except Exception as exc:
                logger.warning("Batch sentiment analyse mislukt bij batch %d: %s", i, exc)
                results.extend([self._neutral_result(t) for t in batch])

        return results

    def aggregate(
        self,
        symbol: str,
        results: List[SentimentResult],
    ) -> AggregateSentiment:
        """Aggregeer meerdere SentimentResults tot één score per asset.

        Args:
            symbol: Asset-symbool.
            results: Lijst van individuele SentimentResults.

        Returns:
            AggregateSentiment met gemiddelden en ratio's.
        """
        if not results:
            return AggregateSentiment(
                symbol=symbol,
                mean_score=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                num_texts=0,
                confidence=0.0,
            )

        numeric_scores = [r.numeric_score for r in results]
        confidences = [r.score for r in results]
        labels = [r.label for r in results]

        n = len(results)
        return AggregateSentiment(
            symbol=symbol,
            mean_score=round(float(np.mean(numeric_scores)), 4),
            positive_ratio=round(labels.count("positive") / n, 4),
            negative_ratio=round(labels.count("negative") / n, 4),
            neutral_ratio=round(labels.count("neutral") / n, 4),
            num_texts=n,
            confidence=round(float(np.mean(confidences)), 4),
        )

    # ── Interne methoden ────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-load het FinBERT model bij eerste gebruik."""
        if self._model_loaded or self._load_attempted:
            return

        self._load_attempted = True
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("FinBERT model laden (%s)...", _MODEL_NAME)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=_MODEL_NAME,
                tokenizer=_MODEL_NAME,
                return_all_scores=True,
                device=-1,  # CPU — geen GPU vereist
            )
            self._model_loaded = True
            logger.info("FinBERT model succesvol geladen.")

        except ImportError:
            logger.warning(
                "transformers en/of torch niet geïnstalleerd. "
                "Sentiment analyse niet beschikbaar. "
                "Installeer via: pip install transformers torch"
            )
        except Exception as exc:
            logger.warning("FinBERT model laden mislukt: %s. Fallback naar neutraal.", exc)

    def _parse_pipeline_output(
        self,
        output: object,
        original_text: str,
    ) -> SentimentResult:
        """Parse HuggingFace pipeline output naar SentimentResult.

        De pipeline met return_all_scores=True retourneert:
        [[{'label': 'positive', 'score': 0.9}, ...]]
        """
        snippet = original_text[:100] if original_text else ""

        try:
            # output is een lijst van dicts met label + score
            if isinstance(output, list) and output and isinstance(output[0], list):
                score_list = output[0]
            elif isinstance(output, list) and output and isinstance(output[0], dict):
                score_list = output
            else:
                return self._neutral_result(original_text)

            scores: Dict[str, float] = {}
            for item in score_list:
                label = item["label"].lower()
                score = float(item["score"])
                scores[label] = round(score, 4)

            # Bepaal primair label
            best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
            best_score = scores[best_label]

            return SentimentResult(
                label=best_label,
                score=round(best_score, 4),
                scores=scores,
                text_snippet=snippet,
            )

        except Exception as exc:
            logger.debug("Pipeline output parsing mislukt: %s", exc)
            return self._neutral_result(original_text)

    @staticmethod
    def _neutral_result(text: str) -> SentimentResult:
        """Retourneer een neutraal standaard-resultaat."""
        return SentimentResult(
            label="neutral",
            score=0.34,
            scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            text_snippet=text[:100] if text else "",
        )
