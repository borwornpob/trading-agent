"""GLM-based news sentiment scoring via OpenAI-compatible API.

Concepts: Ch 14 (NLP, sentiment analysis), Ch 15 (topic modeling), Ch 16 (embeddings).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from agent.config import GenAISettings
from agent.data.news_fetcher import NewsItem


@dataclass
class SentimentResult:
    headline: str
    sentiment: float          # -1.0 to +1.0
    confidence: float         # 0.0 to 1.0
    key_drivers: list[str]
    impact_horizon: str       # short / medium / long
    event_risk: bool          # NFP, FOMC, CPI, etc.
    gold_relevant: bool       # Is this actually about gold/metals?
    content_hash: str


@dataclass
class SentimentCache:
    """TTL cache keyed by content hash to avoid re-scoring identical headlines."""
    store: dict[str, tuple[float, SentimentResult]] = field(default_factory=dict)
    ttl_seconds: float = 14400.0  # 4 hours default

    def get(self, content_hash: str) -> SentimentResult | None:
        self._evict()
        entry = self.store.get(content_hash)
        if entry is None:
            return None
        ts, result = entry
        if time.monotonic() - ts > self.ttl_seconds:
            del self.store[content_hash]
            return None
        return result

    def put(self, content_hash: str, result: SentimentResult) -> None:
        self.store[content_hash] = (time.monotonic(), result)

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self.store.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self.store[k]


SYSTEM_PROMPT = """You are a gold market analyst. Analyze the given news headline and summary.
Return a JSON object with these fields:
- "sentiment": float from -1.0 (very bearish for gold) to +1.0 (very bullish)
- "confidence": float from 0.0 to 1.0
- "key_drivers": list of 1-3 short driver strings (e.g. "dollar_strength", "geopolitical_risk")
- "impact_horizon": one of "short" (< 24h), "medium" (1-5 days), "long" (> 5 days)
- "event_risk": boolean, true if this relates to NFP, FOMC, CPI, rate decisions, or similar high-impact scheduled events
- "gold_relevant": boolean, true if this news is directly relevant to gold/XAUUSD trading

Respond ONLY with valid JSON, no other text."""


BATCH_SYSTEM_PROMPT = """You are a gold market analyst. Analyze each given news headline and summary.
Return a JSON array with exactly one object per input item, in the same order.
Each object must include these fields:
- "id": integer copied from the input item
- "sentiment": float from -1.0 (very bearish for gold) to +1.0 (very bullish)
- "confidence": float from 0.0 to 1.0
- "key_drivers": list of 1-3 short driver strings (e.g. "dollar_strength", "geopolitical_risk")
- "impact_horizon": one of "short", "medium", "long"
- "event_risk": boolean, true if this relates to NFP, FOMC, CPI, rate decisions, or similar high-impact scheduled events
- "gold_relevant": boolean, true if this news is directly relevant to gold/XAUUSD trading

Respond ONLY with valid JSON, no markdown or commentary."""


def _build_user_prompt(item: NewsItem) -> str:
    return f"Headline: {item.headline}\nSummary: {item.summary[:300]}"


def _build_batch_user_prompt(items: list[NewsItem]) -> str:
    payload = [
        {
            "id": i,
            "headline": item.headline,
            "summary": item.summary[:300],
            "source": item.source,
            "published_utc": item.published_utc,
        }
        for i, item in enumerate(items)
    ]
    return "Analyze these gold-market news items:\n" + json.dumps(payload, ensure_ascii=False)


def score_single(
    item: NewsItem,
    settings: GenAISettings,
    cache: SentimentCache | None = None,
) -> SentimentResult:
    """Score a single news item via GLM API."""
    cache = cache or SentimentCache()
    cached = cache.get(item.content_hash)
    if cached is not None:
        return cached

    if not settings.can_call():
        return _default_result(item)

    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(item)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        parsed = _parse_json_response(text)
    except Exception:
        parsed = {}

    result = _result_from_payload(item, parsed)
    cache.put(item.content_hash, result)
    return result


def score_batch(
    items: list[NewsItem],
    settings: GenAISettings,
    cache: SentimentCache | None = None,
) -> list[SentimentResult]:
    """Score multiple news items with one OpenAI-compatible API request."""
    if not items:
        return []

    cache = cache or SentimentCache()
    results: list[SentimentResult | None] = [None] * len(items)
    pending: list[tuple[int, NewsItem]] = []

    for idx, item in enumerate(items):
        cached = cache.get(item.content_hash)
        if cached is not None:
            results[idx] = cached
        else:
            pending.append((idx, item))

    if pending and settings.can_call():
        from openai import OpenAI

        pending_items = [item for _, item in pending]
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_batch_user_prompt(pending_items)},
                ],
                temperature=0.1,
                max_tokens=max(500, min(1500, 250 * len(pending_items))),
            )
            text = response.choices[0].message.content.strip()
            parsed = _parse_json_value(text)
        except Exception:
            parsed = []

        payloads = parsed.get("results", []) if isinstance(parsed, dict) else parsed
        payloads = payloads if isinstance(payloads, list) else []
        payload_by_id = {
            int(payload.get("id", idx)): payload
            for idx, payload in enumerate(payloads)
            if isinstance(payload, dict)
        }

        for pending_idx, (result_idx, item) in enumerate(pending):
            result = _result_from_payload(item, payload_by_id.get(pending_idx, {}))
            results[result_idx] = result
            cache.put(item.content_hash, result)

    for idx, item in enumerate(items):
        if results[idx] is None:
            results[idx] = _default_result(item)

    return [r for r in results if r is not None]


def _parse_json_response(text: str) -> dict[str, Any]:
    """Try to extract JSON from the response text."""
    parsed = _parse_json_value(text)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_value(text: str) -> Any:
    """Try to extract a JSON object or array from the response text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_char, close_char in (("[", "]"), ("{", "}")):
            start = text.find(open_char)
            end = text.rfind(close_char)
            if start < 0 or end <= start:
                continue
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _result_from_payload(item: NewsItem, payload: dict[str, Any]) -> SentimentResult:
    key_drivers = payload.get("key_drivers", [])
    if not isinstance(key_drivers, list):
        key_drivers = []
    return SentimentResult(
        headline=item.headline,
        sentiment=float(payload.get("sentiment", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        key_drivers=[str(driver) for driver in key_drivers[:3]],
        impact_horizon=str(payload.get("impact_horizon", "medium")),
        event_risk=bool(payload.get("event_risk", False)),
        gold_relevant=bool(payload.get("gold_relevant", True)),
        content_hash=item.content_hash,
    )


def _default_result(item: NewsItem) -> SentimentResult:
    return SentimentResult(
        headline=item.headline,
        sentiment=0.0,
        confidence=0.0,
        key_drivers=[],
        impact_horizon="medium",
        event_risk=False,
        gold_relevant=True,
        content_hash=item.content_hash,
    )


def sentiment_to_features(results: list[SentimentResult]) -> pl.DataFrame:
    """Convert scored results into a feature DataFrame for model input.

    Aggregates all headlines into a single row with:
    - weighted_sentiment: confidence-weighted average sentiment
    - max_confidence: highest confidence among recent items
    - event_risk_count: number of items flagging event risk
    - gold_relevant_count: number of gold-relevant items
    """
    if not results:
        return pl.DataFrame({
            "weighted_sentiment": [0.0],
            "sentiment_count": [0],
            "max_confidence": [0.0],
            "event_risk_flag": [False],
            "event_risk_count": [0],
            "gold_relevant_count": [0],
        })

    sentiments = [r.sentiment for r in results]
    confidences = [r.confidence for r in results]
    total_conf = sum(confidences) or 1.0
    weighted = sum(s * c for s, c in zip(sentiments, confidences)) / total_conf
    event_count = sum(1 for r in results if r.event_risk)

    return pl.DataFrame({
        "weighted_sentiment": [weighted],
        "sentiment_count": [len(results)],
        "max_confidence": [max(confidences)],
        "event_risk_flag": [event_count > 0],
        "event_risk_count": [event_count],
        "gold_relevant_count": [sum(1 for r in results if r.gold_relevant)],
    })
