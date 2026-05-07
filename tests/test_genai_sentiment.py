"""Tests for GenAI sentiment scoring."""

import pytest
from unittest.mock import MagicMock
from agent.data.news_fetcher import NewsItem, NewsCache
from agent.data.genai_sentiment import SentimentResult, SentimentCache, _parse_json_response, sentiment_to_features


def test_news_cache_dedup():
    cache = NewsCache(ttl_seconds=60)
    assert not cache.is_duplicate("abc")
    cache.mark_seen("abc")
    assert cache.is_duplicate("abc")
    assert not cache.is_duplicate("xyz")


def test_parse_json_response_valid():
    text = '{"sentiment": 0.5, "confidence": 0.8}'
    result = _parse_json_response(text)
    assert result["sentiment"] == 0.5
    assert result["confidence"] == 0.8


def test_parse_json_response_with_markdown():
    text = '```json\n{"sentiment": -0.3}\n```'
    result = _parse_json_response(text)
    assert result["sentiment"] == -0.3


def test_parse_json_response_invalid():
    result = _parse_json_response("not json at all")
    assert result == {}


def test_sentiment_to_features_empty():
    df = sentiment_to_features([])
    assert len(df) == 1
    assert df["weighted_sentiment"].item() == 0.0


def test_sentiment_to_features():
    results = [
        SentimentResult("h1", 0.5, 0.8, ["dollar"], "short", False, True, "abc"),
        SentimentResult("h2", -0.3, 0.6, ["geopolitical"], "medium", False, True, "def"),
    ]
    df = sentiment_to_features(results)
    assert len(df) == 1
    assert df["sentiment_count"].item() == 2
    assert df["event_risk_count"].item() == 0
