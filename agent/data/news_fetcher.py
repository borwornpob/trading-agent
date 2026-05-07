"""News RSS collection for gold market.

Concepts: Ch 03 (alternative data, source scoring).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import feedparser
import polars as pl


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    url: str
    published_utc: str
    content_hash: str


@dataclass
class NewsCache:
    """In-memory deduplication cache keyed by content hash."""
    seen_hashes: dict[str, float] = field(default_factory=dict)
    ttl_seconds: float = 3600.0

    def is_duplicate(self, content_hash: str) -> bool:
        self._evict()
        return content_hash in self.seen_hashes

    def mark_seen(self, content_hash: str) -> None:
        self.seen_hashes[content_hash] = time.monotonic()

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, t in self.seen_hashes.items() if now - t > self.ttl_seconds]
        for k in expired:
            del self.seen_hashes[k]


def _content_hash(headline: str, url: str) -> str:
    return hashlib.sha256(f"{headline}|{url}".encode()).hexdigest()[:16]


def fetch_rss_headlines(
    feed_urls: list[str],
    *,
    cache: NewsCache | None = None,
    max_items_per_feed: int = 20,
) -> list[NewsItem]:
    """Fetch headlines from RSS feeds, deduplicate by content hash."""
    cache = cache or NewsCache()
    items: list[NewsItem] = []

    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for entry in feed.entries[:max_items_per_feed]:
            headline = getattr(entry, "title", "").strip()
            if not headline:
                continue
            summary = getattr(entry, "summary", "").strip()
            link = getattr(entry, "link", "")
            published = getattr(entry, "published", "")
            source_name = getattr(feed, "feed", {})
            source_name = getattr(source_name, "title", url)

            ch = _content_hash(headline, link)
            if cache.is_duplicate(ch):
                continue
            cache.mark_seen(ch)

            items.append(NewsItem(
                headline=headline,
                summary=summary[:500],
                source=str(source_name),
                url=link,
                published_utc=published,
                content_hash=ch,
            ))

    return items


def news_items_to_dataframe(items: list[NewsItem]) -> pl.DataFrame:
    """Convert news items to a Polars DataFrame for feature merging."""
    if not items:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime("ms"),
            "headline": pl.Utf8,
            "summary": pl.Utf8,
            "source": pl.Utf8,
            "url": pl.Utf8,
            "content_hash": pl.Utf8,
        })

    return pl.DataFrame({
        "timestamp": [pl.lit(None).cast(pl.Datetime("ms")) for _ in items],
        "headline": [i.headline for i in items],
        "summary": [i.summary for i in items],
        "source": [i.source for i in items],
        "url": [i.url for i in items],
        "content_hash": [i.content_hash for i in items],
    })
