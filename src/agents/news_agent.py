"""News sentiment agent using RSS feeds and LLM analysis.

Fetches financial headlines from Google News RSS and uses the LLM
to score overall market sentiment. Falls back to neutral when
LLM is unavailable.
"""

from __future__ import annotations

import json
import time
from typing import Any

import feedparser
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.models import NewsItem

SYSTEM_PROMPT = """You are a financial news sentiment analyst for the Indian stock market.

Given a batch of recent headlines, score the overall market sentiment.

Respond ONLY with valid JSON:
{"sentiment_score": <-1.0 to 1.0>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Scoring guide:
- -1.0: Extremely bearish (crash fears, major crisis)
- -0.5: Moderately bearish (rate hikes, poor earnings)
-  0.0: Neutral (mixed signals)
-  0.5: Moderately bullish (good earnings, FII inflows)
-  1.0: Extremely bullish (breakout, policy support)
"""

NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=indian+stock+market&hl=en-IN&gl=IN",
    "https://news.google.com/rss/search?q=nifty+sensex&hl=en-IN&gl=IN",
]


class NewsAgent(BaseAgent):
    """Analyzes financial news sentiment using LLM.

    Args:
        llm_client: Shared LLM client instance.
        cache_ttl_s: How long to cache fetched headlines in seconds.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache_ttl_s: int = 300,
    ) -> None:
        super().__init__(llm_client)
        self._cache_ttl_s = cache_ttl_s
        self._headline_cache: list[NewsItem] = []
        self._cache_time: float = 0.0

    @property
    def agent_name(self) -> str:
        return "news_agent"

    def fetch_headlines(
        self, query: str = "indian stock market", max_items: int = 15
    ) -> list[NewsItem]:
        """Fetch recent headlines from Google News RSS.

        Args:
            query: Search query for news.
            max_items: Maximum headlines to return.

        Returns:
            List of NewsItem objects.
        """
        now = time.monotonic()
        if self._headline_cache and (now - self._cache_time) < self._cache_ttl_s:
            return self._headline_cache

        items: list[NewsItem] = []
        for feed_url in NEWS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_items]:
                    source_name = ""
                    if hasattr(entry, "source") and hasattr(entry.source, "get"):
                        source_name = entry.source.get("title", "")
                    elif hasattr(entry, "source") and hasattr(entry.source, "title"):
                        source_name = entry.source.title
                    items.append(NewsItem(
                        title=entry.title,
                        source=source_name,
                    ))
            except Exception as e:
                logger.warning("Failed to fetch news from {}: {}", feed_url, e)

        self._headline_cache = items[:max_items]
        self._cache_time = now
        logger.debug("Fetched {} headlines", len(self._headline_cache))
        return self._headline_cache

    def _build_prompt(self, context: dict[str, Any]) -> str:
        headlines = context.get("headlines", [])
        if not headlines:
            return "No headlines available."

        text = "Analyze sentiment of these recent Indian market headlines:\n\n"
        for i, h in enumerate(headlines[:15], 1):
            if isinstance(h, str):
                text += f"{i}. {h}\n"
            else:
                text += f"{i}. {h}\n"
        return text

    def _build_system_prompt(self, lessons: list[str]) -> str:
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about sentiment analysis:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        score = float(data.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))

        return {
            "sentiment_score": score,
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return neutral sentiment when LLM unavailable."""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.3,
            "reasoning": "Fallback: no LLM available, assuming neutral",
        }

    def run(
        self,
        context: dict[str, Any],
        lessons: list[str] | None = None,
        instrument_key: str = "",
    ) -> dict[str, Any]:
        """Override run to handle empty headlines."""
        headlines = context.get("headlines", [])
        if not headlines:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.3,
                "reasoning": "No headlines to analyze",
            }
        return super().run(context, lessons, instrument_key)
