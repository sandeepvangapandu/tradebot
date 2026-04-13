"""Tests for the news sentiment agent."""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.agents.news_agent import NewsAgent
from src.agents.llm_client import LLMClient, LLMResponse


class TestNewsAgent:
    def test_parse_llm_sentiment(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "sentiment_score": 0.3,
                "confidence": 0.75,
                "reasoning": "Mixed signals: positive earnings but trade tensions",
            }),
            success=True,
            latency_ms=400,
        )
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={
            "headlines": [
                "Nifty hits all-time high on strong FII inflows",
                "US-China trade war fears weigh on emerging markets",
            ]
        })
        assert -1.0 <= result["sentiment_score"] <= 1.0
        assert result["confidence"] > 0

    def test_fallback_returns_neutral(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={"headlines": ["Some news"]})
        assert result["sentiment_score"] == 0.0
        assert result["confidence"] > 0

    def test_empty_headlines_returns_neutral(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={"headlines": []})
        assert result["sentiment_score"] == 0.0

    @patch("src.agents.news_agent.feedparser")
    def test_fetch_headlines(self, mock_fp):
        mock_fp.parse.return_value = MagicMock(
            entries=[
                MagicMock(title="Sensex rallies 500 points", source={"title": "ET"}),
                MagicMock(title="RBI holds rates steady", source={"title": "LiveMint"}),
            ]
        )
        client = MagicMock(spec=LLMClient)
        agent = NewsAgent(llm_client=client)
        headlines = agent.fetch_headlines(query="indian stock market")
        assert len(headlines) == 4  # 2 feeds x 2 entries each
        assert "Sensex" in headlines[0].title
