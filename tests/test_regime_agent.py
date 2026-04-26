"""Tests for the LLM-powered market regime agent."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.regime_agent import RegimeAgent
from src.agents.llm_client import LLMClient, LLMResponse
from src.agents.models import RegimeClassification


class TestRegimeAgent:
    def _make_context(self, adx=35.0, rsi=55.0, atr_pct=60.0, bb_width=3.5,
                      price_position=70.0, ema_9_above_21=True):
        return {
            "adx": adx, "adx_trend": "rising", "rsi": rsi,
            "atr_percentile": atr_pct, "bb_width": bb_width,
            "bb_width_percentile": 55.0,
            "price_position_in_range": price_position,
            "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
            "is_squeeze": False,
        }

    def test_llm_response_parsed_correctly(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "regime": "trending_up", "confidence": 0.85,
                "reasoning": "ADX=35 rising, price near 20d high, EMA9 > EMA21",
            }),
            success=True, latency_ms=300,
        )
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(), instrument_key="NSE_EQ:RELIANCE")
        assert result["regime"] == "trending_up"
        assert result["confidence"] == 0.85

    def test_llm_response_with_reasoning_trace(self):
        """Verify parsing works when LLM prefixes JSON with chain-of-thought."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=(
                "<think>\nOkay, let me analyze the indicators.\n"
                "ADX is 35 which suggests strong trend.\n</think>\n"
                '{"regime": "trending_up", "confidence": 0.85, '
                '"reasoning": "Strong ADX with bullish EMAs"}'
            ),
            success=True, latency_ms=500,
        )
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(), instrument_key="NSE_EQ:RELIANCE")
        assert result["regime"] == "trending_up"
        assert result["confidence"] == 0.85

    def test_fallback_trending_up(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(adx=30.0, price_position=75.0))
        assert result["regime"] == "trending_up"

    def test_fallback_trending_down(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(adx=30.0, price_position=20.0))
        assert result["regime"] == "trending_down"

    def test_fallback_ranging(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(adx=15.0, price_position=50.0))
        assert result["regime"] == "ranging"

    def test_fallback_volatile(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(atr_pct=95.0))
        assert result["regime"] == "volatile"

    def test_to_classification(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(adx=30.0, price_position=75.0))
        classification = agent.to_classification(result)
        assert isinstance(classification, RegimeClassification)
        assert classification.regime == "trending_up"
