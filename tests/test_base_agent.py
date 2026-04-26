"""Tests for the base agent class."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.base_agent import BaseAgent, extract_json
from src.agents.llm_client import LLMClient, LLMResponse


class TestExtractJson:
    """Tests for the robust JSON extraction utility."""

    def test_pure_json(self):
        raw = '{"regime": "trending_up", "confidence": 0.85, "reasoning": "Strong"}'
        result = extract_json(raw)
        assert result["regime"] == "trending_up"
        assert result["confidence"] == 0.85

    def test_json_with_markdown_fence(self):
        raw = '```json\n{"regime": "ranging", "confidence": 0.5, "reasoning": "Flat"}\n```'
        result = extract_json(raw)
        assert result["regime"] == "ranging"

    def test_json_with_think_tags(self):
        raw = (
            "<think>\nOkay, let me analyze the indicators. ADX is 35 which is strong.\n"
            "RSI at 55 is neutral. ATR percentile at 60 is moderate.\n</think>\n"
            '{"regime": "trending_up", "confidence": 0.8, "reasoning": "ADX strong"}'
        )
        result = extract_json(raw)
        assert result["regime"] == "trending_up"
        assert result["confidence"] == 0.8

    def test_json_with_reasoning_prefix(self):
        raw = (
            "Let me think about this step by step.\n"
            "The ADX is high, price is near highs, EMAs are bullish.\n\n"
            '{"regime": "trending_up", "confidence": 0.9, "reasoning": "Bullish setup"}'
        )
        result = extract_json(raw)
        assert result["regime"] == "trending_up"

    def test_json_with_reasoning_suffix(self):
        raw = (
            '{"regime": "volatile", "confidence": 0.7, "reasoning": "High ATR"}\n\n'
            "This classification is based on the extreme ATR readings."
        )
        result = extract_json(raw)
        assert result["regime"] == "volatile"

    def test_json_with_think_tags_and_fence(self):
        raw = (
            "<think>Analyzing the data...</think>\n"
            "```json\n"
            '{"action": "approve", "confidence": 0.8, "reasoning": "Good RR"}\n'
            "```"
        )
        result = extract_json(raw)
        assert result["action"] == "approve"

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_json("This response has no JSON at all")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_nested_json_in_reasoning(self):
        raw = (
            '<think>The signal has {"entry": 100} which looks good</think>\n'
            '{"action": "approve", "confidence": 0.75, "reasoning": "Solid"}'
        )
        result = extract_json(raw)
        assert result["action"] == "approve"
        assert result["confidence"] == 0.75

    def test_whitespace_padded(self):
        raw = '\n\n  {"sentiment_score": 0.5, "confidence": 0.8, "reasoning": "Good"}  \n\n'
        result = extract_json(raw)
        assert result["sentiment_score"] == 0.5


class ConcreteAgent(BaseAgent):
    """Test implementation of BaseAgent."""

    @property
    def agent_name(self) -> str:
        return "test_agent"

    def _build_prompt(self, context: dict) -> str:
        return f"Analyze: {json.dumps(context)}"

    def _build_system_prompt(self, lessons: list[str]) -> str:
        base = "You are a test agent. Respond with JSON."
        if lessons:
            base += "\nLessons: " + "; ".join(lessons)
        return base

    def _parse_response(self, raw: str) -> dict:
        return json.loads(raw)

    def _fallback(self, context: dict) -> dict:
        return {"result": "fallback", "confidence": 0.3}


class TestBaseAgent:
    def test_agent_name(self):
        client = MagicMock(spec=LLMClient)
        agent = ConcreteAgent(llm_client=client)
        assert agent.agent_name == "test_agent"

    def test_run_uses_llm_when_available(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "llm_output", "confidence": 0.9}',
            success=True,
            latency_ms=100,
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "llm_output"
        client.invoke.assert_called_once()

    def test_run_falls_back_on_llm_failure(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content="", success=False, error="timeout"
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"
        assert result["confidence"] == 0.3

    def test_run_falls_back_on_parse_error(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content="not valid json at all",
            success=True,
            latency_ms=50,
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"

    def test_run_falls_back_when_not_configured(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"
        client.invoke.assert_not_called()

    def test_lessons_passed_to_system_prompt(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "ok", "confidence": 0.8}',
            success=True,
            latency_ms=100,
        )
        agent = ConcreteAgent(llm_client=client)
        agent.run(
            context={"data": "test"},
            lessons=["Avoid counter-trend longs in strong downtrend"],
        )
        call_kwargs = client.invoke.call_args
        assert "Avoid counter-trend" in call_kwargs.kwargs.get("system_prompt", "") or \
               "Avoid counter-trend" in str(call_kwargs)

    def test_last_decision_tracked(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "ok", "confidence": 0.8}',
            success=True,
            latency_ms=200,
        )
        agent = ConcreteAgent(llm_client=client)
        agent.run(context={"data": "test"})
        assert agent.last_decision is not None
        assert agent.last_decision.agent_name == "test_agent"
        assert not agent.last_decision.used_fallback

    def test_last_decision_tracks_fallback(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = ConcreteAgent(llm_client=client)
        agent.run(context={"data": "test"})
        assert agent.last_decision is not None
        assert agent.last_decision.used_fallback
