"""Tests for the base agent class."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient, LLMResponse


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
