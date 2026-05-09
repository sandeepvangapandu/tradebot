"""Tests for the LLM-powered signal validation agent."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.signal_validator import SignalValidatorAgent
from src.agents.llm_client import LLMClient, LLMResponse


class TestSignalValidatorAgent:
    def _make_context(self, direction="BUY", regime="trending_up", confidence=0.8,
                      stop_loss=49000_00, target=52000_00, entry_price=50000_00):
        return {
            "signal": {
                "signal_id": "sig_001", "instrument_key": "NSE_EQ:RELIANCE",
                "direction": direction, "entry_price": entry_price,
                "stop_loss": stop_loss, "target": target,
                "strategy": "ema_crossover", "confidence": confidence,
            },
            "regime": regime,
            "sentiment_score": 60.0,
            "sentiment_classification": "greed",
        }

    def test_llm_approves_aligned_signal(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "action": "approve", "confidence": 0.9,
                "reasoning": "BUY signal aligned with trending_up regime",
            }),
            success=True, latency_ms=250,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context())
        assert result["action"] == "approve"

    def test_llm_rejects_counter_trend(self):
        # LLM rejects by returning confidence_multiplier=0.0; action field is ignored
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "confidence_multiplier": 0.0,
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "reasoning": "BUY signal in trending_down regime",
            }),
            success=True, latency_ms=200,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(regime="trending_down"))
        assert result["action"] == "reject"

    def test_llm_modifies_stop_loss(self):
        # New API: LLM returns confidence_multiplier and adjusted fields; action is derived
        # from multiplier > 0 → "approve" (with modifications applied via adjusted_stop_loss)
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "confidence_multiplier": 0.75,
                "adjusted_stop_loss": 48500_00,
                "adjusted_target": None,
                "reasoning": "Widening SL due to high volatility",
            }),
            success=True, latency_ms=350,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context())
        # multiplier > 0 → action is "approve"; stop-loss modification is in adjusted_stop_loss
        assert result["action"] == "approve"
        assert result["adjusted_stop_loss"] == 48500_00
        assert result["confidence_multiplier"] == pytest.approx(0.75)

    def test_fallback_approves_trend_aligned_buy(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="BUY", regime="trending_up", confidence=0.7
        ))
        assert result["action"] == "approve"

    def test_fallback_rejects_counter_trend(self):
        # Fallback rejects counter-trend signals outright (confidence_multiplier=0.0).
        # The fallback reads regime from self._latest_context (set via update_context),
        # not from the run() context dict.
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        agent.update_context(regime="trending_down", vix=15.0, sentiment_score=0.0)
        result = agent.run(context=self._make_context(
            direction="BUY", regime="trending_down"
        ))
        assert result["action"] == "reject"
        assert result["confidence_multiplier"] == pytest.approx(0.0)
        assert "counter-trend" in result["reasoning"].lower()

    def test_fallback_rejects_low_confidence(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(confidence=0.3))
        assert result["action"] == "reject"

    def test_fallback_approves_ranging_mean_reversion(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="SELL", regime="ranging", confidence=0.7
        ))
        assert result["action"] == "approve"

    def test_fallback_checks_risk_reward(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="BUY", regime="trending_up", confidence=0.8,
            entry_price=50000_00, stop_loss=49500_00, target=50600_00,
        ))
        assert result["action"] == "reject"
        assert "risk-reward" in result["reasoning"].lower()
