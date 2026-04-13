"""Tests for agent pipeline data models."""

import pytest
from src.agents.models import (
    PipelineState,
    RegimeClassification,
    SignalDecision,
    NewsItem,
    SentimentScore,
    AgentDecision,
)


def test_regime_classification_defaults():
    regime = RegimeClassification(
        regime="trending_up",
        confidence=0.8,
        reasoning="ADX=35, strong uptrend",
    )
    assert regime.regime == "trending_up"
    assert regime.confidence == 0.8
    assert regime.reasoning == "ADX=35, strong uptrend"


def test_regime_classification_low_confidence_flag():
    regime = RegimeClassification(
        regime="ranging",
        confidence=0.2,
        reasoning="unclear signals",
    )
    assert regime.is_low_confidence(threshold=0.3)
    assert not regime.is_low_confidence(threshold=0.1)


def test_signal_decision_approve():
    decision = SignalDecision(
        signal_id="sig_001",
        action="approve",
        adjusted_stop_loss=None,
        adjusted_target=None,
        adjusted_quantity_pct=1.0,
        confidence=0.85,
        reasoning="Strong trend alignment",
    )
    assert decision.is_approved
    assert not decision.is_rejected


def test_signal_decision_reject():
    decision = SignalDecision(
        signal_id="sig_002",
        action="reject",
        confidence=0.9,
        reasoning="Counter-trend in strong downtrend",
    )
    assert decision.is_rejected
    assert not decision.is_approved


def test_signal_decision_modify():
    decision = SignalDecision(
        signal_id="sig_003",
        action="modify",
        adjusted_stop_loss=48000_00,
        adjusted_target=52000_00,
        adjusted_quantity_pct=0.75,
        confidence=0.7,
        reasoning="Widen SL due to high volatility",
    )
    assert decision.is_approved  # modify counts as approved
    assert decision.adjusted_quantity_pct == 0.75


def test_sentiment_score_classification():
    extreme_fear = SentimentScore(score=15.0, classification="extreme_fear")
    assert extreme_fear.classification == "extreme_fear"

    greed = SentimentScore(score=72.0, classification="greed")
    assert greed.classification == "greed"


def test_pipeline_state_creation():
    state = PipelineState(
        instrument_key="NSE_EQ:RELIANCE",
        signals=[],
    )
    assert state.instrument_key == "NSE_EQ:RELIANCE"
    assert state.regime is None
    assert state.sentiment is None
    assert state.validated_signals == []
    assert not state.short_circuited


def test_pipeline_state_short_circuit():
    state = PipelineState(
        instrument_key="NSE_EQ:RELIANCE",
        signals=[],
        short_circuited=True,
        short_circuit_reason="Low regime confidence",
    )
    assert state.short_circuited
    assert state.short_circuit_reason == "Low regime confidence"


def test_agent_decision_record():
    decision = AgentDecision(
        agent_name="regime_agent",
        instrument_key="NSE_EQ:TCS",
        input_summary="ADX=25, ATR_pct=60",
        output_summary="trending_up, confidence=0.75",
        confidence=0.75,
        used_fallback=False,
        latency_ms=450,
    )
    assert decision.agent_name == "regime_agent"
    assert not decision.used_fallback
    assert decision.latency_ms == 450
