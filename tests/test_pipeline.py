"""Tests for the LangGraph agent pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.agents.pipeline import AgentPipeline
from src.agents.llm_client import LLMClient, LLMResponse
from src.agents.models import PipelineState
from src.memory.memory_db import MemoryDB


class TestAgentPipeline:
    def _make_signals(self):
        return [{
            "signal_id": "sig_001",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "entry_price": 50000_00,
            "stop_loss": 49000_00,
            "target": 53000_00,
            "strategy": "ema_crossover",
            "confidence": 0.8,
        }]

    def test_pipeline_with_all_fallbacks(self):
        """When LLM is not configured, all agents use fallbacks."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 30.0, "adx_trend": "rising", "rsi": 55.0,
                "atr_percentile": 50.0, "bb_width": 3.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 70.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        assert isinstance(result, PipelineState)
        assert result.regime is not None
        assert result.regime.regime in {"trending_up", "trending_down", "ranging", "volatile"}

    def test_pipeline_short_circuits_on_low_confidence(self):
        """Pipeline should short-circuit when regime confidence is low."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.8,  # Set high threshold
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 15.0, "adx_trend": "flat", "rsi": 50.0,
                "atr_percentile": 50.0, "bb_width": 2.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 50.0,
                "ema_9": 50000.0, "ema_21": 50000.0, "sma_50": 50000.0,
                "is_squeeze": False,
            },
        )
        assert result.short_circuited
        assert result.validated_signals == []

    def test_pipeline_passes_signals_through(self):
        """Signals aligned with regime should pass validation."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 35.0, "adx_trend": "rising", "rsi": 60.0,
                "atr_percentile": 55.0, "bb_width": 4.0,
                "bb_width_percentile": 60.0, "price_position_in_range": 75.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        assert not result.short_circuited
        # BUY + trending_up + RR > 1.5 should be approved
        assert len(result.validated_signals) >= 1

    def test_pipeline_rejects_counter_trend(self):
        """BUY signal in trending_down should be rejected."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 35.0, "adx_trend": "rising", "rsi": 30.0,
                "atr_percentile": 55.0, "bb_width": 4.0,
                "bb_width_percentile": 60.0, "price_position_in_range": 20.0,
                "ema_9": 49500.0, "ema_21": 49800.0, "sma_50": 50200.0,
                "is_squeeze": False,
            },
        )
        # BUY + trending_down -> rejected
        assert len(result.validated_signals) == 0

    def test_agent_decisions_recorded(self):
        """Pipeline should record all agent decisions."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 30.0, "adx_trend": "rising", "rsi": 55.0,
                "atr_percentile": 50.0, "bb_width": 3.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 70.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        # Should have at least regime_agent and signal_validator decisions
        agent_names = [d.agent_name for d in result.agent_decisions]
        assert "regime_agent" in agent_names
