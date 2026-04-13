"""LangGraph-based agent pipeline for intelligent trade filtering.

Orchestrates the flow: Regime Detection -> Signal Validation.
Each node reads from and writes to a shared PipelineState.
Short-circuits when regime confidence is below threshold.

Note: This implementation uses a simple sequential pipeline.
LangGraph StateGraph can be added later for more complex
conditional routing without changing the interface.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agents.llm_client import LLMClient
from src.agents.models import AgentDecision, PipelineState, RegimeClassification
from src.agents.regime_agent import RegimeAgent
from src.agents.signal_validator import SignalValidatorAgent
from src.memory.injector import MemoryInjector
from src.memory.memory_db import MemoryDB


class AgentPipeline:
    """Orchestrates AI agents in a sequential pipeline.

    Flow:
    1. Regime Agent classifies market state
    2. If confidence < threshold, short-circuit (skip trading)
    3. Signal Validator approves/rejects/modifies each signal
    4. Return validated signals for execution

    Args:
        llm_client: Shared LLM client.
        memory_db: Memory lesson database.
        regime_confidence_threshold: Minimum regime confidence to proceed.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_db: MemoryDB,
        regime_confidence_threshold: float = 0.3,
    ) -> None:
        self._regime_agent = RegimeAgent(llm_client=llm_client)
        self._signal_validator = SignalValidatorAgent(llm_client=llm_client)
        self._injector = MemoryInjector(memory_db=memory_db, max_lessons=5)
        self._confidence_threshold = regime_confidence_threshold

    def run(
        self,
        instrument_key: str,
        signals: list[dict[str, Any]],
        indicator_context: dict[str, Any],
        sentiment_score: float = 50.0,
        sentiment_classification: str = "neutral",
    ) -> PipelineState:
        """Execute the full agent pipeline.

        Args:
            instrument_key: Instrument being evaluated.
            signals: Raw signals from strategy engine.
            indicator_context: Technical indicator values for regime agent.
            sentiment_score: Market mood score (0-100).
            sentiment_classification: Mood classification string.

        Returns:
            PipelineState with regime, validated signals, and decisions.
        """
        state = PipelineState(
            instrument_key=instrument_key,
            signals=signals,
        )

        if not signals:
            logger.debug("Pipeline: no signals to process for {}", instrument_key)
            return state

        # --- Step 1: Regime Detection ---
        regime_lessons = self._injector.get_lessons_for_agent(
            agent_name="regime_agent",
            regime="",  # No regime known yet
        )
        regime_result = self._regime_agent.run(
            context=indicator_context,
            lessons=regime_lessons,
            instrument_key=instrument_key,
        )
        state.regime = self._regime_agent.to_classification(regime_result)

        if self._regime_agent.last_decision:
            state.agent_decisions.append(self._regime_agent.last_decision)

        logger.info(
            "Pipeline regime: {} (confidence={:.2f}) for {}",
            state.regime.regime,
            state.regime.confidence,
            instrument_key,
        )

        # --- Short-circuit on low confidence ---
        if state.regime.is_low_confidence(self._confidence_threshold):
            state.short_circuited = True
            state.short_circuit_reason = (
                f"Regime confidence {state.regime.confidence:.2f} "
                f"< threshold {self._confidence_threshold}"
            )
            logger.warning(
                "Pipeline short-circuited: {} | {}",
                instrument_key,
                state.short_circuit_reason,
            )
            return state

        # --- Step 2: Signal Validation ---
        validated = []
        for signal in signals:
            validator_lessons = self._injector.get_lessons_for_agent(
                agent_name="signal_validator",
                regime=state.regime.regime,
                strategy=signal.get("strategy", ""),
            )
            validation_context = {
                "signal": signal,
                "regime": state.regime.regime,
                "sentiment_score": sentiment_score,
                "sentiment_classification": sentiment_classification,
            }
            val_result = self._signal_validator.run(
                context=validation_context,
                lessons=validator_lessons,
                instrument_key=instrument_key,
            )

            if self._signal_validator.last_decision:
                state.agent_decisions.append(self._signal_validator.last_decision)

            decision = self._signal_validator.to_decision(
                val_result, signal_id=signal.get("signal_id", "")
            )

            if decision.is_approved:
                # Apply modifications if any
                modified_signal = dict(signal)
                if decision.adjusted_stop_loss is not None:
                    modified_signal["stop_loss"] = decision.adjusted_stop_loss
                if decision.adjusted_target is not None:
                    modified_signal["target"] = decision.adjusted_target
                if decision.adjusted_quantity_pct != 1.0:
                    orig_qty = modified_signal.get("quantity", 1)
                    modified_signal["quantity"] = max(
                        1, int(orig_qty * decision.adjusted_quantity_pct)
                    )
                modified_signal["validation_confidence"] = decision.confidence
                modified_signal["validation_reasoning"] = decision.reasoning
                validated.append(modified_signal)
                logger.info(
                    "Signal {} APPROVED | action={} | confidence={:.2f}",
                    signal.get("signal_id", "?"),
                    decision.action,
                    decision.confidence,
                )
            else:
                logger.info(
                    "Signal {} REJECTED | reason={}",
                    signal.get("signal_id", "?"),
                    decision.reasoning,
                )

        state.validated_signals = validated
        logger.info(
            "Pipeline complete for {} | {}/{} signals validated",
            instrument_key,
            len(validated),
            len(signals),
        )

        return state
