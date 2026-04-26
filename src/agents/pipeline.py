"""LangGraph-based agent pipeline for intelligent trade filtering.

Uses LangGraph StateGraph to orchestrate the flow:
  START -> regime_detection -> (short_circuit | signal_validation) -> END

Each node reads from and writes to a shared TypedDict state.
The conditional edge after regime detection short-circuits when
regime confidence falls below the configured threshold.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.llm_client import LLMClient
from src.agents.models import (
    AgentDecision,
    PipelineState,
    RegimeClassification,
    SentimentScore,
)
from src.agents.regime_agent import RegimeAgent
from src.agents.signal_validator import SignalValidatorAgent
from src.memory.injector import MemoryInjector
from src.memory.memory_db import MemoryDB


# ---------------------------------------------------------------------------
# LangGraph state schema — a flat TypedDict that flows through every node.
# ---------------------------------------------------------------------------

class _GraphState(TypedDict, total=False):
    """Internal state carried between graph nodes."""

    instrument_key: str
    signals: list[dict[str, Any]]
    indicator_context: dict[str, Any]
    sentiment_score: float
    sentiment_classification: str

    # Written by regime_detection node
    regime: str
    regime_confidence: float
    regime_reasoning: str
    regime_indicators: dict[str, float]

    # Written by signal_validation node
    validated_signals: list[dict[str, Any]]

    # Accumulator for agent decisions
    agent_decisions: list[AgentDecision]

    # Short-circuit metadata
    short_circuited: bool
    short_circuit_reason: str


class AgentPipeline:
    """Orchestrates AI agents via a LangGraph StateGraph.

    Graph topology::

        START ──> regime_detection ──┬──> signal_validation ──> END
                                     │
                                     └──> short_circuit ──────> END

    The conditional edge after ``regime_detection`` inspects
    regime confidence.  If it falls below *regime_confidence_threshold*
    the graph jumps to the ``short_circuit`` terminal node, skipping
    signal validation entirely.

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

        # Build the compiled graph once at init time.
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph StateGraph."""
        graph = StateGraph(_GraphState)

        # Nodes
        graph.add_node("regime_detection", self._node_regime_detection)
        graph.add_node("signal_validation", self._node_signal_validation)
        graph.add_node("short_circuit", self._node_short_circuit)

        # Edges
        graph.add_edge(START, "regime_detection")

        graph.add_conditional_edges(
            "regime_detection",
            self._route_after_regime,
            {"validate": "signal_validation", "short_circuit": "short_circuit"},
        )

        graph.add_edge("signal_validation", END)
        graph.add_edge("short_circuit", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_regime_detection(self, state: _GraphState) -> dict[str, Any]:
        """Classify market regime from indicator context."""
        indicator_context = state.get("indicator_context", {})
        instrument_key = state.get("instrument_key", "")

        regime_lessons = self._injector.get_lessons_for_agent(
            agent_name="regime_agent",
            regime="",  # No regime known yet
        )
        regime_result = self._regime_agent.run(
            context=indicator_context,
            lessons=regime_lessons,
            instrument_key=instrument_key,
        )

        decisions: list[AgentDecision] = list(state.get("agent_decisions", []))
        if self._regime_agent.last_decision:
            decisions.append(self._regime_agent.last_decision)

        classification = self._regime_agent.to_classification(regime_result)

        logger.info(
            "Pipeline regime: {} (confidence={:.2f}) for {}",
            classification.regime,
            classification.confidence,
            instrument_key,
        )

        return {
            "regime": classification.regime,
            "regime_confidence": classification.confidence,
            "regime_reasoning": classification.reasoning,
            "regime_indicators": classification.indicators,
            "agent_decisions": decisions,
        }

    def _node_signal_validation(self, state: _GraphState) -> dict[str, Any]:
        """Validate each signal against regime and risk-reward rules."""
        signals = state.get("signals", [])
        regime = state.get("regime", "unknown")
        instrument_key = state.get("instrument_key", "")
        sentiment_score = state.get("sentiment_score", 50.0)
        sentiment_classification = state.get("sentiment_classification", "neutral")

        decisions: list[AgentDecision] = list(state.get("agent_decisions", []))
        validated: list[dict[str, Any]] = []

        for signal in signals:
            validator_lessons = self._injector.get_lessons_for_agent(
                agent_name="signal_validator",
                regime=regime,
                strategy=signal.get("strategy", ""),
            )
            validation_context = {
                "signal": signal,
                "regime": regime,
                "sentiment_score": sentiment_score,
                "sentiment_classification": sentiment_classification,
            }
            val_result = self._signal_validator.run(
                context=validation_context,
                lessons=validator_lessons,
                instrument_key=instrument_key,
            )

            if self._signal_validator.last_decision:
                decisions.append(self._signal_validator.last_decision)

            decision = self._signal_validator.to_decision(
                val_result, signal_id=signal.get("signal_id", ""),
            )

            if decision.is_approved:
                modified_signal = dict(signal)
                if decision.adjusted_stop_loss is not None:
                    modified_signal["stop_loss"] = decision.adjusted_stop_loss
                if decision.adjusted_target is not None:
                    modified_signal["target"] = decision.adjusted_target
                if decision.adjusted_quantity_pct != 1.0:
                    orig_qty = modified_signal.get("quantity", 1)
                    modified_signal["quantity"] = max(
                        1, int(orig_qty * decision.adjusted_quantity_pct),
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

        logger.info(
            "Pipeline complete for {} | {}/{} signals validated",
            instrument_key,
            len(validated),
            len(signals),
        )

        return {
            "validated_signals": validated,
            "agent_decisions": decisions,
            "short_circuited": False,
        }

    def _node_short_circuit(self, state: _GraphState) -> dict[str, Any]:
        """Terminal node when regime confidence is too low."""
        instrument_key = state.get("instrument_key", "")
        confidence = state.get("regime_confidence", 0.0)
        reason = (
            f"Regime confidence {confidence:.2f} "
            f"< threshold {self._confidence_threshold}"
        )
        logger.warning("Pipeline short-circuited: {} | {}", instrument_key, reason)

        return {
            "short_circuited": True,
            "short_circuit_reason": reason,
            "validated_signals": [],
        }

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route_after_regime(self, state: _GraphState) -> str:
        """Decide whether to validate signals or short-circuit."""
        confidence = state.get("regime_confidence", 0.0)
        if confidence < self._confidence_threshold:
            return "short_circuit"
        return "validate"

    # ------------------------------------------------------------------
    # Public API (same interface as before)
    # ------------------------------------------------------------------

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
        if not signals:
            logger.debug("Pipeline: no signals to process for {}", instrument_key)
            return PipelineState(instrument_key=instrument_key, signals=signals)

        # Seed the graph state
        initial_state: _GraphState = {
            "instrument_key": instrument_key,
            "signals": signals,
            "indicator_context": indicator_context,
            "sentiment_score": sentiment_score,
            "sentiment_classification": sentiment_classification,
            "agent_decisions": [],
            "validated_signals": [],
            "short_circuited": False,
            "short_circuit_reason": "",
            "regime": "unknown",
            "regime_confidence": 0.0,
            "regime_reasoning": "",
            "regime_indicators": {},
        }

        # Execute the compiled graph
        result = self._graph.invoke(initial_state)

        # Convert flat graph state back to PipelineState dataclass
        regime = RegimeClassification(
            regime=result.get("regime", "unknown"),
            confidence=result.get("regime_confidence", 0.0),
            reasoning=result.get("regime_reasoning", ""),
            indicators=result.get("regime_indicators", {}),
        )

        return PipelineState(
            instrument_key=instrument_key,
            signals=signals,
            regime=regime,
            validated_signals=result.get("validated_signals", []),
            agent_decisions=result.get("agent_decisions", []),
            short_circuited=result.get("short_circuited", False),
            short_circuit_reason=result.get("short_circuit_reason", ""),
        )
