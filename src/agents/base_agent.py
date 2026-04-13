"""Abstract base class for all LLM-powered agents.

Every agent follows the same pattern:
1. Build a prompt from context data.
2. Call the LLM via LLMClient.
3. Parse the JSON response.
4. If any step fails, fall back to a deterministic implementation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from src.agents.llm_client import LLMClient
from src.agents.models import AgentDecision


class BaseAgent(ABC):
    """Abstract base for LLM-powered trading agents.

    Subclasses must implement: agent_name, _build_prompt,
    _build_system_prompt, _parse_response, _fallback.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._last_decision: AgentDecision | None = None

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique name for this agent."""

    @abstractmethod
    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from context data."""

    @abstractmethod
    def _build_system_prompt(self, lessons: list[str]) -> str:
        """Build the system prompt, including injected memory lessons."""

    @abstractmethod
    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse raw LLM text to structured output."""

    @abstractmethod
    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable or fails."""

    @property
    def last_decision(self) -> AgentDecision | None:
        return self._last_decision

    def run(
        self,
        context: dict[str, Any],
        lessons: list[str] | None = None,
        instrument_key: str = "",
    ) -> dict[str, Any]:
        """Run the agent against the provided context.

        Attempts LLM invocation when configured; falls back to the
        deterministic implementation on any failure.

        Args:
            context: Input data the agent should analyse.
            lessons: Optional memory lessons to inject into the system prompt.
            instrument_key: Instrument identifier for decision tracking.

        Returns:
            Structured result dict from either the LLM or the fallback.
        """
        lessons = lessons or []
        start = time.monotonic()

        if not self._llm.is_configured:
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        prompt = self._build_prompt(context)
        system_prompt = self._build_system_prompt(lessons)
        response = self._llm.invoke(prompt, system_prompt=system_prompt)

        if not response.success:
            logger.info(
                "{} LLM failed ({}), using fallback",
                self.agent_name,
                response.error,
            )
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        try:
            result = self._parse_response(response.content)
        except Exception as e:
            logger.warning(
                "{} failed to parse LLM response: {}. Using fallback.",
                self.agent_name,
                e,
            )
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        self._record_decision(
            instrument_key=instrument_key,
            context=context,
            result=result,
            used_fallback=False,
            latency_ms=response.latency_ms,
            lessons_count=len(lessons),
        )
        return result

    def _record_decision(
        self,
        instrument_key: str,
        context: dict[str, Any],
        result: dict[str, Any],
        used_fallback: bool,
        latency_ms: int,
        lessons_count: int,
    ) -> None:
        """Store the most recent decision as an AgentDecision for observability.

        Args:
            instrument_key: Instrument this decision relates to.
            context: Input context (truncated for storage).
            result: Output result (truncated for storage).
            used_fallback: Whether the deterministic fallback was used.
            latency_ms: Total end-to-end latency in milliseconds.
            lessons_count: Number of memory lessons injected.
        """
        confidence = result.get("confidence", 0.5)
        self._last_decision = AgentDecision(
            agent_name=self.agent_name,
            instrument_key=instrument_key,
            input_summary=str(context)[:200],
            output_summary=str(result)[:200],
            confidence=confidence,
            used_fallback=used_fallback,
            latency_ms=latency_ms,
            lessons_injected=lessons_count,
        )
