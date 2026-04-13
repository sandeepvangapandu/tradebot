"""LLM-powered signal validation agent.

Evaluates raw trading signals and decides whether to approve,
reject, or modify them based on market regime, sentiment,
risk-reward, and historical lessons.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.models import SignalDecision

VALID_ACTIONS = {"approve", "reject", "modify"}

SYSTEM_PROMPT = """You are a signal validation agent for an Indian equity/derivatives trading bot.

Your job is quality control: review trading signals and decide whether they should be executed.

For each signal, decide:
- "approve": Signal is good, execute as-is
- "reject": Signal is bad, do not execute
- "modify": Signal has merit but needs adjustment (SL, target, or size)

Respond ONLY with valid JSON:
{
    "action": "approve|reject|modify",
    "adjusted_stop_loss": <paisa or null>,
    "adjusted_target": <paisa or null>,
    "adjusted_quantity_pct": <0.25 to 1.5, default 1.0>,
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence>"
}

Rules:
- REJECT signals that go against the current market regime
- REJECT signals with risk-reward ratio below 1.5:1
- MODIFY signals in volatile markets by reducing quantity
- APPROVE signals that align with regime and have good risk-reward
"""


class SignalValidatorAgent(BaseAgent):
    """Validates trading signals using LLM with deterministic fallback.

    Args:
        llm_client: Configured LLM client for calling the language model.
        min_risk_reward: Minimum acceptable risk-reward ratio (default 1.5).
    """

    def __init__(self, llm_client: LLMClient, min_risk_reward: float = 1.5) -> None:
        super().__init__(llm_client)
        self._min_rr = min_risk_reward

    @property
    def agent_name(self) -> str:
        """Unique name for this agent."""
        return "signal_validator"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from the signal context.

        Args:
            context: Dict containing signal, regime, sentiment fields.

        Returns:
            Formatted prompt string for the LLM.
        """
        sig = context.get("signal", {})
        return (
            "Validate this trading signal:\n"
            f"Direction: {sig.get('direction', 'unknown')}\n"
            f"Instrument: {sig.get('instrument_key', 'unknown')}\n"
            f"Strategy: {sig.get('strategy', 'unknown')}\n"
            f"Entry Price: {sig.get('entry_price', 0)} paisa\n"
            f"Stop Loss: {sig.get('stop_loss', 0)} paisa\n"
            f"Target: {sig.get('target', 0)} paisa\n"
            f"Signal Confidence: {sig.get('confidence', 0):.2f}\n\n"
            f"Market Regime: {context.get('regime', 'unknown')}\n"
            f"Sentiment Score: {context.get('sentiment_score', 50):.0f}/100 "
            f"({context.get('sentiment_classification', 'neutral')})\n"
        )

    def _build_system_prompt(self, lessons: list[str]) -> str:
        """Build the system prompt, optionally injecting memory lessons.

        Args:
            lessons: Past lessons about signal quality from memory.

        Returns:
            Full system prompt string.
        """
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about signal quality:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse and validate raw LLM JSON response.

        Args:
            raw: Raw text response from the LLM.

        Returns:
            Structured result dict with action, adjustments, confidence, reasoning.

        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON.
            ValueError: If required fields are missing or invalid.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        action = data.get("action", "reject")
        if action not in VALID_ACTIONS:
            action = "reject"

        return {
            "action": action,
            "adjusted_stop_loss": data.get("adjusted_stop_loss"),
            "adjusted_target": data.get("adjusted_target"),
            "adjusted_quantity_pct": float(data.get("adjusted_quantity_pct", 1.0)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable or parsing fails.

        Checks signal confidence, risk-reward ratio, and regime alignment
        in priority order and returns approve/reject accordingly.

        Args:
            context: Dict containing signal, regime, and sentiment fields.

        Returns:
            Structured result dict with action and reasoning.
        """
        sig = context.get("signal", {})
        regime = context.get("regime", "unknown")
        direction = sig.get("direction", "").upper()
        confidence = sig.get("confidence", 0.0)
        entry = sig.get("entry_price", 0)
        sl = sig.get("stop_loss", 0)
        target = sig.get("target", 0)

        # 1. Check confidence threshold
        if confidence < 0.5:
            return {
                "action": "reject",
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "adjusted_quantity_pct": 1.0,
                "confidence": 0.8,
                "reasoning": f"Fallback: signal confidence {confidence:.2f} < 0.5",
            }

        # 2. Check risk-reward ratio
        if entry > 0 and sl > 0 and target > 0:
            if direction == "BUY":
                risk = entry - sl
                reward = target - entry
            else:
                risk = sl - entry
                reward = entry - target

            if risk > 0:
                rr = reward / risk
                if rr < self._min_rr:
                    return {
                        "action": "reject",
                        "adjusted_stop_loss": None,
                        "adjusted_target": None,
                        "adjusted_quantity_pct": 1.0,
                        "confidence": 0.85,
                        "reasoning": (
                            f"Fallback: risk-reward {rr:.2f} < {self._min_rr}"
                        ),
                    }

        # 3. Check regime alignment
        counter_trend = (
            (direction == "BUY" and regime == "trending_down")
            or (direction == "SELL" and regime == "trending_up")
        )
        if counter_trend:
            return {
                "action": "reject",
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "adjusted_quantity_pct": 1.0,
                "confidence": 0.75,
                "reasoning": (
                    f"Fallback: {direction} is counter-trend in {regime}"
                ),
            }

        return {
            "action": "approve",
            "adjusted_stop_loss": None,
            "adjusted_target": None,
            "adjusted_quantity_pct": 1.0,
            "confidence": 0.6,
            "reasoning": "Fallback: signal passes basic checks",
        }

    def to_decision(self, result: dict[str, Any], signal_id: str = "") -> SignalDecision:
        """Convert a run() result dict to a typed SignalDecision dataclass.

        Args:
            result: Output from agent.run().
            signal_id: ID of the signal being evaluated.

        Returns:
            SignalDecision dataclass instance.
        """
        return SignalDecision(
            signal_id=signal_id,
            action=result["action"],
            confidence=result["confidence"],
            reasoning=result.get("reasoning", ""),
            adjusted_stop_loss=result.get("adjusted_stop_loss"),
            adjusted_target=result.get("adjusted_target"),
            adjusted_quantity_pct=result.get("adjusted_quantity_pct", 1.0),
        )
