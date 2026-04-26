"""LLM-powered market regime detection agent.

Classifies the current market as trending_up, trending_down,
ranging, or volatile. Uses technical indicators as input and
the LLM for nuanced classification. Falls back to deterministic
rules when LLM is unavailable.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent, extract_json
from src.agents.llm_client import LLMClient
from src.agents.models import RegimeClassification

VALID_REGIMES = {"trending_up", "trending_down", "ranging", "volatile"}

SYSTEM_PROMPT = """You are a market regime classification agent for Indian equity/derivatives markets.

Given technical indicator data, classify the market into one of:
- trending_up: Strong upward momentum, ADX > 25, price near highs
- trending_down: Strong downward momentum, ADX > 25, price near lows
- ranging: Sideways, ADX < 20, no clear direction
- volatile: Extreme ATR, erratic moves, high uncertainty

IMPORTANT: Output ONLY the raw JSON object below. No reasoning, no explanation, no markdown fences, no text before or after the JSON.

{"regime": "<regime>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Rules:
- confidence < 0.3 means you are very uncertain — the pipeline will skip trading
- Consider ALL indicators together, not just one
- ATR percentile > 90 strongly suggests volatile regime
- Bollinger Band squeeze suggests potential breakout from ranging
"""


class RegimeAgent(BaseAgent):
    """Classifies market regime using LLM with deterministic fallback.

    Inherits from BaseAgent which handles the run() loop, LLM invocation,
    circuit-breaker guarding, and fallback dispatch. This class only needs
    to implement the four abstract methods.

    Args:
        llm_client: Configured LLMClient instance (may have is_configured=False).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(llm_client)

    @property
    def agent_name(self) -> str:
        """Unique name for this agent."""
        return "regime_agent"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from technical indicator context.

        Args:
            context: Dict containing ADX, RSI, ATR percentile, Bollinger Band
                     width, price position, EMAs, and squeeze flag.

        Returns:
            Formatted prompt string for the LLM.
        """
        return (
            "Classify the market regime from these indicators:\n"
            f"ADX: {context.get('adx', 0):.1f} ({context.get('adx_trend', 'unknown')})\n"
            f"RSI: {context.get('rsi', 50):.1f}\n"
            f"ATR Percentile: {context.get('atr_percentile', 50):.1f}%\n"
            f"BB Width: {context.get('bb_width', 0):.2f} "
            f"(percentile: {context.get('bb_width_percentile', 50):.1f}%)\n"
            f"Price Position in 20d Range: {context.get('price_position_in_range', 50):.1f}%\n"
            f"EMA 9: {context.get('ema_9', 0):.2f}, EMA 21: {context.get('ema_21', 0):.2f}\n"
            f"SMA 50: {context.get('sma_50', 0):.2f}\n"
            f"BB Squeeze: {context.get('is_squeeze', False)}\n"
        )

    def _build_system_prompt(self, lessons: list[str]) -> str:
        """Build the system prompt, optionally injecting past memory lessons.

        Args:
            lessons: Past lessons about regime misclassification from the
                     learning module.

        Returns:
            System prompt string.
        """
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about regime misclassification:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse raw LLM response into a structured result dict.

        Uses extract_json to handle responses with reasoning traces,
        markdown fences, or other wrapper text. Validates regime value
        and clamps confidence to [0.0, 1.0].

        Args:
            raw: Raw string content from LLM response.

        Returns:
            Dict with keys: regime, confidence, reasoning.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        data = extract_json(raw)
        regime = data.get("regime", "ranging")
        if regime not in VALID_REGIMES:
            logger.warning(
                "regime_agent: invalid regime '{}' from LLM, defaulting to 'ranging'",
                regime,
            )
            regime = "ranging"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "regime": regime,
            "confidence": confidence,
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic regime classification when LLM is unavailable.

        Priority order:
        1. volatile  — ATR percentile > 90
        2. trending_up   — ADX > 25 and price position > 60%
        3. trending_down — ADX > 25 and price position < 40%
        4. ranging   — default

        Args:
            context: Same context dict passed to _build_prompt.

        Returns:
            Dict with keys: regime, confidence, reasoning.
        """
        adx = context.get("adx", 20.0)
        atr_pct = context.get("atr_percentile", 50.0)
        price_pos = context.get("price_position_in_range", 50.0)

        if atr_pct > 90:
            return {
                "regime": "volatile",
                "confidence": 0.7,
                "reasoning": f"Fallback: ATR percentile {atr_pct:.0f}% > 90",
            }
        if adx > 25:
            if price_pos > 60:
                return {
                    "regime": "trending_up",
                    "confidence": 0.6,
                    "reasoning": (
                        f"Fallback: ADX={adx:.0f} > 25, "
                        f"price near highs ({price_pos:.0f}%)"
                    ),
                }
            if price_pos < 40:
                return {
                    "regime": "trending_down",
                    "confidence": 0.6,
                    "reasoning": (
                        f"Fallback: ADX={adx:.0f} > 25, "
                        f"price near lows ({price_pos:.0f}%)"
                    ),
                }
        return {
            "regime": "ranging",
            "confidence": 0.5,
            "reasoning": f"Fallback: ADX={adx:.0f}, no clear trend",
        }

    def to_classification(self, result: dict[str, Any]) -> RegimeClassification:
        """Convert a run() result dict to a typed RegimeClassification dataclass.

        Args:
            result: Dict returned by run(), containing regime, confidence,
                    and reasoning.

        Returns:
            RegimeClassification instance.
        """
        return RegimeClassification(
            regime=result["regime"],
            confidence=result["confidence"],
            reasoning=result.get("reasoning", ""),
        )
