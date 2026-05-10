"""LLM-powered signal validation agent.

Evaluates raw trading signals and decides whether to approve,
reject, or modify them based on market regime, sentiment,
risk-reward, and historical lessons.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent, extract_json
from src.agents.llm_client import LLMClient
from src.agents.models import SignalDecision

VALID_ACTIONS = {"approve", "reject", "modify"}

SYSTEM_PROMPT = """You are the Tactical Signal Validator for an autonomous AI trading desk on the NSE.
You are the final gatekeeper. Every trade MUST pass through you.

You modulate the position size using a CONFIDENCE_MULTIPLIER (0.0 to 2.0).

IMPORTANT: Output ONLY the raw JSON object below. No reasoning, no explanation, no markdown fences, no text before or after the JSON.

{"confidence_multiplier": <0.0 to 2.0>, "adjusted_stop_loss": <paisa or null>, "adjusted_target": <paisa or null>, "reasoning": "<one short sentence>"}

DUAL-CONSENSUS CLASH LOGIC:
- If the algo signal direction matches the Morning Playbook Bias (e.g. BUY & Bullish), output multiplier 2.0 (High Conviction).
- If the algo signal direction is explicitly FORBIDDEN by the Morning Playbook, check the Live Context (VIX/News).
- If the Live Context supports the Algo (e.g., market suddenly crashed since morning), override the Playbook (multiplier 1.0).
- If the Live Context does NOT support the Algo, reject the trade (multiplier 0.0).

OI FORTRESS RULES:
- If the signal is BUY and the entry price is within 0.5% of the OI Ceiling (CE wall), reduce multiplier to 0.3 or reject.
- If the signal is SELL and the entry price is within 0.5% of the OI Floor (PE wall), reduce multiplier to 0.3 or reject.
- On expiry day, Max Pain acts as a gravity center — trade TOWARD max pain, not away from it.

SMART SIZE ESCALATOR:
- Apply the session_size_rules from the Playbook. Multiply the base multiplier by the session percentage.
- Multiply the final confidence_multiplier by the Day-of-Week RL Multiplier (e.g. if base is 1.0, session is 0.5, and DOW is 1.5, final is 0.75).
- During Power Hour (14:30-15:00), only approve trades with multiplier >= power_hour_min_conviction.

SAFETY RULES:
- If VIX is very high (>20), tighten the stop_loss by 20%.
- If risk-reward is poor (< 1.5), output multiplier 0.0.
- If the current time is past the Playbook's new_trade_cutoff_time, reject ALL new trades.
"""


import threading

class SignalValidatorAgent(BaseAgent):
    """Validates trading signals using LLM with deterministic fallback.

    Uses an asynchronous context (updated via update_context) and provides
    synchronous validation returning a confidence multiplier.

    Args:
        llm_client: Configured LLM client for calling the language model.
        min_risk_reward: Minimum acceptable risk-reward ratio (default 1.5).
    """

    def __init__(self, llm_client: LLMClient, min_risk_reward: float = 1.5) -> None:
        super().__init__(llm_client)
        self._min_rr = min_risk_reward
        self._context_lock = threading.Lock()
        self._latest_context: dict[str, Any] = {
            "regime": "unknown",
            "vix": 15.0,
            "sentiment_score": 0.0,
            "playbook": {"bias": "Neutral", "forbidden_directions": []},
        }

    @property
    def agent_name(self) -> str:
        """Unique name for this agent."""
        return "signal_validator"

    def update_context(
        self, 
        regime: str, 
        vix: float, 
        sentiment_score: float = 0.0, 
        playbook: dict[str, Any] | None = None
    ) -> None:
        """Update the overarching macro context asynchronously.
        
        Args:
            regime: Current market regime (e.g., 'bullish_high_vol').
            vix: Current VIX value.
            sentiment_score: Current news sentiment score (-1.0 to 1.0).
            playbook: The morning AI playbook.
        """
        if playbook is None:
            playbook = {"bias": "Neutral", "forbidden_directions": []}
            
        with self._context_lock:
            self._latest_context["regime"] = regime
            self._latest_context["vix"] = vix
            self._latest_context["sentiment_score"] = sentiment_score
            self._latest_context["playbook"] = playbook
        logger.debug("SignalValidator context updated: regime={}, vix={}, sentiment={}", regime, vix, sentiment_score)

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from the signal and the latest cached context.

        Args:
            context: Dict containing the signal to validate.

        Returns:
            Formatted prompt string for the LLM.
        """
        sig = context.get("signal", {})
        
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        current_time_str = now.strftime("%H:%M")
        
        # Determine session phase for Smart Size Escalator
        hour = now.hour
        minute = now.minute
        if hour == 9 and minute >= 30 or hour == 10 and minute <= 30:
            session_phase = "early_morning"
        elif hour == 10 and minute > 30 or hour == 11 and minute < 30:
            session_phase = "mid_morning"
        elif hour >= 13 and hour < 14 or (hour == 14 and minute < 30):
            session_phase = "post_lunch"
        elif hour == 14 and minute >= 30 or hour == 15:
            session_phase = "power_hour"
        else:
            session_phase = "off_hours"
        
        # Read day-of-week sizing if available
        dow_multiplier = 1.0
        dow = now.strftime("%A")
        try:
            from pathlib import Path
            import json
            strategy_name = sig.get('strategy', sig.get('strategy_name', 'unknown'))
            dow_path = Path(f"config/strategies/{strategy_name}_dow_sizing.json")
            if dow_path.exists():
                with open(dow_path, "r") as f:
                    dow_data = json.load(f)
                    dow_multiplier = float(dow_data.get(dow, 1.0))
        except Exception as e:
            logger.debug("Could not read DOW sizing for {}: {}", sig.get('strategy'), e)
        
        with self._context_lock:
            regime = self._latest_context.get("regime", "unknown")
            vix = self._latest_context.get("vix", 15.0)
            sent_score = self._latest_context.get("sentiment_score", 0.0)
            playbook = self._latest_context.get("playbook", {})

        session_rules = playbook.get("session_size_rules", {})

        return (
            "Validate this trading signal instantly:\n"
            f"Direction: {sig.get('direction', sig.get('signal_type', 'unknown'))}\n"
            f"Instrument: {sig.get('instrument_key', 'unknown')}\n"
            f"Strategy: {sig.get('strategy', sig.get('strategy_name', 'unknown'))}\n"
            f"Entry Price: {sig.get('entry_price', sig.get('price', 0))} paisa\n"
            f"Stop Loss: {sig.get('stop_loss', 0)} paisa\n"
            f"Target: {sig.get('target', 0)} paisa\n"
            f"Algo Confidence: {sig.get('confidence', 0.5):.2f}\n\n"
            f"--- LIVE CONTEXT (Current Time: {current_time_str} IST) ---\n"
            f"Market Regime: {regime}\n"
            f"Current VIX: {vix:.2f}\n"
            f"News Sentiment Score: {sent_score:.2f} (-1.0 is crash, 1.0 is extreme bull)\n\n"
            f"--- MORNING AI PLAYBOOK ---\n"
            f"Strategic Bias: {playbook.get('bias', 'Neutral')}\n"
            f"Forbidden Directions: {playbook.get('forbidden_directions', [])}\n"
            f"OI Ceiling (Resistance): {playbook.get('oi_ceiling', 'N/A')} paisa\n"
            f"OI Floor (Support): {playbook.get('oi_floor', 'N/A')} paisa\n"
            f"Max Pain Strike: {playbook.get('max_pain_strike', 'N/A')} paisa\n"
            f"New Trade Cutoff Time: {playbook.get('new_trade_cutoff_time', '15:00')} IST\n\n"
            f"--- SESSION SIZING (Smart Size Escalator) ---\n"
            f"Current Session Phase: {session_phase}\n"
            f"Session Size Rules: {session_rules}\n"
            f"Day-of-Week RL Multiplier: {dow_multiplier:.2f}x\n"
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
        """Parse and validate raw LLM response.

        Uses extract_json to handle responses with reasoning traces,
        markdown fences, or other wrapper text.

        Args:
            raw: Raw text response from the LLM.

        Returns:
            Structured result dict with confidence_multiplier and reasoning.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        data = extract_json(raw)
        
        multiplier = float(data.get("confidence_multiplier", 1.0))
        multiplier = max(0.0, min(2.0, multiplier))

        action = "approve" if multiplier > 0 else "reject"

        return {
            "action": action,
            "adjusted_stop_loss": data.get("adjusted_stop_loss"),
            "adjusted_target": data.get("adjusted_target"),
            "confidence_multiplier": multiplier,
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable or parsing fails.

        Checks signal confidence, risk-reward ratio, and regime alignment
        in priority order and returns a multiplier.

        Args:
            context: Dict containing signal fields.

        Returns:
            Structured result dict with multiplier and reasoning.
        """
        sig = context.get("signal", {})
        
        with self._context_lock:
            regime = self._latest_context.get("regime", "unknown")
            vix = self._latest_context.get("vix", 15.0)

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
                "confidence_multiplier": 0.0,
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
                        "confidence_multiplier": 0.0,
                        "reasoning": (
                            f"Fallback: risk-reward {rr:.2f} < {self._min_rr}"
                        ),
                    }

        # 3. Check regime alignment — reject counter-trend signals outright
        counter_trend = (
            (direction == "BUY" and "down" in regime.lower())
            or (direction == "SELL" and "up" in regime.lower())
        )
        if counter_trend:
            return {
                "action": "reject",
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "confidence_multiplier": 0.0,
                "reasoning": (
                    f"Fallback: {direction} is counter-trend in {regime}, rejecting signal"
                ),
            }

        return {
            "action": "approve",
            "adjusted_stop_loss": None,
            "adjusted_target": None,
            "confidence_multiplier": 1.0,
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
            confidence=0.9, # Confidence is no longer returned explicitly by LLM
            reasoning=result.get("reasoning", ""),
            adjusted_stop_loss=result.get("adjusted_stop_loss"),
            adjusted_target=result.get("adjusted_target"),
            adjusted_quantity_pct=result.get("confidence_multiplier", 1.0),
        )
