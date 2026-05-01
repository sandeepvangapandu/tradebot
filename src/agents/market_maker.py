"""Market Maker Agent — The Veteran Trader's Brain.

Generates the Daily Playbook incorporating:
- Scheme 1: Opening Range Radar (ORB compression/expansion classification)
- Scheme 2: OI Fortress (PCR, Max Pain, CE/PE walls as ceiling/floor)
- Scheme 3: Expiry Day Playbook (gamma-aware sizing, max-pain gravity)
- Scheme 6: Smart Size Escalator session rules
- Scheme 7: Day-of-week bias

Runs at 9:00 AM (pre-market) and optionally re-evaluates at 9:35 AM
after the Opening Range has formed.
"""

from __future__ import annotations

import json
from datetime import datetime, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger

from src.agents.base_agent import BaseAgent, extract_json
from src.agents.llm_client import LLMClient

IST = ZoneInfo("Asia/Kolkata")

SYSTEM_PROMPT = """You are the Chief Market Maker for a BankNifty algorithmic trading desk on the NSE.
You have 25 years of experience on Dalal Street. You think in probabilities, not predictions.

Your job is to write the Daily Playbook that all execution algorithms MUST follow.

You will be provided with:
1. VIX and overnight global macro context
2. News sentiment score (-1.0 to 1.0)
3. Option Chain data: PCR, Max Pain, highest CE OI strike (ceiling), highest PE OI strike (floor)
4. Whether today is an expiry day (0 DTE)
5. Day of week context
6. Opening Range data (if available — 9:35 AM update)

IMPORTANT: Output ONLY the raw JSON object below. No reasoning, no explanation, no markdown fences, no text before or after the JSON.

{
    "bias": "<Bullish | Bearish | Ranging | Volatile>",
    "confidence_multiplier": <0.0 to 2.0>,
    "forbidden_directions": ["<BUY or SELL or NONE>"],
    "oi_ceiling": <strike price in paisa or null>,
    "oi_floor": <strike price in paisa or null>,
    "max_pain_strike": <strike price in paisa or null>,
    "session_size_rules": {
        "early_morning_pct": <0.25 to 1.0>,
        "mid_morning_pct": <0.5 to 1.0>,
        "post_lunch_pct": <0.5 to 1.0>,
        "power_hour_min_conviction": <1.0 to 2.0>
    },
    "new_trade_cutoff_time": "<HH:MM IST>",
    "reasoning": "<one sentence>"
}

VETERAN TRADER RULES:
- If VIX > 20 AND it's expiry day, set confidence_multiplier to 0.5 (gamma will destroy you).
- If PCR > 1.3, bias is Bullish (market makers are selling puts = bullish hedge).
- If PCR < 0.7, bias is Bearish (market makers are selling calls = bearish hedge).
- On expiry day: new_trade_cutoff_time = "14:00", reduce all session sizes by 50%.
- On Monday: early_morning_pct = 0.25 (feel the market first after weekend).
- On Friday: forbidden_directions should avoid overnight risk (prefer same-day exits).
- If ORB range < 0.3% of spot: bias = Volatile (compression breakout coming), use breakout strategies.
- If ORB range > 1.0% of spot: bias = Ranging (big move done), use mean-reversion strategies.
- If the highest CE OI strike is very close to current spot (< 0.5%), it's a resistance ceiling — forbid BUY.
- If the highest PE OI strike is very close to current spot (< 0.5%), it's a support floor — forbid SELL.
"""


class MarketMakerAgent(BaseAgent):
    """Generates the morning strategic playbook for the algorithms.

    Incorporates OI data, ORB classification, expiry context,
    and day-of-week bias to create a veteran trader's game plan.

    Args:
        llm_client: Configured LLMClient instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(llm_client)
        self.latest_playbook: dict[str, Any] = self._default_playbook()

    def _default_playbook(self) -> dict[str, Any]:
        """Return a safe neutral playbook."""
        return {
            "bias": "Neutral",
            "confidence_multiplier": 1.0,
            "forbidden_directions": [],
            "oi_ceiling": None,
            "oi_floor": None,
            "max_pain_strike": None,
            "session_size_rules": {
                "early_morning_pct": 0.5,
                "mid_morning_pct": 1.0,
                "post_lunch_pct": 1.0,
                "power_hour_min_conviction": 1.5,
            },
            "new_trade_cutoff_time": "15:00",
            "reasoning": "Awaiting first morning playbook generation.",
        }

    @property
    def agent_name(self) -> str:
        return "market_maker"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from the morning context data.

        Args:
            context: Dict containing VIX, OI data, ORB data, expiry flag, etc.

        Returns:
            Formatted prompt string for the LLM.
        """
        now = datetime.now(IST)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = day_names[now.weekday()]

        # OI data (from OptionsAnalyzer)
        oi = context.get("oi_data", {})
        pcr = oi.get("pcr_ratio", 1.0)
        max_pain = oi.get("max_pain", None)
        ce_wall = oi.get("max_ce_oi_strike", None)
        pe_wall = oi.get("max_pe_oi_strike", None)

        # ORB data (filled at 9:35 AM update)
        orb = context.get("orb_data", {})
        orb_high = orb.get("orb_high", None)
        orb_low = orb.get("orb_low", None)
        orb_range_pct = orb.get("orb_range_pct", None)

        prompt = (
            "Generate the Daily Playbook based on this morning context:\n\n"
            f"--- GLOBAL MACRO ---\n"
            f"VIX: {context.get('vix', 15.0):.2f}\n"
            f"Overnight US Markets: {context.get('us_markets', 'Neutral')}\n"
            f"News Sentiment Score: {context.get('sentiment_score', 0.0):.2f}\n\n"
            f"--- CALENDAR ---\n"
            f"Day of Week: {day_of_week}\n"
            f"Is Expiry Day (0 DTE): {context.get('is_expiry_day', False)}\n\n"
            f"--- OPTION CHAIN (OI Fortress) ---\n"
            f"Put-Call Ratio (PCR): {pcr:.2f}\n"
            f"Max Pain Strike: {max_pain} paisa\n"
            f"Highest CE OI Strike (Ceiling): {ce_wall} paisa\n"
            f"Highest PE OI Strike (Floor): {pe_wall} paisa\n\n"
        )

        if orb_range_pct is not None:
            prompt += (
                f"--- OPENING RANGE (9:15–9:30) ---\n"
                f"ORB High: {orb_high} paisa\n"
                f"ORB Low: {orb_low} paisa\n"
                f"ORB Range as % of Spot: {orb_range_pct:.2f}%\n\n"
            )
        else:
            prompt += "--- OPENING RANGE ---\nNot yet available (pre-market playbook)\n\n"

        return prompt

    def _build_system_prompt(self, lessons: list[str]) -> str:
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons from bad playbooks:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        data = extract_json(raw)

        # Validate bias
        valid_biases = {"Bullish", "Bearish", "Ranging", "Volatile", "Neutral"}
        bias = data.get("bias", "Neutral")
        if bias not in valid_biases:
            bias = "Neutral"

        # Validate multiplier
        multiplier = float(data.get("confidence_multiplier", 1.0))
        multiplier = max(0.0, min(2.0, multiplier))

        # Validate forbidden directions
        forbidden = data.get("forbidden_directions", [])
        if not isinstance(forbidden, list):
            forbidden = []

        # Parse session size rules with defaults
        session_rules = data.get("session_size_rules", {})
        default_session = self._default_playbook()["session_size_rules"]
        safe_session = {}
        for key, default in default_session.items():
            val = session_rules.get(key, default)
            try:
                safe_session[key] = max(0.1, min(2.0, float(val)))
            except (ValueError, TypeError):
                safe_session[key] = default

        return {
            "bias": bias,
            "confidence_multiplier": multiplier,
            "forbidden_directions": forbidden,
            "oi_ceiling": data.get("oi_ceiling"),
            "oi_floor": data.get("oi_floor"),
            "max_pain_strike": data.get("max_pain_strike"),
            "session_size_rules": safe_session,
            "new_trade_cutoff_time": data.get("new_trade_cutoff_time", "15:00"),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback using pure math when LLM is unavailable."""
        playbook = self._default_playbook()

        # Apply deterministic rules without LLM
        vix = context.get("vix", 15.0)
        is_expiry = context.get("is_expiry_day", False)

        oi = context.get("oi_data", {})
        pcr = oi.get("pcr_ratio", 1.0)

        # PCR-based bias
        if pcr > 1.3:
            playbook["bias"] = "Bullish"
        elif pcr < 0.7:
            playbook["bias"] = "Bearish"

        # Expiry day: reduce size, early cutoff
        if is_expiry:
            playbook["confidence_multiplier"] = 0.5
            playbook["new_trade_cutoff_time"] = "14:00"
            playbook["session_size_rules"]["early_morning_pct"] = 0.25
            playbook["session_size_rules"]["mid_morning_pct"] = 0.5
            playbook["session_size_rules"]["post_lunch_pct"] = 0.5

        # High VIX
        if vix > 20:
            playbook["confidence_multiplier"] = min(playbook["confidence_multiplier"], 0.5)

        # OI walls
        playbook["oi_ceiling"] = oi.get("max_ce_oi_strike")
        playbook["oi_floor"] = oi.get("max_pe_oi_strike")
        playbook["max_pain_strike"] = oi.get("max_pain")

        playbook["reasoning"] = "Fallback: LLM unavailable, using deterministic PCR/VIX rules."
        return playbook

    def update_with_orb(self, orb_data: dict[str, Any]) -> dict[str, Any]:
        """Re-evaluate the playbook at 9:35 AM after the Opening Range forms.

        This is the Scheme 1 dynamic update. The ORB range classification
        can change the bias from the static 9:00 AM playbook.

        Args:
            orb_data: Dict with orb_high, orb_low, orb_range_pct.

        Returns:
            Updated playbook dict.
        """
        current = dict(self.latest_playbook)

        orb_pct = orb_data.get("orb_range_pct", 0.5)

        if orb_pct < 0.3:
            # Compression day — violent breakout incoming
            current["bias"] = "Volatile"
            current["reasoning"] = f"ORB compressed ({orb_pct:.2f}%), expecting breakout. Switching to Volatile bias."
            logger.warning("ORB COMPRESSION DETECTED: {:.2f}% — Playbook updated to Volatile", orb_pct)
        elif orb_pct > 1.0:
            # Expansion day — big move done, mean-revert the rest
            current["bias"] = "Ranging"
            current["reasoning"] = f"ORB expanded ({orb_pct:.2f}%), big move done. Switching to Ranging bias."
            logger.warning("ORB EXPANSION DETECTED: {:.2f}% — Playbook updated to Ranging", orb_pct)

        self.latest_playbook = current
        return current
