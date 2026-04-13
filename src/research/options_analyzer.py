"""Options analyzer for evaluating option trades.

Analyzes implied volatility, Greeks, open interest, and expiry context
to score option trading opportunities.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.research.models import AnalysisComponent, OptionsData
from src.utils.degradation import graceful_degrade


@dataclass
class OptionChainItem:
    """Represents a single strike in the option chain."""

    strike: int  # In paisa
    expiry: str  # YYYY-MM-DD format
    ce_oi: int  # Call open interest
    ce_volume: int
    ce_iv: float
    ce_ltp: int  # Last traded price in paisa
    pe_oi: int  # Put open interest
    pe_volume: int
    pe_iv: float
    pe_ltp: int  # Last traded price in paisa


class OptionsAnalyzer:
    """Analyzer for options-specific metrics.

    Evaluates IV conditions, Greeks risk, open interest patterns,
    and expiry context to score option trades.

    All prices in PAISA (int).
    """

    def __init__(self) -> None:
        """Initialize the options analyzer."""
        self.logger = logger.bind(module="OptionsAnalyzer")

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="iv_analysis",
            score=50.0,
            weight=0.25,
            confidence=30.0,
            reasoning="IV analysis failed - using default neutral score",
            data={"signal": "neutral", "score": 50},
        )
    )
    def analyze_iv(
        self,
        instrument_key: str,
        option_chain: list[OptionChainItem],
        current_iv: float,
        historical_ivs: list[float],
    ) -> AnalysisComponent:
        """Analyze implied volatility conditions for option buying.

        Calculates IV percentile and rank vs 20-day history.
        For BUYING options:
            - <30 percentile = great (90)
            - 30-60 = ok (60)
            - 60-80 = expensive (40)
            - >80 = bad (20)

        Args:
            instrument_key: The option instrument key
            option_chain: List of option chain items
            current_iv: Current implied volatility (as decimal, e.g., 0.25 for 25%)
            historical_ivs: List of historical IV values (20 days)

        Returns:
            AnalysisComponent with score 0-100 and IV data
        """
        if not historical_ivs:
            self.logger.warning(f"No historical IV data for {instrument_key}")
            return AnalysisComponent(
                name="iv_analysis",
                score=50.0,
                weight=0.25,
                confidence=30.0,
                reasoning="Insufficient historical IV data to assess",
                data={"current_iv": current_iv, "error": "no_history"},
            )

        # Calculate IV percentile
        sorted_ivs = sorted(historical_ivs)
        n = len(sorted_ivs)

        # Find position of current IV in sorted history
        position = 0
        for i, iv in enumerate(sorted_ivs):
            if current_iv <= iv:
                position = i
                break
        else:
            position = n

        iv_percentile = (position / n) * 100 if n > 0 else 50.0

        # Calculate IV rank
        iv_min = min(historical_ivs)
        iv_max = max(historical_ivs)
        iv_range = iv_max - iv_min

        if iv_range > 0:
            iv_rank = ((current_iv - iv_min) / iv_range) * 100
        else:
            iv_rank = 50.0

        # Score based on IV percentile for BUYING options
        # Lower IV is better for buying
        if iv_percentile < 30:
            score = 90.0
            assessment = "great"
        elif iv_percentile < 60:
            score = 60.0
            assessment = "ok"
        elif iv_percentile < 80:
            score = 40.0
            assessment = "expensive"
        else:
            score = 20.0
            assessment = "bad"

        reasoning = (
            f"IV at {iv_percentile:.1f}th percentile ({assessment} for buying). "
            f"IV Rank: {iv_rank:.1f}. Current IV: {current_iv*100:.1f}%, "
            f"20-day range: {iv_min*100:.1f}%-{iv_max*100:.1f}%"
        )

        return AnalysisComponent(
            name="iv_analysis",
            score=score,
            weight=0.25,
            confidence=85.0,
            reasoning=reasoning,
            data={
                "current_iv": current_iv,
                "iv_percentile": iv_percentile,
                "iv_rank": iv_rank,
                "iv_min_20d": iv_min,
                "iv_max_20d": iv_max,
                "historical_count": n,
                "assessment": assessment,
            },
        )

    def analyze_greeks(
        self,
        delta: float,
        theta: float,
        gamma: float,
        vega: float,
        premium: int,
        days_to_expiry: int,
    ) -> dict[str, Any]:
        """Analyze Greeks for risk assessment.

        Checks:
        - Delta appropriateness (0.30-0.70 for directional trades)
        - Theta as % of premium (flag if >5% daily decay)
        - Gamma risk near expiry (<3 DTE)
        - Vega exposure in high IV environment

        Args:
            delta: Option delta (-1 to 1)
            theta: Daily theta decay (negative for long options)
            gamma: Option gamma
            vega: Option vega (price change per 1% IV move)
            premium: Option premium in PAISA
            days_to_expiry: Days to expiry

        Returns:
            Dictionary with Greeks assessment
        """
        assessment = {
            "delta": {"value": delta, "appropriate": True, "concern": None},
            "theta": {"value": theta, "pct_of_premium": 0.0, "high_decay": False},
            "gamma": {"value": gamma, "risk_near_expiry": False},
            "vega": {"value": vega, "high_iv_environment": False},
            "overall_risk": "low",
            "warnings": [],
        }

        # Delta check (for directional trades, 0.30-0.70 is sweet spot)
        abs_delta = abs(delta)
        if abs_delta < 0.30:
            assessment["delta"]["appropriate"] = False
            assessment["delta"]["concern"] = "low_delta"
            assessment["warnings"].append("Low delta - option may not move much with underlying")
        elif abs_delta > 0.70:
            assessment["delta"]["appropriate"] = False
            assessment["delta"]["concern"] = "high_delta"
            assessment["warnings"].append("High delta - behaves like stock, consider underlying instead")

        # Theta as % of premium (daily)
        if premium > 0:
            theta_pct = abs(theta) / premium * 100
            assessment["theta"]["pct_of_premium"] = theta_pct

            if theta_pct > 5.0:
                assessment["theta"]["high_decay"] = True
                assessment["warnings"].append(f"High theta decay: {theta_pct:.1f}% of premium per day")
        else:
            assessment["theta"]["pct_of_premium"] = 0.0

        # Gamma risk near expiry
        if days_to_expiry < 3 and abs(gamma) > 0.01:
            assessment["gamma"]["risk_near_expiry"] = True
            assessment["warnings"].append(f"High gamma risk with {days_to_expiry} DTE")

        # Overall risk assessment
        warning_count = len(assessment["warnings"])
        if warning_count >= 3:
            assessment["overall_risk"] = "high"
        elif warning_count >= 2:
            assessment["overall_risk"] = "medium"
        elif warning_count >= 1:
            assessment["overall_risk"] = "low-medium"

        return assessment

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="open_interest_analysis",
            score=50.0,
            weight=0.20,
            confidence=30.0,
            reasoning="Open Interest analysis failed - using default neutral score",
            data={"signal": "neutral", "score": 50},
        )
    )
    def analyze_open_interest(
        self,
        underlying_key: str,
        option_chain: list[OptionChainItem],
        signal_direction: str,  # "BUY" or "SELL"
    ) -> AnalysisComponent:
        """Analyze open interest patterns for directional bias.

        Analyzes:
        - PCR (Put-Call Ratio): >1.3 = bullish, <0.7 = bearish, 0.7-1.3 = neutral
        - Max Pain calculation
        - Change in OI analysis

        Args:
            underlying_key: The underlying instrument key
            option_chain: List of option chain items
            signal_direction: "BUY" or "SELL" signal direction

        Returns:
            AnalysisComponent with score 0-100 and OI data
        """
        if not option_chain:
            self.logger.warning(f"No option chain data for {underlying_key}")
            return AnalysisComponent(
                name="open_interest_analysis",
                score=50.0,
                weight=0.20,
                confidence=30.0,
                reasoning="No option chain data available",
                data={"error": "no_chain_data"},
            )

        # Calculate total OI for PCR
        total_ce_oi = sum(item.ce_oi for item in option_chain)
        total_pe_oi = sum(item.pe_oi for item in option_chain)

        pcr_ratio = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        # Calculate Max Pain
        max_pain = self._calculate_max_pain(option_chain)

        # Analyze PCR for directional bias
        # >1.3 = bullish (more puts written = market makers bullish)
        # <0.7 = bearish (more calls written = market makers bearish)
        if pcr_ratio > 1.3:
            pcr_bias = "bullish"
            pcr_score = 80.0 if signal_direction == "BUY" else 30.0
        elif pcr_ratio < 0.7:
            pcr_bias = "bearish"
            pcr_score = 80.0 if signal_direction == "SELL" else 30.0
        else:
            pcr_bias = "neutral"
            pcr_score = 50.0

        # Find strikes with highest OI (support/resistance levels)
        max_ce_oi_strike = max(option_chain, key=lambda x: x.ce_oi)
        max_pe_oi_strike = max(option_chain, key=lambda x: x.pe_oi)

        # Build reasoning
        reasoning_parts = [
            f"PCR: {pcr_ratio:.2f} ({pcr_bias})",
            f"Max Pain: Rs {max_pain/100:.0f}" if max_pain else "Max Pain: N/A",
            f"CE OI wall at {max_ce_oi_strike.strike/100:.0f}",
            f"PE OI support at {max_pe_oi_strike.strike/100:.0f}",
        ]

        # Check if signal aligns with OI bias
        alignment = "aligned" if pcr_score >= 50 else "contradicts"

        reasoning = (
            f"OI analysis {alignment} with {signal_direction} signal. "
            f"{' '.join(reasoning_parts)}"
        )

        return AnalysisComponent(
            name="open_interest_analysis",
            score=pcr_score,
            weight=0.20,
            confidence=75.0,
            reasoning=reasoning,
            data={
                "pcr_ratio": pcr_ratio,
                "pcr_bias": pcr_bias,
                "max_pain": max_pain,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "max_ce_oi_strike": max_ce_oi_strike.strike,
                "max_pe_oi_strike": max_pe_oi_strike.strike,
                "signal_alignment": alignment,
            },
        )

    def _calculate_max_pain(self, option_chain: list[OptionChainItem]) -> int | None:
        """Calculate max pain strike price.

        Max pain is the strike where option writers (market makers)
        have minimum payout - where most options expire worthless.

        Args:
            option_chain: List of option chain items

        Returns:
            Max pain strike price in paisa, or None if cannot calculate
        """
        if not option_chain:
            return None

        strikes = sorted(set(item.strike for item in option_chain))
        if not strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0.0

            for item in option_chain:
                # Call pain: max(0, strike - test_strike) * OI
                if item.ce_oi > 0:
                    call_pain = max(0, test_strike - item.strike) * item.ce_oi
                    total_pain += call_pain

                # Put pain: max(0, test_strike - strike) * OI
                if item.pe_oi > 0:
                    put_pain = max(0, item.strike - test_strike) * item.pe_oi
                    total_pain += put_pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    def analyze_expiry_context(
        self,
        days_to_expiry: int,
        is_weekly: bool,
        is_monthly: bool,
    ) -> dict[str, Any]:
        """Analyze expiry timing context.

        Flags:
        - <2 DTE: extreme theta decay
        - 0 DTE: bump threshold to 80
        - Weekly vs monthly behavior differences

        Args:
            days_to_expiry: Days to expiry
            is_weekly: Whether this is a weekly expiry
            is_monthly: Whether this is a monthly expiry

        Returns:
            Dictionary with expiry context
        """
        context = {
            "days_to_expiry": days_to_expiry,
            "is_weekly": is_weekly,
            "is_monthly": is_monthly,
            "extreme_theta_decay": False,
            "zero_dte": False,
            "score_threshold_adjustment": 0,
            "warnings": [],
            "recommendations": [],
        }

        # Check DTE conditions
        if days_to_expiry == 0:
            context["zero_dte"] = True
            context["score_threshold_adjustment"] = 15  # Bump threshold by 15 points
            context["warnings"].append("0 DTE - very high risk, require 80+ score")
        elif days_to_expiry < 2:
            context["extreme_theta_decay"] = True
            context["score_threshold_adjustment"] = 10
            context["warnings"].append(f"{days_to_expiry} DTE - extreme theta decay")
        elif days_to_expiry < 5:
            context["score_threshold_adjustment"] = 5
            context["warnings"].append(f"{days_to_expiry} DTE - elevated theta decay")

        # Weekly vs monthly recommendations
        if is_weekly and days_to_expiry < 3:
            context["recommendations"].append("Consider monthly expiry for better theta profile")

        if is_monthly and days_to_expiry > 20:
            context["recommendations"].append("Long dated, consider weekly for better gamma")

        # Liquidity warning for far monthlies
        if is_monthly and days_to_expiry > 30:
            context["warnings"].append("Far month expiry may have lower liquidity")

        return context

    @graceful_degrade(
        default_return=(
            AnalysisComponent(
                name="options_analysis",
                score=50.0,
                weight=0.25,
                confidence=30.0,
                reasoning="Options analysis failed - using default neutral score",
                data={"signal": "neutral", "score": 50},
            ),
            OptionsData()
        )
    )
    def analyze(
        self,
        option_data: dict[str, Any],
        signal_direction: str,
    ) -> tuple[AnalysisComponent, OptionsData]:
        """Run complete options analysis.

        Combines IV, Greeks, OI, and expiry analysis into a comprehensive
        options assessment.

        Args:
            option_data: Dictionary containing:
                - instrument_key: str
                - underlying_key: str
                - option_chain: list[OptionChainItem]
                - current_iv: float
                - historical_ivs: list[float]
                - delta: float
                - theta: float
                - gamma: float
                - vega: float
                - premium: int (paisa)
                - days_to_expiry: int
                - is_weekly: bool
                - is_monthly: bool
            signal_direction: "BUY" or "SELL"

        Returns:
            Tuple of (AnalysisComponent with overall score, OptionsData with details)
        """
        instrument_key = option_data.get("instrument_key", "")
        underlying_key = option_data.get("underlying_key", "")
        option_chain = option_data.get("option_chain", [])

        # IV Analysis
        iv_component = self.analyze_iv(
            instrument_key=instrument_key,
            option_chain=option_chain,
            current_iv=option_data.get("current_iv", 0.0),
            historical_ivs=option_data.get("historical_ivs", []),
        )

        # Greeks Analysis
        greeks_assessment = self.analyze_greeks(
            delta=option_data.get("delta", 0.0),
            theta=option_data.get("theta", 0.0),
            gamma=option_data.get("gamma", 0.0),
            vega=option_data.get("vega", 0.0),
            premium=option_data.get("premium", 0),
            days_to_expiry=option_data.get("days_to_expiry", 0),
        )

        # OI Analysis
        oi_component = self.analyze_open_interest(
            underlying_key=underlying_key,
            option_chain=option_chain,
            signal_direction=signal_direction,
        )

        # Expiry Context
        expiry_context = self.analyze_expiry_context(
            days_to_expiry=option_data.get("days_to_expiry", 0),
            is_weekly=option_data.get("is_weekly", False),
            is_monthly=option_data.get("is_monthly", False),
        )

        # Calculate Greeks-based score
        greeks_score = self._calculate_greeks_score(greeks_assessment)

        # Weighted composite score
        # IV: 25%, Greeks: 30%, OI: 20%, Expiry: 25%
        weights = {
            "iv": 0.25,
            "greeks": 0.30,
            "oi": 0.20,
            "expiry": 0.25,
        }

        # Calculate expiry score (inverse of risk)
        expiry_score = 100 - (expiry_context.get("score_threshold_adjustment", 0) * 5)
        expiry_score = max(0, min(100, expiry_score))

        composite_score = (
            iv_component.score * weights["iv"] +
            greeks_score * weights["greeks"] +
            oi_component.score * weights["oi"] +
            expiry_score * weights["expiry"]
        )

        # Build reasoning
        reasoning_parts = [
            f"IV: {iv_component.score:.0f}/100 ({iv_component.data.get('assessment', 'unknown')})",
            f"Greeks: {greeks_score:.0f}/100 (risk: {greeks_assessment['overall_risk']})",
            f"OI: {oi_component.score:.0f}/100 (PCR: {oi_component.data.get('pcr_ratio', 0):.2f})",
            f"Expiry: {expiry_score:.0f}/100 ({option_data.get('days_to_expiry', 0)} DTE)",
        ]

        all_warnings = (
            greeks_assessment.get("warnings", []) +
            expiry_context.get("warnings", [])
        )

        if all_warnings:
            reasoning_parts.append(f"Warnings: {len(all_warnings)}")

        overall_reasoning = (
            f"Options composite score: {composite_score:.0f}/100. "
            f"Components: {', '.join(reasoning_parts)}"
        )

        # Create overall component
        overall_component = AnalysisComponent(
            name="options_analysis",
            score=composite_score,
            weight=0.25,  # Weight in overall research report
            confidence=min(iv_component.confidence, oi_component.confidence),
            reasoning=overall_reasoning,
            data={
                "iv_component": iv_component.data,
                "greeks_assessment": greeks_assessment,
                "oi_component": oi_component.data,
                "expiry_context": expiry_context,
                "component_scores": {
                    "iv": iv_component.score,
                    "greeks": greeks_score,
                    "oi": oi_component.score,
                    "expiry": expiry_score,
                },
                "warnings": all_warnings,
            },
        )

        # Create OptionsData model
        options_data_model = OptionsData(
            iv_current=option_data.get("current_iv"),
            iv_percentile=iv_component.data.get("iv_percentile"),
            iv_rank=iv_component.data.get("iv_rank"),
            delta=option_data.get("delta"),
            theta=option_data.get("theta"),
            gamma=option_data.get("gamma"),
            vega=option_data.get("vega"),
            theta_pct_of_premium=greeks_assessment["theta"].get("pct_of_premium"),
            pcr_ratio=oi_component.data.get("pcr_ratio"),
            max_pain=oi_component.data.get("max_pain"),
            days_to_expiry=option_data.get("days_to_expiry"),
            is_weekly_expiry=option_data.get("is_weekly", False),
            is_monthly_expiry=option_data.get("is_monthly", False),
        )

        return overall_component, options_data_model

    def _calculate_greeks_score(self, greeks_assessment: dict[str, Any]) -> float:
        """Calculate a score from Greeks assessment.

        Args:
            greeks_assessment: Output from analyze_greeks

        Returns:
            Score 0-100
        """
        score = 100.0

        # Delta penalties
        if not greeks_assessment["delta"]["appropriate"]:
            score -= 20

        # Theta decay penalty
        if greeks_assessment["theta"]["high_decay"]:
            score -= 25

        # Gamma risk penalty
        if greeks_assessment["gamma"]["risk_near_expiry"]:
            score -= 20

        # Overall risk adjustment
        risk_level = greeks_assessment["overall_risk"]
        if risk_level == "high":
            score -= 30
        elif risk_level == "medium":
            score -= 15
        elif risk_level == "low-medium":
            score -= 5

        # Warning penalties
        warning_count = len(greeks_assessment.get("warnings", []))
        score -= warning_count * 10

        return max(0, min(100, score))
