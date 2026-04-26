"""Trade Analyzer - Main orchestrator for the AI Trade Research Engine.

The TradeAnalyzer coordinates all sub-analyzers to provide a comprehensive
research report for each trading signal. It runs analyses in parallel and
combines results into a final trade decision.

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.research.correlation_analyzer import CorrelationAnalyzer
from src.research.event_calendar import EventCalendarAnalyzer
from src.research.market_regime import MarketRegimeDetector, SignalDirection, StrategyType
from src.research.models import (
    AnalysisComponent,
    CorrelationData,
    EventData,
    MarketRegime,
    OptimalEntryType,
    PatternData,
    SignalStrength,
    TradeResearchReport,
    TradeVerdict,
    VolatilityRegime,
)
from src.research.options_analyzer import OptionsAnalyzer
from src.research.pattern_recognition import PatternRecognizer
from src.research.technical_analyzer import TechnicalAnalyzer
from src.research.volume_analyzer import VolumeAnalyzer

# Optional enrichment module
try:
    from src.research.multi_strategy import StrategyConfluenceAnalyzer, ConfluenceLevel
except ImportError:
    StrategyConfluenceAnalyzer = None  # type: ignore[misc,assignment]
    ConfluenceLevel = None  # type: ignore[misc,assignment]

# Optional ML validator
try:
    from src.research.ml_validator import MLSignalValidator
except ImportError:
    MLSignalValidator = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from src.data.bar_builder import BarBuilder
    from src.data.historical import HistoricalDataFetcher
    from src.data.instruments import InstrumentManager
    from src.strategy.engine import Signal

IST = ZoneInfo("Asia/Kolkata")

# Default component weights for final score calculation
DEFAULT_COMPONENT_WEIGHTS = {
    "trend_alignment": 0.25,
    "momentum": 0.20,
    "support_resistance": 0.20,
    "candle_context": 0.15,
    "market_regime": 0.15,
    "volume": 0.15,
    "options": 0.20,
    "pattern": 0.10,
    "correlation": 0.10,
    "events": 0.10,
    "ml_validator": 0.10,
    "strategy_confluence": 0.15,
}


@dataclass
class TradeScorecard:
    """Scorecard for aggregating analysis results into a final decision.

    Attributes:
        components: List of individual analysis components
        weights: Component weights for scoring
        min_execute_score: Minimum score to execute trade
        score_for_full_size: Score required for full position size
    """

    components: list[AnalysisComponent] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_COMPONENT_WEIGHTS.copy())
    min_execute_score: float = 65.0
    score_for_full_size: float = 75.0

    def add_component(self, component: AnalysisComponent) -> None:
        """Add an analysis component to the scorecard."""
        self.components.append(component)

    def calculate_final_score(self) -> float:
        """Calculate weighted final score from all components.

        Returns:
            Weighted composite score (0-100)
        """
        if not self.components:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for component in self.components:
            # If a component severely degraded organically due to missing Backtest data
            # (defaulting to 50 score with low <=50 confidence), we strip its weight entirely
            # so it doesn't drag down the average of valid technical indicators.
            if component.score == 50.0 and component.confidence <= 50.0:
                continue

            weight = self.weights.get(component.name, 0.1)
            weighted_sum += component.score * weight * (component.confidence / 100.0)
            total_weight += weight * (component.confidence / 100.0)

        if total_weight == 0:
            return 0.0

        return min(100.0, max(0.0, weighted_sum / total_weight))

    def get_verdict(self, final_score: float) -> TradeVerdict:
        """Determine trade verdict based on final score.

        Args:
            final_score: The calculated final score

        Returns:
            TradeVerdict: EXECUTE, REDUCE_SIZE, or SKIP
        """
        if final_score >= self.score_for_full_size:
            return TradeVerdict.EXECUTE
        elif final_score >= self.min_execute_score:
            return TradeVerdict.REDUCE_SIZE
        else:
            return TradeVerdict.SKIP

    def get_signal_strength(self, final_score: float) -> SignalStrength:
        """Classify signal strength based on final score.

        Args:
            final_score: The calculated final score

        Returns:
            SignalStrength classification
        """
        if final_score >= 85:
            return SignalStrength.STRONG_BUY
        elif final_score >= 75:
            return SignalStrength.BUY
        elif final_score >= 65:
            return SignalStrength.WEAK_BUY
        elif final_score >= 45:
            return SignalStrength.NEUTRAL
        elif final_score >= 35:
            return SignalStrength.WEAK_SELL
        elif final_score >= 25:
            return SignalStrength.SELL
        else:
            return SignalStrength.STRONG_SELL

    def get_adjustments(self, final_score: float) -> dict[str, float]:
        """Calculate position sizing and SL/target adjustments.

        Args:
            final_score: The calculated final score

        Returns:
            Dictionary with quantity_multiplier, sl_adjustment, target_adjustment
        """
        adjustments = {
            "quantity_multiplier": 1.0,
            "sl_adjustment": 1.0,
            "target_adjustment": 1.0,
            "entry_type": OptimalEntryType.MARKET,
        }

        # Quantity multiplier based on dynamic score targets
        # NOTE: OrderManager._calculate_research_size_multiplier() already
        # applies score-based sizing (0.7x for REDUCE_SIZE). Setting this
        # to 1.0 for executable trades avoids double-penalising position size.
        if final_score >= self.score_for_full_size + 10:
            adjustments["quantity_multiplier"] = 1.0
        elif final_score >= self.score_for_full_size:
            adjustments["quantity_multiplier"] = 1.0  # Full size is explicitly granted for reaching score_for_full_size!
        elif final_score >= self.min_execute_score:
            adjustments["quantity_multiplier"] = 0.5
        else:
            adjustments["quantity_multiplier"] = 0.0

        # SL adjustment - tighter SL for lower confidence
        if final_score >= self.score_for_full_size:
            adjustments["sl_adjustment"] = 1.0
        elif final_score >= self.min_execute_score + 5:
            adjustments["sl_adjustment"] = 0.9
        else:
            adjustments["sl_adjustment"] = 0.8

        # Target adjustment - more conservative targets for lower confidence
        if final_score >= self.score_for_full_size:
            adjustments["target_adjustment"] = 1.0
        elif final_score >= self.min_execute_score + 5:
            adjustments["target_adjustment"] = 0.9
        else:
            adjustments["target_adjustment"] = 0.8

        return adjustments

    def get_key_risks(self) -> list[str]:
        """Extract key risk factors from components."""
        risks = []

        for component in self.components:
            if component.score < 40:
                risks.append(f"{component.name}: {component.reasoning[:100]}")

        return risks[:5]  # Top 5 risks

    def get_key_supports(self) -> list[str]:
        """Extract key supporting factors from components."""
        supports = []

        for component in self.components:
            if component.score > 80:
                supports.append(f"{component.name}: {component.reasoning[:100]}")

        return supports[:5]  # Top 5 supports




class TradeAnalyzer:
    """Main orchestrator for trade signal analysis.

    Coordinates all sub-analyzers to provide comprehensive research
    on trading signals and generates a final execution decision.

    Args:
        settings: Application settings/configuration
        bar_builder: BarBuilder instance for accessing market data
        historical_fetcher: HistoricalDataFetcher for additional data
        instrument_manager: InstrumentManager for instrument lookup
    """

    def __init__(
        self,
        settings: Any,
        bar_builder: BarBuilder,
        historical_fetcher: HistoricalDataFetcher,
        instrument_manager: InstrumentManager,
        ml_validator: Optional[MLSignalValidator] = None,
        confluence_analyzer: Optional[StrategyConfluenceAnalyzer] = None,
    ) -> None:
        """Initialize the trade analyzer with all sub-analyzers.

        Args:
            settings: Application settings/configuration
            bar_builder: BarBuilder instance for accessing market data
            historical_fetcher: HistoricalDataFetcher for additional data
            instrument_manager: InstrumentManager for instrument lookup
            ml_validator: Optional MLSignalValidator for ML-based signal validation
            confluence_analyzer: Optional StrategyConfluenceAnalyzer for multi-strategy confluence
        """
        self._settings = settings
        self._bar_builder = bar_builder
        self._historical_fetcher = historical_fetcher
        self._instrument_manager = instrument_manager

        # Initialize sub-analyzers
        self._technical_analyzer = TechnicalAnalyzer()
        self._regime_detector = MarketRegimeDetector(cache_minutes=5)
        self._volume_analyzer = VolumeAnalyzer()
        self._options_analyzer = OptionsAnalyzer()
        self._correlation_analyzer = CorrelationAnalyzer()
        self._event_analyzer = EventCalendarAnalyzer()
        self._pattern_recognizer = PatternRecognizer()

        self._logger = logger.bind(module="TradeAnalyzer")
        self._executor = ThreadPoolExecutor(max_workers=8)

        # Initialize optional analyzers
        self._ml_validator = ml_validator
        self._confluence_analyzer = confluence_analyzer

        # Create default ML validator if sklearn is available but no validator provided
        if self._ml_validator is None and MLSignalValidator is not None:
            try:
                self._ml_validator = MLSignalValidator()
                self._logger.info("Created default MLSignalValidator instance")
            except Exception as e:
                self._logger.warning(f"Could not create default ML validator: {e}")

    def analyze(self, signal: Signal) -> TradeResearchReport:
        """Run comprehensive analysis on a trading signal.

        Runs independent analyses in parallel and combines results
        into a final research report with execution decision.

        Args:
            signal: Trading signal from strategy engine

        Returns:
            TradeResearchReport with analysis and verdict
        """
        start_time = datetime.now(IST)
        signal_id = str(uuid.uuid4())[:8]

        self._logger.info(
            f"[{signal_id}] Starting analysis for {signal.instrument_key} "
            f"{signal.signal_type.value} at {signal.price}"
        )

        # Gather all required data
        data = self._gather_data(signal)

        # Determine signal direction
        signal_direction = "BUY" if signal.signal_type.value in ["BUY", "BUY_CE"] else "SELL"

        # Run analyses in parallel using thread pool
        scorecard = TradeScorecard(
            min_execute_score=self._settings.research_min_score,
            score_for_full_size=self._settings.research_score_for_full_size,
            weights=self._settings.research_weights or DEFAULT_COMPONENT_WEIGHTS.copy()
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}

            # Submit independent analyses
            if "bars_5min" in data and not data["bars_5min"].empty:
                futures[executor.submit(
                    self._technical_analyzer.analyze_trend_alignment,
                    data.get("bars_all_tf", {}),
                    signal_direction
                )] = "trend"

                futures[executor.submit(
                    self._technical_analyzer.analyze_momentum,
                    data["bars_5min"],
                    signal_direction
                )] = "momentum"

                futures[executor.submit(
                    self._technical_analyzer.analyze_support_resistance,
                    data["bars_5min"],
                    signal.price or 0,
                    signal_direction
                )] = "sr"

                futures[executor.submit(
                    self._technical_analyzer.analyze_candle_context,
                    data["bars_5min"],
                    signal_direction
                )] = "candle"

                futures[executor.submit(
                    self._volume_analyzer.analyze_volume_confirmation,
                    data["bars_5min"],
                    signal_direction.lower()
                )] = "volume"

                futures[executor.submit(
                    self._pattern_recognizer.analyze,
                    data["bars_5min"],
                    signal_direction,
                    "5m"
                )] = "pattern"

            if "bars_15min" in data and "bars_daily" in data:
                futures[executor.submit(
                    self._regime_detector.detect_regime,
                    data["bars_15min"],
                    data["bars_daily"],
                    signal.instrument_key
                )] = "regime"

            futures[executor.submit(
                self._event_analyzer.analyze,
                signal.timestamp,
                signal.instrument_key
            )] = "events"

            # Collect results as they complete
            regime_result = None
            volatility_result = None
            pattern_data = PatternData()
            trend_data = None
            momentum_data = None
            sr_data = None
            volume_data = None
            event_data = EventData()

            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result(timeout=1.5)

                    if analysis_type == "trend":
                        scorecard.add_component(result)
                        trend_data = result.data
                    elif analysis_type == "momentum":
                        scorecard.add_component(result)
                        momentum_data = result.data
                    elif analysis_type == "sr":
                        scorecard.add_component(result)
                        sr_data = result.data
                    elif analysis_type == "candle":
                        scorecard.add_component(result)
                    elif analysis_type == "volume":
                        component = AnalysisComponent(
                            name="volume",
                            score=result.score,
                            weight=0.20,
                            confidence=75.0,
                            reasoning=getattr(result, "reasoning", "Volume analysis"),
                            data=result.data if isinstance(result.data, dict) else (result.data.__dict__ if hasattr(result.data, "__dict__") else {}),
                        )
                        scorecard.add_component(component)
                        volume_data = result.data if isinstance(result.data, dict) else (result.data.__dict__ if hasattr(result.data, "__dict__") else {})
                    elif analysis_type == "pattern":
                        pattern_data = result
                        # Convert pattern to component
                        pattern_score = result.pattern_strength if result.confirms_signal else (
                            100 - result.pattern_strength if result.contradicts_signal else 50
                        )
                        pattern_component = AnalysisComponent(
                            name="pattern",
                            score=pattern_score,
                            weight=0.15,
                            confidence=60.0,
                            reasoning=f"Pattern: {result.pattern_detected}, "
                                      f"confirms: {result.confirms_signal}, "
                                      f"contradicts: {result.contradicts_signal}",
                            data=result.model_dump(),
                        )
                        scorecard.add_component(pattern_component)
                    elif analysis_type == "regime":
                        regime_result, volatility_result, component = result
                        scorecard.add_component(component)
                    elif analysis_type == "events":
                        # EventCalendarAnalyzer.analyze() returns (AnalysisComponent, EventData)
                        if isinstance(result, tuple):
                            component, event_data = result
                            scorecard.add_component(component)
                        else:
                            scorecard.add_component(result)
                            event_data = EventData(**result.data) if result.data else EventData()

                except Exception as e:
                    self._logger.error(f"[{signal_id}] Error in {analysis_type} analysis: {e}")

        # Options analysis if applicable
        options_data = None
        if self._is_option_signal(signal) and "option_data" in data:
            try:
                option_component, options_data_model = self._options_analyzer.analyze(
                    data["option_data"],
                    signal_direction
                )
                scorecard.add_component(option_component)
                options_data = options_data_model
            except Exception as e:
                self._logger.error(f"[{signal_id}] Error in options analysis: {e}")

        # ML Validator analysis
        if self._ml_validator is not None and "bars_5min" in data:
            try:
                signal_dict = {
                    "type": signal.signal_type.value,
                    "entry_price": signal.price,
                    "timestamp": signal.timestamp,
                    "instrument_key": signal.instrument_key,
                    "signal_id": signal.signal_id,
                }
                ml_component = self._ml_validator.validate_signal(signal_dict, data["bars_5min"])
                scorecard.add_component(ml_component)
                self._logger.info(
                    f"[{signal_id}] ML prediction: {ml_component.data.get('prediction', 'unknown')} "
                    f"with confidence {ml_component.data.get('confidence', 0):.1f}%"
                )
            except Exception as e:
                self._logger.warning(f"[{signal_id}] ML validation failed: {e}")

        # Strategy Confluence analysis
        if self._confluence_analyzer is not None:
            try:
                from src.research.multi_strategy import StrategySignal

                # Convert Signal to StrategySignal for confluence analysis
                strategy_signal = StrategySignal(
                    strategy_name=signal.strategy_name,
                    signal_type=signal.signal_type.value,
                    confidence=70.0,  # Default confidence
                    timestamp=signal.timestamp,
                    underlying=signal.underlying or signal.instrument_key,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                )

                confluence_result = self._confluence_analyzer.get_confluence_for_signal(strategy_signal)

                # Create AnalysisComponent from confluence result
                confluence_component = AnalysisComponent(
                    name="strategy_confluence",
                    score=confluence_result.score,
                    weight=0.15,
                    confidence=confluence_result.weighted_confidence,
                    reasoning=(
                        f"Confluence level: {confluence_result.level.name}, "
                        f"Agreeing strategies: {confluence_result.agreeing_strategies}, "
                        f"Conflicting: {confluence_result.conflicting_strategies}"
                    ),
                    data={
                        "level": confluence_result.level.value,
                        "level_name": confluence_result.level.name,
                        "agreeing": confluence_result.agreeing_strategies,
                        "conflicting": confluence_result.conflicting_strategies,
                        "weighted_confidence": confluence_result.weighted_confidence,
                        "recommendation": confluence_result.recommendation,
                    }
                )
                scorecard.add_component(confluence_component)

                self._logger.info(
                    f"[{signal_id}] Strategy confluence: {confluence_result.level.name} "
                    f"(score: {confluence_result.score:.1f}, agreeing: {len(confluence_result.agreeing_strategies)})"
                )

            except Exception as e:
                self._logger.warning(f"[{signal_id}] Confluence analysis failed: {e}")

        # Calculate final results
        final_score = scorecard.calculate_final_score()

        # Apply confluence penalty if no confluence
        if self._confluence_analyzer is not None:
            try:
                from src.research.multi_strategy import ConfluenceLevel

                # Only punish for 'No Confluence' if there are actually multiple strategies activated in the engine
                active_strats = len(self._confluence_analyzer.get_registered_strategies())
                confluence_result = self._confluence_analyzer.get_confluence_for_signal(strategy_signal)
                
                if confluence_result.level == ConfluenceLevel.NONE and active_strats > 1:
                    final_score = max(0, final_score - 20)
                    self._logger.info(f"[{signal_id}] No strategy confluence, reducing score by 20 points")
            except Exception:
                pass  # Ignore errors in confluence penalty application
        verdict = scorecard.get_verdict(final_score)
        signal_strength = scorecard.get_signal_strength(final_score)
        adjustments = scorecard.get_adjustments(final_score)

        # Build the report
        report = TradeResearchReport(
            signal_id=signal_id,
            timestamp=datetime.now(IST),
            instrument_key=signal.instrument_key,
            direction=signal_direction,
            strategy_name=signal.strategy_name,
            final_score=final_score,
            components=scorecard.components,
            trend_data=trend_data if trend_data else {},
            momentum_data=momentum_data if momentum_data else {},
            sr_data=sr_data,
            volume_data=volume_data if volume_data else {},
            options_data=options_data,
            correlation_data={},  # Could be added if market data available
            event_data=event_data,
            pattern_data=pattern_data,
            verdict=verdict,
            signal_strength=signal_strength,
            market_regime=regime_result if regime_result else MarketRegime.RANGING,
            volatility_regime=volatility_result if volatility_result else VolatilityRegime.NORMAL,
            suggested_quantity_multiplier=adjustments["quantity_multiplier"],
            suggested_sl_adjustment=adjustments["sl_adjustment"],
            suggested_target_adjustment=adjustments["target_adjustment"],
            optimal_entry_type=adjustments["entry_type"],
            key_risks=scorecard.get_key_risks(),
            key_supports=scorecard.get_key_supports(),
            summary=self._generate_summary(signal, final_score, verdict, scorecard),
        )

        # Log and store report
        self._log_report(report)

        elapsed = (datetime.now(IST) - start_time).total_seconds()
        self._logger.info(
            f"[{signal_id}] Analysis complete in {elapsed:.2f}s - "
            f"Score: {final_score:.1f}, Verdict: {verdict.value}"
        )

        return report

    def _gather_data(self, signal: Signal) -> dict[str, Any]:
        """Gather all required data for analysis.

        Args:
            signal: Trading signal

        Returns:
            Dictionary with all required data
        """
        data: dict[str, Any] = {}

        # Get bars from BarBuilder for all timeframes
        bars_all_tf = {}
        for tf in [1, 5, 15]:
            try:
                df = self._bar_builder.get_bars(signal.instrument_key, tf, include_current=True)
                if not df.empty:
                    bars_all_tf[self._tf_to_string(tf)] = df
                    if tf == 5:
                        data["bars_5min"] = df
                    elif tf == 15:
                        data["bars_15min"] = df
                    elif tf == 1:
                        data["bars_1min"] = df
            except Exception as e:
                self._logger.warning(f"Could not get {tf}min bars for {signal.instrument_key}: {e}")

        data["bars_all_tf"] = bars_all_tf

        # Get historical daily data if needed
        if "bars_15min" in data and len(data["bars_15min"]) > 0:
            try:
                daily_df = self._historical_fetcher.fetch_candles(
                    signal.instrument_key,
                    interval="day",
                    days=30
                )
                if not daily_df.empty:
                    data["bars_daily"] = daily_df
            except Exception as e:
                self._logger.warning(f"Could not fetch daily data for {signal.instrument_key}: {e}")

        # Get option data if applicable
        if self._is_option_signal(signal):
            data["option_data"] = self._gather_option_data(signal)

        return data

    def _gather_option_data(self, signal: Signal) -> dict[str, Any]:
        """Gather option-specific data.

        Args:
            signal: Trading signal

        Returns:
            Dictionary with option data
        """
        option_data: dict[str, Any] = {
            "instrument_key": signal.instrument_key,
            "underlying_key": signal.underlying or signal.instrument_key,
            "option_chain": [],
            "current_iv": 0.0,
            "historical_ivs": [],
            "delta": 0.0,
            "theta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "premium": signal.price or 0,
            "days_to_expiry": 0,
            "is_weekly": False,
            "is_monthly": False,
        }

        try:
            # Skip option data when instrument manager is unavailable (e.g. backtest)
            if self._instrument_manager is None:
                return option_data

            # Try to get option chain from instrument manager
            if signal.underlying:
                # Find nearest expiry
                weekly_expiry = self._instrument_manager.get_weekly_expiry(
                    signal.underlying.replace("NSE_INDEX|", "").replace("NSE_EQ:", "")
                )
                if weekly_expiry:
                    chain_df = self._instrument_manager.get_option_chain(
                        signal.underlying,
                        weekly_expiry
                    )
                    if not chain_df.empty:
                        # Convert to OptionChainItem format
                        chain_items = []
                        for _, row in chain_df.iterrows():
                            chain_items.append({
                                "strike": int(row.get("strike_price", 0) * 100),
                                "expiry": row.get("expiry", ""),
                                "ce_oi": int(row.get("ce_oi", 0)),
                                "ce_volume": int(row.get("ce_volume", 0)),
                                "ce_iv": float(row.get("ce_iv", 0)),
                                "ce_ltp": int(row.get("ce_ltp", 0) * 100),
                                "pe_oi": int(row.get("pe_oi", 0)),
                                "pe_volume": int(row.get("pe_volume", 0)),
                                "pe_iv": float(row.get("pe_iv", 0)),
                                "pe_ltp": int(row.get("pe_ltp", 0) * 100),
                            })
                        option_data["option_chain"] = chain_items

                    # Calculate days to expiry
                    from datetime import datetime as dt
                    expiry_dt = dt.strptime(weekly_expiry, "%Y-%m-%d")
                    days_to_expiry = (expiry_dt - dt.now()).days
                    option_data["days_to_expiry"] = max(0, days_to_expiry)
                    option_data["is_weekly"] = True

        except Exception as e:
            self._logger.warning(f"Could not gather option data: {e}")

        return option_data

    def _is_option_signal(self, signal: Signal) -> bool:
        """Check if signal is for an option instrument.

        Args:
            signal: Trading signal

        Returns:
            True if option signal
        """
        return signal.signal_type.value in ["BUY_CE", "BUY_PE"] or (
            signal.underlying is not None and signal.underlying != signal.instrument_key
        )

    def _tf_to_string(self, tf: int) -> str:
        """Convert timeframe int to string key.

        Args:
            tf: Timeframe in minutes

        Returns:
            String key like "1min", "5min", etc.
        """
        if tf == 1:
            return "1min"
        elif tf == 5:
            return "5min"
        elif tf == 15:
            return "15min"
        elif tf == 60:
            return "1h"
        elif tf == 240:
            return "4h"
        elif tf == 1440:
            return "daily"
        return f"{tf}min"

    def _generate_summary(
        self,
        signal: Signal,
        final_score: float,
        verdict: TradeVerdict,
        scorecard: TradeScorecard
    ) -> str:
        """Generate a human-readable summary of the analysis.

        Args:
            signal: Trading signal
            final_score: Final calculated score
            verdict: Trade verdict
            scorecard: Scorecard with all components

        Returns:
            Summary string (2-3 sentences)
        """
        direction = "BUY" if signal.signal_type.value in ["BUY", "BUY_CE"] else "SELL"

        parts = [
            f"{direction} signal for {signal.instrument_key} from {signal.strategy_name} strategy "
            f"scored {final_score:.0f}/100 ({verdict.value})."
        ]

        # Add key insight
        if scorecard.components:
            best = max(scorecard.components, key=lambda x: x.score)
            worst = min(scorecard.components, key=lambda x: x.score)

            if best.score > 80:
                parts.append(f"Strong {best.name} analysis supports the trade.")
            if worst.score < 40:
                parts.append(f"Weak {worst.name} ({worst.score:.0f}/100) is a concern.")

        return " ".join(parts)

    def _log_report(self, report: TradeResearchReport) -> None:
        """Log the research report and save to database if configured.

        Args:
            report: TradeResearchReport to log
        """
        # Structured logging
        self._logger.info(
            f"Research Report [{report.signal_id}]: "
            f"{report.instrument_key} {report.direction} | "
            f"Score: {report.final_score:.1f} | Verdict: {report.verdict.value} | "
            f"Regime: {report.market_regime.value}"
        )

        # Log key risks if any
        if report.key_risks:
            for risk in report.key_risks:
                self._logger.warning(f"[{report.signal_id}] Risk: {risk}")

        # Log key supports if any
        if report.key_supports:
            for support in report.key_supports:
                self._logger.info(f"[{report.signal_id}] Support: {support}")

        # TODO: Save to database if configured
        # This would integrate with the database module when available

    def should_execute(self, report: TradeResearchReport) -> tuple[bool, dict[str, Any]]:
        """Determine if trade should be executed based on research report.

        Args:
            report: TradeResearchReport from analyze()

        Returns:
            Tuple of (should_execute, execution_params)
            execution_params includes: quantity_multiplier, sl_adjustment,
            target_adjustment, entry_type
        """
        should_execute = report.verdict in [TradeVerdict.EXECUTE, TradeVerdict.REDUCE_SIZE]

        execution_params = {
            "quantity_multiplier": report.suggested_quantity_multiplier,
            "sl_adjustment": report.suggested_sl_adjustment,
            "target_adjustment": report.suggested_target_adjustment,
            "entry_type": report.optimal_entry_type.value,
            "final_score": report.final_score,
            "verdict": report.verdict.value,
            "signal_strength": report.signal_strength.value,
        }

        # Add additional context
        if report.market_regime == MarketRegime.VOLATILE:
            execution_params["volatility_warning"] = True

        if report.volatility_regime == VolatilityRegime.EXTREME:
            execution_params["extreme_volatility"] = True

        if report.pattern_data and report.pattern_data.contradicts_signal:
            execution_params["pattern_contradiction"] = True

        return should_execute, execution_params

    async def analyze_async(self, signal: Signal) -> TradeResearchReport:
        """Async wrapper for analyze method.

        Args:
            signal: Trading signal

        Returns:
            TradeResearchReport
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, signal)

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
