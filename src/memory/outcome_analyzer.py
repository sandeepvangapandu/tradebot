"""Trade outcome analyzer with MAE/MFE metrics.

Computes Maximum Adverse Excursion (MAE), Maximum Favorable Excursion
(MFE), efficiency, and flags like premature_exit and stop_loss_hit.
All monetary values in paisa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class TradeOutcome:
    """Raw data for a completed trade.

    Attributes:
        trade_id: Unique trade identifier.
        strategy: Strategy that generated this trade.
        instrument_key: Instrument traded.
        direction: BUY or SELL.
        entry_price: Entry price in paisa.
        exit_price: Exit price in paisa.
        stop_loss: Original stop loss in paisa.
        target: Original target in paisa.
        quantity: Number of units.
        realized_pnl: Total P&L in paisa.
        max_adverse_excursion: Maximum price move against position in paisa.
        max_favorable_excursion: Maximum price move in favor in paisa.
        holding_seconds: Time held in seconds.
    """

    trade_id: str
    strategy: str
    instrument_key: str
    direction: str
    entry_price: int
    exit_price: int
    stop_loss: int
    target: int
    quantity: int
    realized_pnl: int
    max_adverse_excursion: int
    max_favorable_excursion: int
    holding_seconds: int


class OutcomeAnalyzer:
    """Analyzes trade outcomes to extract performance metrics.

    Computes MAE/MFE percentages, trade efficiency, and detects
    patterns like premature exits and stop-loss hits.
    """

    def analyze(self, outcome: TradeOutcome) -> dict[str, Any]:
        """Analyze a single trade outcome.

        Args:
            outcome: Completed trade data with MAE/MFE.

        Returns:
            Dictionary with computed metrics and flags.
        """
        was_win = outcome.realized_pnl > 0
        entry = outcome.entry_price

        # MAE/MFE as percentage of entry price
        mae_pct = (outcome.max_adverse_excursion / entry * 100) if entry > 0 else 0.0
        mfe_pct = (outcome.max_favorable_excursion / entry * 100) if entry > 0 else 0.0

        # Efficiency: how much of MFE was captured
        # Positive for winners, negative for losers
        if outcome.max_favorable_excursion > 0:
            actual_gain = abs(outcome.exit_price - outcome.entry_price)
            efficiency = actual_gain / outcome.max_favorable_excursion
            if not was_win:
                efficiency = -efficiency
        else:
            efficiency = 0.0

        # Premature exit: won but captured less than 50% of MFE
        premature_exit = was_win and efficiency < 0.5

        # Late exit: won but gave back more than 40% of MFE
        if was_win and outcome.max_favorable_excursion > 0:
            actual_capture = abs(outcome.exit_price - outcome.entry_price)
            giveback = outcome.max_favorable_excursion - actual_capture
            late_exit = giveback > 0.4 * outcome.max_favorable_excursion
        else:
            late_exit = False

        # Stop loss hit: MAE is within 10% of SL distance
        if outcome.direction == "BUY":
            sl_distance = outcome.entry_price - outcome.stop_loss
        else:
            sl_distance = outcome.stop_loss - outcome.entry_price

        if sl_distance > 0:
            stop_loss_hit = outcome.max_adverse_excursion >= sl_distance * 0.9
        else:
            stop_loss_hit = False

        # Quick stop: hit SL within 10 minutes
        quick_stop = stop_loss_hit and outcome.holding_seconds < 600

        analysis = {
            "trade_id": outcome.trade_id,
            "strategy": outcome.strategy,
            "instrument_key": outcome.instrument_key,
            "direction": outcome.direction,
            "was_win": was_win,
            "pnl_paisa": outcome.realized_pnl,
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "efficiency": efficiency,
            "premature_exit": premature_exit,
            "late_exit": late_exit,
            "stop_loss_hit": stop_loss_hit,
            "quick_stop": quick_stop,
            "holding_seconds": outcome.holding_seconds,
        }

        logger.debug(
            "Outcome analysis | {} | win={} | efficiency={:.2f} | "
            "premature={} | sl_hit={} | quick_stop={}",
            outcome.trade_id,
            was_win,
            efficiency,
            premature_exit,
            stop_loss_hit,
            quick_stop,
        )

        return analysis
