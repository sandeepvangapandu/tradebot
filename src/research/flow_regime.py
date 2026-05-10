"""Flow regime classification (FII/DII streaks, combined signal).

Reads accumulated ``fii_dii_flows_daily`` rows and derives:

* **Per-entity regime** (STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL) based
  on the net flow threshold bands.
* **Consecutive-day streak** (positive = buying days, negative = selling days).
* **Combined signal** (TAILWIND / HEADWIND / MIXED).

The derived values are written to ``flow_regime_daily`` as a daily snapshot and
are also returned as a plain dict for callers that don't need DB access (e.g.
unit tests and condition evaluators).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (₹ crore)
# ---------------------------------------------------------------------------

_STRONG_BUY_THRESHOLD = 2000.0
_BUY_THRESHOLD = 500.0
_SELL_THRESHOLD = -500.0
_STRONG_SELL_THRESHOLD = -2000.0

_STREAK_LOOKBACK = 30  # days


# ---------------------------------------------------------------------------
# FlowRegimeAnalyzer
# ---------------------------------------------------------------------------

class FlowRegimeAnalyzer:
    """Classifies FII/DII flow regime and computes streaks.

    Args:
        db_engine: SQLAlchemy engine connected to the trading DB.  When
            ``None`` the ``compute_regime`` and ``get_today_regime`` methods
            that require DB access will return ``None``.
    """

    def __init__(self, db_engine=None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def classify_fii_regime(self, fii_net: float) -> str:
        """Classify FII flow regime based on net value.

        Args:
            fii_net: FII net flow in ₹ crore (positive = net buy).

        Returns:
            One of 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'.
        """
        return self._classify_regime(fii_net)

    def classify_dii_regime(self, dii_net: float) -> str:
        """Classify DII flow regime based on net value.

        Args:
            dii_net: DII net flow in ₹ crore (positive = net buy).

        Returns:
            One of 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'.
        """
        return self._classify_regime(dii_net)

    @staticmethod
    def _classify_regime(net_crore: float) -> str:
        """Shared regime classification logic.

        Thresholds (₹ crore):
        * > 2000  → STRONG_BUY
        * 500-2000 → BUY
        * -500 to 500 → NEUTRAL
        * -2000 to -500 → SELL
        * < -2000 → STRONG_SELL

        Args:
            net_crore: Net flow value.

        Returns:
            Regime label string.
        """
        if net_crore >= _STRONG_BUY_THRESHOLD:
            return "STRONG_BUY"
        if net_crore >= _BUY_THRESHOLD:
            return "BUY"
        if net_crore > _SELL_THRESHOLD:
            return "NEUTRAL"
        if net_crore > _STRONG_SELL_THRESHOLD:
            return "SELL"
        return "STRONG_SELL"

    # ------------------------------------------------------------------
    # Streak computation
    # ------------------------------------------------------------------

    def compute_streak(self, side: str, lookback_days: int = _STREAK_LOOKBACK) -> int:
        """Compute consecutive same-direction days for FII or DII.

        Positive return value = consecutive buying days.
        Negative return value = consecutive selling days.
        0 = no data or mixed (today's data not committed yet).

        Args:
            side: ``'fii'`` or ``'dii'``.
            lookback_days: How many historical rows to scan.

        Returns:
            Signed integer streak count.  E.g. ``3`` means 3 consecutive
            net-positive (buying) days; ``-2`` means 2 consecutive net-negative
            (selling) days.
        """
        if self._engine is None:
            return 0

        col = "fii_net_value_crore" if side == "fii" else "dii_net_value_crore"
        from sqlalchemy import text as sa_text

        with self._engine.connect() as conn:
            rows = conn.execute(
                sa_text(
                    f"SELECT {col} FROM fii_dii_flows_daily "  # noqa: S608
                    "ORDER BY trade_date DESC LIMIT :n"
                ),
                {"n": lookback_days},
            ).fetchall()

        if not rows:
            return 0

        values = [float(r[0]) for r in rows if r[0] is not None]
        if not values:
            return 0

        first_sign = 1 if values[0] >= 0 else -1
        streak = 0
        for v in values:
            sign = 1 if v >= 0 else -1
            if sign == first_sign:
                streak += 1
            else:
                break

        return streak * first_sign

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def combine_signals(self, fii_regime: str, dii_regime: str) -> str:
        """Combine FII and DII regimes into a single market flow signal.

        Rules:
        * TAILWIND: both FII and DII are BUY or STRONG_BUY.
        * HEADWIND: both FII and DII are SELL or STRONG_SELL.
        * MIXED: any other combination.

        Args:
            fii_regime: Classified FII regime string.
            dii_regime: Classified DII regime string.

        Returns:
            'TAILWIND', 'HEADWIND', or 'MIXED'.
        """
        buy_set = {"BUY", "STRONG_BUY"}
        sell_set = {"SELL", "STRONG_SELL"}

        if fii_regime in buy_set and dii_regime in buy_set:
            return "TAILWIND"
        if fii_regime in sell_set and dii_regime in sell_set:
            return "HEADWIND"
        return "MIXED"

    # ------------------------------------------------------------------
    # Full regime computation + persistence
    # ------------------------------------------------------------------

    def compute_regime(self, trade_date: date | None = None) -> dict | None:
        """Compute the full flow regime for *trade_date* and persist it.

        Reads today's ``fii_dii_flows_daily`` row, classifies regimes,
        computes streaks, and upserts into ``flow_regime_daily``.

        Args:
            trade_date: The date to compute regime for.  Defaults to today.

        Returns:
            Regime dict ``{fii_streak, dii_streak, fii_regime, dii_regime,
            combined_signal}`` or ``None`` if today's flow data is missing.
        """
        if trade_date is None:
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            trade_date = datetime.now(ist).date()

        if self._engine is None:
            return None

        from sqlalchemy import text as sa_text

        with self._engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT fii_net_value_crore, dii_net_value_crore "
                    "FROM fii_dii_flows_daily WHERE trade_date = :d"
                ),
                {"d": trade_date},
            ).fetchone()

        if row is None:
            logger.warning("No FII/DII flow data found for %s", trade_date)
            return None

        fii_net = float(row[0]) if row[0] is not None else 0.0
        dii_net = float(row[1]) if row[1] is not None else 0.0

        fii_regime = self.classify_fii_regime(fii_net)
        dii_regime = self.classify_dii_regime(dii_net)
        combined = self.combine_signals(fii_regime, dii_regime)
        fii_streak = self.compute_streak("fii")
        dii_streak = self.compute_streak("dii")

        regime: dict[str, Any] = {
            "trade_date": trade_date,
            "fii_streak": fii_streak,
            "dii_streak": dii_streak,
            "fii_regime": fii_regime,
            "dii_regime": dii_regime,
            "combined_signal": combined,
        }

        # Persist to flow_regime_daily
        with self._engine.begin() as conn:
            conn.execute(
                sa_text(
                    """
                    INSERT INTO flow_regime_daily
                        (trade_date, fii_streak, dii_streak,
                         fii_regime, dii_regime, combined_signal)
                    VALUES
                        (:d, :fs, :ds, :fr, :dr, :cs)
                    ON CONFLICT (trade_date) DO UPDATE SET
                        fii_streak      = EXCLUDED.fii_streak,
                        dii_streak      = EXCLUDED.dii_streak,
                        fii_regime      = EXCLUDED.fii_regime,
                        dii_regime      = EXCLUDED.dii_regime,
                        combined_signal = EXCLUDED.combined_signal
                    """
                ),
                {
                    "d": trade_date,
                    "fs": fii_streak,
                    "ds": dii_streak,
                    "fr": fii_regime,
                    "dr": dii_regime,
                    "cs": combined,
                },
            )

        logger.info(
            "Flow regime for %s: FII=%s (streak=%d), DII=%s (streak=%d), signal=%s",
            trade_date, fii_regime, fii_streak, dii_regime, dii_streak, combined,
        )
        return regime

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_today_regime(self, trade_date: date | None = None) -> dict | None:
        """Retrieve today's flow regime from the DB (read-only).

        Args:
            trade_date: The date to query.  Defaults to today.

        Returns:
            Regime dict or ``None`` if no record exists.
        """
        if trade_date is None:
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            trade_date = datetime.now(ist).date()

        if self._engine is None:
            return None

        from sqlalchemy import text as sa_text

        with self._engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT fii_streak, dii_streak, fii_regime, dii_regime, combined_signal "
                    "FROM flow_regime_daily WHERE trade_date = :d"
                ),
                {"d": trade_date},
            ).fetchone()

        if row is None:
            return None

        return {
            "trade_date": trade_date,
            "fii_streak": row[0],
            "dii_streak": row[1],
            "fii_regime": row[2],
            "dii_regime": row[3],
            "combined_signal": row[4],
        }
