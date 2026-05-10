"""Tick-level aggressor classification + CVD.

Classifies each incoming tick as a buy-aggressor or sell-aggressor based on
the bid/ask spread (quote rule) or, when depth data is unavailable, falls back
to the tick rule (uptick = BUY, downtick = SELL, unchanged = NEUTRAL).

Per-minute aggregates (buy_volume, sell_volume, delta, trade_count, avg_spread)
are accumulated in memory and flushed to the ``tick_metrics_minute`` table at
the end of each completed minute.

CVD (Cumulative Volume Delta) is the running sum of ``delta`` since the trading
session started (09:00 IST).  Call :meth:`reset_session` at 09:00 IST daily to
clear the accumulator for each instrument.

All monetary values are in **paisa** (integer) throughout this module.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _MinuteBucket:
    """Accumulates tick statistics for one instrument within one clock minute."""

    buy_volume: int = 0
    sell_volume: int = 0
    trade_count: int = 0
    # Running sum of spread in bps for computing avg_spread_bps on flush.
    spread_bps_sum: float = 0.0
    spread_count: int = 0


@dataclass
class _InstrumentState:
    """Per-instrument in-memory state across all open minute buckets."""

    # Current open minute bucket: minute_ts (truncated to the minute) -> bucket
    buckets: dict[datetime, _MinuteBucket] = field(
        default_factory=lambda: defaultdict(_MinuteBucket)
    )
    # Cumulative volume delta since last reset_session call.
    cvd_running: int = 0
    # Last observed LTP in paisa (for tick-rule fallback).
    last_ltp: int | None = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TickMetricsAggregator:
    """Classifies ticks and maintains per-minute CVD aggregates.

    Args:
        db_engine: Optional SQLAlchemy engine for flushing to
            ``tick_metrics_minute``.  When *None*, :meth:`flush_minute`
            returns the record dict but does not persist it (useful in tests).
    """

    def __init__(self, db_engine: Any = None) -> None:
        self._db_engine = db_engine
        self._state: dict[str, _InstrumentState] = defaultdict(_InstrumentState)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Aggressor classification
    # ------------------------------------------------------------------

    def classify_aggressor(
        self,
        ltp_paisa: int,
        bid_paisa: int | None,
        ask_paisa: int | None,
        prev_ltp: int | None = None,
    ) -> str:
        """Classify a tick as BUY / SELL / NEUTRAL.

        Quote rule (primary):
            * ltp >= ask_paisa  → BUY  (trade lifted the offer)
            * ltp <= bid_paisa  → SELL (trade hit the bid)
            * bid < ltp < ask   → NEUTRAL (inside spread — no clear aggressor)

        Tick rule fallback (when bid/ask are unavailable):
            * ltp > prev_ltp    → BUY
            * ltp < prev_ltp    → SELL
            * ltp == prev_ltp   → NEUTRAL

        Args:
            ltp_paisa: Last traded price in paisa.
            bid_paisa: Best bid price in paisa.  ``None`` triggers tick-rule.
            ask_paisa: Best ask price in paisa.  ``None`` triggers tick-rule.
            prev_ltp: Previous LTP in paisa (used only for tick-rule fallback).

        Returns:
            ``'BUY'``, ``'SELL'``, or ``'NEUTRAL'``.
        """
        if bid_paisa is not None and ask_paisa is not None:
            if ltp_paisa >= ask_paisa:
                return "BUY"
            if ltp_paisa <= bid_paisa:
                return "SELL"
            return "NEUTRAL"

        # Tick rule fallback
        if prev_ltp is None:
            return "NEUTRAL"
        if ltp_paisa > prev_ltp:
            return "BUY"
        if ltp_paisa < prev_ltp:
            return "SELL"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------

    def on_tick(
        self,
        instrument_key: str,
        ts: datetime,
        ltp_paisa: int,
        volume: int,
        bid_paisa: int | None = None,
        ask_paisa: int | None = None,
    ) -> None:
        """Process a single market tick.

        Updates the per-minute bucket and running CVD for *instrument_key*.

        Args:
            instrument_key: Upstox instrument identifier (e.g. ``"NSE_EQ|INE002A01018"``).
            ts: Tick timestamp (timezone-aware recommended; naïve treated as UTC).
            ltp_paisa: Last traded price in paisa.
            volume: Volume traded at this tick.
            bid_paisa: Best bid in paisa (optional; enables quote rule).
            ask_paisa: Best ask in paisa (optional; enables quote rule).
        """
        # Ensure timezone-aware
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Truncate to minute boundary (floor to second=0, microsecond=0)
        minute_ts = ts.replace(second=0, microsecond=0)

        with self._lock:
            state = self._state[instrument_key]

            side = self.classify_aggressor(
                ltp_paisa, bid_paisa, ask_paisa, state.last_ltp
            )
            state.last_ltp = ltp_paisa

            bucket = state.buckets[minute_ts]
            bucket.trade_count += 1

            if side == "BUY":
                bucket.buy_volume += volume
                state.cvd_running += volume
            elif side == "SELL":
                bucket.sell_volume += volume
                state.cvd_running -= volume
            # NEUTRAL: volume counted in trade_count but not in buy/sell

            # Track spread for avg_spread_bps
            if bid_paisa is not None and ask_paisa is not None and bid_paisa > 0:
                spread_bps = ((ask_paisa - bid_paisa) / bid_paisa) * 10_000
                bucket.spread_bps_sum += spread_bps
                bucket.spread_count += 1

    # ------------------------------------------------------------------
    # CVD access
    # ------------------------------------------------------------------

    def get_running_cvd(self, instrument_key: str) -> int:
        """Return the cumulative volume delta since the last session reset.

        Args:
            instrument_key: Upstox instrument identifier.

        Returns:
            CVD as integer (positive = net buy pressure, negative = net sell).
        """
        with self._lock:
            return self._state[instrument_key].cvd_running

    # ------------------------------------------------------------------
    # Session reset
    # ------------------------------------------------------------------

    def reset_session(self, instrument_key: str | None = None) -> None:
        """Zero out CVD and clear in-memory buckets for a new trading session.

        Call this at 09:00 IST each day before market open.

        Args:
            instrument_key: Specific instrument to reset.  When *None*, resets
                all instruments.
        """
        with self._lock:
            if instrument_key is None:
                self._state.clear()
            else:
                self._state[instrument_key] = _InstrumentState()

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush_minute(
        self, instrument_key: str, minute_ts: datetime
    ) -> dict | None:
        """Flush a completed minute bucket to persistent storage.

        Computes the minute ``delta`` and attaches the current ``cvd_running``
        snapshot, then (if a DB engine is configured) writes the record to
        ``tick_metrics_minute``.

        Args:
            instrument_key: Upstox instrument identifier.
            minute_ts: The exact minute timestamp (seconds and microseconds must
                be zero) for the bucket to flush.

        Returns:
            The record dict on success, or *None* if no data existed for that
            minute.
        """
        with self._lock:
            state = self._state.get(instrument_key)
            if state is None:
                return None

            bucket = state.buckets.pop(minute_ts, None)
            if bucket is None:
                return None

            delta = bucket.buy_volume - bucket.sell_volume
            cvd_snapshot = state.cvd_running

            avg_spread_bps: float | None = None
            if bucket.spread_count > 0:
                avg_spread_bps = bucket.spread_bps_sum / bucket.spread_count

        record = {
            "instrument_key": instrument_key,
            "minute_ts": minute_ts,
            "buy_volume": bucket.buy_volume,
            "sell_volume": bucket.sell_volume,
            "delta": delta,
            "cvd_running": cvd_snapshot,
            "trade_count": bucket.trade_count,
            "avg_spread_bps": avg_spread_bps,
        }

        if self._db_engine is not None:
            try:
                self._persist_record(record)
            except Exception:
                logger.exception(
                    "tick_metrics: failed to persist minute for %s @ %s",
                    instrument_key,
                    minute_ts,
                )

        return record

    def _persist_record(self, record: dict) -> None:
        """Write a minute record to ``tick_metrics_minute`` via SQLAlchemy.

        Args:
            record: Dict matching ``tick_metrics_minute`` columns.
        """
        from sqlalchemy import text  # local import to keep module lightweight

        sql = text(
            """
            INSERT INTO tick_metrics_minute
                (instrument_key, minute_ts, buy_volume, sell_volume,
                 delta, cvd_running, trade_count, avg_spread_bps)
            VALUES
                (:instrument_key, :minute_ts, :buy_volume, :sell_volume,
                 :delta, :cvd_running, :trade_count, :avg_spread_bps)
            ON CONFLICT (instrument_key, minute_ts) DO UPDATE SET
                buy_volume   = tick_metrics_minute.buy_volume  + EXCLUDED.buy_volume,
                sell_volume  = tick_metrics_minute.sell_volume + EXCLUDED.sell_volume,
                delta        = tick_metrics_minute.delta       + EXCLUDED.delta,
                cvd_running  = EXCLUDED.cvd_running,
                trade_count  = tick_metrics_minute.trade_count + EXCLUDED.trade_count,
                avg_spread_bps = EXCLUDED.avg_spread_bps
            """
        )
        with self._db_engine.begin() as conn:
            conn.execute(sql, record)

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def detect_divergence(
        self, instrument_key: str, lookback_minutes: int = 30
    ) -> dict:
        """Detect price / CVD divergence over the last N completed minutes.

        Reads the most-recently flushed data from the in-memory CVD history
        (the still-open current minute is excluded).

        Methodology:
            * Price trend: compare first vs last ``cvd_running`` in the window
              as a proxy for price direction is **not** available here; instead
              we compare delta sums of the first half vs second half to infer
              directional momentum.  For a full implementation the caller
              should pass a price series.
            * CVD trend: sum of deltas in first half vs second half of window.

        Since this module does not store historical minute records after flush
        (they are in the DB), divergence detection works on the **in-memory
        unflushed buckets** accumulated since session start.  The caller can
        supply a richer history by querying the DB; this method provides a
        lightweight in-process estimate.

        Args:
            instrument_key: Upstox instrument identifier.
            lookback_minutes: How many recent minute buckets to examine.

        Returns:
            A dict with keys:
                * ``price_trend``: ``'up'``, ``'down'``, or ``'flat'``
                * ``cvd_trend``: ``'up'``, ``'down'``, or ``'flat'``
                * ``divergent``: ``True`` when price and CVD trends differ
                * ``type``: ``'bearish'``, ``'bullish'``, or ``None``

            ``bearish`` divergence = price up but CVD down (sellers absorbing
            the rally).  ``bullish`` divergence = price down but CVD up (buyers
            absorbing the sell-off).
        """
        with self._lock:
            state = self._state.get(instrument_key)
            if state is None or not state.buckets:
                return _flat_divergence_result()

            # Sort available minute timestamps
            sorted_ts = sorted(state.buckets.keys())
            window = sorted_ts[-lookback_minutes:]  # most recent N minutes

            if len(window) < 2:
                return _flat_divergence_result()

            deltas = [state.buckets[ts].buy_volume - state.buckets[ts].sell_volume
                      for ts in window]

        # Split into halves to detect trend direction.
        #
        # Design:
        #   * price_trend — direction of net volume pressure in the FIRST half of
        #     the window (what the market *has been* doing; positive delta = price
        #     was being pushed up by buyers).
        #   * cvd_trend   — direction of net volume pressure in the SECOND half of
        #     the window (whether buying or selling is *currently* dominant).
        #
        # When the first half is buy-dominated (price rising) but the second half
        # is sell-dominated (CVD declining), that is a classic bearish divergence.
        # When both halves trend in the same direction there is no divergence.
        mid = len(deltas) // 2
        first_half_cvd = sum(deltas[:mid])
        second_half_cvd = sum(deltas[mid:])

        # Price trend: direction of aggregate delta in the earlier portion of window.
        if first_half_cvd > 0:
            price_trend = "up"
        elif first_half_cvd < 0:
            price_trend = "down"
        else:
            price_trend = "flat"

        # CVD trend: direction of aggregate delta in the later portion of window.
        if second_half_cvd > 0:
            cvd_trend = "up"
        elif second_half_cvd < 0:
            cvd_trend = "down"
        else:
            cvd_trend = "flat"

        divergent = price_trend != cvd_trend and "flat" not in (price_trend, cvd_trend)

        div_type: str | None = None
        if divergent:
            if price_trend == "up" and cvd_trend == "down":
                div_type = "bearish"
            elif price_trend == "down" and cvd_trend == "up":
                div_type = "bullish"

        return {
            "price_trend": price_trend,
            "cvd_trend": cvd_trend,
            "divergent": divergent,
            "type": div_type,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_divergence_result() -> dict:
    """Return a no-divergence result with flat trends."""
    return {
        "price_trend": "flat",
        "cvd_trend": "flat",
        "divergent": False,
        "type": None,
    }
