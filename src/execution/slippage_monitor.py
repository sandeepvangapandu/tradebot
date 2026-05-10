"""Slippage tracking + paper-vs-live drift detection.

Records every fill's real price vs intended price, computes slippage in paisa
and basis-points, persists to ``slippage_log``, and detects when live-mode
slippage drifts significantly worse than paper-mode slippage.

All monetary values are in **paisa** (1 INR = 100 paisa).
All times are in IST (Asia/Kolkata).

Wiring agent is responsible for calling:
  - ``record_fill()`` after every fill comes back from the broker.
  - ``detect_drift()`` periodically (e.g., every 30 fills or every hour).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# IST offset
IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlippageConfig:
    """Runtime configuration for SlippageMonitor.

    Attributes:
        drift_threshold_pct: Alert if live avg slippage is >N% worse than
            paper avg slippage.  Default 50% worse → DRIFT_HIGH alert.
        halt_threshold_pct: Hard halt (DRIFT_BLOCK) if live avg slippage is
            >N% worse than paper avg.  Default 100% worse (2x paper).
        drift_min_sample_size: Minimum fills required in each mode before
            drift evaluation is attempted.  Default 30.
        rolling_window_days: Look-back window used for averaging slippage.
            Default 7 calendar days.
    """

    drift_threshold_pct: float = 50.0
    halt_threshold_pct: float = 100.0
    drift_min_sample_size: int = 30
    rolling_window_days: int = 7


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------


class SlippageMonitor:
    """Track real fill price vs intended price and detect paper/live drift.

    Args:
        db_engine: A SQLAlchemy sync engine pointing at the tradebot database.
            When ``None`` (unit tests), DB calls are skipped or must be mocked.
        config: Optional ``SlippageConfig``; defaults are safe.
    """

    def __init__(
        self,
        db_engine: Any = None,
        config: SlippageConfig | None = None,
    ) -> None:
        self._engine = db_engine
        self._cfg = config or SlippageConfig()

    # ------------------------------------------------------------------
    # Core computation (no DB)
    # ------------------------------------------------------------------

    def compute_slippage(
        self,
        intended_price: int,
        actual_price: int,
        side: str,
        quantity: int,
    ) -> dict:
        """Compute slippage metrics for a single fill.

        Sign convention:
            - BUY: positive slippage = paid *more* than intended (bad).
            - SELL: positive slippage = received *less* than intended (bad).

        Args:
            intended_price: Pre-trade expected fill price in paisa.
            actual_price: Actual broker fill price in paisa.
            side: ``'BUY'`` or ``'SELL'`` (case-insensitive).
            quantity: Number of units filled.

        Returns:
            Dict with keys:
                slippage_paisa (int),
                slippage_bps (float),
                slippage_cost_paisa (int).
        """
        side_upper = side.upper()
        if side_upper == "BUY":
            raw_slip = actual_price - intended_price
        elif side_upper == "SELL":
            raw_slip = intended_price - actual_price
        else:
            raise ValueError(f"side must be BUY or SELL, got {side!r}")

        slippage_paisa = int(raw_slip)

        # Basis-points: slippage / intended_price * 10000
        if intended_price != 0:
            slippage_bps = round(slippage_paisa / intended_price * 10_000, 4)
        else:
            slippage_bps = 0.0

        slippage_cost_paisa = slippage_paisa * quantity

        return {
            "slippage_paisa": slippage_paisa,
            "slippage_bps": slippage_bps,
            "slippage_cost_paisa": slippage_cost_paisa,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def record_fill(
        self,
        fill_id: str,
        order_id: str,
        instrument_key: str,
        side: str,
        intended_price: int,
        actual_fill_price: int,
        quantity: int,
        mode: str,
    ) -> int:
        """Compute slippage and persist one fill record to ``slippage_log``.

        Args:
            fill_id: Broker fill identifier (used as FK to fills table).
            order_id: Broker / client order identifier.
            instrument_key: Upstox instrument key, e.g. ``NSE_EQ|RELIANCE``.
            side: ``'BUY'`` or ``'SELL'``.
            intended_price: Pre-trade intended price in paisa.
            actual_fill_price: Actual fill price in paisa.
            quantity: Filled quantity.
            mode: Execution mode — ``'paper'`` or ``'live'``.

        Returns:
            Database row ``id`` of the inserted record.

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for record_fill()")

        metrics = self.compute_slippage(
            intended_price, actual_fill_price, side, quantity
        )
        now_ist = datetime.now(IST)

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            INSERT INTO slippage_log
                (fill_id, order_id, instrument_key, side,
                 intended_price, actual_fill_price, quantity,
                 slippage_paisa, slippage_bps, slippage_cost_paisa,
                 mode, recorded_at)
            VALUES
                (:fill_id, :order_id, :instrument_key, :side,
                 :intended_price, :actual_fill_price, :quantity,
                 :slippage_paisa, :slippage_bps, :slippage_cost_paisa,
                 :mode, :recorded_at)
            RETURNING id
            """
        )
        params = {
            "fill_id": fill_id,
            "order_id": order_id,
            "instrument_key": instrument_key,
            "side": side.upper(),
            "intended_price": intended_price,
            "actual_fill_price": actual_fill_price,
            "quantity": quantity,
            "slippage_paisa": metrics["slippage_paisa"],
            "slippage_bps": metrics["slippage_bps"],
            "slippage_cost_paisa": metrics["slippage_cost_paisa"],
            "mode": mode.lower(),
            "recorded_at": now_ist,
        }

        with self._engine.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            conn.commit()

        row_id = row[0]
        logger.info(
            "Slippage recorded id=%s instrument=%s side=%s slip_bps=%.2f mode=%s",
            row_id,
            instrument_key,
            side,
            metrics["slippage_bps"],
            mode,
        )
        return row_id

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def get_avg_slippage_bps(
        self,
        instrument_key: str | None = None,
        mode: str | None = None,
        days: int = 7,
    ) -> float:
        """Return the average slippage in basis-points over a rolling window.

        Args:
            instrument_key: Optionally filter to a single instrument.
            mode: Optionally filter to ``'paper'`` or ``'live'``.
            days: Look-back window in calendar days.

        Returns:
            Average slippage in bps (0.0 if no records found).

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for get_avg_slippage_bps()")

        from sqlalchemy import text as sa_text

        cutoff = datetime.now(IST) - timedelta(days=days)

        conditions = ["recorded_at >= :cutoff"]
        params: dict[str, Any] = {"cutoff": cutoff}

        if instrument_key is not None:
            conditions.append("instrument_key = :instrument_key")
            params["instrument_key"] = instrument_key
        if mode is not None:
            conditions.append("mode = :mode")
            params["mode"] = mode.lower()

        where = " AND ".join(conditions)
        sql = sa_text(
            f"SELECT AVG(slippage_bps) FROM slippage_log WHERE {where}"
        )

        with self._engine.connect() as conn:
            result = conn.execute(sql, params).scalar()

        return float(result) if result is not None else 0.0

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self, instrument_key: str | None = None
    ) -> dict | None:
        """Compare paper and live average slippage over the rolling window.

        Returns a drift report if either mode has enough samples, or ``None``
        when the sample size is insufficient for a meaningful comparison.

        Args:
            instrument_key: Optionally scope the comparison to one instrument.

        Returns:
            Dict with keys:
                paper_avg_bps, live_avg_bps, drift_pct,
                alert_type ('DRIFT_HIGH' | 'DRIFT_BLOCK' | None),
                recommendation (str)
            or ``None`` when insufficient data.

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for detect_drift()")

        from sqlalchemy import text as sa_text

        cutoff = datetime.now(IST) - timedelta(days=self._cfg.rolling_window_days)
        params: dict[str, Any] = {"cutoff": cutoff}
        ikey_filter = ""
        if instrument_key is not None:
            ikey_filter = " AND instrument_key = :instrument_key"
            params["instrument_key"] = instrument_key

        count_sql = sa_text(
            f"""
            SELECT mode, COUNT(*) AS cnt, AVG(slippage_bps) AS avg_bps
            FROM slippage_log
            WHERE recorded_at >= :cutoff{ikey_filter}
            GROUP BY mode
            """
        )

        with self._engine.connect() as conn:
            rows = conn.execute(count_sql, params).fetchall()

        stats: dict[str, dict] = {}
        for row in rows:
            stats[row[0]] = {"count": row[1], "avg_bps": float(row[2] or 0.0)}

        paper_stats = stats.get("paper", {"count": 0, "avg_bps": 0.0})
        live_stats = stats.get("live", {"count": 0, "avg_bps": 0.0})

        min_n = self._cfg.drift_min_sample_size
        if paper_stats["count"] < min_n or live_stats["count"] < min_n:
            logger.debug(
                "Insufficient samples for drift detection: paper=%d live=%d (need %d each)",
                paper_stats["count"],
                live_stats["count"],
                min_n,
            )
            return None

        paper_avg = paper_stats["avg_bps"]
        live_avg = live_stats["avg_bps"]

        # drift_pct: how much worse is live vs paper?
        if paper_avg != 0:
            drift_pct = (live_avg - paper_avg) / abs(paper_avg) * 100.0
        else:
            # paper has zero slippage — any live slippage is infinite drift;
            # treat as 0 drift to avoid false positives when paper is perfect.
            drift_pct = 0.0 if live_avg == 0 else float("inf")

        if drift_pct > self._cfg.halt_threshold_pct:
            alert_type = "DRIFT_BLOCK"
            recommendation = (
                "Halt live order submission immediately. "
                f"Live slippage ({live_avg:.2f} bps) is {drift_pct:.1f}% "
                f"worse than paper ({paper_avg:.2f} bps). "
                "Investigate broker routing or market conditions before resuming."
            )
        elif drift_pct > self._cfg.drift_threshold_pct:
            alert_type = "DRIFT_HIGH"
            recommendation = (
                f"Live slippage ({live_avg:.2f} bps) is {drift_pct:.1f}% "
                f"worse than paper ({paper_avg:.2f} bps). "
                "Review smart-router configuration and market depth before next session."
            )
        else:
            alert_type = None
            recommendation = "Slippage within acceptable range."

        return {
            "paper_avg_bps": paper_avg,
            "live_avg_bps": live_avg,
            "drift_pct": drift_pct,
            "alert_type": alert_type,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Alert persistence
    # ------------------------------------------------------------------

    def log_drift_alert(self, drift: dict) -> int:
        """Persist a drift detection result to ``slippage_drift_alerts``.

        Args:
            drift: Dict returned by :meth:`detect_drift` (must have
                ``alert_type``, ``paper_avg_bps``, ``live_avg_bps``,
                ``drift_pct``).

        Returns:
            Database row ``id`` of the inserted alert.

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for log_drift_alert()")

        from sqlalchemy import text as sa_text

        now_ist = datetime.now(IST)
        alert_type = drift.get("alert_type") or "DRIFT_HIGH"
        threshold_pct = (
            self._cfg.halt_threshold_pct
            if alert_type == "DRIFT_BLOCK"
            else self._cfg.drift_threshold_pct
        )
        details = {
            "recommendation": drift.get("recommendation", ""),
        }

        sql = sa_text(
            """
            INSERT INTO slippage_drift_alerts
                (ts, instrument_key, alert_type,
                 paper_avg_bps, live_avg_bps, drift_pct,
                 threshold_pct, details)
            VALUES
                (:ts, :instrument_key, :alert_type,
                 :paper_avg_bps, :live_avg_bps, :drift_pct,
                 :threshold_pct, :details)
            RETURNING id
            """
        )
        params = {
            "ts": now_ist,
            "instrument_key": drift.get("instrument_key"),
            "alert_type": alert_type,
            "paper_avg_bps": drift.get("paper_avg_bps"),
            "live_avg_bps": drift.get("live_avg_bps"),
            "drift_pct": drift.get("drift_pct"),
            "threshold_pct": threshold_pct,
            "details": json.dumps(details),
        }

        with self._engine.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            conn.commit()

        row_id = row[0]
        logger.warning(
            "Drift alert logged id=%s type=%s drift_pct=%.1f%%",
            row_id,
            alert_type,
            drift.get("drift_pct", 0),
        )
        return row_id

    def get_drift_alerts(self, days: int = 7) -> list[dict]:
        """Return recent drift alerts from ``slippage_drift_alerts``.

        Args:
            days: Look-back window in calendar days.

        Returns:
            List of dicts with keys: id, ts, instrument_key, alert_type,
            paper_avg_bps, live_avg_bps, drift_pct, threshold_pct, details.

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for get_drift_alerts()")

        from sqlalchemy import text as sa_text

        cutoff = datetime.now(IST) - timedelta(days=days)
        sql = sa_text(
            """
            SELECT id, ts, instrument_key, alert_type,
                   paper_avg_bps, live_avg_bps, drift_pct,
                   threshold_pct, details
            FROM slippage_drift_alerts
            WHERE ts >= :cutoff
            ORDER BY ts DESC
            """
        )

        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()

        return [
            {
                "id": r[0],
                "ts": r[1],
                "instrument_key": r[2],
                "alert_type": r[3],
                "paper_avg_bps": float(r[4]) if r[4] is not None else None,
                "live_avg_bps": float(r[5]) if r[5] is not None else None,
                "drift_pct": float(r[6]) if r[6] is not None else None,
                "threshold_pct": float(r[7]) if r[7] is not None else None,
                "details": r[8],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self, days: int = 7) -> dict:
        """Return an aggregated slippage summary over a rolling window.

        Args:
            days: Look-back window in calendar days.

        Returns:
            Dict with keys:
                total_fills (int),
                total_slippage_paisa (int),
                avg_bps_by_mode (dict[str, float]),
                by_instrument (list[dict]).

        Raises:
            RuntimeError: If no ``db_engine`` is configured.
        """
        if self._engine is None:
            raise RuntimeError("db_engine is required for get_summary()")

        from sqlalchemy import text as sa_text

        cutoff = datetime.now(IST) - timedelta(days=days)

        overall_sql = sa_text(
            """
            SELECT COUNT(*), COALESCE(SUM(slippage_cost_paisa), 0)
            FROM slippage_log
            WHERE recorded_at >= :cutoff
            """
        )
        mode_sql = sa_text(
            """
            SELECT mode, AVG(slippage_bps)
            FROM slippage_log
            WHERE recorded_at >= :cutoff
            GROUP BY mode
            """
        )
        instrument_sql = sa_text(
            """
            SELECT instrument_key, mode,
                   COUNT(*) AS fills,
                   AVG(slippage_bps) AS avg_bps,
                   SUM(slippage_cost_paisa) AS total_cost_paisa
            FROM slippage_log
            WHERE recorded_at >= :cutoff
            GROUP BY instrument_key, mode
            ORDER BY instrument_key, mode
            """
        )

        params = {"cutoff": cutoff}
        with self._engine.connect() as conn:
            total_row = conn.execute(overall_sql, params).fetchone()
            mode_rows = conn.execute(mode_sql, params).fetchall()
            instr_rows = conn.execute(instrument_sql, params).fetchall()

        total_fills = int(total_row[0]) if total_row else 0
        total_cost = int(total_row[1]) if total_row else 0

        avg_bps_by_mode = {
            row[0]: float(row[1]) if row[1] is not None else 0.0
            for row in mode_rows
        }

        by_instrument = [
            {
                "instrument_key": r[0],
                "mode": r[1],
                "fills": int(r[2]),
                "avg_bps": float(r[3]) if r[3] is not None else 0.0,
                "total_cost_paisa": int(r[4]) if r[4] is not None else 0,
            }
            for r in instr_rows
        ]

        return {
            "total_fills": total_fills,
            "total_slippage_paisa": total_cost,
            "avg_bps_by_mode": avg_bps_by_mode,
            "by_instrument": by_instrument,
        }
