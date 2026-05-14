"""Kronos shadow-mode validator.

Pulls past forecasts from kronos_forecasts, joins with actual bar outcomes,
computes accuracy metrics, persists to kronos_accuracy_daily.

Typical usage::

    from src.research.kronos_validator import KronosValidator
    validator = KronosValidator(db_engine=get_sync_engine())
    metrics = validator.compute_daily(date.today())
    report = validator.generate_report(days=30)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

THRESHOLD_DROP = 45.0         # accuracy < 45% → DROP
THRESHOLD_KEEP = 55.0         # accuracy > 55% → KEEP
# 45 ≤ accuracy ≤ 55 → REDUCE_WEIGHT


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AccuracyMetrics:
    """Per-(instrument, horizon) accuracy summary for a single trade day.

    Attributes:
        instrument_key: Upstox instrument key (e.g. ``'NSE_INDEX|Nifty 50'``).
        horizon_bars: Number of bars the prediction was made for.
        prediction_count: Total number of (prediction, outcome) pairs evaluated.
        direction_correct: Count of rows where predicted_direction == actual_direction.
        direction_accuracy_pct: ``direction_correct / prediction_count * 100``.
        close_mae_pct: Mean absolute percentage error between predicted and actual close.
        range_mae_pct: MAE of (predicted high-low range pct) vs actual high-low range pct.
        above_baseline: ``True`` when ``direction_accuracy_pct > 50.0``.
    """

    instrument_key: str
    horizon_bars: int
    prediction_count: int
    direction_correct: int
    direction_accuracy_pct: float
    close_mae_pct: float
    range_mae_pct: float
    above_baseline: bool


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class KronosValidator:
    """Evaluate Kronos model predictions against actual bar outcomes.

    All DB operations are guarded — when ``db_engine`` is ``None`` the class
    operates in a no-persistence mode (useful for unit testing without a real
    database).

    Args:
        db_engine: Optional SQLAlchemy sync engine connected to a database
            that contains ``kronos_forecasts``, ``bars``, and
            ``kronos_accuracy_daily`` tables.
    """

    def __init__(self, db_engine=None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Primary data fetch
    # ------------------------------------------------------------------

    def fetch_predictions_with_outcomes(
        self,
        trade_date: date,
        instrument_key: Optional[str] = None,
        horizon_bars: Optional[int] = None,
    ) -> list[dict]:
        """For each forecast made on *trade_date*, pull the actual bar outcome.

        Joins ``kronos_forecasts`` with ``bars`` on:
        ``bars.ts = forecast.ts + (horizon_bars * timeframe_minutes_interval)``.

        Args:
            trade_date: The date (IST) on which forecasts were created.
            instrument_key: Optional filter to a single instrument.
            horizon_bars: Optional filter to a single horizon length.

        Returns:
            List of dicts, one per (forecast, outcome) pair, each with keys:
            ``instrument_key``, ``horizon_bars``, ``timeframe``,
            ``predicted_close_paisa``, ``predicted_high_paisa``,
            ``predicted_low_paisa``, ``predicted_direction``,
            ``predicted_change_pct``, ``actual_close``, ``actual_high``,
            ``actual_low``, ``actual_direction``, ``forecast_ts``.

            Returns an empty list when no engine is provided or no data found.
        """
        if self._engine is None:
            logger.debug("No DB engine — returning empty predictions list")
            return []

        from sqlalchemy import text

        filters = ["DATE(f.ts AT TIME ZONE 'Asia/Kolkata') = :trade_date"]
        params: dict[str, Any] = {"trade_date": trade_date.isoformat()}

        if instrument_key is not None:
            filters.append("f.instrument_key = :instrument_key")
            params["instrument_key"] = instrument_key

        if horizon_bars is not None:
            filters.append("f.horizon_bars = :horizon_bars")
            params["horizon_bars"] = horizon_bars

        where_clause = " AND ".join(filters)

        sql = text(
            f"""
            SELECT
                f.ts                    AS forecast_ts,
                f.instrument_key,
                f.horizon_bars,
                f.timeframe,
                f.predicted_close_paisa,
                f.predicted_high_paisa,
                f.predicted_low_paisa,
                f.predicted_direction,
                f.predicted_change_pct,
                b.close                 AS actual_close,
                b.high                  AS actual_high,
                b.low                   AS actual_low
            FROM kronos_forecasts f
            LEFT JOIN bars b
                ON  b.instrument_key = f.instrument_key
                AND b.timeframe      = f.timeframe
                AND b.ts = f.ts + (
                        f.horizon_bars
                        * CASE f.timeframe
                            WHEN '1m'  THEN INTERVAL '1 minute'
                            WHEN '3m'  THEN INTERVAL '3 minutes'
                            WHEN '5m'  THEN INTERVAL '5 minutes'
                            WHEN '15m' THEN INTERVAL '15 minutes'
                            WHEN '30m' THEN INTERVAL '30 minutes'
                            WHEN '1h'  THEN INTERVAL '1 hour'
                            WHEN '1d'  THEN INTERVAL '1 day'
                            ELSE            INTERVAL '5 minutes'
                          END
                    )
            WHERE {where_clause}
            ORDER BY f.ts, f.instrument_key, f.horizon_bars
            """
        )

        rows: list[dict] = []
        try:
            with self._engine.connect() as conn:
                result = conn.execute(sql, params)
                for row in result:
                    r = dict(row._mapping)
                    # Derive actual_direction from actual bars
                    actual_direction = self._derive_actual_direction(
                        forecast_ts=r.get("forecast_ts"),
                        actual_close=r.get("actual_close"),
                        predicted_close_paisa=r.get("predicted_close_paisa"),
                        instrument_key=r.get("instrument_key"),
                        timeframe=r.get("timeframe"),
                    )
                    r["actual_direction"] = actual_direction
                    rows.append(r)
        except Exception:
            logger.exception(
                "Failed to fetch predictions with outcomes for trade_date=%s", trade_date
            )

        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_actual_direction(
        self,
        forecast_ts,
        actual_close,
        predicted_close_paisa,
        instrument_key: Optional[str],
        timeframe: Optional[str],
    ) -> str:
        """Derive actual price direction from the bar that preceded the forecast.

        When ``actual_close`` is ``None`` (bar not found), returns ``'FLAT'``.

        Returns:
            ``'UP'``, ``'DOWN'``, or ``'FLAT'``.
        """
        if actual_close is None or forecast_ts is None:
            return "FLAT"

        if self._engine is None:
            return "FLAT"

        from sqlalchemy import text

        sql = text(
            """
            SELECT close FROM bars
            WHERE instrument_key = :instrument_key
              AND timeframe = :timeframe
              AND ts <= :forecast_ts
            ORDER BY ts DESC
            LIMIT 1
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql,
                    {
                        "instrument_key": instrument_key,
                        "timeframe": timeframe,
                        "forecast_ts": forecast_ts,
                    },
                ).fetchone()
        except Exception:
            logger.debug("Could not fetch reference bar for direction derivation")
            return "FLAT"

        if row is None or row[0] is None:
            return "FLAT"

        ref_close = float(row[0])
        actual = float(actual_close)
        diff_pct = (actual - ref_close) / ref_close * 100.0 if ref_close != 0 else 0.0

        if diff_pct > 0.1:
            return "UP"
        if diff_pct < -0.1:
            return "DOWN"
        return "FLAT"

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics_for_instrument(
        self,
        rows: list[dict],
        instrument_key: str,
        horizon_bars: int,
    ) -> AccuracyMetrics:
        """Aggregate accuracy metrics from a list of (prediction, outcome) rows.

        Args:
            rows: List of dicts as returned by
                :meth:`fetch_predictions_with_outcomes`, already filtered to a
                single ``(instrument_key, horizon_bars)`` combination.
            instrument_key: The instrument being evaluated.
            horizon_bars: The horizon being evaluated.

        Returns:
            :class:`AccuracyMetrics` for the group.
        """
        if not rows:
            return AccuracyMetrics(
                instrument_key=instrument_key,
                horizon_bars=horizon_bars,
                prediction_count=0,
                direction_correct=0,
                direction_accuracy_pct=0.0,
                close_mae_pct=0.0,
                range_mae_pct=0.0,
                above_baseline=False,
            )

        direction_correct = 0
        close_abs_errors: list[float] = []
        range_abs_errors: list[float] = []

        for row in rows:
            predicted_dir = (row.get("predicted_direction") or "FLAT").upper()
            actual_dir = (row.get("actual_direction") or "FLAT").upper()

            if predicted_dir == actual_dir:
                direction_correct += 1

            actual_close = row.get("actual_close")
            predicted_close_paisa = row.get("predicted_close_paisa")

            if actual_close is not None and predicted_close_paisa is not None:
                actual_f = float(actual_close)
                predicted_f = float(predicted_close_paisa)
                if actual_f != 0:
                    close_abs_errors.append(
                        abs(predicted_f - actual_f) / abs(actual_f) * 100.0
                    )

            actual_high = row.get("actual_high")
            actual_low = row.get("actual_low")
            predicted_high = row.get("predicted_high_paisa")
            predicted_low = row.get("predicted_low_paisa")

            if (
                actual_high is not None
                and actual_low is not None
                and predicted_high is not None
                and predicted_low is not None
            ):
                act_h = float(actual_high)
                act_l = float(actual_low)
                pred_h = float(predicted_high)
                pred_l = float(predicted_low)

                mid = (act_h + act_l) / 2.0 if (act_h + act_l) != 0 else 1.0
                actual_range_pct = (act_h - act_l) / mid * 100.0 if mid != 0 else 0.0
                predicted_range_pct = (
                    (pred_h - pred_l) / mid * 100.0 if mid != 0 else 0.0
                )
                range_abs_errors.append(abs(predicted_range_pct - actual_range_pct))

        n = len(rows)
        dir_acc = direction_correct / n * 100.0 if n > 0 else 0.0
        close_mae = (
            sum(close_abs_errors) / len(close_abs_errors)
            if close_abs_errors
            else 0.0
        )
        range_mae = (
            sum(range_abs_errors) / len(range_abs_errors)
            if range_abs_errors
            else 0.0
        )

        return AccuracyMetrics(
            instrument_key=instrument_key,
            horizon_bars=horizon_bars,
            prediction_count=n,
            direction_correct=direction_correct,
            direction_accuracy_pct=dir_acc,
            close_mae_pct=close_mae,
            range_mae_pct=range_mae,
            above_baseline=dir_acc > 50.0,
        )

    # ------------------------------------------------------------------
    # Daily computation + persistence
    # ------------------------------------------------------------------

    def compute_daily(self, trade_date: date) -> list[AccuracyMetrics]:
        """Compute accuracy for every (instrument_key, horizon_bars) on *trade_date*.

        Fetches all forecasts for the day, groups by
        ``(instrument_key, horizon_bars)``, computes
        :class:`AccuracyMetrics` for each, and persists one row per combo
        to ``kronos_accuracy_daily``.

        Args:
            trade_date: The trading date to evaluate.

        Returns:
            List of :class:`AccuracyMetrics`, one per unique combo found.
            Returns an empty list when no forecasts exist for the date.
        """
        all_rows = self.fetch_predictions_with_outcomes(trade_date)
        if not all_rows:
            logger.info("No Kronos forecasts found for %s", trade_date)
            return []

        # Group by (instrument_key, horizon_bars)
        groups: dict[tuple[str, int], list[dict]] = {}
        for row in all_rows:
            key = (
                str(row.get("instrument_key", "")),
                int(row.get("horizon_bars", 0)),
            )
            groups.setdefault(key, []).append(row)

        results: list[AccuracyMetrics] = []
        for (instr, horizon), rows in groups.items():
            m = self.compute_metrics_for_instrument(rows, instr, horizon)
            self._persist_daily_accuracy(trade_date, m)
            results.append(m)

        logger.info(
            "compute_daily(%s): evaluated %d (instrument, horizon) combos",
            trade_date,
            len(results),
        )
        return results

    def _persist_daily_accuracy(self, trade_date: date, m: AccuracyMetrics) -> None:
        """Upsert one row into ``kronos_accuracy_daily``.

        Args:
            trade_date: The trading date the metrics belong to.
            m: Computed accuracy metrics.
        """
        if self._engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO kronos_accuracy_daily (
                trade_date,
                instrument_key,
                horizon_bars,
                prediction_count,
                direction_correct,
                direction_accuracy_pct,
                close_mae_pct,
                range_mae_pct,
                above_baseline,
                computed_at
            ) VALUES (
                :trade_date,
                :instrument_key,
                :horizon_bars,
                :prediction_count,
                :direction_correct,
                :direction_accuracy_pct,
                :close_mae_pct,
                :range_mae_pct,
                :above_baseline,
                NOW()
            )
            ON CONFLICT (trade_date, instrument_key, horizon_bars)
            DO UPDATE SET
                prediction_count       = EXCLUDED.prediction_count,
                direction_correct      = EXCLUDED.direction_correct,
                direction_accuracy_pct = EXCLUDED.direction_accuracy_pct,
                close_mae_pct          = EXCLUDED.close_mae_pct,
                range_mae_pct          = EXCLUDED.range_mae_pct,
                above_baseline         = EXCLUDED.above_baseline,
                computed_at            = EXCLUDED.computed_at
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "trade_date": trade_date.isoformat(),
                        "instrument_key": m.instrument_key,
                        "horizon_bars": m.horizon_bars,
                        "prediction_count": m.prediction_count,
                        "direction_correct": m.direction_correct,
                        "direction_accuracy_pct": m.direction_accuracy_pct,
                        "close_mae_pct": m.close_mae_pct,
                        "range_mae_pct": m.range_mae_pct,
                        "above_baseline": m.above_baseline,
                    },
                )
        except Exception:
            logger.exception(
                "Failed to persist accuracy for instrument=%s horizon=%s date=%s",
                m.instrument_key,
                m.horizon_bars,
                trade_date,
            )

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    def get_accuracy_summary(
        self,
        days: int = 30,
        instrument_key: Optional[str] = None,
    ) -> dict:
        """Aggregate accuracy metrics over a rolling window.

        Reads from ``kronos_accuracy_daily`` for the past *days* calendar days.
        When no DB engine is set or no data exists, returns a neutral empty
        summary without error.

        Args:
            days: Number of calendar days to look back.
            instrument_key: Optional filter to a single instrument.

        Returns:
            Dict with keys:

            * ``'total_predictions'`` – int
            * ``'overall_accuracy_pct'`` – float (0.0 when no data)
            * ``'by_instrument'`` – ``{sym: {accuracy_pct, count, mae_pct}}``
            * ``'by_horizon'`` – ``{horizon: {accuracy_pct, count, mae_pct}}``
            * ``'best_instrument'`` – str or ``''``
            * ``'worst_instrument'`` – str or ``''``
            * ``'recommendation'`` – ``'KEEP'`` / ``'REDUCE_WEIGHT'`` / ``'DROP'``
        """
        empty: dict = {
            "total_predictions": 0,
            "overall_accuracy_pct": 0.0,
            "by_instrument": {},
            "by_horizon": {},
            "best_instrument": "",
            "worst_instrument": "",
            "recommendation": "REDUCE_WEIGHT",
        }

        if self._engine is None:
            logger.debug("No DB engine — returning empty accuracy summary")
            return empty

        from sqlalchemy import text

        cutoff = date.today() - timedelta(days=days)

        filters = ["trade_date >= :cutoff"]
        params: dict[str, Any] = {"cutoff": cutoff.isoformat()}

        if instrument_key is not None:
            filters.append("instrument_key = :instrument_key")
            params["instrument_key"] = instrument_key

        where_clause = " AND ".join(filters)

        sql = text(
            f"""
            SELECT
                instrument_key,
                horizon_bars,
                SUM(prediction_count)       AS total_count,
                SUM(direction_correct)      AS total_correct,
                AVG(close_mae_pct)          AS avg_close_mae,
                AVG(range_mae_pct)          AS avg_range_mae
            FROM kronos_accuracy_daily
            WHERE {where_clause}
            GROUP BY instrument_key, horizon_bars
            ORDER BY instrument_key, horizon_bars
            """
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception:
            logger.exception("Failed to query kronos_accuracy_daily for summary")
            return empty

        if not rows:
            return empty

        total_preds = 0
        total_correct = 0
        by_instrument: dict[str, dict] = {}
        by_horizon: dict[int, dict] = {}

        for row in rows:
            instr = str(row[0])
            horizon = int(row[1])
            count = int(row[2]) if row[2] is not None else 0
            correct = int(row[3]) if row[3] is not None else 0
            mae = float(row[4]) if row[4] is not None else 0.0

            total_preds += count
            total_correct += correct

            # Aggregate by instrument
            if instr not in by_instrument:
                by_instrument[instr] = {"total_count": 0, "total_correct": 0, "mae_sum": 0.0, "mae_n": 0}
            by_instrument[instr]["total_count"] += count
            by_instrument[instr]["total_correct"] += correct
            by_instrument[instr]["mae_sum"] += mae
            by_instrument[instr]["mae_n"] += 1

            # Aggregate by horizon
            if horizon not in by_horizon:
                by_horizon[horizon] = {"total_count": 0, "total_correct": 0, "mae_sum": 0.0, "mae_n": 0}
            by_horizon[horizon]["total_count"] += count
            by_horizon[horizon]["total_correct"] += correct
            by_horizon[horizon]["mae_sum"] += mae
            by_horizon[horizon]["mae_n"] += 1

        overall_acc = (total_correct / total_preds * 100.0) if total_preds > 0 else 0.0

        # Build by_instrument output
        instr_summary: dict[str, dict] = {}
        for sym, d in by_instrument.items():
            acc = (d["total_correct"] / d["total_count"] * 100.0) if d["total_count"] > 0 else 0.0
            mae_avg = d["mae_sum"] / d["mae_n"] if d["mae_n"] > 0 else 0.0
            instr_summary[sym] = {
                "accuracy_pct": round(acc, 2),
                "count": d["total_count"],
                "mae_pct": round(mae_avg, 4),
            }

        # Build by_horizon output
        horizon_summary: dict[int, dict] = {}
        for h, d in by_horizon.items():
            acc = (d["total_correct"] / d["total_count"] * 100.0) if d["total_count"] > 0 else 0.0
            mae_avg = d["mae_sum"] / d["mae_n"] if d["mae_n"] > 0 else 0.0
            horizon_summary[h] = {
                "accuracy_pct": round(acc, 2),
                "count": d["total_count"],
                "mae_pct": round(mae_avg, 4),
            }

        # Best / worst instrument
        best = ""
        worst = ""
        if instr_summary:
            best = max(instr_summary, key=lambda s: instr_summary[s]["accuracy_pct"])
            worst = min(instr_summary, key=lambda s: instr_summary[s]["accuracy_pct"])

        recommendation = self._recommendation(overall_acc)

        return {
            "total_predictions": total_preds,
            "overall_accuracy_pct": round(overall_acc, 2),
            "by_instrument": instr_summary,
            "by_horizon": horizon_summary,
            "best_instrument": best,
            "worst_instrument": worst,
            "recommendation": recommendation,
        }

    @staticmethod
    def _recommendation(accuracy_pct: float) -> str:
        """Map accuracy percentage to a recommendation string.

        Args:
            accuracy_pct: Overall directional accuracy (0–100).

        Returns:
            ``'KEEP'`` if accuracy > 55%, ``'DROP'`` if < 45%,
            ``'REDUCE_WEIGHT'`` otherwise.
        """
        if accuracy_pct > THRESHOLD_KEEP:
            return "KEEP"
        if accuracy_pct < THRESHOLD_DROP:
            return "DROP"
        return "REDUCE_WEIGHT"

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, days: int = 30) -> str:
        """Generate a human-readable Markdown report for *days* rolling window.

        Args:
            days: Number of calendar days to include in the summary.

        Returns:
            Multi-line Markdown string. Suitable for logging or posting to docs.
        """
        summary = self.get_accuracy_summary(days=days)
        now_ist = datetime.now(tz=IST)
        as_of = now_ist.strftime("%Y-%m-%d %H:%M IST")

        lines: list[str] = [
            f"# Kronos Prediction Accuracy Report",
            f"",
            f"**Generated:** {as_of}  ",
            f"**Window:** last {days} days  ",
            f"**Recommendation:** `{summary['recommendation']}`",
            f"",
        ]

        total_preds = summary["total_predictions"]

        if total_preds == 0:
            lines += [
                "## Summary",
                "",
                "_No predictions found for this period._",
                "",
                "Kronos forecasts are stored in `kronos_forecasts`. Run the model",
                "in shadow mode for at least one trading session before generating",
                "a meaningful accuracy report.",
                "",
            ]
            return "\n".join(lines)

        overall_acc = summary["overall_accuracy_pct"]
        lines += [
            "## Overall",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total predictions | {total_preds:,} |",
            f"| Directional accuracy | **{overall_acc:.1f}%** |",
            f"| Best instrument | {summary['best_instrument'] or '—'} |",
            f"| Worst instrument | {summary['worst_instrument'] or '—'} |",
            f"",
        ]

        # Per-instrument breakdown
        by_instr = summary.get("by_instrument", {})
        if by_instr:
            lines += [
                "## Per-Instrument Breakdown",
                "",
                "| Instrument | Predictions | Accuracy % | Close MAE % |",
                "|---|---|---|---|",
            ]
            for sym, d in sorted(by_instr.items()):
                acc_str = f"{d['accuracy_pct']:.1f}%"
                mae_str = f"{d['mae_pct']:.3f}%"
                lines.append(
                    f"| {sym} | {d['count']:,} | {acc_str} | {mae_str} |"
                )
            lines.append("")

        # Per-horizon breakdown
        by_horizon = summary.get("by_horizon", {})
        if by_horizon:
            lines += [
                "## Per-Horizon Breakdown",
                "",
                "| Horizon (bars) | Predictions | Accuracy % | Close MAE % |",
                "|---|---|---|---|",
            ]
            for h in sorted(by_horizon.keys()):
                d = by_horizon[h]
                acc_str = f"{d['accuracy_pct']:.1f}%"
                mae_str = f"{d['mae_pct']:.3f}%"
                lines.append(
                    f"| {h} | {d['count']:,} | {acc_str} | {mae_str} |"
                )
            lines.append("")

        # Decision gate guidance
        lines += [
            "## G.5 Decision Gate Guidance",
            "",
            f"After 4-week shadow mode:",
            f"",
            f"- Hit rate **> 55%** AND useful for ≥ 3 strategies → integrate at weight 0.10+  ",
            f"- Hit rate **50–55%** → keep at low weight as tiebreaker  ",
            f"- Hit rate **< 50%** → drop entirely  ",
            f"",
            f"_Current recommendation: `{summary['recommendation']}`_",
            f"",
        ]

        return "\n".join(lines)
