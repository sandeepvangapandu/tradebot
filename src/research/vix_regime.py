"""India VIX regime classifier.

Classifies India VIX into LOW / NORMAL / HIGH / SPIKE regimes from real-time
data and historical context.  Persists daily summaries to ``vix_regime_daily``
and intraday snapshots to ``vix_regime_intraday``.

Exposes regime as a global filter that gates which strategies are evaluated
each session.

Typical usage::

    from src.research.vix_regime import VIXRegimeClassifier
    clf = VIXRegimeClassifier(db_engine=get_sync_engine())
    regime = clf.get_today_regime(date.today())
    if clf.is_strategy_allowed("trend", regime=regime["regime"]):
        ...
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIX_INSTRUMENT_KEY = "NSE_INDEX|India VIX"

# Static thresholds — could become adaptive / DB-driven in a future phase
REGIME_THRESHOLDS: dict[str, tuple[float, float]] = {
    "LOW":    (0.0,   13.0),
    "NORMAL": (13.0,  18.0),
    "HIGH":   (18.0,  25.0),
    "SPIKE":  (25.0, 999.0),
}

# Which regimes are OK for each broad strategy type.
# Key: strategy_type string; Value: set of allowed regimes
_STRATEGY_REGIME_MAP: dict[str, set[str]] = {
    "trend":        {"NORMAL", "HIGH"},
    "mean_rev":     {"HIGH", "SPIKE"},
    "breakout":     {"NORMAL"},
    "options_sell": {"HIGH"},
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class VIXRegimeClassifier:
    """Classify India VIX and persist regime snapshots.

    Args:
        historical_provider: Optional callable ``(lookback_days: int) ->
            pd.DataFrame`` with columns ``[date, vix]``.  When ``None``,
            :meth:`compute_percentile` falls back to an in-memory cache
            built from :meth:`update_intraday` calls.
        db_engine: Optional SQLAlchemy sync engine.  When ``None``, all
            persistence methods are no-ops (useful for unit testing).
    """

    def __init__(
        self,
        historical_provider=None,
        db_engine=None,
    ) -> None:
        self._historical_provider = historical_provider
        self._engine = db_engine

        # In-memory intraday buffer: list of (ts, vix_value)
        self._intraday_buffer: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify_value(self, vix: float) -> str:
        """Classify a VIX value into a named regime.

        Args:
            vix: India VIX level (e.g. 14.5).

        Returns:
            One of ``'LOW'``, ``'NORMAL'``, ``'HIGH'``, ``'SPIKE'``.
        """
        for regime, (lo, hi) in REGIME_THRESHOLDS.items():
            if lo <= vix < hi:
                return regime
        # Edge case: vix == 999 or larger
        return "SPIKE"

    # ------------------------------------------------------------------
    # Percentile
    # ------------------------------------------------------------------

    def compute_percentile(self, vix: float, lookback_days: int = 60) -> float:
        """Return where *vix* sits in the past *lookback_days* distribution.

        Uses the historical provider if available; otherwise falls back to the
        in-memory intraday buffer (useful in unit tests).

        Args:
            vix: Current VIX value to rank.
            lookback_days: Number of trading days to include.

        Returns:
            Percentile in ``[0.0, 100.0]``.  Returns ``50.0`` when no history
            is available (neutral assumption).
        """
        series: np.ndarray | None = None

        if self._historical_provider is not None:
            try:
                df: pd.DataFrame = self._historical_provider(lookback_days)
                if not df.empty and "vix" in df.columns:
                    series = df["vix"].dropna().to_numpy()
            except Exception:  # noqa: BLE001
                logger.warning("VIX historical_provider failed; using buffer fallback")

        if series is None or len(series) == 0:
            if self._intraday_buffer:
                series = np.array([v for _, v in self._intraday_buffer])
            else:
                return 50.0

        if len(series) == 0:
            return 50.0

        below = np.sum(series < vix)
        return float(below / len(series) * 100.0)

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def detect_spike(
        self,
        current: float,
        prev_close: float,
        threshold_pct: float = 30.0,
    ) -> bool:
        """Return ``True`` when the intraday move exceeds *threshold_pct*.

        Args:
            current: Current VIX reading.
            prev_close: Previous session's VIX close.
            threshold_pct: Percentage move threshold (default 30 %).

        Returns:
            ``True`` when ``abs(current - prev_close) / prev_close * 100 >= threshold_pct``.
        """
        if prev_close == 0:
            return False
        pct_move = abs(current - prev_close) / abs(prev_close) * 100.0
        return pct_move >= threshold_pct

    # ------------------------------------------------------------------
    # Intraday persistence
    # ------------------------------------------------------------------

    def update_intraday(
        self,
        ts: datetime,
        vix_value: float,
        prev_close: float | None = None,
    ) -> None:
        """Persist a VIX tick to ``vix_regime_intraday`` and the in-memory buffer.

        Args:
            ts: Timestamp of the tick (timezone-aware recommended).
            vix_value: VIX reading at *ts*.
            prev_close: Previous day's VIX close used for spike detection.
                If ``None``, spike detection is skipped.
        """
        regime = self.classify_value(vix_value)
        spike = (
            self.detect_spike(vix_value, prev_close)
            if prev_close is not None
            else False
        )

        # Update in-memory buffer (cap at 2000 entries)
        self._intraday_buffer.append((ts, vix_value))
        if len(self._intraday_buffer) > 2000:
            self._intraday_buffer = self._intraday_buffer[-2000:]

        if self._engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO vix_regime_intraday (ts, vix_value, regime, spike_detected)
            VALUES (:ts, :vix_value, :regime, :spike_detected)
            ON CONFLICT (ts) DO UPDATE
                SET vix_value = EXCLUDED.vix_value,
                    regime = EXCLUDED.regime,
                    spike_detected = EXCLUDED.spike_detected
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "ts": ts,
                        "vix_value": vix_value,
                        "regime": regime,
                        "spike_detected": spike,
                    },
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist VIX intraday row ts=%s", ts)

    # ------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------

    def compute_daily(self, trade_date: date) -> dict[str, Any]:
        """Aggregate intraday VIX bars, persist to ``vix_regime_daily``.

        Reads today's rows from ``vix_regime_intraday`` when a DB engine is
        provided; otherwise aggregates from the in-memory buffer.

        Args:
            trade_date: The trading date to aggregate.

        Returns:
            Dict with keys ``vix_open``, ``vix_high``, ``vix_low``,
            ``vix_close``, ``vix_change_pct``, ``regime``, ``percentile_60d``.
            Returns an empty dict if no intraday data is available.
        """
        rows = self._load_intraday_for_date(trade_date)
        if not rows:
            logger.warning("No intraday VIX data for %s", trade_date)
            return {}

        values = [r["vix_value"] for r in rows]
        vix_open  = float(values[0])
        vix_high  = float(max(values))
        vix_low   = float(min(values))
        vix_close = float(values[-1])

        # Fetch previous day's close for change_pct calculation
        prev_close = self._get_previous_close(trade_date)
        vix_change_pct: float | None = None
        if prev_close is not None and prev_close != 0:
            vix_change_pct = (vix_close - prev_close) / abs(prev_close) * 100.0

        regime = self.classify_value(vix_close)
        percentile_60d = self.compute_percentile(vix_close, lookback_days=60)

        row_data: dict[str, Any] = {
            "vix_open": vix_open,
            "vix_high": vix_high,
            "vix_low": vix_low,
            "vix_close": vix_close,
            "vix_change_pct": vix_change_pct,
            "regime": regime,
            "percentile_60d": percentile_60d,
        }

        self._persist_daily(trade_date, row_data)
        return row_data

    # ------------------------------------------------------------------
    # Regime query
    # ------------------------------------------------------------------

    def get_today_regime(self, trade_date: date) -> dict[str, Any] | None:
        """Read the latest regime row for *trade_date* from ``vix_regime_daily``.

        Args:
            trade_date: The date to query.

        Returns:
            Dict with the regime row columns, or ``None`` if not found.
        """
        if self._engine is None:
            return None

        from sqlalchemy import text

        sql = text(
            """
            SELECT trade_date, vix_open, vix_high, vix_low, vix_close,
                   vix_change_pct, regime, percentile_60d, computed_at
            FROM vix_regime_daily
            WHERE trade_date = :trade_date
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"trade_date": trade_date}).fetchone()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to query vix_regime_daily for %s", trade_date)
            return None

        if row is None:
            return None

        return {
            "trade_date": row[0],
            "vix_open": float(row[1]) if row[1] is not None else None,
            "vix_high": float(row[2]) if row[2] is not None else None,
            "vix_low": float(row[3]) if row[3] is not None else None,
            "vix_close": float(row[4]),
            "vix_change_pct": float(row[5]) if row[5] is not None else None,
            "regime": str(row[6]),
            "percentile_60d": float(row[7]) if row[7] is not None else None,
            "computed_at": row[8],
        }

    # ------------------------------------------------------------------
    # Strategy gate
    # ------------------------------------------------------------------

    def is_strategy_allowed(
        self,
        strategy_type: str,
        regime: str | None = None,
    ) -> bool:
        """Return whether *strategy_type* should run in the current regime.

        Strategy type mapping (defaults):

        * ``'trend'``        — allowed in NORMAL and HIGH.
        * ``'mean_rev'``     — allowed in HIGH and SPIKE.
        * ``'breakout'``     — allowed in NORMAL only.
        * ``'options_sell'`` — allowed in HIGH only (premium rich environment).

        Unknown strategy types are allowed by default (fail-open).

        Args:
            strategy_type: One of ``'trend'``, ``'mean_rev'``,
                ``'breakout'``, ``'options_sell'``.
            regime: Override regime string; if ``None``, returns ``True``
                (no regime data = fail-open).

        Returns:
            ``True`` when the strategy is permitted to evaluate signals.
        """
        if regime is None:
            return True  # no data — fail-open

        allowed_regimes = _STRATEGY_REGIME_MAP.get(strategy_type)
        if allowed_regimes is None:
            # Unknown strategy type — fail-open
            return True

        return regime in allowed_regimes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_intraday_for_date(self, trade_date: date) -> list[dict[str, Any]]:
        """Load intraday VIX rows for *trade_date* from DB or in-memory buffer."""
        if self._engine is not None:
            from sqlalchemy import text

            sql = text(
                """
                SELECT ts, vix_value, regime, spike_detected
                FROM vix_regime_intraday
                WHERE ts::date = :trade_date
                ORDER BY ts ASC
                """
            )
            try:
                with self._engine.connect() as conn:
                    rows = conn.execute(sql, {"trade_date": trade_date}).fetchall()
                return [
                    {
                        "ts": r[0],
                        "vix_value": float(r[1]),
                        "regime": str(r[2]),
                        "spike_detected": bool(r[3]),
                    }
                    for r in rows
                ]
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load intraday VIX for %s", trade_date)

        # Fall back to in-memory buffer (used in tests without DB)
        result = []
        for ts, vix_val in self._intraday_buffer:
            ts_date = ts.date() if hasattr(ts, "date") else ts
            if ts_date == trade_date:
                result.append({
                    "ts": ts,
                    "vix_value": vix_val,
                    "regime": self.classify_value(vix_val),
                    "spike_detected": False,
                })
        return result

    def _get_previous_close(self, trade_date: date) -> float | None:
        """Fetch the previous session's VIX close from ``vix_regime_daily``."""
        if self._engine is None:
            return None

        from sqlalchemy import text

        sql = text(
            """
            SELECT vix_close
            FROM vix_regime_daily
            WHERE trade_date < :trade_date
            ORDER BY trade_date DESC
            LIMIT 1
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"trade_date": trade_date}).fetchone()
            return float(row[0]) if row else None
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch previous VIX close for %s", trade_date)
            return None

    def _persist_daily(self, trade_date: date, data: dict[str, Any]) -> None:
        """Upsert a row into ``vix_regime_daily``."""
        if self._engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO vix_regime_daily
                (trade_date, vix_open, vix_high, vix_low, vix_close,
                 vix_change_pct, regime, percentile_60d, computed_at)
            VALUES
                (:trade_date, :vix_open, :vix_high, :vix_low, :vix_close,
                 :vix_change_pct, :regime, :percentile_60d, NOW())
            ON CONFLICT (trade_date) DO UPDATE
                SET vix_open = EXCLUDED.vix_open,
                    vix_high = EXCLUDED.vix_high,
                    vix_low  = EXCLUDED.vix_low,
                    vix_close = EXCLUDED.vix_close,
                    vix_change_pct = EXCLUDED.vix_change_pct,
                    regime = EXCLUDED.regime,
                    percentile_60d = EXCLUDED.percentile_60d,
                    computed_at = NOW()
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "trade_date": trade_date,
                        "vix_open": data.get("vix_open"),
                        "vix_high": data.get("vix_high"),
                        "vix_low": data.get("vix_low"),
                        "vix_close": data["vix_close"],
                        "vix_change_pct": data.get("vix_change_pct"),
                        "regime": data["regime"],
                        "percentile_60d": data.get("percentile_60d"),
                    },
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist vix_regime_daily for %s", trade_date)
