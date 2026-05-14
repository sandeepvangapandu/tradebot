"""Confluence scoring layer.

Combines multi-dimensional condition results into a 0-100 composite score.
Strategies require a score >= threshold (default 70) to fire.
Every evaluation is persisted to the ``confluence_scores`` audit table.

Usage example::

    from src.strategy.confluence_engine import (
        ConfluenceDimension,
        ConfluenceEngine,
        DimensionResult,
    )

    engine = ConfluenceEngine(threshold=70.0, db_engine=db_engine)

    results = [
        DimensionResult(
            dimension=ConfluenceDimension.PRICE_ACTION,
            score=0.9,
            weight=0.25,
            matched=["cpr_breakout", "ema_aligned"],
            failed=[],
        ),
        DimensionResult(
            dimension=ConfluenceDimension.VIX,
            score=0.0,
            weight=0.10,
            skipped=True,           # No data — re-normalised out
        ),
    ]

    outcome = engine.evaluate(
        strategy_name="CPR_VWAP_Bounce_BankNifty",
        instrument_key="NSE_INDEX|Nifty Bank",
        signal_type="BUY_CE",
        dimension_results=results,
    )
    if outcome["passed"]:
        place_order(...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension enum
# ---------------------------------------------------------------------------

class ConfluenceDimension(str, Enum):
    """Scoring dimensions contributing to the confluence composite score."""

    PRICE_ACTION = "price_action"       # Original strategy conditions (CPR, ORB, EMA, etc.)
    MICROSTRUCTURE = "microstructure"   # CVD divergence, depth walls, spread
    OPTIONS = "options"                 # PCR, IV percentile, OI build
    VOLUME_PROFILE = "volume_profile"   # POC/VAH/VAL position
    SECTOR = "sector"                   # Sector rotation rank
    MACRO = "macro"                     # USDINR/Crude/Gold cross-asset
    VIX = "vix"                         # VIX regime
    FLOWS = "flows"                     # FII/DII regime, bond yield trend
    NEWS = "news"                       # Sentiment, news volume
    INSIDER = "insider"                 # Block deals, promoter activity


# ---------------------------------------------------------------------------
# Default weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[ConfluenceDimension, float] = {
    ConfluenceDimension.PRICE_ACTION:   0.25,   # Highest — primary signal source
    ConfluenceDimension.MICROSTRUCTURE: 0.15,
    ConfluenceDimension.OPTIONS:        0.10,
    ConfluenceDimension.VOLUME_PROFILE: 0.10,
    ConfluenceDimension.SECTOR:         0.10,
    ConfluenceDimension.MACRO:          0.05,
    ConfluenceDimension.VIX:            0.10,
    ConfluenceDimension.FLOWS:          0.05,
    ConfluenceDimension.NEWS:           0.05,
    ConfluenceDimension.INSIDER:        0.05,
}
# Sum = 1.000 exactly.


# ---------------------------------------------------------------------------
# DimensionResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    """Result for a single scoring dimension.

    Args:
        dimension: Which confluence dimension this result belongs to.
        score: Normalised score in [0.0, 1.0].
        weight: Contribution weight (used if ``skipped=False``).
        matched: Names of conditions that were met/passed.
        failed: Names of conditions that failed.
        skipped: When True, this dimension has no data and is excluded from
                 the composite score.  The remaining weights are re-normalised
                 so skipped dimensions do not penalise the score.
    """

    dimension: ConfluenceDimension
    score: float                                        # 0-1 normalised
    weight: float                                       # Configurable per dimension
    matched: list[str] = field(default_factory=list)   # Conditions met
    failed: list[str] = field(default_factory=list)    # Conditions failed
    skipped: bool = False                               # No data available


# ---------------------------------------------------------------------------
# ConfluenceEngine
# ---------------------------------------------------------------------------

class ConfluenceEngine:
    """Centralised signal-scoring layer.

    Computes a 0-100 composite score from per-dimension results.
    Strategies fire only when the composite score reaches ``threshold``.
    Every evaluation is written to the ``confluence_scores`` table for audit
    and ML feedback.

    Args:
        weights: Per-dimension weight map.  Defaults to ``DEFAULT_WEIGHTS``.
                 Values must sum to 1.0 (±0.01 tolerance).
        threshold: Minimum composite score (0-100) required to pass.
                   Configurable per strategy; default 70.0.
        db_engine: SQLAlchemy sync engine.  When ``None``, persistence is
                   silently skipped (useful for unit tests that only verify
                   score calculation).
    """

    def __init__(
        self,
        weights: dict | None = None,
        threshold: float = 70.0,
        db_engine: Any = None,
    ) -> None:
        self._weights: dict[ConfluenceDimension, float] = (
            {ConfluenceDimension(k) if isinstance(k, str) else k: float(v)
             for k, v in weights.items()}
            if weights is not None
            else dict(DEFAULT_WEIGHTS)
        )
        self._validate_weights(self._weights)
        self._threshold = float(threshold)
        self._db_engine = db_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        strategy_name: str,
        instrument_key: str,
        signal_type: str,
        dimension_results: list[DimensionResult],
        signal_id: int | None = None,
    ) -> dict[str, Any]:
        """Compute composite score and persist audit row.

        Skipped dimensions do not penalise the score — the remaining weights
        are re-normalised to sum to 1.0 before the weighted average is taken.
        If ALL dimensions are skipped, the composite score is 0.

        Args:
            strategy_name: Strategy identifier.
            instrument_key: Upstox instrument key.
            signal_type: One of BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE.
            dimension_results: Per-dimension condition outcomes.
            signal_id: Optional FK to signals.id.

        Returns:
            Dict with keys:
                composite_score (float 0-100),
                passed (bool),
                threshold (float),
                dimension_breakdown (list[dict]),
                matched_conditions (dict),
                failed_conditions (dict).
        """
        if not dimension_results:
            logger.warning(
                "confluence_engine.evaluate called with no dimension results for %s %s",
                strategy_name,
                instrument_key,
            )
            return self._build_result(
                composite_score=0.0,
                passed=False,
                breakdown=[],
                matched={},
                failed={},
            )

        # --- compute composite score ---
        active = [r for r in dimension_results if not r.skipped]
        if not active:
            composite_score = 0.0
        else:
            # Use the per-result weight first; fall back to engine weight map.
            total_weight = sum(self._effective_weight(r) for r in active)
            if total_weight == 0.0:
                composite_score = 0.0
            else:
                weighted_sum = sum(
                    r.score * self._effective_weight(r) for r in active
                )
                composite_score = (weighted_sum / total_weight) * 100.0

        composite_score = min(100.0, max(0.0, composite_score))
        passed = composite_score >= self._threshold

        # --- build breakdown / condition maps ---
        breakdown = []
        matched_all: dict[str, list[str]] = {}
        failed_all: dict[str, list[str]] = {}

        for r in dimension_results:
            eff_weight = self._effective_weight(r)
            breakdown.append(
                {
                    "dimension": r.dimension.value,
                    "score": r.score,
                    "weight": eff_weight,
                    "skipped": r.skipped,
                    "matched": r.matched,
                    "failed": r.failed,
                }
            )
            if r.matched:
                matched_all[r.dimension.value] = r.matched
            if r.failed:
                failed_all[r.dimension.value] = r.failed

        # --- persist audit row ---
        self._persist(
            strategy_name=strategy_name,
            instrument_key=instrument_key,
            signal_type=signal_type,
            composite_score=composite_score,
            passed=passed,
            breakdown=breakdown,
            matched=matched_all,
            failed=failed_all,
            signal_id=signal_id,
        )

        return self._build_result(
            composite_score=composite_score,
            passed=passed,
            breakdown=breakdown,
            matched=matched_all,
            failed=failed_all,
        )

    def get_recent_scores(
        self,
        strategy_name: str,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Return recent confluence_scores rows for a strategy.

        Args:
            strategy_name: Strategy identifier to filter on.
            hours: Look-back window in hours (default 24).

        Returns:
            List of dicts (id, ts, instrument_key, signal_type, composite_score,
            passed, threshold, dimension_scores).
        """
        if self._db_engine is None:
            return []

        from sqlalchemy import text

        sql = text(
            """
            SELECT id, ts, instrument_key, signal_type,
                   composite_score, passed, threshold, dimension_scores
            FROM confluence_scores
            WHERE strategy_name = :strategy_name
              AND ts >= NOW() - (:hours * INTERVAL '1 hour')
            ORDER BY ts DESC
            """
        )
        try:
            with self._db_engine.connect() as conn:
                rows = conn.execute(
                    sql,
                    {"strategy_name": strategy_name, "hours": hours},
                ).fetchall()
            return [
                {
                    "id": row[0],
                    "ts": row[1],
                    "instrument_key": row[2],
                    "signal_type": row[3],
                    "composite_score": float(row[4]),
                    "passed": bool(row[5]),
                    "threshold": float(row[6]),
                    "dimension_scores": row[7],
                }
                for row in rows
            ]
        except Exception:
            logger.exception("confluence_engine: get_recent_scores failed")
            return []

    def get_pass_rate(
        self,
        strategy_name: str,
        days: int = 7,
    ) -> float:
        """Return fraction of evaluations that passed over the last N days.

        Args:
            strategy_name: Strategy identifier.
            days: Look-back window in days (default 7).

        Returns:
            Pass rate in [0.0, 1.0].  Returns 0.0 when no data or on error.
        """
        if self._db_engine is None:
            return 0.0

        from sqlalchemy import text

        sql = text(
            """
            SELECT
                COUNT(*) FILTER (WHERE passed) AS pass_count,
                COUNT(*) AS total_count
            FROM confluence_scores
            WHERE strategy_name = :strategy_name
              AND ts >= NOW() - (:days * INTERVAL '1 day')
            """
        )
        try:
            with self._db_engine.connect() as conn:
                row = conn.execute(
                    sql,
                    {"strategy_name": strategy_name, "days": days},
                ).fetchone()
            if row is None or row[1] == 0:
                return 0.0
            return float(row[0]) / float(row[1])
        except Exception:
            logger.exception("confluence_engine: get_pass_rate failed")
            return 0.0

    def update_weights(self, weights: dict) -> None:
        """Update dimension weights and optionally persist to bot_config.

        Args:
            weights: New weight map.  Keys may be ConfluenceDimension members
                     or their string values.  Values must sum to 1.0 (±0.01).

        Raises:
            ValueError: If weights do not sum to ~1.0.
        """
        normalised = {
            ConfluenceDimension(k) if isinstance(k, str) else k: float(v)
            for k, v in weights.items()
        }
        self._validate_weights(normalised)
        self._weights = normalised

        # Persist to bot_config if a db engine is available.
        if self._db_engine is not None:
            self._persist_weights_to_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_weight(self, result: DimensionResult) -> float:
        """Return the weight to use for a DimensionResult.

        Prefers the weight stored in the result itself (allows per-call
        overrides); falls back to the engine's weight map.
        """
        if result.weight != 0.0:
            return result.weight
        return self._weights.get(result.dimension, 0.0)

    @staticmethod
    def _validate_weights(weights: dict[ConfluenceDimension, float]) -> None:
        """Raise ValueError if weights do not sum to ~1.0.

        Args:
            weights: Weight map to validate.

        Raises:
            ValueError: When total deviates from 1.0 by more than 0.01.
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Confluence dimension weights must sum to 1.0 (got {total:.4f})"
            )

    def _build_result(
        self,
        composite_score: float,
        passed: bool,
        breakdown: list[dict],
        matched: dict,
        failed: dict,
    ) -> dict[str, Any]:
        """Assemble the public result dict."""
        return {
            "composite_score": composite_score,
            "passed": passed,
            "threshold": self._threshold,
            "dimension_breakdown": breakdown,
            "matched_conditions": matched,
            "failed_conditions": failed,
        }

    def _persist(
        self,
        strategy_name: str,
        instrument_key: str,
        signal_type: str,
        composite_score: float,
        passed: bool,
        breakdown: list[dict],
        matched: dict,
        failed: dict,
        signal_id: int | None,
    ) -> None:
        """Insert a row into confluence_scores.  Silently swallows errors."""
        if self._db_engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO confluence_scores
                (strategy_name, instrument_key, signal_type,
                 composite_score, passed, threshold,
                 dimension_scores, matched_conditions, failed_conditions,
                 signal_id)
            VALUES
                (:strategy_name, :instrument_key, :signal_type,
                 :composite_score, :passed, :threshold,
                 CAST(:dimension_scores AS jsonb),
                 CAST(:matched_conditions AS jsonb),
                 CAST(:failed_conditions AS jsonb),
                 :signal_id)
            """
        )
        try:
            with self._db_engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "strategy_name": strategy_name,
                        "instrument_key": instrument_key,
                        "signal_type": signal_type,
                        "composite_score": composite_score,
                        "passed": passed,
                        "threshold": self._threshold,
                        "dimension_scores": json.dumps(breakdown),
                        "matched_conditions": json.dumps(matched),
                        "failed_conditions": json.dumps(failed),
                        "signal_id": signal_id,
                    },
                )
        except Exception:
            logger.exception("confluence_engine: failed to persist score row")

    def _persist_weights_to_config(self) -> None:
        """Write current weights to bot_config table under key 'confluence_weights'."""
        if self._db_engine is None:
            return

        from sqlalchemy import text

        payload = {dim.value: w for dim, w in self._weights.items()}
        sql = text(
            """
            INSERT INTO bot_config (key, value)
            VALUES ('confluence_weights', CAST(:value AS jsonb))
            ON CONFLICT (key) DO UPDATE
                SET value = CAST(:value AS jsonb), updated_at = NOW()
            """
        )
        try:
            with self._db_engine.begin() as conn:
                conn.execute(sql, {"value": json.dumps(payload)})
        except Exception:
            logger.exception("confluence_engine: failed to persist weights to bot_config")
