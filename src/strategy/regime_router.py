"""Regime router — decides which strategy types are allowed each day.

Reads the daily context from Phase B tables:
  - ``vix_regime_daily``    → VIX regime (LOW/NORMAL/HIGH/SPIKE)
  - ``sector_rank_daily``   → sector breadth (fraction with positive RS)
  - ``macro_regime_daily``  → aggregate macro signal (BULLISH/BEARISH/NEUTRAL)
  - ``flow_regime_daily``   → FII/DII combined signal (TAILWIND/HEADWIND/MIXED)

Classifies a single ``MarketRegime`` for the day, then maps it to a list of
allowed ``StrategyType`` values via ``REGIME_ALLOWLIST``.

The strategy engine calls :meth:`RegimeRouter.is_strategy_type_allowed` before
evaluating any strategy to gate which strategy types may fire today.

Persistence:
  - Each call to :meth:`evaluate_today` writes one row to
    ``regime_decisions_daily`` (upsert — safe to call multiple times).
  - :meth:`get_today_decision` reads that row back without re-computing.

Typical usage::

    from src.strategy.regime_router import RegimeRouter
    from src.storage.db import get_sync_engine

    router = RegimeRouter(db_engine=get_sync_engine())
    decision = router.evaluate_today(date.today())
    allowed = decision["allowed_strategy_types"]
    if router.is_strategy_type_allowed("trend", date.today()):
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    """Top-level market regime for the trading day."""

    TREND_BULL = "TREND_BULL"
    TREND_BEAR = "TREND_BEAR"
    RANGE = "RANGE"
    CHOP = "CHOP"
    HIGH_VOL = "HIGH_VOL"


class StrategyType(str, Enum):
    """Broad strategy-type categories used for regime gating."""

    TREND = "trend"
    MEAN_REV = "mean_rev"
    BREAKOUT = "breakout"
    OPTIONS_SELL = "options_sell"
    OPTIONS_BUY = "options_buy"


# ---------------------------------------------------------------------------
# Regime → allowed strategy types matrix
# ---------------------------------------------------------------------------

#: Which strategy types are permitted for each regime.
#: Options-sell is for premium-collection when vol is elevated but not spiking.
REGIME_ALLOWLIST: dict[MarketRegime, list[StrategyType]] = {
    MarketRegime.TREND_BULL: [
        StrategyType.TREND,
        StrategyType.BREAKOUT,
        StrategyType.OPTIONS_BUY,
    ],
    MarketRegime.TREND_BEAR: [
        StrategyType.TREND,
        StrategyType.BREAKOUT,
    ],
    MarketRegime.RANGE: [
        StrategyType.MEAN_REV,
        StrategyType.OPTIONS_SELL,
    ],
    MarketRegime.CHOP: [
        StrategyType.OPTIONS_SELL,  # Only premium-collection in chop
    ],
    MarketRegime.HIGH_VOL: [
        StrategyType.OPTIONS_BUY,
        StrategyType.MEAN_REV,
    ],
}

# Reasons reported for blocked strategy types (human-readable)
_BLOCK_REASONS: dict[MarketRegime, dict[StrategyType, str]] = {
    MarketRegime.TREND_BULL: {
        StrategyType.MEAN_REV:     "Strong uptrend — mean-reversion against trend",
        StrategyType.OPTIONS_SELL: "Trend day — premium collection risk on directional moves",
    },
    MarketRegime.TREND_BEAR: {
        StrategyType.MEAN_REV:     "Strong downtrend — mean-reversion against trend",
        StrategyType.OPTIONS_SELL: "Trend day — premium collection risk on directional moves",
        StrategyType.OPTIONS_BUY:  "Bear trend — long options directional bias uncertain",
    },
    MarketRegime.RANGE: {
        StrategyType.TREND:       "Ranging market — trend-following whipsaws",
        StrategyType.BREAKOUT:    "Ranging market — false breakouts likely",
        StrategyType.OPTIONS_BUY: "Ranging market — directional options decay risk",
    },
    MarketRegime.CHOP: {
        StrategyType.TREND:       "Choppy market — trend signals unreliable",
        StrategyType.MEAN_REV:    "Choppy market — no clear mean to revert to",
        StrategyType.BREAKOUT:    "Choppy market — false breakout probability high",
        StrategyType.OPTIONS_BUY: "Choppy market — directional bias absent; premium-only mode",
    },
    MarketRegime.HIGH_VOL: {
        StrategyType.TREND:       "High VIX spike — trend strategies face extreme slippage",
        StrategyType.BREAKOUT:    "High VIX spike — breakout signals unreliable",
        StrategyType.OPTIONS_SELL: "High VIX spike — selling premium into spike dangerous",
    },
}

# All strategy types (universe)
_ALL_STRATEGY_TYPES: list[StrategyType] = list(StrategyType)


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegimeContext:
    """Snapshot of regime inputs for a single trading day.

    Args:
        vix_regime: One of LOW / NORMAL / HIGH / SPIKE from Phase B.2.
        sector_breadth: Fraction (0-1) of tracked sectors with positive RS.
        macro_signal: BULLISH / BEARISH / NEUTRAL derived from macro_regime_daily.
        flow_signal: TAILWIND / HEADWIND / MIXED from flow_regime_daily.
    """

    vix_regime: str          # LOW / NORMAL / HIGH / SPIKE
    sector_breadth: float    # 0-1
    macro_signal: str        # BULLISH / BEARISH / NEUTRAL
    flow_signal: str         # TAILWIND / HEADWIND / MIXED


# ---------------------------------------------------------------------------
# Regime Router
# ---------------------------------------------------------------------------


class RegimeRouter:
    """Morning regime classifier and strategy-type gate.

    Args:
        db_engine: SQLAlchemy sync engine connected to the trading DB.
                   When ``None``, all persistence methods are no-ops and
                   :meth:`fetch_context` returns a neutral fallback context.
    """

    def __init__(self, db_engine: Any = None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Context fetch
    # ------------------------------------------------------------------

    def fetch_context(self, trade_date: date) -> RegimeContext:
        """Read regime inputs for *trade_date* from Phase B tables.

        Falls back gracefully when DB is unavailable or rows are missing:
        returns a neutral context (NORMAL VIX, 0.5 breadth, NEUTRAL macro,
        MIXED flow) which classifies as RANGE.

        Args:
            trade_date: The date to query.

        Returns:
            :class:`RegimeContext` populated from DB (or neutral fallback).
        """
        vix_regime = "NORMAL"
        sector_breadth = 0.5
        macro_signal = "NEUTRAL"
        flow_signal = "MIXED"

        if self._engine is None:
            logger.debug("No DB engine — returning neutral RegimeContext fallback")
            return RegimeContext(
                vix_regime=vix_regime,
                sector_breadth=sector_breadth,
                macro_signal=macro_signal,
                flow_signal=flow_signal,
            )

        from sqlalchemy import text

        # --- VIX regime --------------------------------------------------
        try:
            sql = text(
                "SELECT regime FROM vix_regime_daily "
                "WHERE trade_date = :d"
            )
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"d": trade_date}).fetchone()
            if row is not None:
                vix_regime = str(row[0])
        except Exception:
            logger.exception("fetch_context: failed to read vix_regime_daily")

        # --- Sector breadth (fraction of sectors with positive RS) -------
        try:
            sql = text(
                "SELECT COUNT(*) FILTER (WHERE rs_score > 0) AS pos_count, "
                "       COUNT(*) AS total "
                "FROM sector_rank_daily "
                "WHERE trade_date = :d"
            )
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"d": trade_date}).fetchone()
            if row is not None and row[1] and row[1] > 0:
                sector_breadth = float(row[0]) / float(row[1])
        except Exception:
            logger.exception("fetch_context: failed to read sector_rank_daily")

        # --- Macro signal (majority vote across instruments) -------------
        try:
            sql = text(
                "SELECT trend FROM macro_regime_daily "
                "WHERE trade_date = :d"
            )
            with self._engine.connect() as conn:
                rows = conn.execute(sql, {"d": trade_date}).fetchall()
            if rows:
                trends = [str(r[0]) for r in rows if r[0] is not None]
                up = trends.count("UP")
                down = trends.count("DOWN")
                if up > down:
                    macro_signal = "BULLISH"
                elif down > up:
                    macro_signal = "BEARISH"
                else:
                    macro_signal = "NEUTRAL"
        except Exception:
            logger.exception("fetch_context: failed to read macro_regime_daily")

        # --- FII/DII flow signal -----------------------------------------
        try:
            sql = text(
                "SELECT combined_signal FROM flow_regime_daily "
                "WHERE trade_date = :d"
            )
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"d": trade_date}).fetchone()
            if row is not None and row[0] is not None:
                flow_signal = str(row[0])
        except Exception:
            logger.exception("fetch_context: failed to read flow_regime_daily")

        return RegimeContext(
            vix_regime=vix_regime,
            sector_breadth=sector_breadth,
            macro_signal=macro_signal,
            flow_signal=flow_signal,
        )

    # ------------------------------------------------------------------
    # Classification decision tree
    # ------------------------------------------------------------------

    def classify_regime(self, ctx: RegimeContext) -> MarketRegime:
        """Classify the market regime from the given context.

        Decision tree (evaluated top-to-bottom, first match wins):

        1. VIX SPIKE                                         → HIGH_VOL
        2. VIX HIGH + flow HEADWIND                          → CHOP
        3. VIX LOW + sector_breadth > 0.6 + macro BULLISH   → TREND_BULL
        4. VIX LOW/NORMAL + sector_breadth < 0.4            → TREND_BEAR
        5. else                                              → RANGE

        Args:
            ctx: The :class:`RegimeContext` for the day.

        Returns:
            The :class:`MarketRegime` for the day.
        """
        vix = ctx.vix_regime.upper()
        flow = ctx.flow_signal.upper()
        macro = ctx.macro_signal.upper()
        breadth = ctx.sector_breadth

        # Rule 1 — VIX spike overrides everything
        if vix == "SPIKE":
            logger.info("classify_regime: VIX SPIKE → HIGH_VOL")
            return MarketRegime.HIGH_VOL

        # Rule 2 — Elevated VIX + headwind → chop
        if vix == "HIGH" and flow == "HEADWIND":
            logger.info("classify_regime: VIX HIGH + HEADWIND → CHOP")
            return MarketRegime.CHOP

        # Rule 3 — Low VIX + strong breadth + bullish macro → trend bull
        if vix == "LOW" and breadth > 0.6 and macro == "BULLISH":
            logger.info("classify_regime: VIX LOW + breadth %.2f + BULLISH → TREND_BULL", breadth)
            return MarketRegime.TREND_BULL

        # Rule 4 — Weak breadth → bear trend
        if breadth < 0.4:
            logger.info("classify_regime: sector_breadth %.2f < 0.4 → TREND_BEAR", breadth)
            return MarketRegime.TREND_BEAR

        # Default
        logger.info("classify_regime: default → RANGE")
        return MarketRegime.RANGE

    # ------------------------------------------------------------------
    # Strategy type allowlist
    # ------------------------------------------------------------------

    def decide_allowed_strategies(self, regime: MarketRegime) -> list[StrategyType]:
        """Return the list of allowed strategy types for *regime*.

        Reads from :data:`REGIME_ALLOWLIST`. May be extended in future with
        ``bot_config`` overrides read from the DB.

        Args:
            regime: The classified :class:`MarketRegime`.

        Returns:
            List of allowed :class:`StrategyType` values.
        """
        return list(REGIME_ALLOWLIST.get(regime, []))

    def _build_blocked_record(
        self, regime: MarketRegime, allowed: list[StrategyType]
    ) -> dict[str, str]:
        """Build a dict mapping blocked strategy type names to human-readable reasons.

        Args:
            regime: The classified regime.
            allowed: The allowed strategy types.

        Returns:
            Dict ``{strategy_type_value: reason_string}``.
        """
        allowed_values = {st.value for st in allowed}
        blocked: dict[str, str] = {}
        regime_reasons = _BLOCK_REASONS.get(regime, {})
        for st in _ALL_STRATEGY_TYPES:
            if st.value not in allowed_values:
                reason = regime_reasons.get(st, f"Not allowed in {regime.value} regime")
                blocked[st.value] = reason
        return blocked

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def evaluate_today(self, trade_date: date) -> dict[str, Any]:
        """Run the full regime evaluation pipeline for *trade_date*.

        Steps:
        1. :meth:`fetch_context` — read VIX / sector / macro / flow from DB.
        2. :meth:`classify_regime` — decision tree → MarketRegime.
        3. :meth:`decide_allowed_strategies` — map to allowed types.
        4. Persist to ``regime_decisions_daily`` (upsert).

        Args:
            trade_date: The date to evaluate (typically today pre-market).

        Returns:
            Dict matching the ``regime_decisions_daily`` schema, plus
            ``"allowed_strategy_types"`` and ``"blocked_strategy_types"``
            as plain lists/dicts (in addition to the JSONB serialised form).
        """
        ctx = self.fetch_context(trade_date)
        regime = self.classify_regime(ctx)
        allowed = self.decide_allowed_strategies(regime)
        blocked = self._build_blocked_record(regime, allowed)

        allowed_values = [st.value for st in allowed]
        evaluated_at = datetime.now(tz=timezone.utc)

        record: dict[str, Any] = {
            "trade_date": trade_date,
            "vix_regime": ctx.vix_regime,
            "sector_breadth": ctx.sector_breadth,
            "macro_signal": ctx.macro_signal,
            "flow_signal": ctx.flow_signal,
            "market_regime": regime.value,
            "allowed_strategy_types": allowed_values,
            "blocked_strategy_types": blocked,
            "evaluated_at": evaluated_at,
        }

        self._persist(record)
        return record

    # ------------------------------------------------------------------
    # Convenience readers
    # ------------------------------------------------------------------

    def get_today_decision(self, trade_date: date) -> dict[str, Any] | None:
        """Read the persisted regime decision for *trade_date* from DB.

        Args:
            trade_date: The date to query.

        Returns:
            Dict matching ``regime_decisions_daily`` columns, or ``None``
            if no row exists or DB is unavailable.
        """
        if self._engine is None:
            return None

        from sqlalchemy import text

        sql = text(
            """
            SELECT trade_date, vix_regime, sector_breadth, macro_signal,
                   flow_signal, market_regime, allowed_strategy_types,
                   blocked_strategy_types, evaluated_at
            FROM regime_decisions_daily
            WHERE trade_date = :d
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"d": trade_date}).fetchone()
        except Exception:
            logger.exception("get_today_decision: failed to query DB for %s", trade_date)
            return None

        if row is None:
            return None

        allowed_raw = row[6]
        blocked_raw = row[7]

        # SQLAlchemy may return JSONB as dict/list or as string depending on driver
        allowed = self._ensure_list(allowed_raw)
        blocked = self._ensure_dict(blocked_raw)

        return {
            "trade_date": row[0],
            "vix_regime": row[1],
            "sector_breadth": float(row[2]) if row[2] is not None else None,
            "macro_signal": row[3],
            "flow_signal": row[4],
            "market_regime": row[5],
            "allowed_strategy_types": allowed,
            "blocked_strategy_types": blocked,
            "evaluated_at": row[8],
        }

    def is_strategy_type_allowed(self, strategy_type: str, trade_date: date) -> bool:
        """Return whether *strategy_type* is allowed for *trade_date*.

        Queries ``regime_decisions_daily`` for a persisted decision.
        Falls back to ``True`` (permissive) if no decision exists for
        the date, so the strategy engine isn't blocked by missing data.

        Args:
            strategy_type: Strategy type string (e.g. ``"trend"``).
            trade_date: The date to check.

        Returns:
            ``True`` if the strategy type is in the allowed list.
        """
        decision = self.get_today_decision(trade_date)
        if decision is None:
            # No decision yet — permissive fallback
            logger.warning(
                "is_strategy_type_allowed: no decision for %s — defaulting to allowed",
                trade_date,
            )
            return True
        allowed = decision.get("allowed_strategy_types") or []
        return strategy_type in allowed

    def get_blocked_reason(self, strategy_type: str, trade_date: date) -> str | None:
        """Return the human-readable block reason for *strategy_type* on *trade_date*.

        Args:
            strategy_type: Strategy type string.
            trade_date: The date to check.

        Returns:
            Reason string, or ``None`` if the strategy type is allowed
            (or no decision exists).
        """
        decision = self.get_today_decision(trade_date)
        if decision is None:
            return None
        blocked = decision.get("blocked_strategy_types") or {}
        return blocked.get(strategy_type)

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _persist(self, record: dict[str, Any]) -> None:
        """Upsert *record* into ``regime_decisions_daily``.

        No-op when DB engine is unavailable.

        Args:
            record: Dict matching the ``regime_decisions_daily`` columns.
        """
        if self._engine is None:
            logger.debug("_persist: no DB engine — skipping persistence")
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO regime_decisions_daily
              (trade_date, vix_regime, sector_breadth, macro_signal,
               flow_signal, market_regime, allowed_strategy_types,
               blocked_strategy_types, evaluated_at)
            VALUES
              (:trade_date, :vix_regime, :sector_breadth, :macro_signal,
               :flow_signal, :market_regime, :allowed_strategy_types::jsonb,
               :blocked_strategy_types::jsonb, :evaluated_at)
            ON CONFLICT (trade_date) DO UPDATE SET
              vix_regime             = EXCLUDED.vix_regime,
              sector_breadth         = EXCLUDED.sector_breadth,
              macro_signal           = EXCLUDED.macro_signal,
              flow_signal            = EXCLUDED.flow_signal,
              market_regime          = EXCLUDED.market_regime,
              allowed_strategy_types = EXCLUDED.allowed_strategy_types,
              blocked_strategy_types = EXCLUDED.blocked_strategy_types,
              evaluated_at           = EXCLUDED.evaluated_at
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "trade_date": record["trade_date"],
                        "vix_regime": record.get("vix_regime"),
                        "sector_breadth": record.get("sector_breadth"),
                        "macro_signal": record.get("macro_signal"),
                        "flow_signal": record.get("flow_signal"),
                        "market_regime": record["market_regime"],
                        "allowed_strategy_types": json.dumps(
                            record.get("allowed_strategy_types", [])
                        ),
                        "blocked_strategy_types": json.dumps(
                            record.get("blocked_strategy_types", {})
                        ),
                        "evaluated_at": record.get("evaluated_at"),
                    },
                )
            logger.info(
                "_persist: saved regime_decisions_daily for %s — regime=%s allowed=%s",
                record["trade_date"],
                record["market_regime"],
                record.get("allowed_strategy_types"),
            )
        except Exception:
            logger.exception("_persist: failed to upsert regime_decisions_daily")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_list(value: Any) -> list:
        """Coerce a JSONB list (or JSON string) to a Python list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    @staticmethod
    def _ensure_dict(value: Any) -> dict:
        """Coerce a JSONB dict (or JSON string) to a Python dict."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}
