"""Anti-signal veto layer — rejects candidate signals on hard-fail conditions.

This module is a pure gate that runs AFTER raw signal generation and BEFORE
order placement.  Each check returns a :class:`RejectionReason` on failure or
``None`` on pass.  :meth:`RejectionFilter.evaluate` short-circuits on the first
failing check (cheap-to-expensive ordering) and persists a rejection row to the
``rejection_log`` audit table.

Order of checks (cheap → expensive):
  1. market_close     — pure datetime arithmetic, zero I/O
  2. spread           — single float comparison
  3. earnings_blackout — one DB row lookup
  4. corporate_blackout — one DB row lookup
  5. news/insider     — indexed DB query
  6. sector           — indexed DB query
  7. vix_spike        — latest in-memory / DB value
  8. correlated_position_cap — in-memory open positions list
  9. oi_dropping       — options OI recent trend (DB)
  10. liquidity        — ADV lookup (DB)

Typical usage::

    from src.strategy.rejection_filter import RejectionConfig, RejectionFilter

    filt = RejectionFilter(db_engine=db, config=RejectionConfig())
    allowed, reason, details = filt.evaluate(
        signal={
            "strategy_name": "CPR_VWAP_Bounce_BankNifty",
            "instrument_key": "NSE_EQ|RELIANCE",
            "signal_type": "BUY",
            "symbol": "RELIANCE",
        },
        context={
            "current_spread_bps": 15.0,
            "open_positions": [],
            "current_ts": datetime.now(IST),
            "trade_date": date.today(),
        },
    )
    if not allowed:
        logger.info("Signal rejected: %s — %s", reason, details)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any

from src.research.earnings_calendar import EarningsCalendar
from src.research.corporate_calendar import CorporateCalendar
from src.research.news_query import NewsQuery
from src.research.insider_signals import InsiderSignals

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IST timezone constant
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))

# Market close time in IST
_MARKET_CLOSE = time(15, 30, 0)


# ---------------------------------------------------------------------------
# RejectionReason enum
# ---------------------------------------------------------------------------

class RejectionReason(str, Enum):
    """Hard-fail rejection reasons for signal veto."""

    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    EARNINGS_BLACKOUT = "EARNINGS_BLACKOUT"
    CORPORATE_BLACKOUT = "CORPORATE_BLACKOUT"
    NEGATIVE_NEWS_RECENT = "NEGATIVE_NEWS_RECENT"
    VIX_SPIKE = "VIX_SPIKE"
    NEAR_MARKET_CLOSE = "NEAR_MARKET_CLOSE"
    SECTOR_BOTTOM_QUARTILE = "SECTOR_BOTTOM_QUARTILE"
    CORRELATED_POSITION_CAP = "CORRELATED_POSITION_CAP"
    OI_DROPPING = "OI_DROPPING"
    LIQUIDITY_LOW = "LIQUIDITY_LOW"
    POSITIVE_NEWS_RECENT = "POSITIVE_NEWS_RECENT"  # Veto SHORT signals on positive news
    PROMOTER_SELLING = "PROMOTER_SELLING"           # Veto LONG signals
    PROMOTER_BUYING = "PROMOTER_BUYING"             # Veto SHORT signals
    NONE = "NONE"


# ---------------------------------------------------------------------------
# RejectionConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RejectionConfig:
    """Tunable parameters for the rejection filter.

    All thresholds can be overridden per strategy in config YAML.

    Args:
        max_spread_bps: Maximum allowable bid-ask spread in basis points.
            Default 30 bps (0.30%).
        earnings_days_before: Blackout starts N days before earnings date.
        earnings_days_after: Blackout ends N days after earnings date.
        corporate_days_before: Blackout starts N days before corporate ex-date.
        corporate_days_after: Blackout ends N days after corporate ex-date.
        news_lookback_hours: How far back to scan for adverse/positive news.
        vix_spike_check: Whether to apply the VIX spike veto.
        market_close_minutes_buffer: Reject signals after (15:30 - buffer) IST.
            Default 15 means reject after 15:15 IST.
        sector_filter_enabled: Whether to apply sector bottom-quartile veto.
        correlated_position_max_count: Max correlated positions before veto.
        correlated_position_threshold: Correlation coefficient threshold (0-1).
        min_oi_trend_lookback_min: Minutes of OI history to assess trend.
        min_adv_paisa: Minimum Average Daily Volume in paisa (₹ × 100).
            Default ₹50 crore = 50,00,00,000 × 100 paisa
            but stored here as the trade-value floor, not share volume.
            Concretely: 50_00_00_000 paisa = ₹50 crore.
    """

    max_spread_bps: float = 30.0
    earnings_days_before: int = 2
    earnings_days_after: int = 1
    corporate_days_before: int = 1
    corporate_days_after: int = 1
    news_lookback_hours: int = 12
    vix_spike_check: bool = True
    market_close_minutes_buffer: int = 15
    sector_filter_enabled: bool = True
    correlated_position_max_count: int = 3
    correlated_position_threshold: float = 0.8
    min_oi_trend_lookback_min: int = 30
    min_adv_paisa: int = 50_00_00_000  # ₹50 crore in paisa

    # Sector-to-symbol static mapping (mirrors sector_rotation.py)
    sector_map: dict[str, str] = field(default_factory=lambda: {
        "HDFCBANK": "NIFTY_BANK",
        "ICICIBANK": "NIFTY_BANK",
        "AXISBANK": "NIFTY_BANK",
        "KOTAKBANK": "NIFTY_BANK",
        "SBIN": "NIFTY_BANK",
        "TCS": "NIFTY_IT",
        "INFY": "NIFTY_IT",
        "HINDUNILVR": "NIFTY_FMCG",
        "HUL": "NIFTY_FMCG",
        "ITC": "NIFTY_FMCG",
        "RELIANCE": "NIFTY_ENERGY",
    })


# ---------------------------------------------------------------------------
# RejectionFilter
# ---------------------------------------------------------------------------

class RejectionFilter:
    """Veto layer that gates signals on hard-fail conditions.

    Each ``check_*`` method returns a :class:`RejectionReason` on failure or
    ``None`` on pass.  :meth:`evaluate` runs all checks in cheap-to-expensive
    order, short-circuiting on the first failure.

    Args:
        db_engine: Optional SQLAlchemy sync engine.  When ``None``, DB-backed
            checks return ``None`` (pass) so the filter degrades gracefully.
        config: :class:`RejectionConfig` instance.  Defaults to ``RejectionConfig()``.
    """

    def __init__(
        self,
        db_engine: Any = None,
        config: RejectionConfig | None = None,
    ) -> None:
        self._db = db_engine
        self._config = config if config is not None else RejectionConfig()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_spread(
        self,
        instrument_key: str,
        current_spread_bps: float,
    ) -> RejectionReason | None:
        """Reject when the bid-ask spread is too wide.

        Args:
            instrument_key: Upstox instrument key (for logging).
            current_spread_bps: Current spread expressed in basis points.

        Returns:
            :attr:`RejectionReason.SPREAD_TOO_WIDE` when spread exceeds
            ``config.max_spread_bps``, else ``None``.
        """
        if current_spread_bps > self._config.max_spread_bps:
            logger.debug(
                "rejection_filter.spread: %s spread=%.1f bps > max=%.1f bps",
                instrument_key,
                current_spread_bps,
                self._config.max_spread_bps,
            )
            return RejectionReason.SPREAD_TOO_WIDE
        return None

    def check_earnings_blackout(
        self,
        symbol: str,
        trade_date: date,
    ) -> RejectionReason | None:
        """Reject when the trade date falls inside an earnings blackout window.

        Blackout: [earnings_date - days_before, earnings_date + days_after].

        Args:
            symbol: NSE equity symbol.
            trade_date: Calendar date of the proposed trade.

        Returns:
            :attr:`RejectionReason.EARNINGS_BLACKOUT` if in window, else ``None``.
        """
        if self._db is None:
            return None

        try:
            cal = EarningsCalendar(db_engine=self._db)
            in_blackout, _ = cal.is_blackout(
                symbol,
                trade_date,
                days_before=self._config.earnings_days_before,
                days_after=self._config.earnings_days_after,
            )
            if in_blackout:
                logger.debug(
                    "rejection_filter.earnings_blackout: %s on %s", symbol, trade_date
                )
                return RejectionReason.EARNINGS_BLACKOUT
        except Exception:
            logger.exception(
                "rejection_filter.check_earnings_blackout error for %s", symbol
            )
        return None

    def check_corporate_blackout(
        self,
        symbol: str,
        trade_date: date,
    ) -> RejectionReason | None:
        """Reject when the trade date falls inside a corporate-action blackout window.

        Blackout: [ex_date - days_before, ex_date + days_after].

        Args:
            symbol: NSE equity symbol.
            trade_date: Calendar date of the proposed trade.

        Returns:
            :attr:`RejectionReason.CORPORATE_BLACKOUT` if in window, else ``None``.
        """
        if self._db is None:
            return None

        try:
            cal = CorporateCalendar(db_engine=self._db)
            in_blackout, _ = cal.is_blackout(
                symbol,
                trade_date,
                days_before=self._config.corporate_days_before,
                days_after=self._config.corporate_days_after,
            )
            if in_blackout:
                logger.debug(
                    "rejection_filter.corporate_blackout: %s on %s", symbol, trade_date
                )
                return RejectionReason.CORPORATE_BLACKOUT
        except Exception:
            logger.exception(
                "rejection_filter.check_corporate_blackout error for %s", symbol
            )
        return None

    def check_news(
        self,
        symbol: str,
        signal_type: str,
    ) -> RejectionReason | None:
        """Reject based on recent news sentiment.

        - LONG signals rejected when recent negative news is present.
        - SHORT signals rejected when recent positive news is present.

        Args:
            symbol: NSE equity symbol.
            signal_type: One of BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE.

        Returns:
            :attr:`RejectionReason.NEGATIVE_NEWS_RECENT` or
            :attr:`RejectionReason.POSITIVE_NEWS_RECENT`, else ``None``.
        """
        if self._db is None:
            return None

        hours = self._config.news_lookback_hours
        is_long = signal_type.upper() in ("BUY", "BUY_CE")
        is_short = signal_type.upper() in ("SELL", "SELL_PE")

        try:
            nq = NewsQuery(db_engine=self._db)
            if is_long and nq.has_negative_news(symbol, hours=hours):
                logger.debug(
                    "rejection_filter.news: LONG on %s blocked by negative news", symbol
                )
                return RejectionReason.NEGATIVE_NEWS_RECENT
            if is_short and nq.has_positive_news(symbol, hours=hours):
                logger.debug(
                    "rejection_filter.news: SHORT on %s blocked by positive news", symbol
                )
                return RejectionReason.POSITIVE_NEWS_RECENT
        except Exception:
            logger.exception(
                "rejection_filter.check_news error for %s", symbol
            )
        return None

    def check_insider_alignment(
        self,
        symbol: str,
        signal_type: str,
    ) -> RejectionReason | None:
        """Reject when insider/promoter activity conflicts with signal direction.

        - Promoter selling rejects LONG signals (bearish warning).
        - Promoter buying rejects SHORT signals (strong bullish signal).

        Args:
            symbol: NSE equity symbol.
            signal_type: One of BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE.

        Returns:
            :attr:`RejectionReason.PROMOTER_SELLING` or
            :attr:`RejectionReason.PROMOTER_BUYING`, else ``None``.
        """
        if self._db is None:
            return None

        is_long = signal_type.upper() in ("BUY", "BUY_CE")
        is_short = signal_type.upper() in ("SELL", "SELL_PE")

        try:
            signals = InsiderSignals(db_engine=self._db)
            if is_long and signals.promoter_selling_recent(symbol):
                logger.debug(
                    "rejection_filter.insider: LONG on %s blocked — promoter selling",
                    symbol,
                )
                return RejectionReason.PROMOTER_SELLING
            if is_short and signals.promoter_buying_recent(symbol):
                logger.debug(
                    "rejection_filter.insider: SHORT on %s blocked — promoter buying",
                    symbol,
                )
                return RejectionReason.PROMOTER_BUYING
        except Exception:
            logger.exception(
                "rejection_filter.check_insider_alignment error for %s", symbol
            )
        return None

    def check_vix_spike(self) -> RejectionReason | None:
        """Reject when India VIX is in SPIKE regime (VIX > 25).

        Reads the latest intraday VIX value from the DB.  When no DB is
        available or data is missing the check is skipped (returns ``None``).

        Returns:
            :attr:`RejectionReason.VIX_SPIKE` when regime is SPIKE, else ``None``.
        """
        if not self._config.vix_spike_check or self._db is None:
            return None

        try:
            from sqlalchemy import text

            sql = text(
                """
                SELECT vix_value
                FROM vix_regime_intraday
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            with self._db.connect() as conn:
                row = conn.execute(sql).fetchone()

            if row is None:
                return None

            vix_value = float(row[0])
            if vix_value > 25.0:
                logger.debug(
                    "rejection_filter.vix_spike: VIX=%.2f > 25 → SPIKE regime", vix_value
                )
                return RejectionReason.VIX_SPIKE
        except Exception:
            logger.exception("rejection_filter.check_vix_spike error")
        return None

    def check_market_close(
        self,
        current_ts: datetime,
    ) -> RejectionReason | None:
        """Reject signals generated too close to market close.

        Default buffer is 15 minutes — signals after 15:15 IST are rejected.
        This prevents entering positions that cannot be properly managed before
        the mandatory 15:15 IST intraday square-off.

        Args:
            current_ts: Timezone-aware datetime of signal evaluation.  If naive,
                it is assumed to be IST.

        Returns:
            :attr:`RejectionReason.NEAR_MARKET_CLOSE` when past the cutoff,
            else ``None``.
        """
        # Normalise to IST
        if current_ts.tzinfo is None:
            ts_ist = current_ts.replace(tzinfo=IST)
        else:
            ts_ist = current_ts.astimezone(IST)

        buffer_minutes = self._config.market_close_minutes_buffer
        cutoff = datetime.combine(
            ts_ist.date(),
            time(15, 30 - buffer_minutes, 0),
        ).replace(tzinfo=IST)

        if ts_ist >= cutoff:
            logger.debug(
                "rejection_filter.market_close: %s >= cutoff %s",
                ts_ist.strftime("%H:%M:%S"),
                cutoff.strftime("%H:%M:%S"),
            )
            return RejectionReason.NEAR_MARKET_CLOSE
        return None

    def check_sector(
        self,
        symbol: str,
        signal_type: str,
    ) -> RejectionReason | None:
        """Reject LONG signals when the symbol's sector is in the bottom quartile.

        Queries ``sector_rank_daily`` for today's sector rank.  Bottom quartile
        means rank > 6 out of 9 sectors (i.e. rank >= 7 on a 1-9 scale).

        Args:
            symbol: NSE equity symbol.
            signal_type: One of BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE.

        Returns:
            :attr:`RejectionReason.SECTOR_BOTTOM_QUARTILE` when sector is in
            bottom quartile and signal is LONG, else ``None``.
        """
        if not self._config.sector_filter_enabled or self._db is None:
            return None

        is_long = signal_type.upper() in ("BUY", "BUY_CE")
        if not is_long:
            return None

        sector = self._config.sector_map.get(symbol)
        if sector is None:
            return None  # Unknown sector — skip check

        try:
            from sqlalchemy import text

            sql = text(
                """
                SELECT rs_rank, total_sectors
                FROM sector_rank_daily
                WHERE sector_symbol = :sector
                ORDER BY trade_date DESC
                LIMIT 1
                """
            )
            with self._db.connect() as conn:
                row = conn.execute(sql, {"sector": sector}).fetchone()

            if row is None:
                return None

            rs_rank = int(row[0])
            total_sectors = int(row[1]) if row[1] else 9

            # Bottom quartile: rank in worst 25% of sectors
            bottom_quartile_threshold = max(1, total_sectors - total_sectors // 4)
            if rs_rank >= bottom_quartile_threshold:
                logger.debug(
                    "rejection_filter.sector: %s sector=%s rank=%d/%d → bottom quartile",
                    symbol,
                    sector,
                    rs_rank,
                    total_sectors,
                )
                return RejectionReason.SECTOR_BOTTOM_QUARTILE
        except Exception:
            logger.exception(
                "rejection_filter.check_sector error for %s", symbol
            )
        return None

    def check_correlated_positions(
        self,
        symbol: str,
        signal_type: str,
        open_positions: list[dict],
    ) -> RejectionReason | None:
        """Reject when the same-direction correlated position cap is exceeded.

        Uses the ``sector_map`` to identify correlated symbols (same sector).
        When more than ``correlated_position_max_count`` positions in the same
        sector already exist in the same direction, the new signal is vetoed.

        Args:
            symbol: NSE equity symbol for the new signal.
            signal_type: One of BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE.
            open_positions: List of dicts with keys ``symbol`` and ``side``
                (side = "LONG" or "SHORT").

        Returns:
            :attr:`RejectionReason.CORRELATED_POSITION_CAP` when cap exceeded,
            else ``None``.
        """
        if not open_positions:
            return None

        new_sector = self._config.sector_map.get(symbol)
        is_long = signal_type.upper() in ("BUY", "BUY_CE")
        target_side = "LONG" if is_long else "SHORT"

        # Count same-sector, same-direction positions
        correlated_count = 0
        for pos in open_positions:
            pos_symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "").upper()

            pos_sector = self._config.sector_map.get(pos_symbol)
            if pos_sector is not None and pos_sector == new_sector:
                if pos_side == target_side:
                    correlated_count += 1

        max_count = self._config.correlated_position_max_count
        if correlated_count >= max_count:
            logger.debug(
                "rejection_filter.correlated_positions: %s sector=%s already has %d/%d correlated %s positions",
                symbol,
                new_sector,
                correlated_count,
                max_count,
                target_side,
            )
            return RejectionReason.CORRELATED_POSITION_CAP
        return None

    def check_liquidity(
        self,
        symbol: str,
        trade_date: date,
    ) -> RejectionReason | None:
        """Reject when the symbol's average daily volume is below minimum threshold.

        Queries ``liquidity_rank_daily`` (Phase D) for the latest ADV value.
        ADV is stored in paisa in the ``adv_paisa`` column.

        Args:
            symbol: NSE equity symbol.
            trade_date: Calendar date (used for recency check).

        Returns:
            :attr:`RejectionReason.LIQUIDITY_LOW` when ADV < ``config.min_adv_paisa``,
            else ``None``.
        """
        if self._db is None:
            return None

        try:
            from sqlalchemy import text

            sql = text(
                """
                SELECT adv_paisa
                FROM liquidity_rank_daily
                WHERE symbol = :symbol
                ORDER BY trade_date DESC
                LIMIT 1
                """
            )
            with self._db.connect() as conn:
                row = conn.execute(sql, {"symbol": symbol}).fetchone()

            if row is None:
                return None  # No data — don't block

            adv = int(row[0])
            if adv < self._config.min_adv_paisa:
                logger.debug(
                    "rejection_filter.liquidity: %s ADV=%d paisa < min=%d paisa",
                    symbol,
                    adv,
                    self._config.min_adv_paisa,
                )
                return RejectionReason.LIQUIDITY_LOW
        except Exception:
            logger.exception(
                "rejection_filter.check_liquidity error for %s", symbol
            )
        return None

    def check_oi_dropping(
        self,
        instrument_key: str,
        signal_type: str,
    ) -> RejectionReason | None:
        """Reject when options OI has been declining (positions being closed).

        Reads the last ``min_oi_trend_lookback_min`` minutes of OI snapshots
        from ``options_chain_snapshots`` and checks for a net downward trend.
        Only applies to options signal types (BUY_CE, SELL_CE, BUY_PE, SELL_PE).

        Args:
            instrument_key: Upstox instrument key for the underlying index.
            signal_type: Signal type string.

        Returns:
            :attr:`RejectionReason.OI_DROPPING` when OI trend is negative,
            else ``None``.
        """
        is_options = signal_type.upper() in ("BUY_CE", "SELL_CE", "BUY_PE", "SELL_PE")
        if not is_options or self._db is None:
            return None

        lookback_min = self._config.min_oi_trend_lookback_min

        try:
            from sqlalchemy import text

            sql = text(
                f"""
                SELECT total_call_oi, total_put_oi, ts
                FROM options_chain_snapshots
                WHERE underlying_key = :key
                  AND ts >= NOW() - INTERVAL '{int(lookback_min)} minutes'
                ORDER BY ts ASC
                """
            )
            with self._db.connect() as conn:
                rows = conn.execute(sql, {"key": instrument_key}).fetchall()

            if len(rows) < 2:
                return None  # Not enough data to determine trend

            # Determine which OI to track based on signal type
            if "CE" in signal_type.upper():
                first_oi = float(rows[0][0] or 0)   # total_call_oi
                last_oi = float(rows[-1][0] or 0)
            else:
                first_oi = float(rows[0][1] or 0)   # total_put_oi
                last_oi = float(rows[-1][1] or 0)

            if last_oi < first_oi:
                logger.debug(
                    "rejection_filter.oi_dropping: %s OI %.0f→%.0f (signal=%s)",
                    instrument_key,
                    first_oi,
                    last_oi,
                    signal_type,
                )
                return RejectionReason.OI_DROPPING
        except Exception:
            logger.exception(
                "rejection_filter.check_oi_dropping error for %s", instrument_key
            )
        return None

    # ------------------------------------------------------------------
    # Main evaluate entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: dict,
        context: dict,
    ) -> tuple[bool, RejectionReason, dict]:
        """Run all rejection checks in cheap-to-expensive order.

        Short-circuits on the first failure.  Returns ``(True, NONE, {})`` when
        all checks pass.

        Args:
            signal: Dict with keys:
                - ``strategy_name`` (str)
                - ``instrument_key`` (str)
                - ``signal_type`` (str)
                - ``symbol`` (str)
                - ``signal_id`` (int, optional)
            context: Dict with keys:
                - ``current_spread_bps`` (float, default 0.0)
                - ``open_positions`` (list[dict], default [])
                - ``current_ts`` (datetime, default now IST)
                - ``trade_date`` (date, default today IST)

        Returns:
            Tuple of:
            - ``allowed`` (bool): True when all checks pass.
            - ``reason`` (:class:`RejectionReason`): NONE when allowed.
            - ``details`` (dict): Contextual data for the rejection reason.
        """
        symbol = signal.get("symbol", "")
        instrument_key = signal.get("instrument_key", "")
        signal_type = signal.get("signal_type", "")

        current_ts: datetime = context.get("current_ts") or datetime.now(IST)
        trade_date: date = context.get("trade_date") or current_ts.astimezone(IST).date()
        spread_bps: float = float(context.get("current_spread_bps", 0.0))
        open_positions: list[dict] = context.get("open_positions") or []

        # Ordered checks: cheap → expensive
        checks = [
            (
                "market_close",
                lambda: self.check_market_close(current_ts),
                {"current_ts": str(current_ts), "buffer_min": self._config.market_close_minutes_buffer},
            ),
            (
                "spread",
                lambda: self.check_spread(instrument_key, spread_bps),
                {"spread_bps": spread_bps, "max_spread_bps": self._config.max_spread_bps},
            ),
            (
                "earnings_blackout",
                lambda: self.check_earnings_blackout(symbol, trade_date),
                {"symbol": symbol, "trade_date": str(trade_date)},
            ),
            (
                "corporate_blackout",
                lambda: self.check_corporate_blackout(symbol, trade_date),
                {"symbol": symbol, "trade_date": str(trade_date)},
            ),
            (
                "news",
                lambda: self.check_news(symbol, signal_type),
                {"symbol": symbol, "signal_type": signal_type, "hours": self._config.news_lookback_hours},
            ),
            (
                "insider",
                lambda: self.check_insider_alignment(symbol, signal_type),
                {"symbol": symbol, "signal_type": signal_type},
            ),
            (
                "sector",
                lambda: self.check_sector(symbol, signal_type),
                {"symbol": symbol, "signal_type": signal_type},
            ),
            (
                "vix_spike",
                lambda: self.check_vix_spike(),
                {},
            ),
            (
                "correlated_positions",
                lambda: self.check_correlated_positions(symbol, signal_type, open_positions),
                {"symbol": symbol, "open_positions_count": len(open_positions)},
            ),
            (
                "oi_dropping",
                lambda: self.check_oi_dropping(instrument_key, signal_type),
                {"instrument_key": instrument_key, "signal_type": signal_type},
            ),
            (
                "liquidity",
                lambda: self.check_liquidity(symbol, trade_date),
                {"symbol": symbol, "min_adv_paisa": self._config.min_adv_paisa},
            ),
        ]

        for check_name, check_fn, base_details in checks:
            try:
                reason = check_fn()
            except Exception:
                logger.exception(
                    "rejection_filter.evaluate: unexpected error in check '%s'", check_name
                )
                continue

            if reason is not None:
                details = {
                    "check": check_name,
                    **base_details,
                }
                self.log_rejection(signal, reason, details)
                return False, reason, details

        return True, RejectionReason.NONE, {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def log_rejection(
        self,
        signal: dict,
        reason: RejectionReason,
        details: dict,
    ) -> int:
        """Persist a rejection record to the ``rejection_log`` table.

        Args:
            signal: Original signal dict (strategy_name, instrument_key, signal_type).
            reason: :class:`RejectionReason` enum value.
            details: Contextual data dict (spread value, check name, etc.).

        Returns:
            The newly inserted ``id`` (bigserial), or ``-1`` on failure / no DB.
        """
        if self._db is None:
            return -1

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO rejection_log
                (strategy_name, instrument_key, signal_type, rejection_reason, details, signal_id)
            VALUES
                (:strategy_name, :instrument_key, :signal_type,
                 :rejection_reason, CAST(:details AS jsonb), :signal_id)
            RETURNING id
            """
        )
        try:
            with self._db.begin() as conn:
                row = conn.execute(
                    sql,
                    {
                        "strategy_name": signal.get("strategy_name", "unknown"),
                        "instrument_key": signal.get("instrument_key", ""),
                        "signal_type": signal.get("signal_type", ""),
                        "rejection_reason": reason.value,
                        "details": json.dumps(details),
                        "signal_id": signal.get("signal_id"),
                    },
                ).fetchone()
            return int(row[0]) if row else -1
        except Exception:
            logger.exception("rejection_filter.log_rejection: DB insert failed")
            return -1

    def get_rejection_stats(
        self,
        strategy_name: str | None = None,
        days: int = 7,
    ) -> dict:
        """Return aggregated rejection counts over the last N days.

        Args:
            strategy_name: Optional filter to a single strategy.  When ``None``
                all strategies are included.
            days: Look-back window in days.

        Returns:
            Dict with keys:
            - ``total_rejections`` (int)
            - ``by_reason`` (dict[str, int]): rejection count per reason.
        """
        if self._db is None:
            return {"total_rejections": 0, "by_reason": {}}

        from sqlalchemy import text

        if strategy_name:
            sql = text(
                """
                SELECT rejection_reason, COUNT(*) AS cnt
                FROM rejection_log
                WHERE strategy_name = :strategy_name
                  AND ts >= NOW() - (:days * INTERVAL '1 day')
                GROUP BY rejection_reason
                ORDER BY cnt DESC
                """
            )
            params: dict = {"strategy_name": strategy_name, "days": days}
        else:
            sql = text(
                """
                SELECT rejection_reason, COUNT(*) AS cnt
                FROM rejection_log
                WHERE ts >= NOW() - (:days * INTERVAL '1 day')
                GROUP BY rejection_reason
                ORDER BY cnt DESC
                """
            )
            params = {"days": days}

        try:
            with self._db.connect() as conn:
                rows = conn.execute(sql, params).fetchall()

            by_reason = {str(row[0]): int(row[1]) for row in rows}
            total = sum(by_reason.values())
            return {"total_rejections": total, "by_reason": by_reason}
        except Exception:
            logger.exception("rejection_filter.get_rejection_stats: DB query failed")
            return {"total_rejections": 0, "by_reason": {}}
