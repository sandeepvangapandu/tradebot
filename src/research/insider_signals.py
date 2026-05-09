"""Aggregate insider signals + block/bulk deal institutional flow analysis.

This module queries the ``block_deals``, ``bulk_deals``, and
``insider_trades`` tables to derive actionable signals:

* **Block flow** — net buy/sell pressure by large institutions over N days.
* **Promoter buying** — promoter BUY transactions in the past N days (bullish).
* **Promoter selling** — promoter SELL transactions (bearish warning).
* **Insider net position** — aggregate buy/sell across all insider categories.

Signal rationale
----------------
* Significant block buys (> ₹50 cr net in 3 days) indicate institutional
  accumulation — historically a multi-week bullish signal.
* Sustained promoter buying is one of the strongest bullish signals for Indian
  equities (promoters have skin in the game).
* Large promoter selling (via secondary market) can signal near-term downside.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ist_today() -> date:
    """Return today's date in IST (UTC+5:30)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


# ---------------------------------------------------------------------------
# InsiderSignals
# ---------------------------------------------------------------------------

class InsiderSignals:
    """Aggregates insider trade and block/bulk deal signals for a symbol.

    Args:
        db_engine: Optional SQLAlchemy engine.  When ``None`` all query
            methods return safe defaults (False / empty dict) without raising.
    """

    def __init__(self, db_engine=None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Block deal flow
    # ------------------------------------------------------------------

    def block_flow_for_symbol(
        self,
        symbol: str,
        days: int = 5,
    ) -> dict:
        """Compute aggregate block deal buy/sell flow for a symbol.

        Queries both ``block_deals`` and ``bulk_deals`` tables and
        aggregates by side over the past ``days`` calendar days.

        Args:
            symbol: NSE equity symbol.
            days: Lookback window in calendar days.

        Returns:
            Dict with keys:
              - ``buy_value_crore``  : total BUY value (₹ crore)
              - ``sell_value_crore`` : total SELL value (₹ crore)
              - ``net_value_crore``  : buy − sell
              - ``buyer_count``      : number of distinct BUY deals
              - ``seller_count``     : number of distinct SELL deals
        """
        default = {
            "buy_value_crore": 0.0,
            "sell_value_crore": 0.0,
            "net_value_crore": 0.0,
            "buyer_count": 0,
            "seller_count": 0,
        }

        if self._engine is None:
            return default

        from sqlalchemy import text as sa_text

        cutoff = _ist_today() - timedelta(days=days)
        sym = symbol.upper()

        sql = sa_text(
            """
            SELECT
              side,
              COALESCE(SUM(value_crore), 0) AS total_value,
              COUNT(*) AS deal_count
            FROM block_deals
            WHERE symbol = :symbol AND trade_date >= :cutoff
            GROUP BY side

            UNION ALL

            SELECT
              side,
              COALESCE(SUM(
                CAST(quantity AS NUMERIC) * price_paisa / (100.0 * 1e7)
              ), 0) AS total_value,
              COUNT(*) AS deal_count
            FROM bulk_deals
            WHERE symbol = :symbol AND trade_date >= :cutoff
            GROUP BY side
            """
        )

        buy_val = 0.0
        sell_val = 0.0
        buyer_count = 0
        seller_count = 0

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sql, {"symbol": sym, "cutoff": cutoff}
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("block_flow_for_symbol DB error: %s", exc)
            return default

        for row in rows:
            side, total_value, deal_count = row[0], float(row[1]), int(row[2])
            if side == "BUY":
                buy_val += total_value
                buyer_count += deal_count
            elif side == "SELL":
                sell_val += total_value
                seller_count += deal_count

        return {
            "buy_value_crore": round(buy_val, 4),
            "sell_value_crore": round(sell_val, 4),
            "net_value_crore": round(buy_val - sell_val, 4),
            "buyer_count": buyer_count,
            "seller_count": seller_count,
        }

    def has_significant_block_buy(
        self,
        symbol: str,
        threshold_crore: float = 50.0,
        days: int = 3,
    ) -> bool:
        """Return True if net block/bulk BUY value exceeds the threshold.

        Args:
            symbol: NSE equity symbol.
            threshold_crore: Minimum net BUY value in ₹ crore.
            days: Lookback window in calendar days.

        Returns:
            True when net_value_crore >= threshold_crore.
        """
        flow = self.block_flow_for_symbol(symbol, days=days)
        return flow["net_value_crore"] >= threshold_crore

    # ------------------------------------------------------------------
    # Promoter / insider trading signals
    # ------------------------------------------------------------------

    def promoter_buying_recent(
        self,
        symbol: str,
        days: int = 30,
    ) -> bool:
        """Return True if there is at least one promoter BUY in the window.

        Covers both 'PROMOTER' and 'PROMOTER GROUP' categories.

        Args:
            symbol: NSE equity symbol.
            days: Lookback window in calendar days.

        Returns:
            True if any promoter BUY trade was disclosed in the window.
        """
        if self._engine is None:
            return False

        from sqlalchemy import text as sa_text

        cutoff = _ist_today() - timedelta(days=days)

        sql = sa_text(
            """
            SELECT COUNT(*) FROM insider_trades
            WHERE symbol = :symbol
              AND trade_type = 'BUY'
              AND acquirer_category IN ('PROMOTER', 'PROMOTER GROUP')
              AND (disclosure_date >= :cutoff OR trade_date >= :cutoff)
            """
        )

        try:
            with self._engine.connect() as conn:
                count = conn.execute(
                    sql, {"symbol": symbol.upper(), "cutoff": cutoff}
                ).scalar()
            return int(count or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("promoter_buying_recent DB error: %s", exc)
            return False

    def promoter_selling_recent(
        self,
        symbol: str,
        days: int = 30,
    ) -> bool:
        """Return True if there is at least one promoter SELL in the window.

        Args:
            symbol: NSE equity symbol.
            days: Lookback window in calendar days.

        Returns:
            True if any promoter SELL trade was disclosed in the window.
        """
        if self._engine is None:
            return False

        from sqlalchemy import text as sa_text

        cutoff = _ist_today() - timedelta(days=days)

        sql = sa_text(
            """
            SELECT COUNT(*) FROM insider_trades
            WHERE symbol = :symbol
              AND trade_type = 'SELL'
              AND acquirer_category IN ('PROMOTER', 'PROMOTER GROUP')
              AND (disclosure_date >= :cutoff OR trade_date >= :cutoff)
            """
        )

        try:
            with self._engine.connect() as conn:
                count = conn.execute(
                    sql, {"symbol": symbol.upper(), "cutoff": cutoff}
                ).scalar()
            return int(count or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("promoter_selling_recent DB error: %s", exc)
            return False

    def insider_net_position(
        self,
        symbol: str,
        days: int = 30,
    ) -> dict:
        """Compute net insider buy/sell value and dominant direction.

        Aggregates ALL insider categories (promoters + KMP + others).

        Args:
            symbol: NSE equity symbol.
            days: Lookback window in calendar days.

        Returns:
            Dict::

                {
                  "net_buy_value_crore": float,   # positive = net buy
                  "transaction_count": int,
                  "dominant": "BUY" | "SELL" | "NONE",
                }
        """
        default = {
            "net_buy_value_crore": 0.0,
            "transaction_count": 0,
            "dominant": "NONE",
        }

        if self._engine is None:
            return default

        from sqlalchemy import text as sa_text

        cutoff = _ist_today() - timedelta(days=days)

        sql = sa_text(
            """
            SELECT
              trade_type,
              COALESCE(SUM(value_crore), 0) AS total_value,
              COUNT(*) AS tx_count
            FROM insider_trades
            WHERE symbol = :symbol
              AND trade_type IN ('BUY', 'SELL')
              AND (disclosure_date >= :cutoff OR trade_date >= :cutoff)
            GROUP BY trade_type
            """
        )

        buy_val = 0.0
        sell_val = 0.0
        tx_count = 0

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sql, {"symbol": symbol.upper(), "cutoff": cutoff}
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("insider_net_position DB error: %s", exc)
            return default

        for row in rows:
            trade_type, total_value, count = row[0], float(row[1]), int(row[2])
            tx_count += count
            if trade_type == "BUY":
                buy_val += total_value
            elif trade_type == "SELL":
                sell_val += total_value

        net = round(buy_val - sell_val, 4)

        if net > 0:
            dominant = "BUY"
        elif net < 0:
            dominant = "SELL"
        else:
            dominant = "NONE"

        return {
            "net_buy_value_crore": net,
            "transaction_count": tx_count,
            "dominant": dominant,
        }
