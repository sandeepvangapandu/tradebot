"""NSE EOD block + bulk deals scraper.

Block deals: large trades (>= ₹10 crore or 0.5% of total shares) executed in
the separate Block Deal window (8:45–9:00 AM) or Regular session.  NSE
publishes them EOD.

Bulk deals: trades where quantity transacted > 0.5% of the total number of
shares of a listed company in a single trading day.

Primary sources
---------------
* Block deals : https://www.nseindia.com/api/block-deals?date=DD-MM-YYYY
* Bulk deals  : https://www.nseindia.com/api/historical/cm/bulk_deals
                  ?from=DD-MM-YYYY&to=DD-MM-YYYY

Both endpoints require:
  1. A prior GET to https://www.nseindia.com to obtain a session cookie.
  2. Browser-like headers (User-Agent, Referer, Accept).

All monetary values:
  - ``price_paisa``   : price in paisa  (int)  = price_rupee * 100
  - ``value_crore``   : approximate deal value in ₹ crore  (float)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NSE_HOME = "https://www.nseindia.com"
_NSE_BLOCK_API = "https://www.nseindia.com/api/block-deals"
_NSE_BULK_API = "https://www.nseindia.com/api/historical/cm/bulk_deals"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

_HOME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ist_today() -> date:
    """Return today's date in IST (UTC+5:30)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _fmt_date(d: date) -> str:
    """Format date as DD-MM-YYYY (NSE API format)."""
    return d.strftime("%d-%m-%Y")


def _parse_float(value: Any) -> float | None:
    """Parse a string/number to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _rupee_to_paisa(rupee: float | None) -> int | None:
    """Convert rupee float to paisa int."""
    if rupee is None:
        return None
    return round(rupee * 100)


def _crore_from_qty_price(qty: int, price_paisa: int) -> float:
    """Compute deal value in crore from quantity and price in paisa."""
    return round((qty * price_paisa) / (100 * 1e7), 4)


# ---------------------------------------------------------------------------
# BlockDealsScraper
# ---------------------------------------------------------------------------

class BlockDealsScraper:
    """Scrapes and persists NSE EOD block deals and bulk deals.

    Args:
        db_engine: Optional SQLAlchemy engine.  When ``None`` the
            ``persist_*`` methods are no-ops (useful in tests).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine=None, timeout: int = 30) -> None:
        self._engine = db_engine
        self._timeout = timeout

    # ------------------------------------------------------------------
    # NSE session cookie helper
    # ------------------------------------------------------------------

    def _get_nse_cookies(self, client: httpx.Client) -> dict:
        """Fetch NSE homepage to obtain session cookies.

        Args:
            client: Active httpx.Client instance.

        Returns:
            Cookie dict (may be empty if the request fails).
        """
        try:
            resp = client.get(_NSE_HOME, headers=_HOME_HEADERS, follow_redirects=True)
            resp.raise_for_status()
            return dict(resp.cookies)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE homepage fetch for cookies failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Block deals
    # ------------------------------------------------------------------

    def scrape_nse_block_deals(self, trade_date: date | None = None) -> list[dict]:
        """Fetch block deals from NSE API for the given trade date.

        Endpoint: GET /api/block-deals?date=DD-MM-YYYY

        The response JSON looks like::

            {
              "data": [
                {
                  "symbol": "RELIANCE",
                  "clientName": "AXIS MF",
                  "dealType": "BUY",
                  "quantity": 500000,
                  "price": "2450.50"
                },
                ...
              ]
            }

        Args:
            trade_date: Date for which to fetch block deals.  Defaults to
                today (IST).

        Returns:
            List of normalised deal dicts ready for ``persist_block``.
            Returns an empty list on any error.
        """
        if trade_date is None:
            trade_date = _ist_today()

        params = {"date": _fmt_date(trade_date)}

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                cookies = self._get_nse_cookies(client)
                resp = client.get(
                    _NSE_BLOCK_API,
                    headers=_NSE_HEADERS,
                    params=params,
                    cookies=cookies,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE block deals fetch failed for %s: %s", trade_date, exc)
            return []

        return self._parse_block_response(payload, trade_date, exchange="NSE")

    def _parse_block_response(
        self,
        payload: dict | list,
        trade_date: date,
        exchange: str = "NSE",
    ) -> list[dict]:
        """Parse the NSE block deals JSON response.

        Args:
            payload: Parsed JSON from the NSE endpoint.
            trade_date: Trading date these deals belong to.
            exchange: Exchange label ('NSE' or 'BSE').

        Returns:
            List of normalised deal dicts.
        """
        if isinstance(payload, dict):
            rows = payload.get("data", payload.get("blockDeals", []))
        elif isinstance(payload, list):
            rows = payload
        else:
            logger.warning("Unexpected block deals payload type: %s", type(payload))
            return []

        if not isinstance(rows, list):
            logger.warning("Block deals 'data' is not a list: %s", type(rows))
            return []

        deals: list[dict] = []
        for row in rows:
            try:
                deal = self._normalise_block_row(row, trade_date, exchange)
                if deal is not None:
                    deals.append(deal)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping block deal row (parse error): %s — %s", row, exc)

        logger.info("Parsed %d block deals for %s", len(deals), trade_date)
        return deals

    def _normalise_block_row(
        self,
        row: dict,
        trade_date: date,
        exchange: str,
    ) -> dict | None:
        """Normalise a single block deal row.

        Args:
            row: Raw dict from the NSE JSON response.
            trade_date: Trading date.
            exchange: Exchange label.

        Returns:
            Normalised dict or None if required fields are missing.
        """
        symbol = (
            row.get("symbol") or row.get("Symbol") or row.get("scrip_cd")
        )
        if not symbol:
            return None

        side_raw = (
            row.get("dealType") or row.get("deal_type") or
            row.get("buySell") or row.get("BS") or ""
        )
        side = _normalise_side(side_raw)
        if side is None:
            return None

        qty_raw = row.get("quantity") or row.get("Quantity") or row.get("qty") or 0
        qty = _safe_int(qty_raw)
        if qty is None or qty <= 0:
            return None

        price_raw = row.get("price") or row.get("Price") or row.get("wap") or 0
        price_float = _parse_float(price_raw)
        if price_float is None or price_float <= 0:
            return None
        price_paisa = _rupee_to_paisa(price_float)

        client_name = (
            row.get("clientName") or row.get("client_name") or
            row.get("ClientName") or None
        )

        value_crore = _crore_from_qty_price(qty, price_paisa)

        return {
            "trade_date": trade_date,
            "symbol": str(symbol).strip().upper(),
            "client_name": str(client_name).strip() if client_name else None,
            "side": side,
            "quantity": qty,
            "price_paisa": price_paisa,
            "value_crore": value_crore,
            "exchange": exchange,
            "source": "NSE",
        }

    # ------------------------------------------------------------------
    # Bulk deals
    # ------------------------------------------------------------------

    def scrape_nse_bulk_deals(self, trade_date: date | None = None) -> list[dict]:
        """Fetch bulk deals from NSE historical API for the given trade date.

        Endpoint: GET /api/historical/cm/bulk_deals
                    ?from=DD-MM-YYYY&to=DD-MM-YYYY

        The response JSON looks like::

            {
              "data": [
                {
                  "symbol": "INFY",
                  "clientName": "XYZ SECURITIES",
                  "buySell": "B",
                  "quantityTraded": 1000000,
                  "wap": "1700.25"
                },
                ...
              ]
            }

        Args:
            trade_date: Date for which to fetch bulk deals.  Defaults to
                today (IST).

        Returns:
            List of normalised deal dicts ready for ``persist_bulk``.
        """
        if trade_date is None:
            trade_date = _ist_today()

        date_str = _fmt_date(trade_date)
        params = {"from": date_str, "to": date_str}

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                cookies = self._get_nse_cookies(client)
                resp = client.get(
                    _NSE_BULK_API,
                    headers=_NSE_HEADERS,
                    params=params,
                    cookies=cookies,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE bulk deals fetch failed for %s: %s", trade_date, exc)
            return []

        return self._parse_bulk_response(payload, trade_date, exchange="NSE")

    def _parse_bulk_response(
        self,
        payload: dict | list,
        trade_date: date,
        exchange: str = "NSE",
    ) -> list[dict]:
        """Parse the NSE bulk deals JSON response.

        Args:
            payload: Parsed JSON from the NSE endpoint.
            trade_date: Trading date.
            exchange: Exchange label.

        Returns:
            List of normalised bulk deal dicts.
        """
        if isinstance(payload, dict):
            rows = payload.get("data", payload.get("bulkDeals", []))
        elif isinstance(payload, list):
            rows = payload
        else:
            logger.warning("Unexpected bulk deals payload type: %s", type(payload))
            return []

        if not isinstance(rows, list):
            logger.warning("Bulk deals 'data' is not a list: %s", type(rows))
            return []

        deals: list[dict] = []
        for row in rows:
            try:
                deal = self._normalise_bulk_row(row, trade_date, exchange)
                if deal is not None:
                    deals.append(deal)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping bulk deal row (parse error): %s — %s", row, exc)

        logger.info("Parsed %d bulk deals for %s", len(deals), trade_date)
        return deals

    def _normalise_bulk_row(
        self,
        row: dict,
        trade_date: date,
        exchange: str,
    ) -> dict | None:
        """Normalise a single bulk deal row.

        Args:
            row: Raw dict from the NSE JSON.
            trade_date: Trading date.
            exchange: Exchange label.

        Returns:
            Normalised dict or None if required fields are missing.
        """
        symbol = (
            row.get("symbol") or row.get("Symbol") or row.get("scrip_cd")
        )
        if not symbol:
            return None

        side_raw = (
            row.get("buySell") or row.get("dealType") or
            row.get("deal_type") or row.get("BS") or ""
        )
        side = _normalise_side(side_raw)
        if side is None:
            return None

        qty_raw = (
            row.get("quantityTraded") or row.get("quantity") or
            row.get("Quantity") or row.get("qty") or 0
        )
        qty = _safe_int(qty_raw)
        if qty is None or qty <= 0:
            return None

        price_raw = row.get("wap") or row.get("price") or row.get("Price") or 0
        price_float = _parse_float(price_raw)
        if price_float is None or price_float <= 0:
            return None
        price_paisa = _rupee_to_paisa(price_float)

        client_name = (
            row.get("clientName") or row.get("client_name") or
            row.get("ClientName") or None
        )

        return {
            "trade_date": trade_date,
            "symbol": str(symbol).strip().upper(),
            "client_name": str(client_name).strip() if client_name else None,
            "side": side,
            "quantity": qty,
            "price_paisa": price_paisa,
            "exchange": exchange,
            "source": "NSE",
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_block(self, deals: list[dict]) -> int:
        """Upsert block deals into the ``block_deals`` table.

        Uses INSERT … ON CONFLICT DO NOTHING so the method is idempotent.

        Args:
            deals: List of normalised deal dicts (output of
                ``scrape_nse_block_deals``).

        Returns:
            Number of rows actually inserted (0 if engine is None or list
            is empty).
        """
        if self._engine is None or not deals:
            return 0

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            INSERT INTO block_deals
              (trade_date, symbol, client_name, side, quantity, price_paisa,
               value_crore, exchange, source)
            VALUES
              (:trade_date, :symbol, :client_name, :side, :quantity, :price_paisa,
               :value_crore, :exchange, :source)
            ON CONFLICT (trade_date, symbol, client_name, side, quantity, price_paisa)
            DO NOTHING
            """
        )

        inserted = 0
        try:
            with self._engine.begin() as conn:
                for deal in deals:
                    result = conn.execute(sql, deal)
                    inserted += result.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("persist_block DB error: %s", exc)

        logger.info("Persisted %d / %d block deals", inserted, len(deals))
        return inserted

    def persist_bulk(self, deals: list[dict]) -> int:
        """Upsert bulk deals into the ``bulk_deals`` table.

        Uses INSERT … ON CONFLICT DO NOTHING so the method is idempotent.

        Args:
            deals: List of normalised deal dicts (output of
                ``scrape_nse_bulk_deals``).

        Returns:
            Number of rows actually inserted.
        """
        if self._engine is None or not deals:
            return 0

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            INSERT INTO bulk_deals
              (trade_date, symbol, client_name, side, quantity, price_paisa,
               exchange, source)
            VALUES
              (:trade_date, :symbol, :client_name, :side, :quantity, :price_paisa,
               :exchange, :source)
            ON CONFLICT (trade_date, symbol, client_name, side, quantity, price_paisa)
            DO NOTHING
            """
        )

        inserted = 0
        try:
            with self._engine.begin() as conn:
                for deal in deals:
                    result = conn.execute(sql, deal)
                    inserted += result.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("persist_bulk DB error: %s", exc)

        logger.info("Persisted %d / %d bulk deals", inserted, len(deals))
        return inserted

    # ------------------------------------------------------------------
    # Daily runner
    # ------------------------------------------------------------------

    def run_daily(self, trade_date: date | None = None) -> dict:
        """Scrape and persist both block and bulk deals for a single date.

        This is the scheduler entry-point.

        Args:
            trade_date: The trading date to scrape.  Defaults to today (IST).

        Returns:
            Summary dict::

                {
                  "trade_date": date(2026, 5, 8),
                  "block_fetched": 12,
                  "block_inserted": 10,
                  "bulk_fetched": 7,
                  "bulk_inserted": 6,
                }
        """
        if trade_date is None:
            trade_date = _ist_today()

        block_deals = self.scrape_nse_block_deals(trade_date)
        block_inserted = self.persist_block(block_deals)

        bulk_deals = self.scrape_nse_bulk_deals(trade_date)
        bulk_inserted = self.persist_bulk(bulk_deals)

        summary = {
            "trade_date": trade_date,
            "block_fetched": len(block_deals),
            "block_inserted": block_inserted,
            "bulk_fetched": len(bulk_deals),
            "bulk_inserted": bulk_inserted,
        }
        logger.info("BlockDealsScraper.run_daily summary: %s", summary)
        return summary


# ---------------------------------------------------------------------------
# Private helpers (module-level)
# ---------------------------------------------------------------------------

def _normalise_side(raw: str) -> str | None:
    """Map raw side string to 'BUY' or 'SELL'.

    Handles NSE values like 'B', 'BUY', 'Buy', 'S', 'SELL', 'Sell'.

    Args:
        raw: Raw side string from the API.

    Returns:
        'BUY', 'SELL', or None if unrecognised.
    """
    val = str(raw).strip().upper()
    if val in {"B", "BUY", "BUY/SUBSCRIPTION", "PURCHASE", "ACQUISITION"}:
        return "BUY"
    if val in {"S", "SELL", "SALE", "DISPOSAL"}:
        return "SELL"
    return None


def _safe_int(value: Any) -> int | None:
    """Parse a value to int; return None on failure."""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
