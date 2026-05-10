"""NSE/BSE insider trading (SAST / PIT) disclosures scraper.

Regulatory context
------------------
SEBI mandates two types of public disclosures for insider transactions:

1. **SAST (Substantial Acquisition of Shares and Takeovers)** — triggered when
   a promoter/acquirer crosses 5%, 10%, 15%, … thresholds.  NSE publishes
   these at:
   https://www.nseindia.com/companies-listing/corporate-filings-insider-trading
   (via JSON API: /api/corporates-pit?symbol=RELIANCE&from=…&to=…)

2. **PIT (Prohibition of Insider Trading)** — all trades by designated persons
   (KMP, promoters) must be disclosed within 2 trading days.  NSE provides:
   /api/corporates-pit?symbol=RELIANCE&from=…&to=…

This module uses the NSE JSON API for both and parses a common schema.

Design notes
------------
* Acquirer category is normalised to: 'PROMOTER', 'PROMOTER GROUP', 'KMP',
  'OTHER'.
* Trade type is normalised to: 'BUY', 'SELL', 'PLEDGE', 'REVOKE'.
* All DB writes use INSERT … ON CONFLICT DO NOTHING (idempotent).
* ``run_for_universe`` is the scheduler entry-point.
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
_NSE_PIT_API = "https://www.nseindia.com/api/corporates-pit"

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

# Category classification keywords
_PROMOTER_KEYWORDS = {"promoter"}
_PROMOTER_GROUP_KEYWORDS = {"promoter group", "promoter grp", "promoter & promoter group"}
_KMP_KEYWORDS = {"kmp", "key managerial", "managing director", "director", "ceo", "cfo", "cs", "company secretary", "executive"}

# Trade type classification keywords
_BUY_KEYWORDS = {"acquisition", "buy", "purchase", "subscribe", "subscription", "market purchase", "on market purchase"}
_SELL_KEYWORDS = {"disposal", "sell", "sale", "on market sale", "market sale"}
_PLEDGE_KEYWORDS = {"pledge", "pledged", "encumbrance", "creation of encumbrance"}
_REVOKE_KEYWORDS = {"revoke", "revocation", "release", "invocation"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ist_today() -> date:
    """Return today's date in IST (UTC+5:30)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _fmt_date(d: date) -> str:
    """Format date as DD-MM-YYYY (NSE API format)."""
    return d.strftime("%d-%m-%Y")


def _parse_float(value: Any) -> float | None:
    """Parse a value to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Parse a value to int; return None on failure."""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_date(raw: str | None) -> date | None:
    """Parse a date string in various formats to a Python date.

    Tries: DD-MMM-YYYY (e.g. '01-Jan-2026'), DD-MM-YYYY, YYYY-MM-DD.

    Args:
        raw: Raw date string.

    Returns:
        Parsed date or None.
    """
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse date: %s", raw)
    return None


# ---------------------------------------------------------------------------
# InsiderScraper
# ---------------------------------------------------------------------------

class InsiderScraper:
    """Scrapes and persists NSE/BSE insider trading (SAST/PIT) disclosures.

    Args:
        db_engine: Optional SQLAlchemy engine.  When ``None`` the
            ``persist`` method is a no-op (useful in tests).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine=None, timeout: int = 30) -> None:
        self._engine = db_engine
        self._timeout = timeout

    # ------------------------------------------------------------------
    # NSE session cookie helper
    # ------------------------------------------------------------------

    def _get_nse_cookies(self, client: httpx.Client) -> dict:
        """Obtain a session cookie from the NSE homepage.

        Args:
            client: Active httpx.Client instance.

        Returns:
            Cookie dict (may be empty on failure).
        """
        try:
            resp = client.get(_NSE_HOME, headers=_HOME_HEADERS, follow_redirects=True)
            resp.raise_for_status()
            return dict(resp.cookies)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE homepage fetch for cookies failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape_nse_insider(
        self,
        symbol: str,
        lookback_days: int = 90,
    ) -> list[dict]:
        """Fetch insider trading disclosures (PIT/SAST) from NSE for a symbol.

        Endpoint: GET /api/corporates-pit
                    ?symbol=RELIANCE&from=DD-MM-YYYY&to=DD-MM-YYYY

        Sample response structure::

            {
              "data": [
                {
                  "symbol": "RELIANCE",
                  "acqName": "Mukesh D Ambani",
                  "personCategory": "Promoter",
                  "secAcq": "500000",
                  "secVal": "1225.50",
                  "tdpTransactionType": "Acquisition",
                  "befAcqSharesNo": "0",
                  "afterAcqSharesNo": "500000",
                  "intimDt": "01-Jan-2026",
                  "date": "28-Dec-2025"
                }
              ]
            }

        Args:
            symbol: NSE equity symbol (e.g. 'RELIANCE').
            lookback_days: Number of past calendar days to query.

        Returns:
            List of normalised trade dicts ready for ``persist``.
        """
        today = _ist_today()
        from_date = today - timedelta(days=lookback_days)
        params = {
            "symbol": symbol.upper(),
            "from": _fmt_date(from_date),
            "to": _fmt_date(today),
        }

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                cookies = self._get_nse_cookies(client)
                resp = client.get(
                    _NSE_PIT_API,
                    headers=_NSE_HEADERS,
                    params=params,
                    cookies=cookies,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NSE insider fetch failed for %s: %s", symbol, exc
            )
            return []

        return self._parse_pit_response(payload, symbol)

    def _parse_pit_response(
        self,
        payload: dict | list,
        symbol: str,
    ) -> list[dict]:
        """Parse the NSE PIT/SAST JSON response.

        Args:
            payload: Parsed JSON from the NSE endpoint.
            symbol: The equity symbol (for fallback if not in the row).

        Returns:
            List of normalised insider trade dicts.
        """
        if isinstance(payload, dict):
            rows = payload.get("data", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            logger.warning("Unexpected insider payload type: %s", type(payload))
            return []

        if not isinstance(rows, list):
            return []

        trades: list[dict] = []
        for row in rows:
            try:
                trade = self._normalise_pit_row(row, symbol)
                if trade is not None:
                    trades.append(trade)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping insider row (parse error): %s — %s", row, exc)

        logger.info("Parsed %d insider trades for %s", len(trades), symbol)
        return trades

    def _normalise_pit_row(self, row: dict, default_symbol: str) -> dict | None:
        """Normalise a single PIT/SAST row.

        Args:
            row: Raw dict from the NSE JSON.
            default_symbol: Fallback symbol if not present in the row.

        Returns:
            Normalised dict or None if required fields are missing/invalid.
        """
        symbol = (
            row.get("symbol") or row.get("Symbol") or default_symbol
        )
        if not symbol:
            return None

        acquirer_name = (
            row.get("acqName") or row.get("acquirerName") or
            row.get("personName") or None
        )

        category_raw = (
            row.get("personCategory") or row.get("category") or
            row.get("acqCategory") or ""
        )
        acquirer_category = self.parse_acquirer_category(str(category_raw))

        trade_type_raw = (
            row.get("tdpTransactionType") or row.get("transactionType") or
            row.get("tradeType") or ""
        )
        trade_type = self.parse_trade_type(str(trade_type_raw))
        if trade_type is None:
            return None

        qty_raw = (
            row.get("secAcq") or row.get("quantity") or row.get("noOfShares") or 0
        )
        quantity = _safe_int(qty_raw)

        # Value: NSE provides secVal as a number (price per share? or total?)
        # We treat it as total value in lakhs; convert to crore.
        val_raw = (
            row.get("secVal") or row.get("totalValue") or row.get("value") or None
        )
        val_float = _parse_float(val_raw)
        # NSE often provides secVal as total value in rupees; divide by 1e7 for crore
        value_crore: float | None = None
        if val_float is not None:
            value_crore = round(val_float / 1e7, 4) if val_float > 1e4 else val_float

        # Dates
        trade_date_raw = (
            row.get("date") or row.get("tradeDate") or row.get("transactionDate") or None
        )
        disclosure_date_raw = (
            row.get("intimDt") or row.get("disclosureDate") or
            row.get("filingDate") or None
        )

        trade_date = _parse_date(str(trade_date_raw)) if trade_date_raw else None
        disclosure_date = _parse_date(str(disclosure_date_raw)) if disclosure_date_raw else None

        return {
            "symbol": str(symbol).strip().upper(),
            "acquirer_name": str(acquirer_name).strip() if acquirer_name else None,
            "acquirer_category": acquirer_category,
            "trade_type": trade_type,
            "quantity": quantity,
            "value_crore": value_crore,
            "trade_date": trade_date,
            "disclosure_date": disclosure_date,
            "source": "NSE_SAST",
        }

    # ------------------------------------------------------------------
    # Category / trade type parsers
    # ------------------------------------------------------------------

    def parse_acquirer_category(self, raw: str) -> str:
        """Normalise raw acquirer/person category to a standard label.

        Mapping:
          - 'Promoter Group' / 'Promoter Grp'           → 'PROMOTER GROUP'
          - 'Promoter'                                   → 'PROMOTER'
          - 'KMP' / 'Key Managerial' / 'Director' / …   → 'KMP'
          - anything else                                → 'OTHER'

        Args:
            raw: Raw category string from the NSE/BSE response.

        Returns:
            Normalised category string.
        """
        lower = raw.lower().strip()

        # Check promoter group first (more specific)
        for kw in _PROMOTER_GROUP_KEYWORDS:
            if kw in lower:
                return "PROMOTER GROUP"

        for kw in _PROMOTER_KEYWORDS:
            if lower == kw or lower.startswith(kw + " "):
                return "PROMOTER"

        for kw in _KMP_KEYWORDS:
            if kw in lower:
                return "KMP"

        return "OTHER"

    def parse_trade_type(self, raw: str) -> str | None:
        """Normalise raw transaction type to 'BUY', 'SELL', 'PLEDGE', 'REVOKE'.

        Args:
            raw: Raw transaction type string (e.g. 'Acquisition',
                'Disposal', 'Pledge', 'Revoke Pledge').

        Returns:
            Normalised string or None if unrecognised.
        """
        lower = raw.lower().strip()

        for kw in _REVOKE_KEYWORDS:
            if kw in lower:
                return "REVOKE"

        for kw in _PLEDGE_KEYWORDS:
            if kw in lower:
                return "PLEDGE"

        for kw in _BUY_KEYWORDS:
            if kw in lower:
                return "BUY"

        for kw in _SELL_KEYWORDS:
            if kw in lower:
                return "SELL"

        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, trades: list[dict]) -> int:
        """Upsert insider trades into the ``insider_trades`` table.

        Uses INSERT … ON CONFLICT DO NOTHING (idempotent).

        Args:
            trades: List of normalised trade dicts from ``scrape_nse_insider``.

        Returns:
            Number of rows actually inserted.
        """
        if self._engine is None or not trades:
            return 0

        from sqlalchemy import text as sa_text

        sql = sa_text(
            """
            INSERT INTO insider_trades
              (symbol, acquirer_name, acquirer_category, trade_type,
               quantity, value_crore, trade_date, disclosure_date, source)
            VALUES
              (:symbol, :acquirer_name, :acquirer_category, :trade_type,
               :quantity, :value_crore, :trade_date, :disclosure_date, :source)
            ON CONFLICT (symbol, acquirer_name, trade_type, trade_date, quantity)
            DO NOTHING
            """
        )

        inserted = 0
        try:
            with self._engine.begin() as conn:
                for trade in trades:
                    result = conn.execute(sql, trade)
                    inserted += result.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("persist insider_trades DB error: %s", exc)

        logger.info("Persisted %d / %d insider trades", inserted, len(trades))
        return inserted

    # ------------------------------------------------------------------
    # Universe runner
    # ------------------------------------------------------------------

    def run_for_universe(
        self,
        symbols: list[str],
        lookback_days: int = 90,
    ) -> dict:
        """Scrape and persist insider disclosures for all symbols in universe.

        This is the scheduler entry-point.

        Args:
            symbols: List of NSE equity symbols.
            lookback_days: How many past days to look back for each symbol.

        Returns:
            Summary dict::

                {
                  "symbols_processed": 50,
                  "total_fetched": 120,
                  "total_inserted": 98,
                }
        """
        total_fetched = 0
        total_inserted = 0

        for sym in symbols:
            trades = self.scrape_nse_insider(sym, lookback_days=lookback_days)
            total_fetched += len(trades)
            total_inserted += self.persist(trades)

        summary = {
            "symbols_processed": len(symbols),
            "total_fetched": total_fetched,
            "total_inserted": total_inserted,
        }
        logger.info("InsiderScraper.run_for_universe summary: %s", summary)
        return summary
