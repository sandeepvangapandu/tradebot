"""NSE corporate actions scraper (dividends/splits/bonus/etc).

Fetches corporate action events (ex-date, record date, action type, ratio)
for a given symbol universe from NSE's corporateActions API endpoint and
persists them to the local database for use in blackout filtering.

Endpoint reference:
    GET https://www.nseindia.com/api/corporates-corporateActions
        ?index=equities&symbol=<symbol>
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------
TOP_10_SYMBOLS: list[str] = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "SBIN",
    "BAJFINANCE",
    "BHARTIARTL",
    "KOTAKBANK",
]

INDEX_SYMBOLS: list[str] = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

DEFAULT_UNIVERSE: list[str] = TOP_10_SYMBOLS + INDEX_SYMBOLS

# ---------------------------------------------------------------------------
# NSE headers (browser-like; required to avoid 403)
# ---------------------------------------------------------------------------
_NSE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

_NSE_BASE_URL = "https://www.nseindia.com"
_CORP_ACTIONS_API = (
    _NSE_BASE_URL + "/api/corporates-corporateActions?index=equities&symbol={symbol}"
)


class CorporateActionsScraper:
    """Scrapes and persists NSE corporate action events.

    Args:
        db_engine: SQLAlchemy engine or None (skip persistence).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine: Any = None, timeout: int = 30) -> None:
        self.db_engine = db_engine
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_nse_corporate_actions(
        self,
        symbol: str,
        lookback_days: int = 90,
        lookahead_days: int = 90,
    ) -> list[dict]:
        """Fetch corporate actions for *symbol* from NSE.

        Args:
            symbol: NSE trading symbol (e.g. "RELIANCE").
            lookback_days: How many days back from today to include.
            lookahead_days: How many days ahead from today to include.

        Returns:
            List of dicts each with keys:
            ``symbol``, ``action_type``, ``ex_date``, ``record_date``,
            ``details``, ``ratio``, ``source``.
        """
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")

        url = _CORP_ACTIONS_API.format(symbol=symbol.upper())
        today = date.today()
        from_date = today - timedelta(days=lookback_days)
        to_date = today + timedelta(days=lookahead_days)

        try:
            raw_items = self._fetch_json(url)
        except Exception as exc:
            logger.warning(
                "CorporateActionsScraper: fetch failed for {}: {}", symbol, exc
            )
            return []

        results: list[dict] = []
        for item in raw_items:
            parsed = self._parse_item(symbol, item, from_date, to_date)
            if parsed:
                results.append(parsed)

        logger.info(
            "CorporateActionsScraper: {} → {} action(s) in window", symbol, len(results)
        )
        return results

    def parse_action_type(self, raw: str) -> tuple[str, float | None]:
        """Classify a raw NSE action description string.

        Args:
            raw: Raw NSE description e.g. ``"Dividend - Rs.5.00 Per Share"``.

        Returns:
            Tuple ``(action_type, ratio)`` where *action_type* is one of
            ``DIVIDEND``, ``SPLIT``, ``BONUS``, ``RIGHTS``, ``MERGER``,
            ``BUYBACK``, ``OTHER`` and *ratio* is a numeric value or ``None``.

        Examples:
            >>> s = CorporateActionsScraper()
            >>> s.parse_action_type("Dividend - Rs.5.00 Per Share")
            ('DIVIDEND', 5.0)
            >>> s.parse_action_type("Stock Split - From Rs.10/- To Rs.5/-")
            ('SPLIT', 0.5)
            >>> s.parse_action_type("Bonus 1:1")
            ('BONUS', 1.0)
        """
        if not raw:
            return ("OTHER", None)

        text = raw.strip().upper()

        # --- SPLIT (check before DIVIDEND — split text may also contain "Rs.") ---
        if "SPLIT" in text or "SUB-DIVISION" in text or "SUBDIVISION" in text:
            ratio = self._extract_ratio(raw)
            return ("SPLIT", ratio)

        # --- BONUS (check before DIVIDEND for same reason) ---
        if "BONUS" in text:
            ratio = self._extract_ratio(raw)
            return ("BONUS", ratio)

        # --- DIVIDEND ---
        if "DIVIDEND" in text or "DIV" in text:
            ratio = self._extract_rupee_amount(raw)
            return ("DIVIDEND", ratio)

        # --- RIGHTS ---
        if "RIGHTS" in text or "RIGHT ISSUE" in text:
            ratio = self._extract_ratio(raw)
            return ("RIGHTS", ratio)

        # --- MERGER / AMALGAMATION ---
        if "MERGER" in text or "AMALGAMATION" in text or "SCHEME" in text:
            return ("MERGER", None)

        # --- BUYBACK ---
        if "BUYBACK" in text or "BUY BACK" in text or "BUY-BACK" in text:
            ratio = self._extract_rupee_amount(raw)
            return ("BUYBACK", ratio)

        return ("OTHER", None)

    def persist(self, actions: list[dict]) -> int:
        """Bulk-upsert corporate action records.

        Args:
            actions: List of action dicts as returned by
                :meth:`scrape_nse_corporate_actions`.

        Returns:
            Number of rows inserted or updated.
        """
        if not actions:
            return 0
        if self.db_engine is None:
            logger.debug("CorporateActionsScraper.persist: no db_engine, skipping")
            return 0

        try:
            from sqlalchemy import text  # local import to keep optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sqlalchemy required for persistence") from exc

        upsert_sql = text(
            """
            INSERT INTO corporate_actions
                (symbol, instrument_key, action_type, ex_date, record_date,
                 details, ratio, source)
            VALUES
                (:symbol, :instrument_key, :action_type, :ex_date, :record_date,
                 :details, :ratio, :source)
            ON CONFLICT (symbol, action_type, ex_date) DO UPDATE SET
                record_date   = EXCLUDED.record_date,
                details       = EXCLUDED.details,
                ratio         = EXCLUDED.ratio,
                instrument_key= EXCLUDED.instrument_key,
                scraped_at    = NOW()
            """
        )

        count = 0
        with self.db_engine.begin() as conn:
            for action in actions:
                conn.execute(upsert_sql, self._to_row(action))
                count += 1

        logger.info("CorporateActionsScraper.persist: upserted {} row(s)", count)
        return count

    def run_for_universe(self, symbols: list[str] | None = None) -> dict[str, int]:
        """Scrape and persist actions for every symbol in the universe.

        Args:
            symbols: List of NSE symbols. Defaults to
                :data:`DEFAULT_UNIVERSE` (Top-10 + indices).

        Returns:
            Dict mapping each symbol to the count of upserted rows.
        """
        if symbols is None:
            symbols = DEFAULT_UNIVERSE

        results: dict[str, int] = {}
        for symbol in symbols:
            try:
                actions = self.scrape_nse_corporate_actions(symbol)
                count = self.persist(actions)
                results[symbol] = count
            except Exception as exc:
                logger.error(
                    "CorporateActionsScraper.run_for_universe: error for {}: {}",
                    symbol,
                    exc,
                )
                results[symbol] = 0

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_json(self, url: str) -> list[dict]:
        """Perform a GET request with NSE cookies and return parsed JSON.

        NSE requires a valid session cookie; we obtain it by visiting
        the homepage first.

        Args:
            url: Full API URL.

        Returns:
            Parsed JSON list from NSE response.
        """
        with httpx.Client(
            headers=_NSE_HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            # warm-up: acquire session cookie
            client.get(_NSE_BASE_URL, timeout=self.timeout)
            resp = client.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

        # NSE wraps data in {"data": [...]} or returns a bare list
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return []

    def _parse_item(
        self,
        symbol: str,
        item: dict,
        from_date: date,
        to_date: date,
    ) -> dict | None:
        """Parse a single NSE JSON item into an action dict.

        Args:
            symbol: Trading symbol.
            item: Raw dict from NSE API.
            from_date: Earliest ex_date to include.
            to_date: Latest ex_date to include.

        Returns:
            Normalised action dict or ``None`` if outside window or unparseable.
        """
        raw_purpose = item.get("purpose") or item.get("subject") or ""
        raw_ex_date = item.get("exDate") or item.get("ex_date") or ""
        raw_rec_date = item.get("recordDate") or item.get("record_date") or ""

        if not raw_ex_date:
            return None

        try:
            ex_date = self._parse_date(raw_ex_date)
        except ValueError:
            logger.debug("Unparseable ex_date '{}' for {}", raw_ex_date, symbol)
            return None

        if not (from_date <= ex_date <= to_date):
            return None

        try:
            record_date = self._parse_date(raw_rec_date) if raw_rec_date else None
        except ValueError:
            record_date = None

        action_type, ratio = self.parse_action_type(raw_purpose)

        return {
            "symbol": symbol.upper(),
            "instrument_key": item.get("instrumentKey") or item.get("instrument_key"),
            "action_type": action_type,
            "ex_date": ex_date,
            "record_date": record_date,
            "details": raw_purpose or None,
            "ratio": ratio,
            "source": "NSE",
        }

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(raw: str) -> date:
        """Parse a date string in common NSE formats.

        Args:
            raw: Date string e.g. ``"10-May-2026"`` or ``"2026-05-10"``.

        Returns:
            ``datetime.date`` object.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        raw = raw.strip()
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {raw!r}")

    @staticmethod
    def _extract_rupee_amount(text: str) -> float | None:
        """Extract a rupee / paisa amount from an action description.

        Args:
            text: Raw description string.

        Returns:
            Float amount or ``None`` if not found.
        """
        # Match patterns like: Rs.5.00, Rs 5, ₹5.50, Rs.-5, 5.00 Per Share
        patterns = [
            r"(?:Rs\.?|₹)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:per share|\/share)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    @staticmethod
    def _extract_ratio(text: str) -> float | None:
        """Extract a numeric ratio from split/bonus/rights descriptions.

        Handles formats such as ``"1:2"``, ``"2:1"``, ``"From Rs.10 To Rs.5"``.

        Args:
            text: Raw description string.

        Returns:
            Ratio as float (numerator / denominator) or ``None``.
        """
        # "N:M" format → N / M
        m = re.search(r"(\d+)\s*:\s*(\d+)", text)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den == 0:
                return None
            return round(num / den, 6)

        # "From Rs.X/- To Rs.Y/-" split description
        m = re.search(r"from\s+Rs\.?\s*(\d+).+?to\s+Rs\.?\s*(\d+)", text, re.IGNORECASE)
        if m:
            old_fv, new_fv = int(m.group(1)), int(m.group(2))
            if old_fv == 0:
                return None
            return round(new_fv / old_fv, 6)

        return None

    @staticmethod
    def _to_row(action: dict) -> dict:
        """Convert an action dict to a flat DB row dict.

        Args:
            action: Action dict with date objects.

        Returns:
            Dict with string-serialised dates suitable for SQL binding.
        """
        return {
            "symbol": action["symbol"],
            "instrument_key": action.get("instrument_key"),
            "action_type": action["action_type"],
            "ex_date": action["ex_date"].isoformat() if isinstance(action["ex_date"], date) else action["ex_date"],
            "record_date": (
                action["record_date"].isoformat()
                if isinstance(action.get("record_date"), date)
                else action.get("record_date")
            ),
            "details": action.get("details"),
            "ratio": action.get("ratio"),
            "source": action.get("source", "NSE"),
        }
