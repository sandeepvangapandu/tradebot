"""Moneycontrol / NSE earnings calendar scraper.

Scrapes upcoming and past earnings announcement dates for the Top-10 universe,
including EPS estimates, actuals, and surprise percentages.

Primary source:  Moneycontrol earnings page (HTML)
Fallback source: NSE corporate results API

Design notes
------------
- All DB writes are upserts (INSERT … ON CONFLICT DO UPDATE) — safe to re-run.
- ``run_for_universe`` is the single entry-point for the scheduler.
- Reporting time codes: BMO (Before Market Open), AMC (After Market Close),
  DURING (during market hours).
- All dates are in IST.  ``scraped_at`` is stored as UTC.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MC_EARNINGS_URL = (
    "https://www.moneycontrol.com/stocks/marketinfo/upcoming-corporate-actions/"
    "earnings/{symbol}.html"
)
_MC_RESULTS_URL = (
    "https://www.moneycontrol.com/financials/{slug}/results/quarterly-results/"
    "{slug}.html"
)
_NSE_RESULTS_API = (
    "https://www.nseindia.com/api/corporate-announcements"
    "?index=equities&symbol={symbol}&subject=Financial+Results"
)
_NSE_HOME = "https://www.nseindia.com"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_NSE_API_HEADERS = {
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

# Reporting-time keyword maps
_BMO_KEYWORDS = frozenset(
    ["before market", "before market hours", "bmo", "pre-market", "before open"]
)
_AMC_KEYWORDS = frozenset(
    ["after market", "after market hours", "amc", "post-market", "after close"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ist_today() -> date:
    """Return today's date in IST (UTC+5:30)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _parse_numeric(value: Any) -> float | None:
    """Parse a string/number to float, stripping commas and whitespace."""
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("₹", "").strip()
    if cleaned in ("", "-", "N/A", "na", "null"):
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _compute_surprise(expected: float | None, actual: float | None) -> float | None:
    """Compute EPS surprise percentage.

    Formula: (actual - expected) / abs(expected) * 100

    Args:
        expected: Expected EPS value.
        actual: Actual reported EPS value.

    Returns:
        Surprise percentage or None if either input is None or expected is 0.
    """
    if expected is None or actual is None:
        return None
    if expected == 0:
        return None
    return round((actual - expected) / abs(expected) * 100, 2)


# ---------------------------------------------------------------------------
# EarningsScraper
# ---------------------------------------------------------------------------

class EarningsScraper:
    """Orchestrates earnings calendar scraping and DB persistence.

    Args:
        db_engine: Optional SQLAlchemy engine.  When ``None`` the ``persist``
            method is a no-op (useful in unit tests).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine=None, timeout: int = 30) -> None:
        self._engine = db_engine
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Primary: Moneycontrol
    # ------------------------------------------------------------------

    def scrape_moneycontrol_earnings(
        self,
        symbol: str,
        lookback_days: int = 365,
        lookahead_days: int = 90,
    ) -> list[dict]:
        """Scrape earnings dates from Moneycontrol for a given symbol.

        Args:
            symbol: NSE trading symbol (e.g. "RELIANCE").
            lookback_days: How many days back to look for past results.
            lookahead_days: How many days ahead to look for upcoming results.

        Returns:
            List of dicts with keys: symbol, earnings_date, fiscal_quarter,
            reporting_time, expected_eps, actual_eps, surprise_pct.
        """
        today = _ist_today()
        from_date = today - timedelta(days=lookback_days)
        to_date = today + timedelta(days=lookahead_days)

        url = _MC_EARNINGS_URL.format(symbol=symbol.lower())
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=_BROWSER_HEADERS)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Moneycontrol earnings fetch failed for %s: %s", symbol, exc
            )
            return []

        return self._parse_moneycontrol_html(html, symbol, from_date, to_date)

    def _parse_moneycontrol_html(
        self,
        html: str,
        symbol: str,
        from_date: date,
        to_date: date,
    ) -> list[dict]:
        """Parse Moneycontrol earnings HTML into structured dicts.

        Args:
            html: Raw HTML page content.
            symbol: NSE trading symbol.
            from_date: Earliest earnings date to include.
            to_date: Latest earnings date to include.

        Returns:
            List of parsed earnings dicts.
        """
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []

        # Look for earnings tables — Moneycontrol uses several table patterns
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                text_cells = [c.get_text(strip=True) for c in cells]
                entry = self._try_parse_mc_row(text_cells, symbol, from_date, to_date)
                if entry is not None:
                    results.append(entry)

        # Deduplicate by earnings_date
        seen: set[date] = set()
        unique: list[dict] = []
        for r in results:
            if r["earnings_date"] not in seen:
                seen.add(r["earnings_date"])
                unique.append(r)

        logger.info(
            "Moneycontrol: parsed %d earnings entries for %s", len(unique), symbol
        )
        return unique

    def _try_parse_mc_row(
        self,
        cells: list[str],
        symbol: str,
        from_date: date,
        to_date: date,
    ) -> dict | None:
        """Attempt to parse a single table row into an earnings dict.

        Args:
            cells: List of cell text values from the HTML row.
            symbol: NSE trading symbol.
            from_date: Earliest acceptable earnings date.
            to_date: Latest acceptable earnings date.

        Returns:
            Parsed earnings dict or None if the row is not a valid earnings row.
        """
        # Try to find a date in the cells
        earnings_date: date | None = None
        date_idx = -1
        for i, cell in enumerate(cells):
            for fmt in ("%d %b %Y", "%b %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                try:
                    earnings_date = datetime.strptime(cell.strip(), fmt).date()
                    date_idx = i
                    break
                except ValueError:
                    continue
            if earnings_date is not None:
                break

        if earnings_date is None:
            return None
        if earnings_date < from_date or earnings_date > to_date:
            return None

        # Extract other fields from remaining cells
        fiscal_quarter: str | None = None
        reporting_time_raw: str | None = None
        expected_eps: float | None = None
        actual_eps: float | None = None

        for i, cell in enumerate(cells):
            if i == date_idx:
                continue
            cell_lower = cell.lower()
            # Quarter detection: Q1/Q2/Q3/Q4 FY or FQ
            if any(q in cell_lower for q in ("q1", "q2", "q3", "q4")):
                fiscal_quarter = cell.strip()
            # Reporting time
            elif any(kw in cell_lower for kw in ("market", "bmo", "amc", "open", "close")):
                reporting_time_raw = cell.strip()
            # EPS values — two separate numeric cells
            elif _parse_numeric(cell) is not None:
                val = _parse_numeric(cell)
                if expected_eps is None:
                    expected_eps = val
                elif actual_eps is None:
                    actual_eps = val

        surprise_pct = _compute_surprise(expected_eps, actual_eps)

        return {
            "symbol": symbol,
            "earnings_date": earnings_date,
            "fiscal_quarter": fiscal_quarter,
            "reporting_time": self.parse_reporting_time(reporting_time_raw or ""),
            "expected_eps": expected_eps,
            "actual_eps": actual_eps,
            "surprise_pct": surprise_pct,
            "source": "MONEYCONTROL",
        }

    # ------------------------------------------------------------------
    # Fallback: NSE corporate results
    # ------------------------------------------------------------------

    def scrape_nse_corporate_results(self, symbol: str) -> list[dict]:
        """Scrape NSE corporate results announcements as a fallback.

        Hits the NSE API for financial results announcements and parses
        the response into the same dict format as ``scrape_moneycontrol_earnings``.

        Args:
            symbol: NSE trading symbol (e.g. "RELIANCE").

        Returns:
            List of earnings dicts (may be empty on failure).
        """
        url = _NSE_RESULTS_API.format(symbol=symbol.upper())
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                # NSE requires a session cookie from the homepage
                home_resp = client.get(_NSE_HOME, headers=_BROWSER_HEADERS)
                home_resp.raise_for_status()
                cookies = dict(home_resp.cookies)

                api_resp = client.get(url, headers=_NSE_API_HEADERS, cookies=cookies)
                api_resp.raise_for_status()
                data = api_resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NSE corporate results fetch failed for %s: %s", symbol, exc
            )
            return []

        return self._parse_nse_results_json(data, symbol)

    def _parse_nse_results_json(self, data: Any, symbol: str) -> list[dict]:
        """Parse NSE corporate announcements JSON into earnings dicts.

        Args:
            data: Parsed JSON from NSE API (expected: list of announcement dicts).
            symbol: NSE trading symbol.

        Returns:
            List of parsed earnings dicts.
        """
        if not isinstance(data, list):
            logger.warning("NSE results: unexpected JSON structure for %s", symbol)
            return []

        results: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # NSE uses "bcastDate" for the announcement date
            raw_date = item.get("bcastDate") or item.get("date") or ""
            earnings_date: date | None = None
            for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                try:
                    earnings_date = datetime.strptime(raw_date.strip(), fmt).date()
                    break
                except (ValueError, AttributeError):
                    continue

            if earnings_date is None:
                continue

            desc = item.get("subject", "") or item.get("desc", "")
            fiscal_quarter = self._extract_quarter(desc)

            results.append({
                "symbol": symbol,
                "earnings_date": earnings_date,
                "fiscal_quarter": fiscal_quarter,
                "reporting_time": "DURING",  # NSE doesn't specify timing
                "expected_eps": None,
                "actual_eps": None,
                "surprise_pct": None,
                "source": "NSE",
            })

        logger.info(
            "NSE fallback: parsed %d earnings entries for %s", len(results), symbol
        )
        return results

    def _extract_quarter(self, text: str) -> str | None:
        """Extract fiscal quarter string from an announcement description.

        Args:
            text: Raw description text (e.g. "Q4 FY26 Financial Results").

        Returns:
            Quarter string like "Q4 FY26" or None.
        """
        import re
        m = re.search(r"(Q[1-4]\s*FY\d{2,4})", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"(Q[1-4])\s+(?:FY)?(\d{2,4})", text, re.IGNORECASE)
        if m2:
            return f"{m2.group(1).upper()} FY{m2.group(2)}"
        return None

    # ------------------------------------------------------------------
    # Reporting time normalisation
    # ------------------------------------------------------------------

    def parse_reporting_time(self, raw: str) -> str:
        """Normalise a raw reporting-time string to a standard code.

        Args:
            raw: Free-text reporting time (e.g. "Before Market Hours").

        Returns:
            One of 'BMO', 'AMC', or 'DURING'.
        """
        if not raw:
            return "DURING"
        lower = raw.lower().strip()
        if any(kw in lower for kw in _BMO_KEYWORDS):
            return "BMO"
        if any(kw in lower for kw in _AMC_KEYWORDS):
            return "AMC"
        return "DURING"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, earnings: list[dict]) -> int:
        """Upsert earnings records into the earnings_calendar table.

        Uses INSERT … ON CONFLICT (symbol, earnings_date) DO UPDATE so the
        call is idempotent and safe to re-run.

        Args:
            earnings: List of earnings dicts (as returned by scrape methods).

        Returns:
            Number of rows inserted or updated.
        """
        if not earnings or self._engine is None:
            return 0

        from sqlalchemy import text as sa_text

        upsert_sql = sa_text(
            """
            INSERT INTO earnings_calendar
                (symbol, earnings_date, fiscal_quarter, reporting_time,
                 expected_eps, actual_eps, surprise_pct, source)
            VALUES
                (:symbol, :earnings_date, :fiscal_quarter, :reporting_time,
                 :expected_eps, :actual_eps, :surprise_pct, :source)
            ON CONFLICT (symbol, earnings_date)
            DO UPDATE SET
                fiscal_quarter   = EXCLUDED.fiscal_quarter,
                reporting_time   = EXCLUDED.reporting_time,
                expected_eps     = COALESCE(EXCLUDED.expected_eps, earnings_calendar.expected_eps),
                actual_eps       = COALESCE(EXCLUDED.actual_eps,   earnings_calendar.actual_eps),
                surprise_pct     = COALESCE(EXCLUDED.surprise_pct, earnings_calendar.surprise_pct),
                source           = EXCLUDED.source,
                scraped_at       = NOW()
            """
        )

        count = 0
        try:
            with self._engine.begin() as conn:
                for entry in earnings:
                    conn.execute(
                        upsert_sql,
                        {
                            "symbol": entry["symbol"],
                            "earnings_date": entry["earnings_date"],
                            "fiscal_quarter": entry.get("fiscal_quarter"),
                            "reporting_time": entry.get("reporting_time", "DURING"),
                            "expected_eps": entry.get("expected_eps"),
                            "actual_eps": entry.get("actual_eps"),
                            "surprise_pct": entry.get("surprise_pct"),
                            "source": entry.get("source", "MONEYCONTROL"),
                        },
                    )
                    count += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("persist earnings failed: %s", exc)
            return 0

        logger.info("Persisted %d earnings records", count)
        return count

    # ------------------------------------------------------------------
    # Universe run
    # ------------------------------------------------------------------

    def run_for_universe(self, symbols: list[str]) -> dict:
        """Scrape and persist earnings for a list of symbols.

        For each symbol, tries Moneycontrol first; falls back to NSE if
        Moneycontrol returns nothing.

        Args:
            symbols: List of NSE trading symbols.

        Returns:
            Dict mapping symbol → number of records persisted.
        """
        summary: dict[str, int] = {}
        for symbol in symbols:
            try:
                records = self.scrape_moneycontrol_earnings(symbol)
                if not records:
                    logger.info(
                        "%s: Moneycontrol empty, trying NSE fallback", symbol
                    )
                    records = self.scrape_nse_corporate_results(symbol)
                persisted = self.persist(records)
                summary[symbol] = persisted
                logger.info("%s: %d records persisted", symbol, persisted)
            except Exception as exc:  # noqa: BLE001
                logger.error("run_for_universe failed for %s: %s", symbol, exc)
                summary[symbol] = 0
        return summary
