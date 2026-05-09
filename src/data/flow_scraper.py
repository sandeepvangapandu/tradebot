"""Scrape FII/DII cash market activity + 10Y G-Sec yield (EOD).

Primary source:  NSE India  (https://www.nseindia.com/api/fiidiiTradeReact)
Fallback source: Moneycontrol (HTML scrape)

All monetary values (buy/sell/net) are reported in ₹ crore.
The 10-year G-Sec yield is reported as a percentage (e.g. 7.05).

Design notes
------------
- NSE requires browser-like headers + a session cookie obtained from the
  homepage.  We fetch the homepage first to get the cookie, then hit the API
  endpoint.  If that fails (HTTP error, JSON parse error, or missing keys) we
  fall back to Moneycontrol HTML scraping.
- All DB writes are upserts (INSERT … ON CONFLICT DO UPDATE) so the function is
  idempotent — safe to re-run after partial failures.
- ``run_daily`` is the single entry-point to call from the scheduler.  It calls
  scrape + persist for both flows and yield, then delegates regime computation
  to :class:`~src.research.flow_regime.FlowRegimeAnalyzer`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NSE_HOME = "https://www.nseindia.com"
_NSE_FII_DII_API = "https://www.nseindia.com/api/fiidiiTradeReact"
_MC_FII_DII_URL = (
    "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
)
_INVESTING_10Y_URL = "https://in.investing.com/rates-bonds/india-10-year-bond-yield"

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

_MC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _today_date() -> date:
    """Return today's date in IST (Asia/Kolkata = UTC+5:30)."""
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()


def _parse_crore(value: Any) -> float | None:
    """Parse a value to float (crore).  Returns None on parse failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# FlowScraper
# ---------------------------------------------------------------------------

class FlowScraper:
    """Orchestrates scraping and persistence of FII/DII flows and bond yields.

    Args:
        db_engine: Optional SQLAlchemy engine for DB persistence.  When
            ``None`` the ``persist_*`` methods are no-ops (useful in tests).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine=None, timeout: int = 30) -> None:
        self._engine = db_engine
        self._timeout = timeout

    # ------------------------------------------------------------------
    # NSE scrape
    # ------------------------------------------------------------------

    def scrape_nse_fii_dii(self, trade_date: date | None = None) -> dict | None:
        """Fetch FII/DII cash market data from NSE API.

        Args:
            trade_date: The date for which to fetch data.  When ``None``
                today's date is used (IST).

        Returns:
            Dict with keys ``fii_buy``, ``fii_sell``, ``fii_net``,
            ``dii_buy``, ``dii_sell``, ``dii_net`` (all ₹ crore) or
            ``None`` on failure.
        """
        if trade_date is None:
            trade_date = _today_date()

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                # Step 1: get session cookie from homepage
                home_resp = client.get(_NSE_HOME, headers=_MC_HEADERS)
                home_resp.raise_for_status()
                cookies = dict(home_resp.cookies)

                # Step 2: hit the API with the session cookie
                api_resp = client.get(
                    _NSE_FII_DII_API,
                    headers=_NSE_HEADERS,
                    cookies=cookies,
                )
                api_resp.raise_for_status()
                data = api_resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE FII/DII fetch failed: %s", exc)
            return None

        return self._parse_nse_fii_dii_json(data, trade_date)

    def _parse_nse_fii_dii_json(
        self,
        data: list | dict,
        trade_date: date,
    ) -> dict | None:
        """Parse the NSE ``fiidiiTradeReact`` JSON response.

        The API returns a list of dicts with keys like ``category``,
        ``buyValue``, ``sellValue``, ``netValue`` — one entry per category
        (FII/FPI, DII, etc.).

        Args:
            data: Parsed JSON payload from the NSE endpoint.
            trade_date: The trading date this data corresponds to.

        Returns:
            Normalised flow dict or ``None`` if expected keys are absent.
        """
        if not isinstance(data, list):
            logger.warning("Unexpected NSE FII/DII JSON structure: %s", type(data))
            return None

        fii_row: dict | None = None
        dii_row: dict | None = None

        for row in data:
            category = str(row.get("category", "")).upper()
            if "FII" in category or "FPI" in category:
                fii_row = row
            elif "DII" in category:
                dii_row = row

        if fii_row is None or dii_row is None:
            logger.warning("Could not find FII/DII rows in NSE response")
            return None

        fii_buy = _parse_crore(fii_row.get("buyValue"))
        fii_sell = _parse_crore(fii_row.get("sellValue"))
        fii_net = _parse_crore(fii_row.get("netValue"))
        dii_buy = _parse_crore(dii_row.get("buyValue"))
        dii_sell = _parse_crore(dii_row.get("sellValue"))
        dii_net = _parse_crore(dii_row.get("netValue"))

        # Compute derived net if API omits it
        if fii_net is None and fii_buy is not None and fii_sell is not None:
            fii_net = fii_buy - fii_sell
        if dii_net is None and dii_buy is not None and dii_sell is not None:
            dii_net = dii_buy - dii_sell

        return {
            "trade_date": trade_date,
            "fii_buy": fii_buy,
            "fii_sell": fii_sell,
            "fii_net": fii_net,
            "dii_buy": dii_buy,
            "dii_sell": dii_sell,
            "dii_net": dii_net,
        }

    # ------------------------------------------------------------------
    # Moneycontrol fallback scrape
    # ------------------------------------------------------------------

    def scrape_moneycontrol_fii_dii(self, trade_date: date | None = None) -> dict | None:
        """Scrape FII/DII data from Moneycontrol (HTML fallback).

        Args:
            trade_date: The date for which to fetch data.

        Returns:
            Normalised flow dict or ``None`` on failure.
        """
        if trade_date is None:
            trade_date = _today_date()

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(_MC_FII_DII_URL, headers=_MC_HEADERS)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Moneycontrol FII/DII fetch failed: %s", exc)
            return None

        return self._parse_moneycontrol_html(html, trade_date)

    def _parse_moneycontrol_html(self, html: str, trade_date: date) -> dict | None:
        """Parse Moneycontrol FII/DII HTML page.

        Moneycontrol renders a table with rows labelled 'FII' and 'DII'.
        Columns: Date | Buy | Sell | Net.  We extract the first data row
        for each category (most recent).

        Args:
            html: Raw HTML from Moneycontrol page.
            trade_date: The date this data corresponds to.

        Returns:
            Normalised flow dict or ``None`` if table not found.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Look for table rows containing 'FII' / 'DII' text
        fii_data: dict[str, float | None] = {}
        dii_data: dict[str, float | None] = {}

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue
                label = cells[0].upper()
                if "FII" in label or "FPI" in label:
                    fii_data = self._extract_mc_row(cells)
                elif "DII" in label:
                    dii_data = self._extract_mc_row(cells)

        if not fii_data and not dii_data:
            logger.warning("Moneycontrol: could not parse FII/DII table")
            return None

        return {
            "trade_date": trade_date,
            "fii_buy": fii_data.get("buy"),
            "fii_sell": fii_data.get("sell"),
            "fii_net": fii_data.get("net"),
            "dii_buy": dii_data.get("buy"),
            "dii_sell": dii_data.get("sell"),
            "dii_net": dii_data.get("net"),
        }

    @staticmethod
    def _extract_mc_row(cells: list[str]) -> dict[str, float | None]:
        """Extract buy/sell/net from a Moneycontrol table row.

        Expected cell order: [label, date?, buy, sell, net] or [label, buy, sell, net].

        Args:
            cells: List of text strings for one ``<tr>``.

        Returns:
            Dict with ``buy``, ``sell``, ``net`` keys (float or None).
        """
        # Try to find numeric columns after label
        numerics: list[float] = []
        for cell in cells[1:]:
            val = _parse_crore(cell)
            if val is not None:
                numerics.append(val)

        if len(numerics) >= 3:
            return {"buy": numerics[0], "sell": numerics[1], "net": numerics[2]}
        if len(numerics) == 2:
            return {"buy": numerics[0], "sell": numerics[1], "net": numerics[0] - numerics[1]}
        return {}

    # ------------------------------------------------------------------
    # Bond yield scrape
    # ------------------------------------------------------------------

    def scrape_bond_yield(self, trade_date: date | None = None) -> dict | None:
        """Scrape 10-year G-Sec yield from RBI / Investing.com fallback.

        Attempts to parse the yield from Investing.com's India 10Y bond page.
        Returns yield as a percentage (e.g. 7.05 for 7.05%).

        Args:
            trade_date: The date for which to record the yield.

        Returns:
            Dict with ``yield_10y_pct`` and optionally ``change_bps`` or
            ``None`` on failure.
        """
        if trade_date is None:
            trade_date = _today_date()

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(
                    _INVESTING_10Y_URL,
                    headers=_MC_HEADERS,
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bond yield fetch failed: %s", exc)
            return None

        return self._parse_investing_yield_html(html, trade_date)

    def _parse_investing_yield_html(self, html: str, trade_date: date) -> dict | None:
        """Parse 10Y yield from Investing.com bond page.

        Looks for the main price element that Investing.com renders for bond
        pages (class ``last-price-value`` or ``text-5xl``).

        Args:
            html: Raw HTML from Investing.com.
            trade_date: The date this data corresponds to.

        Returns:
            Dict ``{trade_date, yield_10y_pct, change_bps}`` or ``None``.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Primary: data-test="instrument-price-last"
        price_el = soup.find(attrs={"data-test": "instrument-price-last"})
        if price_el:
            yield_pct = _parse_crore(price_el.get_text(strip=True))
            if yield_pct is not None:
                return {"trade_date": trade_date, "yield_10y_pct": yield_pct, "change_bps": None}

        # Fallback: look for a span with class containing 'last-price'
        for tag in soup.find_all(["span", "div"], class_=True):
            classes = " ".join(tag.get("class", []))
            if "last-price" in classes or "price-last" in classes:
                yield_pct = _parse_crore(tag.get_text(strip=True))
                if yield_pct is not None:
                    return {
                        "trade_date": trade_date,
                        "yield_10y_pct": yield_pct,
                        "change_bps": None,
                    }

        logger.warning("Could not parse 10Y yield from Investing.com HTML")
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_flows(self, trade_date: date, flows: dict, source: str) -> None:
        """Upsert FII/DII flow data into ``fii_dii_flows_daily``.

        Args:
            trade_date: The trading date.
            flows: Dict containing fii_buy, fii_sell, fii_net, dii_buy,
                dii_sell, dii_net (all ₹ crore, may be None).
            source: Data source label ('NSE'|'MONEYCONTROL'|'MANUAL').
        """
        if self._engine is None:
            logger.debug("No DB engine — skipping persist_flows")
            return

        fii_net = flows.get("fii_net")
        dii_net = flows.get("dii_net")
        net_total = None
        if fii_net is not None and dii_net is not None:
            net_total = fii_net + dii_net

        sql = """
            INSERT INTO fii_dii_flows_daily (
                trade_date, fii_buy_value_crore, fii_sell_value_crore,
                fii_net_value_crore, dii_buy_value_crore, dii_sell_value_crore,
                dii_net_value_crore, net_flow_total_crore, source, scraped_at
            ) VALUES (
                :trade_date, :fii_buy, :fii_sell, :fii_net,
                :dii_buy, :dii_sell, :dii_net, :net_total, :source, NOW()
            )
            ON CONFLICT (trade_date) DO UPDATE SET
                fii_buy_value_crore  = EXCLUDED.fii_buy_value_crore,
                fii_sell_value_crore = EXCLUDED.fii_sell_value_crore,
                fii_net_value_crore  = EXCLUDED.fii_net_value_crore,
                dii_buy_value_crore  = EXCLUDED.dii_buy_value_crore,
                dii_sell_value_crore = EXCLUDED.dii_sell_value_crore,
                dii_net_value_crore  = EXCLUDED.dii_net_value_crore,
                net_flow_total_crore = EXCLUDED.net_flow_total_crore,
                source               = EXCLUDED.source,
                scraped_at           = EXCLUDED.scraped_at
        """
        from sqlalchemy import text as sa_text

        with self._engine.begin() as conn:
            conn.execute(
                sa_text(sql),
                {
                    "trade_date": trade_date,
                    "fii_buy": flows.get("fii_buy"),
                    "fii_sell": flows.get("fii_sell"),
                    "fii_net": fii_net,
                    "dii_buy": flows.get("dii_buy"),
                    "dii_sell": flows.get("dii_sell"),
                    "dii_net": dii_net,
                    "net_total": net_total,
                    "source": source,
                },
            )
        logger.info("Persisted FII/DII flows for %s (source=%s)", trade_date, source)

    def persist_yield(self, trade_date: date, yield_data: dict, source: str) -> None:
        """Upsert 10Y G-Sec yield into ``bond_yield_daily``.

        Computes ``change_bps`` by fetching the previous day's yield from the
        DB.  Computes ``trend_5d`` from the last 5 trading days.

        Args:
            trade_date: The trading date.
            yield_data: Dict with ``yield_10y_pct`` and optional ``change_bps``.
            source: Data source label.
        """
        if self._engine is None:
            logger.debug("No DB engine — skipping persist_yield")
            return

        from sqlalchemy import text as sa_text

        yield_pct: float = yield_data["yield_10y_pct"]
        change_bps = yield_data.get("change_bps")
        trend_5d: str | None = None

        with self._engine.begin() as conn:
            # Compute change_bps from prev day if not supplied
            if change_bps is None:
                prev_row = conn.execute(
                    sa_text(
                        "SELECT yield_10y_pct FROM bond_yield_daily "
                        "WHERE trade_date < :d ORDER BY trade_date DESC LIMIT 1"
                    ),
                    {"d": trade_date},
                ).fetchone()
                if prev_row:
                    change_bps = round((yield_pct - float(prev_row[0])) * 100, 2)

            # Compute 5-day trend from last 5 records + today
            hist = conn.execute(
                sa_text(
                    "SELECT yield_10y_pct FROM bond_yield_daily "
                    "WHERE trade_date < :d ORDER BY trade_date DESC LIMIT 5"
                ),
                {"d": trade_date},
            ).fetchall()
            if len(hist) >= 3:
                oldest_yield = float(hist[-1][0])
                diff = yield_pct - oldest_yield
                if diff > 0.05:
                    trend_5d = "RISING"
                elif diff < -0.05:
                    trend_5d = "FALLING"
                else:
                    trend_5d = "FLAT"

            conn.execute(
                sa_text(
                    """
                    INSERT INTO bond_yield_daily
                        (trade_date, yield_10y_pct, change_bps, trend_5d, source, scraped_at)
                    VALUES (:d, :y, :cb, :t5, :src, NOW())
                    ON CONFLICT (trade_date) DO UPDATE SET
                        yield_10y_pct = EXCLUDED.yield_10y_pct,
                        change_bps    = EXCLUDED.change_bps,
                        trend_5d      = EXCLUDED.trend_5d,
                        source        = EXCLUDED.source,
                        scraped_at    = EXCLUDED.scraped_at
                    """
                ),
                {"d": trade_date, "y": yield_pct, "cb": change_bps, "t5": trend_5d, "src": source},
            )
        logger.info("Persisted bond yield for %s: %.2f%% (source=%s)", trade_date, yield_pct, source)

    # ------------------------------------------------------------------
    # Daily orchestration
    # ------------------------------------------------------------------

    def run_daily(self, trade_date: date | None = None) -> dict:
        """Run the full EOD scrape + persist pipeline.

        Order of operations:
        1. Try NSE for FII/DII flows; fall back to Moneycontrol on failure.
        2. Scrape 10Y G-Sec yield.
        3. Persist flows and yield to DB.
        4. Trigger flow regime computation via FlowRegimeAnalyzer.

        Args:
            trade_date: The date to process.  Defaults to today (IST).

        Returns:
            Summary dict with keys ``flows_source``, ``flows_ok``,
            ``yield_ok``, ``regime``.
        """
        if trade_date is None:
            trade_date = _today_date()

        summary: dict = {
            "trade_date": str(trade_date),
            "flows_source": None,
            "flows_ok": False,
            "yield_ok": False,
            "regime": None,
        }

        # --- FII/DII flows ---
        flows = self.scrape_nse_fii_dii(trade_date)
        flows_source = "NSE"
        if flows is None:
            logger.info("NSE source failed — trying Moneycontrol fallback")
            flows = self.scrape_moneycontrol_fii_dii(trade_date)
            flows_source = "MONEYCONTROL"

        if flows is not None:
            self.persist_flows(trade_date, flows, source=flows_source)
            summary["flows_source"] = flows_source
            summary["flows_ok"] = True
        else:
            logger.error("All FII/DII sources failed for %s", trade_date)

        # --- Bond yield ---
        yield_data = self.scrape_bond_yield(trade_date)
        if yield_data is not None:
            self.persist_yield(trade_date, yield_data, source="INVESTING_COM")
            summary["yield_ok"] = True
        else:
            logger.error("Bond yield scrape failed for %s", trade_date)

        # --- Flow regime ---
        if summary["flows_ok"] and self._engine is not None:
            try:
                from src.research.flow_regime import FlowRegimeAnalyzer

                analyzer = FlowRegimeAnalyzer(db_engine=self._engine)
                regime = analyzer.compute_regime(trade_date)
                summary["regime"] = regime
            except Exception as exc:  # noqa: BLE001
                logger.warning("Flow regime computation failed: %s", exc)

        return summary
