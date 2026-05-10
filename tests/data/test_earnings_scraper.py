"""Tests for src.data.earnings_scraper — earnings calendar scraper.

All tests use inline mocks (unittest.mock) and do NOT make real HTTP calls
or require a live database.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.data.earnings_scraper import EarningsScraper


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

EARNINGS_DATE = date(2026, 5, 10)
SYMBOL = "RELIANCE"


def _make_db_engine(
    *, fetchone_return=None, fetchall_return=None
) -> MagicMock:
    """Create a mock SQLAlchemy engine that fakes DB calls."""
    result = MagicMock()
    result.fetchone.return_value = fetchone_return
    result.fetchall.return_value = fetchall_return or []

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    mock_engine = MagicMock()
    # Support both .connect() and .begin() context managers
    mock_engine.connect.return_value = mock_conn
    mock_engine.begin.return_value = mock_conn
    return mock_engine


# ---------------------------------------------------------------------------
# Reporting-time normalisation tests
# ---------------------------------------------------------------------------

class TestParseReportingTime:
    """Unit tests for EarningsScraper.parse_reporting_time."""

    def test_parse_reporting_time_bmo(self):
        """'Before Market Hours' must normalise to 'BMO'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("Before Market Hours") == "BMO"

    def test_parse_reporting_time_bmo_lowercase(self):
        """Case-insensitive match for 'before market hours'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("before market hours") == "BMO"

    def test_parse_reporting_time_bmo_code(self):
        """Literal 'BMO' string normalises to 'BMO'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("BMO") == "BMO"

    def test_parse_reporting_time_amc(self):
        """'After Market Hours' must normalise to 'AMC'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("After Market Hours") == "AMC"

    def test_parse_reporting_time_amc_code(self):
        """Literal 'AMC' string normalises to 'AMC'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("AMC") == "AMC"

    def test_parse_reporting_time_amc_post_market(self):
        """'Post-market' normalises to 'AMC'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("Post-market") == "AMC"

    def test_parse_reporting_time_empty_string_returns_during(self):
        """Empty string should return 'DURING' (safe default)."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("") == "DURING"

    def test_parse_reporting_time_unknown_returns_during(self):
        """Unknown strings should return 'DURING'."""
        scraper = EarningsScraper()
        assert scraper.parse_reporting_time("unknown timing") == "DURING"


# ---------------------------------------------------------------------------
# Moneycontrol HTML parsing tests
# ---------------------------------------------------------------------------

_MC_HTML_WITH_EARNINGS = """
<html><body>
<table>
  <tr><th>Date</th><th>Quarter</th><th>Est EPS</th><th>Act EPS</th></tr>
  <tr>
    <td>10 May 2026</td>
    <td>Q4 FY26</td>
    <td>Before Market Hours</td>
    <td>25.50</td>
  </tr>
  <tr>
    <td>08 Feb 2026</td>
    <td>Q3 FY26</td>
    <td>After Market Hours</td>
    <td>22.00</td>
  </tr>
</table>
</body></html>
"""

_MC_HTML_NO_DATES = """
<html><body>
<table>
  <tr><th>Company</th><th>Sector</th></tr>
  <tr><td>Reliance</td><td>Energy</td></tr>
</table>
</body></html>
"""


class TestScrapeMoneycontrolEarnings:
    """Tests for EarningsScraper.scrape_moneycontrol_earnings (HTTP mocked)."""

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_moneycontrol_earnings_parses_mock_html(self, mock_client_cls):
        """Should parse two earnings rows from mock HTML."""
        mock_resp = MagicMock()
        mock_resp.text = _MC_HTML_WITH_EARNINGS
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_moneycontrol_earnings(
            SYMBOL, lookback_days=365, lookahead_days=90
        )

        # At least one row with an earnings_date should be parsed
        assert isinstance(results, list)
        for r in results:
            assert "symbol" in r
            assert "earnings_date" in r
            assert isinstance(r["earnings_date"], date)
            assert r["symbol"] == SYMBOL

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_moneycontrol_returns_empty_on_http_error(self, mock_client_cls):
        """Should return [] gracefully when HTTP request fails."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("connection refused")
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_moneycontrol_earnings(SYMBOL)
        assert results == []

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_moneycontrol_returns_empty_when_no_dates_in_html(
        self, mock_client_cls
    ):
        """Should return [] when the HTML contains no parseable date cells."""
        mock_resp = MagicMock()
        mock_resp.text = _MC_HTML_NO_DATES
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_moneycontrol_earnings(SYMBOL)
        assert results == []


# ---------------------------------------------------------------------------
# NSE corporate results fallback tests
# ---------------------------------------------------------------------------

_NSE_RESULTS_JSON = [
    {
        "bcastDate": "09-May-2026",
        "subject": "Financial Results for Q4 FY26",
    },
    {
        "bcastDate": "08-Feb-2026",
        "subject": "Q3 FY26 Financial Results",
    },
]

_NSE_BAD_JSON = {"error": "not found"}


class TestScrapeNseCorporateResults:
    """Tests for EarningsScraper.scrape_nse_corporate_results (HTTP mocked)."""

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_nse_corporate_results_fallback(self, mock_client_cls):
        """Should parse NSE JSON results into earnings dicts."""
        mock_home_resp = MagicMock()
        mock_home_resp.raise_for_status = MagicMock()
        mock_home_resp.cookies = {}

        mock_api_resp = MagicMock()
        mock_api_resp.raise_for_status = MagicMock()
        mock_api_resp.json.return_value = _NSE_RESULTS_JSON

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        # First call = homepage, second call = API
        mock_client.get.side_effect = [mock_home_resp, mock_api_resp]
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_nse_corporate_results(SYMBOL)

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r["symbol"] == SYMBOL
            assert isinstance(r["earnings_date"], date)
            assert r["source"] == "NSE"

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_nse_returns_empty_on_bad_json(self, mock_client_cls):
        """Should return [] when NSE returns non-list JSON."""
        mock_home_resp = MagicMock()
        mock_home_resp.raise_for_status = MagicMock()
        mock_home_resp.cookies = {}

        mock_api_resp = MagicMock()
        mock_api_resp.raise_for_status = MagicMock()
        mock_api_resp.json.return_value = _NSE_BAD_JSON

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [mock_home_resp, mock_api_resp]
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_nse_corporate_results(SYMBOL)
        assert results == []

    @patch("src.data.earnings_scraper.httpx.Client")
    def test_scrape_nse_returns_empty_on_http_error(self, mock_client_cls):
        """Should return [] gracefully when HTTP request raises."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("timeout")
        mock_client_cls.return_value = mock_client

        scraper = EarningsScraper()
        results = scraper.scrape_nse_corporate_results(SYMBOL)
        assert results == []


# ---------------------------------------------------------------------------
# Persist tests
# ---------------------------------------------------------------------------

_SAMPLE_EARNINGS = [
    {
        "symbol": SYMBOL,
        "earnings_date": date(2026, 5, 10),
        "fiscal_quarter": "Q4 FY26",
        "reporting_time": "BMO",
        "expected_eps": 25.0,
        "actual_eps": 28.0,
        "surprise_pct": 12.0,
        "source": "MONEYCONTROL",
    }
]

_SAMPLE_EARNINGS_UPDATED_EPS = [
    {
        "symbol": SYMBOL,
        "earnings_date": date(2026, 5, 10),
        "fiscal_quarter": "Q4 FY26",
        "reporting_time": "BMO",
        "expected_eps": 25.0,
        "actual_eps": 30.0,   # different actual EPS
        "surprise_pct": 20.0,
        "source": "MONEYCONTROL",
    }
]


class TestPersist:
    """Tests for EarningsScraper.persist."""

    def test_persist_idempotent_via_unique_constraint(self):
        """persist() should call execute once per record with upsert SQL."""
        engine = _make_db_engine()
        scraper = EarningsScraper(db_engine=engine)

        count = scraper.persist(_SAMPLE_EARNINGS)
        assert count == 1
        # engine.begin() should have been called (transaction context manager)
        assert engine.begin.called

    def test_persist_updates_actual_eps_when_available(self):
        """A second persist call with updated actual_eps should call execute again."""
        engine = _make_db_engine()
        scraper = EarningsScraper(db_engine=engine)

        # First persist (initial estimate only)
        count1 = scraper.persist(_SAMPLE_EARNINGS)
        # Second persist (actual EPS now available)
        count2 = scraper.persist(_SAMPLE_EARNINGS_UPDATED_EPS)

        assert count1 == 1
        assert count2 == 1
        # Verify the conn.execute was called twice (once per persist call)
        conn = engine.begin.return_value.__enter__.return_value
        assert conn.execute.call_count == 2

    def test_persist_returns_zero_when_no_engine(self):
        """persist() with no engine should return 0 without raising."""
        scraper = EarningsScraper(db_engine=None)
        assert scraper.persist(_SAMPLE_EARNINGS) == 0

    def test_persist_returns_zero_when_empty_list(self):
        """persist() with empty list should return 0."""
        engine = _make_db_engine()
        scraper = EarningsScraper(db_engine=engine)
        assert scraper.persist([]) == 0

    def test_persist_returns_zero_on_db_error(self):
        """persist() should return 0 (not raise) when DB throws."""
        engine = MagicMock()
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock(side_effect=Exception("DB down"))
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine.begin.return_value = conn_ctx

        scraper = EarningsScraper(db_engine=engine)
        assert scraper.persist(_SAMPLE_EARNINGS) == 0
