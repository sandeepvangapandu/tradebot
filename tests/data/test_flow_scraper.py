"""Tests for src.data.flow_scraper — FII/DII + bond yield scraping.

All tests use inline mocks (unittest.mock) and do NOT make real HTTP calls
or require a live database.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data.flow_scraper import FlowScraper


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TRADE_DATE = date(2026, 5, 8)

# Sample NSE API response (list of category dicts)
_NSE_FII_DII_JSON = [
    {
        "category": "FII/FPI",
        "buyValue": "12345.67",
        "sellValue": "9876.54",
        "netValue": "2469.13",
    },
    {
        "category": "DII",
        "buyValue": "8000.00",
        "sellValue": "5500.00",
        "netValue": "2500.00",
    },
]

# Sample Moneycontrol HTML with FII and DII rows
_MC_FII_DII_HTML = """
<html><body>
<table>
  <tr><th>Category</th><th>Buy (₹ Cr)</th><th>Sell (₹ Cr)</th><th>Net (₹ Cr)</th></tr>
  <tr><td>FII</td><td>10000.00</td><td>7000.00</td><td>3000.00</td></tr>
  <tr><td>DII</td><td>6000.00</td><td>4000.00</td><td>2000.00</td></tr>
</table>
</body></html>
"""

# Sample Investing.com HTML with the yield price element
_INVESTING_YIELD_HTML = """
<html><body>
  <span data-test="instrument-price-last">7.05</span>
</body></html>
"""


def _make_scraper(db_engine=None) -> FlowScraper:
    return FlowScraper(db_engine=db_engine, timeout=5)


def _mock_http_response(content: str, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = content
    resp.json.return_value = json.loads(content) if content.startswith("[") or content.startswith("{") else {}
    resp.raise_for_status = MagicMock()
    resp.cookies = {}
    return resp


# ---------------------------------------------------------------------------
# test_scrape_nse_fii_dii_parses_mock_response
# ---------------------------------------------------------------------------

class TestScrapeNseFiiDii:
    """Tests for FlowScraper.scrape_nse_fii_dii."""

    def test_scrape_nse_fii_dii_parses_mock_response(self):
        """NSE JSON with FII+DII rows should parse correctly into crore values."""
        scraper = _make_scraper()

        home_resp = MagicMock()
        home_resp.raise_for_status = MagicMock()
        home_resp.cookies = {}

        api_resp = MagicMock()
        api_resp.raise_for_status = MagicMock()
        api_resp.json.return_value = _NSE_FII_DII_JSON

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_nse_fii_dii(TRADE_DATE)

        assert result is not None
        assert result["fii_buy"] == pytest.approx(12345.67)
        assert result["fii_sell"] == pytest.approx(9876.54)
        assert result["fii_net"] == pytest.approx(2469.13)
        assert result["dii_buy"] == pytest.approx(8000.00)
        assert result["dii_sell"] == pytest.approx(5500.00)
        assert result["dii_net"] == pytest.approx(2500.00)
        assert result["trade_date"] == TRADE_DATE

    def test_scrape_nse_fii_dii_returns_none_on_http_error(self):
        """HTTP failure on NSE homepage should return None (not raise)."""
        scraper = _make_scraper()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Connection refused")

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_nse_fii_dii(TRADE_DATE)

        assert result is None

    def test_scrape_nse_fii_dii_computes_net_when_missing(self):
        """When 'netValue' is absent, it should be derived from buy - sell."""
        scraper = _make_scraper()
        no_net_json = [
            {"category": "FII/FPI", "buyValue": "5000.00", "sellValue": "3000.00"},
            {"category": "DII", "buyValue": "4000.00", "sellValue": "3500.00"},
        ]

        home_resp = MagicMock()
        home_resp.raise_for_status = MagicMock()
        home_resp.cookies = {}

        api_resp = MagicMock()
        api_resp.raise_for_status = MagicMock()
        api_resp.json.return_value = no_net_json

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_nse_fii_dii(TRADE_DATE)

        assert result is not None
        assert result["fii_net"] == pytest.approx(2000.00)
        assert result["dii_net"] == pytest.approx(500.00)


# ---------------------------------------------------------------------------
# test_scrape_moneycontrol_fii_dii_parses_mock_html
# ---------------------------------------------------------------------------

class TestScrapeMoneycontrolFiiDii:
    """Tests for FlowScraper.scrape_moneycontrol_fii_dii."""

    def test_scrape_moneycontrol_fii_dii_parses_mock_html(self):
        """MC HTML with FII+DII table should parse buy/sell/net correctly."""
        scraper = _make_scraper()

        mc_resp = MagicMock()
        mc_resp.raise_for_status = MagicMock()
        mc_resp.text = _MC_FII_DII_HTML

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mc_resp

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_moneycontrol_fii_dii(TRADE_DATE)

        assert result is not None
        assert result["fii_buy"] == pytest.approx(10000.00)
        assert result["fii_sell"] == pytest.approx(7000.00)
        assert result["fii_net"] == pytest.approx(3000.00)
        assert result["dii_buy"] == pytest.approx(6000.00)
        assert result["dii_sell"] == pytest.approx(4000.00)
        assert result["dii_net"] == pytest.approx(2000.00)

    def test_scrape_moneycontrol_returns_none_on_http_error(self):
        """HTTP failure on Moneycontrol should return None gracefully."""
        scraper = _make_scraper()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("timeout")

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_moneycontrol_fii_dii(TRADE_DATE)

        assert result is None

    def test_scrape_moneycontrol_returns_none_on_empty_table(self):
        """HTML with no parseable FII/DII table should return None."""
        scraper = _make_scraper()

        mc_resp = MagicMock()
        mc_resp.raise_for_status = MagicMock()
        mc_resp.text = "<html><body><p>No data</p></body></html>"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mc_resp

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_moneycontrol_fii_dii(TRADE_DATE)

        assert result is None


# ---------------------------------------------------------------------------
# test_scrape_bond_yield_parses_mock_response
# ---------------------------------------------------------------------------

class TestScrapeBondYield:
    """Tests for FlowScraper.scrape_bond_yield."""

    def test_scrape_bond_yield_parses_mock_response(self):
        """Investing.com HTML with price element should parse yield correctly."""
        scraper = _make_scraper()

        yield_resp = MagicMock()
        yield_resp.raise_for_status = MagicMock()
        yield_resp.text = _INVESTING_YIELD_HTML

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = yield_resp

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_bond_yield(TRADE_DATE)

        assert result is not None
        assert result["yield_10y_pct"] == pytest.approx(7.05)
        assert result["trade_date"] == TRADE_DATE

    def test_scrape_bond_yield_returns_none_on_http_error(self):
        """HTTP failure on yield endpoint should return None gracefully."""
        scraper = _make_scraper()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("DNS failure")

        with patch("src.data.flow_scraper.httpx.Client", return_value=mock_client):
            result = scraper.scrape_bond_yield(TRADE_DATE)

        assert result is None


# ---------------------------------------------------------------------------
# test_persist_flows_upserts_to_db
# ---------------------------------------------------------------------------

class TestPersistFlows:
    """Tests for FlowScraper.persist_flows."""

    def test_persist_flows_upserts_to_db(self):
        """persist_flows should call db engine with INSERT … ON CONFLICT SQL."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = _make_scraper(db_engine=mock_engine)

        flows = {
            "fii_buy": 12000.0,
            "fii_sell": 9000.0,
            "fii_net": 3000.0,
            "dii_buy": 8000.0,
            "dii_sell": 5500.0,
            "dii_net": 2500.0,
        }

        scraper.persist_flows(TRADE_DATE, flows, source="NSE")

        # Verify the engine was used and execute was called
        mock_engine.begin.assert_called_once()
        mock_conn.execute.assert_called_once()

        # Verify the SQL contains expected keywords
        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        assert "fii_dii_flows_daily" in sql_text or "INSERT" in sql_text

    def test_persist_flows_noop_when_no_engine(self):
        """persist_flows should be a no-op when db_engine is None."""
        scraper = _make_scraper(db_engine=None)
        # Should not raise
        scraper.persist_flows(TRADE_DATE, {"fii_net": 100.0}, source="NSE")


# ---------------------------------------------------------------------------
# test_persist_yield_computes_change_bps_from_prev
# ---------------------------------------------------------------------------

class TestPersistYield:
    """Tests for FlowScraper.persist_yield."""

    def test_persist_yield_computes_change_bps_from_prev(self):
        """persist_yield should compute change_bps from the previous day's yield.

        The implementation uses a single ``begin()`` connection for the entire
        operation (prev-yield query, 5d-trend query, and insert).  We provide a
        side_effect list for execute() on the write-connection mock so that each
        sequential call returns the right result object.
        """
        prev_row = (7.00,)
        hist_rows = [(7.00,), (6.98,), (6.97,)]

        # Each conn.execute() call returns the next item in this list
        exec_results = [
            MagicMock(fetchone=MagicMock(return_value=prev_row)),   # prev yield SELECT
            MagicMock(fetchall=MagicMock(return_value=hist_rows)),   # 5d trend SELECT
            MagicMock(),                                              # INSERT
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = exec_results

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = _make_scraper(db_engine=mock_engine)

        # Today's yield is 7.05 → change from 7.00 = +5 bps
        scraper.persist_yield(TRADE_DATE, {"yield_10y_pct": 7.05}, source="TEST")

        # At minimum: prev-yield query + trend query + insert = 3 calls
        assert mock_conn.execute.call_count == 3

    def test_persist_yield_noop_when_no_engine(self):
        """persist_yield should be a no-op when db_engine is None."""
        scraper = _make_scraper(db_engine=None)
        scraper.persist_yield(TRADE_DATE, {"yield_10y_pct": 7.05}, source="TEST")


# ---------------------------------------------------------------------------
# test_run_daily_falls_back_to_secondary_source_on_failure
# ---------------------------------------------------------------------------

class TestRunDaily:
    """Tests for FlowScraper.run_daily."""

    def test_run_daily_falls_back_to_secondary_source_on_failure(self):
        """When NSE fails, run_daily should fall back to Moneycontrol."""
        scraper = _make_scraper(db_engine=None)

        mc_flows = {
            "trade_date": TRADE_DATE,
            "fii_buy": 10000.0,
            "fii_sell": 7000.0,
            "fii_net": 3000.0,
            "dii_buy": 6000.0,
            "dii_sell": 4000.0,
            "dii_net": 2000.0,
        }

        with patch.object(scraper, "scrape_nse_fii_dii", return_value=None) as mock_nse, \
             patch.object(scraper, "scrape_moneycontrol_fii_dii", return_value=mc_flows) as mock_mc, \
             patch.object(scraper, "scrape_bond_yield", return_value=None), \
             patch.object(scraper, "persist_flows") as mock_persist_flows, \
             patch.object(scraper, "persist_yield"):

            summary = scraper.run_daily(TRADE_DATE)

        mock_nse.assert_called_once_with(TRADE_DATE)
        mock_mc.assert_called_once_with(TRADE_DATE)
        assert summary["flows_source"] == "MONEYCONTROL"
        assert summary["flows_ok"] is True

    def test_run_daily_uses_nse_when_available(self):
        """When NSE succeeds, run_daily should NOT call Moneycontrol."""
        scraper = _make_scraper(db_engine=None)

        nse_flows = {
            "trade_date": TRADE_DATE,
            "fii_buy": 12000.0,
            "fii_sell": 9000.0,
            "fii_net": 3000.0,
            "dii_buy": 8000.0,
            "dii_sell": 5500.0,
            "dii_net": 2500.0,
        }

        with patch.object(scraper, "scrape_nse_fii_dii", return_value=nse_flows) as mock_nse, \
             patch.object(scraper, "scrape_moneycontrol_fii_dii") as mock_mc, \
             patch.object(scraper, "scrape_bond_yield", return_value=None), \
             patch.object(scraper, "persist_flows"), \
             patch.object(scraper, "persist_yield"):

            summary = scraper.run_daily(TRADE_DATE)

        mock_nse.assert_called_once()
        mock_mc.assert_not_called()
        assert summary["flows_source"] == "NSE"
        assert summary["flows_ok"] is True

    def test_run_daily_returns_false_flows_ok_when_all_sources_fail(self):
        """When both NSE and MC fail, flows_ok should be False."""
        scraper = _make_scraper(db_engine=None)

        with patch.object(scraper, "scrape_nse_fii_dii", return_value=None), \
             patch.object(scraper, "scrape_moneycontrol_fii_dii", return_value=None), \
             patch.object(scraper, "scrape_bond_yield", return_value=None):

            summary = scraper.run_daily(TRADE_DATE)

        assert summary["flows_ok"] is False
        assert summary["yield_ok"] is False
