"""Tests for src.data.block_deals_scraper — block and bulk deals scraping.

All tests use inline mocks (unittest.mock) and do NOT make real HTTP calls
or require a live database.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.data.block_deals_scraper import BlockDealsScraper


# ---------------------------------------------------------------------------
# Constants / shared fixtures
# ---------------------------------------------------------------------------

TRADE_DATE = date(2026, 5, 8)

_NSE_BLOCK_JSON = {
    "data": [
        {
            "symbol": "RELIANCE",
            "clientName": "AXIS MUTUAL FUND",
            "dealType": "BUY",
            "quantity": 500000,
            "price": "2450.50",
        },
        {
            "symbol": "TCS",
            "clientName": "HDFC SECURITIES",
            "dealType": "SELL",
            "quantity": 100000,
            "price": "3800.00",
        },
    ]
}

_NSE_BULK_JSON = {
    "data": [
        {
            "symbol": "INFY",
            "clientName": "ICICI PRUDENTIAL MF",
            "buySell": "B",
            "quantityTraded": 1000000,
            "wap": "1750.25",
        },
        {
            "symbol": "WIPRO",
            "clientName": "SBI MF",
            "buySell": "S",
            "quantityTraded": 250000,
            "wap": "510.00",
        },
    ]
}


def _make_scraper(db_engine=None) -> BlockDealsScraper:
    return BlockDealsScraper(db_engine=db_engine, timeout=5)


def _mock_http_response(payload: dict | list, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    resp.cookies = {}
    return resp


# ---------------------------------------------------------------------------
# test_scrape_nse_block_parses_mock_response
# ---------------------------------------------------------------------------

class TestScrapeNseBlockDeals:
    """Tests for BlockDealsScraper.scrape_nse_block_deals."""

    def test_scrape_nse_block_parses_mock_response(self):
        """Block deals: correct parsing of a mocked NSE API response."""
        scraper = _make_scraper()

        home_resp = _mock_http_response({})
        api_resp = _mock_http_response(_NSE_BLOCK_JSON)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_block_deals(TRADE_DATE)

        assert len(deals) == 2

        reliance = next(d for d in deals if d["symbol"] == "RELIANCE")
        assert reliance["side"] == "BUY"
        assert reliance["quantity"] == 500_000
        assert reliance["price_paisa"] == 245_050  # 2450.50 * 100
        assert reliance["client_name"] == "AXIS MUTUAL FUND"
        assert reliance["exchange"] == "NSE"
        assert reliance["trade_date"] == TRADE_DATE

        tcs = next(d for d in deals if d["symbol"] == "TCS")
        assert tcs["side"] == "SELL"
        assert tcs["quantity"] == 100_000
        assert tcs["price_paisa"] == 380_000

    def test_scrape_nse_block_returns_empty_on_http_error(self):
        """Block deals: HTTP error should return an empty list gracefully."""
        scraper = _make_scraper()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Connection refused")

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_block_deals(TRADE_DATE)

        assert deals == []

    def test_scrape_nse_block_skips_invalid_rows(self):
        """Block deals: rows missing symbol or invalid side are skipped."""
        payload = {
            "data": [
                # Missing symbol
                {"dealType": "BUY", "quantity": 1000, "price": "100.00"},
                # Invalid side
                {"symbol": "ABC", "dealType": "HOLD", "quantity": 1000, "price": "100.00"},
                # Valid
                {"symbol": "VALID", "clientName": "X", "dealType": "BUY", "quantity": 500, "price": "200.00"},
            ]
        }
        scraper = _make_scraper()
        home_resp = _mock_http_response({})
        api_resp = _mock_http_response(payload)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_block_deals(TRADE_DATE)

        assert len(deals) == 1
        assert deals[0]["symbol"] == "VALID"


# ---------------------------------------------------------------------------
# test_scrape_nse_bulk_parses_mock_response
# ---------------------------------------------------------------------------

class TestScrapeNseBulkDeals:
    """Tests for BlockDealsScraper.scrape_nse_bulk_deals."""

    def test_scrape_nse_bulk_parses_mock_response(self):
        """Bulk deals: correct parsing of a mocked NSE API response."""
        scraper = _make_scraper()

        home_resp = _mock_http_response({})
        api_resp = _mock_http_response(_NSE_BULK_JSON)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_bulk_deals(TRADE_DATE)

        assert len(deals) == 2

        infy = next(d for d in deals if d["symbol"] == "INFY")
        assert infy["side"] == "BUY"
        assert infy["quantity"] == 1_000_000
        assert infy["price_paisa"] == 175_025  # 1750.25 * 100
        assert infy["client_name"] == "ICICI PRUDENTIAL MF"
        assert infy["trade_date"] == TRADE_DATE

        wipro = next(d for d in deals if d["symbol"] == "WIPRO")
        assert wipro["side"] == "SELL"
        assert wipro["quantity"] == 250_000

    def test_scrape_nse_bulk_returns_empty_on_http_error(self):
        """Bulk deals: HTTP error should return an empty list gracefully."""
        scraper = _make_scraper()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Timeout")

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_bulk_deals(TRADE_DATE)

        assert deals == []

    def test_scrape_nse_bulk_handles_buysell_single_char(self):
        """Bulk deals: 'B'/'S' side codes should be normalised correctly."""
        payload = {
            "data": [
                {"symbol": "XYZ", "clientName": "AB", "buySell": "B", "quantityTraded": 100, "wap": "500.00"},
                {"symbol": "DEF", "clientName": "CD", "buySell": "S", "quantityTraded": 200, "wap": "300.00"},
            ]
        }
        scraper = _make_scraper()
        home_resp = _mock_http_response({})
        api_resp = _mock_http_response(payload)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [home_resp, api_resp]

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            deals = scraper.scrape_nse_bulk_deals(TRADE_DATE)

        assert deals[0]["side"] == "BUY"
        assert deals[1]["side"] == "SELL"


# ---------------------------------------------------------------------------
# test_persist_block_idempotent
# ---------------------------------------------------------------------------

class TestPersistBlock:
    """Tests for BlockDealsScraper.persist_block."""

    def test_persist_block_idempotent(self):
        """persist_block: calling twice with the same deals should not double-count."""
        mock_result = MagicMock()
        mock_result.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = _make_scraper(db_engine=mock_engine)

        deals = [
            {
                "trade_date": TRADE_DATE,
                "symbol": "RELIANCE",
                "client_name": "AXIS MF",
                "side": "BUY",
                "quantity": 500_000,
                "price_paisa": 245_050,
                "value_crore": 122.525,
                "exchange": "NSE",
                "source": "NSE",
            }
        ]

        count1 = scraper.persist_block(deals)
        assert count1 == 1

        # Second call — DB engine honours ON CONFLICT DO NOTHING (rowcount 0)
        mock_result.rowcount = 0
        count2 = scraper.persist_block(deals)
        assert count2 == 0

    def test_persist_block_returns_zero_with_no_engine(self):
        """persist_block: returns 0 when no engine is configured."""
        scraper = _make_scraper(db_engine=None)
        count = scraper.persist_block([{"symbol": "TEST", "side": "BUY"}])
        assert count == 0

    def test_persist_block_returns_zero_for_empty_list(self):
        """persist_block: returns 0 for an empty input list."""
        mock_engine = MagicMock()
        scraper = _make_scraper(db_engine=mock_engine)
        assert scraper.persist_block([]) == 0


# ---------------------------------------------------------------------------
# test_persist_bulk_idempotent
# ---------------------------------------------------------------------------

class TestPersistBulk:
    """Tests for BlockDealsScraper.persist_bulk."""

    def test_persist_bulk_idempotent(self):
        """persist_bulk: second identical insert should return 0 (DO NOTHING)."""
        mock_result = MagicMock()
        mock_result.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = _make_scraper(db_engine=mock_engine)

        deals = [
            {
                "trade_date": TRADE_DATE,
                "symbol": "INFY",
                "client_name": "ICICI MF",
                "side": "BUY",
                "quantity": 1_000_000,
                "price_paisa": 175_025,
                "exchange": "NSE",
                "source": "NSE",
            }
        ]

        count1 = scraper.persist_bulk(deals)
        assert count1 == 1

        mock_result.rowcount = 0
        count2 = scraper.persist_bulk(deals)
        assert count2 == 0

    def test_persist_bulk_returns_zero_with_no_engine(self):
        """persist_bulk: returns 0 when no engine is configured."""
        scraper = _make_scraper(db_engine=None)
        assert scraper.persist_bulk([{"symbol": "TEST"}]) == 0

    def test_persist_bulk_returns_zero_for_empty_list(self):
        """persist_bulk: returns 0 for an empty input list."""
        mock_engine = MagicMock()
        scraper = _make_scraper(db_engine=mock_engine)
        assert scraper.persist_bulk([]) == 0


# ---------------------------------------------------------------------------
# test_run_daily
# ---------------------------------------------------------------------------

class TestRunDaily:
    """Integration-style tests for BlockDealsScraper.run_daily."""

    def test_run_daily_returns_summary(self):
        """run_daily: should return a summary dict with fetched/inserted counts."""
        scraper = _make_scraper(db_engine=None)

        home_resp = _mock_http_response({})
        block_resp = _mock_http_response(_NSE_BLOCK_JSON)
        bulk_resp = _mock_http_response(_NSE_BULK_JSON)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        # 4 calls: home + block_api + home + bulk_api
        mock_client.get.side_effect = [home_resp, block_resp, home_resp, bulk_resp]

        with patch("src.data.block_deals_scraper.httpx.Client", return_value=mock_client):
            summary = scraper.run_daily(TRADE_DATE)

        assert summary["trade_date"] == TRADE_DATE
        assert summary["block_fetched"] == 2
        assert summary["bulk_fetched"] == 2
        assert summary["block_inserted"] == 0  # no engine
        assert summary["bulk_inserted"] == 0   # no engine
