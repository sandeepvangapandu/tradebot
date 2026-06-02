"""Tests for src/data/corporate_actions_scraper.py.

All HTTP calls are mocked — no real network access.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data.corporate_actions_scraper import CorporateActionsScraper


# ---------------------------------------------------------------------------
# parse_action_type tests
# ---------------------------------------------------------------------------

class TestParseActionType:
    """Unit tests for CorporateActionsScraper.parse_action_type."""

    def setup_method(self):
        self.scraper = CorporateActionsScraper()

    def test_parse_action_type_dividend(self):
        """Dividend description maps to ('DIVIDEND', 5.0)."""
        action_type, ratio = self.scraper.parse_action_type(
            "Dividend - Rs.5.00 Per Share"
        )
        assert action_type == "DIVIDEND"
        assert ratio == pytest.approx(5.0)

    def test_parse_action_type_dividend_rupee_symbol(self):
        """Rupee symbol variant is also parsed correctly."""
        action_type, ratio = self.scraper.parse_action_type(
            "Interim Dividend ₹2.50 per share"
        )
        assert action_type == "DIVIDEND"
        assert ratio == pytest.approx(2.5)

    def test_parse_action_type_split(self):
        """Stock split 1:2 maps to ('SPLIT', 0.5)."""
        action_type, ratio = self.scraper.parse_action_type("Stock Split 1:2")
        assert action_type == "SPLIT"
        assert ratio == pytest.approx(0.5)

    def test_parse_action_type_split_face_value_description(self):
        """Face-value split description is classified as SPLIT."""
        action_type, ratio = self.scraper.parse_action_type(
            "Sub-Division of Equity Shares From Rs.10/- To Rs.5/-"
        )
        assert action_type == "SPLIT"

    def test_parse_action_type_bonus_one_to_one(self):
        """Bonus 1:1 maps to ('BONUS', 1.0)."""
        action_type, ratio = self.scraper.parse_action_type("Bonus 1:1")
        assert action_type == "BONUS"
        assert ratio == pytest.approx(1.0)

    def test_parse_action_type_bonus_two_to_one(self):
        """Bonus 2:1 maps to ('BONUS', 2.0)."""
        action_type, ratio = self.scraper.parse_action_type("Bonus Issue 2:1")
        assert action_type == "BONUS"
        assert ratio == pytest.approx(2.0)

    def test_parse_action_type_unknown_returns_OTHER(self):
        """Unrecognised description maps to ('OTHER', None)."""
        action_type, ratio = self.scraper.parse_action_type(
            "Some completely unrelated corporate event"
        )
        assert action_type == "OTHER"
        assert ratio is None

    def test_parse_action_type_empty_string_returns_OTHER(self):
        """Empty string maps to ('OTHER', None)."""
        action_type, ratio = self.scraper.parse_action_type("")
        assert action_type == "OTHER"
        assert ratio is None

    def test_parse_action_type_rights(self):
        """Rights issue is classified as RIGHTS."""
        action_type, _ = self.scraper.parse_action_type("Rights Issue 1:5 at Rs.75")
        assert action_type == "RIGHTS"

    def test_parse_action_type_buyback(self):
        """Buyback is classified as BUYBACK."""
        action_type, _ = self.scraper.parse_action_type(
            "Buyback of Equity Shares at Rs.2500"
        )
        assert action_type == "BUYBACK"


# ---------------------------------------------------------------------------
# scrape_nse_corporate_actions with mocked HTTP
# ---------------------------------------------------------------------------

class TestScrapeNseCorporateActions:
    """Tests for the HTTP scraping logic using mocked httpx."""

    def setup_method(self):
        self.scraper = CorporateActionsScraper()
        self.today = date.today()

    def _make_nse_item(
        self,
        purpose: str = "Dividend - Rs.5.00 Per Share",
        ex_date_offset: int = 5,
    ) -> dict:
        """Build a minimal NSE API response item."""
        ex_date = self.today + timedelta(days=ex_date_offset)
        return {
            "purpose": purpose,
            "exDate": ex_date.strftime("%d-%b-%Y"),  # NSE format
            "recordDate": (ex_date + timedelta(days=1)).strftime("%d-%b-%Y"),
            "instrumentKey": None,
        }

    def test_scrape_nse_corporate_actions_parses_mock_response(self):
        """Mock httpx returns one item; scraper produces one parsed action."""
        item = self._make_nse_item()
        mock_response = MagicMock()
        mock_response.json.return_value = [item]
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("src.data.corporate_actions_scraper.httpx.Client", return_value=mock_client):
            actions = self.scraper.scrape_nse_corporate_actions("RELIANCE")

        assert len(actions) == 1
        action = actions[0]
        assert action["symbol"] == "RELIANCE"
        assert action["action_type"] == "DIVIDEND"
        assert action["ratio"] == pytest.approx(5.0)
        assert isinstance(action["ex_date"], date)

    def test_scrape_nse_corporate_actions_filters_outside_window(self):
        """Items with ex_date beyond lookahead_days are excluded."""
        item = self._make_nse_item(ex_date_offset=200)  # beyond default 90-day window
        mock_response = MagicMock()
        mock_response.json.return_value = [item]
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("src.data.corporate_actions_scraper.httpx.Client", return_value=mock_client):
            actions = self.scraper.scrape_nse_corporate_actions("RELIANCE")

        assert len(actions) == 0

    def test_scrape_nse_corporate_actions_handles_http_error(self):
        """HTTP errors are caught and an empty list is returned."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("src.data.corporate_actions_scraper.httpx.Client", return_value=mock_client):
            actions = self.scraper.scrape_nse_corporate_actions("RELIANCE")

        assert actions == []

    def test_scrape_nse_corporate_actions_dict_wrapped_response(self):
        """NSE response wrapped in {'data': [...]} is unwrapped correctly."""
        item = self._make_nse_item()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [item]}
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("src.data.corporate_actions_scraper.httpx.Client", return_value=mock_client):
            actions = self.scraper.scrape_nse_corporate_actions("TCS")

        assert len(actions) == 1
        assert actions[0]["symbol"] == "TCS"


# ---------------------------------------------------------------------------
# persist: idempotency via unique constraint
# ---------------------------------------------------------------------------

class TestPersist:
    """Tests for CorporateActionsScraper.persist using a mock DB engine."""

    def _make_action(self, symbol: str = "RELIANCE", days: int = 5) -> dict:
        ex_date = date.today() + timedelta(days=days)
        return {
            "symbol": symbol,
            "instrument_key": None,
            "action_type": "DIVIDEND",
            "ex_date": ex_date,
            "record_date": ex_date + timedelta(days=1),
            "details": "Dividend - Rs.5.00 Per Share",
            "ratio": 5.0,
            "source": "NSE",
        }

    def test_persist_returns_zero_when_no_engine(self):
        """Returns 0 without error when db_engine is None."""
        scraper = CorporateActionsScraper(db_engine=None)
        count = scraper.persist([self._make_action()])
        assert count == 0

    def test_persist_returns_zero_for_empty_list(self):
        """Returns 0 immediately for an empty actions list."""
        mock_engine = MagicMock()
        scraper = CorporateActionsScraper(db_engine=mock_engine)
        count = scraper.persist([])
        assert count == 0

    def test_persist_upserts_idempotent_via_unique_constraint(self):
        """persist() calls execute once per action; idempotency is SQL-level."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = CorporateActionsScraper(db_engine=mock_engine)
        action = self._make_action()

        # First insert
        count1 = scraper.persist([action])
        # Second insert (would hit ON CONFLICT DO UPDATE in real DB)
        count2 = scraper.persist([action])

        assert count1 == 1
        assert count2 == 1
        # Engine.begin() called twice (once per persist call)
        assert mock_engine.begin.call_count == 2

    def test_persist_multiple_actions(self):
        """persist() returns count equal to number of actions provided."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        scraper = CorporateActionsScraper(db_engine=mock_engine)
        actions = [self._make_action("RELIANCE", 5), self._make_action("TCS", 10)]
        count = scraper.persist(actions)
        assert count == 2


# ---------------------------------------------------------------------------
# run_for_universe
# ---------------------------------------------------------------------------

class TestRunForUniverse:
    """Tests for CorporateActionsScraper.run_for_universe."""

    def test_run_for_universe_returns_per_symbol_counts(self):
        """run_for_universe returns a dict keyed by symbol with int counts."""
        scraper = CorporateActionsScraper(db_engine=None)

        def _fake_scrape(symbol, lookback_days=90, lookahead_days=90):
            # Return one fake action per symbol
            return [{
                "symbol": symbol,
                "instrument_key": None,
                "action_type": "DIVIDEND",
                "ex_date": date.today() + timedelta(days=5),
                "record_date": None,
                "details": "Dividend Rs.1",
                "ratio": 1.0,
                "source": "NSE",
            }]

        with patch.object(scraper, "scrape_nse_corporate_actions", side_effect=_fake_scrape):
            results = scraper.run_for_universe(symbols=["RELIANCE", "TCS"])

        assert set(results.keys()) == {"RELIANCE", "TCS"}
        # persist returns 0 because db_engine is None
        assert results["RELIANCE"] == 0
        assert results["TCS"] == 0

    def test_run_for_universe_handles_per_symbol_exceptions(self):
        """Exceptions per symbol are caught and count is set to 0."""
        scraper = CorporateActionsScraper(db_engine=None)

        def _fail(symbol, **kwargs):
            raise RuntimeError("HTTP error")

        with patch.object(scraper, "scrape_nse_corporate_actions", side_effect=_fail):
            results = scraper.run_for_universe(symbols=["INFY"])

        assert results["INFY"] == 0

    def test_run_for_universe_uses_default_universe_when_no_symbols(self):
        """run_for_universe iterates DEFAULT_UNIVERSE when symbols is None."""
        from src.data.corporate_actions_scraper import DEFAULT_UNIVERSE

        scraper = CorporateActionsScraper(db_engine=None)

        with patch.object(
            scraper,
            "scrape_nse_corporate_actions",
            return_value=[],
        ):
            results = scraper.run_for_universe(symbols=None)

        assert set(results.keys()) == set(DEFAULT_UNIVERSE)
