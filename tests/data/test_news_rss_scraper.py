"""Tests for src.data.news_rss_scraper — RSS feed parsing + persistence.

All tests use inline mocks and the news_rss fixture factory.
No real HTTP calls or live DB connection required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.news_rss_scraper import NewsRSSScraper
from tests.fixtures.news_rss import (
    make_rss_xml,
    sample_articles_negative_for,
    sample_articles_positive_for,
    sample_articles_neutral,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scraper(db_engine=None) -> NewsRSSScraper:
    return NewsRSSScraper(db_engine=db_engine, timeout=5)


def _mock_http_response(content: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = content
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from httpx import HTTPStatusError, Request, Response

        resp.raise_for_status.side_effect = Exception(
            f"HTTP {status_code}"
        )
    return resp


def _mock_db_engine(rowcount: int = 1) -> MagicMock:
    """Build a mock SQLAlchemy engine that simulates upsert execution."""
    mock_result = MagicMock()
    mock_result.rowcount = rowcount

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn
    return mock_engine


# ---------------------------------------------------------------------------
# test_parse_rss_xml_extracts_title_link_pubdate
# ---------------------------------------------------------------------------

class TestParseRssXmlExtractsTitleLinkPubdate:
    """parse_rss_xml should extract title, link, and published_at from items."""

    def test_parse_rss_xml_extracts_title_link_pubdate(self):
        articles_in = sample_articles_negative_for("RELIANCE", count=2)
        xml_str = make_rss_xml(articles_in)

        scraper = _make_scraper()
        parsed = scraper.parse_rss_xml(xml_str, source="test_source")

        assert len(parsed) == 2
        for art in parsed:
            assert art["source"] == "test_source"
            assert art["title"]          # non-empty
            assert art["link"]           # non-empty
            assert art["published_at"] is not None
            assert isinstance(art["published_at"], datetime)
            # Must be timezone-aware UTC
            assert art["published_at"].tzinfo is not None

    def test_parse_rss_xml_sets_source_field(self):
        xml_str = make_rss_xml(sample_articles_positive_for("TCS", count=1))
        parsed = _make_scraper().parse_rss_xml(xml_str, source="moneycontrol_markets")
        assert all(a["source"] == "moneycontrol_markets" for a in parsed)


# ---------------------------------------------------------------------------
# test_parse_rss_xml_handles_atom_format
# ---------------------------------------------------------------------------

class TestParseRssXmlHandlesAtomFormat:
    """parse_rss_xml should handle Atom 1.0 feeds."""

    _ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Reuters India Markets</title>
  <link href="https://www.reuters.com/markets/asia/" rel="alternate"/>
  <updated>2026-04-28T09:15:00Z</updated>
  <entry>
    <title>INFY reports 15% profit surge on strong demand</title>
    <link href="https://www.reuters.com/article/infy-results" rel="alternate"/>
    <published>2026-04-28T09:00:00Z</published>
    <summary>Infosys Q4 results beat Street estimates by wide margin.</summary>
  </entry>
  <entry>
    <title>SBIN faces probe over unsecured loans</title>
    <link href="https://www.reuters.com/article/sbin-probe" rel="alternate"/>
    <updated>2026-04-28T08:30:00Z</updated>
    <summary>RBI launches audit after whistleblower complaint.</summary>
  </entry>
</feed>"""

    def test_parse_rss_xml_handles_atom_format(self):
        scraper = _make_scraper()
        parsed = scraper.parse_rss_xml(self._ATOM_XML, source="reuters_india")

        assert len(parsed) == 2
        titles = [a["title"] for a in parsed]
        assert any("INFY" in t or "Infosys" in t or "profit" in t.lower() for t in titles)
        assert any("SBIN" in t or "probe" in t.lower() for t in titles)
        for art in parsed:
            assert art["link"].startswith("https://")
            assert art["published_at"] is not None

    def test_parse_atom_preserves_summary_as_description(self):
        scraper = _make_scraper()
        parsed = scraper.parse_rss_xml(self._ATOM_XML, source="reuters_india")
        for art in parsed:
            assert art["description"] is not None and len(art["description"]) > 0


# ---------------------------------------------------------------------------
# test_parse_rss_xml_skips_items_missing_required_fields
# ---------------------------------------------------------------------------

class TestParseRssXmlSkipsItemsMissingRequiredFields:
    """Items missing both title and link should be skipped."""

    def test_parse_rss_xml_skips_items_missing_required_fields(self):
        # Build XML with one valid item and one empty item
        broken_xml = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>RELIANCE hits record high on strong results</title>
      <link>https://example.com/reliance-high</link>
      <pubDate>Mon, 28 Apr 2026 09:15:00 +0000</pubDate>
    </item>
    <item>
      <description>No title or link here</description>
    </item>
  </channel>
</rss>"""
        scraper = _make_scraper()
        parsed = scraper.parse_rss_xml(broken_xml, source="test")

        assert len(parsed) == 1
        assert parsed[0]["title"] == "RELIANCE hits record high on strong results"

    def test_parse_rss_xml_returns_empty_on_malformed_xml(self):
        scraper = _make_scraper()
        parsed = scraper.parse_rss_xml("<<NOT XML>>", source="test")
        assert parsed == []


# ---------------------------------------------------------------------------
# test_persist_articles_upserts_idempotent
# ---------------------------------------------------------------------------

class TestPersistArticlesUpsertsIdempotent:
    """persist_articles should call INSERT … ON CONFLICT DO NOTHING."""

    def test_persist_articles_upserts_idempotent(self):
        mock_engine = _mock_db_engine(rowcount=3)
        scraper = _make_scraper(db_engine=mock_engine)

        articles = [
            {
                "source": "test_src",
                "title": f"Article {i}",
                "link": f"https://example.com/article-{i}",
                "published_at": datetime(2026, 4, 28, 9, 15, tzinfo=timezone.utc),
                "description": "Test description",
                "body": None,
            }
            for i in range(3)
        ]

        count = scraper.persist_articles(articles)

        mock_engine.begin.assert_called_once()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        assert mock_conn.execute.call_count == 3

    def test_persist_articles_returns_zero_when_no_engine(self):
        scraper = _make_scraper(db_engine=None)
        count = scraper.persist_articles([{"source": "x", "title": "t", "link": "l"}])
        assert count == 0

    def test_persist_articles_returns_zero_for_empty_list(self):
        mock_engine = _mock_db_engine()
        scraper = _make_scraper(db_engine=mock_engine)
        count = scraper.persist_articles([])
        assert count == 0
        mock_engine.begin.assert_not_called()


# ---------------------------------------------------------------------------
# test_fetch_feed_handles_http_error_returns_empty
# ---------------------------------------------------------------------------

class TestFetchFeedHandlesHttpErrorReturnsEmpty:
    """fetch_feed should return [] on any HTTP or network error."""

    def test_fetch_feed_handles_http_error_returns_empty(self):
        scraper = _make_scraper()

        with patch("src.data.news_rss_scraper.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("Connection refused")
            result = scraper.fetch_feed("test_source", "https://example.com/feed.xml")

        assert result == []

    def test_fetch_feed_returns_empty_on_non_200_response(self):
        scraper = _make_scraper()

        with patch("src.data.news_rss_scraper.httpx") as mock_httpx:
            resp = MagicMock()
            resp.text = ""
            resp.raise_for_status.side_effect = Exception("HTTP 503")
            mock_httpx.get.return_value = resp
            result = scraper.fetch_feed("test_source", "https://example.com/down.xml")

        assert result == []

    def test_fetch_feed_parses_valid_response(self):
        scraper = _make_scraper()
        xml_str = make_rss_xml(sample_articles_positive_for("INFY", count=2))

        with patch("src.data.news_rss_scraper.httpx") as mock_httpx:
            resp = MagicMock()
            resp.text = xml_str
            resp.raise_for_status = MagicMock()
            mock_httpx.get.return_value = resp
            result = scraper.fetch_feed("moneycontrol_markets", "https://example.com/feed.xml")

        assert len(result) == 2


# ---------------------------------------------------------------------------
# test_run_all_sources_iterates_active_only
# ---------------------------------------------------------------------------

class TestRunAllSourcesIteratesActiveOnly:
    """run_all_sources should only process active sources from the DB."""

    def test_run_all_sources_iterates_active_only(self):
        """Only sources with active=TRUE should be fetched."""
        # Mock engine returns 2 active sources
        active_sources = [
            ("moneycontrol_markets", "https://mc.com/rss.xml"),
            ("et_markets", "https://et.com/rss.xml"),
        ]

        fetch_result_xml = make_rss_xml(sample_articles_neutral(count=2))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = active_sources

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        scraper = _make_scraper(db_engine=mock_engine)

        with patch.object(scraper, "fetch_feed", return_value=[{"source": "x", "title": "t", "link": "l", "published_at": None, "description": None, "body": None}]) as mock_fetch, \
             patch.object(scraper, "persist_articles", return_value=1):
            results = scraper.run_all_sources()

        assert mock_fetch.call_count == 2
        assert "moneycontrol_markets" in results
        assert "et_markets" in results

    def test_run_all_sources_returns_empty_when_no_engine(self):
        scraper = _make_scraper(db_engine=None)
        results = scraper.run_all_sources()
        assert results == {}
