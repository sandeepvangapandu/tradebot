"""RSS news scraper for Indian financial sources.

Fetches and parses RSS 2.0 / Atom feeds from Moneycontrol, ET Markets,
BusinessLine, and Reuters India.  Persists de-duplicated articles via
UNIQUE (source, link) upsert in the ``news_articles`` table.

Design principles
-----------------
* ``httpx`` for HTTP — supports timeout, redirects, and easy mocking.
* Pure XML parsing via :mod:`xml.etree.ElementTree` — no external HTML parser
  dependency.
* Idempotent: fetching the same feed twice won't duplicate rows.
* All timestamps stored as timezone-aware UTC.
* Returns gracefully on any network / parse error — strategies must tolerate
  temporary scraper failures.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag string."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_pub_date(raw: str | None) -> datetime | None:
    """Parse RFC 822 or ISO-8601 date strings into UTC-aware datetime.

    Args:
        raw: Raw date string from RSS pubDate or Atom updated/published.

    Returns:
        UTC-aware datetime, or None on failure.
    """
    if not raw:
        return None
    raw = raw.strip()
    # Try RFC 822 (standard RSS pubDate)
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        pass
    # Try ISO-8601 / Atom
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    logger.debug("Could not parse date: %r", raw)
    return None


# ---------------------------------------------------------------------------
# NewsRSSScraper
# ---------------------------------------------------------------------------


class NewsRSSScraper:
    """Scrape, parse, and persist Indian financial news from RSS feeds.

    Args:
        db_engine: SQLAlchemy engine for persisting articles.  When ``None``
            the scraper operates in parse-only mode (useful for testing).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, db_engine=None, timeout: int = 30) -> None:
        self._db = db_engine
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_feed(self, source_name: str, rss_url: str) -> list[dict]:
        """Fetch and parse one RSS feed.

        Args:
            source_name: Identifier for the source (e.g. ``"moneycontrol_markets"``).
            rss_url: Full URL of the RSS/Atom feed.

        Returns:
            List of article dicts with keys:
            ``source``, ``title``, ``link``, ``published_at``,
            ``description``, ``body``.
            Returns an empty list on any error.
        """
        try:
            response = httpx.get(rss_url, timeout=self._timeout, follow_redirects=True)
            response.raise_for_status()
            return self.parse_rss_xml(response.text, source=source_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_feed failed for %s (%s): %s", source_name, rss_url, exc)
            return []

    def parse_rss_xml(self, xml_text: str, source: str) -> list[dict]:
        """Parse RSS 2.0 or Atom XML into article dicts.

        Items missing both title and link are silently skipped.

        Args:
            xml_text: Raw XML string.
            source: Source name to embed in each article dict.

        Returns:
            List of article dicts.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("XML parse error for source %s: %s", source, exc)
            return []

        tag = _strip_ns(root.tag).lower()

        # RSS 2.0 — root is <rss>, items live under <rss><channel><item>
        if tag == "rss":
            return self._parse_rss2(root, source)

        # Atom — root is <feed>, items are <entry>
        if tag == "feed":
            return self._parse_atom(root, source)

        logger.warning("Unrecognised feed root tag %r for source %s", root.tag, source)
        return []

    def persist_articles(self, articles: list[dict]) -> int:
        """Upsert articles into the ``news_articles`` table.

        Uses INSERT … ON CONFLICT (source, link) DO NOTHING for idempotency.

        Args:
            articles: List of article dicts (output of :meth:`parse_rss_xml`).

        Returns:
            Number of rows actually inserted (duplicates silently skipped).
        """
        if not articles or self._db is None:
            return 0

        from sqlalchemy import text as sa_text

        upsert_sql = sa_text(
            """
            INSERT INTO news_articles (source, title, link, published_at, description, body)
            VALUES (:source, :title, :link, :published_at, :description, :body)
            ON CONFLICT (source, link) DO NOTHING
            """
        )

        inserted = 0
        try:
            with self._db.begin() as conn:
                for art in articles:
                    result = conn.execute(
                        upsert_sql,
                        {
                            "source": art.get("source", ""),
                            "title": art.get("title", ""),
                            "link": art.get("link"),
                            "published_at": art.get("published_at"),
                            "description": art.get("description"),
                            "body": art.get("body"),
                        },
                    )
                    inserted += result.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("persist_articles DB error: %s", exc)

        return inserted

    def run_all_sources(self) -> dict[str, int]:
        """Fetch + persist all active sources from the ``news_sources`` table.

        Returns:
            Mapping of source name → article count fetched (not inserted, as
            duplicates are silently skipped).
        """
        sources = self._load_active_sources()
        results: dict[str, int] = {}
        for name, url in sources.items():
            articles = self.fetch_feed(name, url)
            self.persist_articles(articles)
            results[name] = len(articles)
            logger.info("Fetched %d articles from %s", len(articles), name)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_active_sources(self) -> dict[str, str]:
        """Load active sources from DB or return empty dict.

        Returns:
            Mapping of name → rss_url for all active rows.
        """
        if self._db is None:
            return {}

        from sqlalchemy import text as sa_text

        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    sa_text("SELECT name, rss_url FROM news_sources WHERE active = TRUE")
                ).fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as exc:  # noqa: BLE001
            logger.warning("_load_active_sources DB error: %s", exc)
            return {}

    def _parse_rss2(self, root: ET.Element, source: str) -> list[dict]:
        """Parse RSS 2.0 format."""
        articles: list[dict] = []
        # Channel may have namespace prefix — search broadly
        for item in root.iter():
            if _strip_ns(item.tag).lower() != "item":
                continue
            art = self._rss_item_to_dict(item, source)
            if art:
                articles.append(art)
        return articles

    def _parse_atom(self, root: ET.Element, source: str) -> list[dict]:
        """Parse Atom feed format."""
        articles: list[dict] = []
        for entry in root.iter():
            if _strip_ns(entry.tag).lower() != "entry":
                continue
            art = self._atom_entry_to_dict(entry, source)
            if art:
                articles.append(art)
        return articles

    def _rss_item_to_dict(self, item: ET.Element, source: str) -> dict | None:
        """Convert a single <item> element to an article dict."""
        title = self._child_text(item, "title")
        link = self._child_text(item, "link")
        if not title and not link:
            return None
        description = self._child_text(item, "description")
        pub_date_raw = self._child_text(item, "pubDate")
        published_at = _parse_pub_date(pub_date_raw)
        # content:encoded → body
        body = self._child_text(item, f"{{{_CONTENT_NS}}}encoded") or description
        return {
            "source": source,
            "title": title or "",
            "link": link or "",
            "published_at": published_at,
            "description": description,
            "body": body,
        }

    def _atom_entry_to_dict(self, entry: ET.Element, source: str) -> dict | None:
        """Convert a single Atom <entry> element to an article dict."""
        title = self._child_text(entry, f"{{{_ATOM_NS}}}title") or self._child_text(entry, "title")
        # <link href="…" rel="alternate"/>
        link = self._atom_link(entry)
        if not title and not link:
            return None
        summary = (
            self._child_text(entry, f"{{{_ATOM_NS}}}summary")
            or self._child_text(entry, "summary")
        )
        published_raw = (
            self._child_text(entry, f"{{{_ATOM_NS}}}published")
            or self._child_text(entry, "published")
            or self._child_text(entry, f"{{{_ATOM_NS}}}updated")
            or self._child_text(entry, "updated")
        )
        published_at = _parse_pub_date(published_raw)
        return {
            "source": source,
            "title": title or "",
            "link": link or "",
            "published_at": published_at,
            "description": summary,
            "body": summary,
        }

    @staticmethod
    def _child_text(element: ET.Element, tag: str) -> str | None:
        """Return stripped text content of a direct child, or None."""
        child = element.find(tag)
        if child is None:
            return None
        return (child.text or "").strip() or None

    @staticmethod
    def _atom_link(entry: ET.Element) -> str | None:
        """Extract href from Atom <link> element (may be namespaced)."""
        for child in entry:
            if _strip_ns(child.tag).lower() == "link":
                href = child.get("href")
                if href:
                    return href.strip()
                # Some Atom feeds use text content rather than attribute
                if child.text:
                    return child.text.strip()
        return None
