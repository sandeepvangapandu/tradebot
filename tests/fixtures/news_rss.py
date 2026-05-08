"""News RSS fixture factories.

Generates realistic synthetic RSS 2.0 XML feeds and article lists for use in
sentiment / news-filter tests.  All data is fully synthetic — no external
network calls are made.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NEGATIVE_TEMPLATES = [
    "{sym} quarterly profit falls 22% on weak demand, misses estimates",
    "{sym} faces regulatory probe over accounting irregularities",
    "{sym} shares plunge as management lowers FY guidance sharply",
    "{sym} CFO resigns amid internal audit findings",
    "{sym} hit by massive sell-off; FII outflows accelerate",
    "{sym} reports unexpected loss; board suspends dividend",
    "{sym} downgraded to 'Sell' by three brokerages after earnings miss",
    "{sym} plant shutdown extends — supply disruption to hit margins",
]

_POSITIVE_TEMPLATES = [
    "{sym} beats Street estimates; PAT up 18% YoY",
    "{sym} wins ₹5,000 crore government contract — stock hits 52-week high",
    "{sym} upgrades guidance for FY27 after strong Q4 results",
    "{sym} announces buyback at ₹200 premium to current price",
    "{sym} board approves special dividend; investors cheer",
    "{sym} new product line drives record revenue in Q1",
    "{sym} gains market share; brokerages raise target prices",
]

_NEUTRAL_HEADLINES = [
    "Nifty50 trades flat ahead of RBI policy announcement",
    "Sensex opens marginally lower; auto stocks lead gains",
    "Markets watch US Fed minutes for rate trajectory clues",
    "FII data: net buyers in debt, sellers in equity segment today",
    "NSE to introduce new circuit filter rules from next month",
    "SEBI issues circular on algo trading disclosure norms",
    "SGX Nifty flat; Asian markets mixed after China data release",
    "Midcap index outperforms benchmark for fifth straight session",
    "Options expiry week: analysts expect high volatility in BankNifty",
    "India VIX dips below 14 as sentiment stabilises post Budget",
]

_BASE_TS = datetime(2026, 4, 28, 9, 15, 0, tzinfo=timezone.utc)


def _rfc822(dt: datetime) -> str:
    """Format a datetime as RFC 822 (used in RSS pubDate)."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _make_article(title: str, link: str, description: str, pub_offset_seconds: int) -> dict:
    pub_dt = _BASE_TS + timedelta(seconds=pub_offset_seconds)
    return {
        "title": title,
        "link": link,
        "description": description,
        "pubDate": _rfc822(pub_dt),
    }


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_rss_xml(articles: list[dict]) -> str:
    """Build a valid RSS 2.0 XML string from a list of article dicts.

    Args:
        articles: List of dicts with keys ``title``, ``link``, ``description``,
            ``pubDate``.  Missing keys are replaced with empty strings.

    Returns:
        UTF-8 encoded RSS 2.0 XML string.
    """
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Trading Bot Test Feed"
    ET.SubElement(channel, "link").text = "https://example.com/feed"
    ET.SubElement(channel, "description").text = "Synthetic market news for testing"
    ET.SubElement(channel, "language").text = "en-in"
    ET.SubElement(channel, "lastBuildDate").text = _rfc822(_BASE_TS)

    for art in articles:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = art.get("title", "")
        ET.SubElement(item, "link").text = art.get("link", "https://example.com/article")
        ET.SubElement(item, "description").text = art.get("description", "")
        ET.SubElement(item, "pubDate").text = art.get("pubDate", _rfc822(_BASE_TS))

    tree = ET.ElementTree(rss)
    import io
    buf = io.StringIO()
    tree.write(buf, encoding="unicode", xml_declaration=False)
    # Prepend a standard XML declaration manually (BytesIO conflicts with unicode encoding)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + buf.getvalue()


def sample_articles_negative_for(symbol: str, count: int = 3, seed: int = 42) -> list[dict]:
    """Return ``count`` synthetic negative-sentiment articles mentioning ``symbol``.

    Args:
        symbol: Stock/index symbol, e.g. ``"RELIANCE"`` or ``"INFY"``.
        count: Number of articles to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of article dicts compatible with :func:`make_rss_xml`.
    """
    rng = random.Random(seed)
    templates = _NEGATIVE_TEMPLATES.copy()
    rng.shuffle(templates)
    articles: list[dict] = []
    for i in range(count):
        tmpl = templates[i % len(templates)]
        title = tmpl.format(sym=symbol)
        articles.append(_make_article(
            title=title,
            link=f"https://example.com/news/{symbol.lower()}-negative-{i}",
            description=title + ". Markets react sharply.",
            pub_offset_seconds=i * 600,
        ))
    return articles


def sample_articles_positive_for(symbol: str, count: int = 3, seed: int = 42) -> list[dict]:
    """Return ``count`` synthetic positive-sentiment articles mentioning ``symbol``.

    Args:
        symbol: Stock/index symbol.
        count: Number of articles to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of article dicts compatible with :func:`make_rss_xml`.
    """
    rng = random.Random(seed)
    templates = _POSITIVE_TEMPLATES.copy()
    rng.shuffle(templates)
    articles: list[dict] = []
    for i in range(count):
        tmpl = templates[i % len(templates)]
        title = tmpl.format(sym=symbol)
        articles.append(_make_article(
            title=title,
            link=f"https://example.com/news/{symbol.lower()}-positive-{i}",
            description=title + ". Analysts remain bullish.",
            pub_offset_seconds=i * 600,
        ))
    return articles


def sample_articles_neutral(count: int = 5, seed: int = 42) -> list[dict]:
    """Return ``count`` synthetic neutral market news articles.

    Args:
        count: Number of articles to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of article dicts compatible with :func:`make_rss_xml`.
    """
    rng = random.Random(seed)
    headlines = _NEUTRAL_HEADLINES.copy()
    rng.shuffle(headlines)
    articles: list[dict] = []
    for i in range(count):
        title = headlines[i % len(headlines)]
        articles.append(_make_article(
            title=title,
            link=f"https://example.com/news/market-{i}",
            description=title + ". More details awaited.",
            pub_offset_seconds=i * 300,
        ))
    return articles
