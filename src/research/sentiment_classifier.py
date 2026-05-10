"""Rule-based + lexicon sentiment classifier for finance news.

Scores article text against curated positive/negative word lists to produce
a per-symbol sentiment score in [-1, 1].  Easily upgradeable to an LLM-backed
classifier by subclassing :class:`SentimentClassifier` and overriding
:meth:`score_text`.

Design principles
-----------------
* Zero external ML dependencies — pure Python + regex.
* Word-boundary matching prevents false positives (e.g. "ITC" ≠ "WITCH").
* Confidence is the fraction of matched words relative to text length —
  penalises very short texts.
* Idempotent: re-classifying the same article_id skips existing rows.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

POSITIVE_FINANCE_WORDS: list[str] = [
    "beat",
    "surge",
    "rally",
    "gain",
    "profit",
    "upgrade",
    "buy rating",
    "outperform",
    "record high",
    "exceed",
    "strong",
    "robust",
    "jump",
    "soar",
    "bullish",
    "positive",
    "growth",
    "expansion",
    "optimistic",
    "upbeat",
]

NEGATIVE_FINANCE_WORDS: list[str] = [
    "miss",
    "fall",
    "drop",
    "loss",
    "downgrade",
    "sell rating",
    "underperform",
    "record low",
    "disappoint",
    "weak",
    "fraud",
    "probe",
    "investigation",
    "lawsuit",
    "resign",
    "layoff",
    "plunge",
    "crash",
    "bearish",
    "negative",
    "decline",
    "contraction",
    "pessimistic",
    "downbeat",
    "warning",
    "slash",
]

# ---------------------------------------------------------------------------
# Symbol → name alias mapping (top-10 NSE stocks)
# ---------------------------------------------------------------------------

SYMBOL_ALIASES: dict[str, list[str]] = {
    "RELIANCE": ["RELIANCE", "RIL", "Reliance Industries", "Reliance"],
    "HDFCBANK": ["HDFCBANK", "HDFC Bank", "HDFCB"],
    "ICICIBANK": ["ICICIBANK", "ICICI Bank", "ICICIB"],
    "TCS": ["TCS", "Tata Consultancy", "Tata Consultancy Services"],
    "INFY": ["INFY", "Infosys"],
    "HINDUNILVR": ["HINDUNILVR", "HUL", "Hindustan Unilever"],
    "ITC": ["ITC"],
    "AXISBANK": ["AXISBANK", "Axis Bank"],
    "KOTAKBANK": ["KOTAKBANK", "Kotak Bank", "Kotak Mahindra"],
    "SBIN": ["SBIN", "SBI", "State Bank of India"],
}

# Pre-compile alias patterns keyed by symbol for efficiency.
# Patterns use word-boundary anchors so "ITC" won't match inside "WITCH".
_ALIAS_PATTERNS: dict[str, re.Pattern] = {}
for _sym, _aliases in SYMBOL_ALIASES.items():
    # Sort longest alias first to prefer specific matches
    sorted_aliases = sorted(_aliases, key=len, reverse=True)
    pattern_parts = [r"\b" + re.escape(a) + r"\b" for a in sorted_aliases]
    _ALIAS_PATTERNS[_sym] = re.compile(
        "|".join(pattern_parts), re.IGNORECASE
    )

# Pre-compile lexicon patterns
_POSITIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (w, re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE))
    for w in POSITIVE_FINANCE_WORDS
]
_NEGATIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (w, re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE))
    for w in NEGATIVE_FINANCE_WORDS
]


# ---------------------------------------------------------------------------
# SentimentClassifier
# ---------------------------------------------------------------------------


class SentimentClassifier:
    """Classify news articles by symbol using rule-based lexicon matching.

    Args:
        db_engine: SQLAlchemy engine for persisting sentiment rows.  When
            ``None`` the classifier operates in-memory only.
    """

    def __init__(self, db_engine=None) -> None:
        self._db = db_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_symbols(self, text: str) -> list[str]:
        """Find stock symbols mentioned in *text* using alias matching.

        Matching is case-insensitive and respects word boundaries to prevent
        false positives (e.g. "ITC" will not match inside "WITCH").

        Args:
            text: Headline, description, or body text.

        Returns:
            Deduplicated list of matched symbols in definition order.
        """
        matched: list[str] = []
        for sym, pattern in _ALIAS_PATTERNS.items():
            if pattern.search(text):
                matched.append(sym)
        return matched

    def score_text(self, text: str) -> tuple[float, str, dict[str, list[str]]]:
        """Score *text* and return (sentiment_score, classification, matched_words).

        Algorithm
        ---------
        1. Find all positive / negative keyword matches.
        2. Score = (n_positive - n_negative) / (n_positive + n_negative + 1)
           which clips naturally to (-1, 1).
        3. Classification: score > 0.05 → POSITIVE, < -0.05 → NEGATIVE,
           else NEUTRAL.

        Args:
            text: Free-form text to classify.

        Returns:
            Tuple of:
            - ``score``: float in [-1, 1]
            - ``classification``: one of ``"POSITIVE"``, ``"NEGATIVE"``,
              ``"NEUTRAL"``
            - ``matched``: dict with keys ``"positive"`` and ``"negative"``
              listing matched keywords.
        """
        pos_matched: list[str] = []
        neg_matched: list[str] = []

        for word, pattern in _POSITIVE_PATTERNS:
            if pattern.search(text):
                pos_matched.append(word)

        for word, pattern in _NEGATIVE_PATTERNS:
            if pattern.search(text):
                neg_matched.append(word)

        n_pos = len(pos_matched)
        n_neg = len(neg_matched)
        score = (n_pos - n_neg) / (n_pos + n_neg + 1)

        if score > 0.05:
            classification = "POSITIVE"
        elif score < -0.05:
            classification = "NEGATIVE"
        else:
            classification = "NEUTRAL"

        return score, classification, {"positive": pos_matched, "negative": neg_matched}

    def classify_article(self, article: dict) -> list[dict]:
        """Produce per-symbol sentiment records for one article.

        Combines title + description + body for maximum signal, then scores
        once and attaches per-symbol metadata.

        Args:
            article: Dict with at least ``title``; optionally ``description``,
                ``body``, and ``id`` (article_id).

        Returns:
            List of sentiment record dicts with keys:
            ``article_id``, ``symbol``, ``sentiment``, ``classification``,
            ``confidence``, ``matched_keywords``.
        """
        combined = " ".join(
            filter(
                None,
                [
                    article.get("title", ""),
                    article.get("description", ""),
                    article.get("body", ""),
                ],
            )
        )

        symbols = self.extract_symbols(combined)
        if not symbols:
            return []

        score, classification, matched = self.score_text(combined)

        # Confidence: capped fraction of total matched words over text tokens
        word_count = max(len(combined.split()), 1)
        total_matched = len(matched["positive"]) + len(matched["negative"])
        confidence = min(1.0, total_matched / max(word_count * 0.05, 1))

        article_id = article.get("id")
        records: list[dict] = []
        for sym in symbols:
            records.append(
                {
                    "article_id": article_id,
                    "symbol": sym,
                    "sentiment": round(score, 6),
                    "classification": classification,
                    "confidence": round(confidence, 6),
                    "matched_keywords": matched,
                }
            )
        return records

    def classify_and_persist(self, article_id: int, article: dict) -> int:
        """Classify an article and persist per-symbol sentiment rows.

        Skips symbols already present (PRIMARY KEY conflict → DO NOTHING).

        Args:
            article_id: Primary key from ``news_articles``.
            article: Article dict (title / description / body).

        Returns:
            Number of new sentiment rows inserted.
        """
        article = {**article, "id": article_id}
        records = self.classify_article(article)
        if not records or self._db is None:
            return 0

        return self._persist_sentiment_records(records)

    def run_for_unprocessed(self, limit: int = 500) -> int:
        """Classify + persist sentiment for articles not yet processed.

        Finds ``news_articles`` rows that have no corresponding row in
        ``news_sentiment_symbol`` and classifies up to *limit* of them.

        Args:
            limit: Maximum number of articles to process in one call.

        Returns:
            Total number of new sentiment rows inserted.
        """
        if self._db is None:
            return 0

        articles = self._fetch_unprocessed(limit)
        total = 0
        for art in articles:
            article_id = art["id"]
            inserted = self.classify_and_persist(article_id, art)
            total += inserted
            logger.debug(
                "Classified article %d → %d symbol rows", article_id, inserted
            )
        logger.info(
            "run_for_unprocessed: processed %d articles, inserted %d sentiment rows",
            len(articles),
            total,
        )
        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_unprocessed(self, limit: int) -> list[dict]:
        """Fetch articles that have not been sentiment-classified yet."""
        from sqlalchemy import text as sa_text

        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    sa_text(
                        """
                        SELECT a.id, a.title, a.description, a.body
                        FROM news_articles a
                        WHERE NOT EXISTS (
                            SELECT 1 FROM news_sentiment_symbol s
                            WHERE s.article_id = a.id
                        )
                        ORDER BY a.published_at DESC NULLS LAST
                        LIMIT :lim
                        """
                    ),
                    {"lim": limit},
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("_fetch_unprocessed DB error: %s", exc)
            return []

        return [
            {
                "id": row[0],
                "title": row[1] or "",
                "description": row[2] or "",
                "body": row[3] or "",
            }
            for row in rows
        ]

    def _persist_sentiment_records(self, records: list[dict]) -> int:
        """Upsert sentiment records into ``news_sentiment_symbol``."""
        from sqlalchemy import text as sa_text

        upsert_sql = sa_text(
            """
            INSERT INTO news_sentiment_symbol
              (article_id, symbol, sentiment, classification, confidence, matched_keywords)
            VALUES
              (:article_id, :symbol, :sentiment, :classification, :confidence,
               :matched_keywords)
            ON CONFLICT (article_id, symbol) DO NOTHING
            """
        )

        inserted = 0
        try:
            with self._db.begin() as conn:
                for rec in records:
                    result = conn.execute(
                        upsert_sql,
                        {
                            "article_id": rec["article_id"],
                            "symbol": rec["symbol"],
                            "sentiment": rec["sentiment"],
                            "classification": rec["classification"],
                            "confidence": rec["confidence"],
                            "matched_keywords": json.dumps(rec["matched_keywords"]),
                        },
                    )
                    inserted += result.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error("_persist_sentiment_records DB error: %s", exc)

        return inserted
