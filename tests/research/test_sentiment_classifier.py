"""Tests for src.research.sentiment_classifier — rule-based lexicon classifier.

All tests use inline mocks.  No LLM or live DB connection required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.research.sentiment_classifier import (
    NEGATIVE_FINANCE_WORDS,
    POSITIVE_FINANCE_WORDS,
    SYMBOL_ALIASES,
    SentimentClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier(db_engine=None) -> SentimentClassifier:
    return SentimentClassifier(db_engine=db_engine)


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
# test_extract_symbols_matches_aliases
# ---------------------------------------------------------------------------

class TestExtractSymbolsMatchesAliases:
    """extract_symbols should match canonical aliases."""

    def test_extract_symbols_matches_aliases(self):
        classifier = _make_classifier()
        # "RIL" is an alias for RELIANCE; "HDFC Bank" for HDFCBANK
        text = "RIL shares rally after strong Q4; HDFC Bank profits surge."
        symbols = classifier.extract_symbols(text)
        assert "RELIANCE" in symbols
        assert "HDFCBANK" in symbols

    def test_extract_symbols_matches_full_company_name(self):
        classifier = _make_classifier()
        text = "Infosys announces large buyback; State Bank of India profit falls."
        symbols = classifier.extract_symbols(text)
        assert "INFY" in symbols
        assert "SBIN" in symbols

    def test_extract_symbols_matches_canonical_symbol(self):
        classifier = _make_classifier()
        text = "TCS beats estimates; AXISBANK downgraded by brokerages."
        symbols = classifier.extract_symbols(text)
        assert "TCS" in symbols
        assert "AXISBANK" in symbols

    def test_extract_symbols_returns_empty_for_no_match(self):
        classifier = _make_classifier()
        text = "Nifty50 flat ahead of RBI policy; Asian markets mixed."
        symbols = classifier.extract_symbols(text)
        # None of the top-10 symbols are mentioned
        assert symbols == []


# ---------------------------------------------------------------------------
# test_extract_symbols_case_insensitive
# ---------------------------------------------------------------------------

class TestExtractSymbolsCaseInsensitive:
    """Alias matching must be case-insensitive."""

    def test_extract_symbols_case_insensitive(self):
        classifier = _make_classifier()
        # Mixed case
        assert "RELIANCE" in classifier.extract_symbols("reliance industries gains 5%")
        assert "HDFCBANK" in classifier.extract_symbols("hdfc bank reports strong Q3")
        assert "INFY" in classifier.extract_symbols("INFOSYS Q4 PAT up 18%")

    def test_extract_symbols_ril_lowercase(self):
        classifier = _make_classifier()
        assert "RELIANCE" in classifier.extract_symbols("ril shares plunge on probe")


# ---------------------------------------------------------------------------
# test_extract_symbols_word_boundary
# ---------------------------------------------------------------------------

class TestExtractSymbolsWordBoundary:
    """Word boundary anchors prevent substring false-positives."""

    def test_extract_symbols_word_boundary_itc_not_in_witch(self):
        """ITC must NOT match inside 'WITCH'."""
        classifier = _make_classifier()
        text = "WITCH rallies; market sentiment upbeat."
        symbols = classifier.extract_symbols(text)
        assert "ITC" not in symbols

    def test_extract_symbols_word_boundary_sbi_not_in_substring(self):
        """SBI must NOT match inside 'SUBSIDIARIES'."""
        classifier = _make_classifier()
        text = "Subsidiaries of large conglomerates report results."
        symbols = classifier.extract_symbols(text)
        assert "SBIN" not in symbols

    def test_extract_symbols_itc_matches_standalone(self):
        """ITC must match when it appears as a standalone token."""
        classifier = _make_classifier()
        text = "ITC shares plunge on regulatory probe."
        symbols = classifier.extract_symbols(text)
        assert "ITC" in symbols


# ---------------------------------------------------------------------------
# test_score_text_positive_finance_words
# ---------------------------------------------------------------------------

class TestScoreTextPositiveFinanceWords:
    """score_text should produce a positive score for bullish headlines."""

    def test_score_text_positive_finance_words(self):
        classifier = _make_classifier()
        text = "RELIANCE beats estimates; profit surge on robust demand; bullish outlook."
        score, classification, matched = classifier.score_text(text)

        assert score > 0
        assert classification == "POSITIVE"
        assert len(matched["positive"]) > 0
        assert "beat" in matched["positive"] or "surge" in matched["positive"] or "profit" in matched["positive"] or "robust" in matched["positive"]

    def test_score_text_all_positive_words_increase_score(self):
        classifier = _make_classifier()
        text = " ".join(POSITIVE_FINANCE_WORDS)
        score, classification, _ = classifier.score_text(text)
        assert score > 0
        assert classification == "POSITIVE"


# ---------------------------------------------------------------------------
# test_score_text_negative_finance_words
# ---------------------------------------------------------------------------

class TestScoreTextNegativeFinanceWords:
    """score_text should produce a negative score for bearish headlines."""

    def test_score_text_negative_finance_words(self):
        classifier = _make_classifier()
        text = "INFY misses estimates; loss reported; fraud probe launched; stock crashes."
        score, classification, matched = classifier.score_text(text)

        assert score < 0
        assert classification == "NEGATIVE"
        assert len(matched["negative"]) > 0

    def test_score_text_all_negative_words_decrease_score(self):
        classifier = _make_classifier()
        text = " ".join(NEGATIVE_FINANCE_WORDS)
        score, classification, _ = classifier.score_text(text)
        assert score < 0
        assert classification == "NEGATIVE"


# ---------------------------------------------------------------------------
# test_score_text_neutral_no_matches
# ---------------------------------------------------------------------------

class TestScoreTextNeutralNoMatches:
    """score_text should return NEUTRAL when no lexicon words match."""

    def test_score_text_neutral_no_matches(self):
        classifier = _make_classifier()
        text = "Nifty50 trades flat ahead of RBI policy announcement today."
        score, classification, matched = classifier.score_text(text)

        assert classification == "NEUTRAL"
        assert matched["positive"] == [] or abs(score) <= 0.05

    def test_score_text_neutral_balanced_words(self):
        classifier = _make_classifier()
        # Equal positive and negative lexicon words should score near 0 → NEUTRAL
        text = "profit beat but also miss loss"
        score, classification, _ = classifier.score_text(text)
        # With equal counts, score = (2-2)/(2+2+1) = 0 → NEUTRAL
        assert classification == "NEUTRAL"


# ---------------------------------------------------------------------------
# test_classify_article_returns_per_symbol_records
# ---------------------------------------------------------------------------

class TestClassifyArticleReturnsPerSymbolRecords:
    """classify_article should produce one record per matched symbol."""

    def test_classify_article_returns_per_symbol_records(self):
        classifier = _make_classifier()
        article = {
            "id": 42,
            "title": "RELIANCE and TCS both report strong quarterly profit surge.",
            "description": "Both companies beat analyst estimates.",
            "body": None,
        }
        records = classifier.classify_article(article)

        symbols = {r["symbol"] for r in records}
        assert "RELIANCE" in symbols
        assert "TCS" in symbols

        for rec in records:
            assert "article_id" in rec
            assert "sentiment" in rec
            assert "classification" in rec
            assert rec["classification"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
            assert "confidence" in rec
            assert 0.0 <= rec["confidence"] <= 1.0
            assert "matched_keywords" in rec

    def test_classify_article_returns_empty_for_no_symbol(self):
        classifier = _make_classifier()
        article = {
            "id": 1,
            "title": "Market flat; RBI keeps rates unchanged at policy meeting.",
            "description": "No major moves expected.",
        }
        records = classifier.classify_article(article)
        assert records == []

    def test_classify_article_includes_article_id(self):
        classifier = _make_classifier()
        article = {
            "id": 99,
            "title": "SBIN plunge on fraud probe.",
            "description": "Shares drop 8%.",
        }
        records = classifier.classify_article(article)
        assert all(r["article_id"] == 99 for r in records)


# ---------------------------------------------------------------------------
# test_classify_and_persist_writes_to_db
# ---------------------------------------------------------------------------

class TestClassifyAndPersistWritesToDb:
    """classify_and_persist should write rows to news_sentiment_symbol."""

    def test_classify_and_persist_writes_to_db(self):
        mock_engine = _mock_db_engine(rowcount=1)
        classifier = _make_classifier(db_engine=mock_engine)

        article = {
            "title": "HDFCBANK reports record profit surge on robust loan growth.",
            "description": "Analysts upgrade to Buy rating.",
            "body": None,
        }

        inserted = classifier.classify_and_persist(article_id=7, article=article)

        # At least 1 row should be inserted (HDFCBANK matched)
        assert inserted >= 1
        mock_engine.begin.assert_called()

    def test_classify_and_persist_returns_zero_when_no_engine(self):
        classifier = _make_classifier(db_engine=None)
        inserted = classifier.classify_and_persist(
            article_id=1,
            article={"title": "RELIANCE profit surge.", "description": ""},
        )
        assert inserted == 0

    def test_classify_and_persist_returns_zero_for_no_symbol_match(self):
        mock_engine = _mock_db_engine()
        classifier = _make_classifier(db_engine=mock_engine)
        # Article with no symbol mention — should not write anything
        inserted = classifier.classify_and_persist(
            article_id=2,
            article={"title": "Nifty flat today.", "description": ""},
        )
        assert inserted == 0
