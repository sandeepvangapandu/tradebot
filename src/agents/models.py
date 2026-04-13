"""Data models for the AI agent pipeline.

All monetary values are in paisa (1 INR = 100 paisa).
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RegimeClassification:
    """Result from the regime detection agent.

    Attributes:
        regime: One of trending_up, trending_down, ranging, volatile.
        confidence: 0.0 to 1.0.
        reasoning: Human-readable explanation.
        indicators: Raw indicator values used for the decision.
    """

    regime: str
    confidence: float
    reasoning: str
    indicators: dict[str, float] = field(default_factory=dict)

    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if confidence is below the threshold."""
        return self.confidence < threshold


@dataclass
class SignalDecision:
    """Decision on a single trading signal from the validation agent.

    Attributes:
        signal_id: ID of the signal being evaluated.
        action: One of approve, reject, modify.
        adjusted_stop_loss: Modified SL in paisa (None if unchanged).
        adjusted_target: Modified target in paisa (None if unchanged).
        adjusted_quantity_pct: Multiplier for quantity (1.0 = no change).
        confidence: Agent's confidence in this decision.
        reasoning: Why the agent made this decision.
    """

    signal_id: str
    action: str  # "approve", "reject", "modify"
    confidence: float
    reasoning: str
    adjusted_stop_loss: int | None = None
    adjusted_target: int | None = None
    adjusted_quantity_pct: float = 1.0

    @property
    def is_approved(self) -> bool:
        """Signal is approved (includes modified signals)."""
        return self.action in ("approve", "modify")

    @property
    def is_rejected(self) -> bool:
        """Signal was rejected."""
        return self.action == "reject"


@dataclass
class NewsItem:
    """A single news headline with metadata."""

    title: str
    source: str
    published: datetime | None = None
    url: str = ""


@dataclass
class SentimentScore:
    """Composite market sentiment score.

    Attributes:
        score: 0-100 Market Mood Index.
        classification: extreme_fear, fear, neutral, greed, extreme_greed.
        news_score: Sentiment from news analysis (-1 to 1).
        volatility_score: Inverted volatility component (0-100).
        breadth_score: Market breadth component (0-100).
    """

    score: float
    classification: str
    news_score: float = 0.0
    volatility_score: float = 50.0
    breadth_score: float = 50.0


@dataclass
class AgentDecision:
    """Record of a single agent's decision for observability."""

    agent_name: str
    instrument_key: str
    input_summary: str
    output_summary: str
    confidence: float
    used_fallback: bool
    latency_ms: int
    lessons_injected: int = 0


@dataclass
class PipelineState:
    """Shared state flowing through the LangGraph pipeline."""

    instrument_key: str
    signals: list[dict[str, Any]]
    regime: RegimeClassification | None = None
    sentiment: SentimentScore | None = None
    validated_signals: list[dict[str, Any]] = field(default_factory=list)
    agent_decisions: list[AgentDecision] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    short_circuited: bool = False
    short_circuit_reason: str = ""
