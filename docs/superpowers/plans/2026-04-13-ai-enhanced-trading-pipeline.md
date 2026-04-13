# AI-Enhanced Trading Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer LLM-powered AI agents (market regime, signal validation, news sentiment) and a self-improving memory system on top of the existing deterministic trading bot, connected via a LangGraph state-machine pipeline.

**Architecture:** The existing rule-based strategies, risk management, and execution layers remain unchanged. New AI agents sit *between* signal generation and order execution as an intelligent filter layer. A LangGraph `StateGraph` orchestrates the flow: Regime Detection -> Strategy Filtering -> Signal Validation -> Risk Check. Each agent has a deterministic fallback so the system works even if the LLM is unavailable. A trade memory system analyzes outcomes, classifies mistakes, and injects lessons back into agent context.

**Tech Stack:** Python 3.11+, LangChain + LangGraph (agent orchestration), Groq (free Llama-3.3-70b inference), feedparser (news RSS), existing pandas-ta/SQLAlchemy/loguru stack.

---

## File Structure

### New files to create:
```
src/
  agents/
    __init__.py                    # Package init, exports
    llm_client.py                  # Groq LLM client with circuit breaker + rate limiter
    base_agent.py                  # Base class for all LLM agents (prompt, fallback, parse)
    regime_agent.py                # LLM-powered market regime classification
    signal_validator.py            # LLM signal quality control
    news_agent.py                  # News fetching + LLM sentiment scoring
    sentiment.py                   # Composite Market Mood Index
    pipeline.py                    # LangGraph state machine orchestrating all agents
    models.py                      # Pydantic models for pipeline state, agent I/O
  memory/
    __init__.py                    # Package init
    outcome_analyzer.py            # MAE/MFE analysis on completed trades
    mistake_classifier.py          # Rule-based + LLM mistake categorization
    memory_db.py                   # Lesson storage with time-decay and scoring
    injector.py                    # Retrieves and formats relevant lessons for agents
tests/
  test_llm_client.py
  test_regime_agent.py
  test_signal_validator.py
  test_news_agent.py
  test_sentiment.py
  test_pipeline.py
  test_outcome_analyzer.py
  test_mistake_classifier.py
  test_memory_db.py
  test_injector.py
```

### Existing files to modify:
```
config/settings.py               # Add AI agent settings (Groq key, model, thresholds)
src/persistence/models.py        # Add AgentDecisionRecord, MistakeRecord, MemoryLessonRecord tables
src/main.py                      # Wire pipeline into trading loop
requirements.txt                 # Add langchain, langgraph, groq, feedparser
```

---

## Task 1: Dependencies and Configuration

**Files:**
- Modify: `requirements.txt`
- Modify: `config/settings.py:9-131`
- Test: Manual — `pip install` and import check

- [ ] **Step 1: Add new dependencies to requirements.txt**

Add this block at the end of `requirements.txt`:

```
# -----------------------------------------------------------------------------
# AI Agent Pipeline
# -----------------------------------------------------------------------------
# LLM orchestration framework for agent pipeline
langchain>=0.3.0
langchain-groq>=0.2.0
langgraph>=0.2.0

# Groq SDK for free Llama-3.3-70b inference
groq>=0.11.0

# RSS feed parsing for news sentiment agent
feedparser>=6.0.0
```

- [ ] **Step 2: Add AI agent settings to config/settings.py**

Add these fields to the `Settings` class after the existing `learning_export_lessons_path` field (line 119):

```python
    # AI Agent Pipeline Configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024
    groq_rate_limit_rpm: int = 30  # Free tier: 30 requests/min

    # Agent pipeline settings
    agent_pipeline_enabled: bool = True
    regime_confidence_threshold: float = 0.3  # Skip pipeline if below
    signal_min_confidence: float = 0.5  # Reject signals below this
    news_sentiment_weight: float = 0.35
    volatility_sentiment_weight: float = 0.35
    breadth_sentiment_weight: float = 0.30
    news_cache_ttl_seconds: int = 300  # 5 minutes
    sentiment_cache_ttl_seconds: int = 600  # 10 minutes

    # Memory system settings
    memory_enabled: bool = True
    memory_decay_rate: float = 0.05  # 5% per week
    memory_decay_start_days: int = 30  # Start decay after 30 days
    memory_max_lessons_per_agent: int = 5  # Top N lessons injected
    memory_boost_factor: float = 1.1  # Boost useful lessons (capped at 2x)
```

- [ ] **Step 3: Install dependencies**

Run: `pip install langchain langchain-groq langgraph groq feedparser`

- [ ] **Step 4: Verify imports work**

Run: `python -c "import langchain; import langgraph; import groq; import feedparser; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/settings.py
git commit -m "feat: add AI agent pipeline dependencies and configuration"
```

---

## Task 2: Agent Pipeline Data Models

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/models.py`
- Test: `tests/test_agent_models.py`

- [ ] **Step 1: Create the agents package init**

```python
"""AI agent pipeline for intelligent trade filtering."""
```

- [ ] **Step 2: Write the failing test for pipeline state model**

Create `tests/test_agent_models.py`:

```python
"""Tests for agent pipeline data models."""

import pytest
from src.agents.models import (
    PipelineState,
    RegimeClassification,
    SignalDecision,
    NewsItem,
    SentimentScore,
    AgentDecision,
)


def test_regime_classification_defaults():
    regime = RegimeClassification(
        regime="trending_up",
        confidence=0.8,
        reasoning="ADX=35, strong uptrend",
    )
    assert regime.regime == "trending_up"
    assert regime.confidence == 0.8
    assert regime.reasoning == "ADX=35, strong uptrend"


def test_regime_classification_low_confidence_flag():
    regime = RegimeClassification(
        regime="ranging",
        confidence=0.2,
        reasoning="unclear signals",
    )
    assert regime.is_low_confidence(threshold=0.3)
    assert not regime.is_low_confidence(threshold=0.1)


def test_signal_decision_approve():
    decision = SignalDecision(
        signal_id="sig_001",
        action="approve",
        adjusted_stop_loss=None,
        adjusted_target=None,
        adjusted_quantity_pct=1.0,
        confidence=0.85,
        reasoning="Strong trend alignment",
    )
    assert decision.is_approved
    assert not decision.is_rejected


def test_signal_decision_reject():
    decision = SignalDecision(
        signal_id="sig_002",
        action="reject",
        confidence=0.9,
        reasoning="Counter-trend in strong downtrend",
    )
    assert decision.is_rejected
    assert not decision.is_approved


def test_signal_decision_modify():
    decision = SignalDecision(
        signal_id="sig_003",
        action="modify",
        adjusted_stop_loss=48000_00,
        adjusted_target=52000_00,
        adjusted_quantity_pct=0.75,
        confidence=0.7,
        reasoning="Widen SL due to high volatility",
    )
    assert decision.is_approved  # modify counts as approved
    assert decision.adjusted_quantity_pct == 0.75


def test_sentiment_score_classification():
    extreme_fear = SentimentScore(score=15.0, classification="extreme_fear")
    assert extreme_fear.classification == "extreme_fear"

    greed = SentimentScore(score=72.0, classification="greed")
    assert greed.classification == "greed"


def test_pipeline_state_creation():
    state = PipelineState(
        instrument_key="NSE_EQ:RELIANCE",
        signals=[],
    )
    assert state.instrument_key == "NSE_EQ:RELIANCE"
    assert state.regime is None
    assert state.sentiment is None
    assert state.validated_signals == []
    assert not state.short_circuited


def test_pipeline_state_short_circuit():
    state = PipelineState(
        instrument_key="NSE_EQ:RELIANCE",
        signals=[],
        short_circuited=True,
        short_circuit_reason="Low regime confidence",
    )
    assert state.short_circuited
    assert state.short_circuit_reason == "Low regime confidence"


def test_agent_decision_record():
    decision = AgentDecision(
        agent_name="regime_agent",
        instrument_key="NSE_EQ:TCS",
        input_summary="ADX=25, ATR_pct=60",
        output_summary="trending_up, confidence=0.75",
        confidence=0.75,
        used_fallback=False,
        latency_ms=450,
    )
    assert decision.agent_name == "regime_agent"
    assert not decision.used_fallback
    assert decision.latency_ms == 450
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_agent_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.models'`

- [ ] **Step 4: Implement the models**

Create `src/agents/models.py`:

```python
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
    """A single news headline with metadata.

    Attributes:
        title: Headline text.
        source: Publisher name.
        published: Publication timestamp.
        url: Link to the article.
    """

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
    """Record of a single agent's decision for observability.

    Attributes:
        agent_name: Which agent made this decision.
        instrument_key: Instrument this decision applies to.
        input_summary: Condensed representation of inputs.
        output_summary: Condensed representation of output.
        confidence: Agent confidence.
        used_fallback: Whether the deterministic fallback was used.
        latency_ms: Time taken for this agent call in milliseconds.
        lessons_injected: Number of memory lessons injected into context.
    """

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
    """Shared state flowing through the LangGraph pipeline.

    Each agent reads from and writes to this state as it passes
    through the graph nodes.

    Attributes:
        instrument_key: The instrument being evaluated.
        signals: Raw signals from the strategy engine.
        regime: Regime classification result (set by regime agent).
        sentiment: Market sentiment (set by sentiment agent).
        validated_signals: Signals that passed validation.
        agent_decisions: Log of all agent decisions for observability.
        lessons: Memory lessons injected into this pipeline run.
        short_circuited: Whether pipeline was short-circuited.
        short_circuit_reason: Why pipeline was short-circuited.
    """

    instrument_key: str
    signals: list[dict[str, Any]]
    regime: RegimeClassification | None = None
    sentiment: SentimentScore | None = None
    validated_signals: list[dict[str, Any]] = field(default_factory=list)
    agent_decisions: list[AgentDecision] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    short_circuited: bool = False
    short_circuit_reason: str = ""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_agent_models.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/__init__.py src/agents/models.py tests/test_agent_models.py
git commit -m "feat: add data models for AI agent pipeline"
```

---

## Task 3: LLM Client with Circuit Breaker

**Files:**
- Create: `src/agents/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_client.py`:

```python
"""Tests for LLM client with circuit breaker and rate limiting."""

import pytest
from unittest.mock import MagicMock, patch
from src.agents.llm_client import LLMClient, LLMCircuitBreaker, LLMResponse


class TestLLMCircuitBreaker:
    def test_starts_closed(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=30)
        assert cb.state == "closed"
        assert cb.is_available()

    def test_opens_after_threshold_failures(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert not cb.is_available()

    def test_success_resets_failure_count(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_half_open_after_timeout(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        # With 0s timeout, should immediately transition to half_open
        assert cb.is_available()  # triggers half_open check
        assert cb.state == "half_open"

    def test_half_open_success_closes(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.is_available()  # triggers half_open
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        cb = LLMCircuitBreaker(failure_threshold=3, recovery_timeout_s=0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.is_available()  # triggers half_open
        cb.record_failure()
        assert cb.state == "open"


class TestLLMResponse:
    def test_successful_response(self):
        resp = LLMResponse(content='{"regime": "trending_up"}', success=True)
        assert resp.success
        assert resp.content == '{"regime": "trending_up"}'
        assert resp.error is None

    def test_failed_response(self):
        resp = LLMResponse(content="", success=False, error="Rate limited")
        assert not resp.success
        assert resp.error == "Rate limited"


class TestLLMClient:
    def test_client_creation_without_key(self):
        client = LLMClient(api_key="", model="llama-3.3-70b-versatile")
        assert not client.is_configured

    def test_client_creation_with_key(self):
        client = LLMClient(api_key="test-key", model="llama-3.3-70b-versatile")
        assert client.is_configured

    def test_invoke_returns_fallback_when_not_configured(self):
        client = LLMClient(api_key="", model="test")
        response = client.invoke("test prompt")
        assert not response.success
        assert "not configured" in response.error.lower()

    def test_invoke_returns_fallback_when_circuit_open(self):
        client = LLMClient(api_key="test-key", model="test")
        # Force circuit open
        for _ in range(3):
            client._circuit_breaker.record_failure()
        response = client.invoke("test prompt")
        assert not response.success
        assert "circuit" in response.error.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the LLM client**

Create `src/agents/llm_client.py`:

```python
"""LLM client with circuit breaker, rate limiting, and fallback.

Wraps Groq API calls with resilience patterns so the trading bot
never depends on LLM availability for continued operation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from loguru import logger

from src.utils.rate_limiter import RateLimiter


@dataclass
class LLMResponse:
    """Response from an LLM invocation.

    Attributes:
        content: Raw text content from the LLM.
        success: Whether the call succeeded.
        error: Error message if the call failed.
        latency_ms: Round-trip time in milliseconds.
        used_fallback_model: Whether the fallback model was used.
    """

    content: str
    success: bool
    error: str | None = None
    latency_ms: int = 0
    used_fallback_model: bool = False


class LLMCircuitBreaker:
    """Circuit breaker for LLM API calls.

    States: closed (normal) -> open (failing) -> half_open (testing).

    Args:
        failure_threshold: Failures before opening circuit.
        recovery_timeout_s: Seconds before attempting half-open.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 30.0,
    ) -> None:
        self._lock = threading.Lock()
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._failure_count = 0
        self._state = "closed"  # closed, open, half_open
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        """Current circuit breaker state."""
        with self._lock:
            return self._state

    def is_available(self) -> bool:
        """Check if the circuit allows requests.

        Returns:
            True if requests should be attempted.
        """
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout_s:
                    self._state = "half_open"
                    logger.info("LLM circuit breaker: open -> half_open")
                    return True
                return False
            # half_open: allow one request
            return True

    def record_success(self) -> None:
        """Record a successful LLM call."""
        with self._lock:
            self._failure_count = 0
            if self._state == "half_open":
                self._state = "closed"
                logger.info("LLM circuit breaker: half_open -> closed")

    def record_failure(self) -> None:
        """Record a failed LLM call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == "half_open":
                self._state = "open"
                logger.warning("LLM circuit breaker: half_open -> open")
            elif self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning(
                    "LLM circuit breaker: closed -> open after {} failures",
                    self._failure_count,
                )


class LLMClient:
    """Resilient LLM client for agent pipeline.

    Wraps Groq API with circuit breaker, rate limiting, and
    automatic fallback to a smaller model on failure.

    Args:
        api_key: Groq API key.
        model: Primary model ID.
        fallback_model: Fallback model ID for retries.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        rate_limit_rpm: Rate limit in requests per minute.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        fallback_model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        rate_limit_rpm: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._circuit_breaker = LLMCircuitBreaker(
            failure_threshold=3, recovery_timeout_s=30.0
        )
        self._rate_limiter = RateLimiter(
            rate=rate_limit_rpm / 60.0,
            capacity=min(rate_limit_rpm / 60.0 * 5, 10.0),
        )

        self._groq_client = None
        if api_key:
            try:
                from groq import Groq

                self._groq_client = Groq(api_key=api_key)
            except ImportError:
                logger.warning("groq package not installed, LLM client disabled")

    @property
    def is_configured(self) -> bool:
        """Whether the client has a valid API key and SDK."""
        return bool(self._api_key) and self._groq_client is not None

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        use_fallback_model: bool = False,
    ) -> LLMResponse:
        """Send a prompt to the LLM and get a response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            use_fallback_model: Force use of the fallback model.

        Returns:
            LLMResponse with content or error details.
        """
        if not self.is_configured:
            return LLMResponse(
                content="",
                success=False,
                error="LLM client not configured (no API key)",
            )

        if not self._circuit_breaker.is_available():
            return LLMResponse(
                content="",
                success=False,
                error="LLM circuit breaker is open",
            )

        model = self._fallback_model if use_fallback_model else self._model

        # Rate limit
        self._rate_limiter.acquire()

        start = time.monotonic()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            content = response.choices[0].message.content or ""
            latency_ms = int((time.monotonic() - start) * 1000)

            self._circuit_breaker.record_success()
            logger.debug(
                "LLM call OK | model={} | latency={}ms | tokens={}",
                model,
                latency_ms,
                response.usage.total_tokens if response.usage else "?",
            )

            return LLMResponse(
                content=content,
                success=True,
                latency_ms=latency_ms,
                used_fallback_model=use_fallback_model,
            )

        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._circuit_breaker.record_failure()
            logger.warning("LLM call failed | model={} | error={}", model, e)

            # Try fallback model if primary failed
            if not use_fallback_model and model != self._fallback_model:
                logger.info("Retrying with fallback model: {}", self._fallback_model)
                return self.invoke(
                    prompt, system_prompt, use_fallback_model=True
                )

            return LLMResponse(
                content="",
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_llm_client.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/llm_client.py tests/test_llm_client.py
git commit -m "feat: add LLM client with circuit breaker and rate limiting"
```

---

## Task 4: Base Agent Class

**Files:**
- Create: `src/agents/base_agent.py`
- Test: `tests/test_base_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_base_agent.py`:

```python
"""Tests for the base agent class."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient, LLMResponse


class ConcreteAgent(BaseAgent):
    """Test implementation of BaseAgent."""

    @property
    def agent_name(self) -> str:
        return "test_agent"

    def _build_prompt(self, context: dict) -> str:
        return f"Analyze: {json.dumps(context)}"

    def _build_system_prompt(self, lessons: list[str]) -> str:
        base = "You are a test agent. Respond with JSON."
        if lessons:
            base += "\nLessons: " + "; ".join(lessons)
        return base

    def _parse_response(self, raw: str) -> dict:
        return json.loads(raw)

    def _fallback(self, context: dict) -> dict:
        return {"result": "fallback", "confidence": 0.3}


class TestBaseAgent:
    def test_agent_name(self):
        client = MagicMock(spec=LLMClient)
        agent = ConcreteAgent(llm_client=client)
        assert agent.agent_name == "test_agent"

    def test_run_uses_llm_when_available(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "llm_output", "confidence": 0.9}',
            success=True,
            latency_ms=100,
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "llm_output"
        client.invoke.assert_called_once()

    def test_run_falls_back_on_llm_failure(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content="", success=False, error="timeout"
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"
        assert result["confidence"] == 0.3

    def test_run_falls_back_on_parse_error(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content="not valid json at all",
            success=True,
            latency_ms=50,
        )
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"

    def test_run_falls_back_when_not_configured(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = ConcreteAgent(llm_client=client)
        result = agent.run(context={"data": "test"})
        assert result["result"] == "fallback"
        client.invoke.assert_not_called()

    def test_lessons_passed_to_system_prompt(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "ok", "confidence": 0.8}',
            success=True,
            latency_ms=100,
        )
        agent = ConcreteAgent(llm_client=client)
        agent.run(
            context={"data": "test"},
            lessons=["Avoid counter-trend longs in strong downtrend"],
        )
        call_kwargs = client.invoke.call_args
        assert "Avoid counter-trend" in call_kwargs.kwargs.get("system_prompt", "") or \
               "Avoid counter-trend" in str(call_kwargs)

    def test_last_decision_tracked(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content='{"result": "ok", "confidence": 0.8}',
            success=True,
            latency_ms=200,
        )
        agent = ConcreteAgent(llm_client=client)
        agent.run(context={"data": "test"})
        assert agent.last_decision is not None
        assert agent.last_decision.agent_name == "test_agent"
        assert not agent.last_decision.used_fallback

    def test_last_decision_tracks_fallback(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = ConcreteAgent(llm_client=client)
        agent.run(context={"data": "test"})
        assert agent.last_decision is not None
        assert agent.last_decision.used_fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_base_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement BaseAgent**

Create `src/agents/base_agent.py`:

```python
"""Abstract base class for all LLM-powered agents.

Every agent follows the same pattern:
1. Build a prompt from context data.
2. Call the LLM via LLMClient.
3. Parse the JSON response.
4. If any step fails, fall back to a deterministic implementation.

This ensures the trading bot never depends on LLM availability.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from src.agents.llm_client import LLMClient
from src.agents.models import AgentDecision


class BaseAgent(ABC):
    """Abstract base for LLM-powered trading agents.

    Subclasses must implement:
        - agent_name: Unique identifier for this agent.
        - _build_prompt: Convert context dict to an LLM prompt.
        - _build_system_prompt: Build system prompt with injected lessons.
        - _parse_response: Parse LLM text output to structured data.
        - _fallback: Deterministic fallback when LLM is unavailable.

    Args:
        llm_client: Shared LLM client instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._last_decision: AgentDecision | None = None

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique name for this agent."""

    @abstractmethod
    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from context data."""

    @abstractmethod
    def _build_system_prompt(self, lessons: list[str]) -> str:
        """Build the system prompt, including injected memory lessons."""

    @abstractmethod
    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse raw LLM text to structured output."""

    @abstractmethod
    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when LLM is unavailable or fails."""

    @property
    def last_decision(self) -> AgentDecision | None:
        """Most recent decision made by this agent."""
        return self._last_decision

    def run(
        self,
        context: dict[str, Any],
        lessons: list[str] | None = None,
        instrument_key: str = "",
    ) -> dict[str, Any]:
        """Execute the agent: try LLM, fall back to deterministic.

        Args:
            context: Input data for this agent's decision.
            lessons: Memory lessons to inject into the prompt.
            instrument_key: Instrument this decision applies to.

        Returns:
            Structured decision dict from either LLM or fallback.
        """
        lessons = lessons or []
        start = time.monotonic()

        # If LLM is not configured, go straight to fallback
        if not self._llm.is_configured:
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        # Build prompts
        prompt = self._build_prompt(context)
        system_prompt = self._build_system_prompt(lessons)

        # Call LLM
        response = self._llm.invoke(prompt, system_prompt=system_prompt)

        if not response.success:
            logger.info(
                "{} LLM failed ({}), using fallback",
                self.agent_name,
                response.error,
            )
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        # Parse response
        try:
            result = self._parse_response(response.content)
        except Exception as e:
            logger.warning(
                "{} failed to parse LLM response: {}. Using fallback.",
                self.agent_name,
                e,
            )
            result = self._fallback(context)
            self._record_decision(
                instrument_key=instrument_key,
                context=context,
                result=result,
                used_fallback=True,
                latency_ms=int((time.monotonic() - start) * 1000),
                lessons_count=len(lessons),
            )
            return result

        self._record_decision(
            instrument_key=instrument_key,
            context=context,
            result=result,
            used_fallback=False,
            latency_ms=response.latency_ms,
            lessons_count=len(lessons),
        )
        return result

    def _record_decision(
        self,
        instrument_key: str,
        context: dict[str, Any],
        result: dict[str, Any],
        used_fallback: bool,
        latency_ms: int,
        lessons_count: int,
    ) -> None:
        """Record decision for observability."""
        confidence = result.get("confidence", 0.5)
        self._last_decision = AgentDecision(
            agent_name=self.agent_name,
            instrument_key=instrument_key,
            input_summary=str(context)[:200],
            output_summary=str(result)[:200],
            confidence=confidence,
            used_fallback=used_fallback,
            latency_ms=latency_ms,
            lessons_injected=lessons_count,
        )
        logger.debug(
            "{} decision | fallback={} | confidence={:.2f} | latency={}ms",
            self.agent_name,
            used_fallback,
            confidence,
            latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_base_agent.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base_agent.py tests/test_base_agent.py
git commit -m "feat: add base agent class with LLM/fallback pattern"
```

---

## Task 5: Market Regime Agent (LLM-Powered)

**Files:**
- Create: `src/agents/regime_agent.py`
- Test: `tests/test_regime_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_regime_agent.py`:

```python
"""Tests for the LLM-powered market regime agent."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.regime_agent import RegimeAgent
from src.agents.llm_client import LLMClient, LLMResponse
from src.agents.models import RegimeClassification


class TestRegimeAgent:
    def _make_context(self, adx=35.0, rsi=55.0, atr_pct=60.0, bb_width=3.5,
                      price_position=70.0, ema_9_above_21=True):
        return {
            "adx": adx,
            "adx_trend": "rising",
            "rsi": rsi,
            "atr_percentile": atr_pct,
            "bb_width": bb_width,
            "bb_width_percentile": 55.0,
            "price_position_in_range": price_position,
            "ema_9": 50500.0,
            "ema_21": 50200.0,
            "sma_50": 49800.0,
            "is_squeeze": False,
        }

    def test_llm_response_parsed_correctly(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "regime": "trending_up",
                "confidence": 0.85,
                "reasoning": "ADX=35 rising, price near 20d high, EMA9 > EMA21",
            }),
            success=True,
            latency_ms=300,
        )
        agent = RegimeAgent(llm_client=client)
        result = agent.run(context=self._make_context(), instrument_key="NSE_EQ:RELIANCE")
        assert result["regime"] == "trending_up"
        assert result["confidence"] == 0.85

    def test_fallback_trending_up(self):
        """ADX > 25 + price_position > 60 = trending_up."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(
            context=self._make_context(adx=30.0, price_position=75.0),
            instrument_key="NSE_EQ:RELIANCE",
        )
        assert result["regime"] == "trending_up"

    def test_fallback_trending_down(self):
        """ADX > 25 + price_position < 40 = trending_down."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(
            context=self._make_context(adx=30.0, price_position=20.0),
            instrument_key="NSE_EQ:RELIANCE",
        )
        assert result["regime"] == "trending_down"

    def test_fallback_ranging(self):
        """ADX < 20 = ranging."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(
            context=self._make_context(adx=15.0, price_position=50.0),
            instrument_key="NSE_EQ:RELIANCE",
        )
        assert result["regime"] == "ranging"

    def test_fallback_volatile(self):
        """ATR percentile > 90 = volatile."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(
            context=self._make_context(atr_pct=95.0),
            instrument_key="NSE_EQ:RELIANCE",
        )
        assert result["regime"] == "volatile"

    def test_to_classification(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = RegimeAgent(llm_client=client)
        result = agent.run(
            context=self._make_context(adx=30.0, price_position=75.0),
        )
        classification = agent.to_classification(result)
        assert isinstance(classification, RegimeClassification)
        assert classification.regime == "trending_up"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_regime_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RegimeAgent**

Create `src/agents/regime_agent.py`:

```python
"""LLM-powered market regime detection agent.

Classifies the current market as trending_up, trending_down,
ranging, or volatile. Uses technical indicators as input and
the LLM for nuanced classification. Falls back to deterministic
rules when LLM is unavailable.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.models import RegimeClassification

VALID_REGIMES = {"trending_up", "trending_down", "ranging", "volatile"}

SYSTEM_PROMPT = """You are a market regime classification agent for Indian equity/derivatives markets.

Given technical indicator data, classify the market into one of:
- trending_up: Strong upward momentum, ADX > 25, price near highs
- trending_down: Strong downward momentum, ADX > 25, price near lows
- ranging: Sideways, ADX < 20, no clear direction
- volatile: Extreme ATR, erratic moves, high uncertainty

Respond ONLY with valid JSON:
{"regime": "<regime>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Rules:
- confidence < 0.3 means you are very uncertain — the pipeline will skip trading
- Consider ALL indicators together, not just one
- ATR percentile > 90 strongly suggests volatile regime
- Bollinger Band squeeze suggests potential breakout from ranging
"""


class RegimeAgent(BaseAgent):
    """Classifies market regime using LLM with deterministic fallback.

    Args:
        llm_client: Shared LLM client instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(llm_client)

    @property
    def agent_name(self) -> str:
        return "regime_agent"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        return (
            "Classify the market regime from these indicators:\n"
            f"ADX: {context.get('adx', 0):.1f} ({context.get('adx_trend', 'unknown')})\n"
            f"RSI: {context.get('rsi', 50):.1f}\n"
            f"ATR Percentile: {context.get('atr_percentile', 50):.1f}%\n"
            f"BB Width: {context.get('bb_width', 0):.2f} (percentile: {context.get('bb_width_percentile', 50):.1f}%)\n"
            f"Price Position in 20d Range: {context.get('price_position_in_range', 50):.1f}%\n"
            f"EMA 9: {context.get('ema_9', 0):.2f}, EMA 21: {context.get('ema_21', 0):.2f}\n"
            f"SMA 50: {context.get('sma_50', 0):.2f}\n"
            f"BB Squeeze: {context.get('is_squeeze', False)}\n"
        )

    def _build_system_prompt(self, lessons: list[str]) -> str:
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about regime misclassification:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        regime = data.get("regime", "ranging")
        if regime not in VALID_REGIMES:
            regime = "ranging"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "regime": regime,
            "confidence": confidence,
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic regime classification from indicators."""
        adx = context.get("adx", 20.0)
        atr_pct = context.get("atr_percentile", 50.0)
        price_pos = context.get("price_position_in_range", 50.0)

        # Volatile takes priority
        if atr_pct > 90:
            return {
                "regime": "volatile",
                "confidence": 0.7,
                "reasoning": f"Fallback: ATR percentile {atr_pct:.0f}% > 90",
            }

        # Trending
        if adx > 25:
            if price_pos > 60:
                return {
                    "regime": "trending_up",
                    "confidence": 0.6,
                    "reasoning": f"Fallback: ADX={adx:.0f} > 25, price near highs ({price_pos:.0f}%)",
                }
            elif price_pos < 40:
                return {
                    "regime": "trending_down",
                    "confidence": 0.6,
                    "reasoning": f"Fallback: ADX={adx:.0f} > 25, price near lows ({price_pos:.0f}%)",
                }

        # Ranging
        return {
            "regime": "ranging",
            "confidence": 0.5,
            "reasoning": f"Fallback: ADX={adx:.0f}, no clear trend",
        }

    def to_classification(self, result: dict[str, Any]) -> RegimeClassification:
        """Convert raw result dict to a RegimeClassification model."""
        return RegimeClassification(
            regime=result["regime"],
            confidence=result["confidence"],
            reasoning=result.get("reasoning", ""),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_regime_agent.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/regime_agent.py tests/test_regime_agent.py
git commit -m "feat: add LLM-powered market regime agent with fallback"
```

---

## Task 6: Signal Validation Agent

**Files:**
- Create: `src/agents/signal_validator.py`
- Test: `tests/test_signal_validator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_signal_validator.py`:

```python
"""Tests for the LLM-powered signal validation agent."""

import json
import pytest
from unittest.mock import MagicMock
from src.agents.signal_validator import SignalValidatorAgent
from src.agents.llm_client import LLMClient, LLMResponse


class TestSignalValidatorAgent:
    def _make_context(self, direction="BUY", regime="trending_up", confidence=0.8,
                      stop_loss=49000_00, target=52000_00, entry_price=50000_00):
        return {
            "signal": {
                "signal_id": "sig_001",
                "instrument_key": "NSE_EQ:RELIANCE",
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "strategy": "ema_crossover",
                "confidence": confidence,
            },
            "regime": regime,
            "sentiment_score": 60.0,
            "sentiment_classification": "greed",
        }

    def test_llm_approves_aligned_signal(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "action": "approve",
                "confidence": 0.9,
                "reasoning": "BUY signal aligned with trending_up regime",
            }),
            success=True,
            latency_ms=250,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context())
        assert result["action"] == "approve"

    def test_llm_rejects_counter_trend(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "action": "reject",
                "confidence": 0.85,
                "reasoning": "BUY signal in trending_down regime",
            }),
            success=True,
            latency_ms=200,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(regime="trending_down"))
        assert result["action"] == "reject"

    def test_llm_modifies_stop_loss(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "action": "modify",
                "adjusted_stop_loss": 48500_00,
                "adjusted_quantity_pct": 0.75,
                "confidence": 0.7,
                "reasoning": "Widening SL due to high volatility",
            }),
            success=True,
            latency_ms=350,
        )
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context())
        assert result["action"] == "modify"
        assert result["adjusted_stop_loss"] == 48500_00
        assert result["adjusted_quantity_pct"] == 0.75

    def test_fallback_approves_trend_aligned_buy(self):
        """BUY + trending_up + confidence > 0.5 = approve."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="BUY", regime="trending_up", confidence=0.7
        ))
        assert result["action"] == "approve"

    def test_fallback_rejects_counter_trend(self):
        """BUY + trending_down = reject."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="BUY", regime="trending_down"
        ))
        assert result["action"] == "reject"

    def test_fallback_rejects_low_confidence(self):
        """confidence < 0.5 = reject."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(confidence=0.3))
        assert result["action"] == "reject"

    def test_fallback_approves_ranging_mean_reversion(self):
        """SELL + ranging = approve (mean reversion is valid in ranging)."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="SELL", regime="ranging", confidence=0.7
        ))
        assert result["action"] == "approve"

    def test_fallback_checks_risk_reward(self):
        """Risk-reward < 1.5 = reject."""
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = SignalValidatorAgent(llm_client=client)
        result = agent.run(context=self._make_context(
            direction="BUY",
            regime="trending_up",
            confidence=0.8,
            entry_price=50000_00,
            stop_loss=49500_00,  # Risk: 500
            target=50600_00,    # Reward: 600, R:R = 1.2 < 1.5
        ))
        assert result["action"] == "reject"
        assert "risk-reward" in result["reasoning"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_signal_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SignalValidatorAgent**

Create `src/agents/signal_validator.py`:

```python
"""LLM-powered signal validation agent.

Evaluates raw trading signals from the strategy engine and decides
whether to approve, reject, or modify them based on market regime,
sentiment, risk-reward, and historical lessons.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.models import SignalDecision

VALID_ACTIONS = {"approve", "reject", "modify"}

SYSTEM_PROMPT = """You are a signal validation agent for an Indian equity/derivatives trading bot.

Your job is quality control: review trading signals and decide whether they should be executed.

For each signal you receive, decide:
- "approve": Signal is good, execute as-is
- "reject": Signal is bad, do not execute
- "modify": Signal has merit but needs adjustment (SL, target, or size)

Respond ONLY with valid JSON:
{
    "action": "approve|reject|modify",
    "adjusted_stop_loss": <paisa or null>,
    "adjusted_target": <paisa or null>,
    "adjusted_quantity_pct": <0.25 to 1.5, default 1.0>,
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence>"
}

Rules:
- REJECT signals that go against the current market regime (e.g., BUY in trending_down)
- REJECT signals with risk-reward ratio below 1.5:1
- MODIFY signals in volatile markets by reducing quantity (0.5-0.75x)
- APPROVE signals that align with regime and have good risk-reward
- extreme_fear sentiment: be extra cautious with BUY signals
- extreme_greed sentiment: be extra cautious with SELL signals
"""


class SignalValidatorAgent(BaseAgent):
    """Validates trading signals using LLM with deterministic fallback.

    Args:
        llm_client: Shared LLM client instance.
        min_risk_reward: Minimum risk-reward ratio for approval.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_risk_reward: float = 1.5,
    ) -> None:
        super().__init__(llm_client)
        self._min_rr = min_risk_reward

    @property
    def agent_name(self) -> str:
        return "signal_validator"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        sig = context.get("signal", {})
        return (
            "Validate this trading signal:\n"
            f"Direction: {sig.get('direction', 'unknown')}\n"
            f"Instrument: {sig.get('instrument_key', 'unknown')}\n"
            f"Strategy: {sig.get('strategy', 'unknown')}\n"
            f"Entry Price: {sig.get('entry_price', 0)} paisa\n"
            f"Stop Loss: {sig.get('stop_loss', 0)} paisa\n"
            f"Target: {sig.get('target', 0)} paisa\n"
            f"Signal Confidence: {sig.get('confidence', 0):.2f}\n\n"
            f"Market Regime: {context.get('regime', 'unknown')}\n"
            f"Sentiment Score: {context.get('sentiment_score', 50):.0f}/100 "
            f"({context.get('sentiment_classification', 'neutral')})\n"
        )

    def _build_system_prompt(self, lessons: list[str]) -> str:
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about signal quality:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        action = data.get("action", "reject")
        if action not in VALID_ACTIONS:
            action = "reject"

        return {
            "action": action,
            "adjusted_stop_loss": data.get("adjusted_stop_loss"),
            "adjusted_target": data.get("adjusted_target"),
            "adjusted_quantity_pct": float(data.get("adjusted_quantity_pct", 1.0)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic signal validation."""
        sig = context.get("signal", {})
        regime = context.get("regime", "unknown")
        direction = sig.get("direction", "").upper()
        confidence = sig.get("confidence", 0.0)
        entry = sig.get("entry_price", 0)
        sl = sig.get("stop_loss", 0)
        target = sig.get("target", 0)

        # Check confidence threshold
        if confidence < 0.5:
            return {
                "action": "reject",
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "adjusted_quantity_pct": 1.0,
                "confidence": 0.8,
                "reasoning": f"Fallback: signal confidence {confidence:.2f} < 0.5",
            }

        # Check risk-reward ratio
        if entry > 0 and sl > 0 and target > 0:
            if direction == "BUY":
                risk = entry - sl
                reward = target - entry
            else:
                risk = sl - entry
                reward = entry - target

            if risk > 0:
                rr = reward / risk
                if rr < self._min_rr:
                    return {
                        "action": "reject",
                        "adjusted_stop_loss": None,
                        "adjusted_target": None,
                        "adjusted_quantity_pct": 1.0,
                        "confidence": 0.85,
                        "reasoning": f"Fallback: risk-reward {rr:.2f} < {self._min_rr}",
                    }

        # Check regime alignment
        counter_trend = (
            (direction == "BUY" and regime == "trending_down")
            or (direction == "SELL" and regime == "trending_up")
        )
        if counter_trend:
            return {
                "action": "reject",
                "adjusted_stop_loss": None,
                "adjusted_target": None,
                "adjusted_quantity_pct": 1.0,
                "confidence": 0.75,
                "reasoning": f"Fallback: {direction} is counter-trend in {regime}",
            }

        return {
            "action": "approve",
            "adjusted_stop_loss": None,
            "adjusted_target": None,
            "adjusted_quantity_pct": 1.0,
            "confidence": 0.6,
            "reasoning": f"Fallback: signal passes basic checks",
        }

    def to_decision(self, result: dict[str, Any], signal_id: str = "") -> SignalDecision:
        """Convert raw result dict to a SignalDecision model."""
        return SignalDecision(
            signal_id=signal_id,
            action=result["action"],
            confidence=result["confidence"],
            reasoning=result.get("reasoning", ""),
            adjusted_stop_loss=result.get("adjusted_stop_loss"),
            adjusted_target=result.get("adjusted_target"),
            adjusted_quantity_pct=result.get("adjusted_quantity_pct", 1.0),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_signal_validator.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/signal_validator.py tests/test_signal_validator.py
git commit -m "feat: add LLM-powered signal validation agent"
```

---

## Task 7: News Sentiment Agent

**Files:**
- Create: `src/agents/news_agent.py`
- Create: `src/agents/sentiment.py`
- Test: `tests/test_news_agent.py`
- Test: `tests/test_sentiment.py`

- [ ] **Step 1: Write the failing test for news agent**

Create `tests/test_news_agent.py`:

```python
"""Tests for the news sentiment agent."""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.agents.news_agent import NewsAgent
from src.agents.llm_client import LLMClient, LLMResponse


class TestNewsAgent:
    def test_parse_llm_sentiment(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        client.invoke.return_value = LLMResponse(
            content=json.dumps({
                "sentiment_score": 0.3,
                "confidence": 0.75,
                "reasoning": "Mixed signals: positive earnings but trade tensions",
            }),
            success=True,
            latency_ms=400,
        )
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={
            "headlines": [
                "Nifty hits all-time high on strong FII inflows",
                "US-China trade war fears weigh on emerging markets",
            ]
        })
        assert -1.0 <= result["sentiment_score"] <= 1.0
        assert result["confidence"] > 0

    def test_fallback_returns_neutral(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = False
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={"headlines": ["Some news"]})
        assert result["sentiment_score"] == 0.0
        assert result["confidence"] > 0

    def test_empty_headlines_returns_neutral(self):
        client = MagicMock(spec=LLMClient)
        client.is_configured = True
        agent = NewsAgent(llm_client=client)
        result = agent.run(context={"headlines": []})
        assert result["sentiment_score"] == 0.0

    @patch("src.agents.news_agent.feedparser")
    def test_fetch_headlines(self, mock_fp):
        mock_fp.parse.return_value = MagicMock(
            entries=[
                MagicMock(title="Sensex rallies 500 points", source={"title": "ET"}),
                MagicMock(title="RBI holds rates steady", source={"title": "LiveMint"}),
            ]
        )
        client = MagicMock(spec=LLMClient)
        agent = NewsAgent(llm_client=client)
        headlines = agent.fetch_headlines(query="indian stock market")
        assert len(headlines) == 2
        assert "Sensex" in headlines[0].title
```

- [ ] **Step 2: Write the failing test for sentiment composite**

Create `tests/test_sentiment.py`:

```python
"""Tests for the composite Market Mood Index."""

import pytest
from src.agents.sentiment import MarketMoodCalculator, classify_mood


class TestClassifyMood:
    def test_extreme_fear(self):
        assert classify_mood(10.0) == "extreme_fear"

    def test_fear(self):
        assert classify_mood(30.0) == "fear"

    def test_neutral(self):
        assert classify_mood(50.0) == "neutral"

    def test_greed(self):
        assert classify_mood(70.0) == "greed"

    def test_extreme_greed(self):
        assert classify_mood(90.0) == "extreme_greed"


class TestMarketMoodCalculator:
    def test_default_weights(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=0.5,
            volatility_score=60.0,
            breadth_score=70.0,
        )
        assert 0 <= score.score <= 100
        assert score.classification in {
            "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
        }

    def test_all_positive(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=1.0,
            volatility_score=90.0,
            breadth_score=90.0,
        )
        assert score.score > 70
        assert score.classification in {"greed", "extreme_greed"}

    def test_all_negative(self):
        calc = MarketMoodCalculator()
        score = calc.calculate(
            news_sentiment=-1.0,
            volatility_score=10.0,
            breadth_score=10.0,
        )
        assert score.score < 30
        assert score.classification in {"extreme_fear", "fear"}

    def test_custom_weights(self):
        calc = MarketMoodCalculator(
            news_weight=0.5,
            volatility_weight=0.3,
            breadth_weight=0.2,
        )
        score = calc.calculate(
            news_sentiment=0.0,
            volatility_score=50.0,
            breadth_score=50.0,
        )
        assert 40 <= score.score <= 60
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_news_agent.py tests/test_sentiment.py -v`
Expected: FAIL

- [ ] **Step 4: Implement NewsAgent**

Create `src/agents/news_agent.py`:

```python
"""News sentiment agent using RSS feeds and LLM analysis.

Fetches financial headlines from Google News RSS and uses the LLM
to score overall market sentiment. Falls back to neutral when
LLM is unavailable.
"""

from __future__ import annotations

import json
import time
from typing import Any

import feedparser
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.models import NewsItem

SYSTEM_PROMPT = """You are a financial news sentiment analyst for the Indian stock market.

Given a batch of recent headlines, score the overall market sentiment.

Respond ONLY with valid JSON:
{"sentiment_score": <-1.0 to 1.0>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Scoring guide:
- -1.0: Extremely bearish (crash fears, major crisis)
- -0.5: Moderately bearish (rate hikes, poor earnings)
-  0.0: Neutral (mixed signals)
-  0.5: Moderately bullish (good earnings, FII inflows)
-  1.0: Extremely bullish (breakout, policy support)
"""

NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=indian+stock+market&hl=en-IN&gl=IN",
    "https://news.google.com/rss/search?q=nifty+sensex&hl=en-IN&gl=IN",
]


class NewsAgent(BaseAgent):
    """Analyzes financial news sentiment using LLM.

    Args:
        llm_client: Shared LLM client instance.
        cache_ttl_s: How long to cache fetched headlines in seconds.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache_ttl_s: int = 300,
    ) -> None:
        super().__init__(llm_client)
        self._cache_ttl_s = cache_ttl_s
        self._headline_cache: list[NewsItem] = []
        self._cache_time: float = 0.0

    @property
    def agent_name(self) -> str:
        return "news_agent"

    def fetch_headlines(
        self, query: str = "indian stock market", max_items: int = 15
    ) -> list[NewsItem]:
        """Fetch recent headlines from Google News RSS.

        Args:
            query: Search query for news.
            max_items: Maximum headlines to return.

        Returns:
            List of NewsItem objects.
        """
        now = time.monotonic()
        if self._headline_cache and (now - self._cache_time) < self._cache_ttl_s:
            return self._headline_cache

        items: list[NewsItem] = []
        for feed_url in NEWS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_items]:
                    source_name = ""
                    if hasattr(entry, "source") and hasattr(entry.source, "get"):
                        source_name = entry.source.get("title", "")
                    elif hasattr(entry, "source") and hasattr(entry.source, "title"):
                        source_name = entry.source.title
                    items.append(NewsItem(
                        title=entry.title,
                        source=source_name,
                    ))
            except Exception as e:
                logger.warning("Failed to fetch news from {}: {}", feed_url, e)

        self._headline_cache = items[:max_items]
        self._cache_time = now
        logger.debug("Fetched {} headlines", len(self._headline_cache))
        return self._headline_cache

    def _build_prompt(self, context: dict[str, Any]) -> str:
        headlines = context.get("headlines", [])
        if not headlines:
            return "No headlines available."

        text = "Analyze sentiment of these recent Indian market headlines:\n\n"
        for i, h in enumerate(headlines[:15], 1):
            if isinstance(h, str):
                text += f"{i}. {h}\n"
            else:
                text += f"{i}. {h}\n"
        return text

    def _build_system_prompt(self, lessons: list[str]) -> str:
        prompt = SYSTEM_PROMPT
        if lessons:
            prompt += "\n\nPast lessons about sentiment analysis:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        return prompt

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        score = float(data.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))

        return {
            "sentiment_score": score,
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return neutral sentiment when LLM unavailable."""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.3,
            "reasoning": "Fallback: no LLM available, assuming neutral",
        }

    def run(
        self,
        context: dict[str, Any],
        lessons: list[str] | None = None,
        instrument_key: str = "",
    ) -> dict[str, Any]:
        """Override run to handle empty headlines."""
        headlines = context.get("headlines", [])
        if not headlines:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.3,
                "reasoning": "No headlines to analyze",
            }
        return super().run(context, lessons, instrument_key)
```

- [ ] **Step 5: Implement MarketMoodCalculator**

Create `src/agents/sentiment.py`:

```python
"""Composite Market Mood Index calculator.

Combines news sentiment, volatility, and market breadth into a
single 0-100 score classified as extreme_fear to extreme_greed.
"""

from __future__ import annotations

from src.agents.models import SentimentScore


def classify_mood(score: float) -> str:
    """Classify a 0-100 mood score.

    Args:
        score: Market Mood Index (0-100).

    Returns:
        Classification string.
    """
    if score < 20:
        return "extreme_fear"
    if score < 40:
        return "fear"
    if score < 60:
        return "neutral"
    if score < 80:
        return "greed"
    return "extreme_greed"


class MarketMoodCalculator:
    """Calculates composite Market Mood Index.

    Combines three inputs into a 0-100 score:
    - News sentiment (-1 to 1, normalized to 0-100)
    - Volatility score (0-100, high vol = low score)
    - Market breadth (0-100, advance/decline ratio)

    Args:
        news_weight: Weight for news sentiment component.
        volatility_weight: Weight for volatility component.
        breadth_weight: Weight for market breadth component.
    """

    def __init__(
        self,
        news_weight: float = 0.35,
        volatility_weight: float = 0.35,
        breadth_weight: float = 0.30,
    ) -> None:
        total = news_weight + volatility_weight + breadth_weight
        self._news_w = news_weight / total
        self._vol_w = volatility_weight / total
        self._breadth_w = breadth_weight / total

    def calculate(
        self,
        news_sentiment: float,
        volatility_score: float,
        breadth_score: float,
    ) -> SentimentScore:
        """Calculate the composite Market Mood Index.

        Args:
            news_sentiment: -1.0 to 1.0 from news agent.
            volatility_score: 0-100 (inverted: high vol = low score).
            breadth_score: 0-100 from advance/decline ratio.

        Returns:
            SentimentScore with composite score and classification.
        """
        # Normalize news sentiment from [-1, 1] to [0, 100]
        news_normalized = (news_sentiment + 1.0) * 50.0

        # Clamp all components to [0, 100]
        news_normalized = max(0.0, min(100.0, news_normalized))
        volatility_score = max(0.0, min(100.0, volatility_score))
        breadth_score = max(0.0, min(100.0, breadth_score))

        # Weighted composite
        score = (
            news_normalized * self._news_w
            + volatility_score * self._vol_w
            + breadth_score * self._breadth_w
        )

        score = max(0.0, min(100.0, score))
        classification = classify_mood(score)

        return SentimentScore(
            score=score,
            classification=classification,
            news_score=news_sentiment,
            volatility_score=volatility_score,
            breadth_score=breadth_score,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_news_agent.py tests/test_sentiment.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/news_agent.py src/agents/sentiment.py tests/test_news_agent.py tests/test_sentiment.py
git commit -m "feat: add news sentiment agent and Market Mood Index"
```

---

## Task 8: Trade Outcome Analyzer (MAE/MFE)

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/outcome_analyzer.py`
- Test: `tests/test_outcome_analyzer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_outcome_analyzer.py`:

```python
"""Tests for trade outcome analyzer with MAE/MFE metrics."""

import pytest
from src.memory.outcome_analyzer import TradeOutcome, OutcomeAnalyzer


class TestOutcomeAnalyzer:
    def test_winning_trade_metrics(self):
        outcome = TradeOutcome(
            trade_id="t001",
            strategy="ema_crossover",
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            entry_price=50000_00,
            exit_price=51000_00,
            stop_loss=49000_00,
            target=52000_00,
            quantity=10,
            realized_pnl=100000_00,  # +1000 INR
            max_adverse_excursion=300_00,  # Price went 3 INR against
            max_favorable_excursion=1200_00,  # Price went 12 INR in favor
            holding_seconds=3600,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["was_win"]
        assert analysis["mae_pct"] == pytest.approx(0.06, rel=0.01)  # 300/50000 * 100
        assert analysis["mfe_pct"] == pytest.approx(0.24, rel=0.01)  # 1200/50000 * 100
        assert analysis["efficiency"] == pytest.approx(0.833, rel=0.01)  # 1000/1200

    def test_losing_trade_metrics(self):
        outcome = TradeOutcome(
            trade_id="t002",
            strategy="rsi_reversal",
            instrument_key="NSE_EQ:TCS",
            direction="BUY",
            entry_price=40000_00,
            exit_price=39000_00,
            stop_loss=38500_00,
            target=42000_00,
            quantity=5,
            realized_pnl=-50000_00,  # -500 INR
            max_adverse_excursion=1200_00,
            max_favorable_excursion=300_00,
            holding_seconds=600,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert not analysis["was_win"]
        assert analysis["efficiency"] < 0  # Negative for losers

    def test_premature_exit_detection(self):
        """Exit captured < 50% of MFE = premature exit."""
        outcome = TradeOutcome(
            trade_id="t003",
            strategy="vwap_breakout",
            instrument_key="NSE_EQ:INFY",
            direction="BUY",
            entry_price=15000_00,
            exit_price=15200_00,
            stop_loss=14800_00,
            target=16000_00,
            quantity=20,
            realized_pnl=40000_00,  # +200 gain
            max_adverse_excursion=100_00,
            max_favorable_excursion=800_00,  # MFE = 800, captured only 200 = 25%
            holding_seconds=1800,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["was_win"]
        assert analysis["premature_exit"]
        assert analysis["efficiency"] < 0.5

    def test_stop_loss_hit_detection(self):
        """MAE approximately equals entry - stop_loss = stop hit."""
        outcome = TradeOutcome(
            trade_id="t004",
            strategy="supertrend",
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            entry_price=50000_00,
            exit_price=49050_00,
            stop_loss=49000_00,
            target=52000_00,
            quantity=10,
            realized_pnl=-95000_00,
            max_adverse_excursion=1000_00,  # ~equal to SL distance of 1000
            max_favorable_excursion=200_00,
            holding_seconds=300,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["stop_loss_hit"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_outcome_analyzer.py -v`
Expected: FAIL

- [ ] **Step 3: Create memory package init**

Create `src/memory/__init__.py`:

```python
"""Self-improving trade memory system."""
```

- [ ] **Step 4: Implement OutcomeAnalyzer**

Create `src/memory/outcome_analyzer.py`:

```python
"""Trade outcome analyzer with MAE/MFE metrics.

Computes Maximum Adverse Excursion (MAE), Maximum Favorable Excursion
(MFE), efficiency, and flags like premature_exit and stop_loss_hit.
All monetary values in paisa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class TradeOutcome:
    """Raw data for a completed trade.

    Attributes:
        trade_id: Unique trade identifier.
        strategy: Strategy that generated this trade.
        instrument_key: Instrument traded.
        direction: BUY or SELL.
        entry_price: Entry price in paisa.
        exit_price: Exit price in paisa.
        stop_loss: Original stop loss in paisa.
        target: Original target in paisa.
        quantity: Number of units.
        realized_pnl: Total P&L in paisa.
        max_adverse_excursion: Maximum price move against position in paisa.
        max_favorable_excursion: Maximum price move in favor in paisa.
        holding_seconds: Time held in seconds.
    """

    trade_id: str
    strategy: str
    instrument_key: str
    direction: str
    entry_price: int
    exit_price: int
    stop_loss: int
    target: int
    quantity: int
    realized_pnl: int
    max_adverse_excursion: int
    max_favorable_excursion: int
    holding_seconds: int


class OutcomeAnalyzer:
    """Analyzes trade outcomes to extract performance metrics.

    Computes MAE/MFE percentages, trade efficiency, and detects
    patterns like premature exits and stop-loss hits.
    """

    def analyze(self, outcome: TradeOutcome) -> dict[str, Any]:
        """Analyze a single trade outcome.

        Args:
            outcome: Completed trade data with MAE/MFE.

        Returns:
            Dictionary with computed metrics and flags.
        """
        was_win = outcome.realized_pnl > 0
        entry = outcome.entry_price

        # MAE/MFE as percentage of entry price
        mae_pct = (outcome.max_adverse_excursion / entry * 100) if entry > 0 else 0.0
        mfe_pct = (outcome.max_favorable_excursion / entry * 100) if entry > 0 else 0.0

        # Efficiency: how much of MFE was captured
        # Positive for winners, negative for losers
        if outcome.max_favorable_excursion > 0:
            actual_gain = abs(outcome.exit_price - outcome.entry_price)
            efficiency = actual_gain / outcome.max_favorable_excursion
            if not was_win:
                efficiency = -efficiency
        else:
            efficiency = 0.0

        # Premature exit: won but captured less than 50% of MFE
        premature_exit = was_win and efficiency < 0.5

        # Late exit: won but gave back more than 40% of MFE
        if was_win and outcome.max_favorable_excursion > 0:
            actual_capture = abs(outcome.exit_price - outcome.entry_price)
            giveback = outcome.max_favorable_excursion - actual_capture
            late_exit = giveback > 0.4 * outcome.max_favorable_excursion
        else:
            late_exit = False

        # Stop loss hit: MAE is within 10% of SL distance
        if outcome.direction == "BUY":
            sl_distance = outcome.entry_price - outcome.stop_loss
        else:
            sl_distance = outcome.stop_loss - outcome.entry_price

        if sl_distance > 0:
            stop_loss_hit = outcome.max_adverse_excursion >= sl_distance * 0.9
        else:
            stop_loss_hit = False

        # Quick stop: hit SL within 10 minutes
        quick_stop = stop_loss_hit and outcome.holding_seconds < 600

        analysis = {
            "trade_id": outcome.trade_id,
            "strategy": outcome.strategy,
            "instrument_key": outcome.instrument_key,
            "direction": outcome.direction,
            "was_win": was_win,
            "pnl_paisa": outcome.realized_pnl,
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "efficiency": efficiency,
            "premature_exit": premature_exit,
            "late_exit": late_exit,
            "stop_loss_hit": stop_loss_hit,
            "quick_stop": quick_stop,
            "holding_seconds": outcome.holding_seconds,
        }

        logger.debug(
            "Outcome analysis | {} | win={} | efficiency={:.2f} | "
            "premature={} | sl_hit={} | quick_stop={}",
            outcome.trade_id,
            was_win,
            efficiency,
            premature_exit,
            stop_loss_hit,
            quick_stop,
        )

        return analysis
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_outcome_analyzer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/memory/__init__.py src/memory/outcome_analyzer.py tests/test_outcome_analyzer.py
git commit -m "feat: add trade outcome analyzer with MAE/MFE metrics"
```

---

## Task 9: Mistake Classifier

**Files:**
- Create: `src/memory/mistake_classifier.py`
- Test: `tests/test_mistake_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mistake_classifier.py`:

```python
"""Tests for rule-based mistake classifier."""

import pytest
from src.memory.mistake_classifier import MistakeClassifier, MistakeCategory


class TestMistakeClassifier:
    def _make_analysis(self, **overrides):
        base = {
            "trade_id": "t001",
            "strategy": "ema_crossover",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "was_win": False,
            "pnl_paisa": -50000_00,
            "mae_pct": 2.0,
            "mfe_pct": 0.5,
            "efficiency": -0.8,
            "premature_exit": False,
            "late_exit": False,
            "stop_loss_hit": True,
            "quick_stop": False,
            "holding_seconds": 1800,
        }
        base.update(overrides)
        return base

    def test_quick_stop_classified_as_stop_too_tight(self):
        analysis = self._make_analysis(
            quick_stop=True,
            stop_loss_hit=True,
            holding_seconds=300,
        )
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.STOP_TOO_TIGHT in categories

    def test_regime_mismatch(self):
        analysis = self._make_analysis(direction="BUY")
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_down")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.REGIME_MISMATCH in categories

    def test_premature_exit_classified(self):
        analysis = self._make_analysis(was_win=True, premature_exit=True, pnl_paisa=10000_00)
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.PREMATURE_EXIT in categories

    def test_late_exit_classified(self):
        analysis = self._make_analysis(was_win=True, late_exit=True, pnl_paisa=10000_00)
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="ranging")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.LATE_EXIT in categories

    def test_winning_trade_no_mistakes(self):
        analysis = self._make_analysis(
            was_win=True,
            pnl_paisa=100000_00,
            premature_exit=False,
            late_exit=False,
            stop_loss_hit=False,
        )
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        assert len(mistakes) == 0

    def test_severity_levels(self):
        classifier = MistakeClassifier()
        analysis = self._make_analysis(direction="BUY", quick_stop=True, holding_seconds=200)
        mistakes = classifier.classify(analysis, regime="trending_down")
        # Should have regime_mismatch (high) and stop_too_tight (medium)
        severities = {m.category: m.severity for m in mistakes}
        assert severities[MistakeCategory.REGIME_MISMATCH] == "high"
        assert severities[MistakeCategory.STOP_TOO_TIGHT] == "medium"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_mistake_classifier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MistakeClassifier**

Create `src/memory/mistake_classifier.py`:

```python
"""Rule-based trade mistake classifier.

Analyzes trade outcome metrics and classifies mistakes into
categories with severity levels. These feed the memory system
so agents can learn from past errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class MistakeCategory(str, Enum):
    """Types of trading mistakes."""

    REGIME_MISMATCH = "regime_mismatch"
    STOP_TOO_TIGHT = "stop_too_tight"
    STOP_TOO_LOOSE = "stop_too_loose"
    PREMATURE_EXIT = "premature_exit"
    LATE_EXIT = "late_exit"
    POOR_TIMING = "poor_timing"
    OVERTRADING = "overtrading"
    POSITION_SIZING = "position_sizing"
    CHASING = "chasing"
    SIGNAL_QUALITY = "signal_quality"


SEVERITY_SCORES = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


@dataclass
class Mistake:
    """A classified trading mistake.

    Attributes:
        category: Type of mistake.
        severity: critical, high, medium, or low.
        description: Human-readable description.
        lesson: Actionable lesson for future trades.
        score: Severity score for weighting.
    """

    category: MistakeCategory
    severity: str
    description: str
    lesson: str

    @property
    def score(self) -> float:
        return SEVERITY_SCORES.get(self.severity, 1.0)


class MistakeClassifier:
    """Classifies trade mistakes from outcome analysis.

    Uses deterministic rules to identify common trading errors.
    Each rule checks a specific condition and produces a Mistake
    with category, severity, and actionable lesson.
    """

    def classify(
        self, analysis: dict[str, Any], regime: str = "unknown"
    ) -> list[Mistake]:
        """Classify mistakes in a trade outcome.

        Args:
            analysis: Output from OutcomeAnalyzer.analyze().
            regime: Current market regime when the trade was taken.

        Returns:
            List of identified Mistake objects (may be empty for good trades).
        """
        mistakes: list[Mistake] = []
        direction = analysis.get("direction", "").upper()
        was_win = analysis.get("was_win", True)

        # --- Regime mismatch ---
        counter_trend = (
            (direction == "BUY" and regime == "trending_down")
            or (direction == "SELL" and regime == "trending_up")
        )
        if counter_trend:
            mistakes.append(Mistake(
                category=MistakeCategory.REGIME_MISMATCH,
                severity="high",
                description=f"{direction} signal taken in {regime} regime",
                lesson=f"Avoid {direction} trades when market is {regime}",
            ))

        # --- Stop too tight ---
        if analysis.get("quick_stop") and analysis.get("stop_loss_hit"):
            mistakes.append(Mistake(
                category=MistakeCategory.STOP_TOO_TIGHT,
                severity="medium",
                description="Stop loss hit within 10 minutes of entry",
                lesson="Widen stop loss or wait for better entry in this volatility",
            ))

        # --- Stop too loose ---
        if not was_win and not analysis.get("stop_loss_hit") and analysis.get("mae_pct", 0) > 3.0:
            mistakes.append(Mistake(
                category=MistakeCategory.STOP_TOO_LOOSE,
                severity="medium",
                description=f"Loss of {analysis.get('mae_pct', 0):.1f}% without hitting stop loss",
                lesson="Tighten stop loss or use trailing stop for this strategy",
            ))

        # --- Premature exit ---
        if analysis.get("premature_exit"):
            mistakes.append(Mistake(
                category=MistakeCategory.PREMATURE_EXIT,
                severity="low",
                description="Exited winning trade capturing less than 50% of MFE",
                lesson="Let winners run longer or use trailing stop instead of fixed target",
            ))

        # --- Late exit ---
        if analysis.get("late_exit"):
            mistakes.append(Mistake(
                category=MistakeCategory.LATE_EXIT,
                severity="low",
                description="Gave back more than 40% of maximum favorable excursion",
                lesson="Use trailing stop to protect profits once in the money",
            ))

        if mistakes:
            logger.info(
                "Classified {} mistakes for trade {} | categories: {}",
                len(mistakes),
                analysis.get("trade_id", "?"),
                [m.category.value for m in mistakes],
            )

        return mistakes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_mistake_classifier.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/mistake_classifier.py tests/test_mistake_classifier.py
git commit -m "feat: add rule-based trade mistake classifier"
```

---

## Task 10: Memory Database with Time-Decay

**Files:**
- Create: `src/memory/memory_db.py`
- Modify: `src/persistence/models.py` (add MemoryLessonRecord table)
- Test: `tests/test_memory_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_db.py`:

```python
"""Tests for the memory lesson database with time-decay."""

import pytest
from datetime import datetime, timedelta
from src.memory.memory_db import MemoryDB, MemoryLesson


class TestMemoryDB:
    def test_store_and_retrieve_lesson(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001",
            category="regime_mismatch",
            strategy="ema_crossover",
            regime="trending_down",
            description="Avoid BUY in strong downtrend",
            severity="high",
            base_score=1.5,
            created_at=datetime.now(),
        )
        db.store(lesson)
        retrieved = db.get_relevant_lessons(regime="trending_down", strategy="ema_crossover")
        assert len(retrieved) == 1
        assert retrieved[0].lesson_id == "L001"

    def test_lessons_filtered_by_regime(self):
        db = MemoryDB()
        db.store(MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="test1", severity="high",
            base_score=1.5, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L002", category="stop_too_tight", strategy="ema_crossover",
            regime="volatile", description="test2", severity="medium",
            base_score=1.0, created_at=datetime.now(),
        ))
        results = db.get_relevant_lessons(regime="trending_down")
        assert len(results) == 1
        assert results[0].lesson_id == "L001"

    def test_time_decay_reduces_score(self):
        db = MemoryDB(decay_rate=0.05, decay_start_days=0)
        old_lesson = MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="old lesson", severity="high",
            base_score=1.5, created_at=datetime.now() - timedelta(weeks=4),
        )
        db.store(old_lesson)
        retrieved = db.get_relevant_lessons(regime="trending_down")
        assert len(retrieved) == 1
        # Score should be decayed: 1.5 * (1-0.05)^4 = ~1.22
        assert retrieved[0].effective_score < 1.5

    def test_boost_lesson(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="useful lesson", severity="high",
            base_score=1.5, created_at=datetime.now(),
        )
        db.store(lesson)
        db.boost_lesson("L001", factor=1.1)
        retrieved = db.get_relevant_lessons(regime="trending_down")
        assert retrieved[0].base_score == pytest.approx(1.65, rel=0.01)

    def test_boost_capped_at_2x(self):
        db = MemoryDB()
        lesson = MemoryLesson(
            lesson_id="L001", category="test", strategy="test",
            regime="test", description="test", severity="high",
            base_score=1.5, created_at=datetime.now(),
        )
        db.store(lesson)
        # Boost many times
        for _ in range(20):
            db.boost_lesson("L001", factor=1.1)
        retrieved = db.get_relevant_lessons(regime="test")
        assert retrieved[0].base_score <= 3.0  # 1.5 * 2.0 cap

    def test_max_lessons_returned(self):
        db = MemoryDB()
        for i in range(10):
            db.store(MemoryLesson(
                lesson_id=f"L{i:03d}", category="test", strategy="test",
                regime="test", description=f"lesson {i}", severity="high",
                base_score=float(i), created_at=datetime.now(),
            ))
        results = db.get_relevant_lessons(regime="test", max_lessons=5)
        assert len(results) == 5
        # Should return highest-scored first
        assert results[0].base_score >= results[-1].base_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_memory_db.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MemoryDB**

Create `src/memory/memory_db.py`:

```python
"""In-memory lesson database with time-decay scoring.

Stores trade lessons with severity-based scores that decay over
time. Lessons can be boosted when they prove useful. Retrieval
is filtered by regime and strategy for contextual injection.

For persistence, lessons are also stored via the existing
LearningPersistence layer. This in-memory store is the
fast-path for pipeline injection.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


@dataclass
class MemoryLesson:
    """A lesson stored in the memory system.

    Attributes:
        lesson_id: Unique identifier.
        category: Mistake category (e.g., regime_mismatch).
        strategy: Strategy this lesson applies to.
        regime: Market regime when the mistake occurred.
        description: Human-readable lesson text.
        severity: critical, high, medium, low.
        base_score: Severity score (can be boosted).
        created_at: When the lesson was created.
        times_injected: How many times injected into agent context.
        times_useful: How many times the lesson proved useful.
        effective_score: Score after time-decay (computed on retrieval).
    """

    lesson_id: str
    category: str
    strategy: str
    regime: str
    description: str
    severity: str
    base_score: float
    created_at: datetime
    times_injected: int = 0
    times_useful: int = 0
    effective_score: float = 0.0


class MemoryDB:
    """In-memory lesson store with time-decay and boost.

    Args:
        decay_rate: Exponential decay rate per week (default 0.05 = 5%).
        decay_start_days: Days before decay begins (default 30).
        max_base_score_multiplier: Maximum boost cap as multiplier of
            original score (default 2.0).
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        decay_start_days: int = 30,
        max_base_score_multiplier: float = 2.0,
    ) -> None:
        self._lock = threading.Lock()
        self._lessons: dict[str, MemoryLesson] = {}
        self._decay_rate = decay_rate
        self._decay_start_days = decay_start_days
        self._max_multiplier = max_base_score_multiplier
        # Track original scores for cap calculation
        self._original_scores: dict[str, float] = {}

    def store(self, lesson: MemoryLesson) -> None:
        """Store a lesson in the database.

        Args:
            lesson: The lesson to store.
        """
        with self._lock:
            self._lessons[lesson.lesson_id] = lesson
            if lesson.lesson_id not in self._original_scores:
                self._original_scores[lesson.lesson_id] = lesson.base_score
        logger.debug("Stored lesson {} | category={}", lesson.lesson_id, lesson.category)

    def get_relevant_lessons(
        self,
        regime: str = "",
        strategy: str = "",
        max_lessons: int = 10,
    ) -> list[MemoryLesson]:
        """Retrieve lessons relevant to current context.

        Filters by regime and/or strategy, applies time-decay,
        and returns top N by effective score.

        Args:
            regime: Current market regime to filter by.
            strategy: Current strategy to filter by.
            max_lessons: Maximum lessons to return.

        Returns:
            List of MemoryLesson sorted by effective_score descending.
        """
        now = datetime.now()

        with self._lock:
            candidates = []
            for lesson in self._lessons.values():
                # Filter by regime if specified
                if regime and lesson.regime != regime:
                    continue
                # Filter by strategy if specified
                if strategy and lesson.strategy != strategy:
                    continue

                # Calculate effective score with time decay
                effective = self._apply_decay(lesson.base_score, lesson.created_at, now)
                lesson.effective_score = effective
                candidates.append(lesson)

        # Sort by effective score descending
        candidates.sort(key=lambda l: l.effective_score, reverse=True)
        return candidates[:max_lessons]

    def boost_lesson(self, lesson_id: str, factor: float = 1.1) -> None:
        """Boost a lesson's score when it proves useful.

        Args:
            lesson_id: ID of the lesson to boost.
            factor: Multiplicative boost factor.
        """
        with self._lock:
            lesson = self._lessons.get(lesson_id)
            if not lesson:
                return

            original = self._original_scores.get(lesson_id, lesson.base_score)
            max_score = original * self._max_multiplier
            lesson.base_score = min(lesson.base_score * factor, max_score)
            lesson.times_useful += 1

        logger.debug(
            "Boosted lesson {} | new_score={:.2f}",
            lesson_id,
            lesson.base_score,
        )

    def record_injection(self, lesson_id: str) -> None:
        """Record that a lesson was injected into agent context.

        Args:
            lesson_id: ID of the lesson that was injected.
        """
        with self._lock:
            lesson = self._lessons.get(lesson_id)
            if lesson:
                lesson.times_injected += 1

    def _apply_decay(
        self, base_score: float, created_at: datetime, now: datetime
    ) -> float:
        """Apply exponential time-decay to a score.

        Args:
            base_score: Original score.
            created_at: When the lesson was created.
            now: Current time.

        Returns:
            Decayed score.
        """
        age = now - created_at
        age_days = age.total_seconds() / 86400

        if age_days <= self._decay_start_days:
            return base_score

        # Weeks past the decay start
        decay_weeks = (age_days - self._decay_start_days) / 7.0
        decay_factor = (1 - self._decay_rate) ** decay_weeks
        return base_score * decay_factor

    def get_all_lessons(self) -> list[MemoryLesson]:
        """Get all stored lessons."""
        with self._lock:
            return list(self._lessons.values())

    def remove_lesson(self, lesson_id: str) -> None:
        """Remove a lesson from the database."""
        with self._lock:
            self._lessons.pop(lesson_id, None)
            self._original_scores.pop(lesson_id, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_memory_db.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/memory_db.py tests/test_memory_db.py
git commit -m "feat: add memory lesson database with time-decay and boost"
```

---

## Task 11: Memory Injector

**Files:**
- Create: `src/memory/injector.py`
- Test: `tests/test_injector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_injector.py`:

```python
"""Tests for memory injector that formats lessons for agent context."""

import pytest
from datetime import datetime
from src.memory.injector import MemoryInjector
from src.memory.memory_db import MemoryDB, MemoryLesson


class TestMemoryInjector:
    def _populate_db(self, db: MemoryDB) -> None:
        db.store(MemoryLesson(
            lesson_id="L001", category="regime_mismatch", strategy="ema_crossover",
            regime="trending_down", description="Avoid BUY in strong downtrend",
            severity="high", base_score=1.5, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L002", category="stop_too_tight", strategy="ema_crossover",
            regime="volatile", description="Widen SL in volatile markets by 1.5x ATR",
            severity="medium", base_score=1.0, created_at=datetime.now(),
        ))
        db.store(MemoryLesson(
            lesson_id="L003", category="premature_exit", strategy="vwap_breakout",
            regime="trending_up", description="Use trailing stop instead of fixed target",
            severity="low", base_score=0.5, created_at=datetime.now(),
        ))

    def test_get_lessons_for_regime(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(
            agent_name="regime_agent",
            regime="trending_down",
        )
        assert len(lessons) >= 1
        assert any("downtrend" in l.lower() for l in lessons)

    def test_get_lessons_for_strategy(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(
            agent_name="signal_validator",
            regime="volatile",
            strategy="ema_crossover",
        )
        assert len(lessons) >= 1
        assert any("volatile" in l.lower() or "widen" in l.lower() for l in lessons)

    def test_max_lessons_respected(self):
        db = MemoryDB()
        for i in range(20):
            db.store(MemoryLesson(
                lesson_id=f"L{i:03d}", category="test", strategy="test",
                regime="test", description=f"Lesson {i}",
                severity="medium", base_score=1.0, created_at=datetime.now(),
            ))
        injector = MemoryInjector(memory_db=db, max_lessons=3)
        lessons = injector.get_lessons_for_agent(agent_name="test", regime="test")
        assert len(lessons) <= 3

    def test_empty_db_returns_empty(self):
        db = MemoryDB()
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        lessons = injector.get_lessons_for_agent(agent_name="test", regime="unknown")
        assert lessons == []

    def test_injection_recorded(self):
        db = MemoryDB()
        self._populate_db(db)
        injector = MemoryInjector(memory_db=db, max_lessons=5)
        injector.get_lessons_for_agent(agent_name="regime_agent", regime="trending_down")
        lesson = db.get_relevant_lessons(regime="trending_down")[0]
        assert lesson.times_injected == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_injector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MemoryInjector**

Create `src/memory/injector.py`:

```python
"""Memory injector that retrieves and formats lessons for agent context.

Selects the most relevant lessons from the MemoryDB based on current
regime and strategy, formats them as human-readable strings, and
tracks injection counts.
"""

from __future__ import annotations

from loguru import logger

from src.memory.memory_db import MemoryDB


class MemoryInjector:
    """Retrieves and formats lessons for injection into agent prompts.

    Args:
        memory_db: The memory lesson database.
        max_lessons: Maximum lessons to inject per agent call.
    """

    def __init__(self, memory_db: MemoryDB, max_lessons: int = 5) -> None:
        self._db = memory_db
        self._max_lessons = max_lessons

    def get_lessons_for_agent(
        self,
        agent_name: str,
        regime: str = "",
        strategy: str = "",
    ) -> list[str]:
        """Get formatted lesson strings for an agent.

        Retrieves lessons relevant to the current context,
        formats them as human-readable strings, and records
        the injection.

        Args:
            agent_name: Name of the agent requesting lessons.
            regime: Current market regime.
            strategy: Current strategy name.

        Returns:
            List of formatted lesson strings.
        """
        lessons = self._db.get_relevant_lessons(
            regime=regime,
            strategy=strategy,
            max_lessons=self._max_lessons,
        )

        if not lessons:
            return []

        formatted = []
        for lesson in lessons:
            text = f"[{lesson.category}] {lesson.description}"
            formatted.append(text)
            self._db.record_injection(lesson.lesson_id)

        logger.debug(
            "Injected {} lessons into {} | regime={} | strategy={}",
            len(formatted),
            agent_name,
            regime,
            strategy,
        )

        return formatted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_injector.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/injector.py tests/test_injector.py
git commit -m "feat: add memory injector for lesson context injection"
```

---

## Task 12: LangGraph Agent Pipeline

**Files:**
- Create: `src/agents/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
"""Tests for the LangGraph agent pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.agents.pipeline import AgentPipeline
from src.agents.llm_client import LLMClient, LLMResponse
from src.agents.models import PipelineState
from src.memory.memory_db import MemoryDB


class TestAgentPipeline:
    def _make_signals(self):
        return [{
            "signal_id": "sig_001",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "entry_price": 50000_00,
            "stop_loss": 49000_00,
            "target": 53000_00,
            "strategy": "ema_crossover",
            "confidence": 0.8,
        }]

    def test_pipeline_with_all_fallbacks(self):
        """When LLM is not configured, all agents use fallbacks."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 30.0, "adx_trend": "rising", "rsi": 55.0,
                "atr_percentile": 50.0, "bb_width": 3.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 70.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        assert isinstance(result, PipelineState)
        assert result.regime is not None
        assert result.regime.regime in {"trending_up", "trending_down", "ranging", "volatile"}

    def test_pipeline_short_circuits_on_low_confidence(self):
        """Pipeline should short-circuit when regime confidence is low."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.8,  # Set high threshold
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 15.0, "adx_trend": "flat", "rsi": 50.0,
                "atr_percentile": 50.0, "bb_width": 2.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 50.0,
                "ema_9": 50000.0, "ema_21": 50000.0, "sma_50": 50000.0,
                "is_squeeze": False,
            },
        )
        assert result.short_circuited
        assert result.validated_signals == []

    def test_pipeline_passes_signals_through(self):
        """Signals aligned with regime should pass validation."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 35.0, "adx_trend": "rising", "rsi": 60.0,
                "atr_percentile": 55.0, "bb_width": 4.0,
                "bb_width_percentile": 60.0, "price_position_in_range": 75.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        assert not result.short_circuited
        # BUY + trending_up + RR > 1.5 should be approved
        assert len(result.validated_signals) >= 1

    def test_pipeline_rejects_counter_trend(self):
        """BUY signal in trending_down should be rejected."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 35.0, "adx_trend": "rising", "rsi": 30.0,
                "atr_percentile": 55.0, "bb_width": 4.0,
                "bb_width_percentile": 60.0, "price_position_in_range": 20.0,
                "ema_9": 49500.0, "ema_21": 49800.0, "sma_50": 50200.0,
                "is_squeeze": False,
            },
        )
        # BUY + trending_down -> rejected
        assert len(result.validated_signals) == 0

    def test_agent_decisions_recorded(self):
        """Pipeline should record all agent decisions."""
        client = LLMClient(api_key="", model="test")
        memory_db = MemoryDB()
        pipeline = AgentPipeline(
            llm_client=client,
            memory_db=memory_db,
            regime_confidence_threshold=0.3,
        )
        result = pipeline.run(
            instrument_key="NSE_EQ:RELIANCE",
            signals=self._make_signals(),
            indicator_context={
                "adx": 30.0, "adx_trend": "rising", "rsi": 55.0,
                "atr_percentile": 50.0, "bb_width": 3.0,
                "bb_width_percentile": 50.0, "price_position_in_range": 70.0,
                "ema_9": 50500.0, "ema_21": 50200.0, "sma_50": 49800.0,
                "is_squeeze": False,
            },
        )
        # Should have at least regime_agent and signal_validator decisions
        agent_names = [d.agent_name for d in result.agent_decisions]
        assert "regime_agent" in agent_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement AgentPipeline**

Create `src/agents/pipeline.py`:

```python
"""LangGraph-based agent pipeline for intelligent trade filtering.

Orchestrates the flow: Regime Detection -> Signal Validation.
Each node reads from and writes to a shared PipelineState.
Short-circuits when regime confidence is below threshold.

Note: This implementation uses a simple sequential pipeline.
LangGraph StateGraph can be added later for more complex
conditional routing without changing the interface.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agents.llm_client import LLMClient
from src.agents.models import AgentDecision, PipelineState, RegimeClassification
from src.agents.regime_agent import RegimeAgent
from src.agents.signal_validator import SignalValidatorAgent
from src.memory.injector import MemoryInjector
from src.memory.memory_db import MemoryDB


class AgentPipeline:
    """Orchestrates AI agents in a sequential pipeline.

    Flow:
    1. Regime Agent classifies market state
    2. If confidence < threshold, short-circuit (skip trading)
    3. Signal Validator approves/rejects/modifies each signal
    4. Return validated signals for execution

    Args:
        llm_client: Shared LLM client.
        memory_db: Memory lesson database.
        regime_confidence_threshold: Minimum regime confidence to proceed.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_db: MemoryDB,
        regime_confidence_threshold: float = 0.3,
    ) -> None:
        self._regime_agent = RegimeAgent(llm_client=llm_client)
        self._signal_validator = SignalValidatorAgent(llm_client=llm_client)
        self._injector = MemoryInjector(memory_db=memory_db, max_lessons=5)
        self._confidence_threshold = regime_confidence_threshold

    def run(
        self,
        instrument_key: str,
        signals: list[dict[str, Any]],
        indicator_context: dict[str, Any],
        sentiment_score: float = 50.0,
        sentiment_classification: str = "neutral",
    ) -> PipelineState:
        """Execute the full agent pipeline.

        Args:
            instrument_key: Instrument being evaluated.
            signals: Raw signals from strategy engine.
            indicator_context: Technical indicator values for regime agent.
            sentiment_score: Market mood score (0-100).
            sentiment_classification: Mood classification string.

        Returns:
            PipelineState with regime, validated signals, and decisions.
        """
        state = PipelineState(
            instrument_key=instrument_key,
            signals=signals,
        )

        if not signals:
            logger.debug("Pipeline: no signals to process for {}", instrument_key)
            return state

        # --- Step 1: Regime Detection ---
        regime_lessons = self._injector.get_lessons_for_agent(
            agent_name="regime_agent",
            regime="",  # No regime known yet
        )
        regime_result = self._regime_agent.run(
            context=indicator_context,
            lessons=regime_lessons,
            instrument_key=instrument_key,
        )
        state.regime = self._regime_agent.to_classification(regime_result)

        if self._regime_agent.last_decision:
            state.agent_decisions.append(self._regime_agent.last_decision)

        logger.info(
            "Pipeline regime: {} (confidence={:.2f}) for {}",
            state.regime.regime,
            state.regime.confidence,
            instrument_key,
        )

        # --- Short-circuit on low confidence ---
        if state.regime.is_low_confidence(self._confidence_threshold):
            state.short_circuited = True
            state.short_circuit_reason = (
                f"Regime confidence {state.regime.confidence:.2f} "
                f"< threshold {self._confidence_threshold}"
            )
            logger.warning(
                "Pipeline short-circuited: {} | {}",
                instrument_key,
                state.short_circuit_reason,
            )
            return state

        # --- Step 2: Signal Validation ---
        validated = []
        for signal in signals:
            validator_lessons = self._injector.get_lessons_for_agent(
                agent_name="signal_validator",
                regime=state.regime.regime,
                strategy=signal.get("strategy", ""),
            )
            validation_context = {
                "signal": signal,
                "regime": state.regime.regime,
                "sentiment_score": sentiment_score,
                "sentiment_classification": sentiment_classification,
            }
            val_result = self._signal_validator.run(
                context=validation_context,
                lessons=validator_lessons,
                instrument_key=instrument_key,
            )

            if self._signal_validator.last_decision:
                state.agent_decisions.append(self._signal_validator.last_decision)

            decision = self._signal_validator.to_decision(
                val_result, signal_id=signal.get("signal_id", "")
            )

            if decision.is_approved:
                # Apply modifications if any
                modified_signal = dict(signal)
                if decision.adjusted_stop_loss is not None:
                    modified_signal["stop_loss"] = decision.adjusted_stop_loss
                if decision.adjusted_target is not None:
                    modified_signal["target"] = decision.adjusted_target
                if decision.adjusted_quantity_pct != 1.0:
                    orig_qty = modified_signal.get("quantity", 1)
                    modified_signal["quantity"] = max(
                        1, int(orig_qty * decision.adjusted_quantity_pct)
                    )
                modified_signal["validation_confidence"] = decision.confidence
                modified_signal["validation_reasoning"] = decision.reasoning
                validated.append(modified_signal)
                logger.info(
                    "Signal {} APPROVED | action={} | confidence={:.2f}",
                    signal.get("signal_id", "?"),
                    decision.action,
                    decision.confidence,
                )
            else:
                logger.info(
                    "Signal {} REJECTED | reason={}",
                    signal.get("signal_id", "?"),
                    decision.reasoning,
                )

        state.validated_signals = validated
        logger.info(
            "Pipeline complete for {} | {}/{} signals validated",
            instrument_key,
            len(validated),
            len(signals),
        )

        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_pipeline.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/pipeline.py tests/test_pipeline.py
git commit -m "feat: add agent pipeline with regime detection and signal validation"
```

---

## Task 13: Database Models for Agent Decisions

**Files:**
- Modify: `src/persistence/models.py`
- Test: Verify migration works

- [ ] **Step 1: Add new tables to persistence models**

Add the following classes to the end of `src/persistence/models.py` (before the closing of the file):

```python
class AgentDecisionRecord(Base):
    """Records every AI agent decision for observability."""

    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(64), nullable=False, index=True)
    instrument_key = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=0.5)
    used_fallback = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Integer, default=0, nullable=False)
    lessons_injected = Column(Integer, default=0, nullable=False)


class MemoryLessonRecord(Base):
    """Persistent storage for memory lessons with decay tracking."""

    __tablename__ = "memory_lessons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lesson_id = Column(String(64), unique=True, nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    strategy = Column(String(128), nullable=False, index=True)
    regime = Column(String(32), nullable=False, index=True)
    description = Column(Text, nullable=False)
    severity = Column(String(16), nullable=False)
    base_score = Column(Float, nullable=False, default=1.0)
    times_injected = Column(Integer, default=0, nullable=False)
    times_useful = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class MistakeRecord(Base):
    """Records classified mistakes for analysis."""

    __tablename__ = "trade_mistakes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    severity = Column(String(16), nullable=False)
    description = Column(Text, nullable=False)
    lesson = Column(Text, nullable=False)
    regime = Column(String(32), nullable=False)
    strategy = Column(String(128), nullable=False)
    score = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
```

- [ ] **Step 2: Verify the models load correctly**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -c "from src.persistence.models import AgentDecisionRecord, MemoryLessonRecord, MistakeRecord; print('Models OK')"`
Expected: `Models OK`

- [ ] **Step 3: Commit**

```bash
git add src/persistence/models.py
git commit -m "feat: add database models for agent decisions, memory lessons, and mistakes"
```

---

## Task 14: Integration - Wire Pipeline into Main

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add pipeline initialization to TradingBot.startup()**

In `src/main.py`, add the following imports at the top (after existing imports):

```python
from src.agents.llm_client import LLMClient
from src.agents.pipeline import AgentPipeline
from src.memory.memory_db import MemoryDB
from src.memory.outcome_analyzer import OutcomeAnalyzer, TradeOutcome
from src.memory.mistake_classifier import MistakeClassifier
```

Add these attributes to `TradingBot.__init__()`:

```python
        self.llm_client: LLMClient | None = None
        self.agent_pipeline: AgentPipeline | None = None
        self.memory_db: MemoryDB | None = None
        self.outcome_analyzer: OutcomeAnalyzer | None = None
        self.mistake_classifier: MistakeClassifier | None = None
```

Add pipeline initialization to the end of `startup()`, before the "Startup Complete" log:

```python
        # 14. Initialize AI agent pipeline
        if self.settings.agent_pipeline_enabled:
            self.llm_client = LLMClient(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                fallback_model=self.settings.groq_fallback_model,
                temperature=self.settings.groq_temperature,
                max_tokens=self.settings.groq_max_tokens,
                rate_limit_rpm=self.settings.groq_rate_limit_rpm,
            )
            self.memory_db = MemoryDB(
                decay_rate=self.settings.memory_decay_rate,
                decay_start_days=self.settings.memory_decay_start_days,
            )
            self.agent_pipeline = AgentPipeline(
                llm_client=self.llm_client,
                memory_db=self.memory_db,
                regime_confidence_threshold=self.settings.regime_confidence_threshold,
            )
            self.outcome_analyzer = OutcomeAnalyzer()
            self.mistake_classifier = MistakeClassifier()
            logger.info(
                "AI Agent Pipeline initialized | LLM configured={}",
                self.llm_client.is_configured,
            )
```

- [ ] **Step 2: Verify imports work**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -c "from src.agents.pipeline import AgentPipeline; print('Pipeline import OK')"`
Expected: `Pipeline import OK`

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: wire AI agent pipeline into trading bot startup"
```

---

## Task 15: Run Full Test Suite

**Files:**
- No new files

- [ ] **Step 1: Run all new tests together**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/test_agent_models.py tests/test_llm_client.py tests/test_base_agent.py tests/test_regime_agent.py tests/test_signal_validator.py tests/test_news_agent.py tests/test_sentiment.py tests/test_outcome_analyzer.py tests/test_mistake_classifier.py tests/test_memory_db.py tests/test_injector.py tests/test_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run the full test suite to check for regressions**

Run: `cd /Users/sandeepvangapandu/Downloads/Trading && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still PASS, no regressions

- [ ] **Step 3: Final commit with all `__init__.py` updates**

Update `src/agents/__init__.py` with exports:

```python
"""AI agent pipeline for intelligent trade filtering."""

from src.agents.llm_client import LLMClient
from src.agents.models import PipelineState, RegimeClassification, SignalDecision, SentimentScore
from src.agents.pipeline import AgentPipeline
from src.agents.regime_agent import RegimeAgent
from src.agents.signal_validator import SignalValidatorAgent
from src.agents.news_agent import NewsAgent
from src.agents.sentiment import MarketMoodCalculator

__all__ = [
    "LLMClient",
    "PipelineState",
    "RegimeClassification",
    "SignalDecision",
    "SentimentScore",
    "AgentPipeline",
    "RegimeAgent",
    "SignalValidatorAgent",
    "NewsAgent",
    "MarketMoodCalculator",
]
```

```bash
git add src/agents/__init__.py
git commit -m "feat: finalize agent package exports"
```
