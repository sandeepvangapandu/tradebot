"""LLM client with circuit breaker, rate limiting, and fallback.

Supports two providers:
- **Groq**: Fast inference via Groq SDK (groq package).
- **OpenRouter**: OpenAI-compatible API gateway with many free models.

The trading bot never depends on LLM availability for continued operation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.utils.rate_limiter import RateLimiter


@dataclass
class LLMResponse:
    """Response from an LLM invocation."""

    content: str
    success: bool
    error: str | None = None
    latency_ms: int = 0
    used_fallback_model: bool = False


class LLMCircuitBreaker:
    """Circuit breaker for LLM API calls.

    States: closed (normal) -> open (failing) -> half_open (testing).

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout_s: Seconds to wait before transitioning to half_open.
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
        self._state = "closed"
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        """Current circuit breaker state: closed, open, or half_open."""
        with self._lock:
            return self._state

    def is_available(self) -> bool:
        """Check whether calls should be allowed through.

        Returns:
            True if the circuit is closed or half_open (probe allowed).
            False if the circuit is open and the recovery timeout has not elapsed.
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
            # half_open: allow one probe call through
            return True

    def record_success(self) -> None:
        """Record a successful API call; resets failure count and closes circuit."""
        with self._lock:
            self._failure_count = 0
            if self._state == "half_open":
                self._state = "closed"
                logger.info("LLM circuit breaker: half_open -> closed")

    def record_failure(self) -> None:
        """Record a failed API call; may open the circuit."""
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

    Supports Groq and OpenRouter providers with circuit breaker,
    rate limiting, and automatic fallback to a smaller model on failure.

    Args:
        api_key: API key for the selected provider.
        model: Primary model identifier.
        fallback_model: Model to use when the primary model fails.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in the completion.
        rate_limit_rpm: Requests per minute budget.
        provider: LLM provider — "groq" or "openrouter".
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-70b-8192",
        fallback_model: str = "llama3-8b-8192",
        temperature: float = 0.0,
        max_tokens: int = 512,
        rate_limit_rpm: int = 60,
        provider: str = "groq",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider = provider.lower()

        self._circuit_breaker = LLMCircuitBreaker(
            failure_threshold=3, recovery_timeout_s=30.0
        )
        self._rate_limiter = RateLimiter(
            rate=rate_limit_rpm / 60.0,
            capacity=min(rate_limit_rpm / 60.0 * 5, 10.0),
        )

        self._client: Any = None
        if api_key:
            self._client = self._init_client()

    def _init_client(self) -> Any:
        """Initialize the appropriate SDK client based on provider."""
        if self._provider == "groq":
            try:
                from groq import Groq

                client = Groq(api_key=self._api_key)
                logger.info("LLM client initialized: provider=groq, model={}", self._model)
                return client
            except ImportError:
                logger.warning("groq package not installed, LLM client disabled")
                return None

        elif self._provider == "openrouter":
            try:
                from openai import OpenAI

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self._api_key,
                )
                logger.info(
                    "LLM client initialized: provider=openrouter, model={}",
                    self._model,
                )
                return client
            except ImportError:
                logger.warning(
                    "openai package not installed. Install with: pip install openai"
                )
                return None

        else:
            logger.error("Unknown LLM provider: {}", self._provider)
            return None

    @property
    def is_configured(self) -> bool:
        """True when an API key is set and the client was initialised."""
        return bool(self._api_key) and self._client is not None

    @property
    def provider(self) -> str:
        """Active LLM provider name."""
        return self._provider

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        use_fallback_model: bool = False,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a structured response.

        Applies circuit-breaker and rate-limiter guards before every call.
        On primary-model failure, automatically retries with the fallback model.

        Args:
            prompt: User message / prompt text.
            system_prompt: Optional system instruction prepended to the chat.
            use_fallback_model: If True, skip directly to the fallback model.

        Returns:
            LLMResponse with content and success flag.
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
        self._rate_limiter.acquire()

        start = time.monotonic()
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Both Groq SDK and OpenAI SDK share the same chat.completions.create API
            extra_kwargs: dict[str, Any] = {}
            if self._provider == "openrouter":
                extra_kwargs["extra_headers"] = {
                    "HTTP-Referer": "https://github.com/trading-bot",
                    "X-Title": "Trading Agent Pipeline",
                }

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **extra_kwargs,
            )

            content = response.choices[0].message.content or ""
            latency_ms = int((time.monotonic() - start) * 1000)
            self._circuit_breaker.record_success()

            return LLMResponse(
                content=content,
                success=True,
                latency_ms=latency_ms,
                used_fallback_model=use_fallback_model,
            )

        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._circuit_breaker.record_failure()
            logger.warning(
                "LLM call failed | provider={} | model={} | error={}",
                self._provider,
                model,
                e,
            )

            # Retry once with the smaller fallback model (avoids infinite recursion)
            if not use_fallback_model and model != self._fallback_model:
                logger.info("Retrying with fallback model: {}", self._fallback_model)
                return self.invoke(prompt, system_prompt, use_fallback_model=True)

            return LLMResponse(
                content="",
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )
