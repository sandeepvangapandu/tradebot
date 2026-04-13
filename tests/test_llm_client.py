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
