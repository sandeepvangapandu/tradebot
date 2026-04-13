"""Token-bucket rate limiter for API call throttling."""

from __future__ import annotations

import threading
import time as _time
from types import TracebackType


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Args:
        rate: Token refill rate in tokens per second.
        capacity: Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float = 40.0, capacity: float = 40.0) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill: float = _time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill.

        Must be called while holding ``self._lock``.
        """
        now = _time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until the requested number of tokens is available.

        Args:
            tokens: Number of tokens to consume.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate how long to wait for enough tokens.
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate
            _time.sleep(wait_time)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens without blocking.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were successfully acquired, False otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    # -- Context-manager support ------------------------------------------

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Nothing to release; tokens are consumed on entry.
        pass
