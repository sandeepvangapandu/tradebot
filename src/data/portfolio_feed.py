"""Portfolio data feed via Upstox WebSocket (PortfolioDataStreamer).

Provides real-time order and position updates from the broker.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import upstox_client
from loguru import logger


class PortfolioFeed:
    """Manages WebSocket connection for portfolio updates (orders & positions).

    Args:
        access_token: Valid Upstox OAuth2 access token.
    """

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self._streamer: upstox_client.PortfolioDataStreamer | None = None
        self._api_client: upstox_client.ApiClient | None = None
        self._order_callbacks: list[Callable[[dict], None]] = []
        self._position_callbacks: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Connect to the Upstox portfolio WebSocket."""
        if self._running:
            return

        configuration = upstox_client.Configuration()
        configuration.access_token = self._access_token

        # Keep a reference so we can drain the ThreadPool in stop() before GC
        self._api_client = upstox_client.ApiClient(configuration)

        streamer = upstox_client.PortfolioDataStreamer(
            self._api_client,
        )

        streamer.auto_reconnect(enable=True, interval=5, retry_count=50)

        streamer.on("message", self._on_message)
        streamer.on("error", self._on_error)
        streamer.on("close", self._on_close)

        self._streamer = streamer
        self._running = True

        logger.info("Starting PortfolioDataStreamer for order/position updates")
        streamer.connect()

    def stop(self) -> None:
        """Disconnect the portfolio WebSocket and clean up the ApiClient ThreadPool."""
        self._running = False
        if self._streamer is not None:
            logger.info("Stopping PortfolioDataStreamer")
            try:
                self._streamer.disconnect()
            except Exception as exc:
                logger.error("Error while disconnecting portfolio streamer: {}", exc)
            finally:
                self._streamer = None

        # Explicitly drain the ApiClient's ThreadPool before GC runs __del__.
        # Without this, pool.join() raises [Errno 9] at interpreter shutdown
        # and the SDK prints it raw to stdout.
        if self._api_client is not None:
            try:
                self._api_client.pool.close()
                self._api_client.pool.join()
            except Exception:
                pass  # Pool may already be stopped at shutdown — safe to ignore
            finally:
                self._api_client = None

    def register_order_callback(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for order updates.

        Args:
            callback: Function receiving order update dict.
        """
        with self._lock:
            if callback not in self._order_callbacks:
                self._order_callbacks.append(callback)
                logger.debug("Registered order callback: {}", callback.__qualname__)

    def remove_order_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove an order callback."""
        with self._lock:
            try:
                self._order_callbacks.remove(callback)
            except ValueError:
                pass

    def register_position_callback(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for position updates.

        Args:
            callback: Function receiving position update dict.
        """
        with self._lock:
            if callback not in self._position_callbacks:
                self._position_callbacks.append(callback)
                logger.debug("Registered position callback: {}", callback.__qualname__)

    def remove_position_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove a position callback."""
        with self._lock:
            try:
                self._position_callbacks.remove(callback)
            except ValueError:
                pass

    def _on_message(self, message: Any) -> None:
        """Dispatch a portfolio WebSocket message to order/position callbacks.

        The Upstox PortfolioDataStreamer emits order updates and position
        updates as a single `message` event. We inspect the payload and
        route to the appropriate callback list.
        """
        try:
            # Message is typically a dict with a `type` field, or a JSON string
            payload = message
            if isinstance(payload, (bytes, str)):
                import json
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"raw": payload}

            kind = ""
            if isinstance(payload, dict):
                kind = (payload.get("type") or payload.get("update_type") or "").lower()

            if "position" in kind:
                self._on_position_update(payload)
            else:
                # Default to order update for unknown kinds
                self._on_order_update(payload)
        except Exception as exc:
            logger.error("PortfolioDataStreamer message dispatch failed: {}", exc)

    def _on_order_update(self, message: dict[str, Any]) -> None:
        """Handle order update from WebSocket."""
        logger.debug("Order update: {}", message)

        with self._lock:
            callbacks = list(self._order_callbacks)

        for cb in callbacks:
            try:
                cb(message)
            except Exception as exc:
                logger.error("Error in order callback {}: {}", cb.__qualname__, exc)

    def _on_position_update(self, message: dict[str, Any]) -> None:
        """Handle position update from WebSocket."""
        logger.debug("Position update: {}", message)

        with self._lock:
            callbacks = list(self._position_callbacks)

        for cb in callbacks:
            try:
                cb(message)
            except Exception as exc:
                logger.error("Error in position callback {}: {}", cb.__qualname__, exc)

    def _on_error(self, error: Any) -> None:
        """Handle WebSocket error."""
        logger.error("PortfolioDataStreamer error: {}", error)

    def _on_close(self, *args: Any) -> None:
        """Handle WebSocket close."""
        logger.warning("PortfolioDataStreamer connection closed")
        self._running = False

    def is_running(self) -> bool:
        """Check if feed is connected."""
        return self._running
