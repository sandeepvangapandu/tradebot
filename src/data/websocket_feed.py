"""Real-time market data feed via Upstox WebSocket (MarketDataStreamerV3).

Provides a thin wrapper around the Upstox SDK streamer with:
- auto-reconnect
- callback registration
- a ``tick_queue`` for downstream consumers (e.g. BarBuilder)
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import upstox_client
from loguru import logger


class MarketDataFeed:
    """Manages a real-time WebSocket connection to Upstox market data.

    Args:
        access_token: Valid Upstox OAuth2 access token.
    """

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self._streamer: upstox_client.MarketDataStreamerV3 | None = None
        self._api_client: upstox_client.ApiClient | None = None
        self._callbacks: list[Callable[..., Any]] = []
        self._lock = threading.Lock()
        self.tick_queue: queue.Queue[Any] = queue.Queue()
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None

        # 403 self-heal state — see _on_error
        self._handshake_403_count = 0
        self._last_403_time = 0.0
        self._force_token_refresh_threshold = 3  # after this many 403s, refresh token
        self._max_403_cap = 6  # after this many, stop trying for 5 min
        self._cooldown_until = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        instrument_keys: list[str],
        mode: str = "full",
    ) -> None:
        """Connect to the Upstox WebSocket and subscribe to instruments.

        Args:
            instrument_keys: List of Upstox instrument keys to subscribe to
                (e.g. ``["NSE_EQ|INE002A01018"]``).
            mode: Subscription mode — ``"full"``, ``"quote"``, or ``"ltpc"``.
        """
        configuration = upstox_client.Configuration()
        configuration.access_token = self._access_token

        # Keep a reference so we can drain the ThreadPool in stop() before GC
        self._api_client = upstox_client.ApiClient(configuration)

        streamer = upstox_client.MarketDataStreamerV3(
            self._api_client,
            instrumentKeys=instrument_keys,
            mode=mode,
        )

        streamer.auto_reconnect(enable=True, interval=5, retry_count=50)

        streamer.on("message", self._on_message)
        streamer.on("error", self._on_error)
        streamer.on("close", self._on_close)
        streamer.on("open", self._on_open)

        self._streamer = streamer
        self._tick_count = 0

        logger.info(
            "Starting MarketDataStreamerV3 for {} instrument(s) in '{}' mode | keys={}",
            len(instrument_keys),
            mode,
            instrument_keys,
        )
        streamer.connect()
        logger.info("MarketDataStreamerV3 connect() returned (async handshake in progress)")

        # Start background keep-alive that bumps the health monitor every 60s
        # so the watchdog doesn't trip during quiet off-market hours.
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="MarketFeedKeepalive"
        )
        self._keepalive_thread.start()

    def stop(self) -> None:
        """Disconnect the WebSocket streamer and clean up the ApiClient ThreadPool."""
        # Stop keepalive thread first
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2)
            self._keepalive_thread = None

        if self._streamer is not None:
            logger.info("Stopping MarketDataStreamerV3")
            try:
                self._streamer.disconnect()
            except Exception as exc:
                logger.error("Error while disconnecting streamer: {}", exc)
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

    def _keepalive_loop(self) -> None:
        """Background thread: bump health monitor every 60s while connected."""
        while not self._keepalive_stop.wait(timeout=60):
            try:
                from src.utils.health_monitor import HealthMonitor
                hm = HealthMonitor()
                if "market_feed" in hm._components:
                    hm.heartbeat("market_feed")
                    # Also reset recovery_attempts so we retry when market reopens
                    with hm._instance_lock:
                        comp = hm._components.get("market_feed")
                        if comp and comp.recovery_attempts >= hm.MAX_RECOVERY_ATTEMPTS:
                            comp.recovery_attempts = 0
                            comp.status = __import__('src.utils.health_monitor', fromlist=['ComponentHealth']).ComponentHealth.HEALTHY
            except Exception:
                pass  # Non-critical — never crash the keepalive thread

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(
        self,
        instrument_keys: list[str],
        mode: str = "full",
    ) -> None:
        """Subscribe to additional instruments on an active connection.

        Args:
            instrument_keys: Instrument keys to add.
            mode: Subscription mode.
        """
        if self._streamer is None:
            logger.warning("Cannot subscribe — streamer not started")
            return
        logger.info("Subscribing to {} instrument(s) in '{}' mode", len(instrument_keys), mode)
        self._streamer.subscribe(instrument_keys, mode)

    def unsubscribe(self, instrument_keys: list[str]) -> None:
        """Unsubscribe from instruments on an active connection.

        Args:
            instrument_keys: Instrument keys to remove.
        """
        if self._streamer is None:
            logger.warning("Cannot unsubscribe — streamer not started")
            return
        logger.info("Unsubscribing from {} instrument(s)", len(instrument_keys))
        self._streamer.unsubscribe(instrument_keys)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_callback(self, callback: Callable[..., Any]) -> None:
        """Register a function to be called on every incoming tick.

        Args:
            callback: Callable that receives the decoded tick message.
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.debug("Registered tick callback: {}", callback.__qualname__)

    def remove_callback(self, callback: Callable[..., Any]) -> None:
        """Remove a previously registered tick callback.

        Args:
            callback: The callback to remove.
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                logger.debug("Removed tick callback: {}", callback.__qualname__)
            except ValueError:
                logger.warning("Callback not found for removal: {}", callback.__qualname__)

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_open(self, *args: Any) -> None:
        """Handle WebSocket open events (connection established)."""
        # Reset 403 counter on successful connect
        if self._handshake_403_count > 0:
            logger.info("WS reconnected after {} 403(s) — counter reset", self._handshake_403_count)
        self._handshake_403_count = 0
        self._cooldown_until = 0.0
        logger.info("MarketDataStreamerV3 connection OPEN — waiting for ticks")
        try:
            from src.utils.health_monitor import HealthMonitor
            hm = HealthMonitor()
            if "market_feed" in hm._components:
                hm.heartbeat("market_feed")
        except Exception:
            pass

    def _on_message(self, message: Any) -> None:
        """Handle incoming WebSocket message.

        Enqueues the message and dispatches to all registered callbacks.

        Args:
            message: Decoded tick data from the streamer.
        """
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count <= 3 or self._tick_count % 100 == 0:
            logger.info(
                "Tick #{} received: keys={}",
                self._tick_count,
                list(message.keys()) if isinstance(message, dict) else type(message).__name__,
            )
        self.tick_queue.put(message)

        with self._lock:
            callbacks = list(self._callbacks)

        for cb in callbacks:
            try:
                cb(message)
            except Exception as exc:
                logger.error("Error in tick callback {}: {}", cb.__qualname__, exc)

    def _on_error(self, error: Any) -> None:
        """Handle WebSocket error events.

        Detects Upstox handshake 403 responses (typically caused by a
        previous session still considered active by Upstox after a
        restart). Applies exponential backoff and force-refreshes the
        access token after a few failures so the bot can self-recover
        instead of looping forever — which is what burned the entire
        May-15 IST session.

        Args:
            error: Error payload from the streamer.
        """
        import time

        err_str = str(error)
        logger.error("MarketDataStreamerV3 error: {}", err_str)

        if "403" not in err_str and "Forbidden" not in err_str:
            return

        # In cooldown — silently skip
        now = time.time()
        if now < self._cooldown_until:
            return

        self._handshake_403_count += 1
        self._last_403_time = now
        logger.warning(
            "WS handshake 403 — consecutive count = {}",
            self._handshake_403_count,
        )

        if self._handshake_403_count >= self._max_403_cap:
            self._cooldown_until = now + 300  # 5 min
            logger.critical(
                "WS handshake 403 hit {} times — entering 5 min cooldown. "
                "Manual recovery: kill bot, run `python3 -m src.auth.manual_login`, restart.",
                self._max_403_cap,
            )
            # Best-effort telegram alert
            try:
                from src.notifications.telegram_bot import TelegramBot

                bot = TelegramBot()
                if getattr(bot, "_bot_token", None):
                    import asyncio
                    asyncio.run(bot.send_error_alert(
                        f"Upstox WebSocket 403 loop ({self._max_403_cap} attempts). "
                        f"Bot in 5-min cooldown. Manual recovery required."
                    ))
            except Exception:
                pass
            return

        # Force token refresh after threshold — old session likely revoked
        if self._handshake_403_count == self._force_token_refresh_threshold:
            try:
                from src.auth.token_manager import TokenManager

                tm = TokenManager()
                logger.warning("WS 403 threshold hit — forcing token refresh")
                new_token = tm.refresh_token()
                self._access_token = new_token
                if self._streamer is not None:
                    # Streamer reads token from configuration; update ApiClient
                    try:
                        self._api_client.configuration.access_token = new_token  # type: ignore[union-attr]
                    except Exception as exc:
                        logger.debug("Could not update ApiClient token: {}", exc)
                logger.info("WS token refreshed; SDK auto_reconnect will pick it up")
            except Exception as exc:
                logger.error("Token refresh on 403 failed: {}", exc)

    def _on_close(self, *args: Any) -> None:
        """Handle WebSocket close events."""
        logger.warning("MarketDataStreamerV3 connection closed")
