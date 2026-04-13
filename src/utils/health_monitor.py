"""Self-healing health monitor for the trading bot.

This module provides comprehensive health monitoring and automatic recovery
for all trading bot components. It tracks component status, detects failures,
and attempts automatic recovery without human intervention.

Example Usage:
    from src.utils.health_monitor import HealthMonitor, ComponentHealth

    monitor = HealthMonitor()
    monitor.register_component("websocket_feed", heartbeat_timeout=30)

    # In WebSocket thread:
    monitor.heartbeat("websocket_feed")

    # Before critical operation:
    if not monitor.is_healthy("websocket_feed"):
        logger.warning("WebSocket not healthy, skipping operation")
        return

    # Get status dashboard data:
    status = monitor.get_all_status()
"""

from __future__ import annotations

import functools
import gc
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from src.utils.logger import logger

T = TypeVar("T")


class ComponentHealth(str, Enum):
    """Health status enumeration for components.

    Attributes:
        HEALTHY: Component is functioning normally.
        DEGRADED: Component has issues but is still operational.
        FAILED: Component has failed and needs recovery.
        RECOVERING: Component is currently being recovered.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ComponentStatus:
    """Status information for a monitored component.

    Attributes:
        name: Unique identifier for the component.
        status: Current health status of the component.
        last_heartbeat: Timestamp of the last heartbeat received.
        error_count: Number of errors in the current window.
        last_error: Most recent error message (if any).
        recovery_attempts: Number of recovery attempts made.
        error_timestamps: List of recent error timestamps for rate calculation.
        metadata: Additional component-specific data.
    """

    name: str
    status: ComponentHealth = ComponentHealth.HEALTHY
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now())
    error_count: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    error_timestamps: list[datetime] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure error_timestamps is a list."""
        if self.error_timestamps is None:
            self.error_timestamps = []


# Recovery function type
RecoveryFunction = Callable[[str], bool]


class HealthMonitor:
    """Self-healing health monitor for trading bot components.

    This class implements a singleton pattern to provide a centralized
    health monitoring system. It tracks component status through heartbeats,
    detects failures, and automatically attempts recovery.

    Thread-safe operations ensure consistent state across multiple threads.

    Attributes:
        _instance: Singleton instance of HealthMonitor.
        _lock: Class-level lock for singleton creation.
    """

    _instance: Optional[HealthMonitor] = None
    _lock: threading.Lock = threading.Lock()

    # Alert thresholds
    DEGRADED_ERROR_THRESHOLD: int = 3  # Errors in 5 minutes to mark degraded
    DEGRADED_TIME_WINDOW: int = 300  # 5 minutes in seconds
    FAILED_HEARTBEAT_MULTIPLIER: float = 1.5  # Multiplier for timeout to mark failed
    MAX_RECOVERY_ATTEMPTS: int = 3  # Max recovery attempts before giving up
    DEGRADED_ALERT_THRESHOLD: int = 300  # 5 minutes in degraded before alerting

    def __new__(cls) -> HealthMonitor:
        """Create or return the singleton instance.

        Returns:
            The singleton HealthMonitor instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the health monitor (only runs once due to singleton)."""
        if self._initialized:
            return

        self._initialized = True
        self._components: dict[str, ComponentStatus] = {}
        self._heartbeat_timeouts: dict[str, int] = {}
        self._recovery_strategies: dict[str, RecoveryFunction] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_interval: int = 30
        self._stop_monitoring: threading.Event = threading.Event()
        self._instance_lock: threading.RLock = threading.RLock()
        self._alert_callbacks: list[Callable[[str, ComponentStatus], None]] = []

        # Register default recovery strategies
        self._register_default_recovery_strategies()

        logger.info("HealthMonitor initialized")

    def _register_default_recovery_strategies(self) -> None:
        """Register default recovery strategies for common components."""
        # WebSocket feed recovery
        self.register_recovery_strategy("websocket_feed", self._recover_websocket)
        self.register_recovery_strategy("market_feed", self._recover_websocket)

        # Database connection recovery
        self.register_recovery_strategy("database", self._recover_database)
        self.register_recovery_strategy("db_connection", self._recover_database)

        # Memory recovery
        self.register_recovery_strategy("memory", self._recover_memory)
        self.register_recovery_strategy("cache", self._recover_memory)

        # Order manager recovery
        self.register_recovery_strategy("order_manager", self._recover_order_manager)
        self.register_recovery_strategy("order_service", self._recover_order_manager)

    def _recover_websocket(self, component_name: str) -> bool:
        """Recovery strategy for WebSocket connections.

        Attempts to reconnect and resubscribe to symbols.

        Args:
            component_name: Name of the WebSocket component.

        Returns:
            True if recovery was successful, False otherwise.
        """
        logger.info(f"Attempting WebSocket recovery for {component_name}")

        try:
            # Import here to avoid circular dependencies
            from src.data.websocket_feed import MarketFeed

            # Get the market feed instance if it exists
            feed = MarketFeed.get_instance()
            if feed is not None:
                # Stop existing connection
                feed.stop()
                time.sleep(1)  # Brief pause before reconnect

                # Restart connection
                feed.start()

                # Resubscribe to symbols
                subscribed_symbols = getattr(feed, "_subscribed_symbols", set())
                for symbol in subscribed_symbols:
                    feed.subscribe(symbol)

                logger.info(f"WebSocket recovery successful for {component_name}")
                return True

        except ImportError:
            logger.warning("MarketFeed not available for WebSocket recovery")
        except Exception as e:
            logger.error(f"WebSocket recovery failed for {component_name}: {e}")

        return False

    def _recover_database(self, component_name: str) -> bool:
        """Recovery strategy for database connections.

        Closes and reopens the connection pool.

        Args:
            component_name: Name of the database component.

        Returns:
            True if recovery was successful, False otherwise.
        """
        logger.info(f"Attempting database recovery for {component_name}")

        try:
            # Import here to avoid circular dependencies
            from src.persistence.database import get_session as get_db_connection

            # Get database connection and reset it
            db = get_db_connection()
            if hasattr(db, "close"):
                db.close()
            if hasattr(db, "connect"):
                db.connect()

            logger.info(f"Database recovery successful for {component_name}")
            return True

        except ImportError:
            logger.warning("Database module not available for recovery")
        except Exception as e:
            logger.error(f"Database recovery failed for {component_name}: {e}")

        return False

    def _recover_memory(self, component_name: str) -> bool:
        """Recovery strategy for high memory usage.

        Clears caches and runs garbage collection.

        Args:
            component_name: Name of the memory component.

        Returns:
            True if recovery was successful, False otherwise.
        """
        logger.info(f"Attempting memory recovery for {component_name}")

        try:
            # Clear any caches
            import sys

            # Try to clear indicator cache if it exists
            try:
                from src.strategy.indicators import IndicatorEngine as IndicatorCache

                cache = IndicatorCache.get_instance()
                if hasattr(cache, "clear"):
                    cache.clear()
                    logger.info("Indicator cache cleared")
            except ImportError:
                pass

            # Force garbage collection
            gc.collect()

            # Log memory stats
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(
                f"Memory recovery completed. RSS: {memory_info.rss / 1024 / 1024:.2f} MB"
            )

            return True

        except ImportError:
            # psutil not available, just run gc
            gc.collect()
            logger.info("Garbage collection completed (psutil not available)")
            return True
        except Exception as e:
            logger.error(f"Memory recovery failed for {component_name}: {e}")
            return False

    def _recover_order_manager(self, component_name: str) -> bool:
        """Recovery strategy for order manager.

        Resets state and re-syncs with broker.

        Args:
            component_name: Name of the order manager component.

        Returns:
            True if recovery was successful, False otherwise.
        """
        logger.info(f"Attempting order manager recovery for {component_name}")

        try:
            # Import here to avoid circular dependencies
            from src.execution.order_manager import OrderManager

            # Get order manager instance
            om = OrderManager.get_instance()
            if om is not None:
                # Reset state
                if hasattr(om, "reset_state"):
                    om.reset_state()

                # Re-sync with broker
                if hasattr(om, "sync_with_broker"):
                    om.sync_with_broker()

                logger.info(f"Order manager recovery successful for {component_name}")
                return True

        except ImportError:
            logger.warning("OrderManager not available for recovery")
        except Exception as e:
            logger.error(f"Order manager recovery failed for {component_name}: {e}")

        return False

    def register_component(self, name: str, heartbeat_timeout: int = 60) -> None:
        """Register a component for health monitoring.

        Args:
            name: Unique identifier for the component.
            heartbeat_timeout: Seconds before component is considered failed
                if no heartbeat is received.

        Raises:
            ValueError: If component is already registered.
        """
        with self._instance_lock:
            if name in self._components:
                logger.warning(f"Component {name} is already registered")
                return

            self._components[name] = ComponentStatus(name=name)
            self._heartbeat_timeouts[name] = heartbeat_timeout

            logger.info(
                f"Registered component {name} with heartbeat timeout {heartbeat_timeout}s"
            )

    def unregister_component(self, name: str) -> None:
        """Unregister a component from health monitoring.

        Args:
            name: Unique identifier for the component.
        """
        with self._instance_lock:
            if name in self._components:
                del self._components[name]
                del self._heartbeat_timeouts[name]
                logger.info(f"Unregistered component {name}")

    def heartbeat(self, component_name: str) -> None:
        """Update the heartbeat timestamp for a component.

        This should be called regularly by the component to indicate
        it is still alive and functioning.

        Args:
            component_name: Name of the component sending the heartbeat.

        Raises:
            ValueError: If component is not registered.
        """
        with self._instance_lock:
            if component_name not in self._components:
                raise ValueError(f"Component {component_name} is not registered")

            status = self._components[component_name]
            old_status = status.status

            status.last_heartbeat = datetime.now()

            # If we were recovering and got a heartbeat, mark as healthy
            if old_status == ComponentHealth.RECOVERING:
                status.status = ComponentHealth.HEALTHY
                status.recovery_attempts = 0
                logger.info(f"Component {component_name} recovered and is now HEALTHY")

            # Clear error count on successful heartbeat
            if old_status == ComponentHealth.DEGRADED:
                # Only clear if enough time has passed since last error
                if status.error_timestamps:
                    last_error_time = status.error_timestamps[-1]
                    if datetime.now() - last_error_time > timedelta(
                        seconds=self.DEGRADED_TIME_WINDOW
                    ):
                        status.error_count = 0
                        status.error_timestamps = []
                        status.status = ComponentHealth.HEALTHY
                        logger.info(f"Component {component_name} is now HEALTHY")

    def update_status(
        self,
        component_name: str,
        status: ComponentHealth,
        error: Optional[str] = None,
    ) -> None:
        """Update the health status of a component.

        Args:
            component_name: Name of the component to update.
            status: New health status.
            error: Optional error message if status is DEGRADED or FAILED.

        Raises:
            ValueError: If component is not registered.
        """
        with self._instance_lock:
            if component_name not in self._components:
                raise ValueError(f"Component {component_name} is not registered")

            component_status = self._components[component_name]
            old_status = component_status.status

            component_status.status = status

            if error:
                component_status.last_error = error
                component_status.error_count += 1
                component_status.error_timestamps.append(datetime.now())

                # Clean old error timestamps (older than 5 minutes)
                cutoff = datetime.now() - timedelta(seconds=self.DEGRADED_TIME_WINDOW)
                component_status.error_timestamps = [
                    ts for ts in component_status.error_timestamps if ts > cutoff
                ]
                component_status.error_count = len(component_status.error_timestamps)

            # Auto-transition to DEGRADED if too many errors
            if (
                component_status.error_count >= self.DEGRADED_ERROR_THRESHOLD
                and status == ComponentHealth.HEALTHY
            ):
                component_status.status = ComponentHealth.DEGRADED
                status = ComponentHealth.DEGRADED
                logger.warning(
                    f"Component {component_name} auto-transitioned to DEGRADED "
                    f"due to {component_status.error_count} errors in 5 minutes"
                )

            if old_status != status:
                logger.info(
                    f"Component {component_name} status changed: "
                    f"{old_status.value} -> {status.value}"
                )

                # Alert on degraded status
                if status == ComponentHealth.DEGRADED:
                    self._alert_degraded(component_name, component_status)

    def get_status(self, component_name: str) -> ComponentStatus:
        """Get the current status of a component.

        Args:
            component_name: Name of the component.

        Returns:
            The current ComponentStatus.

        Raises:
            ValueError: If component is not registered.
        """
        with self._instance_lock:
            if component_name not in self._components:
                raise ValueError(f"Component {component_name} is not registered")

            # Return a copy to prevent external modification
            status = self._components[component_name]
            return ComponentStatus(
                name=status.name,
                status=status.status,
                last_heartbeat=status.last_heartbeat,
                error_count=status.error_count,
                last_error=status.last_error,
                recovery_attempts=status.recovery_attempts,
                error_timestamps=list(status.error_timestamps),
                metadata=dict(status.metadata),
            )

    def get_all_status(self) -> dict[str, ComponentStatus]:
        """Get the status of all registered components.

        Returns:
            Dictionary mapping component names to their status.
        """
        with self._instance_lock:
            return {
                name: ComponentStatus(
                    name=status.name,
                    status=status.status,
                    last_heartbeat=status.last_heartbeat,
                    error_count=status.error_count,
                    last_error=status.last_error,
                    recovery_attempts=status.recovery_attempts,
                    error_timestamps=list(status.error_timestamps),
                    metadata=dict(status.metadata),
                )
                for name, status in self._components.items()
            }

    def is_healthy(self, component_name: str) -> bool:
        """Check if a component is healthy.

        A component is considered healthy if its status is HEALTHY
        or RECOVERING (actively being fixed).

        Args:
            component_name: Name of the component to check.

        Returns:
            True if the component is healthy, False otherwise.

        Raises:
            ValueError: If component is not registered.
        """
        with self._instance_lock:
            if component_name not in self._components:
                raise ValueError(f"Component {component_name} is not registered")

            status = self._components[component_name].status
            return status in (ComponentHealth.HEALTHY, ComponentHealth.RECOVERING)

    def get_unhealthy_components(self) -> list[str]:
        """Get a list of all unhealthy component names.

        Returns:
            List of component names that are not healthy.
        """
        with self._instance_lock:
            return [
                name
                for name, status in self._components.items()
                if status.status not in (ComponentHealth.HEALTHY, ComponentHealth.RECOVERING)
            ]

    def register_recovery_strategy(
        self, component_pattern: str, recovery_func: RecoveryFunction
    ) -> None:
        """Register a recovery strategy for a component or component type.

        Args:
            component_pattern: Name or prefix of the component.
            recovery_func: Function to call for recovery. Should return True
                if recovery was successful.
        """
        with self._instance_lock:
            self._recovery_strategies[component_pattern] = recovery_func
            logger.debug(f"Registered recovery strategy for {component_pattern}")

    def start_monitoring(self, interval: int = 30) -> None:
        """Start the background health monitoring thread.

        Args:
            interval: Seconds between health checks.
        """
        with self._instance_lock:
            if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
                logger.warning("Health monitoring is already running")
                return

            self._monitoring_interval = interval
            self._stop_monitoring.clear()

            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="HealthMonitor",
                daemon=True,
            )
            self._monitoring_thread.start()

            logger.info(f"Health monitoring started with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop the background health monitoring thread."""
        with self._instance_lock:
            if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
                logger.warning("Health monitoring is not running")
                return

            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)

            if self._monitoring_thread.is_alive():
                logger.warning("Health monitoring thread did not stop gracefully")
            else:
                logger.info("Health monitoring stopped")

            self._monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_health()
            except Exception as e:
                logger.error(f"Error in health check: {e}")

            # Wait for interval or until stopped
            self._stop_monitoring.wait(self._monitoring_interval)

    def _check_health(self) -> None:
        """Check health of all components and trigger recovery if needed."""
        with self._instance_lock:
            now = datetime.now()

            for name, status in self._components.items():
                timeout = self._heartbeat_timeouts.get(name, 60)

                # Check for missed heartbeats (mark as FAILED)
                if status.status not in (
                    ComponentHealth.FAILED,
                    ComponentHealth.RECOVERING,
                ):
                    time_since_heartbeat = (now - status.last_heartbeat).total_seconds()
                    if time_since_heartbeat > timeout * self.FAILED_HEARTBEAT_MULTIPLIER:
                        old_status = status.status
                        status.status = ComponentHealth.FAILED
                        logger.error(
                            f"Component {name} marked FAILED: "
                            f"No heartbeat for {time_since_heartbeat:.1f}s "
                            f"(timeout: {timeout}s)"
                        )

                        # Trigger recovery
                        self._attempt_recovery(name)

                # Check for components stuck in DEGRADED for too long
                elif status.status == ComponentHealth.DEGRADED:
                    if status.error_timestamps:
                        time_in_degraded = (
                            now - status.error_timestamps[0]
                        ).total_seconds()
                        if time_in_degraded > self.DEGRADED_ALERT_THRESHOLD:
                            self._alert_degraded(name, status)

                # Check for FAILED components that need recovery
                elif status.status == ComponentHealth.FAILED:
                    if status.recovery_attempts < self.MAX_RECOVERY_ATTEMPTS:
                        self._attempt_recovery(name)
                    else:
                        logger.critical(
                            f"Component {name} has exceeded max recovery attempts "
                            f"({self.MAX_RECOVERY_ATTEMPTS}). Manual intervention required."
                        )

    def _attempt_recovery(self, component_name: str) -> bool:
        """Attempt to recover a failed component.

        Args:
            component_name: Name of the component to recover.

        Returns:
            True if recovery was initiated successfully, False otherwise.
        """
        with self._instance_lock:
            if component_name not in self._components:
                return False

            status = self._components[component_name]

            if status.recovery_attempts >= self.MAX_RECOVERY_ATTEMPTS:
                logger.error(
                    f"Max recovery attempts exceeded for {component_name}"
                )
                return False

            status.recovery_attempts += 1
            status.status = ComponentHealth.RECOVERING

            logger.info(
                f"Attempting recovery for {component_name} "
                f"(attempt {status.recovery_attempts}/{self.MAX_RECOVERY_ATTEMPTS})"
            )

            # Find recovery strategy
            recovery_func: Optional[RecoveryFunction] = None

            # Try exact match first
            if component_name in self._recovery_strategies:
                recovery_func = self._recovery_strategies[component_name]
            else:
                # Try pattern matching
                for pattern, func in self._recovery_strategies.items():
                    if component_name.startswith(pattern) or pattern in component_name:
                        recovery_func = func
                        break

            if recovery_func is None:
                logger.warning(f"No recovery strategy found for {component_name}")
                status.status = ComponentHealth.FAILED
                return False

        # Execute recovery outside the lock to avoid blocking
        try:
            success = recovery_func(component_name)

            with self._instance_lock:
                if component_name in self._components:
                    if success:
                        # Don't mark as healthy yet - wait for heartbeat
                        logger.info(f"Recovery initiated for {component_name}")
                    else:
                        # Recovery failed, mark as failed again
                        self._components[component_name].status = ComponentHealth.FAILED
                        logger.error(f"Recovery failed for {component_name}")

            return success

        except Exception as e:
            logger.error(f"Recovery error for {component_name}: {e}")
            with self._instance_lock:
                if component_name in self._components:
                    self._components[component_name].status = ComponentHealth.FAILED
            return False

    def _alert_degraded(self, component_name: str, status: ComponentStatus) -> None:
        """Send alert for degraded component.

        Args:
            component_name: Name of the degraded component.
            status: Current status of the component.
        """
        logger.warning(
            f"ALERT: Component {component_name} is DEGRADED. "
            f"Errors: {status.error_count}, Last error: {status.last_error}"
        )

        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(component_name, status)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def register_alert_callback(
        self, callback: Callable[[str, ComponentStatus], None]
    ) -> None:
        """Register a callback for health alerts.

        Args:
            callback: Function to call when a component becomes degraded.
                Receives component_name and ComponentStatus as arguments.
        """
        with self._instance_lock:
            self._alert_callbacks.append(callback)
            logger.debug(f"Registered alert callback: {callback.__name__}")

    def unregister_alert_callback(
        self, callback: Callable[[str, ComponentStatus], None]
    ) -> None:
        """Unregister an alert callback.

        Args:
            callback: The callback function to remove.
        """
        with self._instance_lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
                logger.debug(f"Unregistered alert callback: {callback.__name__}")

    def reset_component(self, component_name: str) -> None:
        """Reset a component's status and recovery attempts.

        This is useful for manual recovery or when a component
        has been fixed externally.

        Args:
            component_name: Name of the component to reset.

        Raises:
            ValueError: If component is not registered.
        """
        with self._instance_lock:
            if component_name not in self._components:
                raise ValueError(f"Component {component_name} is not registered")

            status = self._components[component_name]
            status.status = ComponentHealth.HEALTHY
            status.error_count = 0
            status.error_timestamps = []
            status.recovery_attempts = 0
            status.last_error = None
            status.last_heartbeat = datetime.now()

            logger.info(f"Component {component_name} has been reset to HEALTHY")


# Convenience function to get the singleton instance
def get_health_monitor() -> HealthMonitor:
    """Get the global HealthMonitor instance.

    Returns:
        The singleton HealthMonitor instance.
    """
    return HealthMonitor()


def require_healthy(component_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to require a component to be healthy before executing.

    Usage:
        @require_healthy("websocket_feed")
        def process_market_data(data: dict) -> None:
            # This will only run if websocket_feed is healthy
            pass

    Args:
        component_name: Name of the component that must be healthy.

    Returns:
        Decorator function that checks health before execution.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            monitor = HealthMonitor()

            try:
                if not monitor.is_healthy(component_name):
                    logger.warning(
                        f"Skipping {func.__name__}: {component_name} is not healthy"
                    )
                    raise ComponentNotHealthyError(
                        f"Component {component_name} is not healthy"
                    )
            except ValueError:
                # Component not registered
                logger.warning(
                    f"Component {component_name} not registered, "
                    f"allowing {func.__name__} to proceed"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ComponentNotHealthyError(Exception):
    """Exception raised when a required component is not healthy."""

    pass


# Memory monitoring helper
class MemoryMonitor:
    """Helper class for monitoring memory usage."""

    def __init__(self, threshold_percent: float = 85.0) -> None:
        """Initialize memory monitor.

        Args:
            threshold_percent: Memory usage percentage threshold for alerts.
        """
        self.threshold_percent = threshold_percent
        self._monitor = HealthMonitor()
        self._monitor.register_component("memory", heartbeat_timeout=60)

    def check_memory(self) -> bool:
        """Check current memory usage and update health status.

        Returns:
            True if memory usage is acceptable, False if high.
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent > self.threshold_percent:
                self._monitor.update_status(
                    "memory",
                    ComponentHealth.DEGRADED,
                    error=f"High memory usage: {usage_percent}%",
                )
                return False
            else:
                self._monitor.heartbeat("memory")
                return True

        except ImportError:
            # psutil not available, just heartbeat
            self._monitor.heartbeat("memory")
            return True
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return False
