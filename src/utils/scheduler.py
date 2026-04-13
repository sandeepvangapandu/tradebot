"""Task scheduler wrapper around APScheduler."""

from __future__ import annotations

from typing import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

IST = ZoneInfo("Asia/Kolkata")


class TaskScheduler:
    """Background task scheduler configured for IST timezone.

    Wraps APScheduler's ``BackgroundScheduler`` with convenience methods
    for adding daily cron jobs.
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone=IST)

    def add_daily_job(
        self,
        func: Callable[..., object],
        hour: int,
        minute: int,
        job_id: str,
    ) -> None:
        """Schedule a function to run daily at the given IST time.

        Args:
            func: Callable to execute.
            hour: Hour of day (0-23) in IST.
            minute: Minute of hour (0-59).
            job_id: Unique identifier for this job.
        """
        trigger = CronTrigger(hour=hour, minute=minute, timezone=IST)
        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
        )

    def start(self) -> None:
        """Start the background scheduler."""
        if not self._scheduler.running:
            self._scheduler.start()

    def stop(self) -> None:
        """Gracefully shut down the scheduler, waiting for running jobs."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)


_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    """Get or create the global TaskScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
