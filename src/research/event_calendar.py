"""Event calendar analyzer for trading signal timing and risk assessment.

This module provides event-based analysis for trading signals, including:
- NSE holidays and trading calendar
- RBI MPC and US Fed meeting dates
- Weekly/Monthly expiry detection
- Time-of-day quality scoring
- Event risk assessment

All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

from datetime import datetime, date, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger
from .models import AnalysisComponent, EventData
from src.utils.degradation import graceful_degrade

# IST timezone
IST = ZoneInfo("Asia/Kolkata")


class EventCalendarAnalyzer:
    """Analyzer for event calendar, expiry dates, and time-of-day risk assessment.

    Provides scoring for trading conditions based on:
    - Proximity to market events (RBI/Fed meetings, Budget, etc.)
    - Weekly and monthly expiry dates
    - Time of day (market open/close, prime hours)
    - NSE holidays

    All timestamps are handled in IST (Asia/Kolkata).
    """

    # NSE Holidays 2024-2025 (hardcoded, configurable)
    NSE_HOLIDAYS_2024: set[date] = {
        date(2024, 1, 26),   # Republic Day
        date(2024, 3, 8),    # Mahashivratri
        date(2024, 3, 25),   # Holi
        date(2024, 3, 29),   # Good Friday
        date(2024, 4, 11),   # Id-Ul-Fitr
        date(2024, 4, 17),   # Ram Navami
        date(2024, 5, 1),    # Maharashtra Day
        date(2024, 6, 17),   # Bakri Id
        date(2024, 7, 17),   # Moharram
        date(2024, 8, 15),   # Independence Day
        date(2024, 9, 7),    # Ganesh Chaturthi
        date(2024, 10, 2),   # Gandhi Jayanti
        date(2024, 10, 12),  # Dussehra
        date(2024, 11, 1),   # Diwali (Laxmi Pujan)
        date(2024, 11, 2),   # Diwali (Balipratipada)
        date(2024, 11, 15),  # Guru Nanak Jayanti
        date(2024, 12, 25),  # Christmas
    }

    NSE_HOLIDAYS_2025: set[date] = {
        date(2025, 1, 26),   # Republic Day
        date(2025, 2, 26),   # Mahashivratri
        date(2025, 3, 14),   # Holi
        date(2025, 3, 31),   # Id-Ul-Fitr
        date(2025, 4, 10),   # Good Friday
        date(2025, 4, 14),   # Ambedkar Jayanti
        date(2025, 4, 18),   # Ram Navami
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 6, 7),    # Bakri Id
        date(2025, 7, 6),    # Moharram
        date(2025, 8, 15),   # Independence Day
        date(2025, 8, 27),   # Ganesh Chaturthi
        date(2025, 10, 2),   # Gandhi Jayanti
        date(2025, 10, 21),  # Dussehra
        date(2025, 10, 22),  # Diwali (Laxmi Pujan)
        date(2025, 11, 5),   # Guru Nanak Jayanti
        date(2025, 12, 25),  # Christmas
    }

    # RBI MPC Meeting Dates 2024-2025 (hardcoded)
    RBI_MPC_DATES_2024: set[date] = {
        date(2024, 2, 6),    # Feb MPC
        date(2024, 2, 8),    # Feb MPC (policy day)
        date(2024, 4, 3),    # Apr MPC
        date(2024, 4, 5),    # Apr MPC (policy day)
        date(2024, 6, 5),    # Jun MPC
        date(2024, 6, 7),    # Jun MPC (policy day)
        date(2024, 8, 6),    # Aug MPC
        date(2024, 8, 8),    # Aug MPC (policy day)
        date(2024, 10, 7),   # Oct MPC
        date(2024, 10, 9),   # Oct MPC (policy day)
        date(2024, 12, 4),   # Dec MPC
        date(2024, 12, 6),   # Dec MPC (policy day)
    }

    RBI_MPC_DATES_2025: set[date] = {
        date(2025, 2, 4),    # Feb MPC
        date(2025, 2, 6),    # Feb MPC (policy day)
        date(2025, 4, 2),    # Apr MPC
        date(2025, 4, 4),    # Apr MPC (policy day)
        date(2025, 6, 4),    # Jun MPC
        date(2025, 6, 6),    # Jun MPC (policy day)
        date(2025, 8, 4),    # Aug MPC
        date(2025, 8, 6),    # Aug MPC (policy day)
        date(2025, 10, 6),   # Oct MPC
        date(2025, 10, 8),   # Oct MPC (policy day)
        date(2025, 12, 3),   # Dec MPC
        date(2025, 12, 5),   # Dec MPC (policy day)
    }

    # US Fed Meeting Dates 2024-2025 (hardcoded)
    US_FED_DATES_2024: set[date] = {
        date(2024, 1, 30),   # Jan FOMC
        date(2024, 1, 31),   # Jan FOMC (policy day)
        date(2024, 3, 19),   # Mar FOMC
        date(2024, 3, 20),   # Mar FOMC (policy day)
        date(2024, 4, 30),   # Apr/May FOMC
        date(2024, 5, 1),    # Apr/May FOMC (policy day)
        date(2024, 6, 11),   # Jun FOMC
        date(2024, 6, 12),   # Jun FOMC (policy day)
        date(2024, 7, 30),   # Jul FOMC
        date(2024, 7, 31),   # Jul FOMC (policy day)
        date(2024, 9, 17),   # Sep FOMC
        date(2024, 9, 18),   # Sep FOMC (policy day)
        date(2024, 11, 6),   # Nov FOMC
        date(2024, 11, 7),   # Nov FOMC (policy day)
        date(2024, 12, 17),  # Dec FOMC
        date(2024, 12, 18),  # Dec FOMC (policy day)
    }

    US_FED_DATES_2025: set[date] = {
        date(2025, 1, 28),   # Jan FOMC
        date(2025, 1, 29),   # Jan FOMC (policy day)
        date(2025, 3, 18),   # Mar FOMC
        date(2025, 3, 19),   # Mar FOMC (policy day)
        date(2025, 4, 29),   # Apr/May FOMC
        date(2025, 4, 30),   # Apr/May FOMC (policy day)
        date(2025, 6, 17),   # Jun FOMC
        date(2025, 6, 18),   # Jun FOMC (policy day)
        date(2025, 7, 29),   # Jul FOMC
        date(2025, 7, 30),   # Jul FOMC (policy day)
        date(2025, 9, 16),   # Sep FOMC
        date(2025, 9, 17),   # Sep FOMC (policy day)
        date(2025, 10, 28),  # Oct/Nov FOMC
        date(2025, 10, 29),  # Oct/Nov FOMC (policy day)
        date(2025, 12, 16),  # Dec FOMC
        date(2025, 12, 17),  # Dec FOMC (policy day)
    }

    # Budget Day (February 1st every year)
    BUDGET_DAYS: set[date] = {
        date(2024, 2, 1),
        date(2025, 2, 1),
        date(2026, 2, 1),
    }

    # Quarterly Earnings Seasons (approximate periods)
    # Format: (start_date, end_date) for each quarter
    EARNINGS_SEASONS_2024: list[tuple[date, date]] = [
        (date(2024, 1, 8), date(2024, 2, 15)),   # Q3 FY24
        (date(2024, 4, 8), date(2024, 5, 31)),   # Q4 FY24 (full year)
        (date(2024, 7, 8), date(2024, 8, 14)),   # Q1 FY25
        (date(2024, 10, 7), date(2024, 11, 14)), # Q2 FY25
    ]

    EARNINGS_SEASONS_2025: list[tuple[date, date]] = [
        (date(2025, 1, 8), date(2025, 2, 15)),   # Q3 FY25
        (date(2025, 4, 8), date(2025, 5, 31)),   # Q4 FY25 (full year)
        (date(2025, 7, 8), date(2025, 8, 14)),   # Q1 FY26
        (date(2025, 10, 7), date(2025, 11, 14)), # Q2 FY26
    ]

    # Weekly expiry day (Thursday for Nifty/BankNifty, but varies by underlying)
    # Nifty and BankNifty: Thursday (or previous trading day if holiday)
    DEFAULT_WEEKLY_EXPIRY_DAY = 3  # Thursday (0=Monday, 6=Sunday)

    # Monthly expiry day (last Thursday of month)
    MONTHLY_EXPIRY_DAY = 3  # Thursday

    def __init__(self):
        """Initialize the event calendar analyzer."""
        self.holidays = self.NSE_HOLIDAYS_2024 | self.NSE_HOLIDAYS_2025
        self.rbi_dates = self.RBI_MPC_DATES_2024 | self.RBI_MPC_DATES_2025
        self.fed_dates = self.US_FED_DATES_2024 | self.US_FED_DATES_2025
        self.budget_days = self.BUDGET_DAYS
        self.earnings_seasons = self.EARNINGS_SEASONS_2024 + self.EARNINGS_SEASONS_2025
        logger.info("EventCalendarAnalyzer initialized")

    def is_nse_holiday(self, check_date: date) -> bool:
        """Check if the given date is an NSE holiday.

        Args:
            check_date: Date to check

        Returns:
            True if holiday, False otherwise
        """
        return check_date in self.holidays

    def is_weekly_expiry(
        self,
        check_date: date,
        underlying: Optional[str] = None
    ) -> bool:
        """Check if the given date is a weekly expiry day.

        For Nifty and BankNifty, weekly expiry is Thursday.
        For stock options, it varies (typically Thursday).

        Args:
            check_date: Date to check
            underlying: Optional underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')

        Returns:
            True if weekly expiry day, False otherwise
        """
        # Check if it's a holiday (would have shifted expiry)
        if self.is_nse_holiday(check_date):
            return False

        # Check if it's the last Thursday (monthly expiry takes precedence)
        if self.is_monthly_expiry(check_date):
            return True  # Monthly expiry is also weekly expiry

        # Check if it's the expected expiry day of week
        if check_date.weekday() == self.DEFAULT_WEEKLY_EXPIRY_DAY:
            return True

        # Check if previous day was a holiday (expiry shifted to today)
        prev_day = check_date - timedelta(days=1)
        if prev_day.weekday() == self.DEFAULT_WEEKLY_EXPIRY_DAY - 1:
            if self.is_nse_holiday(prev_day):
                return True

        return False

    def is_monthly_expiry(self, check_date: date) -> bool:
        """Check if the given date is a monthly expiry day (last Thursday).

        Args:
            check_date: Date to check

        Returns:
            True if monthly expiry day, False otherwise
        """
        if check_date.weekday() != self.MONTHLY_EXPIRY_DAY:
            return False

        if self.is_nse_holiday(check_date):
            return False

        # Check if this is the last Thursday of the month
        next_week = check_date + timedelta(days=7)
        return next_week.month != check_date.month

    def days_to_monthly_expiry(self, from_date: date) -> int:
        """Calculate days to the next monthly expiry.

        Args:
            from_date: Starting date

        Returns:
            Number of days to next monthly expiry (0 if today is expiry)
        """
        current = from_date

        # Search forward up to 35 days (max ~5 weeks)
        for _ in range(35):
            if self.is_monthly_expiry(current):
                return (current - from_date).days
            current += timedelta(days=1)

        # Fallback: calculate based on last Thursday logic
        return self._calculate_days_to_last_thursday(from_date)

    def _calculate_days_to_last_thursday(self, from_date: date) -> int:
        """Calculate days to last Thursday of current month.

        Args:
            from_date: Starting date

        Returns:
            Days to last Thursday
        """
        # Get last day of current month
        if from_date.month == 12:
            last_day = date(from_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(from_date.year, from_date.month + 1, 1) - timedelta(days=1)

        # Find last Thursday
        days_back = (last_day.weekday() - self.MONTHLY_EXPIRY_DAY) % 7
        last_thursday = last_day - timedelta(days=days_back)

        # If last Thursday has passed, go to next month
        if last_thursday < from_date:
            if from_date.month == 12:
                next_month_last = date(from_date.year + 1, 2, 1) - timedelta(days=1)
            else:
                next_month_last = date(from_date.year, from_date.month + 2, 1) - timedelta(days=1)
            days_back = (next_month_last.weekday() - self.MONTHLY_EXPIRY_DAY) % 7
            last_thursday = next_month_last - timedelta(days=days_back)

        return (last_thursday - from_date).days

    def is_rbi_policy_day(self, check_date: date) -> bool:
        """Check if the given date is an RBI MPC policy announcement day.

        Args:
            check_date: Date to check

        Returns:
            True if RBI policy day, False otherwise
        """
        return check_date in self.rbi_dates

    def is_fed_day(self, check_date: date) -> bool:
        """Check if the given date is a US Fed meeting/policy day.

        Args:
            check_date: Date to check

        Returns:
            True if Fed day, False otherwise
        """
        return check_date in self.fed_dates

    def is_budget_day(self, check_date: date) -> bool:
        """Check if the given date is Budget day (Feb 1).

        Args:
            check_date: Date to check

        Returns:
            True if Budget day, False otherwise
        """
        return check_date in self.budget_days

    def is_earnings_season(self, check_date: date) -> bool:
        """Check if the given date falls within an earnings season.

        Args:
            check_date: Date to check

        Returns:
            True if during earnings season, False otherwise
        """
        for start, end in self.earnings_seasons:
            if start <= check_date <= end:
                return True
        return False

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="event_risk",
            score=50.0,
            weight=0.15,
            confidence=30.0,
            reasoning="Event risk analysis failed - using default neutral score",
            data={"avoid": False, "score": 50},
        )
    )
    def analyze_event_risk(
        self,
        check_date: date,
        underlying: Optional[str] = None
    ) -> AnalysisComponent:
        """Analyze event risk for a given date and underlying.

        Scoring:
        - No event = score 80
        - Weekly expiry = score 60
        - Monthly expiry week = score 50
        - RBI/Fed before announcement = score 30
        - Budget/Election = score 10

        Args:
            check_date: Date to analyze
            underlying: Optional underlying symbol

        Returns:
            AnalysisComponent with event risk score and EventData
        """
        event_data = EventData()
        reasoning_parts = []
        score = 80  # Default: no event
        risk_level = "low"

        # Check for Budget day (highest risk)
        if self.is_budget_day(check_date):
            score = 10
            risk_level = "extreme"
            event_data.is_budget_day = True
            reasoning_parts.append("Budget Day - extreme volatility expected")

        # Check for RBI policy day
        elif self.is_rbi_policy_day(check_date):
            score = 30
            risk_level = "high"
            event_data.is_rbi_policy_day = True
            reasoning_parts.append("RBI MPC policy announcement day")

        # Check for Fed day
        elif self.is_fed_day(check_date):
            score = 30
            risk_level = "high"
            event_data.is_fed_day = True
            reasoning_parts.append("US Fed policy announcement day")

        # Check for monthly expiry
        elif self.is_monthly_expiry(check_date):
            score = 50
            risk_level = "high"
            event_data.is_monthly_expiry = True
            event_data.is_weekly_expiry = True
            event_data.days_to_monthly_expiry = 0
            reasoning_parts.append("Monthly expiry day")

        # Check for weekly expiry
        elif self.is_weekly_expiry(check_date, underlying):
            score = 60
            risk_level = "medium"
            event_data.is_weekly_expiry = True
            event_data.days_to_monthly_expiry = self.days_to_monthly_expiry(check_date)
            reasoning_parts.append("Weekly expiry day")

        # Check if in monthly expiry week (but not expiry day)
        else:
            days_to_expiry = self.days_to_monthly_expiry(check_date)
            event_data.days_to_monthly_expiry = days_to_expiry

            if days_to_expiry <= 7:
                score = 50
                risk_level = "medium"
                reasoning_parts.append(f"Monthly expiry week ({days_to_expiry} days to expiry)")
            elif days_to_expiry <= 14:
                score = 65
                risk_level = "medium"
                reasoning_parts.append(f"Approaching monthly expiry ({days_to_expiry} days)")
            else:
                reasoning_parts.append(f"{days_to_expiry} days to monthly expiry")

        # Check for earnings season
        if self.is_earnings_season(check_date):
            event_data.is_earnings_season = True
            if risk_level == "low":
                score = max(score - 10, 40)
                risk_level = "medium"
            reasoning_parts.append("Earnings season active")

        # Finalize event data
        event_data.event_risk_level = risk_level

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant events"

        logger.debug(
            f"Event risk analysis for {check_date}: score={score}, "
            f"risk_level={risk_level}, underlying={underlying}"
        )

        return AnalysisComponent(
            name="event_risk",
            score=float(score),
            weight=0.15,
            confidence=90.0,
            reasoning=reasoning,
            data=event_data.model_dump()
        )

    @graceful_degrade(
        default_return=AnalysisComponent(
            name="time_of_day",
            score=60.0,
            weight=0.10,
            confidence=30.0,
            reasoning="Time of day analysis failed - using default score",
            data={"score": 60},
        )
    )
    def analyze_time_of_day(self, timestamp: datetime) -> AnalysisComponent:
        """Analyze trading quality based on time of day.

        Scoring:
        - 9:15-9:30 = score 40 (opening volatility)
        - 9:30-10:00 = score 70 (trend establishing)
        - 10:00-14:00 = score 90 (prime hours)
        - 14:00-15:00 = score 60 (theta decay accelerates)
        - 15:00-15:15 = score 30 (square-off pressure)
        - After 15:15 = score 0 (market closed)

        Args:
            timestamp: Timestamp to analyze (must be IST)

        Returns:
            AnalysisComponent with time-of-day score
        """
        # Ensure timestamp is in IST
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)
        else:
            timestamp = timestamp.astimezone(IST)

        current_time = timestamp.time()

        # Define time ranges
        market_open = time(9, 15)
        volatile_end = time(9, 30)
        trend_establish_end = time(10, 0)
        prime_hours_end = time(14, 0)
        theta_decay_end = time(15, 0)
        square_off_end = time(15, 15)

        if current_time < market_open:
            score = 0
            reasoning = "Pre-market - no trading"
        elif market_open <= current_time < volatile_end:
            score = 40
            reasoning = "Opening volatility (9:15-9:30) - high spreads, avoid new positions"
        elif volatile_end <= current_time < trend_establish_end:
            score = 70
            reasoning = "Trend establishing (9:30-10:00) - good for momentum"
        elif trend_establish_end <= current_time < prime_hours_end:
            score = 90
            reasoning = "Prime trading hours (10:00-14:00) - best liquidity and stability"
        elif prime_hours_end <= current_time < theta_decay_end:
            score = 60
            reasoning = "Afternoon session (14:00-15:00) - theta decay accelerates"
        elif theta_decay_end <= current_time < square_off_end:
            score = 30
            reasoning = "Square-off period (15:00-15:15) - high volatility, reduce positions"
        else:
            score = 0
            reasoning = "Market closed (after 15:15)"

        logger.debug(
            f"Time of day analysis for {current_time}: score={score}"
        )

        return AnalysisComponent(
            name="time_of_day",
            score=float(score),
            weight=0.10,
            confidence=95.0,
            reasoning=reasoning,
            data={"time": current_time.isoformat()}
        )

    @graceful_degrade(
        default_return=(
            AnalysisComponent(
                name="event_calendar",
                score=50.0,
                weight=0.15,
                confidence=30.0,
                reasoning="Event calendar analysis failed - using default neutral score",
                data={"avoid": False, "score": 50},
            ),
            EventData()
        )
    )
    def analyze(
        self,
        timestamp: datetime,
        underlying: Optional[str] = None
    ) -> tuple[AnalysisComponent, EventData]:
        """Run complete event calendar analysis.

        Combines event risk and time-of-day analysis.

        Args:
            timestamp: Timestamp to analyze (must be IST-aware or naive)
            underlying: Optional underlying symbol for expiry checks

        Returns:
            Tuple of (combined AnalysisComponent, EventData)
        """
        logger.info(
            f"Starting event calendar analysis for {timestamp}, underlying={underlying}"
        )

        # Ensure timestamp is in IST
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)
        else:
            timestamp = timestamp.astimezone(IST)

        check_date = timestamp.date()

        # Run individual analyses
        event_risk = self.analyze_event_risk(check_date, underlying)
        time_of_day = self.analyze_time_of_day(timestamp)

        # Extract EventData from event_risk
        event_data = EventData(**event_risk.data)
        event_data.time_of_day_score = time_of_day.score

        # Combine scores with weights
        # Event risk: 60%, Time of day: 40%
        combined_score = (
            event_risk.score * 0.60 +
            time_of_day.score * 0.40
        )

        # Determine combined reasoning
        if combined_score >= 75:
            overall = "Favorable conditions"
        elif combined_score >= 50:
            overall = "Moderate conditions - exercise caution"
        elif combined_score >= 25:
            overall = "Challenging conditions - reduce size"
        else:
            overall = "Avoid trading"

        combined_reasoning = (
            f"{overall}. Event: {event_risk.reasoning}. "
            f"Time: {time_of_day.reasoning}"
        )

        combined_component = AnalysisComponent(
            name="event_calendar",
            score=combined_score,
            weight=0.15,
            confidence=min(event_risk.confidence, time_of_day.confidence),
            reasoning=combined_reasoning,
            data={
                "event_risk": event_risk.model_dump(),
                "time_of_day": time_of_day.model_dump(),
                "event_data": event_data.model_dump()
            }
        )

        logger.info(
            f"Event calendar analysis complete: combined_score={combined_score:.1f}, "
            f"event_risk={event_risk.score}, time_of_day={time_of_day.score}"
        )

        return combined_component, event_data

    def add_custom_holiday(self, holiday_date: date) -> None:
        """Add a custom holiday to the calendar.

        Args:
            holiday_date: Date to add as holiday
        """
        self.holidays.add(holiday_date)
        logger.info(f"Added custom holiday: {holiday_date}")

    def add_custom_event(self, event_date: date, event_type: str) -> None:
        """Add a custom event date.

        Args:
            event_date: Date of the event
            event_type: Type of event ('rbi', 'fed', 'budget')
        """
        if event_type == 'rbi':
            self.rbi_dates.add(event_date)
        elif event_type == 'fed':
            self.fed_dates.add(event_date)
        elif event_type == 'budget':
            self.budget_days.add(event_date)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
        logger.info(f"Added custom {event_type} event: {event_date}")
