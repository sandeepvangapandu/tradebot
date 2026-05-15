#!/usr/bin/env python3
"""Trading Bot Runner - Manages bot lifecycle for Indian market hours.

This wrapper script:
1. Starts the trading bot at 9:15 AM IST on trading days
2. Stops the bot at 3:30 PM IST
3. Sends EOD summary via SMS
4. Handles graceful shutdown

Designed to run as a user service on Zo Computer.
"""

import subprocess
import sys
import time
import signal
import os
import sqlite3
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# Add the Trading directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.holidays import is_trading_day

IST = ZoneInfo("Asia/Kolkata")

# Market hours (IST)
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)

# Setup logging to both stdout and file
def setup_logging():
    """Setup logging for the runner."""
    import logging
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "runner.log")
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class BotRunner:
    def __init__(self):
        self.bot_process = None
        self.running = True
        self._setup_signal_handlers()
        self.script_dir = Path(__file__).parent
        self.db_path = self.script_dir / "data" / "trading_bot.db"
        self._eod_sent_today = None  # Track if EOD was sent today
    
    def _setup_signal_handlers(self):
        """Handle shutdown signals gracefully."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self._stop_bot()
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now(IST)
        
        # Check if it's a trading day (weekday + not a holiday)
        if not is_trading_day(now.date()):
            return False
        
        # Check if within market hours
        current_time = now.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE
    
    def _time_to_open(self) -> float:
        """Calculate seconds until next market open."""
        now = datetime.now(IST)
        
        # If it's a trading day and before market open
        if is_trading_day(now.date()) and now.time() < MARKET_OPEN:
            market_open_dt = datetime.combine(now.date(), MARKET_OPEN)
            market_open_dt = market_open_dt.replace(tzinfo=IST)
            return (market_open_dt - now).total_seconds()
        
        # Otherwise, find next trading day
        next_day = now.date() + timedelta(days=1)
        while not is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        market_open_dt = datetime.combine(next_day, MARKET_OPEN)
        market_open_dt = market_open_dt.replace(tzinfo=IST)
        return (market_open_dt - now).total_seconds()
    
    def _time_to_close(self) -> float:
        """Calculate seconds until market close."""
        now = datetime.now(IST)
        
        if is_trading_day(now.date()) and MARKET_OPEN <= now.time() < MARKET_CLOSE:
            market_close_dt = datetime.combine(now.date(), MARKET_CLOSE)
            market_close_dt = market_close_dt.replace(tzinfo=IST)
            return (market_close_dt - now).total_seconds()
        
        return 0
    
    def _kill_stale_procs(self):
        """Kill any pre-existing python -m src.main procs for this user.

        Prevents the bot-restart cascade that caused 11 starts in 84 minutes
        on 2026-05-15. Without this, two bot instances fight for the same
        Upstox WebSocket session and Upstox 403s the new one.
        """
        try:
            import signal as _sig
            out = subprocess.check_output(["pgrep", "-f", "src.main"], text=True).strip()
            pids = [int(p) for p in out.splitlines() if p.strip().isdigit()]
            own_pid = os.getpid()
            for pid in pids:
                if pid == own_pid:
                    continue
                try:
                    os.kill(pid, _sig.SIGTERM)
                    logger.warning(f"Killed stale src.main PID {pid}")
                except ProcessLookupError:
                    pass
            if pids:
                time.sleep(2)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # No stale procs or pgrep unavailable
        except Exception as exc:
            logger.debug(f"Stale-proc cleanup error (ignored): {exc}")

    def _start_bot(self):
        """Start the trading bot process.

        Hardened 2026-05-15:
        - Kill stale src.main procs first (avoids Upstox session conflicts).
        - Track restart history; halt after 5 restarts in 10 min to prevent
          all-day restart cascades.
        """
        # Restart cascade guard
        now = time.time()
        if not hasattr(self, "_recent_starts"):
            self._recent_starts: list[float] = []
        self._recent_starts = [t for t in self._recent_starts if now - t < 600]
        if len(self._recent_starts) >= 5:
            logger.critical(
                "Restart cascade detected: %d restarts in 10 min. Halting. "
                "Manual recovery: kill runner, investigate logs.",
                len(self._recent_starts),
            )
            self.running = False
            return
        self._recent_starts.append(now)

        self._kill_stale_procs()

        logger.info("=" * 60)
        logger.info("Starting trading bot...")
        logger.info(f"Database: {self.db_path}")

        # Run the bot with Python 3
        self.bot_process = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            cwd=self.script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(self.script_dir)}
        )

        # Boot grace: ignore exit / health checks for 180s after spawn.
        # Covers slow startup (instrument master download + WebSocket
        # handshake + initial bar seeding).
        self._boot_grace_until = time.time() + 180

        logger.info(f"Bot started with PID {self.bot_process.pid}")
    
    def _stop_bot(self):
        """Stop the trading bot process gracefully."""
        if self.bot_process and self.bot_process.poll() is None:
            logger.info("=" * 60)
            logger.info("Stopping trading bot...")
            
            # Send SIGTERM first for graceful shutdown
            self.bot_process.terminate()
            
            # Wait up to 30 seconds for graceful shutdown
            try:
                self.bot_process.wait(timeout=30)
                logger.info("Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                logger.warning("Force killing bot...")
                self.bot_process.kill()
                self.bot_process.wait()
                logger.info("Bot killed")
            
            self.bot_process = None
    
    def _get_today_summary(self) -> dict:
        """Query database for today's trading summary."""
        summary = {
            "date": datetime.now(IST).strftime('%Y-%m-%d'),
            "trades": 0,
            "realized_pnl_paisa": 0,
            "unrealized_pnl_paisa": 0,
            "wins": 0,
            "losses": 0,
            "open_positions": 0,
        }
        
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return summary
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get today's date in IST
            today = datetime.now(IST).strftime('%Y-%m-%d')
            
            # Get daily P&L for today
            cursor.execute(
                "SELECT realized_pnl, unrealized_pnl, trades_count, win_count FROM daily_pnl WHERE date = ?",
                (today,)
            )
            row = cursor.fetchone()
            if row:
                summary["realized_pnl_paisa"] = row[0] or 0
                summary["unrealized_pnl_paisa"] = row[1] or 0
                summary["trades"] = row[2] or 0
                summary["wins"] = row[3] or 0
                summary["losses"] = summary["trades"] - summary["wins"]
            
            # Count open positions
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
            summary["open_positions"] = cursor.fetchone()[0] or 0
            
            conn.close()
            logger.info(f"Retrieved summary for {today}: {summary}")
        except Exception as e:
            logger.error(f"Error querying database: {e}")
        
        return summary
    
    def _send_eod_summary(self):
        """Send end-of-day summary via email/SMS and save to file."""
        now = datetime.now(IST)
        today_str = now.strftime('%Y-%m-%d')
        
        # Check if we already sent EOD for today
        if self._eod_sent_today == today_str:
            logger.info("EOD summary already sent today, skipping")
            return
        
        logger.info("=" * 60)
        logger.info("Generating EOD summary...")
        
        # Get today's summary from database
        summary = self._get_today_summary()
        
        # Convert paisa to rupees
        realized_pnl_rupees = summary["realized_pnl_paisa"] / 100
        unrealized_pnl_rupees = summary["unrealized_pnl_paisa"] / 100
        
        # Format P&L with sign
        pnl_sign = "+" if realized_pnl_rupees >= 0 else ""
        pnl_emoji = "🟢" if realized_pnl_rupees >= 0 else "🔴"
        
        # Create summary text
        summary_text = f"""📊 Trading Bot EOD Summary
{'='*40}
Date: {summary['date']}
Time: {now.strftime('%H:%M:%S IST')}

📈 Trades: {summary['trades']}
   Wins: {summary['wins']} | Losses: {summary['losses']}
   Win Rate: {(summary['wins']/summary['trades']*100):.1f}%""" if summary['trades'] > 0 else """📊 Trading Bot EOD Summary
{'='*40}
Date: {summary['date']}
Time: {now.strftime('%H:%M:%S IST')}

📈 Trades: 0"""

        summary_text += f"""
💰 Realized P&L: {pnl_emoji} ₹{pnl_sign}{realized_pnl_rupees:,.2f}
📊 Open Positions: {summary['open_positions']}

---
Database: {self.db_path}
Logs: {self.script_dir / 'logs'}
"""
        
        # Save to file
        summary_file = self.script_dir / "logs" / "eod_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"EOD Summary saved to {summary_file}")
        logger.info(summary_text)
        
        # Mark as sent for today
        self._eod_sent_today = today_str
    
    def run(self):
        """Main runner loop."""
        logger.info("=" * 60)
        logger.info("Trading Bot Runner Started")
        logger.info(f"Current time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
        logger.info(f"Market hours: {MARKET_OPEN.strftime('%H:%M')} - {MARKET_CLOSE.strftime('%H:%M')} IST")
        logger.info(f"Database: {self.db_path}")
        logger.info("=" * 60)
        
        while self.running:
            now = datetime.now(IST)
            
            # Check if it's market hours
            if self._is_market_hours():
                # Market is open
                if self.bot_process is None or self.bot_process.poll() is not None:
                    # Bot is not running, start it
                    self._start_bot()
                
                # Sleep for 1 minute, then check again
                time.sleep(60)
                
            else:
                # Market is closed
                if self.bot_process and self.bot_process.poll() is None:
                    # Bot is still running, stop it
                    self._stop_bot()
                    self._send_eod_summary()
                
                # Calculate time until next market open
                seconds_to_open = self._time_to_open()
                hours = int(seconds_to_open // 3600)
                minutes = int((seconds_to_open % 3600) // 60)
                
                logger.info(f"Market closed. Next open in {hours}h {minutes}m")
                
                # Sleep until market opens (check every 5 minutes)
                sleep_time = min(seconds_to_open, 300)
                time.sleep(sleep_time)
        
        # Final cleanup
        self._stop_bot()
        logger.info("Shutdown complete")


def main():
    runner = BotRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
