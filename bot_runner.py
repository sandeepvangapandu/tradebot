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


class BotRunner:
    def __init__(self):
        self.bot_process = None
        self.running = True
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Handle shutdown signals gracefully."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[Runner] Received signal {signum}, shutting down...")
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
    
    def _start_bot(self):
        """Start the trading bot process."""
        print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}] Starting trading bot...")
        
        # Run the bot with Python 3
        self.bot_process = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"[Runner] Bot started with PID {self.bot_process.pid}")
    
    def _stop_bot(self):
        """Stop the trading bot process gracefully."""
        if self.bot_process and self.bot_process.poll() is None:
            print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}] Stopping trading bot...")
            
            # Send SIGTERM first for graceful shutdown
            self.bot_process.terminate()
            
            # Wait up to 30 seconds for graceful shutdown
            try:
                self.bot_process.wait(timeout=30)
                print("[Runner] Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                print("[Runner] Force killing bot...")
                self.bot_process.kill()
                self.bot_process.wait()
                print("[Runner] Bot killed")
            
            self.bot_process = None
    
    def _get_today_summary(self) -> dict:
        """Query database for today's trading summary."""
        db_path = Path(__file__).parent / "data" / "trading_bot.db"
        summary = {
            "trades": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_positions": 0,
        }
        
        if not db_path.exists():
            return summary
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get today's date in IST
            today = datetime.now(IST).strftime('%Y-%m-%d')
            
            # Count today's trades
            cursor.execute(
                f"SELECT COUNT(*) FROM trades WHERE date(timestamp) = '{today}'"
            )
            summary["trades"] = cursor.fetchone()[0] or 0

            # Get daily P&L
            cursor.execute(
                "SELECT realized_pnl, unrealized_pnl FROM daily_pnl ORDER BY date DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                summary["realized_pnl"] = row[0] or 0.0
                summary["unrealized_pnl"] = row[1] or 0.0
            
            # Count open positions
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'")
            summary["open_positions"] = cursor.fetchone()[0] or 0
            
            conn.close()
        except Exception as e:
            print(f"[Runner] Error querying database: {e}")
        
        return summary
    
    def _send_eod_summary(self):
        """Send end-of-day summary via SMS."""
        print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}] Sending EOD summary...")
        
        # Get today's summary from database
        summary = self._get_today_summary()
        today = datetime.now(IST).strftime('%Y-%m-%d')
        
        # Format P&L with sign (convert paisa to rupees)
        pnl = summary["realized_pnl"] / 100
        pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"-₹{abs(pnl):.2f}"
        
        # Create concise SMS (under 400 chars)
        sms_message = (
            f"📊 Trading Bot EOD\n"
            f"Date: {today}\n"
            f"Trades: {summary['trades']}\n"
            f"P&L: {pnl_str}\n"
            f"Open Pos: {summary['open_positions']}\n"
            f"Status: Market Closed"
        )
        
        print(f"[Runner] EOD Summary:\n{sms_message}")
        
        # Send SMS via Zo API
        zo_token = os.environ.get("ZO_CLIENT_IDENTITY_TOKEN")
        if zo_token:
            try:
                import requests
                response = requests.post(
                    "https://api.zo.computer/zo/ask",
                    headers={
                        "authorization": zo_token,
                        "content-type": "application/json"
                    },
                    json={
                        "input": f"Send an SMS to the user with this exact message:\n\n{sms_message}",
                        "model_name": "zo:fast"
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    print("[Runner] EOD summary SMS request sent")
                else:
                    print(f"[Runner] Failed to send SMS: {response.status_code}")
            except Exception as e:
                print(f"[Runner] Error sending SMS: {e}")
        else:
            print("[Runner] ZO_CLIENT_IDENTITY_TOKEN not set, skipping SMS")
        
        # Also save to file for reference
        summary_file = Path(__file__).parent / "logs" / "eod_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write(f"📊 Trading Bot EOD Summary\n{'='*40}\n")
            f.write(f"Date: {today}\nTime: {datetime.now(IST).strftime('%H:%M:%S IST')}\n\n")
            f.write(f"📈 Trades: {summary['trades']}\n")
            f.write(f"💰 Realized P&L: {pnl_str}\n")
            f.write(f"📊 Open Positions: {summary['open_positions']}\n")
            f.write(f"\n---\nDatabase: {Path(__file__).parent}/data/trading_bot.db\n")
            f.write(f"Logs: {Path(__file__).parent}/logs/trading_bot.log\n")
    
    def run(self):
        """Main runner loop."""
        print("=" * 60)
        print("Trading Bot Runner Started")
        print(f"Current time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"Market hours: {MARKET_OPEN.strftime('%H:%M')} - {MARKET_CLOSE.strftime('%H:%M')} IST")
        print("=" * 60)
        
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
                
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S IST')}] Market closed")
                print(f"[Runner] Next market open in {hours}h {minutes}m")
                
                # Sleep until market opens (check every 5 minutes)
                sleep_time = min(seconds_to_open, 300)
                time.sleep(sleep_time)
        
        # Final cleanup
        self._stop_bot()
        print("\n[Runner] Shutdown complete")


def main():
    runner = BotRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted by user")
    except Exception as e:
        print(f"\n[Runner] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
