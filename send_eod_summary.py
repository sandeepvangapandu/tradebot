#!/usr/bin/env python3
"""EOD Summary Generator and Sender.

This script:
1. Reads the day's trading activity from the database
2. Generates a summary
3. Sends it via email (SMS not available, user can set up Telegram if preferred)

Called by scheduled agent at 15:30 IST on trading days.
"""

import sys
import sqlite3
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def get_eod_summary():
    """Generate EOD summary from database."""
    db_path = Path(__file__).parent / "data" / "trading_bot.db"
    
    if not db_path.exists():
        return "No trading database found. Bot may not have run today."
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        
        # Get today's closed trades (count by exit_time)
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE date(exit_time) = ?
            """, (today,))
            trade_count = cursor.fetchone()[0] or 0
        except Exception:
            trade_count = 0

        # Get today's P&L from daily_pnl table (one row per day)
        try:
            cursor.execute("""
                SELECT realized_pnl, unrealized_pnl
                FROM daily_pnl
                WHERE date = ?
            """, (today,))
            result = cursor.fetchone()
            realized_pnl = (result[0] if result else 0) or 0
            unrealized_pnl = (result[1] if result else 0) or 0
        except Exception:
            realized_pnl = 0
            unrealized_pnl = 0
        
        conn.close()
        
        # Format summary
        realized_str = f"₹{realized_pnl/100:,.2f}" if realized_pnl else "₹0.00"
        unrealized_str = f"₹{unrealized_pnl/100:,.2f}" if unrealized_pnl else "₹0.00"
        
        summary = f"""
📊 Trading Bot EOD Summary
{'='*40}
Date: {datetime.now(IST).strftime('%Y-%m-%d')}
Time: {datetime.now(IST).strftime('%H:%M:%S IST')}

📈 Trades: {trade_count}
💰 Realized P&L: {realized_str}
📊 Unrealized P&L: {unrealized_str}

---
Database: {db_path}
Logs: /home/workspace/Trading/logs/trading_bot.log
"""
        return summary
        
    except Exception as e:
        return f"Error generating summary: {e}\n\nCheck logs at /home/workspace/Trading/logs/trading_bot.log"


def main():
    """Main entry point."""
    summary = get_eod_summary()
    print(summary)
    
    # Save summary for reference
    summary_file = Path(__file__).parent / "logs" / "eod_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"\nSummary saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
