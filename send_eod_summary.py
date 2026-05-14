#!/usr/bin/env python3
"""EOD Summary Generator and Sender.

Reads today's trades + P&L from the bot's primary SQLite database
(path from settings.database_url) and prints a summary. Run at 15:35
IST or later so the bot's 15:30 daily_summary scheduled job has
written the daily_pnl row.

Called by Zo Computer scheduled agent at 15:35 IST on trading days.
"""

import os
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _resolve_db_path() -> Path:
    """Resolve the SQLite path the running bot uses.

    Prefers DATABASE_URL env (sqlite:///...) then falls back to the
    canonical project path. Raises if the file is missing.
    """
    url = os.environ.get("DATABASE_URL", "sqlite:///data/trading_bot.db")
    if url.startswith("sqlite:///"):
        rel = url[len("sqlite:///"):]
        path = (Path(__file__).parent / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    else:
        path = Path(__file__).parent / "data" / "trading_bot.db"
    return path


def get_eod_summary() -> str:
    db_path = _resolve_db_path()
    if not db_path.exists():
        return f"No trading database at {db_path}. Bot may not have run today."

    now = datetime.now(IST)
    today = now.date().isoformat()

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Real numbers from trades table — closed trades only
        cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(realized_pnl), 0), "
            "       COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0), "
            "       COALESCE(SUM(fees), 0) "
            "FROM trades WHERE date(exit_time) = ?",
            (today,),
        )
        n_trades, realized_paisa, wins, fees_paisa = cur.fetchone()

        # daily_pnl row (written by bot's 15:30 scheduler) — has unrealized too
        cur.execute(
            "SELECT realized_pnl, unrealized_pnl, trades_count, win_count "
            "FROM daily_pnl WHERE date = ?",
            (today,),
        )
        row = cur.fetchone()
        if row:
            dpnl_realized, dpnl_unrealized, dpnl_trades, dpnl_wins = row
        else:
            dpnl_realized = dpnl_unrealized = dpnl_trades = dpnl_wins = None

        # Open positions
        cur.execute(
            "SELECT strategy, side, instrument_key, entry_price, quantity "
            "FROM positions WHERE status = 'open' AND date(opened_at) = ?",
            (today,),
        )
        open_rows = cur.fetchall()

        # Per-strategy breakdown
        cur.execute(
            "SELECT strategy, COUNT(*), COALESCE(SUM(realized_pnl), 0) "
            "FROM trades WHERE date(exit_time) = ? GROUP BY strategy",
            (today,),
        )
        per_strat = cur.fetchall()

        conn.close()
    except Exception as exc:
        return f"Error reading database {db_path}: {exc}"

    realized_str = f"₹{(realized_paisa or 0) / 100:,.2f}"
    unrealized_str = (
        f"₹{(dpnl_unrealized or 0) / 100:,.2f}" if dpnl_unrealized is not None else "n/a (daily_summary not run yet)"
    )
    fees_str = f"₹{(fees_paisa or 0) / 100:,.2f}"
    win_rate = (wins / n_trades * 100) if n_trades else 0.0

    lines = [
        "📊 Trading Bot EOD Summary",
        "=" * 42,
        f"Date: {today}",
        f"Time: {now.strftime('%H:%M:%S IST')}",
        "",
        f"📈 Closed trades:  {n_trades}",
        f"   Wins / Losses:  {wins} / {n_trades - wins} ({win_rate:.1f}% win rate)",
        f"💰 Realized P&L:   {realized_str}",
        f"📊 Unrealized P&L: {unrealized_str}",
        f"💸 Fees:           {fees_str}",
        "",
    ]
    if per_strat:
        lines.append("Per-strategy realized P&L:")
        for strat, n, pnl in per_strat:
            lines.append(f"  {strat:30s} n={n:3d}  pnl=₹{pnl / 100:>12,.2f}")
        lines.append("")
    if open_rows:
        lines.append(f"Open positions ({len(open_rows)}):")
        for strat, side, ikey, px, qty in open_rows:
            lines.append(f"  {strat:30s} {side} {qty} {ikey} @ ₹{px / 100:.2f}")
        lines.append("")
    lines.append(f"DB: {db_path}")
    return "\n".join(lines)


def main() -> int:
    summary = get_eod_summary()
    print(summary)
    out = Path(__file__).parent / "logs" / "eod_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary)
    print(f"\nSummary saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
