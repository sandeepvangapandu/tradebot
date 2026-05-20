#!/usr/bin/env python3
"""
Send a daily P&L summary email after market close.

Usage:
    python3 scripts/daily_email_report.py

Env vars required:
    REPORT_EMAIL_FROM     sender Gmail address
    REPORT_EMAIL_PASSWORD Gmail App Password (not account password)
    REPORT_EMAIL_TO       recipient address (can be same as FROM)

Gmail setup: Google Account → Security → 2-Step Verification ON
             → App Passwords → generate one for "Trading Bot"
"""

import os
import smtplib
import sqlite3
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path(__file__).parent.parent / "data" / "trading_bot.db"


def fetch_report(report_date: date) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Daily P&L row
    cur.execute("SELECT * FROM daily_pnl WHERE date = ?", (report_date.isoformat(),))
    pnl_row = cur.fetchone()

    # Today's trades
    cur.execute(
        """
        SELECT strategy, instrument_key, side, entry_price, exit_price,
               quantity, realized_pnl, entry_time, exit_time, holding_duration_seconds
        FROM trades
        WHERE date(entry_time) = ?
        ORDER BY entry_time
        """,
        (report_date.isoformat(),),
    )
    trades = [dict(r) for r in cur.fetchall()]

    # All-time stats
    cur.execute(
        "SELECT COUNT(*) as total, SUM(realized_pnl) as total_pnl FROM trades WHERE realized_pnl != 0"
    )
    all_time = dict(cur.fetchone())

    conn.close()
    return {"pnl_row": dict(pnl_row) if pnl_row else None, "trades": trades, "all_time": all_time}


def paise_to_inr(paise: int) -> str:
    sign = "-" if paise < 0 else "+"
    return f"{sign}₹{abs(paise) / 100:,.2f}"


def build_html(report_date: date, data: dict) -> str:
    pnl = data["pnl_row"]
    trades = data["trades"]
    all_time = data["all_time"]

    today_pnl = pnl["total_pnl"] if pnl else 0
    today_trades = len(trades)
    today_wins = sum(1 for t in trades if t["realized_pnl"] > 0)
    win_rate = f"{today_wins}/{today_trades}" if today_trades else "0/0"
    pnl_color = "#2e7d32" if today_pnl >= 0 else "#c62828"

    rows = ""
    for t in trades:
        pnl_v = t["realized_pnl"]
        color = "#2e7d32" if pnl_v >= 0 else "#c62828"
        duration = f"{t['holding_duration_seconds']}s" if t["holding_duration_seconds"] else "—"
        entry_t = t["entry_time"][:19] if t["entry_time"] else "—"
        rows += f"""
        <tr>
          <td>{t['strategy']}</td>
          <td>{t['instrument_key']}</td>
          <td>{t['side']}</td>
          <td>{t['quantity']}</td>
          <td>₹{t['entry_price']/100:,.2f}</td>
          <td>₹{t['exit_price']/100:,.2f}</td>
          <td style="color:{color};font-weight:bold">{paise_to_inr(pnl_v)}</td>
          <td>{duration}</td>
          <td>{entry_t}</td>
        </tr>"""

    no_trades_row = "" if trades else "<tr><td colspan='9' style='text-align:center;color:#888'>No trades today</td></tr>"
    all_time_pnl = all_time["total_pnl"] or 0
    all_time_color = "#2e7d32" if all_time_pnl >= 0 else "#c62828"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 14px; color: #333; }}
  h2 {{ color: #1a237e; }}
  .summary {{ background: #f5f5f5; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: flex; gap: 30px; flex-wrap: wrap; }}
  .stat {{ text-align: center; }}
  .stat .val {{ font-size: 24px; font-weight: bold; }}
  .stat .lbl {{ font-size: 12px; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ background: #1a237e; color: white; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f9f9f9; }}
  .footer {{ margin-top: 20px; font-size: 12px; color: #999; }}
</style>
</head>
<body>
<h2>Trading Bot — Daily Report ({report_date.strftime('%d %b %Y')})</h2>

<div class="summary">
  <div class="stat">
    <div class="val" style="color:{pnl_color}">{paise_to_inr(today_pnl)}</div>
    <div class="lbl">Today's P&L</div>
  </div>
  <div class="stat">
    <div class="val">{today_trades}</div>
    <div class="lbl">Trades</div>
  </div>
  <div class="stat">
    <div class="val">{win_rate}</div>
    <div class="lbl">Win/Total</div>
  </div>
  <div class="stat">
    <div class="val" style="color:{all_time_color}">{paise_to_inr(all_time_pnl)}</div>
    <div class="lbl">All-Time P&L</div>
  </div>
</div>

<h3>Trades</h3>
<table>
  <thead>
    <tr>
      <th>Strategy</th><th>Instrument</th><th>Side</th><th>Qty</th>
      <th>Entry</th><th>Exit</th><th>P&L</th><th>Duration</th><th>Time (IST)</th>
    </tr>
  </thead>
  <tbody>
    {rows}
    {no_trades_row}
  </tbody>
</table>

<div class="footer">
  Generated {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')} | Paper Trading Mode
</div>
</body>
</html>"""


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["REPORT_EMAIL_FROM"]
    password = os.environ["REPORT_EMAIL_PASSWORD"]
    recipient = os.environ["REPORT_EMAIL_TO"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    print(f"Report sent to {recipient}")


def main() -> None:
    report_date = date.today()
    data = fetch_report(report_date)
    html = build_html(report_date, data)

    pnl = data["pnl_row"]
    today_pnl_inr = (pnl["total_pnl"] / 100) if pnl else 0
    sign = "+" if today_pnl_inr >= 0 else ""
    subject = f"[TradingBot] {report_date.strftime('%d %b')} P&L: {sign}₹{today_pnl_inr:,.2f}"

    send_email(subject, html)


if __name__ == "__main__":
    main()
