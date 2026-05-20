#!/usr/bin/env python3
"""
Email the Upstox OAuth authorization URL so the user can complete login manually.
Run this before market open. After clicking the link and logging in, copy the
full redirect URL and run:
    python3 scripts/exchange_token.py "<paste full redirect URL here>"
"""

import os
import smtplib
import urllib.parse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

IST = ZoneInfo("Asia/Kolkata")


def build_auth_url() -> str:
    client_id = os.environ["UPSTOX_CLIENT_ID"]
    redirect_uri = os.environ["UPSTOX_REDIRECT_URI"]
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    })
    return f"https://api.upstox.com/v2/login/authorization/dialog?{params}"


def send_email(auth_url: str) -> None:
    sender = os.environ["REPORT_EMAIL_FROM"]
    password = os.environ["REPORT_EMAIL_PASSWORD"]
    recipient = os.environ["REPORT_EMAIL_TO"]

    now = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
    subject = f"[TradingBot] Upstox Login Required — {now}"

    html = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;font-size:14px;color:#333">
<h2 style="color:#1a237e">Upstox Token Refresh Required</h2>
<p>Click the button below to login to Upstox:</p>
<p>
  <a href="{auth_url}"
     style="background:#1a237e;color:white;padding:12px 24px;text-decoration:none;border-radius:6px;display:inline-block">
    Login to Upstox
  </a>
</p>
<p style="color:#666;font-size:12px">
  After login, your browser will redirect to a URL starting with your redirect URI.<br>
  That page may show an error — that's fine.<br><br>
  Copy the <b>full URL</b> from your browser's address bar, then run on the server:<br>
  <code style="background:#f5f5f5;padding:4px 8px;border-radius:4px">
    python3 /data/Trading/scripts/exchange_token.py "&lt;paste full URL here&gt;"
  </code>
</p>
<p style="color:#999;font-size:11px">Sent {now}</p>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

    print(f"Auth URL emailed to {recipient}")
    print(f"Auth URL: {auth_url}")


if __name__ == "__main__":
    url = build_auth_url()
    send_email(url)
