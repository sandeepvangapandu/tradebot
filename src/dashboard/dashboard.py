"""Main Streamlit dashboard entry point.

Run with: streamlit run src/dashboard/dashboard.py

Streamlit auto-discovers pages in the pages/ subdirectory.
Each page file calls its own render() at the bottom.
This file is the landing/home page.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.persistence.database import init_db
from config.settings import get_settings

# Initialize database on startup
settings = get_settings()
init_db(settings.database_url)

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📊 Trading Bot")
mode = settings.trading_mode.upper()
if mode == "PAPER":
    st.sidebar.success(f"Mode: {mode}")
else:
    st.sidebar.error(f"Mode: {mode}")
st.sidebar.markdown("---")
st.sidebar.caption("Trading Bot v1.0")

# --- Landing page content ---
st.title("📈 Trading Bot Dashboard")
st.markdown(
    """
Welcome to the **Trading Bot Dashboard**. Use the sidebar to navigate:

| Page | Description |
|------|-------------|
| **📡 Live Feed** | Real-time Upstox quotes for the watchlist (default view) |
| **Overview** | Live P&L, open positions, today's trades |
| **Trade History** | Historical trades with filters and CSV export |
| **Strategy Perf** | Strategy comparison — P&L, win rate, profit factor |
| **Equity Curve** | Equity over time, drawdown, daily P&L bars |
| **Risk Monitor** | Risk usage, capital deployment, daily loss tracking |
| **Backtester** | Run strategies against uploaded CSV data |

> 👉 Open **Live Feed** from the sidebar to watch live Upstox prices.
"""
)

st.markdown("---")

# Quick stats on landing page
try:
    from src.dashboard.data_service import DashboardDataService
    from src.persistence.database import get_session

    with get_session() as session:
        svc = DashboardDataService(session)
        total_pnl = svc.get_total_pnl()
        positions = svc.get_current_positions()

    col1, col2, col3 = st.columns(3)
    col1.metric("Capital", f"Rs. {settings.capital / 100:,.2f}")
    col2.metric("Total P&L", f"Rs. {total_pnl:,.2f}")
    col3.metric("Open Positions", len(positions))
except Exception:
    pass
