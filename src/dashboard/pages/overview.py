"""Live Overview dashboard page."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for standalone execution
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.persistence.database import get_session, init_db
from config.settings import get_settings

IST = ZoneInfo("Asia/Kolkata")


def render():
    """Render the live overview page."""
    st.title("🏠 Live Overview")
    now = datetime.now(IST)
    st.caption(f"Last updated: {now.strftime('%d %b %Y, %I:%M:%S %p IST')}")

    try:
        from src.dashboard.data_service import DashboardDataService

        settings = get_settings()
        init_db(settings.database_url)

        with get_session() as session:
            service = DashboardDataService(session)

            today_pnl = service.get_today_pnl()
            total_pnl = service.get_total_pnl()
            positions = service.get_current_positions()
            today_trades = service.get_today_trades()

        # --- Top metrics row ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Capital", f"Rs. {settings.capital / 100:,.2f}")
        col2.metric(
            "Today's P&L",
            f"Rs. {today_pnl:,.2f}",
            delta=f"Rs. {today_pnl:,.2f}",
        )
        col3.metric("Total P&L", f"Rs. {total_pnl:,.2f}")
        col4.metric("Open Positions", len(positions))

        st.markdown("---")

        # --- Today's Stats ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Trades Today", len(today_trades))
        wins = sum(1 for t in today_trades if t.realized_pnl > 0)
        col2.metric("Wins", wins)
        win_rate = (wins / len(today_trades) * 100) if today_trades else 0
        col3.metric("Win Rate", f"{win_rate:.0f}%")

        st.markdown("---")

        # --- Open Positions ---
        st.subheader("📊 Open Positions")
        if positions:
            pos_data = []
            for p in positions:
                pos_data.append(
                    {
                        "Strategy": p.strategy,
                        "Instrument": p.instrument_key,
                        "Side": p.side,
                        "Entry Price (Rs.)": p.entry_price,
                        "Quantity": p.quantity,
                        "Unrealized P&L (Rs.)": p.unrealized_pnl,
                        "Opened At": p.opened_at,
                    }
                )
            st.dataframe(pd.DataFrame(pos_data), width='stretch')
        else:
            st.info("No open positions")

        # --- Recent Trades ---
        st.subheader("📜 Today's Trades")
        if today_trades:
            trade_data = []
            for t in today_trades:
                trade_data.append(
                    {
                        "Strategy": t.strategy,
                        "Instrument": t.instrument_key,
                        "Side": t.side,
                        "Entry (Rs.)": t.entry_price,
                        "Exit (Rs.)": t.exit_price,
                        "Qty": t.quantity,
                        "P&L (Rs.)": t.realized_pnl,
                        "Fees (Rs.)": t.fees,
                        "Duration (min)": round(t.duration_minutes, 1),
                    }
                )
            st.dataframe(pd.DataFrame(trade_data), width='stretch')
        else:
            st.info("No trades today")

    except Exception as e:
        st.error(f"Error loading overview: {e}")
        import traceback
        st.code(traceback.format_exc())


# Auto-run when Streamlit executes this page
render()
