"""Trade History dashboard page."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.persistence.database import get_session, init_db
from config.settings import get_settings


def render():
    """Render the trade history page."""
    st.title("📜 Trade History")

    try:
        from src.dashboard.data_service import DashboardDataService

        # Ensure DB is initialized
        settings = get_settings()
        init_db(settings.database_url)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input(
                "Start Date", value=date.today() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input("End Date", value=date.today())
        with col3:
            strategy_filter = st.text_input(
                "Strategy Filter", placeholder="e.g., ema_crossover"
            )

        with get_session() as session:
            service = DashboardDataService(session)
            df = service.get_trade_history(
                start_date=start_date,
                end_date=end_date,
                strategy=strategy_filter if strategy_filter else None,
            )

        if df is not None and not df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            total_pnl = df["realized_pnl"].sum()
            col1.metric("Total P&L", f"Rs. {total_pnl:,.2f}")
            col2.metric("Total Trades", len(df))
            wins = (df["realized_pnl"] > 0).sum()
            col3.metric("Win Rate", f"{wins / len(df) * 100:.1f}%")
            col4.metric("Total Fees", f"Rs. {df['fees'].sum():,.2f}")

            st.markdown("---")

            # Trade table
            st.dataframe(df, use_container_width=True)

            # P&L distribution chart
            st.subheader("P&L Distribution")
            chart_data = df[["exit_time", "realized_pnl"]].copy()
            chart_data = chart_data.set_index("exit_time")
            st.bar_chart(chart_data["realized_pnl"])

            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=f"trades_{start_date}_{end_date}.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades found for the selected date range.")

    except Exception as e:
        st.error(f"Error loading trade history: {e}")
        import traceback
        st.code(traceback.format_exc())


# Allow Streamlit multipage auto-discovery to run this page directly
render()
