"""Equity Curve dashboard page."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from src.persistence.database import get_session, init_db
from config.settings import get_settings


def render():
    """Render the equity curve page."""
    st.title("📉 Equity Curve")

    try:
        from src.dashboard.data_service import DashboardDataService

        settings = get_settings()
        init_db(settings.database_url)

        initial_capital_rupees = settings.capital / 100

        days = st.slider("Days to display", min_value=7, max_value=365, value=30)

        with get_session() as session:
            service = DashboardDataService(session)
            daily_df = service.get_daily_pnl_summary(days=days)

        if daily_df is not None and not daily_df.empty:
            # Compute cumulative equity
            daily_df["cumulative_pnl"] = daily_df["total_pnl"].cumsum()
            daily_df["equity"] = initial_capital_rupees + daily_df["cumulative_pnl"]

            # Drawdown calculation
            daily_df["peak"] = daily_df["equity"].cummax()
            daily_df["drawdown"] = daily_df["equity"] - daily_df["peak"]
            daily_df["drawdown_pct"] = (
                daily_df["drawdown"] / daily_df["peak"]
            ) * 100

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Starting Capital", f"Rs. {initial_capital_rupees:,.2f}")
            col2.metric(
                "Current Equity",
                f"Rs. {daily_df['equity'].iloc[-1]:,.2f}",
            )
            total_return = daily_df["cumulative_pnl"].iloc[-1]
            return_pct = total_return / initial_capital_rupees * 100
            col3.metric("Total Return", f"{return_pct:.2f}%")
            col4.metric("Max Drawdown", f"{daily_df['drawdown_pct'].min():.2f}%")

            st.markdown("---")

            # Equity curve chart
            st.subheader("Equity Over Time (Rs.)")
            st.line_chart(daily_df["equity"])

            # Drawdown chart
            st.subheader("Drawdown (%)")
            st.area_chart(daily_df["drawdown_pct"])

            # Daily P&L bars
            st.subheader("Daily P&L (Rs.)")
            st.bar_chart(daily_df["total_pnl"])

            # Stats table
            st.subheader("Daily Breakdown")
            display_df = daily_df[
                ["total_pnl", "realized_pnl", "trades_count", "win_count"]
            ].copy()
            display_df.columns = [
                "Total P&L (Rs.)",
                "Realized (Rs.)",
                "Trades",
                "Wins",
            ]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info(
                "No daily P&L data yet. The bot records daily summaries after market close."
            )

    except Exception as e:
        st.error(f"Error loading equity curve: {e}")
        import traceback
        st.code(traceback.format_exc())


# Allow Streamlit multipage auto-discovery to run this page directly
render()
