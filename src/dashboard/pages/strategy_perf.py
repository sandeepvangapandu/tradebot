"""Strategy Performance dashboard page."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.persistence.database import get_session, init_db
from config.settings import get_settings


def render():
    """Render the strategy performance page."""
    st.title("🎯 Strategy Performance")

    try:
        from src.dashboard.data_service import DashboardDataService

        settings = get_settings()
        init_db(settings.database_url)

        with get_session() as session:
            service = DashboardDataService(session)
            perf_list = service.get_strategy_performance()

        if perf_list:
            # Convert dataclass list to DataFrame
            records = [asdict(p) for p in perf_list]
            df = pd.DataFrame(records)

            # Summary table with formatted columns
            display_df = df.copy()
            display_df = display_df.rename(columns={
                "strategy_name": "Strategy",
                "total_trades": "Trades",
                "win_count": "Wins",
                "loss_count": "Losses",
                "win_rate": "Win Rate",
                "total_pnl": "P&L (Rs.)",
                "avg_win": "Avg Win (Rs.)",
                "avg_loss": "Avg Loss (Rs.)",
                "profit_factor": "Profit Factor",
            })
            display_df["Win Rate"] = display_df["Win Rate"].apply(
                lambda x: f"{x * 100:.1f}%"
            )

            st.subheader("📊 Strategy Comparison")
            st.dataframe(display_df, width='stretch')

            st.markdown("---")

            # P&L bar chart
            st.subheader("P&L by Strategy (Rs.)")
            chart_df = df.set_index("strategy_name")[["total_pnl"]]
            st.bar_chart(chart_df)

            # Win rate comparison
            st.subheader("Win Rate by Strategy (%)")
            wr_df = df.copy()
            wr_df["win_rate_pct"] = wr_df["win_rate"] * 100
            st.bar_chart(wr_df.set_index("strategy_name")[["win_rate_pct"]])

            # Trade count
            st.subheader("Trade Count by Strategy")
            st.bar_chart(df.set_index("strategy_name")[["total_trades"]])
        else:
            st.info("No strategy performance data available. Run the bot to generate trades.")

    except Exception as e:
        st.error(f"Error loading strategy performance: {e}")
        import traceback
        st.code(traceback.format_exc())


# Allow Streamlit multipage auto-discovery to run this page directly
render()
