"""Tests for backtest report generator."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.backtest.engine import BacktestEngine, BacktestResults, Trade
from src.backtest.report_generator import BacktestReportGenerator

IST = ZoneInfo("Asia/Kolkata")


class TestBacktestReportGenerator:
    """Test report generator functionality."""

    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results."""
        trades = [
            Trade(
                entry_time=pd.Timestamp("2025-01-02 09:15", tz=IST),
                exit_time=pd.Timestamp("2025-01-02 10:30", tz=IST),
                side="BUY",
                entry_price=50000_00,
                exit_price=50500_00,
                quantity=1,
                pnl=500_00,
                fees=50,
                net_pnl=499_50,
                exit_reason="target",
            ),
            Trade(
                entry_time=pd.Timestamp("2025-01-02 11:00", tz=IST),
                exit_time=pd.Timestamp("2025-01-02 12:00", tz=IST),
                side="SELL",
                entry_price=50500_00,
                exit_price=50200_00,
                quantity=1,
                pnl=-300_00,
                fees=50,
                net_pnl=-300_50,
                exit_reason="stop_loss",
            ),
        ]

        return BacktestResults(
            initial_capital=1_000_000_00,
            final_capital=1_000_200_00,
            trades=trades,
            equity_curve=[1_000_000_00, 1_000_500_00, 1_000_200_00],
            total_bars_processed=100,
        )

    def test_initialization(self, sample_results):
        """Test report generator initialization."""
        generator = BacktestReportGenerator(sample_results)
        assert generator.results == sample_results
        assert generator.metrics is not None

    def test_generate_summary_json(self, sample_results, tmp_path):
        """Test JSON summary generation."""
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "summary.json"

        summary = generator.generate_summary_json(output_path)

        assert output_path.exists()
        assert "metrics" in summary
        assert summary["trade_count"] == 2
        assert summary["initial_capital_rupees"] == 1_000_000.0

        # Verify file content
        with open(output_path) as f:
            loaded = json.load(f)
            assert "generated_at" in loaded

    def test_generate_csv_report(self, sample_results, tmp_path):
        """Test CSV report generation."""
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "trades.csv"

        generator.generate_csv_report(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "entry_time" in content
        assert "pnl_rupees" in content

    def test_generate_html_report(self, sample_results, tmp_path):
        """Test HTML report generation."""
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "report.html"

        generator.generate_html_report(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "Backtest Report" in content
        assert "Performance Metrics" in content
        assert "Trade List" in content

    def test_plot_equity_curve(self, sample_results, tmp_path):
        """Test equity curve plot generation."""
        from src.backtest.report_generator import PLOTLY_AVAILABLE
        try:
            import kaleido  # noqa: F401
            kaleido_available = True
        except ImportError:
            kaleido_available = False
        if not PLOTLY_AVAILABLE or not kaleido_available:
            pytest.skip("Plotly or Kaleido not installed")
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "equity.png"

        fig = generator.plot_equity_curve(output_path)

        assert output_path.exists()
        assert fig is not None

    def test_plot_drawdown(self, sample_results, tmp_path):
        """Test drawdown plot generation."""
        from src.backtest.report_generator import PLOTLY_AVAILABLE
        try:
            import kaleido  # noqa: F401
            kaleido_available = True
        except ImportError:
            kaleido_available = False
        if not PLOTLY_AVAILABLE or not kaleido_available:
            pytest.skip("Plotly or Kaleido not installed")
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "drawdown.png"

        fig = generator.plot_drawdown(output_path)

        assert output_path.exists()
        assert fig is not None

    def test_plot_trade_distribution(self, sample_results, tmp_path):
        """Test trade distribution plot generation."""
        from src.backtest.report_generator import PLOTLY_AVAILABLE
        try:
            import kaleido  # noqa: F401
            kaleido_available = True
        except ImportError:
            kaleido_available = False
        if not PLOTLY_AVAILABLE or not kaleido_available:
            pytest.skip("Plotly or Kaleido not installed")
        generator = BacktestReportGenerator(sample_results)
        output_path = tmp_path / "distribution.png"

        fig = generator.plot_trade_distribution(output_path)

        assert output_path.exists()
        assert fig is not None

    def test_generate_full_report(self, sample_results, tmp_path):
        """Test full report generation."""
        from src.backtest.report_generator import PLOTLY_AVAILABLE
        try:
            import kaleido  # noqa: F401
            kaleido_available = True
        except ImportError:
            kaleido_available = False
        if not PLOTLY_AVAILABLE or not kaleido_available:
            pytest.skip("Plotly or Kaleido not installed")
        generator = BacktestReportGenerator(sample_results)
        output_dir = tmp_path / "report"

        paths = generator.generate_full_report(output_dir)

        assert "html" in paths
        assert "csv" in paths
        assert "json" in paths
        assert "equity_chart" in paths
        assert "drawdown_chart" in paths
        assert "distribution_chart" in paths

        assert all(p.exists() for p in paths.values())

    def test_empty_trades_csv(self, tmp_path):
        """Test CSV generation with no trades."""
        results = BacktestResults(
            initial_capital=1_000_000_00,
            final_capital=1_000_000_00,
            trades=[],
            equity_curve=[1_000_000_00],
            total_bars_processed=10,
        )
        generator = BacktestReportGenerator(results)
        output_path = tmp_path / "empty.csv"

        generator.generate_csv_report(output_path)
        # Should log warning but not crash
        assert not output_path.exists()  # No file created for empty trades

    def test_trades_table_html_empty(self, tmp_path):
        """Test HTML trades table with no trades."""
        results = BacktestResults(
            initial_capital=1_000_000_00,
            final_capital=1_000_000_00,
            trades=[],
            equity_curve=[1_000_000_00],
            total_bars_processed=10,
        )
        generator = BacktestReportGenerator(results)
        output_path = tmp_path / "empty.html"

        generator.generate_html_report(output_path)
        content = output_path.read_text()
        assert "No trades executed" in content
