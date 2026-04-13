"""Backtest report generator for detailed analysis and visualization."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
from loguru import logger

from src.backtest.engine import BacktestResults


class BacktestReportGenerator:
    """Generate detailed reports from backtest results.

    Supports HTML, CSV, JSON, and interactive Plotly charts.
    """

    def __init__(self, results: BacktestResults):
        """Initialize with backtest results.

        Args:
            results: BacktestResults instance to generate reports from.
        """
        self.results = results
        self.metrics = results.get_metrics()

    def generate_html_report(self, output_path: str | Path) -> None:
        """Generate a comprehensive HTML report.

        Args:
            output_path: Path to save the HTML file.
        """
        output_path = Path(output_path)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-container {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if self.metrics.get('total_pnl_rupees', 0) >= 0 else 'negative'}">
                    ₹{self.metrics.get('total_pnl_rupees', 0):,.2f}
                </div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('return_pct', 0):.2f}%</div>
                <div class="metric-label">Return %</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('win_rate', 0):.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('total_trades', 0)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{self.metrics.get('max_drawdown_pct', 0):.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('sharpe_ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">₹{self.metrics.get('avg_winner', 0):,.0f}</div>
                <div class="metric-label">Avg Winner</div>
            </div>
        </div>

        <h2>Trade List</h2>
        {self._generate_trades_table_html()}

        <h2>Charts</h2>
        <div class="chart-container">
            <p>See exported PNG charts or run interactive report for charts.</p>
        </div>
    </div>
</body>
</html>"""

        output_path.write_text(html_content, encoding='utf-8')
        logger.info(f"HTML report saved to {output_path}")

    def _generate_trades_table_html(self) -> str:
        """Generate HTML table for trades."""
        if not self.results.trades:
            return "<p>No trades executed</p>"

        rows = []
        for trade in self.results.trades:
            pnl_class = "positive" if trade.pnl >= 0 else "negative"
            rows.append(f"""
                <tr>
                    <td>{trade.entry_time}</td>
                    <td>{trade.exit_time}</td>
                    <td>{trade.side}</td>
                    <td>₹{trade.entry_price / 100:,.2f}</td>
                    <td>₹{trade.exit_price / 100:,.2f}</td>
                    <td>{trade.quantity}</td>
                    <td class="{pnl_class}">₹{trade.pnl / 100:,.2f}</td>
                    <td>{trade.exit_reason}</td>
                </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Side</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Quantity</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def generate_csv_report(self, output_path: str | Path) -> None:
        """Export trades to CSV.

        Args:
            output_path: Path to save the CSV file.
        """
        output_path = Path(output_path)

        if not self.results.trades:
            logger.warning("No trades to export")
            return

        df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'side': t.side,
                'entry_price_rupees': t.entry_price / 100,
                'exit_price_rupees': t.exit_price / 100,
                'quantity': t.quantity,
                'pnl_rupees': t.pnl / 100,
                'exit_reason': t.exit_reason,
            }
            for t in self.results.trades
        ])

        df.to_csv(output_path, index=False)
        logger.info(f"CSV report saved to {output_path}")

    def generate_summary_json(self, output_path: str | Path | None = None) -> dict:
        """Generate JSON summary.

        Args:
            output_path: Optional path to save JSON file.

        Returns:
            Dictionary with summary data.
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'metrics': self.metrics,
            'trade_count': len(self.results.trades),
            'initial_capital_rupees': self.results.initial_capital / 100,
            'final_capital_rupees': self.results.final_capital / 100,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json.dumps(summary, indent=2, default=str))
            logger.info(f"JSON summary saved to {output_path}")

        return summary

    def plot_equity_curve(self, output_path: str | Path | None = None) -> Any:
        """Generate equity curve chart.

        Args:
            output_path: Optional path to save PNG.

        Returns:
            Plotly figure object.
        """
        equity_rupees = [e / 100 for e in self.results.equity_curve]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_rupees))),
            y=equity_rupees,
            mode='lines',
            name='Equity',
            line=dict(color='#4CAF50', width=2),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)',
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Bar',
            yaxis_title='Equity (₹)',
            hovermode='x unified',
            template='plotly_white',
        )

        if output_path:
            fig.write_image(str(output_path))
            logger.info(f"Equity curve saved to {output_path}")

        return fig

    def plot_drawdown(self, output_path: str | Path | None = None) -> Any:
        """Generate drawdown chart.

        Args:
            output_path: Optional path to save PNG.

        Returns:
            Plotly figure object.
        """
        # Calculate drawdown series
        equity = self.results.equity_curve
        peak = equity[0]
        drawdowns = []

        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(drawdowns))),
            y=drawdowns,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#f44336', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.2)',
        ))

        fig.update_layout(
            title='Drawdown Chart',
            xaxis_title='Bar',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
        )

        if output_path:
            fig.write_image(str(output_path))
            logger.info(f"Drawdown chart saved to {output_path}")

        return fig

    def plot_trade_distribution(self, output_path: str | Path | None = None) -> Any:
        """Generate P&L distribution histogram.

        Args:
            output_path: Optional path to save PNG.

        Returns:
            Plotly figure object.
        """
        if not self.results.trades:
            logger.warning("No trades for distribution plot")
            return go.Figure()

        pnls = [t.pnl / 100 for t in self.results.trades]

        colors = ['green' if p >= 0 else 'red' for p in pnls]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(pnls))),
            y=pnls,
            marker_color=colors,
            name='P&L',
        ))

        fig.update_layout(
            title='Trade P&L Distribution',
            xaxis_title='Trade #',
            yaxis_title='P&L (₹)',
            template='plotly_white',
        )

        if output_path:
            fig.write_image(str(output_path))
            logger.info(f"Trade distribution saved to {output_path}")

        return fig

    def generate_full_report(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate all report formats in a directory.

        Args:
            output_dir: Directory to save all reports.

        Returns:
            Dictionary mapping report type to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paths = {}

        # HTML report
        html_path = output_dir / f"backtest_report_{timestamp}.html"
        self.generate_html_report(html_path)
        paths['html'] = html_path

        # CSV report
        csv_path = output_dir / f"trades_{timestamp}.csv"
        self.generate_csv_report(csv_path)
        paths['csv'] = csv_path

        # JSON summary
        json_path = output_dir / f"summary_{timestamp}.json"
        self.generate_summary_json(json_path)
        paths['json'] = json_path

        # Charts
        equity_path = output_dir / f"equity_{timestamp}.png"
        self.plot_equity_curve(equity_path)
        paths['equity_chart'] = equity_path

        drawdown_path = output_dir / f"drawdown_{timestamp}.png"
        self.plot_drawdown(drawdown_path)
        paths['drawdown_chart'] = drawdown_path

        if self.results.trades:
            dist_path = output_dir / f"distribution_{timestamp}.png"
            self.plot_trade_distribution(dist_path)
            paths['distribution_chart'] = dist_path

        logger.info(f"Full report generated in {output_dir}")
        return paths
