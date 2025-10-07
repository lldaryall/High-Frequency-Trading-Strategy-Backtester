"""
Backtest reporter for generating comprehensive results and artifacts.

This module handles the creation of all backtest output files including
performance metrics, trade data, position history, and visualizations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

from .plots import PlotGenerator
from ..metrics.performance import PerformanceAnalyzer
from ..config.config import BacktestConfig


class BacktestReporter:
    """Generates comprehensive backtest reports and artifacts."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize the reporter."""
        self.config = config
        self.output_dir = Path(config.report.output_dir)
        self.plot_generator = PlotGenerator()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, 
                       trades: List[Dict[str, Any]], 
                       positions: List[Dict[str, Any]], 
                       blotter: pd.DataFrame,
                       performance_metrics: Dict[str, Any]) -> None:
        """Generate complete backtest report with all artifacts.
        
        Args:
            trades: List of trade records
            positions: List of position records  
            blotter: Order blotter DataFrame
            performance_metrics: Performance metrics dictionary
        """
        print(f"📊 Generating backtest report in {self.output_dir}")
        
        # Save configuration
        self._save_config()
        
        # Save performance metrics
        self._save_performance_metrics(performance_metrics)
        
        # Save trade data
        self._save_trades(trades)
        
        # Save position data
        self._save_positions(positions)
        
        # Save blotter data
        self._save_blotter(blotter)
        
        # Generate plots
        if self.config.report.plots:
            self._generate_plots(trades, positions, performance_metrics)
        
        print(f"✅ Report generated successfully in {self.output_dir}")
    
    def _save_config(self) -> None:
        """Save the backtest configuration."""
        config_file = self.output_dir / "config.yaml"
        
        # Convert config to dictionary
        config_dict = {
            "name": self.config.name,
            "description": self.config.description,
            "data": {
                "path": self.config.data.path,
                "kind": self.config.data.kind,
                "symbol": self.config.data.symbol
            },
            "strategy": {
                "name": self.config.strategy.name,
                "symbol": self.config.strategy.symbol,
                "enabled": self.config.strategy.enabled,
                "max_position": self.config.strategy.max_position,
                "max_order_size": self.config.strategy.max_order_size,
                "params": self.config.strategy.params
            },
            "execution": {
                "fees": self.config.execution.fees,
                "latency": self.config.execution.latency
            },
            "risk": {
                "max_gross": self.config.risk.max_gross,
                "max_pos_per_symbol": self.config.risk.max_pos_per_symbol,
                "max_daily_loss": self.config.risk.max_daily_loss
            },
            "report": {
                "output_dir": self.config.report.output_dir,
                "format": self.config.report.format,
                "plots": self.config.report.plots,
                "detailed_trades": self.config.report.detailed_trades,
                "performance_metrics": self.config.report.performance_metrics
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _save_performance_metrics(self, metrics: Any) -> None:
        """Save performance metrics in JSON and CSV formats."""
        # Convert metrics to dict if it's an object
        if hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        elif hasattr(metrics, '__dict__'):
            metrics_dict = metrics.__dict__
        else:
            metrics_dict = metrics
        
        # Save JSON
        json_file = self.output_dir / "performance.json"
        with open(json_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Save CSV
        if self.config.report.format in ["csv", "both"]:
            csv_file = self.output_dir / "performance.csv"
            metrics_df = pd.DataFrame([metrics_dict])
            metrics_df.to_csv(csv_file, index=False)
    
    def _save_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Save trade data."""
        if not trades:
            # Create empty DataFrame with expected columns
            trades_df = pd.DataFrame(columns=[
                'trade_id', 'symbol', 'side', 'quantity', 'entry_price', 
                'exit_price', 'entry_time', 'exit_time', 'duration_ns', 
                'pnl', 'commission', 'slippage', 'strategy_id'
            ])
        else:
            trades_df = pd.DataFrame(trades)
        
        # Save CSV
        trades_file = self.output_dir / "trades.csv"
        trades_df.to_csv(trades_file, index=False)
        
        # Save Parquet if detailed trades requested
        if self.config.report.detailed_trades:
            trades_parquet = self.output_dir / "trades.parquet"
            trades_df.to_parquet(trades_parquet, index=False)
    
    def _save_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Save position data."""
        if not positions:
            # Create empty DataFrame with expected columns
            positions_df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'position', 'price', 'unrealized_pnl', 
                'realized_pnl', 'strategy_id'
            ])
        else:
            positions_df = pd.DataFrame(positions)
        
        # Save CSV
        positions_file = self.output_dir / "positions.csv"
        positions_df.to_csv(positions_file, index=False)
        
        # Save Parquet
        positions_parquet = self.output_dir / "positions.parquet"
        positions_df.to_parquet(positions_parquet, index=False)
    
    def _save_blotter(self, blotter: pd.DataFrame) -> None:
        """Save order blotter data."""
        if blotter.empty:
            # Create empty DataFrame with expected columns
            blotter_df = pd.DataFrame(columns=[
                'order_id', 'strategy_id', 'symbol', 'side', 'order_type',
                'quantity', 'price', 'status', 'submitted_at', 'filled_at',
                'cancelled_at', 'total_filled_qty', 'avg_fill_price'
            ])
        else:
            blotter_df = blotter.copy()
        
        # Save Parquet
        blotter_file = self.output_dir / "blotter.parquet"
        blotter_df.to_parquet(blotter_file, index=False)
        
        # Save CSV if requested
        if self.config.report.format in ["csv", "both"]:
            blotter_csv = self.output_dir / "blotter.csv"
            blotter_df.to_csv(blotter_csv, index=False)
    
    def _generate_plots(self, 
                       trades: List[Dict[str, Any]], 
                       positions: List[Dict[str, Any]], 
                       performance_metrics: Dict[str, Any]) -> None:
        """Generate visualization plots."""
        print("📈 Generating plots...")
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        positions_df = pd.DataFrame(positions) if positions else pd.DataFrame()
        
        # Generate equity curve
        if not positions_df.empty:
            equity_plot = self.output_dir / "equity_curve.png"
            self.plot_generator.plot_equity_curve(positions_df, equity_plot)
        
        # Generate drawdown curve
        if not positions_df.empty:
            drawdown_plot = self.output_dir / "drawdown_curve.png"
            self.plot_generator.plot_drawdown_curve(positions_df, drawdown_plot)
        
        # Generate trade PnL histogram
        if not trades_df.empty:
            pnl_hist_plot = self.output_dir / "trade_pnl_histogram.png"
            self.plot_generator.plot_trade_pnl_histogram(trades_df, pnl_hist_plot)
        
        # Generate performance summary
        summary_plot = self.output_dir / "performance_summary.png"
        self.plot_generator.plot_performance_summary(performance_metrics, summary_plot)
    
    def save_latency_sweep_results(self, sweep_results: List[Dict[str, Any]]) -> None:
        """Save latency sensitivity analysis results."""
        if not sweep_results:
            return
        
        # Save as CSV
        sweep_df = pd.DataFrame(sweep_results)
        sweep_file = self.output_dir / "latency_sweep.csv"
        sweep_df.to_csv(sweep_file, index=False)
        
        # Save as Parquet
        sweep_parquet = self.output_dir / "latency_sweep.parquet"
        sweep_df.to_parquet(sweep_parquet, index=False)
        
        # Generate latency sensitivity plot
        if self.config.report.plots:
            latency_plot = self.output_dir / "latency_sensitivity.png"
            self.plot_generator.plot_latency_sensitivity(sweep_df, latency_plot)
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get a summary of the generated report."""
        summary = {
            "output_dir": str(self.output_dir),
            "files": [],
            "plots": []
        }
        
        # List all files
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                file_info = {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                summary["files"].append(file_info)
        
        # List plots
        plot_extensions = ['.png', '.jpg', '.jpeg', '.svg']
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in plot_extensions:
                summary["plots"].append(file_path.name)
        
        return summary
