"""
Plot generation for backtest results visualization.

This module provides comprehensive plotting capabilities for backtest results
including equity curves, drawdown analysis, trade distributions, and performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PlotGenerator:
    """Generates visualization plots for backtest results."""
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 300):
        """Initialize the plot generator."""
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_equity_curve(self, positions_df: pd.DataFrame, output_path: Path) -> None:
        """Plot equity curve over time."""
        if positions_df.empty:
            self._create_empty_plot("Equity Curve", output_path)
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in positions_df.columns:
            positions_df = positions_df.copy()
            if positions_df['timestamp'].dtype == 'int64':
                positions_df['datetime'] = pd.to_datetime(positions_df['timestamp'], unit='ns')
            else:
                positions_df['datetime'] = pd.to_datetime(positions_df['timestamp'])
        else:
            self._create_empty_plot("Equity Curve", output_path)
            return
        
        # Calculate cumulative PnL
        if 'realized_pnl' in positions_df.columns:
            positions_df['cumulative_pnl'] = positions_df['realized_pnl'].cumsum()
        else:
            positions_df['cumulative_pnl'] = 0
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.plot(positions_df['datetime'], positions_df['cumulative_pnl'], 
                linewidth=2, color='blue', alpha=0.8)
        
        ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def plot_drawdown_curve(self, positions_df: pd.DataFrame, output_path: Path) -> None:
        """Plot drawdown curve over time."""
        if positions_df.empty:
            self._create_empty_plot("Drawdown Curve", output_path)
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in positions_df.columns:
            positions_df = positions_df.copy()
            if positions_df['timestamp'].dtype == 'int64':
                positions_df['datetime'] = pd.to_datetime(positions_df['timestamp'], unit='ns')
            else:
                positions_df['datetime'] = pd.to_datetime(positions_df['timestamp'])
        else:
            self._create_empty_plot("Drawdown Curve", output_path)
            return
        
        # Calculate cumulative PnL and running maximum
        if 'realized_pnl' in positions_df.columns:
            positions_df['cumulative_pnl'] = positions_df['realized_pnl'].cumsum()
        else:
            positions_df['cumulative_pnl'] = 0
        
        positions_df['running_max'] = positions_df['cumulative_pnl'].expanding().max()
        positions_df['drawdown'] = positions_df['cumulative_pnl'] - positions_df['running_max']
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Fill area under drawdown curve
        ax.fill_between(positions_df['datetime'], positions_df['drawdown'], 0, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(positions_df['datetime'], positions_df['drawdown'], 
                linewidth=2, color='red', alpha=0.8)
        
        ax.set_title('Drawdown Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Drawdown ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def plot_trade_pnl_histogram(self, trades_df: pd.DataFrame, output_path: Path) -> None:
        """Plot trade PnL distribution histogram."""
        if trades_df.empty or 'pnl' not in trades_df.columns:
            self._create_empty_plot("Trade PnL Distribution", output_path)
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Histogram
        ax1.hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax1.axvline(trades_df['pnl'].mean(), color='green', linestyle='-', alpha=0.7, 
                   label=f'Mean: ${trades_df["pnl"].mean():.2f}')
        
        ax1.set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PnL ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(trades_df['pnl'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_title('Trade PnL Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('PnL ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: ${trades_df['pnl'].mean():.2f}
Median: ${trades_df['pnl'].median():.2f}
Std: ${trades_df['pnl'].std():.2f}
Min: ${trades_df['pnl'].min():.2f}
Max: ${trades_df['pnl'].max():.2f}
Winning Trades: {(trades_df['pnl'] > 0).sum()}
Losing Trades: {(trades_df['pnl'] < 0).sum()}"""
        
        ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def plot_performance_summary(self, metrics: Any, output_path: Path) -> None:
        """Plot performance metrics summary."""
        # Convert metrics to dict if it's an object
        if hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        elif hasattr(metrics, '__dict__'):
            metrics_dict = metrics.__dict__
        else:
            metrics_dict = metrics
        
        # Create plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # Key metrics
        key_metrics = [
            ('Total Return', metrics_dict.get('total_return', 0) * 100, '%'),
            ('Sharpe Ratio', metrics_dict.get('sharpe_ratio', 0), ''),
            ('Max Drawdown', abs(metrics_dict.get('max_drawdown', 0)) * 100, '%'),
            ('Total Trades', metrics_dict.get('total_trades', 0), ''),
            ('Hit Rate', metrics_dict.get('hit_rate', 0) * 100, '%'),
            ('Turnover', metrics_dict.get('turnover', 0), ''),
        ]
        
        # Plot key metrics as bar chart
        names, values, units = zip(*key_metrics)
        colors = ['green' if v > 0 else 'red' if v < 0 else 'blue' for v in values]
        
        bars = ax1.bar(range(len(names)), values, color=colors, alpha=0.7)
        ax1.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, unit in zip(bars, values, units):
            height = bar.get_height()
            label = f'{value:.2f}{unit}' if unit else f'{value:.2f}'
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=10)
        
        # Risk metrics
        risk_metrics = {
            'Volatility': metrics_dict.get('volatility', 0) * 100,
            'Skewness': metrics_dict.get('skewness', 0),
            'Kurtosis': metrics_dict.get('kurtosis', 0),
            'VaR (95%)': abs(metrics_dict.get('var_95', 0)) * 100,
        }
        
        # Filter out NaN values and ensure we have valid data
        valid_risk_metrics = {k: v for k, v in risk_metrics.items() if not np.isnan(v) and v != 0}
        if valid_risk_metrics:
            ax2.pie([abs(v) for v in valid_risk_metrics.values()], 
                   labels=valid_risk_metrics.keys(), autopct='%1.1f%%', startangle=90)
        else:
            ax2.text(0.5, 0.5, 'No risk data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Risk Metrics Distribution', fontsize=14, fontweight='bold')
        
        # Trade statistics
        trade_stats = {
            'Winning': metrics_dict.get('winning_trades', 0),
            'Losing': metrics_dict.get('losing_trades', 0),
            'Breakeven': metrics_dict.get('breakeven_trades', 0),
        }
        
        # Filter out zero values and ensure we have valid data
        valid_trade_stats = {k: v for k, v in trade_stats.items() if v > 0}
        if valid_trade_stats:
            ax3.pie(valid_trade_stats.values(), labels=valid_trade_stats.keys(), autopct='%1.1f%%', startangle=90)
        else:
            ax3.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Trade Outcomes', fontsize=14, fontweight='bold')
        
        # Performance comparison (if available)
        benchmark_return = 0.05  # 5% benchmark
        strategy_return = metrics_dict.get('total_return', 0)
        
        ax4.bar(['Benchmark', 'Strategy'], [benchmark_return * 100, strategy_return * 100],
               color=['gray', 'blue'], alpha=0.7)
        ax4.set_title('Strategy vs Benchmark', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def plot_latency_sensitivity(self, sweep_df: pd.DataFrame, output_path: Path) -> None:
        """Plot latency sensitivity analysis."""
        if sweep_df.empty:
            self._create_empty_plot("Latency Sensitivity", output_path)
            return
        
        # Create plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # Convert latency to microseconds
        sweep_df = sweep_df.copy()
        sweep_df['latency_us'] = sweep_df['latency_ns'] / 1000
        
        # Total return vs latency
        ax1.plot(sweep_df['latency_us'], sweep_df['total_return'] * 100, 
                'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Total Return vs Latency', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Latency (μs)', fontsize=12)
        ax1.set_ylabel('Total Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Sharpe ratio vs latency
        ax2.plot(sweep_df['latency_us'], sweep_df['sharpe_ratio'], 
                'o-', linewidth=2, markersize=8, color='green')
        ax2.set_title('Sharpe Ratio vs Latency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Latency (μs)', fontsize=12)
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Max drawdown vs latency
        ax3.plot(sweep_df['latency_us'], abs(sweep_df['max_drawdown']) * 100, 
                'o-', linewidth=2, markersize=8, color='red')
        ax3.set_title('Max Drawdown vs Latency', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Latency (μs)', fontsize=12)
        ax3.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Total trades vs latency
        ax4.plot(sweep_df['latency_us'], sweep_df['total_trades'], 
                'o-', linewidth=2, markersize=8, color='orange')
        ax4.set_title('Total Trades vs Latency', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Latency (μs)', fontsize=12)
        ax4.set_ylabel('Total Trades', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def _create_empty_plot(self, title: str, output_path: Path) -> None:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.text(0.5, 0.5, f'No data available for {title}', 
               ha='center', va='center', fontsize=16, 
               transform=ax.transAxes)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
