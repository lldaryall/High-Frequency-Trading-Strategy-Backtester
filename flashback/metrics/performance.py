"""
Performance metrics calculation and visualization for flashback HFT backtesting engine.

This module provides comprehensive performance analysis including returns, risk metrics,
trading statistics, and visualization capabilities.
"""

from __future__ import annotations

import json
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from ..risk.portfolio import PortfolioSnapshot, Position


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'BUY' or 'SELL'
    pnl: float
    fees: float
    duration_ns: int
    strategy_id: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in nanoseconds
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float
    
    # Portfolio metrics
    turnover: float
    avg_position_size: float
    max_position_size: float
    
    # Latency metrics
    mean_latency_ns: float
    p95_latency_ns: float
    p99_latency_ns: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    calmar_ratio: float
    
    # Additional metrics
    start_time: int
    end_time: int
    duration_days: float
    initial_capital: float
    final_capital: float
    total_fees: float
    total_slippage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "hit_rate": self.hit_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "profit_factor": self.profit_factor,
            "turnover": self.turnover,
            "avg_position_size": self.avg_position_size,
            "max_position_size": self.max_position_size,
            "mean_latency_ns": self.mean_latency_ns,
            "p95_latency_ns": self.p95_latency_ns,
            "p99_latency_ns": self.p99_latency_ns,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "calmar_ratio": self.calmar_ratio,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_days": self.duration_days,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage
        }


class PerformanceAnalyzer:
    """
    Performance analysis and metrics calculation for HFT backtesting.
    
    Calculates comprehensive performance metrics from portfolio snapshots,
    trade records, and latency data.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.snapshots: List[PortfolioSnapshot] = []
        self.trades: List[TradeRecord] = []
        self.latencies: List[float] = []  # in nanoseconds
        
    def add_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Add a portfolio snapshot for analysis."""
        self.snapshots.append(snapshot)
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a completed trade record."""
        self.trades.append(trade)
    
    def add_latency(self, latency_ns: float) -> None:
        """Add a latency measurement in nanoseconds."""
        self.latencies.append(latency_ns)
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if not self.snapshots:
            raise ValueError("No snapshots available for analysis")
        
        # Sort snapshots by timestamp
        self.snapshots.sort(key=lambda x: x.timestamp)
        
        # Basic portfolio data
        initial_capital = self.snapshots[0].cash + self.snapshots[0].total_market_value
        final_capital = self.snapshots[-1].cash + self.snapshots[-1].total_market_value
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Time calculations
        start_time = self.snapshots[0].timestamp
        end_time = self.snapshots[-1].timestamp
        duration_ns = end_time - start_time
        duration_days = duration_ns / (24 * 60 * 60 * 1e9)  # Convert ns to days
        
        # Annualized return
        if duration_days > 0 and duration_days < 365.25 * 10:  # Reasonable bounds
            try:
                annualized_return = (1 + total_return) ** (365.25 / duration_days) - 1
            except (OverflowError, ZeroDivisionError):
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Calculate returns series
        returns = self._calculate_returns_series()
        
        # Volatility (annualized)
        volatility = self._calculate_volatility(returns, duration_days)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
        
        # Drawdown analysis
        max_drawdown, max_drawdown_duration = self._calculate_drawdown()
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics()
        
        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        
        # Latency metrics
        latency_metrics = self._calculate_latency_metrics()
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return PerformanceMetrics(
            # Basic metrics
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            
            # Trading metrics
            total_trades=trading_metrics['total_trades'],
            winning_trades=trading_metrics['winning_trades'],
            losing_trades=trading_metrics['losing_trades'],
            hit_rate=trading_metrics['hit_rate'],
            avg_win=trading_metrics['avg_win'],
            avg_loss=trading_metrics['avg_loss'],
            win_loss_ratio=trading_metrics['win_loss_ratio'],
            profit_factor=trading_metrics['profit_factor'],
            
            # Portfolio metrics
            turnover=portfolio_metrics['turnover'],
            avg_position_size=portfolio_metrics['avg_position_size'],
            max_position_size=portfolio_metrics['max_position_size'],
            
            # Latency metrics
            mean_latency_ns=latency_metrics['mean_latency_ns'],
            p95_latency_ns=latency_metrics['p95_latency_ns'],
            p99_latency_ns=latency_metrics['p99_latency_ns'],
            
            # Risk metrics
            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            calmar_ratio=calmar_ratio,
            
            # Additional metrics
            start_time=start_time,
            end_time=end_time,
            duration_days=duration_days,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_fees=sum(trade.fees for trade in self.trades),
            total_slippage=0.0  # Placeholder - would need slippage data
        )
    
    def _calculate_returns_series(self) -> List[float]:
        """Calculate returns series from portfolio snapshots."""
        if len(self.snapshots) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.snapshots)):
            prev_value = self.snapshots[i-1].cash + self.snapshots[i-1].total_market_value
            curr_value = self.snapshots[i].cash + self.snapshots[i].total_market_value
            
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        return returns
    
    def _calculate_volatility(self, returns: List[float], duration_days: float) -> float:
        """Calculate annualized volatility."""
        if not returns or duration_days <= 0:
            return 0.0
        
        # Calculate daily volatility
        daily_vol = np.std(returns) if len(returns) > 1 else 0.0
        
        # Annualize (assuming 252 trading days per year)
        annualized_vol = daily_vol * math.sqrt(252)
        
        return annualized_vol
    
    def _calculate_sharpe_ratio(self, returns: List[float], volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if not returns or volatility == 0:
            return 0.0
        
        avg_return = np.mean(returns)
        excess_return = avg_return - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        return excess_return / volatility * math.sqrt(252)  # Annualized
    
    def _calculate_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not self.snapshots:
            return 0.0, 0
        
        # Calculate cumulative returns
        cumulative_returns = []
        for snapshot in self.snapshots:
            total_value = snapshot.cash + snapshot.total_market_value
            cumulative_returns.append(total_value)
        
        # Calculate running maximum
        running_max = []
        current_max = cumulative_returns[0]
        for value in cumulative_returns:
            if value > current_max:
                current_max = value
            running_max.append(current_max)
        
        # Calculate drawdowns
        drawdowns = []
        for i, (value, max_val) in enumerate(zip(cumulative_returns, running_max)):
            if max_val > 0:
                drawdown = (value - max_val) / max_val
                drawdowns.append(drawdown)
            else:
                drawdowns.append(0.0)
        
        # Find maximum drawdown
        max_drawdown = min(drawdowns) if drawdowns else 0.0
        
        # Find maximum drawdown duration
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for i, drawdown in enumerate(drawdowns):
            if drawdown < 0:
                current_drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
            else:
                current_drawdown_duration = 0
        
        # Convert duration to nanoseconds
        if len(self.snapshots) > 1:
            time_per_snapshot = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) / (len(self.snapshots) - 1)
            max_drawdown_duration = int(max_drawdown_duration * time_per_snapshot)
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_trading_metrics(self) -> Dict[str, Any]:
        """Calculate trading performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'hit_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'profit_factor': 0.0
            }
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_trades = len(self.trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        hit_rate = winning_count / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'hit_rate': hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor
        }
    
    def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        if not self.snapshots:
            return {
                'turnover': 0.0,
                'avg_position_size': 0.0,
                'max_position_size': 0.0
            }
        
        # Calculate turnover (simplified - would need trade volume data)
        total_trade_value = sum(abs(trade.quantity * trade.entry_price) for trade in self.trades)
        avg_portfolio_value = np.mean([s.cash + s.total_market_value for s in self.snapshots])
        turnover = total_trade_value / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
        
        # Calculate position size metrics
        position_sizes = []
        for snapshot in self.snapshots:
            for position in snapshot.positions.values():
                if not position.is_flat:
                    position_sizes.append(abs(position.quantity))
        
        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
        max_position_size = max(position_sizes) if position_sizes else 0.0
        
        return {
            'turnover': turnover,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size
        }
    
    def _calculate_latency_metrics(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self.latencies:
            return {
                'mean_latency_ns': 0.0,
                'p95_latency_ns': 0.0,
                'p99_latency_ns': 0.0
            }
        
        latencies = np.array(self.latencies)
        
        return {
            'mean_latency_ns': float(np.mean(latencies)),
            'p95_latency_ns': float(np.percentile(latencies, 95)),
            'p99_latency_ns': float(np.percentile(latencies, 99))
        }
    
    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate risk metrics (VaR, CVaR)."""
        if not returns:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        returns_array = np.array(returns)
        
        # Value at Risk 95%
        var_95 = float(np.percentile(returns_array, 5))  # 5th percentile
        
        # Conditional Value at Risk 95%
        cvar_95 = float(np.mean(returns_array[returns_array <= var_95])) if np.any(returns_array <= var_95) else 0.0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def generate_plots(self, output_dir: str = "output") -> None:
        """
        Generate performance visualization plots.
        
        Args:
            output_dir: Directory to save plot files
        """
        if not self.snapshots:
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Sort snapshots by timestamp
        self.snapshots.sort(key=lambda x: x.timestamp)
        
        # Convert timestamps to datetime for plotting
        timestamps = [pd.Timestamp(s.timestamp, unit='ns') for s in self.snapshots]
        
        # Calculate equity curve
        equity_values = [s.cash + s.total_market_value for s in self.snapshots]
        
        # 1. Equity Curve
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity_values, linewidth=2, color='blue')
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/equity_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Drawdown Curve
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = [(value - max_val) / max_val * 100 for value, max_val in zip(equity_values, running_max)]
        
        plt.fill_between(timestamps, drawdowns, 0, alpha=0.3, color='red')
        plt.plot(timestamps, drawdowns, linewidth=2, color='red')
        plt.title('Drawdown Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drawdown_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Trade PnL Histogram
        if self.trades:
            plt.figure(figsize=(10, 6))
            pnl_values = [trade.pnl for trade in self.trades]
            
            plt.hist(pnl_values, bins=50, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
            plt.title('Trade PnL Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('PnL ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/trade_pnl_histogram.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def export_json(self, filepath: str) -> None:
        """
        Export performance metrics to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        metrics = self.calculate_metrics()
        
        # Convert to dictionary
        metrics_dict = {
            'basic_metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'max_drawdown_duration_ns': metrics.max_drawdown_duration
            },
            'trading_metrics': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'hit_rate': metrics.hit_rate,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'win_loss_ratio': metrics.win_loss_ratio,
                'profit_factor': metrics.profit_factor
            },
            'portfolio_metrics': {
                'turnover': metrics.turnover,
                'avg_position_size': metrics.avg_position_size,
                'max_position_size': metrics.max_position_size
            },
            'latency_metrics': {
                'mean_latency_ns': metrics.mean_latency_ns,
                'p95_latency_ns': metrics.p95_latency_ns,
                'p99_latency_ns': metrics.p99_latency_ns
            },
            'risk_metrics': {
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'calmar_ratio': metrics.calmar_ratio
            },
            'summary': {
                'start_time': metrics.start_time,
                'end_time': metrics.end_time,
                'duration_days': metrics.duration_days,
                'initial_capital': metrics.initial_capital,
                'final_capital': metrics.final_capital,
                'total_fees': metrics.total_fees,
                'total_slippage': metrics.total_slippage
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def export_csv(self, filepath: str) -> None:
        """
        Export performance metrics to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        metrics = self.calculate_metrics()
        
        # Flatten metrics for CSV
        csv_data = {
            'Metric': [
                'Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio',
                'Max Drawdown', 'Max Drawdown Duration (ns)',
                'Total Trades', 'Winning Trades', 'Losing Trades', 'Hit Rate',
                'Avg Win', 'Avg Loss', 'Win/Loss Ratio', 'Profit Factor',
                'Turnover', 'Avg Position Size', 'Max Position Size',
                'Mean Latency (ns)', 'P95 Latency (ns)', 'P99 Latency (ns)',
                'VaR 95%', 'CVaR 95%', 'Calmar Ratio',
                'Start Time', 'End Time', 'Duration (days)',
                'Initial Capital', 'Final Capital', 'Total Fees', 'Total Slippage'
            ],
            'Value': [
                metrics.total_return, metrics.annualized_return, metrics.volatility,
                metrics.sharpe_ratio, metrics.max_drawdown, metrics.max_drawdown_duration,
                metrics.total_trades, metrics.winning_trades, metrics.losing_trades,
                metrics.hit_rate, metrics.avg_win, metrics.avg_loss,
                metrics.win_loss_ratio, metrics.profit_factor,
                metrics.turnover, metrics.avg_position_size, metrics.max_position_size,
                metrics.mean_latency_ns, metrics.p95_latency_ns, metrics.p99_latency_ns,
                metrics.var_95, metrics.cvar_95, metrics.calmar_ratio,
                metrics.start_time, metrics.end_time, metrics.duration_days,
                metrics.initial_capital, metrics.final_capital, metrics.total_fees,
                metrics.total_slippage
            ]
        }
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
    
    def generate_report(self, output_dir: str = "output") -> None:
        """
        Generate complete performance report with plots and data files.
        
        Args:
            output_dir: Directory to save all output files
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self.generate_plots(output_dir)
        
        # Export data files
        self.export_json(f"{output_dir}/performance.json")
        self.export_csv(f"{output_dir}/performance.csv")
        
        print(f"Performance report generated in {output_dir}/")
        print(f"- performance.json: Complete metrics in JSON format")
        print(f"- performance.csv: Metrics in CSV format")
        print(f"- equity_curve.png: Portfolio value over time")
        print(f"- drawdown_curve.png: Drawdown analysis")
        print(f"- trade_pnl_histogram.png: Trade PnL distribution")
