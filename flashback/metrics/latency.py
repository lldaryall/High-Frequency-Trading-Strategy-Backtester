"""Latency metrics calculation and analysis."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class LatencyCalculator:
    """Calculate latency metrics and analysis."""
    
    def __init__(self) -> None:
        """Initialize latency calculator."""
        pass
        
    def calculate(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate latency metrics.
        
        Args:
            trades: DataFrame with trade data including latency_us column
            
        Returns:
            Dictionary with latency metrics
        """
        if trades.empty or 'latency_us' not in trades.columns:
            return {"avg_latency_us": 0.0, "p99_latency_us": 0.0, "max_latency_us": 0.0}
            
        latencies = trades['latency_us']
        
        # Basic statistics
        avg_latency = latencies.mean()
        median_latency = latencies.median()
        std_latency = latencies.std()
        min_latency = latencies.min()
        max_latency = latencies.max()
        
        # Percentiles
        p90_latency = latencies.quantile(0.90)
        p95_latency = latencies.quantile(0.95)
        p99_latency = latencies.quantile(0.99)
        p99_9_latency = latencies.quantile(0.999)
        
        # Latency distribution
        latency_distribution = self._calculate_latency_distribution(latencies)
        
        # Latency by symbol
        latency_by_symbol = self._calculate_latency_by_symbol(trades)
        
        # Latency by time period
        latency_by_period = self._calculate_latency_by_period(trades)
        
        return {
            "avg_latency_us": avg_latency,
            "median_latency_us": median_latency,
            "std_latency_us": std_latency,
            "min_latency_us": min_latency,
            "max_latency_us": max_latency,
            "p90_latency_us": p90_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency,
            "p99_9_latency_us": p99_9_latency,
            "latency_distribution": latency_distribution,
            "latency_by_symbol": latency_by_symbol,
            "latency_by_period": latency_by_period,
        }
        
    def _calculate_latency_distribution(self, latencies: pd.Series) -> Dict[str, int]:
        """Calculate latency distribution by bins."""
        if latencies.empty:
            return {}
            
        # Define latency bins (microseconds)
        bins = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, float('inf')]
        bin_labels = ['0-10μs', '10-50μs', '50-100μs', '100-200μs', '200-500μs', 
                     '500μs-1ms', '1-2ms', '2-5ms', '5-10ms', '10ms+']
        
        distribution = pd.cut(latencies, bins=bins, labels=bin_labels, include_lowest=True)
        return distribution.value_counts().to_dict()
        
    def _calculate_latency_by_symbol(self, trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate latency metrics by symbol."""
        if trades.empty or 'latency_us' not in trades.columns:
            return {}
            
        latency_by_symbol = {}
        
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            symbol_latencies = symbol_trades['latency_us']
            
            latency_by_symbol[symbol] = {
                "avg_latency_us": symbol_latencies.mean(),
                "median_latency_us": symbol_latencies.median(),
                "p99_latency_us": symbol_latencies.quantile(0.99),
                "max_latency_us": symbol_latencies.max(),
            }
            
        return latency_by_symbol
        
    def _calculate_latency_by_period(self, trades: pd.DataFrame, 
                                   period: str = 'H') -> Dict[str, Dict[str, float]]:
        """Calculate latency metrics by time period."""
        if trades.empty or 'latency_us' not in trades.columns:
            return {}
            
        # Group trades by period
        trades_by_period = trades.groupby(pd.Grouper(freq=period))
        
        latency_by_period = {}
        
        for period, period_trades in trades_by_period:
            period_latencies = period_trades['latency_us']
            
            latency_by_period[period.strftime('%Y-%m-%d %H:%M')] = {
                "avg_latency_us": period_latencies.mean(),
                "median_latency_us": period_latencies.median(),
                "p99_latency_us": period_latencies.quantile(0.99),
                "max_latency_us": period_latencies.max(),
            }
            
        return latency_by_period
        
    def calculate_rolling_latency(self, trades: pd.DataFrame, 
                                window: int = 100) -> pd.Series:
        """Calculate rolling average latency."""
        if trades.empty or 'latency_us' not in trades.columns:
            return pd.Series(dtype=float)
            
        return trades['latency_us'].rolling(window).mean()
        
    def calculate_latency_trend(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate latency trend over time."""
        if trades.empty or 'latency_us' not in trades.columns:
            return {}
            
        # Calculate daily average latency
        daily_latency = trades.groupby(trades.index.date)['latency_us'].mean()
        
        if len(daily_latency) < 2:
            return {}
            
        # Calculate trend using linear regression
        x = np.arange(len(daily_latency))
        y = daily_latency.values
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        return {
            "latency_trend_slope": slope,
            "latency_trend_intercept": intercept,
            "latency_trend_r_squared": np.corrcoef(x, y)[0, 1] ** 2,
        }
        
    def calculate_latency_volatility(self, trades: pd.DataFrame) -> float:
        """Calculate latency volatility."""
        if trades.empty or 'latency_us' not in trades.columns:
            return 0.0
            
        # Calculate daily latency volatility
        daily_latency = trades.groupby(trades.index.date)['latency_us'].std()
        
        if len(daily_latency) < 2:
            return 0.0
            
        return daily_latency.mean()
        
    def calculate_latency_skewness(self, trades: pd.DataFrame) -> float:
        """Calculate latency skewness."""
        if trades.empty or 'latency_us' not in trades.columns:
            return 0.0
            
        latencies = trades['latency_us']
        
        if len(latencies) < 3:
            return 0.0
            
        return latencies.skew()
        
    def calculate_latency_kurtosis(self, trades: pd.DataFrame) -> float:
        """Calculate latency kurtosis."""
        if trades.empty or 'latency_us' not in trades.columns:
            return 0.0
            
        latencies = trades['latency_us']
        
        if len(latencies) < 4:
            return 0.0
            
        return latencies.kurtosis()
        
    def calculate_latency_autocorrelation(self, trades: pd.DataFrame, 
                                        lag: int = 1) -> float:
        """Calculate latency autocorrelation."""
        if trades.empty or 'latency_us' not in trades.columns:
            return 0.0
            
        # Calculate daily average latency
        daily_latency = trades.groupby(trades.index.date)['latency_us'].mean()
        
        if len(daily_latency) < lag + 1:
            return 0.0
            
        return daily_latency.autocorr(lag=lag)
        
    def calculate_latency_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive latency metrics."""
        if trades.empty or 'latency_us' not in trades.columns:
            return {}
            
        return {
            "latency_volatility": self.calculate_latency_volatility(trades),
            "latency_skewness": self.calculate_latency_skewness(trades),
            "latency_kurtosis": self.calculate_latency_kurtosis(trades),
            "latency_autocorrelation": self.calculate_latency_autocorrelation(trades),
            "latency_trend": self.calculate_latency_trend(trades),
        }
