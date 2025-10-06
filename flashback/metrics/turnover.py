"""Turnover calculation and analysis."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class TurnoverCalculator:
    """Calculate turnover metrics and analysis."""
    
    def __init__(self) -> None:
        """Initialize turnover calculator."""
        pass
        
    def calculate(self, trades: pd.DataFrame, positions: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate turnover metrics.
        
        Args:
            trades: DataFrame with trade data
            positions: Current positions
            
        Returns:
            Dictionary with turnover metrics
        """
        if trades.empty:
            return {"turnover_rate": 0.0, "avg_trade_size": 0.0, "trade_frequency": 0.0}
            
        # Calculate turnover rate
        turnover_rate = self._calculate_turnover_rate(trades, positions)
        
        # Calculate average trade size
        avg_trade_size = self._calculate_avg_trade_size(trades)
        
        # Calculate trade frequency
        trade_frequency = self._calculate_trade_frequency(trades)
        
        # Calculate turnover by symbol
        turnover_by_symbol = self._calculate_turnover_by_symbol(trades, positions)
        
        # Calculate turnover by time period
        turnover_by_period = self._calculate_turnover_by_period(trades)
        
        return {
            "turnover_rate": turnover_rate,
            "avg_trade_size": avg_trade_size,
            "trade_frequency": trade_frequency,
            "turnover_by_symbol": turnover_by_symbol,
            "turnover_by_period": turnover_by_period,
        }
        
    def _calculate_turnover_rate(self, trades: pd.DataFrame, 
                               positions: Dict[str, int]) -> float:
        """Calculate overall turnover rate."""
        if trades.empty:
            return 0.0
            
        # Calculate total volume traded
        total_volume = trades['quantity'].sum()
        
        # Calculate average position size
        avg_position = np.mean([abs(pos) for pos in positions.values()]) if positions else 0
        
        if avg_position == 0:
            return 0.0
            
        # Turnover rate = total volume / average position
        return total_volume / avg_position
        
    def _calculate_avg_trade_size(self, trades: pd.DataFrame) -> float:
        """Calculate average trade size."""
        if trades.empty:
            return 0.0
            
        return trades['quantity'].mean()
        
    def _calculate_trade_frequency(self, trades: pd.DataFrame) -> float:
        """Calculate trade frequency (trades per day)."""
        if trades.empty:
            return 0.0
            
        # Calculate time span
        start_time = trades.index.min()
        end_time = trades.index.max()
        time_span = (end_time - start_time).total_seconds() / 86400  # Days
        
        if time_span == 0:
            return 0.0
            
        return len(trades) / time_span
        
    def _calculate_turnover_by_symbol(self, trades: pd.DataFrame, 
                                    positions: Dict[str, int]) -> Dict[str, float]:
        """Calculate turnover rate by symbol."""
        if trades.empty:
            return {}
            
        turnover_by_symbol = {}
        
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            symbol_volume = symbol_trades['quantity'].sum()
            symbol_position = abs(positions.get(symbol, 0))
            
            if symbol_position > 0:
                turnover_by_symbol[symbol] = symbol_volume / symbol_position
            else:
                turnover_by_symbol[symbol] = 0.0
                
        return turnover_by_symbol
        
    def _calculate_turnover_by_period(self, trades: pd.DataFrame, 
                                    period: str = 'D') -> Dict[str, float]:
        """Calculate turnover by time period."""
        if trades.empty:
            return {}
            
        # Group trades by period
        trades_by_period = trades.groupby(pd.Grouper(freq=period))
        
        turnover_by_period = {}
        
        for period, period_trades in trades_by_period:
            period_volume = period_trades['quantity'].sum()
            turnover_by_period[period.strftime('%Y-%m-%d')] = period_volume
            
        return turnover_by_period
        
    def calculate_rolling_turnover(self, trades: pd.DataFrame, 
                                 window: int = 30) -> pd.Series:
        """Calculate rolling turnover."""
        if trades.empty:
            return pd.Series(dtype=float)
            
        # Resample to daily frequency
        daily_volume = trades.groupby(trades.index.date)['quantity'].sum()
        daily_volume.index = pd.to_datetime(daily_volume.index)
        
        # Calculate rolling turnover
        rolling_turnover = daily_volume.rolling(window).sum()
        
        return rolling_turnover
        
    def calculate_turnover_volatility(self, trades: pd.DataFrame) -> float:
        """Calculate turnover volatility."""
        if trades.empty:
            return 0.0
            
        # Calculate daily turnover
        daily_turnover = trades.groupby(trades.index.date)['quantity'].sum()
        
        if len(daily_turnover) < 2:
            return 0.0
            
        return daily_turnover.std()
        
    def calculate_turnover_skewness(self, trades: pd.DataFrame) -> float:
        """Calculate turnover skewness."""
        if trades.empty:
            return 0.0
            
        # Calculate daily turnover
        daily_turnover = trades.groupby(trades.index.date)['quantity'].sum()
        
        if len(daily_turnover) < 3:
            return 0.0
            
        return daily_turnover.skew()
        
    def calculate_turnover_kurtosis(self, trades: pd.DataFrame) -> float:
        """Calculate turnover kurtosis."""
        if trades.empty:
            return 0.0
            
        # Calculate daily turnover
        daily_turnover = trades.groupby(trades.index.date)['quantity'].sum()
        
        if len(daily_turnover) < 4:
            return 0.0
            
        return daily_turnover.kurtosis()
        
    def calculate_turnover_autocorrelation(self, trades: pd.DataFrame, 
                                         lag: int = 1) -> float:
        """Calculate turnover autocorrelation."""
        if trades.empty:
            return 0.0
            
        # Calculate daily turnover
        daily_turnover = trades.groupby(trades.index.date)['quantity'].sum()
        
        if len(daily_turnover) < lag + 1:
            return 0.0
            
        return daily_turnover.autocorr(lag=lag)
        
    def calculate_turnover_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive turnover metrics."""
        if trades.empty:
            return {}
            
        return {
            "turnover_volatility": self.calculate_turnover_volatility(trades),
            "turnover_skewness": self.calculate_turnover_skewness(trades),
            "turnover_kurtosis": self.calculate_turnover_kurtosis(trades),
            "turnover_autocorrelation": self.calculate_turnover_autocorrelation(trades),
        }
