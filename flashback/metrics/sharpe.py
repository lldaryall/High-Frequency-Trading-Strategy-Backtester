"""Sharpe ratio calculation."""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class SharpeCalculator:
    """Calculate Sharpe ratio and related metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize Sharpe calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio and related metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with Sharpe metrics
        """
        if returns.empty:
            return {"sharpe_ratio": 0.0, "information_ratio": 0.0, "sortino_ratio": 0.0}
            
        # Basic Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = self._calculate_sharpe_ratio(excess_returns)
        
        # Information ratio (vs benchmark)
        information_ratio = self._calculate_information_ratio(returns)
        
        # Sortino ratio (downside deviation)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "information_ratio": information_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "excess_return": excess_returns.mean(),
            "volatility": returns.std(),
            "downside_deviation": self._calculate_downside_deviation(returns),
        }
        
    def _calculate_sharpe_ratio(self, excess_returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if excess_returns.empty or excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
    def _calculate_information_ratio(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series] = None) -> float:
        """Calculate information ratio."""
        if benchmark_returns is None:
            # Use zero as benchmark
            active_returns = returns
        else:
            # Align returns with benchmark
            aligned_returns = returns.reindex(benchmark_returns.index, method='ffill')
            active_returns = aligned_returns - benchmark_returns
            
        if active_returns.empty or active_returns.std() == 0:
            return 0.0
            
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_deviation = self._calculate_downside_deviation(returns)
        
        if downside_deviation == 0:
            return 0.0
            
        excess_return = returns.mean() - (self.risk_free_rate / 252)
        return excess_return / downside_deviation * np.sqrt(252)
        
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if returns.empty:
            return 0.0
            
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if max_drawdown == 0:
            return 0.0
            
        annualized_return = returns.mean() * 252
        return annualized_return / abs(max_drawdown)
        
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation."""
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            return 0.0
            
        return negative_returns.std()
        
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        if returns.empty:
            return pd.Series(dtype=float)
            
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Avoid division by zero
        rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
        
        return rolling_sharpe.fillna(0)
        
    def calculate_rolling_sortino(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling Sortino ratio."""
        if returns.empty:
            return pd.Series(dtype=float)
            
        rolling_mean = returns.rolling(window).mean()
        rolling_downside = returns.rolling(window).apply(
            lambda x: self._calculate_downside_deviation(x), raw=False
        )
        
        # Avoid division by zero
        rolling_sortino = rolling_mean / rolling_downside.replace(0, np.nan) * np.sqrt(252)
        
        return rolling_sortino.fillna(0)
