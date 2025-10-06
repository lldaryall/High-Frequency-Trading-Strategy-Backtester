"""Drawdown calculation and analysis."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DrawdownCalculator:
    """Calculate drawdown metrics and analysis."""
    
    def __init__(self) -> None:
        """Initialize drawdown calculator."""
        pass
        
    def calculate(self, pnl_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate drawdown metrics.
        
        Args:
            pnl_data: DataFrame with PnL data
            
        Returns:
            Dictionary with drawdown metrics
        """
        if pnl_data.empty:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0, "avg_drawdown": 0.0}
            
        # Get PnL series
        if 'total_pnl' in pnl_data.columns:
            pnl_series = pnl_data['total_pnl']
        else:
            pnl_series = pnl_data.iloc[:, 0]
            
        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(pnl_series)
        
        # Calculate metrics
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1] if not drawdowns.empty else 0.0
        avg_drawdown = drawdowns.mean()
        
        # Calculate drawdown duration
        duration_stats = self._calculate_drawdown_duration(drawdowns)
        
        # Calculate recovery time
        recovery_time = self._calculate_recovery_time(pnl_series)
        
        return {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "avg_drawdown": avg_drawdown,
            "max_drawdown_duration": duration_stats["max_duration"],
            "avg_drawdown_duration": duration_stats["avg_duration"],
            "recovery_time": recovery_time,
            "drawdown_series": drawdowns,
        }
        
    def _calculate_drawdowns(self, pnl_series: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        # Calculate running maximum
        running_max = pnl_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (pnl_series - running_max) / running_max
        
        # Handle case where running_max is 0
        drawdown = drawdown.replace([np.inf, -np.inf], 0)
        
        return drawdown
        
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> Dict[str, int]:
        """Calculate drawdown duration statistics."""
        if drawdowns.empty:
            return {"max_duration": 0, "avg_duration": 0}
            
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        
        current_period = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
                
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
            
        if not drawdown_periods:
            return {"max_duration": 0, "avg_duration": 0}
            
        return {
            "max_duration": max(drawdown_periods),
            "avg_duration": np.mean(drawdown_periods),
        }
        
    def _calculate_recovery_time(self, pnl_series: pd.Series) -> int:
        """Calculate time to recover from maximum drawdown."""
        if pnl_series.empty:
            return 0
            
        # Find maximum drawdown point
        drawdowns = self._calculate_drawdowns(pnl_series)
        max_dd_idx = drawdowns.idxmin()
        
        # Find when PnL recovered to pre-drawdown level
        max_dd_value = pnl_series.loc[max_dd_idx]
        recovery_idx = pnl_series[pnl_series.index > max_dd_idx].ge(max_dd_value).idxmax()
        
        if pd.isna(recovery_idx):
            return len(pnl_series) - pnl_series.index.get_loc(max_dd_idx)
        else:
            return pnl_series.index.get_loc(recovery_idx) - pnl_series.index.get_loc(max_dd_idx)
            
    def calculate_rolling_drawdown(self, pnl_series: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling drawdown."""
        if pnl_series.empty:
            return pd.Series(dtype=float)
            
        rolling_max = pnl_series.rolling(window).max()
        rolling_drawdown = (pnl_series - rolling_max) / rolling_max
        
        return rolling_drawdown.replace([np.inf, -np.inf], 0)
        
    def find_drawdown_periods(self, pnl_series: pd.Series, 
                            threshold: float = -0.05) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Find drawdown periods below threshold.
        
        Args:
            pnl_series: PnL series
            threshold: Drawdown threshold (e.g., -0.05 for -5%)
            
        Returns:
            List of (start, end) tuples for drawdown periods
        """
        drawdowns = self._calculate_drawdowns(pnl_series)
        below_threshold = drawdowns < threshold
        
        periods = []
        start_idx = None
        
        for i, is_below in enumerate(below_threshold):
            if is_below and start_idx is None:
                start_idx = i
            elif not is_below and start_idx is not None:
                periods.append((pnl_series.index[start_idx], pnl_series.index[i-1]))
                start_idx = None
                
        # Handle case where drawdown period extends to end
        if start_idx is not None:
            periods.append((pnl_series.index[start_idx], pnl_series.index[-1]))
            
        return periods
        
    def calculate_underwater_curve(self, pnl_series: pd.Series) -> pd.Series:
        """Calculate underwater curve (cumulative drawdown)."""
        drawdowns = self._calculate_drawdowns(pnl_series)
        return drawdowns.cumsum()
        
    def calculate_drawdown_at_risk(self, pnl_series: pd.Series, 
                                 confidence_level: float = 0.95) -> float:
        """Calculate drawdown at risk (DaR)."""
        drawdowns = self._calculate_drawdowns(pnl_series)
        return np.percentile(drawdowns, (1 - confidence_level) * 100)
