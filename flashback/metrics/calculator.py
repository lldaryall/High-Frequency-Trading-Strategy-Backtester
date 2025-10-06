"""Main performance metrics calculator."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .sharpe import SharpeCalculator
from .drawdown import DrawdownCalculator
from .turnover import TurnoverCalculator
from .latency import LatencyCalculator
from ..utils.logger import get_logger


class PerformanceCalculator:
    """Main performance metrics calculator."""
    
    def __init__(self) -> None:
        """Initialize performance calculator."""
        self.logger = get_logger(__name__)
        
        # Initialize component calculators
        self.sharpe_calc = SharpeCalculator()
        self.drawdown_calc = DrawdownCalculator()
        self.turnover_calc = TurnoverCalculator()
        self.latency_calc = LatencyCalculator()
        
    def calculate(self, pnl_data: pd.DataFrame, positions: Dict[str, int], 
                 trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            pnl_data: DataFrame with PnL data
            positions: Current positions
            trades: Optional DataFrame with trade data
            
        Returns:
            Dictionary with performance metrics
        """
        if pnl_data.empty:
            return self._empty_metrics()
            
        # Calculate returns
        returns = self._calculate_returns(pnl_data)
        
        # Basic metrics
        metrics = self._calculate_basic_metrics(pnl_data, returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        metrics.update(risk_metrics)
        
        # Sharpe ratio
        sharpe_metrics = self.sharpe_calc.calculate(returns)
        metrics.update(sharpe_metrics)
        
        # Drawdown metrics
        drawdown_metrics = self.drawdown_calc.calculate(pnl_data)
        metrics.update(drawdown_metrics)
        
        # Turnover metrics
        if trades is not None and not trades.empty:
            turnover_metrics = self.turnover_calc.calculate(trades, positions)
            metrics.update(turnover_metrics)
            
        # Latency metrics
        if trades is not None and not trades.empty:
            latency_metrics = self.latency_calc.calculate(trades)
            metrics.update(latency_metrics)
            
        return metrics
        
    def _calculate_returns(self, pnl_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from PnL data."""
        if 'total_pnl' in pnl_data.columns:
            pnl_series = pnl_data['total_pnl']
        else:
            pnl_series = pnl_data.iloc[:, 0]  # Use first column if no total_pnl
            
        # Calculate period returns
        returns = pnl_series.pct_change().dropna()
        return returns
        
    def _calculate_basic_metrics(self, pnl_data: pd.DataFrame, 
                               returns: pd.Series) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        if pnl_data.empty:
            return {}
            
        total_pnl = pnl_data['total_pnl'].iloc[-1] if 'total_pnl' in pnl_data.columns else 0
        initial_pnl = pnl_data['total_pnl'].iloc[0] if 'total_pnl' in pnl_data.columns else 0
        
        # Calculate total return
        if initial_pnl != 0:
            total_return = (total_pnl - initial_pnl) / abs(initial_pnl)
        else:
            total_return = 0.0
            
        # Calculate annualized return
        if len(pnl_data) > 1:
            start_date = pnl_data.index[0]
            end_date = pnl_data.index[-1]
            years = (end_date - start_date).days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annualized_return = 0.0
            
        return {
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "num_periods": len(pnl_data),
        }
        
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics."""
        if returns.empty:
            return {}
            
        # Volatility
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(252)  # Assuming daily data
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Conditional Value at Risk)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        return {
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
        }
        
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no data."""
        return {
            "total_pnl": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "num_periods": 0,
            "volatility": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
        }
        
    def calculate_rolling_metrics(self, pnl_data: pd.DataFrame, 
                                window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            pnl_data: DataFrame with PnL data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        if pnl_data.empty:
            return pd.DataFrame()
            
        # Calculate rolling returns
        returns = pnl_data['total_pnl'].pct_change().dropna()
        
        # Rolling metrics
        rolling_metrics = pd.DataFrame(index=returns.index)
        rolling_metrics['rolling_return'] = returns.rolling(window).mean()
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std()
        rolling_metrics['rolling_sharpe'] = (rolling_metrics['rolling_return'] / 
                                           rolling_metrics['rolling_volatility'])
        
        # Rolling drawdown
        rolling_metrics['rolling_max'] = pnl_data['total_pnl'].rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (pnl_data['total_pnl'] - 
                                             rolling_metrics['rolling_max']) / rolling_metrics['rolling_max']
        
        return rolling_metrics.dropna()
        
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted performance report."""
        report = []
        report.append("=" * 50)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 50)
        
        # Basic metrics
        report.append("\nBASIC METRICS:")
        report.append(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        
        # Risk metrics
        report.append("\nRISK METRICS:")
        report.append(f"Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Current Drawdown: {metrics.get('current_drawdown', 0):.2%}")
        
        # VaR metrics
        report.append("\nVALUE AT RISK:")
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        report.append(f"VaR (99%): {metrics.get('var_99', 0):.2%}")
        report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        report.append(f"CVaR (99%): {metrics.get('cvar_99', 0):.2%}")
        
        # Distribution metrics
        report.append("\nDISTRIBUTION:")
        report.append(f"Skewness: {metrics.get('skewness', 0):.2f}")
        report.append(f"Kurtosis: {metrics.get('kurtosis', 0):.2f}")
        
        # Turnover metrics
        if 'turnover_rate' in metrics:
            report.append("\nTURNOVER:")
            report.append(f"Turnover Rate: {metrics.get('turnover_rate', 0):.2%}")
            report.append(f"Avg Trade Size: {metrics.get('avg_trade_size', 0):.2f}")
            
        # Latency metrics
        if 'avg_latency_us' in metrics:
            report.append("\nLATENCY:")
            report.append(f"Avg Latency: {metrics.get('avg_latency_us', 0):.0f} μs")
            report.append(f"P99 Latency: {metrics.get('p99_latency_us', 0):.0f} μs")
            
        report.append("=" * 50)
        
        return "\n".join(report)
