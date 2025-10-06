"""Risk limits and controls."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from ..core.events import OrderEvent
from ..utils.logger import get_logger


class RiskLimits:
    """Risk limits and controls system."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize risk limits.
        
        Args:
            config: Risk limits configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Position limits
        self.max_position = config.get("max_position", 1000)
        self.max_position_per_symbol = config.get("max_position_per_symbol", 1000)
        self.max_total_exposure = config.get("max_total_exposure", 10000)
        
        # PnL limits
        self.max_drawdown = config.get("max_drawdown", 0.05)  # 5%
        self.max_daily_loss = config.get("max_daily_loss", 1000.0)
        self.max_trade_size = config.get("max_trade_size", 100)
        
        # Risk metrics
        self.max_var = config.get("max_var", 0.02)  # 2% VaR
        self.max_volatility = config.get("max_volatility", 0.1)  # 10%
        
        # Time-based limits
        self.max_trades_per_minute = config.get("max_trades_per_minute", 10)
        self.max_trades_per_hour = config.get("max_trades_per_hour", 100)
        
        # Tracking
        self.trade_counts: Dict[str, int] = {}  # timestamp -> count
        self.daily_pnl: Dict[str, float] = {}  # date -> PnL
        
    def check_order(self, order: OrderEvent, positions: Dict[str, int]) -> bool:
        """
        Check if order passes risk limits.
        
        Args:
            order: Order to check
            positions: Current positions
            
        Returns:
            True if order passes risk checks
        """
        # Check position limits
        if not self._check_position_limits(order, positions):
            return False
            
        # Check trade size limits
        if not self._check_trade_size_limits(order):
            return False
            
        # Check time-based limits
        if not self._check_time_limits(order):
            return False
            
        # Check PnL limits
        if not self._check_pnl_limits():
            return False
            
        return True
        
    def _check_position_limits(self, order: OrderEvent, positions: Dict[str, int]) -> bool:
        """Check position limits."""
        symbol = order.symbol
        current_position = positions.get(symbol, 0)
        
        # Calculate new position
        quantity = order.quantity if order.side == 'B' else -order.quantity
        new_position = current_position + quantity
        
        # Check per-symbol limit
        if abs(new_position) > self.max_position_per_symbol:
            self.logger.warning(f"Position limit exceeded for {symbol}: {new_position}")
            return False
            
        # Check total position limit
        total_position = sum(abs(pos) for pos in positions.values())
        if total_position + abs(quantity) > self.max_position:
            self.logger.warning(f"Total position limit exceeded: {total_position + abs(quantity)}")
            return False
            
        return True
        
    def _check_trade_size_limits(self, order: OrderEvent) -> bool:
        """Check trade size limits."""
        if order.quantity > self.max_trade_size:
            self.logger.warning(f"Trade size limit exceeded: {order.quantity}")
            return False
            
        return True
        
    def _check_time_limits(self, order: OrderEvent) -> bool:
        """Check time-based limits."""
        timestamp = order.timestamp
        minute_key = timestamp.strftime("%Y-%m-%d %H:%M")
        hour_key = timestamp.strftime("%Y-%m-%d %H")
        
        # Check trades per minute
        minute_count = self.trade_counts.get(minute_key, 0)
        if minute_count >= self.max_trades_per_minute:
            self.logger.warning(f"Trades per minute limit exceeded: {minute_count}")
            return False
            
        # Check trades per hour
        hour_count = sum(count for key, count in self.trade_counts.items() 
                        if key.startswith(hour_key))
        if hour_count >= self.max_trades_per_hour:
            self.logger.warning(f"Trades per hour limit exceeded: {hour_count}")
            return False
            
        # Update trade count
        self.trade_counts[minute_key] = minute_count + 1
        
        return True
        
    def _check_pnl_limits(self) -> bool:
        """Check PnL limits."""
        # This would typically check against current PnL
        # For now, we'll assume it passes
        return True
        
    def update_daily_pnl(self, date: str, pnl: float) -> None:
        """Update daily PnL tracking."""
        self.daily_pnl[date] = pnl
        
    def check_daily_loss_limit(self, date: str) -> bool:
        """Check if daily loss limit is exceeded."""
        daily_pnl = self.daily_pnl.get(date, 0.0)
        return daily_pnl >= -self.max_daily_loss
        
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: List of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value
        """
        if not returns:
            return 0.0
            
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        return abs(var)
        
    def calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate volatility of returns.
        
        Args:
            returns: List of returns
            
        Returns:
            Volatility (standard deviation)
        """
        if len(returns) < 2:
            return 0.0
            
        return np.std(returns)
        
    def check_volatility_limit(self, returns: List[float]) -> bool:
        """Check if volatility limit is exceeded."""
        volatility = self.calculate_volatility(returns)
        return volatility <= self.max_volatility
        
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary."""
        return {
            "max_position": self.max_position,
            "max_position_per_symbol": self.max_position_per_symbol,
            "max_total_exposure": self.max_total_exposure,
            "max_drawdown": self.max_drawdown,
            "max_daily_loss": self.max_daily_loss,
            "max_trade_size": self.max_trade_size,
            "max_var": self.max_var,
            "max_volatility": self.max_volatility,
            "max_trades_per_minute": self.max_trades_per_minute,
            "max_trades_per_hour": self.max_trades_per_hour,
        }
        
    def reset(self) -> None:
        """Reset risk limits state."""
        self.trade_counts.clear()
        self.daily_pnl.clear()
