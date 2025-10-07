"""
Mean reversion trading strategy.

This strategy trades based on the z-score of the mid-price relative to a short rolling mean.
When the price deviates significantly from its recent average, the strategy takes contrarian positions.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .base import BaseStrategy, StrategyConfig, StrategyState
from ..core.events import MarketDataEvent, FillEvent
from ..market.orders import OrderSide, OrderType, TimeInForce
from ..exec.router import NewOrder


class MeanReversionConfig(BaseModel):
    """Configuration for mean reversion strategy."""
    
    # Base strategy config
    strategy_id: str
    symbol: str
    enabled: bool = True
    max_position: Optional[int] = None
    max_order_size: Optional[int] = None
    risk_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Strategy parameters
    lookback_period: int = Field(default=20, ge=5, le=100, description="Rolling mean lookback period")
    z_score_threshold: float = Field(default=2.0, ge=0.5, le=5.0, description="Z-score threshold for entry")
    exit_z_score: float = Field(default=0.5, ge=0.1, le=2.0, description="Z-score threshold for exit")
    position_size: int = Field(default=100, ge=1, le=10000, description="Position size in shares")
    min_price_history: int = Field(default=30, ge=10, le=100, description="Minimum price history before trading")
    
    # Risk management
    stop_loss_pct: Optional[float] = Field(default=0.02, ge=0.001, le=0.1, description="Stop loss percentage")
    
    @field_validator('exit_z_score')
    @classmethod
    def exit_must_be_less_than_entry(cls, v, info):
        """Validate that exit threshold is less than entry threshold."""
        if hasattr(info, 'data') and 'z_score_threshold' in info.data and v >= info.data['z_score_threshold']:
            raise ValueError('exit_z_score must be less than z_score_threshold')
        return v


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    This strategy:
    1. Calculates a rolling mean of mid-prices over a lookback period
    2. Computes the z-score of the current price relative to this mean
    3. Enters long positions when price is significantly below mean (negative z-score)
    4. Enters short positions when price is significantly above mean (positive z-score)
    5. Exits positions when z-score returns to neutral levels
    """
    
    def __init__(self, config: MeanReversionConfig):
        """
        Initialize mean reversion strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        self.config: MeanReversionConfig = config
        
        # Strategy state
        self.current_position: int = 0  # Current position size (positive = long, negative = short)
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[int] = None
        
        # Price statistics
        self.rolling_mean: Optional[float] = None
        self.rolling_std: Optional[float] = None
        self.current_z_score: Optional[float] = None
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
    
    def on_bar(self, book_update: MarketDataEvent) -> List[NewOrder]:
        """
        Handle order book updates.
        
        Args:
            book_update: Market data event
            
        Returns:
            List of new order intents
        """
        if not self.is_active():
            return []
        
        # Update market data
        self._update_market_data(book_update)
        
        # Extract mid price
        mid_price = self._get_mid_price(book_update)
        if mid_price is None or mid_price <= 0:
            return []
        
        # Need sufficient price history
        if len(self.price_history) < self.config.min_price_history:
            return []
        
        # Calculate rolling statistics
        self._update_rolling_statistics()
        
        # Generate trading signals
        return self._generate_signals(mid_price)
    
    def on_trade(self, trade: FillEvent) -> List[NewOrder]:
        """
        Handle trade executions.
        
        Args:
            trade: Fill event
            
        Returns:
            List of new order intents
        """
        if not self.is_active():
            return []
        
        # Update position based on trade
        if trade.side == 'BUY':
            self.current_position += trade.quantity
        else:
            self.current_position -= trade.quantity
        
        # Update entry price and timestamp
        if self.entry_price is None:
            self.entry_price = trade.price
            self.entry_timestamp = trade.timestamp.value
        else:
            # Update average entry price
            if self.current_position != 0:
                self.entry_price = (self.entry_price * abs(self.current_position - trade.quantity) + 
                                  trade.price * trade.quantity) / abs(self.current_position)
        
        # Update statistics
        self.total_fills += 1
        
        # Check if position is closed
        if self.current_position == 0:
            # Store the position size before it was zeroed out
            # For a SELL trade that closes a long position, use the original position size
            # For a BUY trade that closes a short position, use the original position size
            if trade.side == 'SELL':
                # This was a sell that closed a long position
                position_size = trade.quantity
            else:
                # This was a buy that closed a short position
                position_size = -trade.quantity
            self._close_position(trade.price, position_size)
        
        return []
    
    def on_timer(self, timestamp: int) -> List[NewOrder]:
        """
        Handle timer events.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of new order intents
        """
        if not self.is_active():
            return []
        
        # Check for stop loss
        if self.current_position != 0 and self.entry_price is not None:
            current_price = self.price_history[-1] if self.price_history else None
            if current_price and self._should_stop_loss(current_price):
                return self._create_exit_orders(current_price)
        
        return []
    
    def _update_rolling_statistics(self) -> None:
        """Update rolling mean and standard deviation."""
        if len(self.price_history) < self.config.lookback_period:
            return
        
        # Get recent prices
        recent_prices = self.price_history[-self.config.lookback_period:]
        
        # Calculate rolling statistics
        self.rolling_mean = np.mean(recent_prices)
        self.rolling_std = np.std(recent_prices)
        
        # Calculate current z-score
        if self.rolling_std > 0:
            self.current_z_score = (self.price_history[-1] - self.rolling_mean) / self.rolling_std
        else:
            self.current_z_score = 0.0
    
    def _generate_signals(self, current_price: float) -> List[NewOrder]:
        """
        Generate trading signals based on z-score.
        
        Args:
            current_price: Current mid price
            
        Returns:
            List of new order intents
        """
        if self.current_z_score is None:
            return []
        
        orders = []
        
        # Entry signals
        if self.current_position == 0:
            # Long signal: price significantly below mean
            if self.current_z_score <= -self.config.z_score_threshold:
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=self.config.position_size
                ))
            
            # Short signal: price significantly above mean
            elif self.current_z_score >= self.config.z_score_threshold:
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=self.config.position_size
                ))
        
        # Exit signals
        elif self.current_position != 0:
            # Exit long position
            if self.current_position > 0 and self.current_z_score >= -self.config.exit_z_score:
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=abs(self.current_position)
                ))
            
            # Exit short position
            elif self.current_position < 0 and self.current_z_score <= self.config.exit_z_score:
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=abs(self.current_position)
                ))
        
        return orders
    
    def _create_exit_orders(self, current_price: float) -> List[NewOrder]:
        """
        Create exit orders for current position.
        
        Args:
            current_price: Current mid price
            
        Returns:
            List of exit order intents
        """
        if self.current_position == 0:
            return []
        
        orders = []
        
        if self.current_position > 0:
            # Exit long position
            orders.append(self._create_order_intent(
                side=OrderSide.SELL,
                price=current_price,
                quantity=self.current_position
            ))
        else:
            # Exit short position
            orders.append(self._create_order_intent(
                side=OrderSide.BUY,
                price=current_price,
                quantity=abs(self.current_position)
            ))
        
        return orders
    
    def _should_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss should be triggered.
        
        Args:
            current_price: Current price
            
        Returns:
            True if stop loss should be triggered
        """
        if self.entry_price is None or self.config.stop_loss_pct is None:
            return False
        
        if self.current_position > 0:
            # Long position: check if price dropped below stop loss
            return current_price <= self.entry_price * (1 - self.config.stop_loss_pct)
        elif self.current_position < 0:
            # Short position: check if price rose above stop loss
            return current_price >= self.entry_price * (1 + self.config.stop_loss_pct)
        
        return False
    
    def _close_position(self, exit_price: float, position_size: int = None) -> None:
        """
        Close current position and update statistics.
        
        Args:
            exit_price: Price at which position was closed
            position_size: Size of position being closed (if None, use current_position)
        """
        if self.entry_price is None:
            return
        
        # Use provided position size or current position
        if position_size is None:
            position_size = self.current_position
        
        # Calculate P&L
        if position_size > 0:
            # Long position
            pnl = (exit_price - self.entry_price) * position_size
        else:
            # Short position
            pnl = (self.entry_price - exit_price) * abs(position_size)
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Reset position
        self.current_position = 0
        self.entry_price = None
        self.entry_timestamp = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get strategy statistics.
        
        Returns:
            Dictionary containing strategy statistics
        """
        base_stats = super().get_statistics()
        
        # Add strategy-specific statistics
        strategy_stats = {
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "current_z_score": self.current_z_score,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "avg_pnl_per_trade": self.total_pnl / max(self.total_trades, 1)
        }
        
        base_stats.update(strategy_stats)
        return base_stats