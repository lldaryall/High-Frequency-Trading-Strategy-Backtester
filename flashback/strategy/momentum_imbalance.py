"""
Momentum and order flow imbalance trading strategy.

This strategy combines order flow imbalance with short/long EMA crossover signals
to identify momentum opportunities in the market.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .base import BaseStrategy, StrategyConfig, StrategyState
from ..core.events import MarketDataEvent, FillEvent
from ..market.orders import OrderSide, OrderType, TimeInForce
from ..exec.router import NewOrder


class MomentumImbalanceConfig(BaseModel):
    """Configuration for momentum imbalance strategy."""
    
    # Base strategy config
    strategy_id: str
    symbol: str
    enabled: bool = True
    max_position: Optional[int] = None
    max_order_size: Optional[int] = None
    risk_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Order flow imbalance parameters
    imbalance_lookback: int = Field(default=10, ge=5, le=50, description="Lookback period for order flow imbalance")
    imbalance_threshold: float = Field(default=0.6, ge=0.1, le=0.9, description="Order flow imbalance threshold")
    
    # EMA parameters
    short_ema_period: int = Field(default=5, ge=2, le=20, description="Short EMA period")
    long_ema_period: int = Field(default=20, ge=5, le=50, description="Long EMA period")
    
    # Trading parameters
    position_size: int = Field(default=100, ge=1, le=10000, description="Position size in shares")
    min_price_history: int = Field(default=50, ge=20, le=100, description="Minimum price history before trading")
    
    # Risk management
    stop_loss_pct: Optional[float] = Field(default=0.015, ge=0.001, le=0.1, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(default=0.03, ge=0.001, le=0.1, description="Take profit percentage")
    
    @field_validator('long_ema_period')
    @classmethod
    def long_ema_must_be_greater_than_short(cls, v, info):
        """Validate that long EMA period is greater than short EMA period."""
        if hasattr(info, 'data') and 'short_ema_period' in info.data and v <= info.data['short_ema_period']:
            raise ValueError('long_ema_period must be greater than short_ema_period')
        return v
    
    @field_validator('take_profit_pct')
    @classmethod
    def take_profit_must_be_greater_than_stop_loss(cls, v, info):
        """Validate that take profit is greater than stop loss."""
        if hasattr(info, 'data') and 'stop_loss_pct' in info.data and v <= info.data['stop_loss_pct']:
            raise ValueError('take_profit_pct must be greater than stop_loss_pct')
        return v


class MomentumImbalanceStrategy(BaseStrategy):
    """
    Momentum and order flow imbalance trading strategy.
    
    This strategy:
    1. Calculates order flow imbalance (buy volume vs sell volume)
    2. Computes short and long EMAs of prices
    3. Enters long positions when:
       - Order flow imbalance is positive (more buying pressure)
       - Short EMA crosses above long EMA (bullish momentum)
    4. Enters short positions when:
       - Order flow imbalance is negative (more selling pressure)
       - Short EMA crosses below long EMA (bearish momentum)
    5. Exits positions based on stop loss, take profit, or signal reversal
    """
    
    def __init__(self, config: MomentumImbalanceConfig):
        """
        Initialize momentum imbalance strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        self.config: MomentumImbalanceConfig = config
        
        # Strategy state (inherited from BaseStrategy.position)
        self.current_position = self.position.current_position
        self.entry_price = self.position.entry_price
        self.entry_timestamp = self.position.entry_timestamp
        self.total_pnl = self.position.total_pnl
        self.total_trades = self.position.total_trades
        
        # Price history tracking
        self.price_history: deque[float] = deque(maxlen=config.long_ema_period * 2)
        
        # Order flow tracking
        self.buy_volume_history: List[int] = []
        self.sell_volume_history: List[int] = []
        self.current_imbalance: Optional[float] = None
        
        # EMA calculations
        self.short_ema: Optional[float] = None
        self.long_ema: Optional[float] = None
        self.ema_alpha_short: float = 2.0 / (self.config.short_ema_period + 1)
        self.ema_alpha_long: float = 2.0 / (self.config.long_ema_period + 1)
        
        # Signal tracking
        self.last_ema_signal: Optional[str] = None  # 'bullish', 'bearish', or None
        self.last_imbalance_signal: Optional[str] = None  # 'bullish', 'bearish', or None
        
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
        
        # Extract price and volume
        mid_price = self._get_mid_price(book_update)
        volume = self._get_volume(book_update)
        
        if mid_price is None or mid_price <= 0:
            return []
        
        # Add price to history
        self.price_history.append(mid_price)
        
        # Need sufficient price history
        if len(self.price_history) < self.config.min_price_history:
            return []
        
        # Update order flow imbalance
        self._update_order_flow_imbalance(volume)
        
        # Update EMAs
        self._update_emas(mid_price)
        
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
            self.position.current_position += trade.quantity
        else:
            self.current_position -= trade.quantity
            self.position.current_position -= trade.quantity
        
        # Update entry price and timestamp
        if self.entry_price is None:
            self.entry_price = trade.price
            self.entry_timestamp = trade.timestamp.value
            self.position.entry_price = trade.price
            self.position.entry_timestamp = trade.timestamp.value
        else:
            # Update average entry price
            if self.current_position != 0:
                self.entry_price = (self.entry_price * abs(self.current_position - trade.quantity) + 
                                  trade.price * trade.quantity) / abs(self.current_position)
                self.position.entry_price = self.entry_price
        
        # Update statistics
        self.total_fills += 1
        
        # Check if position is closed
        if self.current_position == 0:
            # Calculate P&L before position was closed
            if self.entry_price is not None:
                if trade.side == 'BUY':
                    # We were short, now closed
                    pnl = (self.entry_price - trade.price) * trade.quantity
                else:
                    # We were long, now closed
                    pnl = (trade.price - self.entry_price) * trade.quantity
                
                # Update statistics
                self.total_trades += 1
                self.total_pnl += pnl
                self.position.total_trades += 1
                self.position.total_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Reset position
                self.entry_price = None
                self.entry_timestamp = None
                self.position.entry_price = None
                self.position.entry_timestamp = None
        
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
        
        # Check for stop loss or take profit
        if self.current_position != 0 and self.entry_price is not None:
            current_price = self.price_history[-1] if self.price_history else None
            if current_price:
                return self._check_exit_conditions(current_price)
        
        return []
    
    def _update_order_flow_imbalance(self, volume: int) -> None:
        """
        Update order flow imbalance calculation.
        
        Args:
            volume: Current volume (simplified - in practice would be buy/sell volume)
        """
        # Simplified implementation: assume 50/50 split for now
        # In practice, you'd track actual buy/sell volumes separately
        buy_volume = int(volume * 0.5)
        sell_volume = volume - buy_volume
        
        self.buy_volume_history.append(buy_volume)
        self.sell_volume_history.append(sell_volume)
        
        # Keep only recent history
        max_history = self.config.imbalance_lookback * 2
        if len(self.buy_volume_history) > max_history:
            self.buy_volume_history = self.buy_volume_history[-max_history:]
            self.sell_volume_history = self.sell_volume_history[-max_history:]
        
        # Calculate imbalance if we have enough data
        if len(self.buy_volume_history) >= self.config.imbalance_lookback:
            recent_buy = sum(self.buy_volume_history[-self.config.imbalance_lookback:])
            recent_sell = sum(self.sell_volume_history[-self.config.imbalance_lookback:])
            total_volume = recent_buy + recent_sell
            
            if total_volume > 0:
                self.current_imbalance = (recent_buy - recent_sell) / total_volume
            else:
                self.current_imbalance = 0.0
    
    def _update_emas(self, price: float) -> None:
        """
        Update EMA calculations.
        
        Args:
            price: Current price
        """
        if self.short_ema is None:
            # Initialize EMAs with current price
            self.short_ema = price
            self.long_ema = price
        else:
            # Update EMAs
            self.short_ema = self.ema_alpha_short * price + (1 - self.ema_alpha_short) * self.short_ema
            self.long_ema = self.ema_alpha_long * price + (1 - self.ema_alpha_long) * self.long_ema
    
    def _generate_signals(self, current_price: float) -> List[NewOrder]:
        """
        Generate trading signals based on momentum and imbalance.
        
        Args:
            current_price: Current mid price
            
        Returns:
            List of new order intents
        """
        if self.current_imbalance is None or self.short_ema is None or self.long_ema is None:
            return []
        
        orders = []
        
        # Determine current signals
        ema_signal = self._get_ema_signal()
        imbalance_signal = self._get_imbalance_signal()
        
        # Entry signals
        if self.current_position == 0:
            # Long signal: bullish EMA crossover + positive imbalance
            if (ema_signal == 'bullish' and imbalance_signal == 'bullish' and
                self.last_ema_signal != 'bullish'):  # Crossover condition
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=self.config.position_size
                ))
            
            # Short signal: bearish EMA crossover + negative imbalance
            elif (ema_signal == 'bearish' and imbalance_signal == 'bearish' and
                  self.last_ema_signal != 'bearish'):  # Crossover condition
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=self.config.position_size
                ))
        
        # Exit signals
        elif self.current_position != 0:
            # Exit long position on bearish signals
            if (self.current_position > 0 and 
                (ema_signal == 'bearish' or imbalance_signal == 'bearish')):
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=self.current_position
                ))
            
            # Exit short position on bullish signals
            elif (self.current_position < 0 and 
                  (ema_signal == 'bullish' or imbalance_signal == 'bullish')):
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=abs(self.current_position)
                ))
        
        # Update signal history
        self.last_ema_signal = ema_signal
        self.last_imbalance_signal = imbalance_signal
        
        return orders
    
    def _get_ema_signal(self) -> Optional[str]:
        """Get current EMA signal."""
        if self.short_ema is None or self.long_ema is None:
            return None
        
        if self.short_ema > self.long_ema:
            return 'bullish'
        elif self.short_ema < self.long_ema:
            return 'bearish'
        else:
            return None
    
    def _get_imbalance_signal(self) -> Optional[str]:
        """Get current imbalance signal."""
        if self.current_imbalance is None:
            return None
        
        if self.current_imbalance > self.config.imbalance_threshold:
            return 'bullish'
        elif self.current_imbalance < -self.config.imbalance_threshold:
            return 'bearish'
        else:
            return None
    
    def _check_exit_conditions(self, current_price: float) -> List[NewOrder]:
        """
        Check for stop loss or take profit conditions.
        
        Args:
            current_price: Current price
            
        Returns:
            List of exit order intents
        """
        if self.entry_price is None or self.current_position == 0:
            return []
        
        orders = []
        
        if self.current_position > 0:
            # Long position
            if self.config.stop_loss_pct and current_price <= self.entry_price * (1 - self.config.stop_loss_pct):
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=self.current_position
                ))
            elif (self.config.take_profit_pct and 
                  current_price >= self.entry_price * (1 + self.config.take_profit_pct)):
                orders.append(self._create_order_intent(
                    side=OrderSide.SELL,
                    price=current_price,
                    quantity=self.current_position
                ))
        
        elif self.current_position < 0:
            # Short position
            if self.config.stop_loss_pct and current_price >= self.entry_price * (1 + self.config.stop_loss_pct):
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=abs(self.current_position)
                ))
            elif (self.config.take_profit_pct and 
                  current_price <= self.entry_price * (1 - self.config.take_profit_pct)):
                orders.append(self._create_order_intent(
                    side=OrderSide.BUY,
                    price=current_price,
                    quantity=abs(self.current_position)
                ))
        
        return orders
    
    def _close_position(self, exit_price: float) -> None:
        """
        Close current position and update statistics.
        
        Args:
            exit_price: Price at which position was closed
        """
        if self.entry_price is None:
            return
        
        # Calculate P&L using the position before it was reset
        position_size = self.current_position
        if position_size > 0:
            # Long position
            pnl = (exit_price - self.entry_price) * position_size
        else:
            # Short position
            pnl = (self.entry_price - exit_price) * abs(position_size)
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += pnl
        self.position.total_trades += 1
        self.position.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Reset position
        self.current_position = 0
        self.entry_price = None
        self.entry_timestamp = None
        self.position.current_position = 0
        self.position.entry_price = None
        self.position.entry_timestamp = None
    
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
            "current_imbalance": self.current_imbalance,
            "short_ema": self.short_ema,
            "long_ema": self.long_ema,
            "ema_signal": self._get_ema_signal(),
            "imbalance_signal": self._get_imbalance_signal(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "avg_pnl_per_trade": self.total_pnl / max(self.total_trades, 1)
        }
        
        base_stats.update(strategy_stats)
        return base_stats
