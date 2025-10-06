"""
Base strategy interfaces and protocols.

This module defines the Strategy protocol that all trading strategies must implement,
along with base classes and utilities for strategy development.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from ..core.events import MarketDataEvent, OrderEvent, FillEvent, RejectEvent
from ..market.orders import Order, OrderSide, OrderType, TimeInForce
from ..exec.router import NewOrder, CancelOrder


class StrategyState(Enum):
    """Strategy execution states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyPosition:
    """Strategy position tracking."""
    current_position: int = 0
    entry_price: Optional[float] = None
    entry_timestamp: Optional[int] = None
    total_pnl: float = 0.0
    total_trades: int = 0


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""
    strategy_id: str
    symbol: str
    enabled: bool = True
    max_position: Optional[int] = None
    max_order_size: Optional[int] = None
    risk_limits: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.risk_limits is None:
            self.risk_limits = {}


class Strategy(Protocol):
    """
    Protocol defining the interface that all trading strategies must implement.
    
    Strategies receive market data updates and emit trading intents (orders).
    Position management and risk limits are handled by the risk management system.
    """
    
    def on_bar(self, book_update: MarketDataEvent) -> List[NewOrder]:
        """
        Handle order book updates (market data events).
        
        Args:
            book_update: Market data event containing order book information
            
        Returns:
            List of new order intents to submit
        """
        ...
    
    def on_trade(self, trade: FillEvent) -> List[NewOrder]:
        """
        Handle trade executions (fill events).
        
        Args:
            trade: Fill event containing trade execution information
            
        Returns:
            List of new order intents to submit
        """
        ...
    
    def on_timer(self, timestamp: int) -> List[NewOrder]:
        """
        Handle timer events for periodic strategy logic.
        
        Args:
            timestamp: Current simulation timestamp in nanoseconds
            
        Returns:
            List of new order intents to submit
        """
        ...
    
    def get_state(self) -> StrategyState:
        """
        Get current strategy state.
        
        Returns:
            Current strategy execution state
        """
        ...
    
    def get_config(self) -> StrategyConfig:
        """
        Get strategy configuration.
        
        Returns:
            Strategy configuration object
        """
        ...


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Provides common functionality and enforces the Strategy protocol.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.state = StrategyState.INITIALIZED
        self.symbol = config.symbol
        self.position = StrategyPosition()
        self.strategy_id = config.strategy_id
        
        # Strategy state tracking
        self.last_update_time: Optional[int] = None
        self.total_orders: int = 0
        self.total_fills: int = 0
        self.total_pnl: float = 0.0
        
        # Market data storage for strategy logic
        self.price_history: List[float] = []
        self.volume_history: List[int] = []
        self.timestamp_history: List[int] = []
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
    
    @abstractmethod
    def on_bar(self, book_update: MarketDataEvent) -> List[NewOrder]:
        """Handle order book updates."""
        pass
    
    @abstractmethod
    def on_trade(self, trade: FillEvent) -> List[NewOrder]:
        """Handle trade executions."""
        pass
    
    @abstractmethod
    def on_timer(self, timestamp: int) -> List[NewOrder]:
        """Handle timer events."""
        pass
    
    def get_state(self) -> StrategyState:
        """Get current strategy state."""
        return self.state
    
    def get_config(self) -> StrategyConfig:
        """Get strategy configuration."""
        return self.config
    
    def start(self) -> None:
        """Start the strategy."""
        self.state = StrategyState.RUNNING
    
    def stop(self) -> None:
        """Stop the strategy."""
        self.state = StrategyState.STOPPED
    
    def pause(self) -> None:
        """Pause the strategy."""
        self.state = StrategyState.PAUSED
    
    def resume(self) -> None:
        """Resume the strategy."""
        self.state = StrategyState.RUNNING
    
    def is_active(self) -> bool:
        """Check if strategy is active (running)."""
        return self.state == StrategyState.RUNNING
    
    def _create_order_intent(self, side: OrderSide, price: float, quantity: int, 
                           order_type: OrderType = OrderType.LIMIT,
                           time_in_force: TimeInForce = TimeInForce.DAY) -> NewOrder:
        """
        Create a new order intent.
        
        Args:
            side: Order side (BUY/SELL)
            price: Order price
            quantity: Order quantity
            order_type: Order type (LIMIT/MARKET)
            time_in_force: Time in force (DAY/IOC/FOK)
            
        Returns:
            New order intent
        """
        order_id = f"{self.strategy_id}_{self.total_orders + 1}"
        self.total_orders += 1
        
        return NewOrder(
            intent_id=order_id,
            timestamp=self.last_update_time or 0,
            symbol=self.symbol,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
            time_in_force=time_in_force,
            strategy_id=self.strategy_id
        )
    
    def _create_cancel_intent(self, order_id: str) -> CancelOrder:
        """
        Create a cancel order intent.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            Cancel order intent
        """
        return CancelOrder(
            intent_id=f"cancel_{order_id}",
            timestamp=self.last_update_time or 0,
            symbol=self.symbol,
            order_id=order_id,
            strategy_id=self.strategy_id
        )
    
    def _update_market_data(self, book_update: MarketDataEvent) -> None:
        """
        Update internal market data storage.
        
        Args:
            book_update: Market data event
        """
        self.last_update_time = book_update.timestamp.value
        
        # Extract price and volume from market data
        # This is a simplified implementation - in practice, you'd parse the actual market data
        if hasattr(book_update, 'data') and book_update.data:
            price = book_update.data.get('mid_price', 0.0)
            volume = book_update.data.get('volume', 0)
            
            if price > 0:
                self.price_history.append(price)
                self.volume_history.append(volume)
                self.timestamp_history.append(book_update.timestamp.value)
                
                # Keep only recent history (e.g., last 1000 points)
                max_history = 1000
                if len(self.price_history) > max_history:
                    self.price_history = self.price_history[-max_history:]
                    self.volume_history = self.volume_history[-max_history:]
                    self.timestamp_history = self.timestamp_history[-max_history:]
    
    def _get_mid_price(self, book_update: MarketDataEvent) -> Optional[float]:
        """
        Extract mid price from market data event.
        
        Args:
            book_update: Market data event
            
        Returns:
            Mid price if available, None otherwise
        """
        if hasattr(book_update, 'data') and book_update.data:
            return book_update.data.get('mid_price')
        return None
    
    def _get_bid_ask(self, book_update: MarketDataEvent) -> tuple[Optional[float], Optional[float]]:
        """
        Extract bid and ask prices from market data event.
        
        Args:
            book_update: Market data event
            
        Returns:
            Tuple of (bid_price, ask_price) if available, (None, None) otherwise
        """
        if hasattr(book_update, 'data') and book_update.data:
            bid = book_update.data.get('bid_price')
            ask = book_update.data.get('ask_price')
            return bid, ask
        return None, None
    
    def _get_volume(self, book_update: MarketDataEvent) -> int:
        """
        Extract volume from market data event.
        
        Args:
            book_update: Market data event
            
        Returns:
            Volume if available, 0 otherwise
        """
        if hasattr(book_update, 'data') and book_update.data:
            return book_update.data.get('volume', 0)
        return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get strategy statistics.
        
        Returns:
            Dictionary containing strategy statistics
        """
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "state": self.state.value,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_pnl": self.total_pnl,
            "active_orders": len(self.active_orders),
            "price_history_length": len(self.price_history),
            "last_update_time": self.last_update_time
        }


class StrategyError(Exception):
    """Exception raised by strategy implementations."""
    pass


class StrategyValidationError(StrategyError):
    """Exception raised when strategy validation fails."""
    pass