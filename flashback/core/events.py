"""Event system for the backtesting engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import pandas as pd


class EventType(Enum):
    """Types of events in the system."""
    MARKET_DATA = "market_data"
    ORDER = "order"
    FILL = "fill"
    CANCEL = "cancel"
    REJECT = "reject"
    TIMER = "timer"


@dataclass
class Event(ABC):
    """Base class for all events."""
    timestamp: pd.Timestamp
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        pass


@dataclass
class MarketDataEvent(Event):
    """Market data event containing tick or order book data."""
    symbol: str
    side: str  # 'B' for bid, 'S' for ask
    price: float
    size: int
    event_type_str: str  # 'TICK', 'L2_UPDATE', etc.
    event_type: EventType = EventType.MARKET_DATA
    data: Optional[Dict[str, Any]] = None  # Additional market data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "event_type_str": self.event_type_str,
        }


@dataclass
class OrderEvent(Event):
    """Order event for strategy-generated orders."""
    order_id: str
    symbol: str
    side: str  # 'B' for buy, 'S' for sell
    order_type: str  # 'LIMIT', 'MARKET', 'IOC', 'FOK'
    quantity: int
    event_type: EventType = EventType.ORDER
    price: Optional[float] = None
    time_in_force: str = "DAY"  # 'DAY', 'IOC', 'FOK'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "time_in_force": self.time_in_force,
        }


@dataclass
class FillEvent(Event):
    """Fill event when an order is executed."""
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    event_type: EventType = EventType.FILL
    commission: float = 0.0
    latency_us: int = 0  # Latency in microseconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "latency_us": self.latency_us,
        }


@dataclass
class CancelEvent(Event):
    """Cancel event for order cancellations."""
    order_id: str
    symbol: str
    event_type: EventType = EventType.CANCEL
    reason: str = "USER_REQUEST"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "reason": self.reason,
        }


@dataclass
class RejectEvent(Event):
    """Reject event for order rejections."""
    order_id: str
    symbol: str
    reason: str
    original_order: OrderEvent
    event_type: EventType = EventType.REJECT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "reason": self.reason,
            "original_order": self.original_order.to_dict(),
        }


@dataclass
class TimerEvent(Event):
    """Timer event for scheduled tasks."""
    timer_id: str
    callback: str
    event_type: EventType = EventType.TIMER
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "timer_id": self.timer_id,
            "callback": self.callback,
            "data": self.data,
        }
