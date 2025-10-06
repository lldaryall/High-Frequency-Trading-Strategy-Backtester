"""
Order routing and execution management.

This module handles strategy intents, latency scheduling, order routing to matching engines,
and maintains a comprehensive blotter for tracking all order states and fills.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import pandas as pd

from ..core.events import Event, OrderEvent, FillEvent, CancelEvent, RejectEvent, EventType
from ..market.orders import Order, Fill, Cancel, OrderSide, OrderType, TimeInForce
from ..market.book import MatchingEngine
from ..market.fees import FeeModel, create_standard_fee_model
from ..market.latency import LatencyModel, create_standard_latency_model


class StrategyIntentType(Enum):
    """Types of strategy intents."""
    NEW_ORDER = "new_order"
    CANCEL_ORDER = "cancel_order"


@dataclass
class StrategyIntent:
    """Base class for strategy intents."""
    intent_id: str
    timestamp: int
    symbol: str
    intent_type: StrategyIntentType = StrategyIntentType.NEW_ORDER


@dataclass
class NewOrder(StrategyIntent):
    """Strategy intent to place a new order."""
    side: OrderSide
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent_type: StrategyIntentType = field(default=StrategyIntentType.NEW_ORDER, init=False)


@dataclass
class CancelOrder(StrategyIntent):
    """Strategy intent to cancel an existing order."""
    order_id: str
    strategy_id: str = ""
    intent_type: StrategyIntentType = field(default=StrategyIntentType.CANCEL_ORDER, init=False)


@dataclass
class OrderState:
    """Tracks the state of an order through its lifecycle."""
    order: Order
    strategy_id: str
    created_at: int
    submitted_at: Optional[int] = None
    filled_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    rejected_at: Optional[int] = None
    fills: List[Fill] = field(default_factory=list)
    total_filled_qty: int = 0
    total_filled_value: float = 0.0
    total_commission: float = 0.0
    average_fill_price: float = 0.0
    vwap: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.order.is_active()
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.order.is_filled()
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.order.is_partially_filled()
    
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.order.is_cancelled()
    
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.order.status == "REJECTED"
    
    def add_fill(self, fill: Fill, commission: float = 0.0) -> None:
        """Add a fill to the order state."""
        self.fills.append(fill)
        self.total_filled_qty += fill.quantity
        self.total_filled_value += fill.quantity * fill.price
        self.total_commission += commission
        
        # Update order status
        if self.total_filled_qty >= self.order.quantity:
            self.order.status = "FILLED"
            self.filled_at = fill.timestamp
        elif self.total_filled_qty > 0:
            self.order.status = "PARTIALLY_FILLED"
        
        # Update average fill price and VWAP
        if self.total_filled_qty > 0:
            self.average_fill_price = self.total_filled_value / self.total_filled_qty
            self.vwap = self.total_filled_value / self.total_filled_qty
    
    def get_remaining_qty(self) -> int:
        """Get remaining quantity to be filled."""
        return self.order.quantity - self.total_filled_qty
    
    def get_fill_ratio(self) -> float:
        """Get fill ratio (0.0 to 1.0)."""
        if self.order.quantity == 0:
            return 0.0
        return self.total_filled_qty / self.order.quantity


class Blotter:
    """Maintains a comprehensive record of all orders and fills."""
    
    def __init__(self):
        """Initialize the blotter."""
        self.orders: Dict[str, OrderState] = {}
        self.fills: List[Fill] = []
        self.cancels: List[Cancel] = []
        self.rejects: List[Dict[str, Any]] = []
        
        # Indexes for efficient lookups
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.fills_by_order: Dict[str, List[Fill]] = defaultdict(list)
        self.fills_by_strategy: Dict[str, List[Fill]] = defaultdict(list)
    
    def add_order(self, order_state: OrderState) -> None:
        """Add an order to the blotter."""
        order_id = order_state.order.order_id
        self.orders[order_id] = order_state
        
        # Update indexes
        if order_state.strategy_id:
            self.orders_by_strategy[order_state.strategy_id].append(order_id)
        self.orders_by_symbol[order_state.order.symbol].append(order_id)
    
    def add_fill(self, fill: Fill, order_id: str, strategy_id: str = "") -> None:
        """Add a fill to the blotter."""
        self.fills.append(fill)
        self.fills_by_order[order_id].append(fill)
        if strategy_id:
            self.fills_by_strategy[strategy_id].append(fill)
    
    def add_cancel(self, cancel: Cancel) -> None:
        """Add a cancel to the blotter."""
        self.cancels.append(cancel)
    
    def add_reject(self, reject_info: Dict[str, Any]) -> None:
        """Add a reject to the blotter."""
        self.rejects.append(reject_info)
    
    def get_order(self, order_id: str) -> Optional[OrderState]:
        """Get order state by ID."""
        return self.orders.get(order_id)
    
    def get_fills(self, order_id: Optional[str] = None, strategy_id: Optional[str] = None) -> List[Fill]:
        """Get fills, optionally filtered by order ID or strategy ID."""
        if order_id:
            return self.fills_by_order.get(order_id, [])
        elif strategy_id:
            return self.fills_by_strategy.get(strategy_id, [])
        else:
            return self.fills.copy()
    
    def get_open_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[OrderState]:
        """Get open orders, optionally filtered by symbol or strategy ID."""
        open_orders = []
        
        for order_state in self.orders.values():
            if not order_state.is_active():
                continue
            
            if symbol and order_state.order.symbol != symbol:
                continue
            
            if strategy_id and order_state.strategy_id != strategy_id:
                continue
            
            open_orders.append(order_state)
        
        return open_orders
    
    def get_filled_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[OrderState]:
        """Get filled orders, optionally filtered by symbol or strategy ID."""
        filled_orders = []
        
        for order_state in self.orders.values():
            if not order_state.is_filled():
                continue
            
            if symbol and order_state.order.symbol != symbol:
                continue
            
            if strategy_id and order_state.strategy_id != strategy_id:
                continue
            
            filled_orders.append(order_state)
        
        return filled_orders
    
    def get_cancelled_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[OrderState]:
        """Get cancelled orders, optionally filtered by symbol or strategy ID."""
        cancelled_orders = []
        
        for order_state in self.orders.values():
            if not order_state.is_cancelled():
                continue
            
            if symbol and order_state.order.symbol != symbol:
                continue
            
            if strategy_id and order_state.strategy_id != strategy_id:
                continue
            
            cancelled_orders.append(order_state)
        
        return cancelled_orders
    
    def get_rejected_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[OrderState]:
        """Get rejected orders, optionally filtered by symbol or strategy ID."""
        rejected_orders = []
        
        for order_state in self.orders.values():
            if not order_state.is_rejected():
                continue
            
            if symbol and order_state.order.symbol != symbol:
                continue
            
            if strategy_id and order_state.strategy_id != strategy_id:
                continue
            
            rejected_orders.append(order_state)
        
        return rejected_orders
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blotter statistics."""
        total_orders = len(self.orders)
        open_orders = len(self.get_open_orders())
        filled_orders = len(self.get_filled_orders())
        cancelled_orders = len(self.get_cancelled_orders())
        rejected_orders = len(self.get_rejected_orders())
        
        total_fills = len(self.fills)
        total_volume = sum(fill.quantity for fill in self.fills)
        total_notional = sum(fill.quantity * fill.price for fill in self.fills)
        total_commission = sum(order_state.total_commission for order_state in self.orders.values())
        
        return {
            "total_orders": total_orders,
            "open_orders": open_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "rejected_orders": rejected_orders,
            "total_fills": total_fills,
            "total_volume": total_volume,
            "total_notional": total_notional,
            "total_commission": total_commission,
        }


class OrderRouter:
    """Routes strategy intents to matching engines with latency and fee modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the order router.
        
        Args:
            config: Router configuration
        """
        self.config = config
        self.blotter = Blotter()
        self.fee_model = create_standard_fee_model()
        self.latency_model = create_standard_latency_model()
        
        # Pending orders waiting for latency delay
        self.pending_orders: deque = deque()
        
        # Order books for each symbol
        self.order_books: Dict[str, MatchingEngine] = {}
        
        # Event callbacks
        self.on_fill_callback: Optional[callable] = None
        self.on_cancel_callback: Optional[callable] = None
        self.on_reject_callback: Optional[callable] = None
    
    def set_fee_model(self, fee_model: FeeModel) -> None:
        """Set the fee model."""
        self.fee_model = fee_model
    
    def set_latency_model(self, latency_model: LatencyModel) -> None:
        """Set the latency model."""
        self.latency_model = latency_model
    
    def set_callbacks(self, on_fill: Optional[callable] = None, 
                     on_cancel: Optional[callable] = None,
                     on_reject: Optional[callable] = None) -> None:
        """Set event callbacks."""
        self.on_fill_callback = on_fill
        self.on_cancel_callback = on_cancel
        self.on_reject_callback = on_reject
    
    def get_order_book(self, symbol: str) -> MatchingEngine:
        """Get or create order book for symbol."""
        if symbol not in self.order_books:
            self.order_books[symbol] = MatchingEngine(symbol)
        return self.order_books[symbol]
    
    def process_intent(self, intent: StrategyIntent, current_time: int) -> None:
        """
        Process a strategy intent.
        
        Args:
            intent: Strategy intent to process
            current_time: Current simulation time in nanoseconds
        """
        if isinstance(intent, NewOrder):
            self._process_new_order(intent, current_time)
        elif isinstance(intent, CancelOrder):
            self._process_cancel_order(intent, current_time)
        else:
            raise ValueError(f"Unknown intent type: {type(intent)}")
    
    def _process_new_order(self, intent: NewOrder, current_time: int) -> None:
        """Process a new order intent."""
        # Create order
        order = Order(
            order_id=intent.intent_id,
            timestamp=current_time,
            symbol=intent.symbol,
            side=intent.side,
            price=intent.price,
            quantity=intent.quantity,
            time_in_force=intent.time_in_force,
            order_type=intent.order_type
        )
        
        # Create order state
        order_state = OrderState(
            order=order,
            strategy_id=intent.strategy_id,
            created_at=current_time,
            metadata=intent.metadata
        )
        
        # Add to blotter
        self.blotter.add_order(order_state)
        
        # Calculate latency
        latency_ns = self.latency_model.calculate_latency(order, LatencyType.SUBMISSION)
        scheduled_time = current_time + latency_ns
        
        # Schedule order submission
        self.pending_orders.append((scheduled_time, order_state))
        
        # Create order book for symbol if it doesn't exist
        if intent.symbol not in self.order_books:
            self.order_books[intent.symbol] = MatchingEngine(intent.symbol)
    
    def _process_cancel_order(self, intent: CancelOrder, current_time: int) -> None:
        """Process a cancel order intent."""
        order_state = self.blotter.get_order(intent.order_id)
        if not order_state:
            # Order not found
            reject_info = {
                "order_id": intent.order_id,
                "reason": "Order not found",
                "timestamp": current_time
            }
            self.blotter.add_reject(reject_info)
            if self.on_reject_callback:
                self.on_reject_callback(reject_info)
            return
        
        if not order_state.is_active():
            # Order already filled or cancelled
            reject_info = {
                "order_id": intent.order_id,
                "reason": "Order not active",
                "timestamp": current_time
            }
            self.blotter.add_reject(reject_info)
            if self.on_reject_callback:
                self.on_reject_callback(reject_info)
            return
        
        # Calculate latency for cancel
        latency_ns = self.latency_model.calculate_latency(order_state.order, LatencyType.CANCELLATION)
        scheduled_time = current_time + latency_ns
        
        # Schedule cancel
        self.pending_orders.append((scheduled_time, "cancel", order_state))
    
    def process_pending_orders(self, current_time: int) -> List[Event]:
        """
        Process pending orders that are ready for execution.
        
        Args:
            current_time: Current simulation time in nanoseconds
            
        Returns:
            List of events generated
        """
        events = []
        
        # Process orders in chronological order
        while self.pending_orders and self.pending_orders[0][0] <= current_time:
            scheduled_time, *data = self.pending_orders.popleft()
            
            if len(data) == 1 and isinstance(data[0], OrderState):
                # New order submission
                order_state = data[0]
                events.extend(self._submit_order(order_state, current_time))
            elif len(data) == 2 and data[0] == "cancel":
                # Cancel order
                order_state = data[1]
                events.extend(self._cancel_order(order_state, current_time))
        
        return events
    
    def _submit_order(self, order_state: OrderState, current_time: int) -> List[Event]:
        """Submit an order to the matching engine."""
        events = []
        order = order_state.order
        
        # Update order state
        order_state.submitted_at = current_time
        
        # Get order book
        order_book = self.get_order_book(order.symbol)
        
        # Submit to matching engine
        fills = order_book.add_order(order)
        
        # Process fills
        for fill in fills:
            # Calculate commission
            commission = self.fee_model.calculate_fee(fill)
            
            # Update order state
            order_state.add_fill(fill, commission)
            
            # Add to blotter
            self.blotter.add_fill(fill, order.order_id, order_state.strategy_id)
            
            # Create fill event
            fill_event = FillEvent(
                timestamp=pd.Timestamp(current_time, unit='ns'),
                event_type=EventType.FILL,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=fill.quantity,
                price=fill.price,
                commission=commission
            )
            events.append(fill_event)
            
            # Call callback
            if self.on_fill_callback:
                self.on_fill_callback(fill_event)
        
        
        # Update order status
        if order.is_filled():
            order_state.filled_at = current_time
        elif order.is_partially_filled():
            pass  # Still active
        elif order.is_cancelled():
            order_state.cancelled_at = current_time
        
        return events
    
    def _cancel_order(self, order_state: OrderState, current_time: int) -> List[Event]:
        """Cancel an order."""
        events = []
        order = order_state.order
        
        # Get order book
        order_book = self.get_order_book(order.symbol)
        
        # Cancel order
        success = order_book.cancel_order(order.order_id, current_time)
        
        if success:
            # Update order state
            order_state.cancelled_at = current_time
            
            # Create cancel event
            cancel_event = CancelEvent(
                timestamp=pd.Timestamp(current_time, unit='ns'),
                event_type=EventType.CANCEL,
                order_id=order.order_id,
                symbol=order.symbol,
                reason="Strategy request"
            )
            events.append(cancel_event)
            
            # Call callback
            if self.on_cancel_callback:
                self.on_cancel_callback(cancel_event)
        else:
            # Order not found or already filled
            reject_info = {
                "order_id": order.order_id,
                "reason": "Order not found or already filled",
                "timestamp": current_time
            }
            self.blotter.add_reject(reject_info)
            
            # Create reject event
            reject_event = RejectEvent(
                timestamp=pd.Timestamp(current_time, unit='ns'),
                event_type=EventType.REJECT,
                order_id=order.order_id,
                symbol=order.symbol,
                reason="Order not found or already filled"
            )
            events.append(reject_event)
            
            # Call callback
            if self.on_reject_callback:
                self.on_reject_callback(reject_event)
        
        return events
    
    def get_blotter(self) -> Blotter:
        """Get the blotter."""
        return self.blotter
    
    def get_fills(self, order_id: Optional[str] = None, strategy_id: Optional[str] = None) -> List[Fill]:
        """Get fills from the blotter."""
        return self.blotter.get_fills(order_id, strategy_id)
    
    def get_open_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[OrderState]:
        """Get open orders from the blotter."""
        return self.blotter.get_open_orders(symbol, strategy_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = self.blotter.get_statistics()
        stats["pending_orders"] = len(self.pending_orders)
        stats["symbols"] = list(self.order_books.keys())
        return stats


# Import LatencyType for the router
from ..market.latency import LatencyType