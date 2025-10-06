"""Order and trade dataclasses for the matching engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "DAY"      # Good for the day
    IOC = "IOC"      # Immediate or Cancel
    FOK = "FOK"      # Fill or Kill


@dataclass
class Order:
    """Order dataclass representing a trading order."""
    order_id: str
    timestamp: int  # nanoseconds since epoch
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    time_in_force: TimeInForce
    order_type: OrderType
    remaining_qty: Optional[int] = None
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    status: str = "NEW"  # NEW, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED
    
    def __post_init__(self) -> None:
        """Initialize remaining quantity if not provided."""
        if self.remaining_qty is None:
            self.remaining_qty = self.quantity
    
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY
    
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL
    
    def is_limit(self) -> bool:
        """Check if order is a limit order."""
        return self.order_type == OrderType.LIMIT
    
    def is_market(self) -> bool:
        """Check if order is a market order."""
        return self.order_type == OrderType.MARKET
    
    def is_day_order(self) -> bool:
        """Check if order is a day order."""
        return self.time_in_force == TimeInForce.DAY
    
    def is_ioc(self) -> bool:
        """Check if order is IOC (Immediate or Cancel)."""
        return self.time_in_force == TimeInForce.IOC
    
    def is_fok(self) -> bool:
        """Check if order is FOK (Fill or Kill)."""
        return self.time_in_force == TimeInForce.FOK
    
    def is_active(self) -> bool:
        """Check if order is still active (not filled or cancelled)."""
        return self.status in ["NEW", "PARTIALLY_FILLED"]
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == "FILLED"
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == "PARTIALLY_FILLED"
    
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == "CANCELLED"
    
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.status == "REJECTED"
    
    def get_fill_ratio(self) -> float:
        """Get the fill ratio (filled_qty / total_qty)."""
        if self.quantity == 0:
            return 0.0
        return self.filled_qty / self.quantity
    
    def get_remaining_ratio(self) -> float:
        """Get the remaining ratio (remaining_qty / total_qty)."""
        if self.quantity == 0:
            return 0.0
        return self.remaining_qty / self.quantity
    
    def update_fill(self, fill_qty: int, fill_price: float) -> None:
        """Update order with a fill."""
        if fill_qty > self.remaining_qty:
            raise ValueError(f"Fill quantity {fill_qty} exceeds remaining {self.remaining_qty}")
        
        self.filled_qty += fill_qty
        self.remaining_qty -= fill_qty
        
        # Update average fill price
        if self.filled_qty > 0:
            total_value = self.avg_fill_price * (self.filled_qty - fill_qty) + fill_price * fill_qty
            self.avg_fill_price = total_value / self.filled_qty
        
        # Update status
        if self.remaining_qty == 0:
            self.status = "FILLED"
        else:
            self.status = "PARTIALLY_FILLED"
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status in ["FILLED", "CANCELLED", "REJECTED"]:
            raise ValueError(f"Cannot cancel order with status {self.status}")
        
        self.status = "CANCELLED"
        self.remaining_qty = 0
    
    def reject(self, reason: str = "Unknown") -> None:
        """Reject the order."""
        if self.status in ["FILLED", "CANCELLED", "REJECTED"]:
            raise ValueError(f"Cannot reject order with status {self.status}")
        
        self.status = "REJECTED"
        self.remaining_qty = 0
    
    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "time_in_force": self.time_in_force.value,
            "order_type": self.order_type.value,
            "remaining_qty": self.remaining_qty,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        """Create order from dictionary."""
        return cls(
            order_id=data["order_id"],
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            price=data["price"],
            quantity=data["quantity"],
            time_in_force=TimeInForce(data["time_in_force"]),
            order_type=OrderType(data["order_type"]),
            remaining_qty=data.get("remaining_qty"),
            filled_qty=data.get("filled_qty", 0),
            avg_fill_price=data.get("avg_fill_price", 0.0),
            status=data.get("status", "NEW"),
        )


@dataclass
class Fill:
    """Fill dataclass representing a trade execution."""
    fill_id: str
    order_id: str
    timestamp: int  # nanoseconds since epoch
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    commission: float = 0.0
    maker_taker: str = "UNKNOWN"  # MAKER, TAKER, UNKNOWN
    
    def is_buy(self) -> bool:
        """Check if fill is a buy."""
        return self.side == OrderSide.BUY
    
    def is_sell(self) -> bool:
        """Check if fill is a sell."""
        return self.side == OrderSide.SELL
    
    def is_maker(self) -> bool:
        """Check if fill is from a maker order."""
        return self.maker_taker == "MAKER"
    
    def is_taker(self) -> bool:
        """Check if fill is from a taker order."""
        return self.maker_taker == "TAKER"
    
    def get_notional(self) -> float:
        """Get the notional value of the fill."""
        return self.price * self.quantity
    
    def get_net_value(self) -> float:
        """Get the net value after commission."""
        return self.get_notional() - self.commission
    
    def to_dict(self) -> dict:
        """Convert fill to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "commission": self.commission,
            "maker_taker": self.maker_taker,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Fill':
        """Create fill from dictionary."""
        return cls(
            fill_id=data["fill_id"],
            order_id=data["order_id"],
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            price=data["price"],
            quantity=data["quantity"],
            commission=data.get("commission", 0.0),
            maker_taker=data.get("maker_taker", "UNKNOWN"),
        )


@dataclass
class Cancel:
    """Cancel dataclass representing an order cancellation."""
    cancel_id: str
    order_id: str
    timestamp: int  # nanoseconds since epoch
    symbol: str
    cancelled_qty: int
    reason: str = "USER_REQUEST"
    
    def to_dict(self) -> dict:
        """Convert cancel to dictionary."""
        return {
            "cancel_id": self.cancel_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "cancelled_qty": self.cancelled_qty,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Cancel':
        """Create cancel from dictionary."""
        return cls(
            cancel_id=data["cancel_id"],
            order_id=data["order_id"],
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            cancelled_qty=data["cancelled_qty"],
            reason=data.get("reason", "USER_REQUEST"),
        )


@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book."""
    price: float
    total_qty: int
    order_count: int
    orders: list[Order] = None
    
    def __post_init__(self) -> None:
        """Initialize orders list if not provided."""
        if self.orders is None:
            self.orders = []
    
    def add_order(self, order: Order) -> None:
        """Add an order to this price level."""
        self.orders.append(order)
        self.total_qty += order.remaining_qty
        self.order_count += 1
    
    def remove_order(self, order: Order) -> None:
        """Remove an order from this price level."""
        if order in self.orders:
            self.orders.remove(order)
            self.total_qty -= order.remaining_qty
            self.order_count -= 1
    
    def get_best_order(self) -> Optional[Order]:
        """Get the best (first) order at this price level."""
        if not self.orders:
            return None
        return self.orders[0]
    
    def is_empty(self) -> bool:
        """Check if this price level is empty."""
        return len(self.orders) == 0
    
    def get_depth(self) -> int:
        """Get the number of orders at this price level."""
        return len(self.orders)


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    timestamp: int
    symbol: str
    bids: list[tuple[float, int]]  # (price, quantity) tuples
    asks: list[tuple[float, int]]  # (price, quantity) tuples
    
    def get_best_bid(self) -> Optional[tuple[float, int]]:
        """Get the best bid (highest price)."""
        if not self.bids:
            return None
        return self.bids[0]
    
    def get_best_ask(self) -> Optional[tuple[float, int]]:
        """Get the best ask (lowest price)."""
        if not self.asks:
            return None
        return self.asks[0]
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask[0] - best_bid[0]
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid[0] + best_ask[0]) / 2.0
    
    def to_dict(self) -> dict:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
        }
