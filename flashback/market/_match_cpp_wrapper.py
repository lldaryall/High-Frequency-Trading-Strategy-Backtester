"""
Python wrapper for the C++ matching engine extension.

This module provides a Python interface to the high-performance C++ matching engine,
maintaining compatibility with the existing Python matching engine interface.
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

# Try to import the C++ extension
try:
    from . import _match_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    _match_cpp = None

from .orders import OrderSide, TimeInForce


class CppOrderSide(IntEnum):
    """C++ order side enumeration."""
    BUY = 1
    SELL = -1


class CppTimeInForce(IntEnum):
    """C++ time in force enumeration."""
    DAY = 0
    IOC = 1
    FOK = 2


class CppOrderStatus(IntEnum):
    """C++ order status enumeration."""
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4


@dataclass
class CppFill:
    """Fill result from C++ matching engine."""
    order_id: str
    price: float
    quantity: int


class CppMatchEngine:
    """
    High-performance C++ matching engine wrapper.
    
    This class provides a Python interface to the C++ matching engine
    while maintaining compatibility with the existing Python interface.
    """
    
    def __init__(self, symbol: str = "DEFAULT"):
        """Initialize the C++ matching engine."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ matching engine extension not available. Run 'make cpp' to build it.")
        
        self.symbol = symbol
        self._engine = _match_cpp.MatchEngine()
        self._order_map: Dict[str, Any] = {}  # Store order metadata
    
    def _convert_side(self, side: OrderSide) -> int:
        """Convert Python OrderSide to C++ side."""
        return CppOrderSide.BUY if side == OrderSide.BUY else CppOrderSide.SELL
    
    def _convert_tif(self, tif: TimeInForce) -> int:
        """Convert Python TimeInForce to C++ TIF."""
        if tif == TimeInForce.DAY:
            return CppTimeInForce.DAY
        elif tif == TimeInForce.IOC:
            return CppTimeInForce.IOC
        elif tif == TimeInForce.FOK:
            return CppTimeInForce.FOK
        else:
            return CppTimeInForce.DAY
    
    def add_order(self, order: Any) -> bool:
        """
        Add an order to the matching engine.
        
        Args:
            order: Order object with order_id, side, price, quantity, tif attributes
            
        Returns:
            bool: True if order was accepted, False if rejected
        """
        try:
            # Convert order attributes to C++ types
            side = self._convert_side(order.side)
            price = order.price if hasattr(order, 'price') else 0.0
            quantity = order.quantity if hasattr(order, 'quantity') else 0
            tif = self._convert_tif(order.tif) if hasattr(order, 'tif') else CppTimeInForce.DAY
            
            # Submit to C++ engine
            success = self._engine.submit_order(
                order.order_id, side, price, quantity, tif
            )
            
            if success:
                # Store order metadata
                self._order_map[order.order_id] = {
                    'order': order,
                    'status': CppOrderStatus.PENDING
                }
            
            return success
            
        except Exception as e:
            print(f"Error adding order {order.order_id}: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if order was cancelled, False if not found
        """
        success = self._engine.cancel_order(order_id)
        
        if success and order_id in self._order_map:
            self._order_map[order_id]['status'] = CppOrderStatus.CANCELLED
        
        return success
    
    def match_order(self, order: Any) -> List[CppFill]:
        """
        Match an incoming order against the book.
        
        Args:
            order: Incoming order to match
            
        Returns:
            List[CppFill]: List of fills generated
        """
        # Add the order first
        if not self.add_order(order):
            return []
        
        # Get fills
        cpp_fills = self._engine.get_fills()
        
        # Convert to Python Fill objects
        fills = []
        for order_id, price, quantity in cpp_fills:
            fills.append(CppFill(order_id, price, quantity))
        
        return fills
    
    def process_tick(self, price: float, size: int, side: OrderSide) -> List[CppFill]:
        """
        Process a market tick.
        
        Args:
            price: Tick price
            size: Tick size
            side: Tick side
            
        Returns:
            List[CppFill]: List of fills generated
        """
        cpp_side = self._convert_side(side)
        self._engine.process_tick(price, size, cpp_side)
        
        # Get fills
        cpp_fills = self._engine.get_fills()
        
        # Convert to Python Fill objects
        fills = []
        for order_id, fill_price, quantity in cpp_fills:
            fills.append(CppFill(order_id, fill_price, quantity))
        
        return fills
    
    def get_fills(self) -> List[CppFill]:
        """
        Get all fills since last call.
        
        Returns:
            List[CppFill]: List of fills
        """
        cpp_fills = self._engine.get_fills()
        
        fills = []
        for order_id, price, quantity in cpp_fills:
            fills.append(CppFill(order_id, price, quantity))
        
        return fills
    
    def get_best_levels(self) -> Dict[float, int]:
        """
        Get best bid/ask levels for debugging.
        
        Returns:
            Dict[float, int]: Price -> quantity mapping
        """
        return dict(self._engine.get_best_levels())
    
    def get_order_count(self) -> int:
        """
        Get current number of active orders.
        
        Returns:
            int: Number of active orders
        """
        return self._engine.get_order_count()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get order book snapshot for debugging.
        
        Returns:
            Dict[str, Any]: Order book snapshot
        """
        levels = self.get_best_levels()
        
        # Separate bid and ask levels
        bid_levels = []
        ask_levels = []
        
        for price, quantity in levels.items():
            if price > 0:  # Valid price
                if len(bid_levels) == 0 or price < bid_levels[0]['price']:
                    ask_levels.append({'price': price, 'quantity': quantity})
                else:
                    bid_levels.append({'price': price, 'quantity': quantity})
        
        # Sort bid levels (descending) and ask levels (ascending)
        bid_levels.sort(key=lambda x: x['price'], reverse=True)
        ask_levels.sort(key=lambda x: x['price'])
        
        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'order_count': self.get_order_count()
        }


def create_cpp_matching_engine(symbol: str = "DEFAULT") -> Optional[CppMatchEngine]:
    """
    Create a C++ matching engine if available.
    
    Args:
        symbol: Symbol for the matching engine
        
    Returns:
        CppMatchEngine or None if C++ extension not available
    """
    if not CPP_AVAILABLE:
        return None
    
    try:
        return CppMatchEngine(symbol)
    except Exception as e:
        print(f"Failed to create C++ matching engine: {e}")
        return None


# Export the main classes
__all__ = [
    'CppMatchEngine',
    'CppFill', 
    'CppOrderSide',
    'CppTimeInForce',
    'CppOrderStatus',
    'create_cpp_matching_engine',
    'CPP_AVAILABLE'
]
