"""Price-time priority L1 matching engine with partial fill support."""

import heapq
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Iterator
import uuid
from dataclasses import dataclass

from .orders import Order, Fill, Cancel, OrderSide, OrderType, TimeInForce, OrderBookLevel, OrderBookSnapshot

# Try to import C++ matching engine
try:
    from . import _match_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    _match_cpp = None


class CppMatchingEngineWrapper:
    """Wrapper for C++ matching engine to match Python interface."""
    
    def __init__(self, symbol: str):
        """Initialize the C++ matching engine wrapper."""
        if not HAS_CPP:
            raise ImportError("C++ matching engine not available")
        
        self.symbol = symbol
        self._engine = _match_cpp.MatchEngine()
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._cancels: List[Cancel] = []
        
        # Statistics
        self.total_volume: int = 0
        self.total_trades: int = 0
        self.last_trade_price: Optional[float] = None
    
    def add_order(self, order: Order) -> List[Fill]:
        """Add an order to the matching engine."""
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol} does not match engine symbol {self.symbol}")
        
        # Store the order
        self._orders[order.order_id] = order
        
        # Convert to C++ format
        side = 1 if order.is_buy() else -1
        tif = 0 if order.time_in_force == TimeInForce.DAY else (1 if order.time_in_force == TimeInForce.IOC else 2)
        
        # Submit to C++ engine
        success = self._engine.submit_order(
            order.order_id, side, order.price, order.quantity, tif
        )
        
        if not success:
            order.cancel()
            return []
        
        # Get fills from C++ engine
        cpp_fills = self._engine.get_fills()
        fills = []
        
        for order_id, price, quantity in cpp_fills:
            # Create Python Fill objects
            taker_fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order_id,
                timestamp=order.timestamp,
                symbol=self.symbol,
                side=order.side,
                price=price,
                quantity=quantity,
                maker_taker="TAKER"
            )
            fills.append(taker_fill)
            self._fills.append(taker_fill)
            
            # Update the order with the fill
            if order_id in self._orders:
                self._orders[order_id].update_fill(quantity, price)
            
            # Update statistics
            self.total_volume += quantity
            self.total_trades += 1
            self.last_trade_price = price
        
        return fills
    
    def cancel_order(self, order_id: str, timestamp: int) -> Optional[Cancel]:
        """Cancel an order."""
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        
        if not order.is_active():
            return None
        
        # Cancel in C++ engine
        success = self._engine.cancel_order(order_id)
        
        if not success:
            return None
        
        # Create cancel record
        cancel = Cancel(
            cancel_id=str(uuid.uuid4()),
            order_id=order_id,
            timestamp=timestamp,
            symbol=self.symbol,
            cancelled_qty=order.remaining_qty,
            reason="USER_REQUEST"
        )
        
        # Cancel the order
        order.cancel()
        self._cancels.append(cancel)
        return cancel
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        levels = self._engine.get_best_levels()
        bid_prices = [price for price in levels.keys() if price > 0]
        return max(bid_prices) if bid_prices else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        levels = self._engine.get_best_levels()
        ask_prices = [price for price in levels.keys() if price > 0]
        return min(ask_prices) if ask_prices else None
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def get_snapshot(self, timestamp: int) -> OrderBookSnapshot:
        """Get a snapshot of the order book."""
        levels = self._engine.get_best_levels()
        
        bid_levels = []
        ask_levels = []
        
        for price, quantity in levels.items():
            if price > 0:
                if len(bid_levels) == 0 or price < bid_levels[0][0]:
                    ask_levels.append((price, quantity))
                else:
                    bid_levels.append((price, quantity))
        
        # Sort bid levels (descending) and ask levels (ascending)
        bid_levels.sort(key=lambda x: x[0], reverse=True)
        ask_levels.sort(key=lambda x: x[0])
        
        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=self.symbol,
            bids=bid_levels[:10],  # Top 10 levels
            asks=ask_levels[:10]
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self._orders.get(order_id)
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self._fills.copy()
    
    def get_cancels(self) -> List[Cancel]:
        """Get all cancels."""
        return self._cancels.copy()
    
    def get_statistics(self) -> dict:
        """Get engine statistics."""
        return {
            "symbol": self.symbol,
            "total_orders": len(self._orders),
            "active_orders": len([o for o in self._orders.values() if o.is_active()]),
            "total_fills": len(self._fills),
            "total_cancels": len(self._cancels),
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
            "last_trade_price": self.last_trade_price,
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
        }


class MatchingEngine:
    """Unified matching engine that automatically selects C++ or Python implementation."""
    
    def __init__(self, symbol: str, use_cpp: bool = HAS_CPP):
        """
        Initialize the matching engine for a symbol.
        
        Args:
            symbol: Trading symbol
            use_cpp: Whether to use C++ implementation (if available)
        """
        self.symbol = symbol
        self.use_cpp = use_cpp and HAS_CPP
        
        if self.use_cpp:
            self.engine = CppMatchingEngineWrapper(symbol)
        else:
            self.engine = PythonMatchingEngine(symbol)
    
    def add_order(self, order: Order) -> List[Fill]:
        """Add an order to the matching engine."""
        return self.engine.add_order(order)
    
    def cancel_order(self, order_id: str, timestamp: int) -> Optional[Cancel]:
        """Cancel an order."""
        return self.engine.cancel_order(order_id, timestamp)
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self.engine.get_best_bid()
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self.engine.get_best_ask()
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        return self.engine.get_spread()
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price."""
        return self.engine.get_mid_price()
    
    def get_snapshot(self, timestamp: int) -> OrderBookSnapshot:
        """Get a snapshot of the order book."""
        return self.engine.get_snapshot(timestamp)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self.engine.get_order(order_id)
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self.engine.get_fills()
    
    def get_cancels(self) -> List[Cancel]:
        """Get all cancels."""
        return self.engine.get_cancels()
    
    def get_statistics(self) -> dict:
        """Get engine statistics."""
        return self.engine.get_statistics()


class PythonMatchingEngine:
    """Pure Python matching engine implementation."""
    
    def __init__(self, symbol: str):
        """
        Initialize the matching engine for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        
        # Order books: price -> OrderBookLevel
        # Bids: highest price first (descending)
        # Asks: lowest price first (ascending)
        self.bids: Dict[float, OrderBookLevel] = {}
        self.asks: Dict[float, OrderBookLevel] = {}
        
        # Price queues for efficient access
        self.bid_prices: List[float] = []  # Max heap (negated for min heap behavior)
        self.ask_prices: List[float] = []  # Min heap
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.cancels: List[Cancel] = []
        
        # Statistics
        self.total_volume: int = 0
        self.total_trades: int = 0
        self.last_trade_price: Optional[float] = None
        
    def add_order(self, order: Order) -> List[Fill]:
        """
        Add an order to the matching engine.
        
        Args:
            order: Order to add
            
        Returns:
            List of fills generated
        """
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol} does not match engine symbol {self.symbol}")
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Generate fills
        fills = self._match_order(order)
        
        # Add remaining quantity to book if not filled
        if order.remaining_qty > 0 and order.is_limit():
            self._add_to_book(order)
        
        return fills
    
    def cancel_order(self, order_id: str, timestamp: int) -> Optional[Cancel]:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            timestamp: Cancellation timestamp
            
        Returns:
            Cancel object if successful, None if order not found
        """
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        if not order.is_active():
            return None
        
        # Create cancel record
        cancel = Cancel(
            cancel_id=str(uuid.uuid4()),
            order_id=order_id,
            timestamp=timestamp,
            symbol=self.symbol,
            cancelled_qty=order.remaining_qty,
            reason="USER_REQUEST"
        )
        
        # Cancel the order
        order.cancel()
        
        # Remove from book if it's there
        if order.is_limit():
            self._remove_from_book(order)
        
        self.cancels.append(cancel)
        return cancel
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        if not self.bid_prices:
            return None
        return -self.bid_prices[0]  # Negate because we use min heap for max behavior
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        if not self.ask_prices:
            return None
        return self.ask_prices[0]
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def get_snapshot(self, timestamp: int) -> OrderBookSnapshot:
        """Get a snapshot of the order book."""
        # Get top 10 levels for each side
        bid_levels = []
        ask_levels = []
        
        # Bids (highest first)
        for price_neg in sorted(self.bid_prices, reverse=True)[:10]:
            actual_price = -price_neg
            if actual_price in self.bids:
                level = self.bids[actual_price]
                bid_levels.append((actual_price, level.total_qty))
        
        # Sort bids by price descending (highest first)
        bid_levels.sort(key=lambda x: x[0], reverse=True)
        
        # Asks (lowest first)
        for price in sorted(self.ask_prices)[:10]:
            if price in self.asks:
                level = self.asks[price]
                ask_levels.append((price, level.total_qty))
        
        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=self.symbol,
            bids=bid_levels,
            asks=ask_levels
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self.orders.get(order_id)
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self.fills.copy()
    
    def get_cancels(self) -> List[Cancel]:
        """Get all cancels."""
        return self.cancels.copy()
    
    def get_statistics(self) -> dict:
        """Get engine statistics."""
        return {
            "symbol": self.symbol,
            "total_orders": len(self.orders),
            "active_orders": len([o for o in self.orders.values() if o.is_active()]),
            "total_fills": len(self.fills),
            "total_cancels": len(self.cancels),
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
            "last_trade_price": self.last_trade_price,
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
        }
    
    def _match_order(self, order: Order) -> List[Fill]:
        """Match an order against the book."""
        fills = []
        
        if order.is_buy():
            fills = self._match_buy_order(order)
        else:
            fills = self._match_sell_order(order)
        
        # Update statistics
        for fill in fills:
            # Only count each fill once (not both maker and taker)
            if fill.maker_taker == "TAKER":
                self.total_volume += fill.quantity
                self.total_trades += 1
                self.last_trade_price = fill.price
        
        return fills
    
    def _match_buy_order(self, order: Order) -> List[Fill]:
        """Match a buy order against ask side."""
        fills = []
        
        # For market orders, use worst possible price
        if order.is_market():
            order.price = float('inf')
        
        # For FOK orders, check if entire quantity can be filled first
        if order.is_fok():
            available_qty = self._get_available_ask_quantity(order.price)
            if available_qty < order.quantity:
                # Can't fill completely, cancel entire order
                order.cancel()
                return fills
        
        # Match against asks (ascending price order)
        ask_prices = sorted(self.ask_prices)
        
        for ask_price in ask_prices:
            if order.remaining_qty <= 0:
                break
            
            if order.is_limit() and order.price < ask_price:
                break  # Can't match at this price
            
            if ask_price not in self.asks:
                continue
            
            level = self.asks[ask_price]
            if level.is_empty():
                continue
            
            # Match against orders at this price level (FIFO)
            orders_to_remove = []
            
            for resting_order in level.orders[:]:  # Copy to avoid modification during iteration
                if order.remaining_qty <= 0:
                    break
                
                if not resting_order.is_active():
                    orders_to_remove.append(resting_order)
                    continue
                
                # Calculate fill quantity
                fill_qty = min(order.remaining_qty, resting_order.remaining_qty)
                
                # Create fills
                taker_fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    timestamp=order.timestamp,
                    symbol=self.symbol,
                    side=order.side,
                    price=ask_price,
                    quantity=fill_qty,
                    maker_taker="TAKER"
                )
                
                maker_fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=resting_order.order_id,
                    timestamp=order.timestamp,
                    symbol=self.symbol,
                    side=resting_order.side,
                    price=ask_price,
                    quantity=fill_qty,
                    maker_taker="MAKER"
                )
                
                fills.extend([taker_fill, maker_fill])
                self.fills.extend([taker_fill, maker_fill])
                
                # Update orders
                order.update_fill(fill_qty, ask_price)
                resting_order.update_fill(fill_qty, ask_price)
                
                # Remove filled orders
                if resting_order.is_filled():
                    orders_to_remove.append(resting_order)
                
                # Handle TIF rules
                if order.is_ioc() and order.remaining_qty > 0:
                    # IOC: cancel remaining quantity
                    order.cancel()
                    break
            
            # Remove filled orders from level
            for filled_order in orders_to_remove:
                level.remove_order(filled_order)
            
            # Remove empty levels
            if level.is_empty():
                self._remove_price_level(ask_price, is_ask=True)
        
        return fills
    
    def _get_available_ask_quantity(self, max_price: float) -> int:
        """Get total available quantity at or below max_price."""
        total_qty = 0
        ask_prices = sorted(self.ask_prices)
        
        for ask_price in ask_prices:
            if ask_price > max_price:
                break
            if ask_price in self.asks:
                level = self.asks[ask_price]
                total_qty += level.total_qty
        
        return total_qty
    
    def _get_available_bid_quantity(self, min_price: float) -> int:
        """Get total available quantity at or above min_price."""
        total_qty = 0
        bid_prices = sorted(self.bid_prices, reverse=True)
        
        for bid_price_neg in bid_prices:
            bid_price = -bid_price_neg
            if bid_price < min_price:
                break
            if bid_price in self.bids:
                level = self.bids[bid_price]
                total_qty += level.total_qty
        
        return total_qty
    
    def _match_sell_order(self, order: Order) -> List[Fill]:
        """Match a sell order against bid side."""
        fills = []
        
        # For market orders, use worst possible price
        if order.is_market():
            order.price = 0.0
        
        # For FOK orders, check if entire quantity can be filled first
        if order.is_fok():
            available_qty = self._get_available_bid_quantity(order.price)
            if available_qty < order.quantity:
                # Can't fill completely, cancel entire order
                order.cancel()
                return fills
        
        # Match against bids (descending price order)
        bid_prices = sorted(self.bid_prices, reverse=True)
        
        for bid_price_neg in bid_prices:
            bid_price = -bid_price_neg  # Convert back from negated price
            
            if order.remaining_qty <= 0:
                break
            
            if order.is_limit() and order.price > bid_price:
                break  # Can't match at this price
            
            if bid_price not in self.bids:
                continue
            
            level = self.bids[bid_price]
            if level.is_empty():
                continue
            
            # Match against orders at this price level (FIFO)
            orders_to_remove = []
            
            for resting_order in level.orders[:]:  # Copy to avoid modification during iteration
                if order.remaining_qty <= 0:
                    break
                
                if not resting_order.is_active():
                    orders_to_remove.append(resting_order)
                    continue
                
                # Calculate fill quantity
                fill_qty = min(order.remaining_qty, resting_order.remaining_qty)
                
                # Create fills
                taker_fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    timestamp=order.timestamp,
                    symbol=self.symbol,
                    side=order.side,
                    price=bid_price,
                    quantity=fill_qty,
                    maker_taker="TAKER"
                )
                
                maker_fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=resting_order.order_id,
                    timestamp=order.timestamp,
                    symbol=self.symbol,
                    side=resting_order.side,
                    price=bid_price,
                    quantity=fill_qty,
                    maker_taker="MAKER"
                )
                
                fills.extend([taker_fill, maker_fill])
                self.fills.extend([taker_fill, maker_fill])
                
                # Update orders
                order.update_fill(fill_qty, bid_price)
                resting_order.update_fill(fill_qty, bid_price)
                
                # Remove filled orders
                if resting_order.is_filled():
                    orders_to_remove.append(resting_order)
                
                # Handle TIF rules
                if order.is_ioc() and order.remaining_qty > 0:
                    # IOC: cancel remaining quantity
                    order.cancel()
                    break
            
            # Remove filled orders from level
            for filled_order in orders_to_remove:
                level.remove_order(filled_order)
            
            # Remove empty levels
            if level.is_empty():
                self._remove_price_level(bid_price, is_ask=False)
        
        return fills
    
    def _add_to_book(self, order: Order) -> None:
        """Add a limit order to the book."""
        price = order.price
        
        if order.is_buy():
            # Add to bids
            if price not in self.bids:
                self.bids[price] = OrderBookLevel(price=price, total_qty=0, order_count=0)
                heapq.heappush(self.bid_prices, -price)  # Negate for max heap behavior
            
            self.bids[price].add_order(order)
        else:
            # Add to asks
            if price not in self.asks:
                self.asks[price] = OrderBookLevel(price=price, total_qty=0, order_count=0)
                heapq.heappush(self.ask_prices, price)
            
            self.asks[price].add_order(order)
    
    def _remove_from_book(self, order: Order) -> None:
        """Remove a limit order from the book."""
        price = order.price
        
        if order.is_buy() and price in self.bids:
            self.bids[price].remove_order(order)
            if self.bids[price].is_empty():
                self._remove_price_level(price, is_ask=False)
        elif order.is_sell() and price in self.asks:
            self.asks[price].remove_order(order)
            if self.asks[price].is_empty():
                self._remove_price_level(price, is_ask=True)
    
    def _remove_price_level(self, price: float, is_ask: bool) -> None:
        """Remove an empty price level."""
        if is_ask:
            if price in self.asks:
                del self.asks[price]
            # Remove from heap (inefficient but simple)
            if price in self.ask_prices:
                self.ask_prices.remove(price)
                heapq.heapify(self.ask_prices)
        else:
            if price in self.bids:
                del self.bids[price]
            # Remove from heap (inefficient but simple)
            neg_price = -price
            if neg_price in self.bid_prices:
                self.bid_prices.remove(neg_price)
                heapq.heapify(self.bid_prices)
