# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

"""
High-performance Cython implementation of order matching engine.

This module provides optimized order matching algorithms for the Flashback HFT
backtesting engine, with identical results to the Python implementation.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython

# Define numpy types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

# Order side constants
cdef int BUY = 1
cdef int SELL = -1

# Order type constants  
cdef int LIMIT = 0
cdef int MARKET = 1

# Time in force constants
cdef int DAY = 0
cdef int IOC = 1
cdef int FOK = 2

# Order status constants
cdef int NEW = 0
cdef int PARTIALLY_FILLED = 1
cdef int FILLED = 2
cdef int CANCELLED = 3
cdef int REJECTED = 4


cdef struct Order:
    """C struct for order representation."""
    ITYPE_t order_id
    ITYPE_t timestamp
    int side
    DTYPE_t price
    ITYPE_t quantity
    ITYPE_t remaining_qty
    ITYPE_t filled_qty
    DTYPE_t avg_fill_price
    int time_in_force
    int order_type
    int status


cdef struct Fill:
    """C struct for fill representation."""
    ITYPE_t order_id
    ITYPE_t timestamp
    int side
    DTYPE_t price
    ITYPE_t quantity
    int maker_taker


cdef struct OrderBookLevel:
    """C struct for order book level."""
    DTYPE_t price
    ITYPE_t total_qty
    ITYPE_t order_count


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CythonMatchingEngine:
    """High-performance Cython matching engine."""
    
    cdef Order* orders
    cdef Fill* fills
    cdef OrderBookLevel* bid_levels
    cdef OrderBookLevel* ask_levels
    cdef int max_orders
    cdef int max_fills
    cdef int max_levels
    cdef int num_orders
    cdef int num_fills
    cdef int num_bid_levels
    cdef int num_ask_levels
    cdef ITYPE_t total_volume
    cdef ITYPE_t total_trades
    cdef DTYPE_t last_trade_price
    
    def __init__(self, int max_orders=10000, int max_fills=50000, int max_levels=1000):
        """Initialize the Cython matching engine."""
        self.max_orders = max_orders
        self.max_fills = max_fills
        self.max_levels = max_levels
        
        # Allocate memory
        self.orders = <Order*>malloc(max_orders * sizeof(Order))
        self.fills = <Fill*>malloc(max_fills * sizeof(Fill))
        self.bid_levels = <OrderBookLevel*>malloc(max_levels * sizeof(OrderBookLevel))
        self.ask_levels = <OrderBookLevel*>malloc(max_levels * sizeof(OrderBookLevel))
        
        # Initialize counters
        self.num_orders = 0
        self.num_fills = 0
        self.num_bid_levels = 0
        self.num_ask_levels = 0
        self.total_volume = 0
        self.total_trades = 0
        self.last_trade_price = 0.0
    
    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.orders != NULL:
            free(self.orders)
        if self.fills != NULL:
            free(self.fills)
        if self.bid_levels != NULL:
            free(self.bid_levels)
        if self.ask_levels != NULL:
            free(self.ask_levels)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _find_bid_level(self, DTYPE_t price) nogil:
        """Find bid level index for given price."""
        cdef int i
        for i in range(self.num_bid_levels):
            if self.bid_levels[i].price == price:
                return i
        return -1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _find_ask_level(self, DTYPE_t price) nogil:
        """Find ask level index for given price."""
        cdef int i
        for i in range(self.num_ask_levels):
            if self.ask_levels[i].price == price:
                return i
        return -1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _add_bid_level(self, DTYPE_t price, ITYPE_t qty) nogil:
        """Add or update bid level."""
        cdef int idx = self._find_bid_level(price)
        if idx >= 0:
            self.bid_levels[idx].total_qty += qty
            self.bid_levels[idx].order_count += 1
        else:
            if self.num_bid_levels < self.max_levels:
                self.bid_levels[self.num_bid_levels].price = price
                self.bid_levels[self.num_bid_levels].total_qty = qty
                self.bid_levels[self.num_bid_levels].order_count = 1
                self.num_bid_levels += 1
                return 1
        return 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _add_ask_level(self, DTYPE_t price, ITYPE_t qty) nogil:
        """Add or update ask level."""
        cdef int idx = self._find_ask_level(price)
        if idx >= 0:
            self.ask_levels[idx].total_qty += qty
            self.ask_levels[idx].order_count += 1
        else:
            if self.num_ask_levels < self.max_levels:
                self.ask_levels[self.num_ask_levels].price = price
                self.ask_levels[self.num_ask_levels].total_qty = qty
                self.ask_levels[self.num_ask_levels].order_count = 1
                self.num_ask_levels += 1
                return 1
        return 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _match_buy_order(self, Order* order) nogil:
        """Match a buy order against ask levels."""
        cdef int i, j
        cdef ITYPE_t fill_qty
        cdef DTYPE_t match_price
        cdef int fills_created = 0
        
        # For market orders, use worst possible price
        if order.order_type == MARKET:
            order.price = 1e10
        
        # Match against asks (ascending price order)
        for i in range(self.num_ask_levels):
            if order.remaining_qty <= 0:
                break
            
            if order.order_type == LIMIT and order.price < self.ask_levels[i].price:
                break  # Can't match at this price
            
            match_price = self.ask_levels[i].price
            fill_qty = min(order.remaining_qty, self.ask_levels[i].total_qty)
            
            # Create fills
            if self.num_fills < self.max_fills - 1:
                # Taker fill
                self.fills[self.num_fills].order_id = order.order_id
                self.fills[self.num_fills].timestamp = order.timestamp
                self.fills[self.num_fills].side = order.side
                self.fills[self.num_fills].price = match_price
                self.fills[self.num_fills].quantity = fill_qty
                self.fills[self.num_fills].maker_taker = 1  # TAKER
                self.num_fills += 1
                fills_created += 1
                
                # Maker fill (simplified - would need to track individual orders)
                self.fills[self.num_fills].order_id = -1  # Placeholder
                self.fills[self.num_fills].timestamp = order.timestamp
                self.fills[self.num_fills].side = SELL
                self.fills[self.num_fills].price = match_price
                self.fills[self.num_fills].quantity = fill_qty
                self.fills[self.num_fills].maker_taker = 0  # MAKER
                self.num_fills += 1
                fills_created += 1
            
            # Update order
            order.remaining_qty -= fill_qty
            order.filled_qty += fill_qty
            order.avg_fill_price = (order.avg_fill_price * (order.filled_qty - fill_qty) + 
                                  match_price * fill_qty) / order.filled_qty
            
            # Update level
            self.ask_levels[i].total_qty -= fill_qty
            
            # Update statistics
            self.total_volume += fill_qty
            self.total_trades += 1
            self.last_trade_price = match_price
            
            # Remove empty levels
            if self.ask_levels[i].total_qty <= 0:
                for j in range(i, self.num_ask_levels - 1):
                    self.ask_levels[j] = self.ask_levels[j + 1]
                self.num_ask_levels -= 1
                i -= 1
        
        # Update order status
        if order.remaining_qty == 0:
            order.status = FILLED
        elif order.filled_qty > 0:
            order.status = PARTIALLY_FILLED
        
        return fills_created
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _match_sell_order(self, Order* order) nogil:
        """Match a sell order against bid levels."""
        cdef int i, j
        cdef ITYPE_t fill_qty
        cdef DTYPE_t match_price
        cdef int fills_created = 0
        
        # For market orders, use worst possible price
        if order.order_type == MARKET:
            order.price = 0.0
        
        # Match against bids (descending price order)
        for i in range(self.num_bid_levels):
            if order.remaining_qty <= 0:
                break
            
            if order.order_type == LIMIT and order.price > self.bid_levels[i].price:
                break  # Can't match at this price
            
            match_price = self.bid_levels[i].price
            fill_qty = min(order.remaining_qty, self.bid_levels[i].total_qty)
            
            # Create fills
            if self.num_fills < self.max_fills - 1:
                # Taker fill
                self.fills[self.num_fills].order_id = order.order_id
                self.fills[self.num_fills].timestamp = order.timestamp
                self.fills[self.num_fills].side = order.side
                self.fills[self.num_fills].price = match_price
                self.fills[self.num_fills].quantity = fill_qty
                self.fills[self.num_fills].maker_taker = 1  # TAKER
                self.num_fills += 1
                fills_created += 1
                
                # Maker fill (simplified - would need to track individual orders)
                self.fills[self.num_fills].order_id = -1  # Placeholder
                self.fills[self.num_fills].timestamp = order.timestamp
                self.fills[self.num_fills].side = BUY
                self.fills[self.num_fills].price = match_price
                self.fills[self.num_fills].quantity = fill_qty
                self.fills[self.num_fills].maker_taker = 0  # MAKER
                self.num_fills += 1
                fills_created += 1
            
            # Update order
            order.remaining_qty -= fill_qty
            order.filled_qty += fill_qty
            order.avg_fill_price = (order.avg_fill_price * (order.filled_qty - fill_qty) + 
                                  match_price * fill_qty) / order.filled_qty
            
            # Update level
            self.bid_levels[i].total_qty -= fill_qty
            
            # Update statistics
            self.total_volume += fill_qty
            self.total_trades += 1
            self.last_trade_price = match_price
            
            # Remove empty levels
            if self.bid_levels[i].total_qty <= 0:
                for j in range(i, self.num_bid_levels - 1):
                    self.bid_levels[j] = self.bid_levels[j + 1]
                self.num_bid_levels -= 1
                i -= 1
        
        # Update order status
        if order.remaining_qty == 0:
            order.status = FILLED
        elif order.filled_qty > 0:
            order.status = PARTIALLY_FILLED
        
        return fills_created
    
    def add_order(self, order_id, timestamp, side, price, quantity, time_in_force=DAY, order_type=LIMIT):
        """Add an order to the matching engine."""
        cdef Order order
        cdef int fills_created = 0
        
        # Initialize order
        order.order_id = order_id
        order.timestamp = timestamp
        order.side = BUY if side == 'BUY' else SELL
        order.price = price
        order.quantity = quantity
        order.remaining_qty = quantity
        order.filled_qty = 0
        order.avg_fill_price = 0.0
        order.time_in_force = time_in_force
        order.order_type = order_type
        order.status = NEW
        
        # Store order
        if self.num_orders < self.max_orders:
            self.orders[self.num_orders] = order
            self.num_orders += 1
        
        # Match order
        if order.side == BUY:
            fills_created = self._match_buy_order(&order)
        else:
            fills_created = self._match_sell_order(&order)
        
        # Add to book if not filled and is limit order
        if order.remaining_qty > 0 and order.order_type == LIMIT:
            if order.side == BUY:
                self._add_bid_level(order.price, order.remaining_qty)
            else:
                self._add_ask_level(order.price, order.remaining_qty)
        
        return fills_created
    
    def get_fills(self):
        """Get all fills as list of tuples."""
        cdef list fills = []
        cdef int i
        for i in range(self.num_fills):
            fills.append((
                self.fills[i].order_id,
                self.fills[i].price,
                self.fills[i].quantity
            ))
        return fills
    
    def get_statistics(self):
        """Get engine statistics."""
        return {
            'total_volume': self.total_volume,
            'total_trades': self.total_trades,
            'last_trade_price': self.last_trade_price,
            'num_orders': self.num_orders,
            'num_fills': self.num_fills,
            'num_bid_levels': self.num_bid_levels,
            'num_ask_levels': self.num_ask_levels
        }


# EMA calculation function for momentum strategy
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CythonEMA:
    """High-performance Cython EMA calculator."""
    
    cdef DTYPE_t alpha
    cdef DTYPE_t ema_value
    cdef int initialized
    
    def __init__(self, int period):
        """Initialize EMA with given period."""
        self.alpha = 2.0 / (period + 1.0)
        self.ema_value = 0.0
        self.initialized = 0
    
    def update(self, DTYPE_t value):
        """Update EMA with new value."""
        if not self.initialized:
            self.ema_value = value
            self.initialized = 1
        else:
            self.ema_value = self.alpha * value + (1.0 - self.alpha) * self.ema_value
        
        return self.ema_value
    
    def get_value(self):
        """Get current EMA value."""
        return self.ema_value


# Order flow imbalance calculation
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CythonImbalance:
    """High-performance Cython order flow imbalance calculator."""
    
    cdef ITYPE_t buy_volume
    cdef ITYPE_t sell_volume
    cdef int window_size
    cdef ITYPE_t* buy_history
    cdef ITYPE_t* sell_history
    cdef int history_index
    
    def __init__(self, int window_size=100):
        """Initialize imbalance calculator with window size."""
        self.window_size = window_size
        self.buy_volume = 0
        self.sell_volume = 0
        self.history_index = 0
        
        # Allocate history arrays
        self.buy_history = <ITYPE_t*>malloc(window_size * sizeof(ITYPE_t))
        self.sell_history = <ITYPE_t*>malloc(window_size * sizeof(ITYPE_t))
        
        # Initialize to zero
        cdef int i
        for i in range(window_size):
            self.buy_history[i] = 0
            self.sell_history[i] = 0
    
    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.buy_history != NULL:
            free(self.buy_history)
        if self.sell_history != NULL:
            free(self.sell_history)
    
    def update(self, int side, ITYPE_t volume):
        """Update imbalance with new trade."""
        # Remove oldest values
        self.buy_volume -= self.buy_history[self.history_index]
        self.sell_volume -= self.sell_history[self.history_index]
        
        # Add new values
        if side == BUY:
            self.buy_history[self.history_index] = volume
            self.sell_history[self.history_index] = 0
            self.buy_volume += volume
        else:
            self.buy_history[self.history_index] = 0
            self.sell_history[self.history_index] = volume
            self.sell_volume += volume
        
        # Update index
        self.history_index = (self.history_index + 1) % self.window_size
    
    def get_imbalance(self):
        """Get current imbalance ratio."""
        cdef ITYPE_t total_volume = self.buy_volume + self.sell_volume
        if total_volume == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total_volume