# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cython implementation of order matching hot paths.

This module provides optimized implementations of the most performance-critical
parts of the order matching engine, particularly the sorting and matching logic.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython

# Type definitions
ctypedef struct OrderLevel:
    double price
    long timestamp
    int total_qty
    int order_count

ctypedef struct Order:
    char* order_id
    double price
    long timestamp
    int quantity
    int remaining_qty
    int side  # 0 = BUY, 1 = SELL
    int order_type
    int time_in_force

@cython.boundscheck(False)
@cython.wraparound(False)
def sort_orders_by_price_time(cnp.ndarray[double, ndim=1] prices,
                             cnp.ndarray[long, ndim=1] timestamps,
                             cnp.ndarray[int, ndim=1] sides,
                             int ascending=1):
    """
    Sort orders by price (ascending for asks, descending for bids) then by time.
    
    This is a critical hot path that was identified in profiling.
    
    Args:
        prices: Array of order prices
        timestamps: Array of order timestamps
        sides: Array of order sides (0=BUY, 1=SELL)
        ascending: 1 for ascending sort (asks), 0 for descending (bids)
    
    Returns:
        Array of sorted indices
    """
    cdef int n = prices.shape[0]
    cdef cnp.ndarray[long, ndim=1] indices = np.arange(n, dtype=np.int64)
    
    # Create array of tuples for sorting
    cdef double[:, :] sort_keys = np.column_stack([
        prices if ascending else -prices,  # Negate for descending sort
        timestamps.astype(np.float64)
    ])
    
    # Use numpy's argsort which is highly optimized
    return np.lexsort((timestamps, prices if ascending else -prices))


@cython.boundscheck(False)
@cython.wraparound(False)
def match_orders_cython(cnp.ndarray[double, ndim=1] incoming_prices,
                       cnp.ndarray[long, ndim=1] incoming_timestamps,
                       cnp.ndarray[int, ndim=1] incoming_quantities,
                       cnp.ndarray[int, ndim=1] incoming_sides,
                       cnp.ndarray[double, ndim=1] book_prices,
                       cnp.ndarray[long, ndim=1] book_timestamps,
                       cnp.ndarray[int, ndim=1] book_quantities,
                       cnp.ndarray[int, ndim=1] book_sides):
    """
    Match incoming orders against the order book.
    
    This is the core matching logic that was identified as a hot path.
    
    Args:
        incoming_*: Arrays for incoming orders
        book_*: Arrays for existing book orders
    
    Returns:
        Tuple of (fill_prices, fill_quantities, fill_timestamps, matched_indices)
    """
    cdef int n_incoming = incoming_prices.shape[0]
    cdef int n_book = book_prices.shape[0]
    
    # Pre-allocate result arrays (over-allocate to avoid resizing)
    cdef cnp.ndarray[double, ndim=1] fill_prices = np.empty(n_incoming * 2, dtype=np.float64)
    cdef cnp.ndarray[int, ndim=1] fill_quantities = np.empty(n_incoming * 2, dtype=np.int32)
    cdef cnp.ndarray[long, ndim=1] fill_timestamps = np.empty(n_incoming * 2, dtype=np.int64)
    cdef cnp.ndarray[int, ndim=1] matched_indices = np.empty(n_incoming * 2, dtype=np.int32)
    
    cdef int fill_count = 0
    cdef int i, j
    cdef double incoming_price, book_price
    cdef int incoming_qty, book_qty, fill_qty
    cdef int incoming_side, book_side
    cdef long incoming_time, book_time
    
    for i in range(n_incoming):
        incoming_price = incoming_prices[i]
        incoming_qty = incoming_quantities[i]
        incoming_side = incoming_sides[i]
        incoming_time = incoming_timestamps[i]
        
        for j in range(n_book):
            book_price = book_prices[j]
            book_qty = book_quantities[j]
            book_side = book_sides[j]
            book_time = book_timestamps[j]
            
            # Check if orders can match
            if (incoming_side == 0 and book_side == 1 and incoming_price >= book_price) or \
               (incoming_side == 1 and book_side == 0 and incoming_price <= book_price):
                
                # Determine fill quantity
                fill_qty = min(incoming_qty, book_qty)
                
                if fill_qty > 0:
                    # Record the fill
                    fill_prices[fill_count] = book_price  # Price-time priority
                    fill_quantities[fill_count] = fill_qty
                    fill_timestamps[fill_count] = min(incoming_time, book_time)
                    matched_indices[fill_count] = j
                    fill_count += 1
                    
                    # Update remaining quantities
                    incoming_qty -= fill_qty
                    book_quantities[j] -= fill_qty
                    
                    if incoming_qty <= 0:
                        break
    
    # Trim arrays to actual size
    return (fill_prices[:fill_count], 
            fill_quantities[:fill_count], 
            fill_timestamps[:fill_count], 
            matched_indices[:fill_count])


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_ema_cython(cnp.ndarray[double, ndim=1] prices,
                        double alpha,
                        cnp.ndarray[double, ndim=1] result):
    """
    Calculate Exponential Moving Average using Cython for performance.
    
    This is used in strategy calculations and can be a hot path.
    
    Args:
        prices: Array of price values
        alpha: EMA smoothing factor (2 / (period + 1))
        result: Pre-allocated result array
    
    Returns:
        None (modifies result in-place)
    """
    cdef int n = prices.shape[0]
    cdef int i
    cdef double ema = prices[0] if n > 0 else 0.0
    
    result[0] = ema
    
    for i in range(1, n):
        ema = alpha * prices[i] + (1.0 - alpha) * ema
        result[i] = ema


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_imbalance_cython(cnp.ndarray[int, ndim=1] bid_sizes,
                              cnp.ndarray[int, ndim=1] ask_sizes):
    """
    Calculate order flow imbalance using Cython.
    
    Args:
        bid_sizes: Array of bid sizes
        ask_sizes: Array of ask sizes
    
    Returns:
        Array of imbalance values
    """
    cdef int n = bid_sizes.shape[0]
    cdef cnp.ndarray[double, ndim=1] imbalance = np.empty(n, dtype=np.float64)
    cdef int i
    cdef int bid_total, ask_total, total
    
    for i in range(n):
        bid_total = bid_sizes[i]
        ask_total = ask_sizes[i]
        total = bid_total + ask_total
        
        if total > 0:
            imbalance[i] = (bid_total - ask_total) / <double>total
        else:
            imbalance[i] = 0.0
    
    return imbalance


@cython.boundscheck(False)
@cython.wraparound(False)
def heapify_cython(cnp.ndarray[double, ndim=1] prices,
                  cnp.ndarray[long, ndim=1] timestamps,
                  cnp.ndarray[int, ndim=1] indices,
                  int n,
                  int i):
    """
    Heapify operation for maintaining order book heap properties.
    
    This is used in the order book data structure maintenance.
    """
    cdef int largest = i
    cdef int left = 2 * i + 1
    cdef int right = 2 * i + 2
    
    # Compare with left child
    if left < n and prices[indices[left]] > prices[indices[largest]]:
        largest = left
    
    # Compare with right child
    if right < n and prices[indices[right]] > prices[indices[largest]]:
        largest = right
    
    # If largest is not root
    if largest != i:
        # Swap
        indices[i], indices[largest] = indices[largest], indices[i]
        # Recursively heapify the affected sub-tree
        heapify_cython(prices, timestamps, indices, n, largest)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_heap_cython(cnp.ndarray[double, ndim=1] prices,
                     cnp.ndarray[long, ndim=1] timestamps,
                     cnp.ndarray[int, ndim=1] indices,
                     int n):
    """
    Build a max heap from the given arrays.
    """
    cdef int i
    for i in range(n // 2 - 1, -1, -1):
        heapify_cython(prices, timestamps, indices, n, i)
