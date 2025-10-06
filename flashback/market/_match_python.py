"""
Pure Python fallback implementation of order matching hot paths.

This module provides identical functionality to the Cython implementation
but using pure Python for environments where Cython is not available.
"""

import numpy as np
from typing import Tuple


def sort_orders_by_price_time(prices: np.ndarray,
                             timestamps: np.ndarray,
                             sides: np.ndarray,
                             ascending: int = 1) -> np.ndarray:
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
    # Use numpy's lexsort which is highly optimized even in pure Python
    if ascending:
        return np.lexsort((timestamps, prices))
    else:
        return np.lexsort((timestamps, -prices))


def match_orders_python(incoming_prices: np.ndarray,
                       incoming_timestamps: np.ndarray,
                       incoming_quantities: np.ndarray,
                       incoming_sides: np.ndarray,
                       book_prices: np.ndarray,
                       book_timestamps: np.ndarray,
                       book_quantities: np.ndarray,
                       book_sides: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match incoming orders against the order book.
    
    This is the core matching logic that was identified as a hot path.
    
    Args:
        incoming_*: Arrays for incoming orders
        book_*: Arrays for existing book orders
    
    Returns:
        Tuple of (fill_prices, fill_quantities, fill_timestamps, matched_indices)
    """
    n_incoming = len(incoming_prices)
    n_book = len(book_prices)
    
    # Pre-allocate result arrays (over-allocate to avoid resizing)
    max_fills = n_incoming * 2
    fill_prices = np.empty(max_fills, dtype=np.float64)
    fill_quantities = np.empty(max_fills, dtype=np.int32)
    fill_timestamps = np.empty(max_fills, dtype=np.int64)
    matched_indices = np.empty(max_fills, dtype=np.int32)
    
    fill_count = 0
    
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


def calculate_ema_python(prices: np.ndarray,
                        alpha: float,
                        result: np.ndarray) -> None:
    """
    Calculate Exponential Moving Average using pure Python.
    
    This is used in strategy calculations and can be a hot path.
    
    Args:
        prices: Array of price values
        alpha: EMA smoothing factor (2 / (period + 1))
        result: Pre-allocated result array
    
    Returns:
        None (modifies result in-place)
    """
    n = len(prices)
    if n == 0:
        return
    
    ema = prices[0]
    result[0] = ema
    
    for i in range(1, n):
        ema = alpha * prices[i] + (1.0 - alpha) * ema
        result[i] = ema


def calculate_imbalance_python(bid_sizes: np.ndarray,
                              ask_sizes: np.ndarray) -> np.ndarray:
    """
    Calculate order flow imbalance using pure Python.
    
    Args:
        bid_sizes: Array of bid sizes
        ask_sizes: Array of ask sizes
    
    Returns:
        Array of imbalance values
    """
    n = len(bid_sizes)
    imbalance = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        bid_total = bid_sizes[i]
        ask_total = ask_sizes[i]
        total = bid_total + ask_total
        
        if total > 0:
            imbalance[i] = (bid_total - ask_total) / total
        else:
            imbalance[i] = 0.0
    
    return imbalance


def heapify_python(prices: np.ndarray,
                  timestamps: np.ndarray,
                  indices: np.ndarray,
                  n: int,
                  i: int) -> None:
    """
    Heapify operation for maintaining order book heap properties.
    
    This is used in the order book data structure maintenance.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
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
        heapify_python(prices, timestamps, indices, n, largest)


def build_heap_python(prices: np.ndarray,
                     timestamps: np.ndarray,
                     indices: np.ndarray,
                     n: int) -> None:
    """
    Build a max heap from the given arrays.
    """
    for i in range(n // 2 - 1, -1, -1):
        heapify_python(prices, timestamps, indices, n, i)
