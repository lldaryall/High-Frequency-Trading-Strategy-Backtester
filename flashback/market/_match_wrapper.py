"""
Wrapper module that automatically chooses between Cython and Python implementations.

This module provides a unified interface that falls back to pure Python
when Cython extensions are not available.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import Cython implementation, fall back to Python if not available
try:
    from ._match import (
        sort_orders_by_price_time as sort_orders_by_price_time_cython,
        match_orders_cython,
        calculate_ema_cython,
        calculate_imbalance_cython,
        heapify_cython,
        build_heap_cython
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# Always import Python fallback
from ._match_python import (
    sort_orders_by_price_time as sort_orders_by_price_time_python,
    match_orders_python,
    calculate_ema_python,
    calculate_imbalance_python,
    heapify_python,
    build_heap_python
)


def sort_orders_by_price_time(prices: np.ndarray,
                             timestamps: np.ndarray,
                             sides: np.ndarray,
                             ascending: int = 1) -> np.ndarray:
    """
    Sort orders by price (ascending for asks, descending for bids) then by time.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        return sort_orders_by_price_time_cython(prices, timestamps, sides, ascending)
    else:
        return sort_orders_by_price_time_python(prices, timestamps, sides, ascending)


def match_orders(incoming_prices: np.ndarray,
                incoming_timestamps: np.ndarray,
                incoming_quantities: np.ndarray,
                incoming_sides: np.ndarray,
                book_prices: np.ndarray,
                book_timestamps: np.ndarray,
                book_quantities: np.ndarray,
                book_sides: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match incoming orders against the order book.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        return match_orders_cython(
            incoming_prices, incoming_timestamps, incoming_quantities, incoming_sides,
            book_prices, book_timestamps, book_quantities, book_sides
        )
    else:
        return match_orders_python(
            incoming_prices, incoming_timestamps, incoming_quantities, incoming_sides,
            book_prices, book_timestamps, book_quantities, book_sides
        )


def calculate_ema(prices: np.ndarray,
                 alpha: float,
                 result: np.ndarray) -> None:
    """
    Calculate Exponential Moving Average.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        calculate_ema_cython(prices, alpha, result)
    else:
        calculate_ema_python(prices, alpha, result)


def calculate_imbalance(bid_sizes: np.ndarray,
                       ask_sizes: np.ndarray) -> np.ndarray:
    """
    Calculate order flow imbalance.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        return calculate_imbalance_cython(bid_sizes, ask_sizes)
    else:
        return calculate_imbalance_python(bid_sizes, ask_sizes)


def heapify(prices: np.ndarray,
           timestamps: np.ndarray,
           indices: np.ndarray,
           n: int,
           i: int) -> None:
    """
    Heapify operation for maintaining order book heap properties.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        heapify_cython(prices, timestamps, indices, n, i)
    else:
        heapify_python(prices, timestamps, indices, n, i)


def build_heap(prices: np.ndarray,
              timestamps: np.ndarray,
              indices: np.ndarray,
              n: int) -> None:
    """
    Build a max heap from the given arrays.
    
    Automatically uses Cython implementation if available, otherwise falls back to Python.
    """
    if CYTHON_AVAILABLE:
        build_heap_cython(prices, timestamps, indices, n)
    else:
        build_heap_python(prices, timestamps, indices, n)


def get_implementation_info() -> dict:
    """
    Get information about which implementation is being used.
    
    Returns:
        Dictionary with implementation details
    """
    return {
        "cython_available": CYTHON_AVAILABLE,
        "implementation": "cython" if CYTHON_AVAILABLE else "python",
        "performance_note": "Using optimized Cython implementation" if CYTHON_AVAILABLE else "Using pure Python fallback"
    }
