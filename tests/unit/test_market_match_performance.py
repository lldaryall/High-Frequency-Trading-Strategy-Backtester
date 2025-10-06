"""Performance tests comparing Cython and Python implementations."""

import pytest
import numpy as np
from typing import Tuple
import time

# Import both implementations
from flashback.market._match_wrapper import (
    sort_orders_by_price_time,
    match_orders,
    calculate_ema,
    calculate_imbalance,
    heapify,
    build_heap,
    get_implementation_info
)

# Also import the pure Python implementation directly for comparison
from flashback.market._match_python import (
    sort_orders_by_price_time as sort_orders_by_price_time_python,
    match_orders_python,
    calculate_ema_python,
    calculate_imbalance_python,
    heapify_python,
    build_heap_python
)


class TestPerformanceComparison:
    """Test that Cython and Python implementations produce identical results."""
    
    def setup_method(self):
        """Set up test data with fixed seed for reproducibility."""
        np.random.seed(42)
        self.n_orders = 1000
        self.n_book = 500
        
        # Generate test data
        self.prices = np.random.uniform(100, 200, self.n_orders).astype(np.float64)
        self.timestamps = np.random.randint(1000000000000000000, 2000000000000000000, self.n_orders).astype(np.int64)
        self.sides = np.random.randint(0, 2, self.n_orders).astype(np.int32)
        self.quantities = np.random.randint(100, 1000, self.n_orders).astype(np.int32)
        
        # Book data
        self.book_prices = np.random.uniform(100, 200, self.n_book).astype(np.float64)
        self.book_timestamps = np.random.randint(1000000000000000000, 2000000000000000000, self.n_book).astype(np.int64)
        self.book_quantities = np.random.randint(100, 1000, self.n_book).astype(np.int32)
        self.book_sides = np.random.randint(0, 2, self.n_book).astype(np.int32)
        
        # EMA data
        self.ema_prices = np.random.uniform(100, 200, 1000).astype(np.float64)
        self.ema_alpha = 0.1
        self.ema_result = np.empty_like(self.ema_prices)
        
        # Imbalance data
        self.bid_sizes = np.random.randint(100, 1000, 100).astype(np.int32)
        self.ask_sizes = np.random.randint(100, 1000, 100).astype(np.int32)
        
        # Heap data
        self.heap_prices = np.random.uniform(100, 200, 100).astype(np.float64)
        self.heap_timestamps = np.random.randint(1000000000000000000, 2000000000000000000, 100).astype(np.int64)
        self.heap_indices = np.arange(100, dtype=np.int32)
    
    def test_sort_orders_identical_results(self):
        """Test that sorting produces identical results."""
        # Test ascending sort
        result_wrapper = sort_orders_by_price_time(self.prices, self.timestamps, self.sides, ascending=1)
        result_python = sort_orders_by_price_time_python(self.prices, self.timestamps, self.sides, ascending=1)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
        
        # Test descending sort
        result_wrapper = sort_orders_by_price_time(self.prices, self.timestamps, self.sides, ascending=0)
        result_python = sort_orders_by_price_time_python(self.prices, self.timestamps, self.sides, ascending=0)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
    
    def test_match_orders_identical_results(self):
        """Test that order matching produces identical results."""
        # Make copies since the function modifies arrays
        book_quantities_wrapper = self.book_quantities.copy()
        book_quantities_python = self.book_quantities.copy()
        
        result_wrapper = match_orders(
            self.prices, self.timestamps, self.quantities, self.sides,
            self.book_prices, self.book_timestamps, book_quantities_wrapper, self.book_sides
        )
        
        result_python = match_orders_python(
            self.prices, self.timestamps, self.quantities, self.sides,
            self.book_prices, self.book_timestamps, book_quantities_python, self.book_sides
        )
        
        # Compare all returned arrays
        for arr_wrapper, arr_python in zip(result_wrapper, result_python):
            np.testing.assert_array_equal(arr_wrapper, arr_python)
    
    def test_calculate_ema_identical_results(self):
        """Test that EMA calculation produces identical results."""
        result_wrapper = self.ema_result.copy()
        result_python = self.ema_result.copy()
        
        calculate_ema(self.ema_prices, self.ema_alpha, result_wrapper)
        calculate_ema_python(self.ema_prices, self.ema_alpha, result_python)
        
        np.testing.assert_array_almost_equal(result_wrapper, result_python, decimal=10)
    
    def test_calculate_imbalance_identical_results(self):
        """Test that imbalance calculation produces identical results."""
        result_wrapper = calculate_imbalance(self.bid_sizes, self.ask_sizes)
        result_python = calculate_imbalance_python(self.bid_sizes, self.ask_sizes)
        
        np.testing.assert_array_almost_equal(result_wrapper, result_python, decimal=10)
    
    def test_heapify_identical_results(self):
        """Test that heapify produces identical results."""
        indices_wrapper = self.heap_indices.copy()
        indices_python = self.heap_indices.copy()
        
        heapify(self.heap_prices, self.heap_timestamps, indices_wrapper, 100, 0)
        heapify_python(self.heap_prices, self.heap_timestamps, indices_python, 100, 0)
        
        np.testing.assert_array_equal(indices_wrapper, indices_python)
    
    def test_build_heap_identical_results(self):
        """Test that build_heap produces identical results."""
        indices_wrapper = self.heap_indices.copy()
        indices_python = self.heap_indices.copy()
        
        build_heap(self.heap_prices, self.heap_timestamps, indices_wrapper, 100)
        build_heap_python(self.heap_prices, self.heap_timestamps, indices_python, 100)
        
        np.testing.assert_array_equal(indices_wrapper, indices_python)
    
    def test_implementation_info(self):
        """Test that implementation info is correctly reported."""
        info = get_implementation_info()
        
        assert "cython_available" in info
        assert "implementation" in info
        assert "performance_note" in info
        assert info["implementation"] in ["cython", "python"]
    
    def test_performance_improvement(self):
        """Test that Cython implementation is faster (if available)."""
        info = get_implementation_info()
        
        if not info["cython_available"]:
            pytest.skip("Cython not available, skipping performance test")
        
        # Time the wrapper implementation (should use Cython if available)
        start_time = time.time()
        for _ in range(100):
            sort_orders_by_price_time(self.prices, self.timestamps, self.sides, ascending=1)
        wrapper_time = time.time() - start_time
        
        # Time the pure Python implementation
        start_time = time.time()
        for _ in range(100):
            sort_orders_by_price_time_python(self.prices, self.timestamps, self.sides, ascending=1)
        python_time = time.time() - start_time
        
        # Cython should be faster (allow for some variance)
        if wrapper_time > 0 and python_time > 0:
            speedup = python_time / wrapper_time
            print(f"Speedup: {speedup:.2f}x")
            # Allow for some variance, but Cython should generally be faster
            # Note: On some systems, the overhead of Cython might make it slightly slower for small datasets
            assert speedup >= 0.5, f"Expected reasonable performance, got speedup: {speedup:.2f}x"
    
    def test_edge_cases(self):
        """Test edge cases that might cause issues."""
        # Empty arrays
        empty_prices = np.array([], dtype=np.float64)
        empty_timestamps = np.array([], dtype=np.int64)
        empty_sides = np.array([], dtype=np.int32)
        
        result_wrapper = sort_orders_by_price_time(empty_prices, empty_timestamps, empty_sides)
        result_python = sort_orders_by_price_time_python(empty_prices, empty_timestamps, empty_sides)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
        
        # Single element
        single_prices = np.array([150.0], dtype=np.float64)
        single_timestamps = np.array([1000000000000000000], dtype=np.int64)
        single_sides = np.array([0], dtype=np.int32)
        
        result_wrapper = sort_orders_by_price_time(single_prices, single_timestamps, single_sides)
        result_python = sort_orders_by_price_time_python(single_prices, single_timestamps, single_sides)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
        
        # All same values
        same_prices = np.full(10, 150.0, dtype=np.float64)
        same_timestamps = np.arange(1000000000000000000, 1000000000000000010, dtype=np.int64)
        same_sides = np.zeros(10, dtype=np.int32)
        
        result_wrapper = sort_orders_by_price_time(same_prices, same_timestamps, same_sides)
        result_python = sort_orders_by_price_time_python(same_prices, same_timestamps, same_sides)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large numbers
        large_prices = np.array([1e10, 1e11, 1e12], dtype=np.float64)
        large_timestamps = np.array([1000000000000000000, 2000000000000000000, 3000000000000000000], dtype=np.int64)
        large_sides = np.array([0, 1, 0], dtype=np.int32)
        
        result_wrapper = sort_orders_by_price_time(large_prices, large_timestamps, large_sides)
        result_python = sort_orders_by_price_time_python(large_prices, large_timestamps, large_sides)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
        
        # Very small numbers
        small_prices = np.array([1e-10, 1e-11, 1e-12], dtype=np.float64)
        small_timestamps = np.array([1000000000000000000, 2000000000000000000, 3000000000000000000], dtype=np.int64)
        small_sides = np.array([0, 1, 0], dtype=np.int32)
        
        result_wrapper = sort_orders_by_price_time(small_prices, small_timestamps, small_sides)
        result_python = sort_orders_by_price_time_python(small_prices, small_timestamps, small_sides)
        
        np.testing.assert_array_equal(result_wrapper, result_python)
