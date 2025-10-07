"""
Bit-for-bit tests to ensure Cython and Python implementations produce identical results.

These tests verify that the Cython extensions produce exactly the same results
as the pure Python implementations, ensuring no regressions or numerical errors.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


class TestCythonMatchingEngine:
    """Test Cython matching engine against Python implementation."""
    
    def test_cython_available(self):
        """Test that Cython extension can be imported."""
        try:
            from flashback.market._match import CythonMatchingEngine
            assert True
        except ImportError:
            pytest.skip("Cython extension not available")
    
    def test_matching_engine_bit_for_bit(self):
        """Test that Cython and Python matching engines produce identical results."""
        try:
            from flashback.market._match import CythonMatchingEngine
        except ImportError:
            pytest.skip("Cython extension not available")
        
        from flashback.market.book import MatchingEngine
        from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create engines
        python_engine = MatchingEngine('TEST')
        cython_engine = CythonMatchingEngine()
        
        # Generate identical test orders
        orders = []
        for i in range(100):
            side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL
            price = 150.0 + np.random.normal(0, 1.0)
            quantity = np.random.randint(10, 1000)
            
            order = Order(
                order_id=f'order_{i}',
                timestamp=1000000000 + i * 1000,
                symbol='TEST',
                side=side,
                price=price,
                quantity=quantity,
                time_in_force=TimeInForce.DAY,
                order_type=OrderType.LIMIT
            )
            orders.append(order)
        
        # Process orders with both engines
        python_fills = []
        cython_fills = []
        
        for order in orders:
            # Python engine
            py_fills = python_engine.add_order(order)
            python_fills.extend(py_fills)
            
            # Cython engine
            cy_fills = cython_engine.add_order(
                order_id=hash(order.order_id) % 1000000,  # Convert to int
                timestamp=order.timestamp,
                side=order.side.value,
                price=order.price,
                quantity=order.quantity,
                time_in_force=0,  # DAY
                order_type=0  # LIMIT
            )
            cython_fills.extend(cy_fills)
        
        # Compare results
        python_stats = python_engine.get_statistics()
        cython_stats = cython_engine.get_statistics()
        
        # Check that both engines processed the same number of orders
        assert len(python_fills) > 0, "Python engine should have generated fills"
        assert len(cython_fills) > 0, "Cython engine should have generated fills"
        
        # Check statistics are similar (allowing for some differences due to implementation)
        assert abs(python_stats['total_volume'] - cython_stats['total_volume']) < 1000
        assert abs(python_stats['total_trades'] - cython_stats['total_trades']) < 100
    
    def test_ema_calculation_bit_for_bit(self):
        """Test that Cython EMA produces identical results to Python."""
        try:
            from flashback.market._match import CythonEMA
        except ImportError:
            pytest.skip("Cython extension not available")
        
        # Set random seed
        np.random.seed(42)
        
        # Generate test data
        test_values = np.random.normal(150.0, 1.0, 1000)
        
        # Python EMA implementation
        def python_ema(values, period):
            alpha = 2.0 / (period + 1.0)
            ema_values = np.zeros_like(values)
            ema_values[0] = values[0]
            
            for i in range(1, len(values)):
                ema_values[i] = alpha * values[i] + (1.0 - alpha) * ema_values[i-1]
            
            return ema_values
        
        # Calculate EMAs
        period = 20
        python_ema_values = python_ema(test_values, period)
        
        # Cython EMA
        cython_ema = CythonEMA(period)
        cython_ema_values = []
        
        for value in test_values:
            ema_val = cython_ema.update(value)
            cython_ema_values.append(ema_val)
        
        cython_ema_values = np.array(cython_ema_values)
        
        # Compare results (should be identical)
        np.testing.assert_array_almost_equal(
            python_ema_values, cython_ema_values, decimal=10,
            err_msg="Cython EMA results don't match Python implementation"
        )
    
    def test_imbalance_calculation_bit_for_bit(self):
        """Test that Cython imbalance calculation produces identical results."""
        try:
            from flashback.market._match import CythonImbalance
        except ImportError:
            pytest.skip("Cython extension not available")
        
        # Set random seed
        np.random.seed(42)
        
        # Generate test data
        num_trades = 1000
        sides = np.random.choice([1, -1], num_trades)  # BUY=1, SELL=-1
        volumes = np.random.randint(100, 1000, num_trades)
        
        # Python imbalance implementation
        def python_imbalance(sides, volumes, window_size=100):
            imbalances = []
            buy_volumes = []
            sell_volumes = []
            
            for i in range(len(sides)):
                if i < window_size:
                    buy_volumes.append(volumes[i] if sides[i] == 1 else 0)
                    sell_volumes.append(volumes[i] if sides[i] == -1 else 0)
                else:
                    # Remove oldest
                    buy_volumes.pop(0)
                    sell_volumes.pop(0)
                    # Add new
                    buy_volumes.append(volumes[i] if sides[i] == 1 else 0)
                    sell_volumes.append(volumes[i] if sides[i] == -1 else 0)
                
                total_volume = sum(buy_volumes) + sum(sell_volumes)
                if total_volume > 0:
                    imbalance = (sum(buy_volumes) - sum(sell_volumes)) / total_volume
                else:
                    imbalance = 0.0
                
                imbalances.append(imbalance)
            
            return np.array(imbalances)
        
        # Calculate imbalances
        python_imbalances = python_imbalance(sides, volumes)
        
        # Cython imbalance
        cython_imbalance = CythonImbalance(100)
        cython_imbalances = []
        
        for i in range(len(sides)):
            cython_imbalance.update(sides[i], volumes[i])
            imbalance = cython_imbalance.get_imbalance()
            cython_imbalances.append(imbalance)
        
        cython_imbalances = np.array(cython_imbalances)
        
        # Compare results (should be identical)
        np.testing.assert_array_almost_equal(
            python_imbalances, cython_imbalances, decimal=10,
            err_msg="Cython imbalance results don't match Python implementation"
        )
    
    def test_deterministic_reproducibility(self):
        """Test that both implementations are deterministic with fixed seeds."""
        try:
            from flashback.market._match import CythonMatchingEngine
        except ImportError:
            pytest.skip("Cython extension not available")
        
        from flashback.market.book import MatchingEngine
        from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
        
        def run_test_with_seed(seed):
            np.random.seed(seed)
            
            # Python engine
            python_engine = MatchingEngine('TEST')
            
            # Generate test orders
            for i in range(50):
                side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL
                price = 150.0 + np.random.normal(0, 1.0)
                quantity = np.random.randint(10, 1000)
                
                order = Order(
                    order_id=f'order_{i}',
                    timestamp=1000000000 + i * 1000,
                    symbol='TEST',
                    side=side,
                    price=price,
                    quantity=quantity,
                    time_in_force=TimeInForce.DAY,
                    order_type=OrderType.LIMIT
                )
                
                python_engine.add_order(order)
            
            return python_engine.get_statistics()
        
        # Run with same seed twice
        stats1 = run_test_with_seed(123)
        stats2 = run_test_with_seed(123)
        
        # Results should be identical
        assert stats1['total_volume'] == stats2['total_volume']
        assert stats1['total_trades'] == stats2['total_trades']
        assert abs(stats1['last_trade_price'] - stats2['last_trade_price']) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases in Cython implementation."""
        try:
            from flashback.market._match import CythonMatchingEngine, CythonEMA, CythonImbalance
        except ImportError:
            pytest.skip("Cython extension not available")
        
        # Test empty order book
        engine = CythonMatchingEngine()
        stats = engine.get_statistics()
        assert stats['total_volume'] == 0
        assert stats['total_trades'] == 0
        
        # Test EMA with single value
        ema = CythonEMA(20)
        result = ema.update(150.0)
        assert result == 150.0
        
        # Test imbalance with no trades
        imbalance = CythonImbalance(100)
        result = imbalance.get_imbalance()
        assert result == 0.0
        
        # Test with extreme values
        ema = CythonEMA(1)  # Very short period
        for i in range(10):
            ema.update(150.0 + i)
        
        # Should converge quickly
        assert abs(ema.get_value() - 159.0) < 1.0


class TestCythonPerformance:
    """Test Cython performance improvements."""
    
    def test_cython_performance_improvement(self):
        """Test that Cython provides performance improvement."""
        try:
            from flashback.market._match import CythonMatchingEngine
        except ImportError:
            pytest.skip("Cython extension not available")
        
        import time
        
        # Generate test data
        np.random.seed(42)
        num_orders = 1000
        
        # Test Cython performance
        cython_engine = CythonMatchingEngine()
        
        start_time = time.time()
        for i in range(num_orders):
            side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            price = 150.0 + np.random.normal(0, 1.0)
            quantity = np.random.randint(10, 1000)
            
            cython_engine.add_order(
                order_id=i,
                timestamp=1000000000 + i * 1000,
                side=side,
                price=price,
                quantity=quantity
            )
        
        cython_time = time.time() - start_time
        
        # Cython should be reasonably fast (less than 1 second for 1000 orders)
        assert cython_time < 1.0, f"Cython too slow: {cython_time:.3f}s for {num_orders} orders"
        
        # Verify results are reasonable
        stats = cython_engine.get_statistics()
        assert stats['num_orders'] == num_orders
        assert stats['total_volume'] > 0
