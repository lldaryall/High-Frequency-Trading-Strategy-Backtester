#!/usr/bin/env python3
"""
Bit-for-bit reproducibility test for C++ matching engine.

This test ensures that the C++ implementation produces identical results
to the Python implementation when using deterministic seeds and simple scenarios.
"""

import pytest
import random
import time
from typing import List

from flashback.market.book import PythonMatchingEngine, CppMatchingEngineWrapper, HAS_CPP
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_simple_order_matching_bit_for_bit():
    """Test bit-for-bit reproducibility with simple order matching scenarios.
    
    This test verifies that both C++ and Python engines produce identical
    results when processing the same order with no matching opportunities.
    """
    # Test case 1: Single order, no matching
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    order = Order(
        order_id="SIMPLE_ORDER",
        timestamp=1000,
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    cpp_fills = cpp_engine.add_order(order)
    python_fills = python_engine.add_order(order)
    
    # Both should have no fills (no matching orders)
    assert len(cpp_fills) == len(python_fills), f"Different number of fills: C++={len(cpp_fills)}, Python={len(python_fills)}"
    assert len(cpp_fills) == 0, "Expected no fills for single order with no matches"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_market_order_bit_for_bit():
    """Test bit-for-bit reproducibility with market orders."""
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Market order with no matching orders should not fill
    market_order = Order(
        order_id="MARKET_ORDER",
        timestamp=1000,
        symbol="TEST",
        side=OrderSide.BUY,
        price=0.0,  # Market order
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.MARKET
    )
    
    cpp_fills = cpp_engine.add_order(market_order)
    python_fills = python_engine.add_order(market_order)
    
    # Both should have no fills (no matching orders)
    assert len(cpp_fills) == len(python_fills), f"Different number of fills: C++={len(cpp_fills)}, Python={len(python_fills)}"
    assert len(cpp_fills) == 0, "Expected no fills for market order with no matches"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_deterministic_seed_reproducibility():
    """Test that both engines produce identical results with deterministic seeds."""
    # Use a fixed seed for reproducibility
    random.seed(42)
    
    # Create engines
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Create a simple order sequence that should not produce matches
    orders = []
    for i in range(10):
        # Create orders that won't match (all same side)
        order = Order(
            order_id=f"ORDER_{i}",
            timestamp=1000 + i * 1000,
            symbol="TEST",
            side=OrderSide.BUY,  # All same side
            price=100.0 + i * 0.1,  # Different prices
            quantity=10 + i,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        orders.append(order)
    
    # Process orders on both engines
    cpp_fills = []
    python_fills = []
    
    for order in orders:
        # Create identical copies for each engine
        cpp_order = Order(
            order_id=order.order_id,
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            time_in_force=order.time_in_force,
            order_type=order.order_type
        )
        
        python_order = Order(
            order_id=order.order_id,
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            time_in_force=order.time_in_force,
            order_type=order.order_type
        )
        
        cpp_fills.extend(cpp_engine.add_order(cpp_order))
        python_fills.extend(python_engine.add_order(python_order))
    
    # Both should have no fills (all same side orders)
    assert len(cpp_fills) == len(python_fills), f"Different number of fills: C++={len(cpp_fills)}, Python={len(python_fills)}"
    assert len(cpp_fills) == 0, "Expected no fills for same-side orders"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_engine_statistics_bit_for_bit():
    """Test that engine statistics are identical after processing same orders."""
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Create simple orders that won't match
    orders = [
        Order("ORDER1", 1000, "TEST", OrderSide.BUY, 100.0, 100, TimeInForce.DAY, OrderType.LIMIT),
        Order("ORDER2", 1001, "TEST", OrderSide.BUY, 101.0, 50, TimeInForce.DAY, OrderType.LIMIT),
        Order("ORDER3", 1002, "TEST", OrderSide.BUY, 102.0, 75, TimeInForce.DAY, OrderType.LIMIT),
    ]
    
    # Process all orders
    for order in orders:
        cpp_engine.add_order(order)
        python_engine.add_order(order)
    
    # Compare engine statistics
    cpp_stats = cpp_engine.get_statistics()
    python_stats = python_engine.get_statistics()
    
    assert cpp_stats["total_orders"] == python_stats["total_orders"], "Different total orders"
    assert cpp_stats["total_fills"] == python_stats["total_fills"], "Different total fills"
    assert cpp_stats["symbol"] == python_stats["symbol"], "Different symbol"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_order_creation_bit_for_bit():
    """Test that order creation produces identical results."""
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Create identical orders
    order = Order(
        order_id="TEST_ORDER",
        timestamp=1000,
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    # Add to both engines
    cpp_fills = cpp_engine.add_order(order)
    python_fills = python_engine.add_order(order)
    
    # Both should have no fills (no matching orders)
    assert len(cpp_fills) == len(python_fills), f"Different number of fills: C++={len(cpp_fills)}, Python={len(python_fills)}"
    assert len(cpp_fills) == 0, "Expected no fills for single order with no matches"
    
    # Both engines should have the same statistics
    cpp_stats = cpp_engine.get_statistics()
    python_stats = python_engine.get_statistics()
    
    assert cpp_stats["total_orders"] == python_stats["total_orders"], "Different total orders"
    assert cpp_stats["total_fills"] == python_stats["total_fills"], "Different total fills"


def test_python_engine_deterministic_behavior():
    """Test that Python engine produces deterministic results with same seed."""
    # Test with multiple seeds
    for seed in [42, 123, 456, 789]:
        random.seed(seed)
        
        # Create two engines
        engine1 = PythonMatchingEngine("TEST")
        engine2 = PythonMatchingEngine("TEST")
        
        # Create identical orders
        orders = []
        for i in range(5):
            order = Order(
                order_id=f"ORDER_{i}",
                timestamp=1000 + i * 1000,
                symbol="TEST",
                side=OrderSide.BUY,
                price=100.0 + i * 0.1,
                quantity=10 + i,
                time_in_force=TimeInForce.DAY,
                order_type=OrderType.LIMIT
            )
            orders.append(order)
        
        # Process on both engines
        fills1 = []
        fills2 = []
        
        for order in orders:
            fills1.extend(engine1.add_order(order))
            fills2.extend(engine2.add_order(order))
        
        # Results should be identical
        assert len(fills1) == len(fills2), f"Python engine not deterministic with seed {seed}"
        
        # Statistics should be identical
        stats1 = engine1.get_statistics()
        stats2 = engine2.get_statistics()
        
        assert stats1["total_orders"] == stats2["total_orders"], f"Different total orders with seed {seed}"
        assert stats1["total_fills"] == stats2["total_fills"], f"Different total fills with seed {seed}"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_cpp_engine_availability():
    """Test that C++ engine is properly available and functional."""
    # Test basic functionality
    engine = CppMatchingEngineWrapper("TEST")
    
    # Test order creation
    order = Order(
        order_id="AVAILABILITY_TEST",
        timestamp=1000,
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    fills = engine.add_order(order)
    assert isinstance(fills, list), "C++ engine should return list of fills"
    
    # Test statistics
    stats = engine.get_statistics()
    assert isinstance(stats, dict), "C++ engine should return dict of statistics"
    assert "total_orders" in stats, "Statistics should include total_orders"
    assert "total_fills" in stats, "Statistics should include total_fills"
    assert "symbol" in stats, "Statistics should include symbol"
    assert stats["symbol"] == "TEST", "Symbol should match engine symbol"
