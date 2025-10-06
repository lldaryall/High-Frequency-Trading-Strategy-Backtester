"""
Benchmark comparison tests for matching engine implementations.

Tests that both C++ and Python implementations produce identical results
and that C++ version is at least 8x faster.
"""

import pytest
import time
import random
from typing import List, Tuple
import uuid

from flashback.market.book import MatchingEngine, PythonMatchingEngine, CppMatchingEngineWrapper, HAS_CPP
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce


def create_random_order(order_id: str, symbol: str = "TEST") -> Order:
    """Create a random order for testing."""
    side = random.choice([OrderSide.BUY, OrderSide.SELL])
    price = round(random.uniform(100.0, 200.0), 2)
    quantity = random.randint(10, 1000)
    tif = random.choice([TimeInForce.DAY, TimeInForce.IOC, TimeInForce.FOK])
    
    return Order(
        order_id=order_id,
        timestamp=int(time.time() * 1e9),
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        time_in_force=tif,
        order_type=OrderType.LIMIT
    )


def run_benchmark(engine, num_orders: int = 100000) -> Tuple[float, List]:
    """Run benchmark with random orders and cancellations."""
    orders = []
    fills = []
    
    start_time = time.time()
    
    # Submit orders
    for i in range(num_orders):
        order = create_random_order(f"ORDER_{i}")
        orders.append(order)
        fills.extend(engine.add_order(order))
    
    # Cancel some orders randomly
    cancel_count = num_orders // 10  # Cancel 10% of orders
    for i in range(cancel_count):
        order_id = f"ORDER_{random.randint(0, num_orders - 1)}"
        engine.cancel_order(order_id, int(time.time() * 1e9))
    
    end_time = time.time()
    duration = end_time - start_time
    
    return duration, fills


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_cpp_vs_python_identical_results():
    """Test that C++ and Python implementations produce identical results."""
    # Create engines
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Use same random seed for reproducible results
    random.seed(42)
    
    # Generate test orders
    num_orders = 1000
    orders = [create_random_order(f"ORDER_{i}") for i in range(num_orders)]
    
    # Run on both engines
    cpp_fills = []
    python_fills = []
    
    for order in orders:
        # Create copies for each engine
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
    
    # Compare results
    assert len(cpp_fills) == len(python_fills), f"Different number of fills: C++={len(cpp_fills)}, Python={len(python_fills)}"
    
    # Sort fills by order_id and price for comparison
    cpp_fills_sorted = sorted(cpp_fills, key=lambda f: (f.order_id, f.price, f.quantity))
    python_fills_sorted = sorted(python_fills, key=lambda f: (f.order_id, f.price, f.quantity))
    
    for cpp_fill, python_fill in zip(cpp_fills_sorted, python_fills_sorted):
        assert cpp_fill.order_id == python_fill.order_id
        assert cpp_fill.price == python_fill.price
        assert cpp_fill.quantity == python_fill.quantity
        assert cpp_fill.side == python_fill.side
    
    # Compare statistics
    cpp_stats = cpp_engine.get_statistics()
    python_stats = python_engine.get_statistics()
    
    assert cpp_stats["total_fills"] == python_stats["total_fills"]
    assert cpp_stats["total_volume"] == python_stats["total_volume"]
    assert cpp_stats["total_trades"] == python_stats["total_trades"]


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_cpp_performance_benchmark():
    """Test that C++ implementation is at least 8x faster than Python."""
    # Create engines
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Use same random seed for fair comparison
    random.seed(123)
    
    # Run benchmark on both engines
    num_orders = 100000
    
    # Python benchmark
    random.seed(123)  # Reset seed
    python_duration, python_fills = run_benchmark(python_engine, num_orders)
    
    # C++ benchmark
    random.seed(123)  # Reset seed
    cpp_duration, cpp_fills = run_benchmark(cpp_engine, num_orders)
    
    # Calculate speedup
    speedup = python_duration / cpp_duration
    
    print(f"\nBenchmark Results:")
    print(f"Python duration: {python_duration:.4f} seconds")
    print(f"C++ duration: {cpp_duration:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Python fills: {len(python_fills)}")
    print(f"C++ fills: {len(cpp_fills)}")
    
    # Assert minimum speedup
    assert speedup >= 8.0, f"C++ implementation is only {speedup:.2f}x faster, expected at least 8x"
    
    # Assert identical results
    assert len(cpp_fills) == len(python_fills), "Different number of fills"
    
    # Compare fill counts by order
    cpp_fill_counts = {}
    python_fill_counts = {}
    
    for fill in cpp_fills:
        cpp_fill_counts[fill.order_id] = cpp_fill_counts.get(fill.order_id, 0) + fill.quantity
    
    for fill in python_fills:
        python_fill_counts[fill.order_id] = python_fill_counts.get(fill.order_id, 0) + fill.quantity
    
    assert cpp_fill_counts == python_fill_counts, "Fill quantities don't match"


def test_unified_engine_automatic_fallback():
    """Test that unified engine automatically falls back to Python when C++ unavailable."""
    # Test with C++ disabled
    engine = MatchingEngine("TEST", use_cpp=False)
    assert isinstance(engine.engine, PythonMatchingEngine)
    assert not engine.use_cpp
    
    # Test with C++ enabled (if available)
    if HAS_CPP:
        engine = MatchingEngine("TEST", use_cpp=True)
        assert isinstance(engine.engine, CppMatchingEngineWrapper)
        assert engine.use_cpp
    else:
        # Should fall back to Python even when C++ is requested
        engine = MatchingEngine("TEST", use_cpp=True)
        assert isinstance(engine.engine, PythonMatchingEngine)
        assert not engine.use_cpp


def test_unified_engine_interface():
    """Test that unified engine provides consistent interface."""
    engine = MatchingEngine("TEST")
    
    # Test basic functionality with a simple order
    order = Order(
        order_id="TEST_ORDER",
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    fills = engine.add_order(order)
    
    # Test order retrieval
    retrieved_order = engine.get_order("TEST_ORDER")
    assert retrieved_order is not None
    assert retrieved_order.order_id == "TEST_ORDER"
    
    # Test cancellation (only if order is still active)
    if retrieved_order.is_active():
        cancel = engine.cancel_order("TEST_ORDER", int(time.time() * 1e9))
        assert cancel is not None
        assert cancel.order_id == "TEST_ORDER"
    
    # Test statistics
    stats = engine.get_statistics()
    assert "symbol" in stats
    assert "total_orders" in stats
    assert "total_fills" in stats
    assert stats["symbol"] == "TEST"


@pytest.mark.skipif(not HAS_CPP, reason="C++ matching engine not available")
def test_cpp_engine_detailed_comparison():
    """Detailed comparison of C++ and Python engines with specific scenarios."""
    cpp_engine = CppMatchingEngineWrapper("TEST")
    python_engine = PythonMatchingEngine("TEST")
    
    # Test scenario 1: Simple buy/sell match
    buy_order = Order(
        order_id="BUY1",
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=100,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    sell_order = Order(
        order_id="SELL1",
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=OrderSide.SELL,
        price=100.0,
        quantity=50,
        time_in_force=TimeInForce.DAY,
        order_type=OrderType.LIMIT
    )
    
    # Add orders to both engines
    cpp_fills1 = cpp_engine.add_order(buy_order)
    cpp_fills2 = cpp_engine.add_order(sell_order)
    
    python_fills1 = python_engine.add_order(buy_order)
    python_fills2 = python_engine.add_order(sell_order)
    
    # Compare results
    assert len(cpp_fills1) == len(python_fills1)
    assert len(cpp_fills2) == len(python_fills2)
    
    # Test scenario 2: IOC order
    ioc_order = Order(
        order_id="IOC1",
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=OrderSide.BUY,
        price=99.0,
        quantity=100,
        time_in_force=TimeInForce.IOC,
        order_type=OrderType.LIMIT
    )
    
    cpp_ioc_fills = cpp_engine.add_order(ioc_order)
    python_ioc_fills = python_engine.add_order(ioc_order)
    
    assert len(cpp_ioc_fills) == len(python_ioc_fills)
    
    # Test scenario 3: FOK order
    fok_order = Order(
        order_id="FOK1",
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=OrderSide.BUY,
        price=100.0,
        quantity=200,  # More than available
        time_in_force=TimeInForce.FOK,
        order_type=OrderType.LIMIT
    )
    
    cpp_fok_fills = cpp_engine.add_order(fok_order)
    python_fok_fills = python_engine.add_order(fok_order)
    
    assert len(cpp_fok_fills) == len(python_fok_fills)
    
    # Compare final statistics
    cpp_stats = cpp_engine.get_statistics()
    python_stats = python_engine.get_statistics()
    
    assert cpp_stats["total_fills"] == python_stats["total_fills"]
    assert cpp_stats["total_volume"] == python_stats["total_volume"]
    assert cpp_stats["total_trades"] == python_stats["total_trades"]


if __name__ == "__main__":
    # Run benchmarks directly
    if HAS_CPP:
        print("Running C++ vs Python benchmark...")
        test_cpp_performance_benchmark()
        print("Benchmark completed successfully!")
    else:
        print("C++ matching engine not available. Install with 'make cpp'")
