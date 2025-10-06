#!/usr/bin/env python3
"""
Test script for the C++ matching engine extension.

This script tests the C++ matching engine functionality and performance
compared to the Python implementation.
"""

import sys
import os
import time
import random
from typing import List, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from flashback.market import CppMatchEngine, CppFill, CPP_MATCHING_AVAILABLE
    from flashback.market.orders import Order, OrderSide, TimeInForce, OrderType
    print("✅ Successfully imported C++ matching engine")
except ImportError as e:
    print(f"❌ Failed to import C++ matching engine: {e}")
    print("Run 'make cpp' to build the extension")
    sys.exit(1)

def create_test_order(order_id: str, side: OrderSide, price: float, quantity: int, tif: TimeInForce = TimeInForce.DAY) -> Order:
    """Create a test order."""
    return Order(
        order_id=order_id,
        timestamp=int(time.time() * 1e9),
        symbol="TEST",
        side=side,
        price=price,
        quantity=quantity,
        time_in_force=tif,
        order_type=OrderType.LIMIT
    )

def test_basic_matching():
    """Test basic order matching functionality."""
    print("\n🧪 Testing basic order matching...")
    
    engine = CppMatchEngine("TEST")
    
    # Add a buy order
    buy_order = create_test_order("BUY1", OrderSide.BUY, 100.0, 100)
    fills = engine.match_order(buy_order)
    print(f"  Buy order added: {len(fills)} fills")
    
    # Add a matching sell order
    sell_order = create_test_order("SELL1", OrderSide.SELL, 100.0, 50)
    fills = engine.match_order(sell_order)
    print(f"  Sell order added: {len(fills)} fills")
    
    for fill in fills:
        print(f"    Fill: {fill.order_id} @ {fill.price} x {fill.quantity}")
    
    # Check order book
    snapshot = engine.get_snapshot()
    print(f"  Order book: {len(snapshot['bids'])} bids, {len(snapshot['asks'])} asks")
    print(f"  Active orders: {engine.get_order_count()}")
    
    return len(fills) > 0

def test_partial_fills():
    """Test partial fill functionality."""
    print("\n🧪 Testing partial fills...")
    
    engine = CppMatchEngine("TEST")
    
    # Add a large buy order
    buy_order = create_test_order("BUY1", OrderSide.BUY, 100.0, 1000)
    fills = engine.match_order(buy_order)
    print(f"  Large buy order: {len(fills)} fills")
    
    # Add multiple smaller sell orders
    total_fills = 0
    for i in range(5):
        sell_order = create_test_order(f"SELL{i+1}", OrderSide.SELL, 100.0, 200)
        fills = engine.match_order(sell_order)
        total_fills += len(fills)
        print(f"    Sell order {i+1}: {len(fills)} fills")
    
    print(f"  Total fills: {total_fills}")
    print(f"  Remaining orders: {engine.get_order_count()}")
    
    return total_fills > 0

def test_ioc_fok_orders():
    """Test IOC and FOK order types."""
    print("\n🧪 Testing IOC and FOK orders...")
    
    engine = CppMatchEngine("TEST")
    
    # Add a resting order
    resting_order = create_test_order("REST1", OrderSide.SELL, 100.0, 100)
    fills = engine.match_order(resting_order)
    print(f"  Resting order: {len(fills)} fills")
    
    # Test IOC order (should fill partially)
    ioc_order = create_test_order("IOC1", OrderSide.BUY, 100.0, 50, TimeInForce.IOC)
    fills = engine.match_order(ioc_order)
    print(f"  IOC order: {len(fills)} fills")
    
    # Test FOK order (should fill completely or not at all)
    fok_order = create_test_order("FOK1", OrderSide.BUY, 100.0, 200, TimeInForce.FOK)
    fills = engine.match_order(fok_order)
    print(f"  FOK order: {len(fills)} fills")
    
    return True

def test_cancellation():
    """Test order cancellation."""
    print("\n🧪 Testing order cancellation...")
    
    engine = CppMatchEngine("TEST")
    
    # Add an order
    order = create_test_order("ORDER1", OrderSide.BUY, 100.0, 100)
    fills = engine.match_order(order)
    print(f"  Order added: {len(fills)} fills, {engine.get_order_count()} active orders")
    
    # Cancel the order
    success = engine.cancel_order("ORDER1")
    print(f"  Cancellation: {success}, {engine.get_order_count()} active orders")
    
    return success

def test_performance():
    """Test performance with many orders."""
    print("\n🧪 Testing performance...")
    
    engine = CppMatchEngine("TEST")
    
    # Generate random orders
    orders = []
    for i in range(1000):
        side = OrderSide.BUY if random.random() < 0.5 else OrderSide.SELL
        price = 100.0 + random.uniform(-10, 10)
        quantity = random.randint(10, 100)
        order = create_test_order(f"ORDER{i}", side, price, quantity)
        orders.append(order)
    
    # Time the matching
    start_time = time.time()
    total_fills = 0
    
    for order in orders:
        fills = engine.match_order(order)
        total_fills += len(fills)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"  Processed {len(orders)} orders in {duration:.4f} seconds")
    print(f"  Rate: {len(orders)/duration:.0f} orders/second")
    print(f"  Total fills: {total_fills}")
    print(f"  Final active orders: {engine.get_order_count()}")
    
    return duration < 1.0  # Should be very fast

def main():
    """Run all tests."""
    print("🚀 Testing C++ Matching Engine Extension")
    print("=" * 50)
    
    if not CPP_MATCHING_AVAILABLE:
        print("❌ C++ extension not available")
        return False
    
    tests = [
        ("Basic Matching", test_basic_matching),
        ("Partial Fills", test_partial_fills),
        ("IOC/FOK Orders", test_ioc_fok_orders),
        ("Cancellation", test_cancellation),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! C++ extension is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
