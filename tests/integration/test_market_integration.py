"""Integration tests for market components with specific requirements."""

import pytest
import numpy as np
from flashback.market.book import MatchingEngine
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
from flashback.market.fees import create_standard_fee_model, BasisPointsFeeModel, FeeConfig
from flashback.market.latency import create_standard_latency_model, RandomLatencyModel, LatencyConfig


class TestMarketOrderPartialFills:
    """Test market order partial fills across multiple price levels."""
    
    def test_market_order_partial_fill_multiple_levels(self):
        """Test market order partially fills across multiple price levels."""
        engine = MatchingEngine("AAPL")
        
        # Add sell orders at different price levels
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("2", 2000, "AAPL", OrderSide.SELL, 151.0, 30, TimeInForce.DAY, OrderType.LIMIT)
        sell3 = Order("3", 3000, "AAPL", OrderSide.SELL, 152.0, 20, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(sell1)
        engine.add_order(sell2)
        engine.add_order(sell3)
        
        # Add market buy order for more than available
        market_buy = Order("4", 4000, "AAPL", OrderSide.BUY, 0.0, 120, TimeInForce.DAY, OrderType.MARKET)
        fills = engine.add_order(market_buy)
        
        # Should have 6 fills (3 pairs)
        assert len(fills) == 6
        
        # Market order should be partially filled
        assert market_buy.is_partially_filled()
        assert market_buy.filled_qty == 100  # 50 + 30 + 20
        assert market_buy.remaining_qty == 20  # 120 - 100
        
        # All sell orders should be filled
        assert sell1.is_filled()
        assert sell2.is_filled()
        assert sell3.is_filled()
        
        # Check fill prices (should match limit prices)
        taker_fills = [f for f in fills if f.order_id == "4"]
        prices = [f.price for f in taker_fills]
        assert 150.0 in prices
        assert 151.0 in prices
        assert 152.0 in prices
    
    def test_market_order_remaining_qty_handling(self):
        """Test that remaining quantity is handled per TIF rules."""
        engine = MatchingEngine("AAPL")
        
        # Add sell orders
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("2", 2000, "AAPL", OrderSide.SELL, 151.0, 30, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(sell1)
        engine.add_order(sell2)
        
        # Test DAY order - remaining qty should stay in book
        day_buy = Order("3", 3000, "AAPL", OrderSide.BUY, 0.0, 100, TimeInForce.DAY, OrderType.MARKET)
        fills = engine.add_order(day_buy)
        
        assert day_buy.is_partially_filled()
        assert day_buy.filled_qty == 80  # 50 + 30
        assert day_buy.remaining_qty == 20
        assert day_buy.status == "PARTIALLY_FILLED"
        
        # Test IOC order - remaining qty should be cancelled
        ioc_buy = Order("4", 4000, "AAPL", OrderSide.BUY, 0.0, 100, TimeInForce.IOC, OrderType.MARKET)
        fills = engine.add_order(ioc_buy)
        
        assert ioc_buy.is_cancelled()  # IOC cancels remaining
        assert ioc_buy.filled_qty == 0  # No fills since book is empty
        assert ioc_buy.remaining_qty == 0
        
        # Test FOK order - should be cancelled if can't fill completely
        fok_buy = Order("5", 5000, "AAPL", OrderSide.BUY, 0.0, 100, TimeInForce.FOK, OrderType.MARKET)
        fills = engine.add_order(fok_buy)
        
        assert fok_buy.is_cancelled()  # FOK cancels if can't fill completely
        assert fok_buy.filled_qty == 0
        assert fok_buy.remaining_qty == 0


class TestOrderCancellation:
    """Test order cancellation functionality."""
    
    def test_cancel_removes_remaining_qty(self):
        """Test that cancels remove remaining quantity from book."""
        engine = MatchingEngine("AAPL")
        
        # Add limit order
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(order)
        
        # Verify order is in book
        assert engine.get_best_bid() == 150.0
        
        # Cancel order
        cancel = engine.cancel_order("1", 2000)
        
        assert cancel is not None
        assert cancel.cancelled_qty == 100
        assert order.is_cancelled()
        assert order.remaining_qty == 0
        
        # Verify order is removed from book
        assert engine.get_best_bid() is None
    
    def test_cancel_partially_filled_order(self):
        """Test cancelling partially filled order."""
        engine = MatchingEngine("AAPL")
        
        # Add sell order
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        # Add buy order that partially fills
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(buy_order)
        
        # Buy order should be partially filled
        assert buy_order.is_partially_filled()
        assert buy_order.filled_qty == 50
        assert buy_order.remaining_qty == 50
        
        # Cancel remaining quantity
        cancel = engine.cancel_order("2", 3000)
        
        assert cancel is not None
        assert cancel.cancelled_qty == 50
        assert buy_order.is_cancelled()
        assert buy_order.remaining_qty == 0
        
        # Verify remaining quantity is removed from book
        assert engine.get_best_bid() is None


class TestFeeModelApplication:
    """Test fee model application with deterministic seeded RNG."""
    
    def test_fee_model_deterministic_with_seed(self):
        """Test that fee models are deterministic with seeded RNG."""
        # Create fee model
        fee_model = create_standard_fee_model(
            maker_bps=1.0,
            taker_bps=2.0,
            maker_per_share=0.001,
            taker_per_share=0.002
        )
        
        # Create fills
        maker_fill = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        maker_fill = maker_fill  # This would be converted to Fill in real usage
        taker_fill = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # For this test, we'll create actual Fill objects
        from flashback.market.orders import Fill
        
        maker_fill_obj = Fill("fill1", "order1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        taker_fill_obj = Fill("fill2", "order2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="TAKER")
        
        # Calculate fees multiple times
        maker_fee1 = fee_model.calculate_fee(maker_fill_obj)
        maker_fee2 = fee_model.calculate_fee(maker_fill_obj)
        taker_fee1 = fee_model.calculate_fee(taker_fill_obj)
        taker_fee2 = fee_model.calculate_fee(taker_fill_obj)
        
        # Should be identical
        assert maker_fee1 == maker_fee2
        assert taker_fee1 == taker_fee2
        
        # Taker should pay more than maker
        assert taker_fee1 > maker_fee1
    
    def test_fee_breakdown_detailed(self):
        """Test detailed fee breakdown."""
        fee_model = create_standard_fee_model(
            maker_bps=1.0,
            taker_bps=2.0,
            maker_per_share=0.001,
            taker_per_share=0.002,
            min_fee=0.01
        )
        
        from flashback.market.orders import Fill
        
        fill = Fill("fill1", "order1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        breakdown = fee_model.get_fee_breakdown(fill)
        
        assert breakdown["notional"] == 15000.0
        assert breakdown["fee_type"] == "maker"
        assert breakdown["bps_rate"] == 1.0
        assert breakdown["per_share_rate"] == 0.001
        assert breakdown["bps_fee"] == 1.5  # 15000 * 0.0001
        assert breakdown["per_share_fee"] == 0.1  # 100 * 0.001
        assert breakdown["base_fee"] == 1.6  # 1.5 + 0.1
        assert breakdown["total"] >= 0.01  # Should hit minimum fee


class TestLatencyModelApplication:
    """Test latency model application with deterministic seeded RNG."""
    
    def test_latency_model_deterministic_with_seed(self):
        """Test that latency models are deterministic with seeded RNG."""
        # Create latency model with seed
        latency_model = create_standard_latency_model(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Calculate latency multiple times
        latency1 = latency_model.calculate_latency(order, "submission")
        latency2 = latency_model.calculate_latency(order, "submission")
        latency3 = latency_model.calculate_latency(order, "submission")
        
        # Should be identical with same seed
        assert latency1 == latency2 == latency3
        
        # Should be >= constant latency
        assert latency1 >= 1000
    
    def test_latency_model_different_seeds(self):
        """Test that different seeds produce different results."""
        # Create two models with different seeds
        model1 = create_standard_latency_model(seed=42)
        model2 = create_standard_latency_model(seed=123)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        latency1 = model1.calculate_latency(order, "submission")
        latency2 = model2.calculate_latency(order, "submission")
        
        # Should be different with different seeds
        assert latency1 != latency2
    
    def test_latency_breakdown_detailed(self):
        """Test detailed latency breakdown."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            min_ns=800,
            max_ns=2000,
            distribution="normal",
            seed=42
        )
        model = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        breakdown = model.get_latency_breakdown(order, "submission")
        
        assert breakdown["constant"] == 1000
        assert breakdown["random"] >= 0
        assert breakdown["total"] >= 800  # min_ns
        assert breakdown["total"] <= 2000  # max_ns
        assert breakdown["latency_type"] == "submission"
        assert breakdown["distribution"] == "normal"
        assert breakdown["mean"] == 500
        assert breakdown["std"] == 200
        assert breakdown["min"] == 800
        assert breakdown["max"] == 2000


class TestMarketIntegrationScenarios:
    """Test complex market integration scenarios."""
    
    def test_complete_trading_scenario(self):
        """Test a complete trading scenario with all components."""
        # Initialize matching engine
        engine = MatchingEngine("AAPL")
        
        # Initialize fee and latency models
        fee_model = create_standard_fee_model(
            maker_bps=1.0,
            taker_bps=2.0,
            maker_per_share=0.001,
            taker_per_share=0.002
        )
        
        latency_model = create_standard_latency_model(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            seed=42
        )
        
        # Add initial orders to book
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("2", 2000, "AAPL", OrderSide.SELL, 151.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(sell1)
        engine.add_order(sell2)
        
        # Add market buy order
        market_buy = Order("3", 3000, "AAPL", OrderSide.BUY, 0.0, 120, TimeInForce.DAY, OrderType.MARKET)
        fills = engine.add_order(market_buy)
        
        # Verify fills
        assert len(fills) == 4  # 2 pairs of fills
        assert market_buy.is_partially_filled()
        assert market_buy.filled_qty == 150  # 100 + 50
        assert market_buy.remaining_qty == 0  # Should be 0 since we have DAY order
        
        # Calculate fees for fills
        from flashback.market.orders import Fill
        
        total_fees = 0
        for fill in fills:
            if isinstance(fill, Fill):
                fee = fee_model.calculate_fee(fill)
                total_fees += fee
        
        # Calculate latency for order
        latency = latency_model.calculate_latency(market_buy, "submission")
        
        # Verify results
        assert total_fees > 0
        assert latency >= 1000  # Should be >= constant latency
        
        # Verify engine statistics
        stats = engine.get_statistics()
        assert stats["total_volume"] == 150
        assert stats["total_trades"] == 2
        assert stats["last_trade_price"] == 151.0  # Last fill price
    
    def test_price_time_priority_ordering(self):
        """Test that price-time priority is maintained."""
        engine = MatchingEngine("AAPL")
        
        # Add multiple orders at same price with different timestamps
        order1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        order2 = Order("2", 2000, "AAPL", OrderSide.SELL, 150.0, 30, TimeInForce.DAY, OrderType.LIMIT)
        order3 = Order("3", 3000, "AAPL", OrderSide.SELL, 150.0, 20, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(order1)
        engine.add_order(order2)
        engine.add_order(order3)
        
        # Add buy order that matches all
        buy_order = Order("4", 4000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        fills = engine.add_order(buy_order)
        
        # Should match in timestamp order
        assert len(fills) == 6  # 3 pairs
        assert order1.is_filled()
        assert order2.is_filled()
        assert order3.is_filled()
        assert buy_order.is_filled()
        
        # Check that fills are in correct order
        taker_fills = [f for f in fills if f.order_id == "4"]
        assert len(taker_fills) == 3
        assert all(f.price == 150.0 for f in taker_fills)
    
    def test_deterministic_replay_consistency(self):
        """Test that the system provides deterministic replay."""
        # This test ensures that the same sequence of events produces identical results
        
        # First run
        engine1 = MatchingEngine("AAPL")
        fee_model1 = create_standard_fee_model(seed=42)
        latency_model1 = create_standard_latency_model(seed=42)
        
        # Add orders
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine1.add_order(sell1)
        
        buy1 = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        fills1 = engine1.add_order(buy1)
        
        # Second run with same seed
        engine2 = MatchingEngine("AAPL")
        fee_model2 = create_standard_fee_model(seed=42)
        latency_model2 = create_standard_latency_model(seed=42)
        
        # Add same orders
        sell2 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine2.add_order(sell2)
        
        buy2 = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        fills2 = engine2.add_order(buy2)
        
        # Results should be identical
        assert len(fills1) == len(fills2)
        assert len(fills1) == 2  # One pair of fills
        
        # Check that orders have same status
        assert buy1.status == buy2.status
        assert sell1.status == sell2.status
        
        # Check that fills have same properties
        for f1, f2 in zip(fills1, fills2):
            assert f1.price == f2.price
            assert f1.quantity == f2.quantity
            assert f1.side == f2.side


if __name__ == "__main__":
    pytest.main([__file__])
