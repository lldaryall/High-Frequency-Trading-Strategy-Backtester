"""Unit tests for market matching engine functionality."""

import pytest
from flashback.market.book import MatchingEngine
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce


class TestMatchingEngine:
    """Test MatchingEngine functionality."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = MatchingEngine("AAPL")
        
        assert engine.symbol == "AAPL"
        assert len(engine.orders) == 0
        assert len(engine.fills) == 0
        assert len(engine.cancels) == 0
        assert engine.total_volume == 0
        assert engine.total_trades == 0
        assert engine.last_trade_price is None
    
    def test_add_limit_buy_order(self):
        """Test adding a limit buy order."""
        engine = MatchingEngine("AAPL")
        
        order = Order(
            order_id="1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        fills = engine.add_order(order)
        
        assert len(fills) == 0  # No matching orders
        assert len(engine.orders) == 1
        assert engine.get_best_bid() == 150.0
        assert engine.get_best_ask() is None
    
    def test_add_limit_sell_order(self):
        """Test adding a limit sell order."""
        engine = MatchingEngine("AAPL")
        
        order = Order(
            order_id="1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.SELL,
            price=151.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        fills = engine.add_order(order)
        
        assert len(fills) == 0  # No matching orders
        assert len(engine.orders) == 1
        assert engine.get_best_bid() is None
        assert engine.get_best_ask() == 151.0
    
    def test_matching_buy_sell_orders(self):
        """Test matching buy and sell orders."""
        engine = MatchingEngine("AAPL")
        
        # Add sell order first
        sell_order = Order(
            order_id="1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        engine.add_order(sell_order)
        
        # Add matching buy order
        buy_order = Order(
            order_id="2",
            timestamp=2000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 2  # One for each side
        assert len(engine.orders) == 2
        assert buy_order.is_filled()
        assert sell_order.is_filled()
        assert engine.total_volume == 100
        assert engine.total_trades == 1
        assert engine.last_trade_price == 150.0
    
    def test_partial_fill(self):
        """Test partial fill scenario."""
        engine = MatchingEngine("AAPL")
        
        # Add sell order
        sell_order = Order(
            order_id="1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        engine.add_order(sell_order)
        
        # Add larger buy order
        buy_order = Order(
            order_id="2",
            timestamp=2000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=150,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 2
        assert sell_order.is_filled()
        assert buy_order.is_partially_filled()
        assert buy_order.filled_qty == 100
        assert buy_order.remaining_qty == 50
        assert engine.get_best_bid() == 150.0  # Remaining buy order in book
    
    def test_market_order_matching(self):
        """Test market order matching."""
        engine = MatchingEngine("AAPL")
        
        # Add limit sell order
        sell_order = Order(
            order_id="1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        engine.add_order(sell_order)
        
        # Add market buy order
        buy_order = Order(
            order_id="2",
            timestamp=2000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=0.0,  # Market order price
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.MARKET
        )
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 2
        assert buy_order.is_filled()
        assert sell_order.is_filled()
        assert fills[0].price == 150.0  # Matched at limit price
    
    def test_price_time_priority(self):
        """Test price-time priority matching."""
        engine = MatchingEngine("AAPL")
        
        # Add multiple sell orders at same price
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("2", 2000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell3 = Order("3", 3000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(sell1)
        engine.add_order(sell2)
        engine.add_order(sell3)
        
        # Add buy order that matches all
        buy_order = Order("4", 4000, "AAPL", OrderSide.BUY, 150.0, 150, TimeInForce.DAY, OrderType.LIMIT)
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 6  # 3 pairs of fills
        assert buy_order.is_filled()
        assert sell1.is_filled()
        assert sell2.is_filled()
        assert sell3.is_filled()
        
        # Check fill order (should be by timestamp)
        fill_times = [fill.timestamp for fill in fills if fill.order_id == "4"]
        assert fill_times == [4000, 4000, 4000]  # All at same time
    
    def test_ioc_order_behavior(self):
        """Test IOC (Immediate or Cancel) order behavior."""
        engine = MatchingEngine("AAPL")
        
        # Add sell order
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        # Add IOC buy order for more than available
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.IOC, OrderType.LIMIT)
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 2  # Partial fill
        assert buy_order.is_cancelled()  # Remaining quantity cancelled
        assert buy_order.filled_qty == 50
        assert sell_order.is_filled()
    
    def test_fok_order_behavior(self):
        """Test FOK (Fill or Kill) order behavior."""
        engine = MatchingEngine("AAPL")
        
        # Add sell order
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        # Add FOK buy order for more than available
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.FOK, OrderType.LIMIT)
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 0  # No fills
        assert buy_order.is_cancelled()  # Entire order cancelled
        assert buy_order.filled_qty == 0
        assert sell_order.is_active()  # Sell order still active
    
    def test_cancel_order(self):
        """Test order cancellation."""
        engine = MatchingEngine("AAPL")
        
        # Add order
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(order)
        
        # Cancel order
        cancel = engine.cancel_order("1", 2000)
        
        assert cancel is not None
        assert cancel.order_id == "1"
        assert cancel.cancelled_qty == 100
        assert order.is_cancelled()
        assert engine.get_best_bid() is None  # Order removed from book
    
    def test_cancel_nonexistent_order(self):
        """Test cancelling non-existent order."""
        engine = MatchingEngine("AAPL")
        
        cancel = engine.cancel_order("nonexistent", 2000)
        assert cancel is None
    
    def test_cancel_filled_order(self):
        """Test cancelling filled order."""
        engine = MatchingEngine("AAPL")
        
        # Add and fill order
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(buy_order)
        
        # Try to cancel filled order
        cancel = engine.cancel_order("2", 3000)
        assert cancel is None  # Cannot cancel filled order
    
    def test_get_snapshot(self):
        """Test order book snapshot."""
        engine = MatchingEngine("AAPL")
        
        # Add orders
        buy1 = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        buy2 = Order("2", 2000, "AAPL", OrderSide.BUY, 149.9, 200, TimeInForce.DAY, OrderType.LIMIT)
        sell1 = Order("3", 3000, "AAPL", OrderSide.SELL, 150.1, 150, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("4", 4000, "AAPL", OrderSide.SELL, 150.2, 250, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(buy1)
        engine.add_order(buy2)
        engine.add_order(sell1)
        engine.add_order(sell2)
        
        snapshot = engine.get_snapshot(5000)
        
        assert snapshot.symbol == "AAPL"
        assert snapshot.timestamp == 5000
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.get_best_bid() == (150.0, 100)
        assert snapshot.get_best_ask() == (150.1, 150)
        assert abs(snapshot.get_spread() - 0.1) < 1e-10
        assert snapshot.get_mid_price() == 150.05
    
    def test_get_statistics(self):
        """Test engine statistics."""
        engine = MatchingEngine("AAPL")
        
        # Add and match orders
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(buy_order)
        
        stats = engine.get_statistics()
        
        assert stats["symbol"] == "AAPL"
        assert stats["total_orders"] == 2
        assert stats["active_orders"] == 0
        assert stats["total_fills"] == 2
        assert stats["total_volume"] == 100
        assert stats["total_trades"] == 1
        assert stats["last_trade_price"] == 150.0
    
    def test_wrong_symbol_order(self):
        """Test adding order with wrong symbol."""
        engine = MatchingEngine("AAPL")
        
        order = Order("1", 1000, "MSFT", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        with pytest.raises(ValueError, match="Order symbol MSFT does not match engine symbol AAPL"):
            engine.add_order(order)
    
    def test_multiple_price_levels(self):
        """Test matching across multiple price levels."""
        engine = MatchingEngine("AAPL")
        
        # Add sell orders at different prices
        sell1 = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell2 = Order("2", 2000, "AAPL", OrderSide.SELL, 151.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        sell3 = Order("3", 3000, "AAPL", OrderSide.SELL, 152.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        
        engine.add_order(sell1)
        engine.add_order(sell2)
        engine.add_order(sell3)
        
        # Add market buy order that should match all
        buy_order = Order("4", 4000, "AAPL", OrderSide.BUY, 0.0, 150, TimeInForce.DAY, OrderType.MARKET)
        fills = engine.add_order(buy_order)
        
        assert len(fills) == 6  # 3 pairs of fills
        assert buy_order.is_filled()
        assert all(sell.is_filled() for sell in [sell1, sell2, sell3])
        
        # Check that fills are at correct prices
        taker_fills = [f for f in fills if f.order_id == "4"]
        prices = [f.price for f in taker_fills]
        assert 150.0 in prices
        assert 151.0 in prices
        assert 152.0 in prices
    
    def test_empty_book_after_fills(self):
        """Test that book is empty after all orders are filled."""
        engine = MatchingEngine("AAPL")
        
        # Add matching orders
        sell_order = Order("1", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(sell_order)
        
        buy_order = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        engine.add_order(buy_order)
        
        # Book should be empty
        assert engine.get_best_bid() is None
        assert engine.get_best_ask() is None
        assert len(engine.bids) == 0
        assert len(engine.asks) == 0


if __name__ == "__main__":
    pytest.main([__file__])
