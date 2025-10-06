"""Unit tests for market orders functionality."""

import pytest
from flashback.market.orders import (
    Order, Fill, Cancel, OrderSide, OrderType, TimeInForce,
    OrderBookLevel, OrderBookSnapshot
)


class TestOrder:
    """Test Order dataclass functionality."""
    
    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            order_id="12345",
            timestamp=1704067200000000000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        assert order.order_id == "12345"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.price == 150.0
        assert order.quantity == 100
        assert order.remaining_qty == 100
        assert order.filled_qty == 0
        assert order.status == "NEW"
    
    def test_order_side_methods(self):
        """Test order side helper methods."""
        buy_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        sell_order = Order("2", 1000, "AAPL", OrderSide.SELL, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        assert buy_order.is_buy()
        assert not buy_order.is_sell()
        assert sell_order.is_sell()
        assert not sell_order.is_buy()
    
    def test_order_type_methods(self):
        """Test order type helper methods."""
        limit_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        market_order = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.MARKET)
        
        assert limit_order.is_limit()
        assert not limit_order.is_market()
        assert market_order.is_market()
        assert not market_order.is_limit()
    
    def test_time_in_force_methods(self):
        """Test time in force helper methods."""
        day_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        ioc_order = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.IOC, OrderType.LIMIT)
        fok_order = Order("3", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.FOK, OrderType.LIMIT)
        
        assert day_order.is_day_order()
        assert ioc_order.is_ioc()
        assert fok_order.is_fok()
    
    def test_order_status_methods(self):
        """Test order status helper methods."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        assert order.is_active()
        assert not order.is_filled()
        assert not order.is_cancelled()
        assert not order.is_rejected()
        
        order.status = "FILLED"
        assert not order.is_active()
        assert order.is_filled()
        
        order.status = "CANCELLED"
        assert not order.is_active()
        assert order.is_cancelled()
    
    def test_fill_ratios(self):
        """Test fill ratio calculations."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        assert order.get_fill_ratio() == 0.0
        assert order.get_remaining_ratio() == 1.0
        
        order.filled_qty = 30
        order.remaining_qty = 70
        
        assert order.get_fill_ratio() == 0.3
        assert order.get_remaining_ratio() == 0.7
    
    def test_update_fill(self):
        """Test order fill updates."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # First fill
        order.update_fill(30, 150.0)
        assert order.filled_qty == 30
        assert order.remaining_qty == 70
        assert order.avg_fill_price == 150.0
        assert order.status == "PARTIALLY_FILLED"
        
        # Second fill
        order.update_fill(20, 151.0)
        assert order.filled_qty == 50
        assert order.remaining_qty == 50
        assert order.avg_fill_price == 150.4  # (30*150 + 20*151) / 50
        assert order.status == "PARTIALLY_FILLED"
        
        # Complete fill
        order.update_fill(50, 152.0)
        assert order.filled_qty == 100
        assert order.remaining_qty == 0
        assert order.status == "FILLED"
    
    def test_update_fill_invalid_quantity(self):
        """Test fill update with invalid quantity."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        with pytest.raises(ValueError, match="Fill quantity 150 exceeds remaining 100"):
            order.update_fill(150, 150.0)
    
    def test_cancel_order(self):
        """Test order cancellation."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        order.cancel()
        assert order.status == "CANCELLED"
        assert order.remaining_qty == 0
    
    def test_cancel_filled_order(self):
        """Test cancelling a filled order."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order.status = "FILLED"
        
        with pytest.raises(ValueError, match="Cannot cancel order with status FILLED"):
            order.cancel()
    
    def test_reject_order(self):
        """Test order rejection."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        order.reject("INSUFFICIENT_FUNDS")
        assert order.status == "REJECTED"
        assert order.remaining_qty == 0
    
    def test_to_dict_from_dict(self):
        """Test order serialization."""
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order.filled_qty = 30
        order.remaining_qty = 70
        order.avg_fill_price = 150.5
        order.status = "PARTIALLY_FILLED"
        
        data = order.to_dict()
        restored_order = Order.from_dict(data)
        
        assert restored_order.order_id == order.order_id
        assert restored_order.symbol == order.symbol
        assert restored_order.side == order.side
        assert restored_order.price == order.price
        assert restored_order.quantity == order.quantity
        assert restored_order.time_in_force == order.time_in_force
        assert restored_order.order_type == order.order_type
        assert restored_order.filled_qty == order.filled_qty
        assert restored_order.remaining_qty == order.remaining_qty
        assert restored_order.avg_fill_price == order.avg_fill_price
        assert restored_order.status == order.status


class TestFill:
    """Test Fill dataclass functionality."""
    
    def test_fill_creation(self):
        """Test basic fill creation."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            timestamp=1704067200000000000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            commission=0.5,
            maker_taker="TAKER"
        )
        
        assert fill.fill_id == "fill_123"
        assert fill.order_id == "order_456"
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.price == 150.0
        assert fill.quantity == 100
        assert fill.commission == 0.5
        assert fill.maker_taker == "TAKER"
    
    def test_fill_side_methods(self):
        """Test fill side helper methods."""
        buy_fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100)
        sell_fill = Fill("1", "2", 1000, "AAPL", OrderSide.SELL, 150.0, 100)
        
        assert buy_fill.is_buy()
        assert not buy_fill.is_sell()
        assert sell_fill.is_sell()
        assert not sell_fill.is_buy()
    
    def test_fill_maker_taker_methods(self):
        """Test fill maker/taker helper methods."""
        maker_fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        taker_fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="TAKER")
        
        assert maker_fill.is_maker()
        assert not maker_fill.is_taker()
        assert taker_fill.is_taker()
        assert not taker_fill.is_maker()
    
    def test_fill_calculations(self):
        """Test fill value calculations."""
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, commission=0.5)
        
        assert fill.get_notional() == 15000.0
        assert fill.get_net_value() == 14999.5
    
    def test_fill_serialization(self):
        """Test fill serialization."""
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, commission=0.5, maker_taker="TAKER")
        
        data = fill.to_dict()
        restored_fill = Fill.from_dict(data)
        
        assert restored_fill.fill_id == fill.fill_id
        assert restored_fill.order_id == fill.order_id
        assert restored_fill.timestamp == fill.timestamp
        assert restored_fill.symbol == fill.symbol
        assert restored_fill.side == fill.side
        assert restored_fill.price == fill.price
        assert restored_fill.quantity == fill.quantity
        assert restored_fill.commission == fill.commission
        assert restored_fill.maker_taker == fill.maker_taker


class TestCancel:
    """Test Cancel dataclass functionality."""
    
    def test_cancel_creation(self):
        """Test basic cancel creation."""
        cancel = Cancel(
            cancel_id="cancel_123",
            order_id="order_456",
            timestamp=1704067200000000000,
            symbol="AAPL",
            cancelled_qty=100,
            reason="USER_REQUEST"
        )
        
        assert cancel.cancel_id == "cancel_123"
        assert cancel.order_id == "order_456"
        assert cancel.symbol == "AAPL"
        assert cancel.cancelled_qty == 100
        assert cancel.reason == "USER_REQUEST"
    
    def test_cancel_serialization(self):
        """Test cancel serialization."""
        cancel = Cancel("1", "2", 1000, "AAPL", 100, "USER_REQUEST")
        
        data = cancel.to_dict()
        restored_cancel = Cancel.from_dict(data)
        
        assert restored_cancel.cancel_id == cancel.cancel_id
        assert restored_cancel.order_id == cancel.order_id
        assert restored_cancel.timestamp == cancel.timestamp
        assert restored_cancel.symbol == cancel.symbol
        assert restored_cancel.cancelled_qty == cancel.cancelled_qty
        assert restored_cancel.reason == cancel.reason


class TestOrderBookLevel:
    """Test OrderBookLevel functionality."""
    
    def test_level_creation(self):
        """Test basic level creation."""
        level = OrderBookLevel(price=150.0, total_qty=1000, order_count=5)
        
        assert level.price == 150.0
        assert level.total_qty == 1000
        assert level.order_count == 5
        assert len(level.orders) == 0
    
    def test_add_remove_order(self):
        """Test adding and removing orders from level."""
        level = OrderBookLevel(price=150.0, total_qty=0, order_count=0)
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Add order
        level.add_order(order)
        assert len(level.orders) == 1
        assert level.total_qty == 100
        assert level.order_count == 1
        assert order in level.orders
        
        # Remove order
        level.remove_order(order)
        assert len(level.orders) == 0
        assert level.total_qty == 0
        assert level.order_count == 0
        assert order not in level.orders
    
    def test_get_best_order(self):
        """Test getting best order from level."""
        level = OrderBookLevel(price=150.0, total_qty=0, order_count=0)
        
        # Empty level
        assert level.get_best_order() is None
        
        # Add orders
        order1 = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order2 = Order("2", 2000, "AAPL", OrderSide.BUY, 150.0, 200, TimeInForce.DAY, OrderType.LIMIT)
        
        level.add_order(order1)
        level.add_order(order2)
        
        # First order should be best (FIFO)
        assert level.get_best_order() == order1
    
    def test_is_empty(self):
        """Test empty level detection."""
        level = OrderBookLevel(price=150.0, total_qty=0, order_count=0)
        
        assert level.is_empty()
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        level.add_order(order)
        
        assert not level.is_empty()


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot functionality."""
    
    def test_snapshot_creation(self):
        """Test basic snapshot creation."""
        snapshot = OrderBookSnapshot(
            timestamp=1704067200000000000,
            symbol="AAPL",
            bids=[(150.0, 1000), (149.9, 500)],
            asks=[(150.1, 800), (150.2, 300)]
        )
        
        assert snapshot.timestamp == 1704067200000000000
        assert snapshot.symbol == "AAPL"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
    
    def test_best_prices(self):
        """Test best price calculations."""
        snapshot = OrderBookSnapshot(
            timestamp=1704067200000000000,
            symbol="AAPL",
            bids=[(150.0, 1000), (149.9, 500)],
            asks=[(150.1, 800), (150.2, 300)]
        )
        
        assert snapshot.get_best_bid() == (150.0, 1000)
        assert snapshot.get_best_ask() == (150.1, 800)
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        snapshot = OrderBookSnapshot(
            timestamp=1704067200000000000,
            symbol="AAPL",
            bids=[(150.0, 1000)],
            asks=[(150.1, 800)]
        )
        
        assert abs(snapshot.get_spread() - 0.1) < 1e-10
        assert snapshot.get_mid_price() == 150.05
    
    def test_empty_book(self):
        """Test empty book handling."""
        snapshot = OrderBookSnapshot(
            timestamp=1704067200000000000,
            symbol="AAPL",
            bids=[],
            asks=[]
        )
        
        assert snapshot.get_best_bid() is None
        assert snapshot.get_best_ask() is None
        assert snapshot.get_spread() is None
        assert snapshot.get_mid_price() is None
    
    def test_snapshot_serialization(self):
        """Test snapshot serialization."""
        snapshot = OrderBookSnapshot(
            timestamp=1704067200000000000,
            symbol="AAPL",
            bids=[(150.0, 1000)],
            asks=[(150.1, 800)]
        )
        
        data = snapshot.to_dict()
        
        assert data["timestamp"] == 1704067200000000000
        assert data["symbol"] == "AAPL"
        assert data["bids"] == [(150.0, 1000)]
        assert data["asks"] == [(150.1, 800)]
        assert data["best_bid"] == (150.0, 1000)
        assert data["best_ask"] == (150.1, 800)
        assert abs(data["spread"] - 0.1) < 1e-10
        assert data["mid_price"] == 150.05


if __name__ == "__main__":
    pytest.main([__file__])
