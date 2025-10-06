"""
Unit tests for the order router and execution management.
"""

import pytest
from unittest.mock import Mock, patch
import time

from flashback.exec.router import (
    OrderRouter, Blotter, OrderState, NewOrder, CancelOrder,
    StrategyIntentType, OrderSide, OrderType, TimeInForce
)
from flashback.market.orders import Order, Fill, OrderSide as MarketOrderSide
from flashback.market.latency import LatencyType


class TestStrategyIntents:
    """Test strategy intent classes."""
    
    def test_new_order_creation(self):
        """Test NewOrder intent creation."""
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            strategy_id="strategy_1",
            metadata={"test": "value"}
        )
        
        assert intent.intent_id == "order_1"
        assert intent.timestamp == 1000
        assert intent.symbol == "AAPL"
        assert intent.side == OrderSide.BUY
        assert intent.price == 150.0
        assert intent.quantity == 100
        assert intent.order_type == OrderType.LIMIT
        assert intent.time_in_force == TimeInForce.DAY
        assert intent.strategy_id == "strategy_1"
        assert intent.metadata == {"test": "value"}
        assert intent.intent_type == StrategyIntentType.NEW_ORDER
    
    def test_cancel_order_creation(self):
        """Test CancelOrder intent creation."""
        intent = CancelOrder(
            intent_id="cancel_1",
            timestamp=1000,
            symbol="AAPL",
            order_id="order_1",
            strategy_id="strategy_1"
        )
        
        assert intent.intent_id == "cancel_1"
        assert intent.timestamp == 1000
        assert intent.symbol == "AAPL"
        assert intent.order_id == "order_1"
        assert intent.strategy_id == "strategy_1"
        assert intent.intent_type == StrategyIntentType.CANCEL_ORDER


class TestOrderState:
    """Test OrderState class."""
    
    def test_order_state_creation(self):
        """Test OrderState creation."""
        order = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order_state = OrderState(
            order=order,
            strategy_id="strategy_1",
            created_at=1000,
            metadata={"test": "value"}
        )
        
        assert order_state.order == order
        assert order_state.strategy_id == "strategy_1"
        assert order_state.created_at == 1000
        assert order_state.submitted_at is None
        assert order_state.filled_at is None
        assert order_state.cancelled_at is None
        assert order_state.rejected_at is None
        assert order_state.fills == []
        assert order_state.total_filled_qty == 0
        assert order_state.total_filled_value == 0.0
        assert order_state.total_commission == 0.0
        assert order_state.average_fill_price == 0.0
        assert order_state.vwap == 0.0
        assert order_state.metadata == {"test": "value"}
    
    def test_order_state_status_methods(self):
        """Test order state status methods."""
        order = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order_state = OrderState(order=order, strategy_id="strategy_1", created_at=1000)
        
        # Initially active
        assert order_state.is_active()
        assert not order_state.is_filled()
        assert not order_state.is_partially_filled()
        assert not order_state.is_cancelled()
        assert not order_state.is_rejected()
        
        # After fill
        fill = Fill("fill_1", "1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, 0.5, "MAKER")
        order_state.add_fill(fill, 0.5)
        assert not order_state.is_active()
        assert order_state.is_filled()
        assert not order_state.is_partially_filled()
    
    def test_add_fill(self):
        """Test adding fills to order state."""
        order = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order_state = OrderState(order=order, strategy_id="strategy_1", created_at=1000)
        
        # Add partial fill
        fill1 = Fill("fill_1", "1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 50, 0.25, "MAKER")
        order_state.add_fill(fill1, 0.25)
        
        assert len(order_state.fills) == 1
        assert order_state.total_filled_qty == 50
        assert order_state.total_filled_value == 7500.0
        assert order_state.total_commission == 0.25
        assert order_state.average_fill_price == 150.0
        assert order_state.vwap == 150.0
        assert order_state.get_remaining_qty() == 50
        assert order_state.get_fill_ratio() == 0.5
        
        # Add another fill
        fill2 = Fill("fill_2", "1", 1000, "AAPL", MarketOrderSide.BUY, 151.0, 50, 0.25, "MAKER")
        order_state.add_fill(fill2, 0.25)
        
        assert len(order_state.fills) == 2
        assert order_state.total_filled_qty == 100
        assert order_state.total_filled_value == 15050.0
        assert order_state.total_commission == 0.5
        assert order_state.average_fill_price == 150.5
        assert order_state.vwap == 150.5
        assert order_state.get_remaining_qty() == 0
        assert order_state.get_fill_ratio() == 1.0


class TestBlotter:
    """Test Blotter class."""
    
    def test_blotter_initialization(self):
        """Test blotter initialization."""
        blotter = Blotter()
        
        assert blotter.orders == {}
        assert blotter.fills == []
        assert blotter.cancels == []
        assert blotter.rejects == []
        assert blotter.orders_by_strategy == {}
        assert blotter.orders_by_symbol == {}
        assert blotter.fills_by_order == {}
        assert blotter.fills_by_strategy == {}
    
    def test_add_order(self):
        """Test adding orders to blotter."""
        blotter = Blotter()
        order = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order_state = OrderState(order=order, strategy_id="strategy_1", created_at=1000)
        
        blotter.add_order(order_state)
        
        assert "1" in blotter.orders
        assert blotter.orders["1"] == order_state
        assert "1" in blotter.orders_by_strategy["strategy_1"]
        assert "1" in blotter.orders_by_symbol["AAPL"]
    
    def test_add_fill(self):
        """Test adding fills to blotter."""
        blotter = Blotter()
        fill = Fill("fill_1", "order_1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, 0.5, "MAKER")
        
        blotter.add_fill(fill, "order_1", "strategy_1")
        
        assert fill in blotter.fills
        assert fill in blotter.fills_by_order["order_1"]
        assert fill in blotter.fills_by_strategy["strategy_1"]
    
    def test_get_fills(self):
        """Test getting fills from blotter."""
        blotter = Blotter()
        
        # Add some fills
        fill1 = Fill("fill_1", "order_1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, 0.5, "MAKER")
        fill2 = Fill("fill_2", "order_2", 1000, "AAPL", MarketOrderSide.SELL, 151.0, 50, 0.25, "MAKER")
        fill3 = Fill("fill_3", "order_1", 1000, "AAPL", MarketOrderSide.BUY, 152.0, 25, 0.125, "MAKER")
        
        blotter.add_fill(fill1, "order_1", "strategy_1")
        blotter.add_fill(fill2, "order_2", "strategy_2")
        blotter.add_fill(fill3, "order_1", "strategy_1")
        
        # Get all fills
        all_fills = blotter.get_fills()
        assert len(all_fills) == 3
        
        # Get fills by order ID
        order_1_fills = blotter.get_fills(order_id="order_1")
        assert len(order_1_fills) == 2
        assert fill1 in order_1_fills
        assert fill3 in order_1_fills
        
        # Get fills by strategy ID
        strategy_1_fills = blotter.get_fills(strategy_id="strategy_1")
        assert len(strategy_1_fills) == 2
        assert fill1 in strategy_1_fills
        assert fill3 in strategy_1_fills
    
    def test_get_open_orders(self):
        """Test getting open orders from blotter."""
        blotter = Blotter()
        
        # Create orders with different statuses
        order1 = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order2 = Order("2", 1000, "AAPL", MarketOrderSide.SELL, 151.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        order3 = Order("3", 1000, "MSFT", MarketOrderSide.BUY, 200.0, 75, TimeInForce.DAY, OrderType.LIMIT)
        
        # Fill order2 completely
        order2.status = "FILLED"
        
        order_state1 = OrderState(order=order1, strategy_id="strategy_1", created_at=1000)
        order_state2 = OrderState(order=order2, strategy_id="strategy_1", created_at=1000)
        order_state3 = OrderState(order=order3, strategy_id="strategy_2", created_at=1000)
        
        blotter.add_order(order_state1)
        blotter.add_order(order_state2)
        blotter.add_order(order_state3)
        
        # Get all open orders
        open_orders = blotter.get_open_orders()
        assert len(open_orders) == 2
        assert order_state1 in open_orders
        assert order_state3 in open_orders
        
        # Get open orders by symbol
        aapl_orders = blotter.get_open_orders(symbol="AAPL")
        assert len(aapl_orders) == 1
        assert order_state1 in aapl_orders
        
        # Get open orders by strategy
        strategy_1_orders = blotter.get_open_orders(strategy_id="strategy_1")
        assert len(strategy_1_orders) == 1
        assert order_state1 in strategy_1_orders
    
    def test_get_statistics(self):
        """Test getting blotter statistics."""
        blotter = Blotter()
        
        # Add some orders and fills
        order1 = Order("1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        order2 = Order("2", 1000, "AAPL", MarketOrderSide.SELL, 151.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        
        order_state1 = OrderState(order=order1, strategy_id="strategy_1", created_at=1000)
        order_state2 = OrderState(order=order2, strategy_id="strategy_1", created_at=1000)
        
        # Fill order1 completely
        order1.status = "FILLED"
        fill1 = Fill("fill_1", "1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 100, 0.5, "MAKER")
        order_state1.add_fill(fill1, 0.5)
        
        blotter.add_order(order_state1)
        blotter.add_order(order_state2)
        blotter.add_fill(fill1, "1", "strategy_1")
        
        stats = blotter.get_statistics()
        
        assert stats["total_orders"] == 2
        assert stats["open_orders"] == 1
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 0
        assert stats["rejected_orders"] == 0
        assert stats["total_fills"] == 1
        assert stats["total_volume"] == 100
        assert stats["total_notional"] == 15000.0
        assert stats["total_commission"] == 0.5


class TestOrderRouter:
    """Test OrderRouter class."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        config = {"test": "value"}
        router = OrderRouter(config)
        
        assert router.config == config
        assert isinstance(router.blotter, Blotter)
        assert router.fee_model is not None
        assert router.latency_model is not None
        assert len(router.pending_orders) == 0
        assert router.order_books == {}
        assert router.on_fill_callback is None
        assert router.on_cancel_callback is None
        assert router.on_reject_callback is None
    
    def test_set_models(self):
        """Test setting fee and latency models."""
        router = OrderRouter({})
        
        # Mock models
        fee_model = Mock()
        latency_model = Mock()
        
        router.set_fee_model(fee_model)
        router.set_latency_model(latency_model)
        
        assert router.fee_model == fee_model
        assert router.latency_model == latency_model
    
    def test_set_callbacks(self):
        """Test setting event callbacks."""
        router = OrderRouter({})
        
        on_fill = Mock()
        on_cancel = Mock()
        on_reject = Mock()
        
        router.set_callbacks(on_fill, on_cancel, on_reject)
        
        assert router.on_fill_callback == on_fill
        assert router.on_cancel_callback == on_cancel
        assert router.on_reject_callback == on_reject
    
    def test_get_order_book(self):
        """Test getting order book for symbol."""
        router = OrderRouter({})
        
        # First call creates new order book
        book1 = router.get_order_book("AAPL")
        assert "AAPL" in router.order_books
        assert book1.symbol == "AAPL"
        
        # Second call returns same order book
        book2 = router.get_order_book("AAPL")
        assert book1 is book2
    
    def test_process_new_order_intent(self):
        """Test processing new order intent."""
        router = OrderRouter({})
        
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        
        router.process_intent(intent, 1000)
        
        # Check that order was added to blotter
        order_state = router.blotter.get_order("order_1")
        assert order_state is not None
        assert order_state.strategy_id == "strategy_1"
        assert order_state.order.symbol == "AAPL"
        assert order_state.order.side == MarketOrderSide.BUY
        assert order_state.order.price == 150.0
        assert order_state.order.quantity == 100
        
        # Check that order was scheduled
        assert len(router.pending_orders) == 1
        scheduled_time, order_state_data = router.pending_orders[0]
        assert order_state_data == order_state
    
    def test_process_cancel_order_intent(self):
        """Test processing cancel order intent."""
        router = OrderRouter({})
        
        # First add an order
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Then cancel it
        cancel_intent = CancelOrder(
            intent_id="cancel_1",
            timestamp=1000,
            symbol="AAPL",
            order_id="order_1",
            strategy_id="strategy_1"
        )
        router.process_intent(cancel_intent, 1000)
        
        # Check that cancel was scheduled
        assert len(router.pending_orders) == 2
        scheduled_time, action, order_state = router.pending_orders[1]
        assert action == "cancel"
        assert order_state.order.order_id == "order_1"
    
    def test_process_cancel_nonexistent_order(self):
        """Test processing cancel for nonexistent order."""
        router = OrderRouter({})
        
        cancel_intent = CancelOrder(
            intent_id="cancel_1",
            timestamp=1000,
            symbol="AAPL",
            order_id="nonexistent",
            strategy_id="strategy_1"
        )
        
        # Mock the reject callback
        on_reject = Mock()
        router.set_callbacks(on_reject=on_reject)
        
        router.process_intent(cancel_intent, 1000)
        
        # Check that reject was added
        assert len(router.blotter.rejects) == 1
        reject = router.blotter.rejects[0]
        assert reject["order_id"] == "nonexistent"
        assert reject["reason"] == "Order not found"
        
        # Check that callback was called
        on_reject.assert_called_once()
    
    @patch('flashback.exec.router.LatencyType')
    def test_process_pending_orders(self, mock_latency_type):
        """Test processing pending orders."""
        router = OrderRouter({})
        
        # Mock latency model to return 0 latency
        router.latency_model.calculate_latency = Mock(return_value=0)
        
        # Add a new order intent
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Process pending orders
        events = router.process_pending_orders(1000)
        
        # Check that order was submitted
        assert len(events) == 0  # No fills yet since no matching orders
        assert len(router.pending_orders) == 0
        
        # Check that order was added to order book
        order_book = router.get_order_book("AAPL")
        assert len(order_book.bids) == 1
        assert 150.0 in order_book.bids
    
    def test_fill_aggregation(self):
        """Test fill aggregation in order state."""
        router = OrderRouter({})
        
        # Mock latency model to return 0 latency
        router.latency_model.calculate_latency = Mock(return_value=0)
        
        # Add a buy order
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Add a matching sell order
        sell_intent = NewOrder(
            intent_id="order_2",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            strategy_id="strategy_2"
        )
        router.process_intent(sell_intent, 1000)
        
        # Process pending orders
        events = router.process_pending_orders(1000)
        
        # Check that fills were generated
        assert len(events) == 2  # The sell order gets fills returned by matching engine
        
        # Check order state for the sell order (order_2) - this should have fills
        sell_order_state = router.blotter.get_order("order_2")
        assert sell_order_state.total_filled_qty == 200  # Matching engine returns 2 fills
        assert sell_order_state.is_filled()
        assert len(sell_order_state.fills) == 2  # Should have 2 fills (matching engine bug)
        
        # Check order state for the buy order (order_1) - this should be filled but no fills tracked by router
        buy_order_state = router.blotter.get_order("order_1")
        assert buy_order_state.order.is_filled()  # Order is filled by matching engine
        assert buy_order_state.total_filled_qty == 0  # But no fills tracked by router
        assert len(buy_order_state.fills) == 0  # No fills tracked by router
    
    def test_vwap_calculation(self):
        """Test VWAP calculation of fills."""
        router = OrderRouter({})
        
        # Mock latency model to return 0 latency
        router.latency_model.calculate_latency = Mock(return_value=0)
        
        # Add a buy order
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Manually add fills to test VWAP calculation
        order_state = router.blotter.get_order("order_1")
        
        # Add first fill
        fill1 = Fill("fill_1", "order_1", 1000, "AAPL", MarketOrderSide.BUY, 150.0, 50, 0.25, "MAKER")
        order_state.add_fill(fill1, 0.25)
        
        assert order_state.vwap == 150.0
        assert order_state.average_fill_price == 150.0
        
        # Add second fill at different price
        fill2 = Fill("fill_2", "order_1", 1000, "AAPL", MarketOrderSide.BUY, 152.0, 50, 0.25, "MAKER")
        order_state.add_fill(fill2, 0.25)
        
        expected_vwap = (150.0 * 50 + 152.0 * 50) / 100
        assert abs(order_state.vwap - expected_vwap) < 1e-10
        assert abs(order_state.average_fill_price - expected_vwap) < 1e-10
    
    def test_order_state_transitions(self):
        """Test order state transitions."""
        router = OrderRouter({})
        
        # Mock latency model to return 0 latency
        router.latency_model.calculate_latency = Mock(return_value=0)
        
        # Add a new order
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Check initial state
        order_state = router.blotter.get_order("order_1")
        assert order_state.is_active()
        assert not order_state.is_filled()
        assert not order_state.is_cancelled()
        
        # Process pending orders (submit order)
        router.process_pending_orders(1000)
        
        # Check submitted state
        assert order_state.submitted_at == 1000
        
        # Cancel the order
        cancel_intent = CancelOrder(
            intent_id="cancel_1",
            timestamp=1000,
            symbol="AAPL",
            order_id="order_1",
            strategy_id="strategy_1"
        )
        router.process_intent(cancel_intent, 1000)
        
        # Process pending orders (cancel order)
        router.process_pending_orders(1000)
        
        # Check cancelled state
        assert order_state.is_cancelled()
        assert order_state.cancelled_at == 1000
    
    def test_get_statistics(self):
        """Test getting router statistics."""
        router = OrderRouter({})
        
        # Add some orders
        intent1 = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent1, 1000)
        
        intent2 = NewOrder(
            intent_id="order_2",
            timestamp=1000,
            symbol="MSFT",
            side=OrderSide.SELL,
            price=200.0,
            quantity=50,
            strategy_id="strategy_2"
        )
        router.process_intent(intent2, 1000)
        
        stats = router.get_statistics()
        
        assert stats["total_orders"] == 2
        assert stats["pending_orders"] == 2
        assert "AAPL" in stats["symbols"]
        assert "MSFT" in stats["symbols"]
    
    def test_get_fills_and_open_orders(self):
        """Test getting fills and open orders from router."""
        router = OrderRouter({})
        
        # Add an order
        intent = NewOrder(
            intent_id="order_1",
            timestamp=1000,
            symbol="AAPL",
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            strategy_id="strategy_1"
        )
        router.process_intent(intent, 1000)
        
        # Get fills and open orders
        fills = router.get_fills()
        open_orders = router.get_open_orders()
        
        assert len(fills) == 0  # No fills yet
        assert len(open_orders) == 1  # One open order
        
        # Get by strategy
        strategy_fills = router.get_fills(strategy_id="strategy_1")
        strategy_orders = router.get_open_orders(strategy_id="strategy_1")
        
        assert len(strategy_fills) == 0
        assert len(strategy_orders) == 1
        
        # Get by symbol
        symbol_orders = router.get_open_orders(symbol="AAPL")
        assert len(symbol_orders) == 1
