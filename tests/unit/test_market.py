"""Unit tests for market modules."""

import pytest
import pandas as pd
import numpy as np

from flashback.market.book import MatchingEngine
from flashback.market.orders import Order, OrderSide, OrderType
from flashback.market.matching import MatchingEngine
from flashback.market.fees import FeeCalculator
from flashback.market.latency import LatencyModel
from flashback.core.events import MarketDataEvent


class TestOrderBook:
    """Test OrderBook class."""
    
    def test_order_book_initialization(self):
        """Test order book initialization."""
        ob = OrderBook('AAPL')
        assert ob.symbol == 'AAPL'
        assert ob.last_bid is None
        assert ob.last_ask is None
        assert ob.last_mid is None
        
    def test_order_book_update(self):
        """Test updating order book with market data."""
        ob = OrderBook('AAPL')
        
        # Update with bid
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        ob.update(event)
        assert ob.last_bid == 150.0
        
        # Update with ask
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:01'),
            symbol='AAPL',
            side='S',
            price=151.0,
            size=100,
            event_type_str='TICK'
        )
        ob.update(event)
        assert ob.last_ask == 151.0
        assert ob.last_mid == 150.5
        assert ob.last_spread == 1.0
        
    def test_order_book_add_order(self):
        """Test adding orders to order book."""
        ob = OrderBook('AAPL')
        
        order = Order(
            order_id='order_1',
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        success = ob.add_order(order)
        assert success
        assert 'order_1' in ob.orders
        assert 150.0 in ob.bid_levels
        
    def test_order_book_cancel_order(self):
        """Test cancelling orders."""
        ob = OrderBook('AAPL')
        
        order = Order(
            order_id='order_1',
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        ob.add_order(order)
        success = ob.cancel_order('order_1')
        assert success
        assert order.status == OrderStatus.CANCELLED
        
    def test_order_book_get_best_bid_ask(self):
        """Test getting best bid and ask."""
        ob = OrderBook('AAPL')
        
        # Add multiple orders
        order1 = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        order2 = Order('order_2', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 50, 149.0)
        order3 = Order('order_3', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 75, 151.0)
        order4 = Order('order_4', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 25, 152.0)
        
        ob.add_order(order1)
        ob.add_order(order2)
        ob.add_order(order3)
        ob.add_order(order4)
        
        best_bid = ob.get_best_bid()
        best_ask = ob.get_best_ask()
        
        assert best_bid[0] == 150.0  # Highest bid price
        assert best_ask[0] == 151.0  # Lowest ask price
        
    def test_order_book_get_snapshot(self):
        """Test getting order book snapshot."""
        ob = OrderBook('AAPL')
        
        # Add orders
        order1 = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        order2 = Order('order_2', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 50, 151.0)
        
        ob.add_order(order1)
        ob.add_order(order2)
        
        snapshot = ob.get_snapshot()
        assert 'bids' in snapshot
        assert 'asks' in snapshot
        assert len(snapshot['bids']) == 1
        assert len(snapshot['asks']) == 1


class TestMatchingEngine:
    """Test MatchingEngine class."""
    
    def test_matching_engine_initialization(self):
        """Test matching engine initialization."""
        engine = MatchingEngine()
        assert engine.partial_fills_enabled is True
        assert engine.min_tick_size == 0.01
        
    def test_match_market_order(self):
        """Test matching market orders."""
        engine = MatchingEngine()
        ob = OrderBook('AAPL')
        
        # Add some orders to the book
        order1 = Order('order_1', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 100, 150.0)
        order2 = Order('order_2', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 50, 151.0)
        ob.add_order(order1)
        ob.add_order(order2)
        
        # Create market buy order
        market_order = Order('market_1', 'AAPL', OrderSide.BUY, OrderType.MARKET, 75, None)
        
        fills = engine.match_market_order(market_order, ob)
        assert len(fills) > 0
        assert market_order.is_filled
        
    def test_match_limit_order(self):
        """Test matching limit orders."""
        engine = MatchingEngine()
        ob = OrderBook('AAPL')
        
        # Add some orders to the book
        order1 = Order('order_1', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 100, 150.0)
        ob.add_order(order1)
        
        # Create limit buy order that crosses the spread
        limit_order = Order('limit_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 50, 150.0)
        
        fills = engine.match_limit_order(limit_order, ob)
        assert len(fills) > 0
        
    def test_match_ioc_order(self):
        """Test matching IOC orders."""
        engine = MatchingEngine()
        ob = OrderBook('AAPL')
        
        # Add some orders to the book
        order1 = Order('order_1', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 100, 150.0)
        ob.add_order(order1)
        
        # Create IOC order
        ioc_order = Order('ioc_1', 'AAPL', OrderSide.BUY, OrderType.IOC, 50, None)
        
        fills = engine.match_ioc_order(ioc_order, ob)
        assert len(fills) > 0
        assert ioc_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]
        
    def test_match_fok_order(self):
        """Test matching FOK orders."""
        engine = MatchingEngine()
        ob = OrderBook('AAPL')
        
        # Add some orders to the book
        order1 = Order('order_1', 'AAPL', OrderSide.SELL, OrderType.LIMIT, 100, 150.0)
        ob.add_order(order1)
        
        # Create FOK order that can be filled
        fok_order = Order('fok_1', 'AAPL', OrderSide.BUY, OrderType.FOK, 50, None)
        
        fills = engine.match_fok_order(fok_order, ob)
        assert len(fills) > 0
        assert fok_order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]


class TestFeeCalculator:
    """Test FeeCalculator class."""
    
    def test_fee_calculator_initialization(self):
        """Test fee calculator initialization."""
        calc = FeeCalculator()
        assert calc.commission_per_share == 0.001
        assert calc.commission_per_trade == 0.0
        
    def test_calculate_fees(self):
        """Test calculating fees for a fill."""
        calc = FeeCalculator()
        
        fill = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='FILL'
        )
        
        fees = calc.calculate_fees(fill)
        assert 'total' in fees
        assert fees['total'] > 0
        
    def test_calculate_commission(self):
        """Test calculating commission for an order."""
        calc = FeeCalculator()
        
        order = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        commission = calc.calculate_commission(order)
        assert commission > 0
        
    def test_calculate_slippage(self):
        """Test calculating slippage."""
        calc = FeeCalculator()
        
        order = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        fill_price = 150.5
        slippage = calc.calculate_slippage(order, fill_price)
        assert slippage >= 0


class TestLatencyModel:
    """Test LatencyModel class."""
    
    def test_latency_model_initialization(self):
        """Test latency model initialization."""
        model = LatencyModel()
        assert model.base_latency_us == 100
        assert model.jitter_us == 10
        
    def test_calculate_latency(self):
        """Test calculating latency for an order."""
        model = LatencyModel()
        
        order = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        latency = model.calculate_latency(order)
        assert latency > 0
        
    def test_calculate_latency_with_market_conditions(self):
        """Test calculating latency with market conditions."""
        model = LatencyModel()
        
        order = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        market_conditions = {
            'volatility': 0.02,
            'volume': 1000000
        }
        
        latency = model.calculate_latency(order, market_conditions)
        assert latency > 0
        
    def test_calculate_queue_latency(self):
        """Test calculating queue latency."""
        model = LatencyModel()
        
        order = Order('order_1', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 100, 150.0)
        queue_latency = model.calculate_queue_latency(order, 5)
        assert queue_latency > 0
        
    def test_simulate_latency_distribution(self):
        """Test simulating latency distribution."""
        model = LatencyModel()
        
        latencies = model.simulate_latency_distribution(1000)
        assert len(latencies) == 1000
        assert all(l > 0 for l in latencies)
        
    def test_get_latency_statistics(self):
        """Test getting latency statistics."""
        model = LatencyModel()
        
        latencies = [100, 150, 200, 250, 300]
        stats = model.get_latency_statistics(latencies)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
