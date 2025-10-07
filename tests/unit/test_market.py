"""Unit tests for market modules."""

import pytest
import pandas as pd
import numpy as np

from flashback.market.book import MatchingEngine
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce, Fill
from flashback.market.fees import FeeCalculator
from flashback.market.latency import RandomLatencyModel
from flashback.core.events import MarketDataEvent


class TestMatchingEngine:
    """Test MatchingEngine class."""
    
    def test_matching_engine_initialization(self):
        """Test matching engine initialization."""
        engine = MatchingEngine('AAPL')
        assert engine.symbol == 'AAPL'
        
    def test_match_market_order(self):
        """Test matching market orders."""
        engine = MatchingEngine('AAPL')
        
        # Add a limit order to the book
        limit_order = Order(
            order_id='limit_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        engine.add_order(limit_order)
        
        # Add a market order that should match
        market_order = Order(
            order_id='market_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:01').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=0.0,  # Market order
            quantity=50,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.MARKET
        )
        
        fills = engine.add_order(market_order)
        assert len(fills) == 1
        assert fills[0].order_id == 'market_1'
        assert fills[0].price == 150.0
        assert fills[0].quantity == 50
        
    def test_match_limit_order(self):
        """Test matching limit orders."""
        engine = MatchingEngine('AAPL')
        
        # Add a sell order
        sell_order = Order(
            order_id='sell_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        engine.add_order(sell_order)
        
        # Add a buy order that should match
        buy_order = Order(
            order_id='buy_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:01').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=50,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        fills = engine.add_order(buy_order)
        assert len(fills) == 1
        assert fills[0].order_id == 'buy_1'
        assert fills[0].price == 150.0
        assert fills[0].quantity == 50
        
    def test_match_ioc_order(self):
        """Test matching IOC orders."""
        engine = MatchingEngine('AAPL')
        
        # Add a sell order
        sell_order = Order(
            order_id='sell_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        engine.add_order(sell_order)
        
        # Add an IOC order that should match
        ioc_order = Order(
            order_id='ioc_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:01').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=50,
            time_in_force=TimeInForce.IOC,
            order_type=OrderType.LIMIT
        )
        
        fills = engine.add_order(ioc_order)
        assert len(fills) == 1
        assert fills[0].order_id == 'ioc_1'
        assert fills[0].price == 150.0
        assert fills[0].quantity == 50
        
    def test_match_fok_order(self):
        """Test matching FOK orders."""
        engine = MatchingEngine('AAPL')
        
        # Add a sell order
        sell_order = Order(
            order_id='sell_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.SELL,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        engine.add_order(sell_order)
        
        # Add a FOK order that should match
        fok_order = Order(
            order_id='fok_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:01').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=50,
            time_in_force=TimeInForce.FOK,
            order_type=OrderType.LIMIT
        )
        
        fills = engine.add_order(fok_order)
        assert len(fills) == 1
        assert fills[0].order_id == 'fok_1'
        assert fills[0].price == 150.0
        assert fills[0].quantity == 50


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
        
        # Create a mock fill
        fill = Fill(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100
        )
        
        fees = calc.calculate_fees(fill)
        assert 'total' in fees
        assert 'commission' in fees
        assert 'slippage' in fees
        
    def test_calculate_commission(self):
        """Test calculating commission for an order."""
        calc = FeeCalculator()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        commission = calc.calculate_commission(order)
        assert commission == 0.1  # 100 * 0.001
        
    def test_calculate_slippage(self):
        """Test calculating slippage."""
        calc = FeeCalculator()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.MARKET
        )
        
        slippage = calc.calculate_slippage(order, 150.1)
        assert slippage == 10.0  # (150.1 - 150.0) * 100


class TestLatencyModel:
    """Test LatencyModel class."""
    
    def test_latency_model_initialization(self):
        """Test latency model initialization."""
        model = RandomLatencyModel()
        assert model is not None
        
    def test_calculate_latency(self):
        """Test calculating latency for an order."""
        model = RandomLatencyModel()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        latency = model.calculate_latency(order)
        assert latency >= 0
        
    def test_calculate_latency_with_market_conditions(self):
        """Test calculating latency with market conditions."""
        model = RandomLatencyModel()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        market_data = {
            'volume': 1000,
            'volatility': 0.02,
            'spread': 0.01
        }
        
        latency = model.calculate_latency(order, market_data)
        assert latency >= 0
        
    def test_calculate_queue_latency(self):
        """Test calculating queue latency."""
        model = RandomLatencyModel()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        queue_position = 5
        latency = model.calculate_queue_latency(order, queue_position)
        assert latency >= 0
        
    def test_simulate_latency_distribution(self):
        """Test simulating latency distribution."""
        model = RandomLatencyModel()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        latencies = model.simulate_latency_distribution(order, 100)
        assert len(latencies) == 100
        assert all(l >= 0 for l in latencies)
        
    def test_get_latency_statistics(self):
        """Test getting latency statistics."""
        model = RandomLatencyModel()
        
        order = Order(
            order_id='order_1',
            timestamp=pd.Timestamp('2024-01-01 09:30:00').value,
            symbol='AAPL',
            side=OrderSide.BUY,
            price=150.0,
            quantity=100,
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.LIMIT
        )
        
        stats = model.get_latency_statistics(order)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats