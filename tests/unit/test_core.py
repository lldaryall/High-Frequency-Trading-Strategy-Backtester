"""Unit tests for core modules."""

import pytest
import pandas as pd
from datetime import datetime

from flashback.core.clock import SimClock
from flashback.core.events import MarketDataEvent, OrderEvent, FillEvent
from flashback.core.engine import BacktestEngine


class TestSimClock:
    """Test SimClock class."""
    
    def test_clock_initialization(self):
        """Test clock initialization."""
        clock = SimClock()
        assert clock.now() is not None
        assert not clock.is_running()
        
    def test_clock_advance_time(self):
        """Test advancing clock time."""
        clock = SimClock()
        start_time = clock.now()
        clock.sleep_ns(1000000000)  # 1 second in nanoseconds
        assert clock.now() > start_time
        
    def test_clock_advance_to(self):
        """Test advancing clock to specific time."""
        clock = SimClock()
        target_time = clock.now() + 2000000000  # 2 seconds from now
        clock.advance_to(target_time)
        assert clock.now() == target_time
        
    def test_clock_schedule_timer(self):
        """Test scheduling timer events."""
        clock = SimClock()
        clock.schedule_timer("test_timer", 1000000000, "test_callback")
        assert clock.get_event_count() == 1
        
    def test_clock_get_next_event(self):
        """Test getting next event."""
        clock = SimClock()
        clock.add_data_event(clock.now() + 1000000000, "AAPL", {"price": 150.0})
        event = clock.get_next_event()
        assert event is not None
        assert event.symbol == "AAPL"
        
    def test_clock_reset(self):
        """Test clock reset."""
        clock = SimClock()
        clock.schedule_timer("test", 1000000000, "callback")
        clock.reset()
        assert clock.get_event_count() == 0


class TestEvents:
    """Test event classes."""
    
    def test_market_data_event(self):
        """Test MarketDataEvent creation."""
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        
        assert event.symbol == 'AAPL'
        assert event.side == 'B'
        assert event.price == 150.0
        assert event.size == 100
        assert event.event_type_str == 'TICK'
        
    def test_order_event(self):
        """Test OrderEvent creation."""
        event = OrderEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            order_id='order_1',
            symbol='AAPL',
            side='B',
            order_type='LIMIT',
            quantity=100,
            price=150.0
        )
        
        assert event.order_id == 'order_1'
        assert event.symbol == 'AAPL'
        assert event.side == 'B'
        assert event.order_type == 'LIMIT'
        assert event.quantity == 100
        assert event.price == 150.0
        
    def test_fill_event(self):
        """Test FillEvent creation."""
        event = FillEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            order_id='order_1',
            symbol='AAPL',
            side='B',
            quantity=100,
            price=150.0,
            commission=0.1,
            latency_us=50
        )
        
        assert event.order_id == 'order_1'
        assert event.symbol == 'AAPL'
        assert event.side == 'B'
        assert event.quantity == 100
        assert event.price == 150.0
        assert event.commission == 0.1
        assert event.latency_us == 50
        
    def test_event_to_dict(self):
        """Test event to dictionary conversion."""
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        
        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict['symbol'] == 'AAPL'
        assert event_dict['price'] == 150.0


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_engine_initialization(self, sample_config):
        """Test engine initialization."""
        engine = BacktestEngine(sample_config)
        assert engine.config == sample_config
        assert engine.clock is not None
        assert engine.data_loader is not None
        assert engine.order_router is not None
        assert engine.risk_manager is not None
        
    def test_engine_add_event(self, sample_config):
        """Test adding events to engine."""
        engine = BacktestEngine(sample_config)
        
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        
        engine.add_event(event)
        assert len(engine.event_queue) == 1
        
    def test_engine_get_next_event(self, sample_config):
        """Test getting next event from queue."""
        engine = BacktestEngine(sample_config)
        
        event1 = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        
        event2 = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:01'),
            symbol='AAPL',
            side='S',
            price=151.0,
            size=100,
            event_type_str='TICK'
        )
        
        engine.add_event(event2)  # Later event first
        engine.add_event(event1)  # Earlier event second
        
        next_event = engine._get_next_event()
        assert next_event.timestamp == event1.timestamp
        
    def test_engine_handle_market_data(self, sample_config):
        """Test handling market data events."""
        engine = BacktestEngine(sample_config)
        
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            symbol='AAPL',
            side='B',
            price=150.0,
            size=100,
            event_type_str='TICK'
        )
        
        engine._handle_market_data(event)
        assert 'AAPL' in engine.order_books
        assert engine.stats['events_processed'] == 1
        
    def test_engine_handle_order(self, sample_config):
        """Test handling order events."""
        engine = BacktestEngine(sample_config)
        
        event = OrderEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            order_id='order_1',
            symbol='AAPL',
            side='B',
            order_type='LIMIT',
            quantity=100,
            price=150.0
        )
        
        engine._handle_order(event)
        assert engine.stats['orders_placed'] == 1
        
    def test_engine_handle_fill(self, sample_config):
        """Test handling fill events."""
        engine = BacktestEngine(sample_config)
        
        event = FillEvent(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            order_id='order_1',
            symbol='AAPL',
            side='B',
            quantity=100,
            price=150.0
        )
        
        engine._handle_fill(event)
        assert engine.stats['fills_generated'] == 1
        
    def test_engine_generate_results(self, sample_config):
        """Test generating results."""
        engine = BacktestEngine(sample_config)
        
        # Add some test data
        engine.stats['events_processed'] = 100
        engine.stats['orders_placed'] = 10
        engine.stats['fills_generated'] = 8
        
        results = engine._generate_results()
        assert 'statistics' in results
        assert 'positions' in results
        assert 'pnl' in results
        assert 'trades' in results
        assert 'order_books' in results
