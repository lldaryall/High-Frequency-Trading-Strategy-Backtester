"""Unit tests for SimClock functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, call
from typing import List

from flashback.core.clock import SimClock, Event, EventType, DataEvent, TimerEvent, ControlEvent


class TestSimClock:
    """Test SimClock functionality."""
    
    def test_clock_initialization(self):
        """Test clock initialization."""
        clock = SimClock()
        assert clock.now() > 0
        assert not clock.is_running()
        assert clock.get_event_count() == 0
        
    def test_clock_initialization_with_start_time(self):
        """Test clock initialization with specific start time."""
        start_time = 1704067200000000000  # 2024-01-01 00:00:00
        clock = SimClock(start_time)
        assert clock.now() == start_time
        
    def test_now_and_now_pd(self):
        """Test now() and now_pd() methods."""
        clock = SimClock()
        current_time = clock.now()
        current_time_pd = clock.now_pd()
        
        assert isinstance(current_time, int)
        assert isinstance(current_time_pd, pd.Timestamp)
        assert current_time_pd.value == current_time
        
    def test_sleep_ns(self):
        """Test sleep_ns() method."""
        clock = SimClock(1000)
        clock.sleep_ns(500)
        assert clock.now() == 1500
        
    def test_advance_to(self):
        """Test advance_to() method."""
        clock = SimClock(1000)
        clock.advance_to(2000)
        assert clock.now() == 2000
        
    def test_advance_to_past_time_raises_error(self):
        """Test that advancing to past time raises error."""
        clock = SimClock(2000)
        with pytest.raises(ValueError, match="Cannot advance to past timestamp"):
            clock.advance_to(1000)
            
    def test_schedule_timer(self):
        """Test scheduling a timer."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1", {"data": "test"})
        
        assert clock.get_event_count() == 1
        assert clock.has_events()
        
        event = clock.get_next_event()
        assert isinstance(event, TimerEvent)
        assert event.timer_id == "timer1"
        assert event.timestamp == 1500
        assert event.callback == "callback1"
        assert event.data == {"data": "test"}
        
    def test_schedule_timer_at(self):
        """Test scheduling a timer at specific time."""
        clock = SimClock(1000)
        clock.schedule_timer_at("timer1", 3000, "callback1")
        
        event = clock.get_next_event()
        assert event.timestamp == 3000
        
    def test_schedule_timer_at_past_time_raises_error(self):
        """Test that scheduling timer in past raises error."""
        clock = SimClock(2000)
        with pytest.raises(ValueError, match="Cannot schedule timer in the past"):
            clock.schedule_timer_at("timer1", 1000, "callback1")
            
    def test_cancel_timer(self):
        """Test canceling a timer."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        clock.schedule_timer("timer2", 1000, "callback2")
        
        assert clock.get_event_count() == 2
        
        # Cancel timer1
        cancelled = clock.cancel_timer("timer1")
        assert cancelled is True
        assert clock.get_event_count() == 1
        
        # Verify only timer2 remains
        event = clock.get_next_event()
        assert event.timer_id == "timer2"
        
    def test_cancel_nonexistent_timer(self):
        """Test canceling a non-existent timer."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        
        cancelled = clock.cancel_timer("nonexistent")
        assert cancelled is False
        assert clock.get_event_count() == 1
        
    def test_add_data_event(self):
        """Test adding data events."""
        clock = SimClock(1000)
        clock.add_data_event(1500, "AAPL", {"price": 150.0, "size": 100})
        
        event = clock.get_next_event()
        assert isinstance(event, DataEvent)
        assert event.timestamp == 1500
        assert event.symbol == "AAPL"
        assert event.data == {"price": 150.0, "size": 100}
        
    def test_add_control_event(self):
        """Test adding control events."""
        clock = SimClock(1000)
        clock.add_control_event(1500, "STOP", {"reason": "test"})
        
        event = clock.get_next_event()
        assert isinstance(event, ControlEvent)
        assert event.timestamp == 1500
        assert event.command == "STOP"
        assert event.data == {"reason": "test"}
        
    def test_event_ordering(self):
        """Test that events are processed in timestamp order."""
        clock = SimClock(1000)
        
        # Add events out of order
        clock.add_data_event(3000, "AAPL", {"price": 150.0})
        clock.schedule_timer("timer1", 1000, "callback1")  # 2000
        clock.add_data_event(1500, "MSFT", {"price": 200.0})
        clock.schedule_timer("timer2", 500, "callback2")   # 1500
        
        events = []
        while clock.has_events():
            events.append(clock.get_next_event())
            
        # Check ordering
        assert len(events) == 4
        assert events[0].timestamp == 1500  # timer2
        assert events[1].timestamp == 1500  # MSFT data
        assert events[2].timestamp == 2000  # timer1
        assert events[3].timestamp == 3000  # AAPL data
        
    def test_peek_next_event(self):
        """Test peeking at next event without removing it."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        
        # Peek at event
        peeked_event = clock.peek_next_event()
        assert peeked_event.timer_id == "timer1"
        
        # Event should still be in queue
        assert clock.get_event_count() == 1
        
        # Get event should return same event
        event = clock.get_next_event()
        assert event.timer_id == "timer1"
        assert clock.get_event_count() == 0
        
    def test_clear_events(self):
        """Test clearing all events."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        clock.add_data_event(1500, "AAPL", {"price": 150.0})
        
        assert clock.get_event_count() == 2
        clock.clear_events()
        assert clock.get_event_count() == 0
        assert not clock.has_events()
        
    def test_get_events_in_range(self):
        """Test getting events in time range."""
        clock = SimClock(1000)
        
        # Add events at different times
        clock.add_data_event(1500, "AAPL", {"price": 150.0})
        clock.schedule_timer("timer1", 1000, "callback1")  # 2000
        clock.add_data_event(2500, "MSFT", {"price": 200.0})
        clock.schedule_timer("timer2", 2000, "callback2")  # 3000
        
        # Get events in range 1500-2500 (inclusive)
        events = clock.get_events_in_range(1500, 2500)
        assert len(events) == 3
        
        timestamps = [event.timestamp for event in events]
        assert 1500 in timestamps
        assert 2000 in timestamps
        assert 2500 in timestamps
        assert 3000 not in timestamps
        
    def test_register_timer_callback(self):
        """Test registering timer callbacks."""
        clock = SimClock(1000)
        
        callback_func = Mock()
        clock.register_timer_callback("test_callback", callback_func)
        
        retrieved_callback = clock.get_timer_callback("test_callback")
        assert retrieved_callback == callback_func
        
        # Test non-existent callback
        assert clock.get_timer_callback("nonexistent") is None
        
    def test_start_stop(self):
        """Test start/stop functionality."""
        clock = SimClock(1000)
        
        assert not clock.is_running()
        
        clock.start()
        assert clock.is_running()
        
        clock.stop()
        assert not clock.is_running()
        
    def test_reset(self):
        """Test reset functionality."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        clock.start()
        
        # Reset with new start time
        clock.reset(2000)
        
        assert clock.now() == 2000
        assert clock.get_event_count() == 0
        assert not clock.is_running()
        
    def test_get_stats(self):
        """Test getting clock statistics."""
        clock = SimClock(1000)
        clock.schedule_timer("timer1", 500, "callback1")
        clock.register_timer_callback("callback1", Mock())
        
        stats = clock.get_stats()
        
        assert stats["current_time"] == 1000
        assert stats["event_count"] == 1
        assert stats["is_running"] is False
        assert stats["next_event_time"] == 1500
        assert "callback1" in stats["registered_callbacks"]


class TestEventTypes:
    """Test event type functionality."""
    
    def test_data_event(self):
        """Test DataEvent creation."""
        event = DataEvent(
            timestamp=1704067200000000000,
            event_type=EventType.DATA,
            data={"price": 150.0, "size": 100},
            symbol="AAPL"
        )
        
        assert event.timestamp == 1704067200000000000
        assert event.event_type == EventType.DATA
        assert event.symbol == "AAPL"
        assert event.data == {"price": 150.0, "size": 100}
        
    def test_timer_event(self):
        """Test TimerEvent creation."""
        event = TimerEvent(
            timestamp=1704067200000000000,
            event_type=EventType.TIMER,
            data={"param": "value"},
            timer_id="timer1",
            callback="callback1"
        )
        
        assert event.timestamp == 1704067200000000000
        assert event.event_type == EventType.TIMER
        assert event.timer_id == "timer1"
        assert event.callback == "callback1"
        assert event.data == {"param": "value"}
        
    def test_control_event(self):
        """Test ControlEvent creation."""
        event = ControlEvent(
            timestamp=1704067200000000000,
            event_type=EventType.CONTROL,
            data={"reason": "test"},
            command="STOP"
        )
        
        assert event.timestamp == 1704067200000000000
        assert event.event_type == EventType.CONTROL
        assert event.command == "STOP"
        assert event.data == {"reason": "test"}
        
    def test_event_ordering(self):
        """Test that events can be ordered by timestamp."""
        events = [
            Event(3000, EventType.DATA),
            Event(1000, EventType.TIMER),
            Event(2000, EventType.CONTROL),
        ]
        
        # Sort events
        events.sort()
        
        timestamps = [event.timestamp for event in events]
        assert timestamps == [1000, 2000, 3000]


class TestClockIntegration:
    """Test clock integration scenarios."""
    
    def test_deterministic_replay(self):
        """Test that clock provides deterministic replay."""
        clock1 = SimClock(1000)
        clock2 = SimClock(1000)
        
        # Add same events to both clocks
        events = [
            (1500, "timer", "timer1", "callback1"),
            (2000, "data", "AAPL", {"price": 150.0}),
            (2500, "timer", "timer2", "callback2"),
        ]
        
        for event_data in events:
            if event_data[1] == "timer":  # Timer event
                clock1.schedule_timer_at(event_data[2], event_data[0], event_data[3])
                clock2.schedule_timer_at(event_data[2], event_data[0], event_data[3])
            else:  # Data event
                clock1.add_data_event(event_data[0], event_data[2], event_data[3])
                clock2.add_data_event(event_data[0], event_data[2], event_data[3])
        
        # Process events from both clocks
        events1 = []
        events2 = []
        
        while clock1.has_events():
            events1.append(clock1.get_next_event())
            
        while clock2.has_events():
            events2.append(clock2.get_next_event())
        
        # Should be identical
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.timestamp == e2.timestamp
            assert e1.event_type == e2.event_type
            
    def test_timer_firing_accuracy(self):
        """Test that timers fire at requested nanoseconds."""
        clock = SimClock(1000)
        fired_times = []
        
        def record_time(data):
            fired_times.append(clock.now())
            
        clock.register_timer_callback("record", record_time)
        
        # Schedule timers at specific times
        clock.schedule_timer_at("timer1", 1500, "record")
        clock.schedule_timer_at("timer2", 2000, "record")
        clock.schedule_timer_at("timer3", 2500, "record")
        
        # Process events
        while clock.has_events():
            event = clock.get_next_event()
            if event.event_type == EventType.TIMER:
                callback = clock.get_timer_callback(event.callback)
                if callback:
                    callback(event.data)
        
        # Check that timers fired at exact times
        assert fired_times == [1500, 2000, 2500]
        
    def test_mixed_event_types(self):
        """Test processing mixed event types."""
        clock = SimClock(1000)
        processed_events = []
        
        # Add mixed events
        clock.add_data_event(1500, "AAPL", {"price": 150.0})
        clock.schedule_timer("timer1", 500, "callback1")  # 1500
        clock.add_control_event(2000, "STOP")
        clock.schedule_timer("timer2", 1000, "callback2")  # 2000
        
        # Process events
        while clock.has_events():
            event = clock.get_next_event()
            processed_events.append((event.timestamp, event.event_type))
        
        # Check ordering and types
        expected = [
            (1500, EventType.DATA),
            (1500, EventType.TIMER),
            (2000, EventType.TIMER),
            (2000, EventType.CONTROL),
        ]
        
        assert processed_events == expected


if __name__ == "__main__":
    pytest.main([__file__])
