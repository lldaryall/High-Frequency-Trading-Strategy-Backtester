"""Simulation clock for event-driven backtesting."""

from typing import Optional, Callable, List, Tuple
import heapq
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class EventType(Enum):
    """Event types in the simulation."""
    DATA = "data"
    TIMER = "timer"
    CONTROL = "control"


@dataclass
class Event:
    """Base event class for the simulation."""
    timestamp: int  # nanoseconds since epoch
    event_type: EventType
    data: Optional[dict] = None
    
    def __lt__(self, other: 'Event') -> bool:
        """Enable heap ordering by timestamp."""
        return self.timestamp < other.timestamp


@dataclass
class TimerEvent(Event):
    """Timer event for scheduled callbacks."""
    timer_id: str = ""
    callback: str = ""
    
    def __post_init__(self) -> None:
        self.event_type = EventType.TIMER


@dataclass
class DataEvent(Event):
    """Data event containing market data."""
    symbol: str = ""
    
    def __post_init__(self) -> None:
        self.event_type = EventType.DATA


@dataclass
class ControlEvent(Event):
    """Control event for simulation control."""
    command: str = ""
    
    def __post_init__(self) -> None:
        self.event_type = EventType.CONTROL


class SimClock:
    """Simulation clock that advances by next event timestamp."""
    
    def __init__(self, start_time: Optional[int] = None):
        """
        Initialize simulation clock.
        
        Args:
            start_time: Initial timestamp in nanoseconds (default: current time)
        """
        self._current_time = start_time or pd.Timestamp.now().value
        self._event_queue: List[Event] = []
        self._timer_callbacks: dict[str, Callable] = {}
        self._is_running = False
        
    def now(self) -> int:
        """
        Get current simulation time in nanoseconds.
        
        Returns:
            Current timestamp in nanoseconds
        """
        return self._current_time
    
    def now_pd(self) -> pd.Timestamp:
        """
        Get current simulation time as pandas Timestamp.
        
        Returns:
            Current timestamp as pandas Timestamp
        """
        return pd.Timestamp(self._current_time)
    
    def sleep_ns(self, duration_ns: int) -> None:
        """
        Advance clock by specified duration.
        
        Args:
            duration_ns: Duration to advance in nanoseconds
        """
        self._current_time += duration_ns
    
    def advance_to(self, timestamp: int) -> None:
        """
        Advance clock to specific timestamp.
        
        Args:
            timestamp: Target timestamp in nanoseconds
        """
        if timestamp < self._current_time:
            raise ValueError(f"Cannot advance to past timestamp: {timestamp} < {self._current_time}")
        self._current_time = timestamp
    
    def schedule_timer(self, timer_id: str, delay_ns: int, callback: str, data: Optional[dict] = None) -> None:
        """
        Schedule a timer event.
        
        Args:
            timer_id: Unique identifier for the timer
            delay_ns: Delay from current time in nanoseconds
            callback: Callback function name
            data: Optional data to pass to callback
        """
        fire_time = self._current_time + delay_ns
        timer_event = TimerEvent(
            timestamp=fire_time,
            event_type=EventType.TIMER,
            data=data,
            timer_id=timer_id,
            callback=callback
        )
        heapq.heappush(self._event_queue, timer_event)
    
    def schedule_timer_at(self, timer_id: str, timestamp: int, callback: str, data: Optional[dict] = None) -> None:
        """
        Schedule a timer event at specific timestamp.
        
        Args:
            timer_id: Unique identifier for the timer
            timestamp: Absolute timestamp in nanoseconds
            callback: Callback function name
            data: Optional data to pass to callback
        """
        if timestamp < self._current_time:
            raise ValueError(f"Cannot schedule timer in the past: {timestamp} < {self._current_time}")
        
        timer_event = TimerEvent(
            timestamp=timestamp,
            event_type=EventType.TIMER,
            data=data,
            timer_id=timer_id,
            callback=callback
        )
        heapq.heappush(self._event_queue, timer_event)
    
    def cancel_timer(self, timer_id: str) -> bool:
        """
        Cancel a scheduled timer.
        
        Args:
            timer_id: Timer identifier to cancel
            
        Returns:
            True if timer was found and cancelled, False otherwise
        """
        # Remove timer from queue
        original_queue = self._event_queue.copy()
        self._event_queue.clear()
        
        cancelled = False
        for event in original_queue:
            if isinstance(event, TimerEvent) and event.timer_id == timer_id:
                cancelled = True
            else:
                heapq.heappush(self._event_queue, event)
        
        return cancelled
    
    def add_data_event(self, timestamp: int, symbol: str, data: dict) -> None:
        """
        Add a data event to the queue.
        
        Args:
            timestamp: Event timestamp in nanoseconds
            symbol: Trading symbol
            data: Event data
        """
        data_event = DataEvent(
            timestamp=timestamp,
            event_type=EventType.DATA,
            data=data,
            symbol=symbol
        )
        heapq.heappush(self._event_queue, data_event)
    
    def add_control_event(self, timestamp: int, command: str, data: Optional[dict] = None) -> None:
        """
        Add a control event to the queue.
        
        Args:
            timestamp: Event timestamp in nanoseconds
            command: Control command
            data: Optional command data
        """
        control_event = ControlEvent(
            timestamp=timestamp,
            event_type=EventType.CONTROL,
            data=data,
            command=command
        )
        heapq.heappush(self._event_queue, control_event)
    
    def get_next_event(self) -> Optional[Event]:
        """
        Get the next event from the queue.
        
        Returns:
            Next event or None if queue is empty
        """
        if not self._event_queue:
            return None
        
        event = heapq.heappop(self._event_queue)
        self._current_time = event.timestamp
        return event
    
    def peek_next_event(self) -> Optional[Event]:
        """
        Peek at the next event without removing it.
        
        Returns:
            Next event or None if queue is empty
        """
        if not self._event_queue:
            return None
        
        return self._event_queue[0]
    
    def has_events(self) -> bool:
        """
        Check if there are any events in the queue.
        
        Returns:
            True if events are available, False otherwise
        """
        return len(self._event_queue) > 0
    
    def get_event_count(self) -> int:
        """
        Get the number of events in the queue.
        
        Returns:
            Number of events in queue
        """
        return len(self._event_queue)
    
    def clear_events(self) -> None:
        """Clear all events from the queue."""
        self._event_queue.clear()
    
    def get_events_in_range(self, start_time: int, end_time: int) -> List[Event]:
        """
        Get all events in a time range.
        
        Args:
            start_time: Start timestamp in nanoseconds
            end_time: End timestamp in nanoseconds
            
        Returns:
            List of events in the time range
        """
        events_in_range = []
        temp_queue = []
        
        # Extract events in range
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if start_time <= event.timestamp <= end_time:
                events_in_range.append(event)
            else:
                temp_queue.append(event)
        
        # Restore events not in range
        for event in temp_queue:
            heapq.heappush(self._event_queue, event)
        
        return events_in_range
    
    def register_timer_callback(self, callback_name: str, callback_func: Callable) -> None:
        """
        Register a timer callback function.
        
        Args:
            callback_name: Name of the callback
            callback_func: Callback function
        """
        self._timer_callbacks[callback_name] = callback_func
    
    def get_timer_callback(self, callback_name: str) -> Optional[Callable]:
        """
        Get a registered timer callback.
        
        Args:
            callback_name: Name of the callback
            
        Returns:
            Callback function or None if not found
        """
        return self._timer_callbacks.get(callback_name)
    
    def is_running(self) -> bool:
        """
        Check if the clock is running.
        
        Returns:
            True if clock is running, False otherwise
        """
        return self._is_running
    
    def start(self) -> None:
        """Start the clock."""
        self._is_running = True
    
    def stop(self) -> None:
        """Stop the clock."""
        self._is_running = False
    
    def reset(self, start_time: Optional[int] = None) -> None:
        """
        Reset the clock to initial state.
        
        Args:
            start_time: New start time in nanoseconds (default: current time)
        """
        self._current_time = start_time or pd.Timestamp.now().value
        self._event_queue.clear()
        self._is_running = False
    
    def get_stats(self) -> dict:
        """
        Get clock statistics.
        
        Returns:
            Dictionary with clock statistics
        """
        return {
            "current_time": self._current_time,
            "current_time_pd": self.now_pd(),
            "event_count": len(self._event_queue),
            "is_running": self._is_running,
            "next_event_time": self.peek_next_event().timestamp if self.peek_next_event() else None,
            "registered_callbacks": list(self._timer_callbacks.keys())
        }