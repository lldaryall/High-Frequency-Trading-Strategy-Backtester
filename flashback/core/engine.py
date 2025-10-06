"""Main backtesting engine with event loop."""

import heapq
from typing import Any, Callable, Dict, List, Optional, Type, Iterator
import pandas as pd
import numpy as np

from .clock import SimClock, Event, EventType, DataEvent, TimerEvent, ControlEvent
from .events import MarketDataEvent, OrderEvent, FillEvent, CancelEvent, RejectEvent
from ..strategy.base import Strategy
from ..market.book import MatchingEngine
from ..exec.router import OrderRouter
from ..risk.manager import RiskManager
from ..data.loader import DataLoader
from ..utils.logger import get_logger


class BacktestEngine:
    """Main backtesting engine with single-thread event loop."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize clock
        start_time = config.get('start_time')
        if start_time:
            start_time = pd.Timestamp(start_time).value
        self.clock = SimClock(start_time)
        
        # Initialize components
        self.data_loader = DataLoader(config.get('data', {}))
        self.order_books: Dict[str, MatchingEngine] = {}
        self.order_router = OrderRouter(config.get('execution', {}))
        self.risk_manager = RiskManager(config.get('risk', {}))
        
        # Event hooks
        self.on_data_hook: Optional[Callable[[DataEvent], None]] = None
        self.on_timer_hook: Optional[Callable[[int], None]] = None
        self.on_end_hook: Optional[Callable[[], None]] = None
        
        # Strategy and state
        self.strategies: List[Strategy] = []
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_log = []
        self.position_log = []
        self.event_log = []
        
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a strategy to the engine.
        
        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
        strategy.set_engine(self)
        
    def set_data_hook(self, hook: Callable[[DataEvent], None]) -> None:
        """
        Set the data event hook.
        
        Args:
            hook: Function to call on data events
        """
        self.on_data_hook = hook
        
    def set_timer_hook(self, hook: Callable[[int], None]) -> None:
        """
        Set the timer event hook.
        
        Args:
            hook: Function to call on timer events
        """
        self.on_timer_hook = hook
        
    def set_end_hook(self, hook: Callable[[], None]) -> None:
        """
        Set the end of simulation hook.
        
        Args:
            hook: Function to call at the end
        """
        self.on_end_hook = hook
        
    def load_data(self, data_path: str) -> None:
        """
        Load market data from file.
        
        Args:
            data_path: Path to data file
        """
        self.logger.info(f"Loading data from {data_path}")
        self.data = self.data_loader.load_data(data_path)
        
        # Validate data
        if not self.data_loader.validate_data(self.data):
            raise ValueError("Data validation failed")
            
        self.logger.info(f"Loaded {len(self.data)} records")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting backtest")
        self.is_running = True
        self.clock.start()
        
        try:
            # Initialize strategies
            for strategy in self.strategies:
                strategy.initialize()
                
            # Process events
            self._process_events()
            
            # Finalize strategies
            for strategy in self.strategies:
                strategy.finalize()
                
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Call end hook
            if self.on_end_hook:
                self.on_end_hook()
                
            self.logger.info("Backtest completed successfully")
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
            self.clock.stop()
            
        return self._get_results()
        
    def run_with_events(self, events: Iterator[Event]) -> Dict[str, Any]:
        """
        Run backtest with a custom event iterator.
        
        Args:
            events: Iterator of events to process
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting backtest with custom events")
        self.is_running = True
        self.clock.start()
        
        try:
            # Initialize strategies
            for strategy in self.strategies:
                strategy.initialize()
                
            # Process events from iterator
            self._process_event_iterator(events)
            
            # Finalize strategies
            for strategy in self.strategies:
                strategy.finalize()
                
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Call end hook
            if self.on_end_hook:
                self.on_end_hook()
                
            self.logger.info("Backtest completed successfully")
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
            self.clock.stop()
            
        return self._get_results()
        
    def _process_events(self) -> None:
        """Process all events in the clock's queue."""
        while self.clock.has_events() and self.is_running:
            event = self.clock.get_next_event()
            if event:
                self._handle_event(event)
                
    def _process_event_iterator(self, events: Iterator[Event]) -> None:
        """Process events from an iterator."""
        for event in events:
            if not self.is_running:
                break
                
            # Advance clock to event time
            if event.timestamp > self.clock.now():
                self.clock.advance_to(event.timestamp)
                
            self._handle_event(event)
            
    def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        # Log event
        self.event_log.append({
            'timestamp': event.timestamp,
            'type': event.event_type.value,
            'data': event.data
        })
        
        # Handle by event type
        if event.event_type == EventType.DATA:
            self._handle_data_event(event)
        elif event.event_type == EventType.TIMER:
            self._handle_timer_event(event)
        elif event.event_type == EventType.CONTROL:
            self._handle_control_event(event)
        else:
            self.logger.warning(f"Unknown event type: {event.event_type}")
            
    def _handle_data_event(self, event: DataEvent) -> None:
        """Handle data event."""
        # Call data hook
        if self.on_data_hook:
            self.on_data_hook(event)
            
        # Notify strategies
        for strategy in self.strategies:
            strategy.on_market_data(event)
            
    def _handle_timer_event(self, event: TimerEvent) -> None:
        """Handle timer event."""
        # Call timer hook
        if self.on_timer_hook:
            self.on_timer_hook(event.timestamp)
            
        # Execute timer callback
        callback = self.clock.get_timer_callback(event.callback)
        if callback:
            callback(event.data)
        else:
            self.logger.warning(f"Timer callback not found: {event.callback}")
            
    def _handle_control_event(self, event: ControlEvent) -> None:
        """Handle control event."""
        if event.command == "STOP":
            self.is_running = False
        elif event.command == "PAUSE":
            # Could implement pause functionality
            pass
        else:
            self.logger.warning(f"Unknown control command: {event.command}")
            
    def schedule_timer(self, timer_id: str, delay_ns: int, callback: str, data: Optional[dict] = None) -> None:
        """
        Schedule a timer event.
        
        Args:
            timer_id: Unique identifier for the timer
            delay_ns: Delay from current time in nanoseconds
            callback: Callback function name
            data: Optional data to pass to callback
        """
        self.clock.schedule_timer(timer_id, delay_ns, callback, data)
        
    def schedule_timer_at(self, timer_id: str, timestamp: int, callback: str, data: Optional[dict] = None) -> None:
        """
        Schedule a timer event at specific timestamp.
        
        Args:
            timer_id: Unique identifier for the timer
            timestamp: Absolute timestamp in nanoseconds
            callback: Callback function name
            data: Optional data to pass to callback
        """
        self.clock.schedule_timer_at(timer_id, timestamp, callback, data)
        
    def register_timer_callback(self, callback_name: str, callback_func: Callable) -> None:
        """
        Register a timer callback function.
        
        Args:
            callback_name: Name of the callback
            callback_func: Callback function
        """
        self.clock.register_timer_callback(callback_name, callback_func)
        
    def add_data_event(self, timestamp: int, symbol: str, data: dict) -> None:
        """
        Add a data event to the queue.
        
        Args:
            timestamp: Event timestamp in nanoseconds
            symbol: Trading symbol
            data: Event data
        """
        self.clock.add_data_event(timestamp, symbol, data)
        
    def add_control_event(self, timestamp: int, command: str, data: Optional[dict] = None) -> None:
        """
        Add a control event to the queue.
        
        Args:
            timestamp: Event timestamp in nanoseconds
            command: Control command
            data: Optional command data
        """
        self.clock.add_control_event(timestamp, command, data)
        
    def stop(self) -> None:
        """Stop the backtest."""
        self.is_running = False
        
    def get_current_time(self) -> int:
        """
        Get current simulation time.
        
        Returns:
            Current timestamp in nanoseconds
        """
        return self.clock.now()
        
    def get_current_time_pd(self) -> pd.Timestamp:
        """
        Get current simulation time as pandas Timestamp.
        
        Returns:
            Current timestamp as pandas Timestamp
        """
        return self.clock.now_pd()
        
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics."""
        if not self.trade_log:
            self.performance_metrics = {
                'total_trades': 0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
            }
            return
            
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(self.trade_log)
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        total_pnl = trades_df['quantity'].sum() * trades_df['price'].mean()
        
        # Calculate returns
        returns = trades_df['price'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }
        
    def _get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        return {
            'performance': self.performance_metrics,
            'trades': self.trade_log,
            'positions': self.position_log,
            'events': self.event_log,
            'config': self.config,
            'clock_stats': self.clock.get_stats()
        }