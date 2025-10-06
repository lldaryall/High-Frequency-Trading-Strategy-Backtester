"""Timer utilities for performance measurement."""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional
import pandas as pd


class Timer:
    """High-precision timer for performance measurement."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize timer.
        
        Args:
            name: Optional timer name
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        
    def stop(self) -> float:
        """
        Stop the timer and return duration.
        
        Returns:
            Duration in seconds
        """
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
        
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
    def __str__(self) -> str:
        """String representation."""
        if self.duration is not None:
            return f"Timer({self.name}): {self.duration:.6f}s"
        else:
            return f"Timer({self.name}): Not started"


@contextmanager
def time_function(name: str):
    """
    Context manager for timing function execution.
    
    Args:
        name: Timer name
        
    Yields:
        Timer instance
    """
    timer = Timer(name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def time_function_decorator(func: Callable) -> Callable:
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with time_function(func.__name__) as timer:
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {timer.duration:.6f}s")
            return result
    return wrapper


class PerformanceTimer:
    """Timer for measuring multiple operations."""
    
    def __init__(self):
        """Initialize performance timer."""
        self.timings = {}
        
    def start(self, name: str) -> None:
        """
        Start timing an operation.
        
        Args:
            name: Operation name
        """
        self.timings[name] = {
            'start': time.perf_counter(),
            'duration': None
        }
        
    def stop(self, name: str) -> float:
        """
        Stop timing an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Duration in seconds
        """
        if name not in self.timings:
            raise ValueError(f"Operation '{name}' not started")
            
        end_time = time.perf_counter()
        start_time = self.timings[name]['start']
        duration = end_time - start_time
        
        self.timings[name]['duration'] = duration
        return duration
        
    def get_duration(self, name: str) -> Optional[float]:
        """
        Get duration of an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Duration in seconds or None if not completed
        """
        if name in self.timings:
            return self.timings[name]['duration']
        return None
        
    def get_summary(self) -> dict:
        """
        Get timing summary.
        
        Returns:
            Dictionary with timing summary
        """
        summary = {}
        for name, timing in self.timings.items():
            if timing['duration'] is not None:
                summary[name] = timing['duration']
        return summary
        
    def get_total_time(self) -> float:
        """
        Get total time of all completed operations.
        
        Returns:
            Total time in seconds
        """
        return sum(timing['duration'] for timing in self.timings.values() 
                  if timing['duration'] is not None)
        
    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()


def measure_latency(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure latency of a function call.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, latency_in_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return result, latency


def benchmark_function(func: Callable, iterations: int = 1000, *args, **kwargs) -> dict:
    """
    Benchmark a function with multiple iterations.
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    return {
        'iterations': iterations,
        'total_time': sum(times),
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }
