"""
Flashback: High-Frequency Trading Strategy Backtester

A high-performance backtesting engine designed for HFT strategies with
microsecond precision and realistic market simulation.
"""

__version__ = "0.1.0"
__author__ = "Flashback Team"

from .core.engine import BacktestEngine
from .core.clock import SimClock
from .strategy.base import Strategy
from .market.book import MatchingEngine
from .exec.router import OrderRouter

__all__ = [
    "BacktestEngine",
    "SimClock", 
    "Strategy",
    "MatchingEngine",
    "OrderRouter",
]
