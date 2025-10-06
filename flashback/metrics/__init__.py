"""
Metrics package for flashback HFT backtesting engine.

This package provides performance analysis, metrics calculation,
and visualization capabilities for backtesting results.
"""

from .performance import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    TradeRecord
)

__all__ = [
    "PerformanceAnalyzer",
    "PerformanceMetrics", 
    "TradeRecord"
]