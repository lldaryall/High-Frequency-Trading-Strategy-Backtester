"""
Analysis package for flashback HFT backtesting engine.

This package provides analysis tools including latency sensitivity analysis
and other performance analysis capabilities.
"""

from .latency_sensitivity import (
    LatencySensitivityAnalyzer,
    LatencySweepConfig,
    LatencySweepResult,
    LatencySensitivitySummary,
    create_latency_sensitivity_analyzer
)

__all__ = [
    "LatencySensitivityAnalyzer",
    "LatencySweepConfig", 
    "LatencySweepResult",
    "LatencySensitivitySummary",
    "create_latency_sensitivity_analyzer"
]
