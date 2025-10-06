"""
Latency sensitivity analysis for flashback HFT backtesting engine.

This module provides tools to analyze the impact of latency on strategy performance
by running backtests across a grid of latency values.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

from ..metrics.performance import PerformanceAnalyzer, PerformanceMetrics
from ..market.latency import LatencyModel, RandomLatencyModel, NetworkLatencyModel
from ..market.slippage import SlippageModel, ImbalanceSlippageModel, SlippageConfig
from ..market.transaction_costs import TransactionCostModel, SimpleTransactionCostModel, TransactionCostConfig


@dataclass
class LatencySweepConfig:
    """Configuration for latency sensitivity sweep."""
    latency_min_ns: float = 1000.0  # Minimum latency in nanoseconds
    latency_max_ns: float = 10000000.0  # Maximum latency in nanoseconds
    latency_steps: int = 10  # Number of latency steps
    latency_scale: str = "log"  # Scale type: "linear" or "log"
    latency_models: List[str] = field(default_factory=lambda: ["random", "network"])
    include_baseline: bool = True  # Include zero-latency baseline


@dataclass
class LatencySweepResult:
    """Result of latency sensitivity sweep."""
    latency_ns: float
    latency_model: str
    metrics: PerformanceMetrics
    config_hash: str  # Hash of configuration used


@dataclass
class LatencySensitivitySummary:
    """Summary of latency sensitivity analysis."""
    results: List[LatencySweepResult]
    latency_grid: List[float]
    performance_metrics: List[str]
    sensitivity_analysis: Dict[str, Any]
    summary_table: pd.DataFrame


class LatencySensitivityAnalyzer:
    """
    Analyzer for latency sensitivity of HFT strategies.
    
    This class runs backtests across a grid of latency values to analyze
    how strategy performance degrades with increased latency.
    """
    
    def __init__(
        self,
        config: LatencySweepConfig,
        slippage_model: Optional[SlippageModel] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None
    ):
        """
        Initialize latency sensitivity analyzer.
        
        Args:
            config: Latency sweep configuration
            slippage_model: Slippage model to use
            transaction_cost_model: Transaction cost model to use
        """
        self.config = config
        self.slippage_model = slippage_model or ImbalanceSlippageModel(SlippageConfig())
        self.transaction_cost_model = transaction_cost_model or SimpleTransactionCostModel(TransactionCostConfig())
        self.results: List[LatencySweepResult] = []
    
    def run_sweep(
        self,
        backtest_runner,
        strategy_config: Dict[str, Any],
        market_data: List[Any],
        output_dir: str = "latency_sweep_output"
    ) -> LatencySensitivitySummary:
        """
        Run latency sensitivity sweep.
        
        Args:
            backtest_runner: Function that runs a backtest and returns results
            strategy_config: Strategy configuration
            market_data: Market data for backtesting
            output_dir: Directory to save results
            
        Returns:
            Latency sensitivity summary
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate latency grid
        latency_grid = self._generate_latency_grid()
        
        # Run backtests for each latency value
        for latency_ns in latency_grid:
            for model_type in self.config.latency_models:
                result = self._run_single_backtest(
                    backtest_runner,
                    strategy_config,
                    market_data,
                    latency_ns,
                    model_type
                )
                self.results.append(result)
        
        # Include baseline if requested
        if self.config.include_baseline:
            baseline_result = self._run_single_backtest(
                backtest_runner,
                strategy_config,
                market_data,
                0.0,
                "baseline"
            )
            self.results.append(baseline_result)
        
        # Analyze results
        summary = self._analyze_results()
        
        # Save results
        self._save_results(summary, output_dir)
        
        return summary
    
    def _generate_latency_grid(self) -> List[float]:
        """Generate grid of latency values."""
        if self.config.latency_scale == "log":
            # Logarithmic scale
            min_log = np.log10(self.config.latency_min_ns)
            max_log = np.log10(self.config.latency_max_ns)
            log_values = np.linspace(min_log, max_log, self.config.latency_steps)
            return [10**x for x in log_values]
        else:
            # Linear scale
            return np.linspace(
                self.config.latency_min_ns,
                self.config.latency_max_ns,
                self.config.latency_steps
            ).tolist()
    
    def _run_single_backtest(
        self,
        backtest_runner,
        strategy_config: Dict[str, Any],
        market_data: List[Any],
        latency_ns: float,
        model_type: str
    ) -> LatencySweepResult:
        """Run a single backtest with specified latency."""
        # Create latency model
        if model_type == "baseline":
            latency_model = None  # No latency
        elif model_type == "random":
            latency_model = RandomLatencyModel(
                base_latency_ns=latency_ns,
                random_component_ns=latency_ns * 0.1
            )
        elif model_type == "network":
            latency_model = NetworkLatencyModel(
                base_latency_ns=latency_ns,
                network_latency_ns=latency_ns * 0.3,
                processing_latency_ns=latency_ns * 0.7
            )
        else:
            raise ValueError(f"Unknown latency model type: {model_type}")
        
        # Run backtest
        backtest_config = {
            **strategy_config,
            "latency_model": latency_model,
            "slippage_model": self.slippage_model,
            "transaction_cost_model": self.transaction_cost_model
        }
        
        # Execute backtest (this would be implemented by the backtest runner)
        backtest_results = backtest_runner(backtest_config, market_data)
        
        # Extract performance metrics
        performance_analyzer = PerformanceAnalyzer()
        
        # Add snapshots, trades, and latencies to analyzer
        for snapshot in backtest_results.get("snapshots", []):
            performance_analyzer.add_snapshot(snapshot)
        
        for trade in backtest_results.get("trades", []):
            performance_analyzer.add_trade(trade)
        
        for latency in backtest_results.get("latencies", []):
            performance_analyzer.add_latency(latency)
        
        # Calculate metrics
        metrics = performance_analyzer.calculate_metrics()
        
        # Create result
        config_hash = self._hash_config(backtest_config)
        
        return LatencySweepResult(
            latency_ns=latency_ns,
            latency_model=model_type,
            metrics=metrics,
            config_hash=config_hash
        )
    
    def _analyze_results(self) -> LatencySensitivitySummary:
        """Analyze latency sensitivity results."""
        # Create summary table
        summary_data = []
        for result in self.results:
            row = {
                "latency_ns": result.latency_ns,
                "latency_model": result.latency_model,
                "total_return": result.metrics.total_return,
                "annualized_return": result.metrics.annualized_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "volatility": result.metrics.volatility,
                "total_trades": result.metrics.total_trades,
                "hit_rate": result.metrics.hit_rate,
                "mean_latency_ns": result.metrics.mean_latency_ns,
                "p95_latency_ns": result.metrics.p95_latency_ns,
                "total_fees": result.metrics.total_fees,
                "final_capital": result.metrics.final_capital
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate sensitivity metrics
        sensitivity_analysis = self._calculate_sensitivity_metrics(summary_df)
        
        # Get performance metrics columns
        performance_metrics = [
            "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
            "volatility", "total_trades", "hit_rate", "mean_latency_ns",
            "p95_latency_ns", "total_fees", "final_capital"
        ]
        
        return LatencySensitivitySummary(
            results=self.results,
            latency_grid=self._generate_latency_grid(),
            performance_metrics=performance_metrics,
            sensitivity_analysis=sensitivity_analysis,
            summary_table=summary_df
        )
    
    def _calculate_sensitivity_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sensitivity metrics from results."""
        analysis = {}
        
        # Group by latency model
        for model in df["latency_model"].unique():
            model_df = df[df["latency_model"] == model].sort_values("latency_ns")
            
            if len(model_df) < 2:
                continue
            
            # Calculate sensitivity for key metrics
            key_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "hit_rate"]
            
            for metric in key_metrics:
                if metric in model_df.columns:
                    values = model_df[metric].values
                    latencies = model_df["latency_ns"].values
                    
                    # Calculate correlation with latency
                    correlation = np.corrcoef(latencies, values)[0, 1]
                    
                    # Calculate sensitivity (change per microsecond)
                    if len(values) > 1:
                        sensitivity = (values[-1] - values[0]) / (latencies[-1] - latencies[0]) * 1000000  # per microsecond
                    else:
                        sensitivity = 0.0
                    
                    analysis[f"{model}_{metric}_correlation"] = correlation
                    analysis[f"{model}_{metric}_sensitivity"] = sensitivity
        
        # Calculate overall sensitivity
        baseline_df = df[df["latency_model"] == "baseline"]
        if not baseline_df.empty:
            baseline_metrics = baseline_df.iloc[0]
            
            for model in df["latency_model"].unique():
                if model == "baseline":
                    continue
                
                model_df = df[df["latency_model"] == model]
                if model_df.empty:
                    continue
                
                # Calculate performance degradation
                for metric in ["total_return", "sharpe_ratio", "hit_rate"]:
                    if metric in model_df.columns and metric in baseline_metrics:
                        baseline_value = baseline_metrics[metric]
                        if baseline_value != 0:
                            degradation = (model_df[metric].mean() - baseline_value) / abs(baseline_value) * 100
                            analysis[f"{model}_{metric}_degradation_pct"] = degradation
        
        return analysis
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for tracking."""
        import hashlib
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _save_results(self, summary: LatencySensitivitySummary, output_dir: str) -> None:
        """Save results to files."""
        # Save summary table
        summary.summary_table.to_csv(f"{output_dir}/latency_sensitivity_summary.csv", index=False)
        
        # Save sensitivity analysis
        with open(f"{output_dir}/sensitivity_analysis.json", "w") as f:
            json.dump(summary.sensitivity_analysis, f, indent=2)
        
        # Save detailed results
        detailed_results = []
        for result in summary.results:
            detailed_results.append({
                "latency_ns": result.latency_ns,
                "latency_model": result.latency_model,
                "config_hash": result.config_hash,
                "metrics": {
                    "total_return": result.metrics.total_return,
                    "annualized_return": result.metrics.annualized_return,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "max_drawdown": result.metrics.max_drawdown,
                    "volatility": result.metrics.volatility,
                    "total_trades": result.metrics.total_trades,
                    "winning_trades": result.metrics.winning_trades,
                    "losing_trades": result.metrics.losing_trades,
                    "hit_rate": result.metrics.hit_rate,
                    "avg_win": result.metrics.avg_win,
                    "avg_loss": result.metrics.avg_loss,
                    "win_loss_ratio": result.metrics.win_loss_ratio,
                    "profit_factor": result.metrics.profit_factor,
                    "turnover": result.metrics.turnover,
                    "mean_latency_ns": result.metrics.mean_latency_ns,
                    "p95_latency_ns": result.metrics.p95_latency_ns,
                    "p99_latency_ns": result.metrics.p99_latency_ns,
                    "var_95": result.metrics.var_95,
                    "cvar_95": result.metrics.cvar_95,
                    "calmar_ratio": result.metrics.calmar_ratio,
                    "start_time": result.metrics.start_time,
                    "end_time": result.metrics.end_time,
                    "duration_days": result.metrics.duration_days,
                    "initial_capital": result.metrics.initial_capital,
                    "final_capital": result.metrics.final_capital,
                    "total_fees": result.metrics.total_fees,
                    "total_slippage": result.metrics.total_slippage
                }
            })
        
        with open(f"{output_dir}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Latency sensitivity analysis saved to {output_dir}/")
        print(f"- latency_sensitivity_summary.csv: Summary table")
        print(f"- sensitivity_analysis.json: Sensitivity metrics")
        print(f"- detailed_results.json: Complete results")


def create_latency_sensitivity_analyzer(
    latency_min_ns: float = 1000.0,
    latency_max_ns: float = 10000000.0,
    latency_steps: int = 10,
    latency_scale: str = "log"
) -> LatencySensitivityAnalyzer:
    """
    Factory function to create latency sensitivity analyzer.
    
    Args:
        latency_min_ns: Minimum latency in nanoseconds
        latency_max_ns: Maximum latency in nanoseconds
        latency_steps: Number of latency steps
        latency_scale: Scale type ("linear" or "log")
        
    Returns:
        Latency sensitivity analyzer
    """
    config = LatencySweepConfig(
        latency_min_ns=latency_min_ns,
        latency_max_ns=latency_max_ns,
        latency_steps=latency_steps,
        latency_scale=latency_scale
    )
    
    return LatencySensitivityAnalyzer(config)
