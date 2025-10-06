"""Latency sensitivity sweeper for running multiple backtests."""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json

from ..config import BacktestConfig
from ..analysis.latency_sensitivity import LatencySensitivityAnalyzer, LatencySweepConfig
from .runner import BacktestRunner

logger = logging.getLogger(__name__)


class LatencySweeper:
    """Runs latency sensitivity analysis across multiple latency values."""
    
    def __init__(self, base_config: BacktestConfig):
        """Initialize the latency sweeper."""
        self.base_config = base_config
        self.logger = logging.getLogger(__name__)
        
        # Create latency sensitivity analyzer
        sweep_config = LatencySweepConfig(
            latency_min_ns=min([1000, 10000, 100000]),  # Will be overridden
            latency_max_ns=max([1000, 10000, 100000]),  # Will be overridden
            latency_steps=10,
            latency_scale="linear",
            latency_models=["random"],
            include_baseline=True
        )
        self.analyzer = LatencySensitivityAnalyzer(sweep_config)
    
    def run_sweep(self, latency_values: List[int]) -> Dict[str, Any]:
        """Run backtest across multiple latency values."""
        self.logger.info(f"Starting latency sweep with values: {latency_values}")
        
        results = []
        
        for latency_ns in latency_values:
            self.logger.info(f"Running backtest with latency: {latency_ns}ns")
            
            # Create modified config with new latency
            config = self._create_config_with_latency(latency_ns)
            
            # Run backtest
            runner = BacktestRunner(config)
            backtest_result = runner.run()
            
            # Extract performance metrics
            performance = backtest_result["performance"]
            
            # Create sweep result
            sweep_result = {
                "latency_ns": latency_ns,
                "latency_model": "random",
                "config_hash": str(hash(str(config.to_dict()))),
                "performance": performance,
                "summary": backtest_result["summary"]
            }
            
            results.append(sweep_result)
        
        # Analyze sensitivity
        sensitivity_analysis = self._analyze_sensitivity(results)
        
        # Compile final results
        final_results = {
            "latency_values": latency_values,
            "results": results,
            "sensitivity_analysis": sensitivity_analysis,
            "summary": {
                "total_runs": len(results),
                "latency_range": f"{min(latency_values)}ns - {max(latency_values)}ns",
                "base_strategy": self.base_config.strategy.name
            }
        }
        
        self.logger.info("Latency sweep completed successfully")
        return final_results
    
    def _create_config_with_latency(self, latency_ns: int) -> BacktestConfig:
        """Create a new config with modified latency."""
        # Deep copy the base config
        config_dict = self.base_config.to_dict()
        
        # Update latency parameters
        config_dict["execution"]["latency"]["mean_ns"] = latency_ns
        config_dict["execution"]["latency"]["std_ns"] = latency_ns * 0.2  # 20% std
        
        # Update output directory to include latency
        base_output_dir = config_dict["report"]["output_dir"]
        config_dict["report"]["output_dir"] = f"{base_output_dir}/latency_{latency_ns}ns"
        
        # Create new config
        from ..config import BacktestConfig
        return BacktestConfig.from_dict(config_dict)
    
    def _analyze_sensitivity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sensitivity of performance to latency."""
        if len(results) < 2:
            return {"error": "Need at least 2 results for sensitivity analysis"}
        
        # Extract data for analysis
        latencies = [r["latency_ns"] for r in results]
        returns = [r["performance"]["total_return"] for r in results]
        sharpe_ratios = [r["performance"]["sharpe_ratio"] for r in results]
        max_drawdowns = [r["performance"]["max_drawdown"] for r in results]
        
        # Calculate correlations
        import numpy as np
        
        latency_array = np.array(latencies)
        return_array = np.array(returns)
        sharpe_array = np.array(sharpe_ratios)
        drawdown_array = np.array(max_drawdowns)
        
        # Calculate correlations
        return_correlation = np.corrcoef(latency_array, return_array)[0, 1]
        sharpe_correlation = np.corrcoef(latency_array, sharpe_array)[0, 1]
        drawdown_correlation = np.corrcoef(latency_array, drawdown_array)[0, 1]
        
        # Calculate sensitivity (change per microsecond)
        latency_range = max(latencies) - min(latencies)
        return_sensitivity = (max(returns) - min(returns)) / (latency_range / 1000) if latency_range > 0 else 0
        sharpe_sensitivity = (max(sharpe_ratios) - min(sharpe_ratios)) / (latency_range / 1000) if latency_range > 0 else 0
        
        # Calculate degradation from best to worst
        best_return_idx = np.argmax(returns)
        worst_return_idx = np.argmin(returns)
        return_degradation = (returns[worst_return_idx] - returns[best_return_idx]) / abs(returns[best_return_idx]) * 100 if returns[best_return_idx] != 0 else 0
        
        best_sharpe_idx = np.argmax(sharpe_ratios)
        worst_sharpe_idx = np.argmin(sharpe_ratios)
        sharpe_degradation = (sharpe_ratios[worst_sharpe_idx] - sharpe_ratios[best_sharpe_idx]) / abs(sharpe_ratios[best_sharpe_idx]) * 100 if sharpe_ratios[best_sharpe_idx] != 0 else 0
        
        return {
            "correlations": {
                "total_return": float(return_correlation),
                "sharpe_ratio": float(sharpe_correlation),
                "max_drawdown": float(drawdown_correlation)
            },
            "sensitivity_per_us": {
                "total_return": float(return_sensitivity),
                "sharpe_ratio": float(sharpe_sensitivity)
            },
            "degradation_pct": {
                "total_return": float(return_degradation),
                "sharpe_ratio": float(sharpe_degradation)
            },
            "latency_range_ns": {
                "min": int(min(latencies)),
                "max": int(max(latencies)),
                "range": int(latency_range)
            },
            "performance_range": {
                "total_return": {"min": float(min(returns)), "max": float(max(returns))},
                "sharpe_ratio": {"min": float(min(sharpe_ratios)), "max": float(max(sharpe_ratios))},
                "max_drawdown": {"min": float(min(max_drawdowns)), "max": float(max(max_drawdowns))}
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save latency sweep results to files."""
        # Create runs directory structure
        run_dir = self._create_run_directory()
        
        # Save summary table to both locations
        summary_data = []
        for result in results["results"]:
            row = {
                "latency_ns": result["latency_ns"],
                "latency_model": result["latency_model"],
                "total_return": result["performance"]["total_return"],
                "annualized_return": result["performance"]["annualized_return"],
                "sharpe_ratio": result["performance"]["sharpe_ratio"],
                "max_drawdown": result["performance"]["max_drawdown"],
                "hit_rate": result["performance"]["hit_rate"],
                "total_trades": result["performance"]["total_trades"],
                "mean_latency_ns": result["performance"]["mean_latency_ns"],
                "p95_latency_ns": result["performance"]["p95_latency_ns"]
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to original output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "latency_sensitivity_summary.csv", index=False)
        
        # Save to runs directory
        summary_df.to_csv(run_dir / "latency_sweep.csv", index=False)
        
        # Save sensitivity analysis
        with open(output_dir / "sensitivity_analysis.json", "w") as f:
            json.dump(results["sensitivity_analysis"], f, indent=2)
        
        # Save detailed results
        with open(output_dir / "detailed_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save config to runs directory
        self._save_config_yaml(run_dir)
        
        # Create summary performance files for the sweep
        self._create_sweep_summary_files(run_dir, results)
        
        self.logger.info(f"Latency sweep results saved to {output_dir} and {run_dir}")
    
    def _create_run_directory(self) -> Path:
        """Create timestamped run directory."""
        from datetime import datetime
        
        # Create runs directory if it doesn't exist
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = runs_dir / f"{timestamp}_sweep"
        run_dir.mkdir(exist_ok=True)
        
        return run_dir
    
    def _save_config_yaml(self, run_dir: Path):
        """Save resolved configuration as YAML."""
        import yaml
        
        config_file = run_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.base_config.to_dict(), f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved configuration to {config_file}")
    
    def _create_sweep_summary_files(self, run_dir: Path, results: Dict[str, Any]):
        """Create summary performance files for the sweep."""
        import json
        import pandas as pd
        
        # Create summary performance metrics (average across all runs)
        if results["results"]:
            # Calculate average performance metrics
            avg_metrics = {}
            for key in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", 
                       "hit_rate", "total_trades", "mean_latency_ns", "p95_latency_ns"]:
                values = [r["performance"][key] for r in results["results"] if key in r["performance"]]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    avg_metrics[key] = 0.0
            
            # Add sweep-specific metrics
            avg_metrics.update({
                "sweep_type": "latency_sensitivity",
                "latency_range_ns": f"{min(r['latency_ns'] for r in results['results'])}-{max(r['latency_ns'] for r in results['results'])}",
                "total_runs": len(results["results"]),
                "sensitivity_analysis": results["sensitivity_analysis"]
            })
            
            # Save performance.json
            with open(run_dir / "performance.json", 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            # Save performance.csv
            perf_df = pd.DataFrame([avg_metrics])
            perf_df.to_csv(run_dir / "performance.csv", index=False)
            
            # Create empty blotter and positions files (sweep doesn't have individual trades)
            empty_blotter = pd.DataFrame(columns=[
                'trade_id', 'symbol', 'side', 'quantity', 'price', 'timestamp', 
                'pnl', 'commission', 'latency_us'
            ])
            empty_blotter.to_parquet(run_dir / "blotter.parquet", index=False)
            
            empty_positions = pd.DataFrame(columns=[
                'timestamp', 'portfolio_value', 'cash', 'positions', 
                'unrealized_pnl', 'realized_pnl'
            ])
            empty_positions.to_parquet(run_dir / "positions.parquet", index=False)
            
            self.logger.info(f"Created sweep summary files in {run_dir}")
    
    def _generate_sweep_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate plots for latency sensitivity analysis."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            latencies = [r["latency_ns"] for r in results["results"]]
            returns = [r["performance"]["total_return"] for r in results["results"]]
            sharpe_ratios = [r["performance"]["sharpe_ratio"] for r in results["results"]]
            max_drawdowns = [r["performance"]["max_drawdown"] for r in results["results"]]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Latency Sensitivity Analysis", fontsize=16)
            
            # Total return vs latency
            axes[0, 0].plot(latencies, returns, 'bo-', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel("Latency (ns)")
            axes[0, 0].set_ylabel("Total Return")
            axes[0, 0].set_title("Total Return vs Latency")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe ratio vs latency
            axes[0, 1].plot(latencies, sharpe_ratios, 'ro-', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel("Latency (ns)")
            axes[0, 1].set_ylabel("Sharpe Ratio")
            axes[0, 1].set_title("Sharpe Ratio vs Latency")
            axes[0, 1].grid(True, alpha=0.3)
            
            # Max drawdown vs latency
            axes[1, 0].plot(latencies, max_drawdowns, 'go-', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel("Latency (ns)")
            axes[1, 0].set_ylabel("Max Drawdown")
            axes[1, 0].set_title("Max Drawdown vs Latency")
            axes[1, 0].grid(True, alpha=0.3)
            
            # Performance degradation
            best_return = max(returns)
            worst_return = min(returns)
            degradation = [(r - best_return) / abs(best_return) * 100 for r in returns]
            
            axes[1, 1].plot(latencies, degradation, 'mo-', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel("Latency (ns)")
            axes[1, 1].set_ylabel("Performance Degradation (%)")
            axes[1, 1].set_title("Performance Degradation vs Latency")
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "latency_sensitivity_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Latency sensitivity plots generated successfully")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
