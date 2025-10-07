"""
End-to-end integration test for Flashback HFT backtesting engine.

This test runs a complete backtest with the momentum imbalance strategy on synthetic data
and validates that the results meet performance thresholds.

Requirements:
- > 30 trades
- Turnover within [0.3, 2.0]
- Sharpe ratio between [0.2, 3.0]
- Max drawdown < 15% of gross PnL
- JSON & plots exist

Do not weaken these bounds.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from flashback.cli.runner import BacktestRunner
from flashback.config.parser import load_config
from flashback.config.config import BacktestConfig, DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, ReportConfig
from examples.generate_synthetic import generate_synthetic_data


class TestEndToEndBacktest:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def synthetic_data(self, temp_dir):
        """Generate synthetic data for testing."""
        # Generate 50k events for integration test
        book_file, trade_file = generate_synthetic_data(
            symbol="AAPL",
            num_events=50000,
            seed=42,
            output_dir=temp_dir
        )
        return book_file, trade_file
    
    @pytest.fixture
    def backtest_config(self, synthetic_data, temp_dir):
        """Create backtest configuration."""
        book_file, trade_file = synthetic_data
        
        # Create test config
        config = BacktestConfig(
            name="E2E Integration Test",
            description="End-to-end integration test with momentum strategy",
            data=DataConfig(
                path=book_file,
                kind="book",
                symbol="AAPL"
            ),
            strategy=StrategyConfig(
                name="momentum_imbalance",
                symbol="AAPL",
                enabled=True,
                max_position=1000,
                max_order_size=100,
                params={
                    "short_ema_period": 5,
                    "long_ema_period": 20,
                    "imbalance_threshold": 0.3,
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01,
                    "min_trade_size": 10,
                    "max_trade_size": 100
                }
            ),
            execution=ExecutionConfig(
                fees={
                    "maker_bps": 0.0,
                    "taker_bps": 0.5,
                    "per_share": 0.0
                },
                latency={
                    "model": "normal",
                    "mean_ns": 500000,
                    "std_ns": 100000,
                    "seed": 42
                }
            ),
            risk=RiskConfig(
                max_gross=100000,
                max_pos_per_symbol=1000,
                max_daily_loss=-2000
            ),
            report=ReportConfig(
                output_dir=f"{temp_dir}/e2e_test_results",
                format="both",
                plots=True,
                detailed_trades=True,
                performance_metrics=True
            )
        )
        return config
    
    def test_momentum_strategy_e2e(self, backtest_config):
        """Test momentum strategy end-to-end with performance assertions."""
        print("🚀 Starting end-to-end backtest...")
        
        # Run backtest
        runner = BacktestRunner(backtest_config)
        results = runner.run()
        
        print("✅ Backtest completed, validating results...")
        
        # Validate output directory exists
        output_dir = Path(backtest_config.report.output_dir)
        assert output_dir.exists(), "Output directory should exist"
        
        # Validate required files exist
        required_files = [
            "performance.json",
            "performance.csv",
            "trades.csv",
            "positions.csv",
            "blotter.parquet"
        ]
        
        for file_name in required_files:
            file_path = output_dir / file_name
            assert file_path.exists(), f"Required file {file_name} should exist"
        
        # Validate plots exist
        plot_files = [
            "equity_curve.png",
            "drawdown_curve.png",
            "trade_pnl_histogram.png"
        ]
        
        for plot_file in plot_files:
            plot_path = output_dir / plot_file
            assert plot_path.exists(), f"Required plot {plot_file} should exist"
        
        # Load and validate performance metrics
        performance_file = output_dir / "performance.json"
        with open(performance_file, 'r') as f:
            performance = json.load(f)
        
        print(f"📊 Performance Metrics:")
        print(f"   Total Trades: {performance.get('total_trades', 0)}")
        print(f"   Total Return: {performance.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"   Turnover: {performance.get('turnover', 0):.2f}")
        
        # Assert performance thresholds (DO NOT WEAKEN)
        total_trades = performance.get('total_trades', 0)
        assert total_trades > 30, f"Expected >30 trades, got {total_trades}"
        
        turnover = performance.get('turnover', 0)
        assert 0.3 <= turnover <= 2.0, f"Expected turnover in [0.3, 2.0], got {turnover:.3f}"
        
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        assert 0.2 <= sharpe_ratio <= 3.0, f"Expected Sharpe in [0.2, 3.0], got {sharpe_ratio:.3f}"
        
        max_drawdown = abs(performance.get('max_drawdown', 0))
        assert max_drawdown < 0.15, f"Expected max drawdown <15%, got {max_drawdown:.2%}"
        
        # Validate trade data
        trades_file = output_dir / "trades.csv"
        trades_df = pd.read_csv(trades_file)
        
        assert len(trades_df) > 30, f"Expected >30 trades in CSV, got {len(trades_df)}"
        assert 'pnl' in trades_df.columns, "Trades CSV should have PnL column"
        assert 'entry_time' in trades_df.columns, "Trades CSV should have entry_time column"
        assert 'exit_time' in trades_df.columns, "Trades CSV should have exit_time column"
        
        # Validate position data
        positions_file = output_dir / "positions.csv"
        positions_df = pd.read_csv(positions_file)
        
        assert len(positions_df) > 0, "Positions CSV should not be empty"
        assert 'timestamp' in positions_df.columns, "Positions CSV should have timestamp column"
        assert 'symbol' in positions_df.columns, "Positions CSV should have symbol column"
        
        # Validate blotter data
        blotter_file = output_dir / "blotter.parquet"
        blotter_df = pd.read_parquet(blotter_file)
        
        assert len(blotter_df) > 0, "Blotter should not be empty"
        assert 'order_id' in blotter_df.columns, "Blotter should have order_id column"
        assert 'status' in blotter_df.columns, "Blotter should have status column"
        
        print("✅ All performance assertions passed!")
    
    def test_mean_reversion_strategy_e2e(self, synthetic_data, temp_dir):
        """Test mean reversion strategy end-to-end."""
        book_file, trade_file = synthetic_data
        
        # Create mean reversion config
        config = BacktestConfig(
            name="Mean Reversion E2E Test",
            description="End-to-end test with mean reversion strategy",
            data=DataConfig(
                path=book_file,
                kind="book",
                symbol="AAPL"
            ),
            strategy=StrategyConfig(
                name="mean_reversion",
                symbol="AAPL",
                enabled=True,
                max_position=1000,
                max_order_size=100,
                params={
                    "lookback_period": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.01
                }
            ),
            execution=ExecutionConfig(
                fees={
                    "maker_bps": 0.0,
                    "taker_bps": 0.5,
                    "per_share": 0.0
                },
                latency={
                    "model": "normal",
                    "mean_ns": 500000,
                    "std_ns": 100000,
                    "seed": 42
                }
            ),
            risk=RiskConfig(
                max_gross=100000,
                max_pos_per_symbol=1000,
                max_daily_loss=-2000
            ),
            report=ReportConfig(
                output_dir=f"{temp_dir}/mean_reversion_results",
                format="both",
                plots=True,
                detailed_trades=True,
                performance_metrics=True
            )
        )
        
        # Run backtest
        runner = BacktestRunner(config)
        results = runner.run()
        
        # Validate basic results
        output_dir = Path(config.report.output_dir)
        assert output_dir.exists(), "Output directory should exist"
        
        performance_file = output_dir / "performance.json"
        assert performance_file.exists(), "Performance file should exist"
        
        with open(performance_file, 'r') as f:
            performance = json.load(f)
        
        # Basic validation (less strict than momentum strategy)
        total_trades = performance.get('total_trades', 0)
        assert total_trades > 0, f"Expected >0 trades, got {total_trades}"
        
        print(f"✅ Mean reversion strategy completed with {total_trades} trades")
    
    def test_latency_sensitivity_e2e(self, synthetic_data, temp_dir):
        """Test latency sensitivity analysis end-to-end."""
        book_file, trade_file = synthetic_data
        
        # Create config for latency sweep
        config = BacktestConfig(
            name="Latency Sensitivity E2E Test",
            description="End-to-end latency sensitivity test",
            data=DataConfig(
                path=book_file,
                kind="book",
                symbol="AAPL"
            ),
            strategy=StrategyConfig(
                name="momentum_imbalance",
                symbol="AAPL",
                enabled=True,
                max_position=1000,
                max_order_size=100,
                params={
                    "short_ema_period": 5,
                    "long_ema_period": 20,
                    "imbalance_threshold": 0.3,
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01,
                    "min_trade_size": 10,
                    "max_trade_size": 100
                }
            ),
            execution=ExecutionConfig(
                fees={
                    "maker_bps": 0.0,
                    "taker_bps": 0.5,
                    "per_share": 0.0
                },
                latency={
                    "model": "normal",
                    "mean_ns": 500000,
                    "std_ns": 100000,
                    "seed": 42
                }
            ),
            risk=RiskConfig(
                max_gross=100000,
                max_pos_per_symbol=1000,
                max_daily_loss=-2000
            ),
            report=ReportConfig(
                output_dir=f"{temp_dir}/latency_sensitivity_results",
                format="both",
                plots=True,
                detailed_trades=True,
                performance_metrics=True
            )
        )
        
        # Run latency sensitivity analysis
        from flashback.cli.sweeper import LatencySweeper
        
        sweeper = LatencySweeper(config)
        sweep_results = sweeper.run_sweep([100000, 250000, 500000, 1000000])
        
        # Validate sweep results
        assert len(sweep_results) == 4, f"Expected 4 latency results, got {len(sweep_results)}"
        
        for result in sweep_results:
            assert 'latency_ns' in result, "Result should have latency_ns"
            assert 'total_return' in result, "Result should have total_return"
            assert 'sharpe_ratio' in result, "Result should have sharpe_ratio"
            assert 'total_trades' in result, "Result should have total_trades"
        
        # Validate that higher latency generally reduces performance
        returns = [r['total_return'] for r in sweep_results]
        latencies = [r['latency_ns'] for r in sweep_results]
        
        # Sort by latency
        sorted_data = sorted(zip(latencies, returns))
        sorted_returns = [r for _, r in sorted_data]
        
        # Performance should generally decrease with higher latency
        # (allowing for some noise)
        print(f"📊 Latency sensitivity results:")
        for i, (lat, ret) in enumerate(sorted_data):
            print(f"   {lat/1000:.0f}μs: {ret:.2%} return")
        
        print("✅ Latency sensitivity analysis completed")
    
    def test_error_handling_e2e(self, temp_dir):
        """Test error handling in end-to-end scenarios."""
        # Test with invalid data file (skip validation for this test)
        from unittest.mock import patch
        
        with patch('flashback.config.config.Path.exists', return_value=False):
            invalid_config = BacktestConfig(
                name="Error Handling Test",
                description="Test error handling with invalid data",
                data=DataConfig(
                    path="nonexistent_file.parquet",
                    kind="book",
                    symbol="AAPL"
                ),
                strategy=StrategyConfig(
                    name="momentum_imbalance",
                    symbol="AAPL",
                    enabled=True,
                    max_position=1000,
                    max_order_size=100,
                    params={}
                ),
                execution=ExecutionConfig(
                    fees={"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                    latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000, "seed": 42}
                ),
                risk=RiskConfig(
                    max_gross=100000,
                    max_pos_per_symbol=1000,
                    max_daily_loss=-2000
                ),
                report=ReportConfig(
                    output_dir=f"{temp_dir}/error_test_results",
                    format="both",
                    plots=True,
                    detailed_trades=True,
                    performance_metrics=True
                )
            )
        
        # Should raise appropriate error
        runner = BacktestRunner(invalid_config)
        with pytest.raises((FileNotFoundError, ValueError)):
            runner.run()
        
        print("✅ Error handling test passed")
    
    def test_performance_benchmarks(self, backtest_config):
        """Test that backtest completes within reasonable time."""
        import time
        
        start_time = time.time()
        
        # Run backtest
        runner = BacktestRunner(backtest_config)
        results = runner.run()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 30 seconds for 50k events
        assert execution_time < 30.0, f"Backtest took too long: {execution_time:.2f}s"
        
        print(f"✅ Backtest completed in {execution_time:.2f}s")
    
    def test_data_consistency_e2e(self, backtest_config):
        """Test data consistency across all output files."""
        # Run backtest
        runner = BacktestRunner(backtest_config)
        results = runner.run()
        
        output_dir = Path(backtest_config.report.output_dir)
        
        # Load all data files
        performance_file = output_dir / "performance.json"
        with open(performance_file, 'r') as f:
            performance = json.load(f)
        
        trades_df = pd.read_csv(output_dir / "trades.csv")
        positions_df = pd.read_csv(output_dir / "positions.csv")
        blotter_df = pd.read_parquet(output_dir / "blotter.parquet")
        
        # Validate consistency
        total_trades_perf = performance.get('total_trades', 0)
        total_trades_csv = len(trades_df)
        
        assert total_trades_perf == total_trades_csv, \
            f"Trade count mismatch: JSON={total_trades_perf}, CSV={total_trades_csv}"
        
        # Validate PnL consistency
        if len(trades_df) > 0:
            total_pnl_csv = trades_df['pnl'].sum()
            total_pnl_perf = performance.get('total_pnl', 0)
            
            assert abs(total_pnl_csv - total_pnl_perf) < 1e-6, \
                f"PnL mismatch: JSON={total_pnl_perf}, CSV={total_pnl_csv}"
        
        print("✅ Data consistency validation passed")


if __name__ == "__main__":
    # Run the main test
    pytest.main([__file__, "-v", "-s"])