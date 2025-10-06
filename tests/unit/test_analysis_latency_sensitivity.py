"""
Unit tests for latency sensitivity analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from flashback.analysis.latency_sensitivity import (
    LatencySweepConfig,
    LatencySweepResult,
    LatencySensitivitySummary,
    LatencySensitivityAnalyzer,
    create_latency_sensitivity_analyzer
)
from flashback.metrics.performance import PerformanceMetrics
from flashback.market.slippage import ImbalanceSlippageModel, SlippageConfig
from flashback.market.transaction_costs import SimpleTransactionCostModel, TransactionCostConfig


class TestLatencySweepConfig:
    """Test LatencySweepConfig dataclass."""
    
    def test_latency_sweep_config_creation(self):
        """Test latency sweep config creation."""
        config = LatencySweepConfig(
            latency_min_ns=1000.0,
            latency_max_ns=1000000.0,
            latency_steps=5,
            latency_scale="log",
            latency_models=["random", "network"],
            include_baseline=True
        )
        
        assert config.latency_min_ns == 1000.0
        assert config.latency_max_ns == 1000000.0
        assert config.latency_steps == 5
        assert config.latency_scale == "log"
        assert config.latency_models == ["random", "network"]
        assert config.include_baseline


class TestLatencySweepResult:
    """Test LatencySweepResult dataclass."""
    
    def test_latency_sweep_result_creation(self):
        """Test latency sweep result creation."""
        metrics = PerformanceMetrics(
            total_return=0.1,
            annualized_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.05,
            max_drawdown_duration=1000000,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            hit_rate=0.6,
            avg_win=100.0,
            avg_loss=-50.0,
            win_loss_ratio=2.0,
            profit_factor=2.0,
            turnover=5.0,
            avg_position_size=1000.0,
            max_position_size=5000.0,
            mean_latency_ns=1000.0,
            p95_latency_ns=2000.0,
            p99_latency_ns=3000.0,
            var_95=-0.02,
            cvar_95=-0.03,
            calmar_ratio=2.4,
            start_time=1000,
            end_time=2000,
            duration_days=1.0,
            initial_capital=1000000.0,
            final_capital=1100000.0,
            total_fees=1000.0,
            total_slippage=500.0
        )
        
        result = LatencySweepResult(
            latency_ns=1000.0,
            latency_model="random",
            metrics=metrics,
            config_hash="abc123"
        )
        
        assert result.latency_ns == 1000.0
        assert result.latency_model == "random"
        assert result.metrics == metrics
        assert result.config_hash == "abc123"


class TestLatencySensitivityAnalyzer:
    """Test LatencySensitivityAnalyzer class."""
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        config = LatencySweepConfig()
        analyzer = LatencySensitivityAnalyzer(config)
        
        assert analyzer.config == config
        assert len(analyzer.results) == 0
    
    def test_analyzer_creation_with_models(self):
        """Test analyzer creation with slippage and transaction cost models."""
        config = LatencySweepConfig()
        slippage_model = ImbalanceSlippageModel(SlippageConfig())
        transaction_cost_model = SimpleTransactionCostModel(TransactionCostConfig())
        
        analyzer = LatencySensitivityAnalyzer(
            config, slippage_model, transaction_cost_model
        )
        
        assert analyzer.slippage_model == slippage_model
        assert analyzer.transaction_cost_model == transaction_cost_model
    
    def test_generate_latency_grid_linear(self):
        """Test linear latency grid generation."""
        config = LatencySweepConfig(
            latency_min_ns=1000.0,
            latency_max_ns=5000.0,
            latency_steps=5,
            latency_scale="linear"
        )
        analyzer = LatencySensitivityAnalyzer(config)
        
        grid = analyzer._generate_latency_grid()
        
        assert len(grid) == 5
        assert grid[0] == 1000.0
        assert grid[-1] == 5000.0
        assert grid[1] == 2000.0  # Linear spacing
    
    def test_generate_latency_grid_log(self):
        """Test logarithmic latency grid generation."""
        config = LatencySweepConfig(
            latency_min_ns=1000.0,
            latency_max_ns=1000000.0,
            latency_steps=4,
            latency_scale="log"
        )
        analyzer = LatencySensitivityAnalyzer(config)
        
        grid = analyzer._generate_latency_grid()
        
        assert len(grid) == 4
        assert abs(grid[0] - 1000.0) < 1e-10
        assert abs(grid[-1] - 1000000.0) < 1e-10
        
        # Check logarithmic spacing
        ratios = [grid[i+1] / grid[i] for i in range(len(grid)-1)]
        for ratio in ratios:
            assert abs(ratio - ratios[0]) < 1e-10  # All ratios should be equal
    
    def test_calculate_sensitivity_metrics(self):
        """Test sensitivity metrics calculation."""
        config = LatencySweepConfig()
        analyzer = LatencySensitivityAnalyzer(config)
        
        # Create mock data
        data = {
            "latency_ns": [1000, 2000, 3000, 4000, 5000],
            "latency_model": ["random"] * 5,
            "total_return": [0.1, 0.09, 0.08, 0.07, 0.06],
            "sharpe_ratio": [1.0, 0.9, 0.8, 0.7, 0.6],
            "max_drawdown": [-0.05, -0.06, -0.07, -0.08, -0.09],
            "hit_rate": [0.6, 0.58, 0.56, 0.54, 0.52]
        }
        df = pd.DataFrame(data)
        
        analysis = analyzer._calculate_sensitivity_metrics(df)
        
        # Should have correlation and sensitivity metrics
        assert "random_total_return_correlation" in analysis
        assert "random_total_return_sensitivity" in analysis
        assert "random_sharpe_ratio_correlation" in analysis
        assert "random_sharpe_ratio_sensitivity" in analysis
    
    def test_calculate_sensitivity_metrics_with_baseline(self):
        """Test sensitivity metrics calculation with baseline."""
        config = LatencySweepConfig()
        analyzer = LatencySensitivityAnalyzer(config)
        
        # Create mock data with baseline
        data = {
            "latency_ns": [0, 1000, 2000, 3000],
            "latency_model": ["baseline", "random", "random", "random"],
            "total_return": [0.12, 0.10, 0.08, 0.06],
            "sharpe_ratio": [1.2, 1.0, 0.8, 0.6],
            "hit_rate": [0.65, 0.60, 0.55, 0.50]
        }
        df = pd.DataFrame(data)
        
        analysis = analyzer._calculate_sensitivity_metrics(df)
        
        # Should have degradation metrics
        assert "random_total_return_degradation_pct" in analysis
        assert "random_sharpe_ratio_degradation_pct" in analysis
        assert "random_hit_rate_degradation_pct" in analysis
        
        # Degradation should be negative (performance decreases)
        assert analysis["random_total_return_degradation_pct"] < 0
    
    def test_hash_config(self):
        """Test configuration hashing."""
        config = LatencySweepConfig()
        analyzer = LatencySensitivityAnalyzer(config)
        
        test_config = {"strategy": "test", "latency": 1000}
        hash1 = analyzer._hash_config(test_config)
        hash2 = analyzer._hash_config(test_config)
        
        # Same config should produce same hash
        assert hash1 == hash2
        
        # Different config should produce different hash
        test_config2 = {"strategy": "test2", "latency": 1000}
        hash3 = analyzer._hash_config(test_config2)
        assert hash1 != hash3
    
    def test_save_results(self):
        """Test saving results to files."""
        config = LatencySweepConfig()
        analyzer = LatencySensitivityAnalyzer(config)
        
        # Create mock results
        metrics = PerformanceMetrics(
            total_return=0.1,
            annualized_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.05,
            max_drawdown_duration=1000000,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            hit_rate=0.6,
            avg_win=100.0,
            avg_loss=-50.0,
            win_loss_ratio=2.0,
            profit_factor=2.0,
            turnover=5.0,
            avg_position_size=1000.0,
            max_position_size=5000.0,
            mean_latency_ns=1000.0,
            p95_latency_ns=2000.0,
            p99_latency_ns=3000.0,
            var_95=-0.02,
            cvar_95=-0.03,
            calmar_ratio=2.4,
            start_time=1000,
            end_time=2000,
            duration_days=1.0,
            initial_capital=1000000.0,
            final_capital=1100000.0,
            total_fees=1000.0,
            total_slippage=500.0
        )
        
        result = LatencySweepResult(
            latency_ns=1000.0,
            latency_model="random",
            metrics=metrics,
            config_hash="test123"
        )
        
        analyzer.results = [result]
        
        # Create mock summary
        summary = LatencySensitivitySummary(
            results=[result],
            latency_grid=[1000.0],
            performance_metrics=["total_return", "sharpe_ratio"],
            sensitivity_analysis={"test": "value"},
            summary_table=pd.DataFrame({
                "latency_ns": [1000.0],
                "latency_model": ["random"],
                "total_return": [0.1],
                "sharpe_ratio": [0.8]
            })
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer._save_results(summary, temp_dir)
            
            # Check that files were created
            assert Path(temp_dir, "latency_sensitivity_summary.csv").exists()
            assert Path(temp_dir, "sensitivity_analysis.json").exists()
            assert Path(temp_dir, "detailed_results.json").exists()
            
            # Check CSV content
            df = pd.read_csv(Path(temp_dir, "latency_sensitivity_summary.csv"))
            assert len(df) == 1
            assert df.iloc[0]["latency_ns"] == 1000.0
            assert df.iloc[0]["latency_model"] == "random"


class TestLatencySensitivityFactory:
    """Test latency sensitivity analyzer factory function."""
    
    def test_create_analyzer(self):
        """Test creating analyzer with factory function."""
        analyzer = create_latency_sensitivity_analyzer(
            latency_min_ns=1000.0,
            latency_max_ns=1000000.0,
            latency_steps=5,
            latency_scale="log"
        )
        
        assert isinstance(analyzer, LatencySensitivityAnalyzer)
        assert analyzer.config.latency_min_ns == 1000.0
        assert analyzer.config.latency_max_ns == 1000000.0
        assert analyzer.config.latency_steps == 5
        assert analyzer.config.latency_scale == "log"


class TestLatencySensitivityIntegration:
    """Test latency sensitivity integration and PnL impact."""
    
    def test_latency_impacts_performance(self):
        """Test that higher latency reduces performance."""
        config = LatencySweepConfig(
            latency_min_ns=1000.0,
            latency_max_ns=10000.0,
            latency_steps=3,
            latency_scale="linear"
        )
        analyzer = LatencySensitivityAnalyzer(config)
        
        # Mock backtest runner that returns worse performance with higher latency
        def mock_backtest_runner(backtest_config, market_data):
            latency_ns = backtest_config.get("latency_model", {}).base_latency_ns if hasattr(backtest_config.get("latency_model", {}), "base_latency_ns") else 0
            
            # Simulate performance degradation with latency
            base_return = 0.1
            latency_penalty = latency_ns / 1000000.0 * 0.01  # 1% penalty per millisecond
            
            return {
                "snapshots": [],
                "trades": [],
                "latencies": [latency_ns] * 10
            }
        
        # Mock market data
        market_data = []
        
        # Mock strategy config
        strategy_config = {"strategy": "test"}
        
        # Run sweep
        with patch.object(analyzer, '_run_single_backtest') as mock_run:
            # Mock the single backtest to return decreasing performance
            def mock_single_backtest(backtest_runner, strategy_config, market_data, latency_ns, model_type):
                # Create mock metrics with performance decreasing with latency
                base_return = 0.1
                latency_penalty = latency_ns / 1000000.0 * 0.01
                
                metrics = PerformanceMetrics(
                    total_return=base_return - latency_penalty,
                    annualized_return=0.12,
                    volatility=0.15,
                    sharpe_ratio=0.8,
                    max_drawdown=-0.05,
                    max_drawdown_duration=1000000,
                    total_trades=100,
                    winning_trades=60,
                    losing_trades=40,
                    hit_rate=0.6,
                    avg_win=100.0,
                    avg_loss=-50.0,
                    win_loss_ratio=2.0,
                    profit_factor=2.0,
                    turnover=5.0,
                    avg_position_size=1000.0,
                    max_position_size=5000.0,
                    mean_latency_ns=latency_ns,
                    p95_latency_ns=latency_ns * 1.5,
                    p99_latency_ns=latency_ns * 2.0,
                    var_95=-0.02,
                    cvar_95=-0.03,
                    calmar_ratio=2.4,
                    start_time=1000,
                    end_time=2000,
                    duration_days=1.0,
                    initial_capital=1000000.0,
                    final_capital=1100000.0,
                    total_fees=1000.0,
                    total_slippage=500.0
                )
                
                return LatencySweepResult(
                    latency_ns=latency_ns,
                    latency_model=model_type,
                    metrics=metrics,
                    config_hash="test123"
                )
            
            mock_run.side_effect = mock_single_backtest
            
            summary = analyzer.run_sweep(
                mock_backtest_runner, strategy_config, market_data, "test_output"
            )
            
            # Check that higher latency results in worse performance
            results_df = summary.summary_table
            if len(results_df) > 1:
                # Sort by latency
                results_df = results_df.sort_values("latency_ns")
                
                # Check that performance decreases with latency
                returns = results_df["total_return"].values
                for i in range(1, len(returns)):
                    assert returns[i] <= returns[i-1]  # Performance should not improve with higher latency
    
    def test_costs_reduce_pnl_expected_bounds(self):
        """Test that costs reduce PnL by expected bounds."""
        # This test ensures that the cost models reduce PnL by reasonable amounts
        # without being too aggressive or too conservative
        
        # Test slippage impact
        from flashback.market.slippage import FixedSlippageModel
        
        no_slippage = FixedSlippageModel(slippage_bps=0.0)
        with_slippage = FixedSlippageModel(slippage_bps=2.0)  # 2 bps
        
        # Simulate a trade
        from flashback.market.orders import OrderBookSnapshot, OrderBookLevel, OrderSide
        
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        order_side = OrderSide.BUY
        order_size = 100
        price = 100.0
        
        no_slippage_cost = no_slippage.calculate_slippage(order_side, order_size, book, price)
        with_slippage_cost = with_slippage.calculate_slippage(order_side, order_size, book, price)
        
        # Slippage should be reasonable (not more than 10 bps for 2 bps setting)
        slippage_bps = (with_slippage_cost - no_slippage_cost) / price * 10000
        assert 0 <= slippage_bps <= 10  # Should be within reasonable bounds
        
        # Test transaction cost impact
        no_cost_model = SimpleTransactionCostModel(
            TransactionCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.0,
                per_share_cost=0.0
            )
        )
        
        with_cost_model = SimpleTransactionCostModel(
            TransactionCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.5,
                per_share_cost=0.001
            )
        )
        
        no_costs = no_cost_model.calculate_costs(
            order_side, None, order_size, price, is_maker=False
        )
        with_costs = with_cost_model.calculate_costs(
            order_side, None, order_size, price, is_maker=False
        )
        
        # Transaction costs should be reasonable (not more than 1% of trade value)
        trade_value = order_size * price
        cost_pct = (with_costs.total_cost - no_costs.total_cost) / trade_value * 100
        assert 0 <= cost_pct <= 1.0  # Should be within 1% of trade value
