"""
Unit tests for performance metrics calculation.
"""

import pytest
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from flashback.metrics.performance import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    TradeRecord
)
from flashback.risk.portfolio import PortfolioSnapshot, Position


class TestTradeRecord:
    """Test TradeRecord dataclass."""
    
    def test_trade_record_creation(self):
        """Test trade record creation."""
        trade = TradeRecord(
            symbol="AAPL",
            entry_time=1000,
            exit_time=2000,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            side="BUY",
            pnl=500.0,
            fees=1.0,
            duration_ns=1000,
            strategy_id="test_strategy"
        )
        
        assert trade.symbol == "AAPL"
        assert trade.entry_time == 1000
        assert trade.exit_time == 2000
        assert trade.entry_price == 150.0
        assert trade.exit_price == 155.0
        assert trade.quantity == 100
        assert trade.side == "BUY"
        assert trade.pnl == 500.0
        assert trade.fees == 1.0
        assert trade.duration_ns == 1000
        assert trade.strategy_id == "test_strategy"


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        
        assert analyzer.risk_free_rate == 0.02
        assert len(analyzer.snapshots) == 0
        assert len(analyzer.trades) == 0
        assert len(analyzer.latencies) == 0
    
    def test_add_snapshot(self):
        """Test adding portfolio snapshots."""
        analyzer = PerformanceAnalyzer()
        
        # Create mock snapshot
        snapshot = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=100000.0,
            total_market_value=50000.0,
            total_unrealized_pnl=1000.0,
            total_realized_pnl=2000.0,
            total_pnl=3000.0,
            gross_exposure=50000.0,
            net_exposure=50000.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot)
        
        assert len(analyzer.snapshots) == 1
        assert analyzer.snapshots[0] == snapshot
    
    def test_add_trade(self):
        """Test adding trade records."""
        analyzer = PerformanceAnalyzer()
        
        trade = TradeRecord(
            symbol="AAPL",
            entry_time=1000,
            exit_time=2000,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            side="BUY",
            pnl=500.0,
            fees=1.0,
            duration_ns=1000,
            strategy_id="test_strategy"
        )
        
        analyzer.add_trade(trade)
        
        assert len(analyzer.trades) == 1
        assert analyzer.trades[0] == trade
    
    def test_add_latency(self):
        """Test adding latency measurements."""
        analyzer = PerformanceAnalyzer()
        
        analyzer.add_latency(1000.0)  # 1 microsecond
        analyzer.add_latency(2000.0)  # 2 microseconds
        
        assert len(analyzer.latencies) == 2
        assert analyzer.latencies == [1000.0, 2000.0]
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots with known values
        initial_capital = 1000000.0
        final_capital = 1100000.0
        
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=initial_capital,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=final_capital,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        metrics = analyzer.calculate_metrics()
        
        # Check basic metrics
        expected_return = (final_capital - initial_capital) / initial_capital
        assert abs(metrics.total_return - expected_return) < 1e-10
        assert metrics.initial_capital == initial_capital
        assert metrics.final_capital == final_capital
        assert metrics.start_time == 1000000000
        assert metrics.end_time == 2000000000
    
    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trade data."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        # Add winning trade
        winning_trade = TradeRecord(
            symbol="AAPL",
            entry_time=1000,
            exit_time=1500,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            side="BUY",
            pnl=500.0,
            fees=1.0,
            duration_ns=500,
            strategy_id="test_strategy"
        )
        
        # Add losing trade
        losing_trade = TradeRecord(
            symbol="MSFT",
            entry_time=1000,
            exit_time=1500,
            entry_price=300.0,
            exit_price=295.0,
            quantity=50,
            side="BUY",
            pnl=-250.0,
            fees=1.0,
            duration_ns=500,
            strategy_id="test_strategy"
        )
        
        analyzer.add_trade(winning_trade)
        analyzer.add_trade(losing_trade)
        
        metrics = analyzer.calculate_metrics()
        
        # Check trading metrics
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.hit_rate == 0.5
        assert metrics.avg_win == 500.0
        assert metrics.avg_loss == -250.0
        assert metrics.win_loss_ratio == 2.0  # 500 / 250
        assert metrics.profit_factor == 2.0  # 500 / 250
    
    def test_calculate_metrics_with_latency(self):
        """Test metrics calculation with latency data."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        # Add latency data
        latencies = [1000.0, 2000.0, 1500.0, 3000.0, 1200.0]
        for latency in latencies:
            analyzer.add_latency(latency)
        
        metrics = analyzer.calculate_metrics()
        
        # Check latency metrics
        assert metrics.mean_latency_ns == np.mean(latencies)
        assert metrics.p95_latency_ns == np.percentile(latencies, 95)
        assert metrics.p99_latency_ns == np.percentile(latencies, 99)
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots with known drawdown pattern
        snapshots = []
        values = [1000000, 1100000, 1200000, 1050000, 900000, 950000, 1300000]
        
        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=1000000000 + i * 1000000000,  # 1 second intervals
                cash=value,
                total_market_value=0.0,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                total_pnl=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                positions={}
            )
            snapshots.append(snapshot)
            analyzer.add_snapshot(snapshot)
        
        metrics = analyzer.calculate_metrics()
        
        # The maximum drawdown should be from 1200000 to 900000
        # Drawdown = (900000 - 1200000) / 1200000 = -0.25 = -25%
        expected_drawdown = (900000 - 1200000) / 1200000
        assert abs(metrics.max_drawdown - expected_drawdown) < 1e-10
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots with known returns
        snapshots = []
        values = [1000000, 1010000, 1020000, 1015000, 1030000]  # 1%, 1%, -0.5%, 1.5%
        
        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=1000000000 + i * 1000000000,  # 1 second intervals
                cash=value,
                total_market_value=0.0,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                total_pnl=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                positions={}
            )
            snapshots.append(snapshot)
            analyzer.add_snapshot(snapshot)
        
        metrics = analyzer.calculate_metrics()
        
        # Check that volatility is calculated (should be > 0)
        assert metrics.volatility > 0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        
        # Create snapshots with known returns
        snapshots = []
        values = [1000000, 1010000, 1020000, 1015000, 1030000]
        
        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=1000000000 + i * 1000000000,  # 1 second intervals
                cash=value,
                total_market_value=0.0,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                total_pnl=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                positions={}
            )
            snapshots.append(snapshot)
            analyzer.add_snapshot(snapshot)
        
        metrics = analyzer.calculate_metrics()
        
        # Check that Sharpe ratio is calculated
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_export_json(self):
        """Test JSON export functionality."""
        analyzer = PerformanceAnalyzer()
        
        # Create minimal data
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=1100000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_json(temp_path)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'basic_metrics' in data
            assert 'trading_metrics' in data
            assert 'portfolio_metrics' in data
            assert 'latency_metrics' in data
            assert 'risk_metrics' in data
            assert 'summary' in data
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_export_csv(self):
        """Test CSV export functionality."""
        analyzer = PerformanceAnalyzer()
        
        # Create minimal data
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=1100000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_csv(temp_path)
            
            # Verify file was created and contains data
            df = pd.read_csv(temp_path)
            
            assert 'Metric' in df.columns
            assert 'Value' in df.columns
            assert len(df) > 0
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_generate_plots(self):
        """Test plot generation functionality."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots for plotting
        snapshots = []
        values = [1000000, 1010000, 1020000, 1015000, 1030000, 1050000]
        
        for i, value in enumerate(values):
            snapshot = PortfolioSnapshot(
                timestamp=1000000000 + i * 1000000000,  # 1 second intervals
                cash=value,
                total_market_value=0.0,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                total_pnl=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                positions={}
            )
            snapshots.append(snapshot)
            analyzer.add_snapshot(snapshot)
        
        # Add some trades for histogram
        trade1 = TradeRecord(
            symbol="AAPL",
            entry_time=1000,
            exit_time=1500,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            side="BUY",
            pnl=500.0,
            fees=1.0,
            duration_ns=500,
            strategy_id="test_strategy"
        )
        
        trade2 = TradeRecord(
            symbol="MSFT",
            entry_time=1000,
            exit_time=1500,
            entry_price=300.0,
            exit_price=295.0,
            quantity=50,
            side="BUY",
            pnl=-250.0,
            fees=1.0,
            duration_ns=500,
            strategy_id="test_strategy"
        )
        
        analyzer.add_trade(trade1)
        analyzer.add_trade(trade2)
        
        # Generate plots in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer.generate_plots(temp_dir)
            
            # Check that plot files were created
            plot_files = [
                "equity_curve.png",
                "drawdown_curve.png", 
                "trade_pnl_histogram.png"
            ]
            
            for plot_file in plot_files:
                plot_path = Path(temp_dir) / plot_file
                assert plot_path.exists()
    
    def test_generate_report(self):
        """Test complete report generation."""
        analyzer = PerformanceAnalyzer()
        
        # Create snapshots
        snapshot1 = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        snapshot2 = PortfolioSnapshot(
            timestamp=2000000000,  # 2 seconds in nanoseconds
            cash=1100000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot1)
        analyzer.add_snapshot(snapshot2)
        
        # Add some trades for histogram
        trade1 = TradeRecord(
            symbol="AAPL",
            entry_time=1000000000,
            exit_time=1500000000,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            side="BUY",
            pnl=500.0,
            fees=1.0,
            duration_ns=500000000,
            strategy_id="test_strategy"
        )
        
        trade2 = TradeRecord(
            symbol="MSFT",
            entry_time=1000000000,
            exit_time=1500000000,
            entry_price=300.0,
            exit_price=295.0,
            quantity=50,
            side="BUY",
            pnl=-250.0,
            fees=1.0,
            duration_ns=500000000,
            strategy_id="test_strategy"
        )
        
        analyzer.add_trade(trade1)
        analyzer.add_trade(trade2)
        
        # Generate report in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer.generate_report(temp_dir)
            
            # Check that all files were created
            expected_files = [
                "performance.json",
                "performance.csv",
                "equity_curve.png",
                "drawdown_curve.png",
                "trade_pnl_histogram.png"
            ]
            
            for file in expected_files:
                file_path = Path(temp_dir) / file
                assert file_path.exists()
    
    def test_empty_analyzer(self):
        """Test behavior with empty analyzer."""
        analyzer = PerformanceAnalyzer()
        
        # Should raise error when trying to calculate metrics with no data
        with pytest.raises(ValueError, match="No snapshots available"):
            analyzer.calculate_metrics()
    
    def test_single_snapshot(self):
        """Test behavior with single snapshot."""
        analyzer = PerformanceAnalyzer()
        
        snapshot = PortfolioSnapshot(
            timestamp=1000000000,  # 1 second in nanoseconds
            cash=1000000.0,
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            positions={}
        )
        
        analyzer.add_snapshot(snapshot)
        
        metrics = analyzer.calculate_metrics()
        
        # With single snapshot, should have zero return
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.volatility == 0.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
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
        
        assert metrics.total_return == 0.1
        assert metrics.annualized_return == 0.12
        assert metrics.volatility == 0.15
        assert metrics.sharpe_ratio == 0.8
        assert metrics.max_drawdown == -0.05
        assert metrics.total_trades == 100
        assert metrics.hit_rate == 0.6
        assert metrics.avg_win == 100.0
        assert metrics.avg_loss == -50.0
        assert metrics.win_loss_ratio == 2.0
        assert metrics.profit_factor == 2.0
        assert metrics.turnover == 5.0
        assert metrics.mean_latency_ns == 1000.0
        assert metrics.p95_latency_ns == 2000.0
        assert metrics.p99_latency_ns == 3000.0
        assert metrics.var_95 == -0.02
        assert metrics.cvar_95 == -0.03
        assert metrics.calmar_ratio == 2.4
        assert metrics.start_time == 1000
        assert metrics.end_time == 2000
        assert metrics.duration_days == 1.0
        assert metrics.initial_capital == 1000000.0
        assert metrics.final_capital == 1100000.0
        assert metrics.total_fees == 1000.0
        assert metrics.total_slippage == 500.0
