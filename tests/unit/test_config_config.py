"""
Unit tests for configuration dataclasses.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from flashback.config.config import (
    DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, 
    ReportConfig, BacktestConfig
)


class TestDataConfig:
    """Test DataConfig dataclass."""
    
    def test_data_config_creation(self):
        """Test data config creation."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = DataConfig(
                path=temp_path,
                kind="book"
            )
            assert config.path == temp_path
            assert config.kind == "book"
            assert config.symbol is None
        finally:
            Path(temp_path).unlink()
    
    def test_data_config_validation(self):
        """Test data config validation."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            # Valid config
            config = DataConfig(path=temp_path, kind="book")
            assert config.kind == "book"
            
            # Invalid kind
            with pytest.raises(ValueError, match="Data kind must be 'book' or 'trade'"):
                DataConfig(path=temp_path, kind="invalid")
        finally:
            Path(temp_path).unlink()
    
    def test_data_config_file_not_found(self):
        """Test data config with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DataConfig(path="nonexistent.csv", kind="book")


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_strategy_config_creation(self):
        """Test strategy config creation."""
        config = StrategyConfig(
            name="momentum_imbalance",
            params={"short_ema_period": 5, "long_ema_period": 20}
        )
        assert config.name == "momentum_imbalance"
        assert config.params["short_ema_period"] == 5
        assert config.enabled is True
        assert config.max_position == 1000
    
    def test_strategy_config_validation(self):
        """Test strategy config validation."""
        # Valid config
        config = StrategyConfig(
            name="mean_reversion",
            params={"lookback_period": 20}
        )
        assert config.name == "mean_reversion"
        
        # Invalid strategy name
        with pytest.raises(ValueError, match="Unknown strategy"):
            StrategyConfig(name="invalid_strategy", params={})


class TestExecutionConfig:
    """Test ExecutionConfig dataclass."""
    
    def test_execution_config_creation(self):
        """Test execution config creation."""
        config = ExecutionConfig(
            fees={"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
            latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000}
        )
        assert config.fees["maker_bps"] == 0.0
        assert config.latency["mean_ns"] == 500000
    
    def test_execution_config_validation(self):
        """Test execution config validation."""
        # Missing fee parameter
        with pytest.raises(ValueError, match="Missing required fee parameter"):
            ExecutionConfig(
                fees={"maker_bps": 0.0},  # Missing taker_bps and per_share
                latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000}
            )
        
        # Missing latency parameter
        with pytest.raises(ValueError, match="Missing required latency parameter"):
            ExecutionConfig(
                fees={"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                latency={"model": "normal"}  # Missing mean_ns and std_ns
            )


class TestRiskConfig:
    """Test RiskConfig dataclass."""
    
    def test_risk_config_creation(self):
        """Test risk config creation."""
        config = RiskConfig(
            max_gross=100000,
            max_pos_per_symbol=1000,
            max_daily_loss=-2000
        )
        assert config.max_gross == 100000
        assert config.max_pos_per_symbol == 1000
        assert config.max_daily_loss == -2000
    
    def test_risk_config_validation(self):
        """Test risk config validation."""
        # Valid config
        config = RiskConfig(
            max_gross=100000,
            max_pos_per_symbol=1000,
            max_daily_loss=-2000
        )
        assert config.max_gross > 0
        
        # Invalid max_gross
        with pytest.raises(ValueError, match="max_gross must be positive"):
            RiskConfig(
                max_gross=0,
                max_pos_per_symbol=1000,
                max_daily_loss=-2000
            )
        
        # Invalid max_daily_loss
        with pytest.raises(ValueError, match="max_daily_loss must be negative"):
            RiskConfig(
                max_gross=100000,
                max_pos_per_symbol=1000,
                max_daily_loss=2000  # Should be negative
            )


class TestReportConfig:
    """Test ReportConfig dataclass."""
    
    def test_report_config_creation(self):
        """Test report config creation."""
        config = ReportConfig(output_dir="runs/test")
        assert config.output_dir == "runs/test"
        assert config.format == "json"
        assert config.plots is True
    
    def test_report_config_validation(self):
        """Test report config validation."""
        # Valid config
        config = ReportConfig(output_dir="runs/test", format="csv")
        assert config.format == "csv"
        
        # Invalid format
        with pytest.raises(ValueError, match="Report format must be"):
            ReportConfig(output_dir="runs/test", format="invalid")


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""
    
    def test_backtest_config_creation(self):
        """Test backtest config creation."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(name="momentum_imbalance", params={}),
                execution=ExecutionConfig(
                    fees={"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                    latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000}
                ),
                risk=RiskConfig(
                    max_gross=100000,
                    max_pos_per_symbol=1000,
                    max_daily_loss=-2000
                ),
                report=ReportConfig(output_dir="runs/test")
            )
            assert config.data.kind == "book"
            assert config.strategy.name == "momentum_imbalance"
            assert config.risk.max_gross == 100000
        finally:
            Path(temp_path).unlink()
    
    def test_backtest_config_from_dict(self):
        """Test creating BacktestConfig from dictionary."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config_dict = {
                "data": {
                    "path": temp_path,
                    "kind": "book"
                },
                "strategy": {
                    "name": "momentum_imbalance",
                    "params": {"short_ema_period": 5}
                },
                "execution": {
                    "fees": {"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                    "latency": {"model": "normal", "mean_ns": 500000, "std_ns": 100000}
                },
                "risk": {
                    "max_gross": 100000,
                    "max_pos_per_symbol": 1000,
                    "max_daily_loss": -2000
                },
                "report": {
                    "output_dir": "runs/test"
                }
            }
            
            config = BacktestConfig.from_dict(config_dict)
            assert config.data.path == temp_path
            assert config.strategy.params["short_ema_period"] == 5
        finally:
            Path(temp_path).unlink()
    
    def test_backtest_config_to_dict(self):
        """Test converting BacktestConfig to dictionary."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(name="momentum_imbalance", params={}),
                execution=ExecutionConfig(
                    fees={"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                    latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000}
                ),
                risk=RiskConfig(
                    max_gross=100000,
                    max_pos_per_symbol=1000,
                    max_daily_loss=-2000
                ),
                report=ReportConfig(output_dir="runs/test")
            )
            
            config_dict = config.to_dict()
            assert config_dict["data"]["path"] == temp_path
            assert config_dict["strategy"]["name"] == "momentum_imbalance"
            assert config_dict["risk"]["max_gross"] == 100000
        finally:
            Path(temp_path).unlink()
