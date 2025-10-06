"""
Unit tests for configuration parser.
"""

import pytest
import yaml
import tempfile
from pathlib import Path

from flashback.config.parser import ConfigParser, load_config, validate_config
from flashback.config.config import BacktestConfig, DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, ReportConfig


class TestConfigParser:
    """Test ConfigParser class."""
    
    def test_parser_creation(self):
        """Test parser creation."""
        parser = ConfigParser()
        assert parser is not None
    
    def test_load_yaml_valid_file(self):
        """Test loading valid YAML file."""
        parser = ConfigParser()
        
        # Create temporary YAML file
        config_data = {
            "data": {"path": "test.csv", "kind": "book"},
            "strategy": {"name": "momentum_imbalance", "params": {}},
            "execution": {
                "fees": {"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                "latency": {"model": "normal", "mean_ns": 500000, "std_ns": 100000}
            },
            "risk": {
                "max_gross": 100000,
                "max_pos_per_symbol": 1000,
                "max_daily_loss": -2000
            },
            "report": {"output_dir": "runs/test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            result = parser.load_yaml(temp_path)
            assert result == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_load_yaml_nonexistent_file(self):
        """Test loading non-existent YAML file."""
        parser = ConfigParser()
        
        with pytest.raises(FileNotFoundError):
            parser.load_yaml("nonexistent.yaml")
    
    def test_load_yaml_invalid_extension(self):
        """Test loading file with invalid extension."""
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration file must be YAML format"):
                parser.load_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_yaml_invalid_yaml(self):
        """Test loading invalid YAML file."""
        parser = ConfigParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML syntax"):
                parser.load_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_parse_config_valid(self):
        """Test parsing valid configuration."""
        parser = ConfigParser()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config_dict = {
                "data": {"path": temp_path, "kind": "book"},
                "strategy": {"name": "momentum_imbalance", "params": {"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}},
                "execution": {
                    "fees": {"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                    "latency": {"model": "normal", "mean_ns": 500000, "std_ns": 100000}
                },
                "risk": {
                    "max_gross": 100000,
                    "max_pos_per_symbol": 1000,
                    "max_daily_loss": -2000
                },
                "report": {"output_dir": "runs/test"}
            }
            
            config = parser.parse_config(config_dict)
            assert isinstance(config, BacktestConfig)
            assert config.data.kind == "book"
            assert config.strategy.name == "momentum_imbalance"
            assert config.strategy.params["short_ema_period"] == 5
        finally:
            Path(temp_path).unlink()
    
    def test_parse_config_missing_section(self):
        """Test parsing config with missing section."""
        parser = ConfigParser()
        
        config_dict = {
            "data": {"path": "test.csv", "kind": "book"},
            # Missing strategy section
            "execution": {
                "fees": {"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                "latency": {"model": "normal", "mean_ns": 500000, "std_ns": 100000}
            },
            "risk": {
                "max_gross": 100000,
                "max_pos_per_symbol": 1000,
                "max_daily_loss": -2000
            },
            "report": {"output_dir": "runs/test"}
        }
        
        with pytest.raises(ValueError, match="Missing required configuration section"):
            parser.parse_config(config_dict)
    
    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        parser = ConfigParser()
        
        # Create temporary file for validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,symbol,side,price,size,event_type\n")
            f.write("1000000000,AAPL,BUY,150.00,100,TRADE\n")
            temp_path = f.name
        
        try:
            # Create a valid config
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(
                    name="momentum_imbalance",
                    params={"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}
                ),
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
            
            result = parser.validate_config(config)
            assert result is True
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_missing_data_file(self):
        """Test validating config with missing data file."""
        parser = ConfigParser()
        
        # Create a config with a non-existent file by bypassing the DataConfig constructor
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            # Create config with valid file first
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(name="momentum_imbalance", params={"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}),
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
            
            # Now delete the file to test validation
            Path(temp_path).unlink()
            
            with pytest.raises(ValueError, match="Data file does not exist"):
                parser.validate_config(config)
        finally:
            # Clean up if file still exists
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def test_validate_config_missing_strategy_params(self):
        """Test validating config with missing strategy parameters."""
        parser = ConfigParser()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(
                    name="momentum_imbalance",
                    params={"short_ema_period": 5}  # Missing required params
                ),
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
            
            with pytest.raises(ValueError, match="Missing required parameter for momentum_imbalance strategy"):
                parser.validate_config(config)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_invalid_fees(self):
        """Test validating config with invalid fees."""
        parser = ConfigParser()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(name="momentum_imbalance", params={"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}),
                execution=ExecutionConfig(
                    fees={"maker_bps": -0.1, "taker_bps": 0.5, "per_share": 0.0},  # Negative fee
                    latency={"model": "normal", "mean_ns": 500000, "std_ns": 100000}
                ),
                risk=RiskConfig(
                    max_gross=100000,
                    max_pos_per_symbol=1000,
                    max_daily_loss=-2000
                ),
                report=ReportConfig(output_dir="runs/test")
            )
            
            with pytest.raises(ValueError, match="Fee rates cannot be negative"):
                parser.validate_config(config)
        finally:
            Path(temp_path).unlink()


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    def test_load_config(self):
        """Test load_config function."""
        config_data = {
            "data": {"path": "test.csv", "kind": "book"},
            "strategy": {"name": "momentum_imbalance", "params": {"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}},
            "execution": {
                "fees": {"maker_bps": 0.0, "taker_bps": 0.5, "per_share": 0.0},
                "latency": {"model": "normal", "mean_ns": 500000, "std_ns": 100000}
            },
            "risk": {
                "max_gross": 100000,
                "max_pos_per_symbol": 1000,
                "max_daily_loss": -2000
            },
            "report": {"output_dir": "runs/test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
            data_file.write("timestamp,symbol,side,price,size,event_type\n")
            data_file.write("1000000000,AAPL,BUY,150.00,100,TRADE\n")
            data_path = data_file.name
        
        try:
            # Update config to use the data file
            config_data["data"]["path"] = data_path
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                config_path = f.name
            
            config = load_config(config_path)
            assert isinstance(config, BacktestConfig)
            assert config.data.kind == "book"
            assert config.strategy.name == "momentum_imbalance"
        finally:
            Path(temp_path).unlink()
            Path(data_path).unlink()
            Path(config_path).unlink()
    
    def test_validate_config_function(self):
        """Test validate_config function."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        
        try:
            config = BacktestConfig(
                data=DataConfig(path=temp_path, kind="book"),
                strategy=StrategyConfig(name="momentum_imbalance", params={"short_ema_period": 5, "long_ema_period": 20, "imbalance_threshold": 0.3}),
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
            
            # This should succeed with valid config
            result = validate_config(config)
            assert result is True
        finally:
            Path(temp_path).unlink()
