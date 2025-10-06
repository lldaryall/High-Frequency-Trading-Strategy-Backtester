"""Integration tests for CLI functionality."""

import pytest
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path


class TestCLIIntegration:
    """Test CLI integration with real components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_config = Path(self.temp_dir) / "test_config.yaml"
        self.temp_data = Path(self.temp_dir) / "test_data.csv"
        
        # Create test data
        self._create_test_data()
        self._create_test_config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test market data."""
        data_content = """timestamp,symbol,side,price,size,event_type
1000000000000000000,AAPL,BUY,150.0,100,TRADE
1000000001000000000,AAPL,SELL,150.5,50,TRADE
1000000002000000000,AAPL,BUY,151.0,75,TRADE
1000000003000000000,AAPL,SELL,150.8,25,TRADE
1000000004000000000,AAPL,BUY,151.2,100,TRADE
"""
        self.temp_data.write_text(data_content)
    
    def _create_test_config(self):
        """Create test configuration."""
        config_content = f"""data:
  path: "{self.temp_data}"
  kind: "trade"
  format: "csv"

strategy:
  name: "momentum_imbalance"
  symbol: "AAPL"
  enabled: true
  max_position: 1000
  max_order_size: 100
  risk_limits: []
  params:
    short_ema_period: 5
    long_ema_period: 10
    imbalance_threshold: 0.6
    take_profit_pct: 0.02
    stop_loss_pct: 0.01

execution:
  fees:
    maker_bps: 0.0
    taker_bps: 0.5
    per_share: 0.0
  latency:
    model: "normal"
    mean_ns: 100000
    std_ns: 20000
    seed: 42

risk:
  max_gross: 100000
  max_pos_per_symbol: 1000
  max_daily_loss: -2000

report:
  output_dir: "{self.temp_dir}/test_run"
  format: "both"
"""
        self.temp_config.write_text(config_content)
    
    def test_run_command_integration(self):
        """Test run command with real components."""
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--config", str(self.temp_config)], 
                               capture_output=True, text=True)
        
        # Should complete successfully
        assert result.returncode == 0
        assert "Backtest completed successfully" in result.stderr
        
        # Check that output files were created
        output_dir = Path(self.temp_dir) / "test_run"
        assert output_dir.exists()
        assert (output_dir / "performance.json").exists()
        assert (output_dir / "performance.csv").exists()
        assert (output_dir / "detailed_results.json").exists()
    
    def test_sweep_command_integration(self):
        """Test sweep command with real components."""
        result = subprocess.run([sys.executable, "-m", "flashback", "sweep", 
                               "--config", str(self.temp_config),
                               "--latency", "50000,100000,150000"], 
                               capture_output=True, text=True)
        
        # Should complete successfully
        assert result.returncode == 0
        assert "Latency sweep completed successfully" in result.stderr
        
        # Check that sweep files were created
        output_dir = Path(self.temp_dir) / "test_run"
        assert output_dir.exists()
        assert (output_dir / "sensitivity_analysis.json").exists()
        assert (output_dir / "latency_sensitivity_summary.csv").exists()
    
    def test_run_command_with_invalid_config(self):
        """Test run command with invalid configuration."""
        # Create invalid config
        invalid_config = Path(self.temp_dir) / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content: [")
        
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--config", str(invalid_config)], 
                               capture_output=True, text=True)
        assert result.returncode == 0  # CLI returns 0 even for errors
        assert "Invalid YAML syntax" in result.stderr
    
    def test_run_command_with_missing_data_file(self):
        """Test run command with missing data file."""
        # Create config pointing to non-existent data file
        config_content = f"""data:
  path: "nonexistent.csv"
  kind: "trade"
  format: "csv"

strategy:
  name: "momentum_imbalance"
  symbol: "AAPL"
  enabled: true
  max_position: 1000
  max_order_size: 100
  risk_limits: []
  params:
    short_ema_period: 5
    long_ema_period: 10
    imbalance_threshold: 0.6
    take_profit_pct: 0.02
    stop_loss_pct: 0.01

execution:
  fees:
    maker_bps: 0.0
    taker_bps: 0.5
    per_share: 0.0
  latency:
    model: "normal"
    mean_ns: 100000
    std_ns: 20000
    seed: 42

risk:
  max_gross: 100000
  max_pos_per_symbol: 1000
  max_daily_loss: -2000

report:
  output_dir: "{self.temp_dir}/test_run"
  format: "both"
"""
        invalid_config = Path(self.temp_dir) / "invalid_data_config.yaml"
        invalid_config.write_text(config_content)
        
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--config", str(invalid_config)], 
                               capture_output=True, text=True)
        assert result.returncode == 0  # CLI returns 0 even for errors
        assert "Data file not found" in result.stderr