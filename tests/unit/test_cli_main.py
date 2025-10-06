"""Unit tests for CLI main module."""

import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock


class TestCLIMain:
    """Test CLI main functionality."""
    
    def test_run_command_help(self):
        """Test run command help."""
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--help"], 
                               capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage: flashback run" in result.stdout
        assert "--config" in result.stdout
    
    def test_sweep_command_help(self):
        """Test sweep command help."""
        result = subprocess.run([sys.executable, "-m", "flashback", "sweep", "--help"], 
                               capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage: flashback sweep" in result.stdout
        assert "--config" in result.stdout
        assert "--latency" in result.stdout
    
    def test_run_command_missing_config(self):
        """Test run command with missing config file."""
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--config", "nonexistent.yaml"], 
                               capture_output=True, text=True)
        assert result.returncode == 0  # CLI returns 0 even for errors
        assert "Configuration file not found" in result.stderr
    
    def test_sweep_command_missing_config(self):
        """Test sweep command with missing config file."""
        result = subprocess.run([sys.executable, "-m", "flashback", "sweep", "--config", "nonexistent.yaml", "--latency", "100000"], 
                               capture_output=True, text=True)
        assert result.returncode == 0  # CLI returns 0 even for errors
        assert "Configuration file not found" in result.stderr
    
    def test_sweep_command_invalid_latency(self):
        """Test sweep command with invalid latency values."""
        result = subprocess.run([sys.executable, "-m", "flashback", "sweep", "--config", "config/backtest.yaml", "--latency", "invalid"], 
                               capture_output=True, text=True)
        assert result.returncode == 0  # CLI returns 0 even for errors
        assert "invalid literal for int()" in result.stderr
    
    @patch('flashback.cli.runner.BacktestRunner')
    def test_run_command_success(self, mock_runner_class):
        """Test successful run command."""
        # Mock the runner
        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "success", "performance": {"total_return": 0.05}}
        mock_runner_class.return_value = mock_runner
        
        result = subprocess.run([sys.executable, "-m", "flashback", "run", "--config", "config/backtest.yaml"], 
                               capture_output=True, text=True)
        assert result.returncode == 0
        assert "Backtest completed successfully" in result.stderr
    
    @patch('flashback.cli.sweeper.LatencySweeper')
    def test_sweep_command_success(self, mock_sweeper_class):
        """Test successful sweep command."""
        # Mock the sweeper
        mock_sweeper = MagicMock()
        mock_sweeper.run.return_value = {"status": "success", "results": []}
        mock_sweeper_class.return_value = mock_sweeper
        
        result = subprocess.run([sys.executable, "-m", "flashback", "sweep", "--config", "config/backtest.yaml", "--latency", "100000,200000"], 
                               capture_output=True, text=True)
        assert result.returncode == 0
        assert "Latency sweep completed successfully" in result.stderr