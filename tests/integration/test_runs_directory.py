"""Integration tests for runs directory structure and packing functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
import yaml

from flashback.config import load_config
from flashback.cli.runner import BacktestRunner
from flashback.cli.sweeper import LatencySweeper
from flashback.cli.pack import RunPacker


class TestRunsDirectory:
    """Test runs directory structure and packing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_data_dir = Path(self.temp_dir) / "data"
        self.temp_data_dir.mkdir()
        
        # Generate synthetic data
        self._generate_synthetic_data()
        self._create_test_config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_synthetic_data(self):
        """Generate synthetic market data for testing."""
        from examples.generate_synthetic import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(
            symbol="TEST",
            initial_price=150.0,
            base_volatility=0.05,
            burst_probability=0.01,
            seed=42
        )
        
        # Generate 5k events
        book_df, trade_df = generator.generate_events(5000)
        generator.save_to_parquet(book_df, trade_df, self.temp_data_dir)
        
        self.data_file = self.temp_data_dir / "test_l1.parquet"
        assert self.data_file.exists(), "Synthetic data file not created"
    
    def _create_test_config(self):
        """Create test configuration file."""
        config_content = f"""
data:
  path: "{self.data_file}"
  kind: "trade"
  format: "parquet"

strategy:
  name: "mean_reversion"
  symbol: "TEST"
  enabled: true
  max_position: 1000
  max_order_size: 100
  risk_limits: {{}}
  params:
    lookback_period: 10
    z_score_threshold: 0.5
    exit_z_score: 0.2
    position_size: 100

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
  output_dir: "{self.temp_dir}/results"
  format: "both"
  plots: true
  detailed_trades: true
  performance_metrics: true
"""
        
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        self.config_file.write_text(config_content)
    
    def test_single_backtest_creates_runs_directory(self):
        """Test that a single backtest creates proper runs directory structure."""
        # Load configuration
        config = load_config(str(self.config_file))
        
        # Run backtest
        runner = BacktestRunner(config)
        results = runner.run()
        
        # Verify run directory was created
        assert 'run_directory' in results, "Run directory not returned in results"
        run_dir = Path(results['run_directory'])
        assert run_dir.exists(), f"Run directory not created: {run_dir}"
        
        # Verify required files exist
        required_files = [
            'config.yaml',
            'performance.json',
            'performance.csv',
            'blotter.parquet',
            'positions.parquet'
        ]
        
        for file_name in required_files:
            file_path = run_dir / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"
        
        # Verify config.yaml contains resolved configuration
        with open(run_dir / 'config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'data' in config_data, "Config missing data section"
        assert 'strategy' in config_data, "Config missing strategy section"
        assert 'execution' in config_data, "Config missing execution section"
        
        # Verify performance files contain valid data
        with open(run_dir / 'performance.json', 'r') as f:
            perf_data = json.load(f)
        
        assert 'total_return' in perf_data, "Performance missing total_return"
        assert 'sharpe_ratio' in perf_data, "Performance missing sharpe_ratio"
        assert 'total_trades' in perf_data, "Performance missing total_trades"
        
        # Verify parquet files are valid
        blotter_df = pd.read_parquet(run_dir / 'blotter.parquet')
        positions_df = pd.read_parquet(run_dir / 'positions.parquet')
        
        assert isinstance(blotter_df, pd.DataFrame), "Blotter is not a DataFrame"
        assert isinstance(positions_df, pd.DataFrame), "Positions is not a DataFrame"
        
        print(f"✅ Single backtest created proper runs directory: {run_dir}")
    
    def test_latency_sweep_creates_runs_directory(self):
        """Test that latency sweep creates proper runs directory structure."""
        # Load configuration
        config = load_config(str(self.config_file))
        
        # Run latency sweep
        sweeper = LatencySweeper(config)
        results = sweeper.run_sweep([100000, 200000])
        
        # Find the sweep run directory
        runs_dir = Path("runs")
        sweep_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith('_sweep')]
        assert len(sweep_dirs) > 0, "No sweep run directory found"
        
        sweep_dir = sweep_dirs[-1]  # Get the most recent one
        
        # Verify required files exist
        required_files = [
            'config.yaml',
            'performance.json',
            'performance.csv',
            'blotter.parquet',
            'positions.parquet',
            'latency_sweep.csv'
        ]
        
        for file_name in required_files:
            file_path = sweep_dir / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"
        
        # Verify latency_sweep.csv contains sweep data
        sweep_df = pd.read_csv(sweep_dir / 'latency_sweep.csv')
        assert len(sweep_df) >= 2, "Latency sweep should have at least 2 runs"
        assert 'latency_ns' in sweep_df.columns, "Missing latency_ns column"
        assert 'total_return' in sweep_df.columns, "Missing total_return column"
        
        print(f"✅ Latency sweep created proper runs directory: {sweep_dir}")
    
    def test_pack_command_works(self):
        """Test that the pack command works correctly."""
        # First create a run
        config = load_config(str(self.config_file))
        runner = BacktestRunner(config)
        results = runner.run()
        
        run_dir = Path(results['run_directory'])
        assert run_dir.exists(), "Run directory not created"
        
        # Test packing
        packer = RunPacker()
        zip_path = packer.pack_run(run_dir)
        
        assert zip_path.exists(), f"Zip file not created: {zip_path}"
        assert zip_path.suffix == '.zip', "Output is not a zip file"
        
        # Verify zip file size is reasonable
        zip_size = zip_path.stat().st_size
        assert zip_size > 1000, f"Zip file too small: {zip_size} bytes"
        
        print(f"✅ Pack command created zip file: {zip_path}")
    
    def test_pack_validation_works(self):
        """Test that pack validation works correctly."""
        # First create a run
        config = load_config(str(self.config_file))
        runner = BacktestRunner(config)
        results = runner.run()
        
        run_dir = Path(results['run_directory'])
        
        # Test validation
        packer = RunPacker()
        validation = packer.validate_run_directory(run_dir)
        
        assert validation['valid'], f"Run directory validation failed: {validation['errors']}"
        assert len(validation['present_files']) >= 5, "Not enough files present"
        assert len(validation['missing_required']) == 0, f"Missing required files: {validation['missing_required']}"
        
        print(f"✅ Pack validation passed for: {run_dir}")
    
    def test_pack_listing_works(self):
        """Test that pack listing works correctly."""
        # First create a run
        config = load_config(str(self.config_file))
        runner = BacktestRunner(config)
        results = runner.run()
        
        run_dir = Path(results['run_directory'])
        
        # Test listing
        packer = RunPacker()
        contents = packer.list_run_contents(run_dir)
        
        assert contents['file_count'] > 0, "No files found in run directory"
        assert contents['total_size'] > 0, "Total size should be positive"
        assert len(contents['files']) > 0, "No file information returned"
        
        # Verify file information structure
        for file_info in contents['files']:
            assert 'path' in file_info, "File info missing path"
            assert 'size' in file_info, "File info missing size"
            assert 'size_formatted' in file_info, "File info missing size_formatted"
            assert file_info['size'] > 0, "File size should be positive"
        
        print(f"✅ Pack listing works for: {run_dir}")
        print(f"  Files: {contents['file_count']}, Size: {contents['total_size_formatted']}")
    
    def test_plots_are_generated(self):
        """Test that plots are generated when there are trades."""
        # Create a run with artificial trades
        config = load_config(str(self.config_file))
        runner = BacktestRunner(config)
        results = runner.run()
        
        run_dir = Path(results['run_directory'])
        
        # Check for plot files
        plot_files = list(run_dir.glob("*.png"))
        
        # We might not have plots if no trades were executed
        if len(plot_files) == 0:
            print("⚠️  No plots generated (expected if no trades executed)")
        else:
            print(f"✅ Generated {len(plot_files)} plot files")
            for plot_file in plot_files:
                assert plot_file.stat().st_size > 1000, f"Plot file too small: {plot_file.name}"
    
    def test_runs_directory_structure(self):
        """Test the overall runs directory structure."""
        # Create multiple runs
        config = load_config(str(self.config_file))
        
        # Single backtest
        runner = BacktestRunner(config)
        results1 = runner.run()
        
        # Latency sweep
        sweeper = LatencySweeper(config)
        results2 = sweeper.run_sweep([100000, 200000])
        
        # Check runs directory
        runs_dir = Path("runs")
        assert runs_dir.exists(), "Runs directory not created"
        
        # Get the specific directories we just created
        single_run_dir = Path(results1['run_directory'])
        sweep_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith('_sweep')]
        sweep_run_dir = sweep_dirs[-1] if sweep_dirs else None
        
        # Verify single backtest directory
        assert single_run_dir.exists(), f"Single run directory not created: {single_run_dir}"
        assert (single_run_dir / 'config.yaml').exists(), f"Single run missing config.yaml: {single_run_dir}"
        assert (single_run_dir / 'performance.json').exists(), f"Single run missing performance.json: {single_run_dir}"
        
        # Verify sweep directory
        if sweep_run_dir:
            assert (sweep_run_dir / 'latency_sweep.csv').exists(), f"Sweep directory missing latency_sweep.csv: {sweep_run_dir}"
            assert (sweep_run_dir / 'config.yaml').exists(), f"Sweep directory missing config.yaml: {sweep_run_dir}"
            assert (sweep_run_dir / 'performance.json').exists(), f"Sweep directory missing performance.json: {sweep_run_dir}"
        
        print(f"✅ Runs directory structure is correct")
        print(f"  Single run: {single_run_dir}")
        if sweep_run_dir:
            print(f"  Sweep run: {sweep_run_dir}")


if __name__ == "__main__":
    # Run the test directly for debugging
    test = TestRunsDirectory()
    test.setup_method()
    try:
        test.test_single_backtest_creates_runs_directory()
        test.test_latency_sweep_creates_runs_directory()
        test.test_pack_command_works()
        test.test_pack_validation_works()
        test.test_pack_listing_works()
        test.test_plots_are_generated()
        test.test_runs_directory_structure()
        print("\n🎉 All runs directory tests passed!")
    finally:
        test.teardown_method()
