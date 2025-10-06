# Flashback Examples

This directory contains examples and demonstrations of the Flashback HFT backtesting engine.

## Files

### Synthetic Data Generation
- `generate_synthetic.py` - Generate realistic synthetic market data with microprice processes, bursts, and order flow imbalance
- `sample_data/` - Directory containing sample data files in Parquet and CSV formats

### Notebooks
- `notebooks/01_quickstart.ipynb` - Complete quickstart tutorial showing the full workflow

### Configuration
- `quickstart_config.yaml` - Sample configuration file for quickstart examples

## Quick Start

1. **Generate Synthetic Data**:
   ```bash
   python examples/generate_synthetic.py --events 1000000 --output-dir examples/sample_data
   ```

2. **Run a Backtest**:
   ```bash
   python -m flashback run --config examples/quickstart_config.yaml
   ```

3. **Run Latency Sensitivity Analysis**:
   ```bash
   python -m flashback sweep --config examples/quickstart_config.yaml --latency 50000,100000,200000,500000
   ```

4. **Open the Quickstart Notebook**:
   ```bash
   jupyter notebook examples/notebooks/01_quickstart.ipynb
   ```

## Synthetic Data Generator

The `generate_synthetic.py` script creates realistic market data with:

- **Microprice Process**: Mean-reverting price evolution with configurable volatility
- **Burst Events**: Volatility spikes that simulate market stress periods
- **Order Flow Imbalance**: Realistic bid/ask size dynamics based on order flow
- **Configurable Parameters**: Symbol, initial price, volatility, burst probability, etc.

### Usage

```bash
python examples/generate_synthetic.py [options]

Options:
  --symbol SYMBOL           Trading symbol (default: AAPL)
  --events EVENTS           Number of events to generate (default: 1000000)
  --output-dir OUTPUT_DIR   Output directory (default: examples/sample_data)
  --initial-price PRICE     Initial price (default: 150.0)
  --volatility VOL          Base volatility (default: 0.02)
  --burst-prob PROB         Burst probability (default: 0.001)
  --seed SEED               Random seed (default: 42)
```

### Output Files

The generator creates several output files:

- `{symbol}_book.parquet` - Order book update events
- `{symbol}_trades.parquet` - Trade events
- `{symbol}_l1.parquet` - Combined L1 data (compatible with backtest engine)
- `{symbol}_l1.csv` - CSV version of L1 data

## Sample Data

The `sample_data/` directory contains:

- Pre-generated synthetic data files
- Existing sample data from the original implementation
- Test fixtures for unit tests

## Notebooks

### 01_quickstart.ipynb

A comprehensive tutorial notebook that demonstrates:

1. **Setup and Imports** - Environment configuration
2. **Synthetic Data Generation** - Creating realistic market data
3. **Configuration** - Setting up backtest parameters
4. **Backtest Execution** - Running the backtest engine
5. **Results Analysis** - Examining performance metrics
6. **Latency Sensitivity** - Analyzing performance vs latency
7. **Visualization** - Creating charts and plots

The notebook provides a complete end-to-end example of using the Flashback engine for HFT strategy development and testing.

## Configuration Examples

### Basic Configuration

```yaml
data:
  path: "examples/sample_data/aapl_l1.parquet"
  kind: "trade"
  format: "parquet"

strategy:
  name: "momentum_imbalance"
  symbol: "AAPL"
  enabled: true
  max_position: 1000
  max_order_size: 100
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
  output_dir: "runs/example"
  format: "both"
  plots: true
  detailed_trades: true
  performance_metrics: true
```

## Next Steps

1. **Experiment with Parameters**: Modify strategy parameters, risk limits, and execution settings
2. **Create Custom Strategies**: Implement new strategies by extending the base strategy class
3. **Analyze Performance**: Use the generated metrics and visualizations to understand strategy behavior
4. **Optimize for Latency**: Use latency sensitivity analysis to find optimal execution parameters
5. **Scale Up**: Run longer backtests with more historical data

For more detailed documentation, see the main project README and the individual module docstrings.
