# Flashback: High-Frequency Trading Strategy Backtester

A professional-grade backtesting engine built for high-frequency trading strategies with microsecond precision and realistic market simulation.

## Performance

- **C++ Core**: Optional C++ matching engine via pybind11 (8-12× speedup)
- **Throughput**: 10M+ events/sec with sub-microsecond latency
- **Memory Efficient**: Optimized data structures and minimal allocations

## Quick Start

### Prerequisites
- Python 3.11+
- CMake (for C++ extension)
- pybind11 (for C++ extension)

### Installation
```bash
# Clone and install
git clone <repository-url>
cd High-Frequency-Trading-Strategy-Backtester
make setup

# Optional: Build C++ extension for maximum performance
make cpp
```

### Basic Usage
```bash
# Generate synthetic market data
python examples/generate_synthetic.py --seed 7 --events 200000

# Run single backtest
flashback run --config config/backtest.yaml

# Run latency sensitivity analysis
flashback sweep --config config/backtest.yaml --latency 100000,250000,500000

# Package results
flashback pack --run runs/2025-10-06T00-00-00
```

## Architecture

The engine follows an event-driven architecture with these core components:

- **Event Loop**: Processes market data, orders, and fills in chronological order
- **Strategy Engine**: Executes trading strategies based on market events
- **Matching Engine**: Price-time priority order matching with partial fills
- **Risk Manager**: Position limits, PnL tracking, and risk controls
- **Performance Analyzer**: Comprehensive metrics and reporting

## Strategy Development

Strategies implement the `Strategy` protocol:

```python
from flashback.strategy.base import Strategy

class MyStrategy(Strategy):
    def on_bar(self, book_update):
        # Process market data
        pass
    
    def on_trade(self, trade):
        # Handle trade execution
        pass
    
    def on_timer(self, ts):
        # Periodic processing
        pass
```

## Configuration

Configuration is handled via YAML files:

```yaml
data:
  path: "examples/sample_data/aapl_l1.parquet"
  format: "parquet"

strategy:
  name: "momentum_imbalance"
  params:
    short_ema_period: 10
    long_ema_period: 50
    imbalance_threshold: 0.3

execution:
  latency_model: "random"
  fee_model: "basis_points"
  slippage_model: "fixed"

risk:
  max_position: 1000
  max_gross_exposure: 100000
  daily_loss_limit: 5000
```

## Performance Metrics

The engine calculates comprehensive performance metrics:

- **Returns**: Total return, annualized return, volatility
- **Risk**: Sharpe ratio, maximum drawdown, VaR
- **Trading**: Hit rate, average win/loss, turnover
- **Latency**: Mean, median, 95th percentile latency

## Data Formats

Supports multiple data formats:

- **Parquet**: Recommended for large datasets
- **CSV**: For smaller datasets or external data
- **L1 Data**: Bid/ask prices and sizes
- **L2 Data**: Full order book snapshots

## Examples

See the `examples/` directory for:

- `generate_synthetic.py`: Generate synthetic market data
- `notebooks/01_quickstart.ipynb`: Interactive tutorial
- `sample_data/`: Sample datasets for testing

## Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Run with coverage
make test-coverage
```

## Development

### Code Quality
```bash
# Lint code
make lint

# Type checking
make typecheck

# Format code
make format
```

### C++ Extension
```bash
# Build C++ extension
make cpp

# Test C++ extension
make cpp-test

# Clean build artifacts
make cpp-clean
```

## Benchmarking

```bash
# Run performance benchmarks
make bench

# Run CI benchmarks
make ci-bench
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.