# Sample Data

This directory contains sample data generation scripts and example data files for testing the flashback engine.

## Files

- `generate_sample_data.py` - Script to generate sample market data
- `README.md` - This file

## Generating Sample Data

### Basic Usage

```bash
# Generate tick data for AAPL
python examples/sample_data/generate_sample_data.py --symbol AAPL --data-type tick

# Generate L2 order book data
python examples/sample_data/generate_sample_data.py --symbol AAPL --data-type l2

# Generate imbalance data
python examples/sample_data/generate_sample_data.py --symbol AAPL --data-type imbalance
```

### Advanced Usage

```bash
# Generate data for multiple symbols
python examples/sample_data/generate_sample_data.py --symbol AAPL --symbol MSFT --data-type tick

# Specify date range
python examples/sample_data/generate_sample_data.py --start-date 2024-01-01 --end-date 2024-12-31 --data-type tick

# Generate more records
python examples/sample_data/generate_sample_data.py --num-records 50000 --data-type tick

# Specify output directory
python examples/sample_data/generate_sample_data.py --output-dir data/ --data-type tick
```

### Command Line Options

- `--symbol`: Symbol to generate data for (default: AAPL)
- `--start-date`: Start date in YYYY-MM-DD format (default: 2024-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2024-01-31)
- `--data-type`: Type of data to generate (tick, l2, imbalance) (default: tick)
- `--num-records`: Number of records to generate (default: 10000)
- `--output-dir`: Output directory (default: examples/sample_data)

## Data Formats

### Tick Data

Contains individual bid/ask ticks with the following columns:
- `ts`: Timestamp
- `symbol`: Trading symbol
- `side`: 'B' for bid, 'S' for ask
- `price`: Price level
- `size`: Order size
- `event_type`: 'TICK'

### L2 Data

Contains Level 2 order book snapshots with the following columns:
- `ts`: Timestamp
- `symbol`: Trading symbol
- `side`: 'B' for bid, 'S' for ask
- `price`: Price level
- `size`: Order size
- `event_type`: 'L2_BID' or 'L2_ASK'

### Imbalance Data

Contains order book imbalance data with the following columns:
- `ts`: Timestamp
- `symbol`: Trading symbol
- `side`: 'I' for imbalance
- `price`: Imbalance value (-1 to 1)
- `size`: 0
- `event_type`: 'IMBALANCE'

## Using Generated Data

Once you've generated sample data, you can use it in your backtests:

```python
from flashback.core.engine import BacktestEngine

config = {
    "data": {
        "source": "examples/sample_data/aapl_tick_data.parquet",
        "symbols": ["AAPL"]
    },
    # ... rest of config
}

engine = BacktestEngine(config)
results = engine.run()
```

## Data Characteristics

The generated data includes:

- **Realistic Price Movement**: Sine wave with random noise
- **Variable Spreads**: Random spreads between 0.01 and 0.015
- **Random Sizes**: Order sizes between 100 and 1000
- **Microsecond Timestamps**: High-frequency timing
- **Multiple Symbols**: Support for generating data for different symbols

## Customization

You can modify the `generate_sample_data.py` script to:

- Change price movement patterns
- Adjust spread distributions
- Modify order size ranges
- Add more realistic market behavior
- Include additional data fields
