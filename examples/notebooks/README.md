# Flashback Examples

This directory contains example notebooks and scripts demonstrating how to use the flashback HFT backtesting engine.

## Files

- `basic_backtest.py` - Basic backtest example showing how to run a simple backtest
- `strategy_comparison.py` - Example comparing different trading strategies
- `basic_backtest.ipynb` - Jupyter notebook version of the basic backtest example
- `strategy_comparison.ipynb` - Jupyter notebook version of the strategy comparison example

## Running the Examples

### Python Scripts

```bash
# Run basic backtest example
python examples/notebooks/basic_backtest.py

# Run strategy comparison example
python examples/notebooks/strategy_comparison.py
```

### Jupyter Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Open the desired notebook in your browser
```

## Example Outputs

The examples will generate:

1. **Sample Data**: Market data in Parquet format
2. **Backtest Results**: Trade blotter, PnL data, and performance metrics
3. **Visualizations**: Charts showing PnL, drawdown, and trade distributions
4. **Output Files**: CSV files with results saved to the `output/` directory

## Customization

You can modify the examples to:

- Use your own market data
- Test different strategies
- Adjust risk parameters
- Change execution settings
- Add custom visualizations

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install -e .
```

The examples require:
- pandas
- numpy
- matplotlib
- seaborn
- pyarrow
- click
- pyyaml
