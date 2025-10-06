"""Basic Backtest Example

This script demonstrates how to run a basic backtest using the flashback engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from flashback.core.engine import BacktestEngine
from flashback.data.loader import DataLoader
from flashback.strategy.mean_reversion import MeanReversionStrategy
from flashback.strategy.momentum import MomentumStrategy


def generate_sample_data():
    """Generate sample market data."""
    timestamps = pd.date_range('2024-01-01', periods=10000, freq='1ms')
    data = []
    
    for i, ts in enumerate(timestamps):
        base_price = 150.0 + np.sin(i * 0.01) * 2.0 + np.random.normal(0, 0.1)
        spread = 0.01 + np.random.uniform(0, 0.005)
        
        data.append({
            'ts': ts,
            'symbol': 'AAPL',
            'side': 'B',
            'price': base_price - spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
        
        data.append({
            'ts': ts + pd.Timedelta(microseconds=100),
            'symbol': 'AAPL',
            'side': 'S',
            'price': base_price + spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
    
    df = pd.DataFrame(data)
    df.to_parquet('sample_data.parquet')
    print(f"Generated {len(df)} market data records")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
    
    return df


def create_config():
    """Create backtest configuration."""
    config = {
        "data": {
            "source": "sample_data.parquet",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "symbols": ["AAPL"]
        },
        "strategy": {
            "name": "mean_reversion",
            "params": {
                "lookback": 100,
                "threshold": 0.001,
                "position_size": 100
            }
        },
        "risk": {
            "max_position": 1000,
            "max_drawdown": 0.05,
            "max_trade_size": 100
        },
        "execution": {
            "fees": {
                "commission_per_share": 0.001,
                "slippage_bps": 1.0
            },
            "latency": {
                "base_latency_us": 100,
                "jitter_us": 10
            }
        },
        "output": {
            "directory": "output",
            "trade_blotter": "trades.csv",
            "performance": "performance.json"
        }
    }
    
    return config


def run_backtest(config):
    """Run the backtest."""
    engine = BacktestEngine(config)
    results = engine.run()
    
    print("Backtest completed!")
    print(f"Events processed: {results['statistics']['events_processed']:,}")
    print(f"Orders placed: {results['statistics']['orders_placed']:,}")
    print(f"Fills generated: {results['statistics']['fills_generated']:,}")
    
    return engine, results


def analyze_results(engine, results):
    """Analyze backtest results."""
    # Get performance metrics
    metrics = engine.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get trade blotter
    trades = results['trades']
    if not trades.empty:
        print(f"\nTrade Blotter ({len(trades)} trades):")
        print(trades.head(10))
        
        print(f"\nTrade Statistics:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Buy trades: {len(trades[trades['side'] == 'B'])}")
        print(f"  Sell trades: {len(trades[trades['side'] == 'S'])}")
        print(f"  Average trade size: {trades['quantity'].mean():.2f}")
        print(f"  Average price: {trades['price'].mean():.2f}")
        print(f"  Total volume: {trades['quantity'].sum():,}")
        print(f"  Total commission: ${trades['commission'].sum():.2f}")
    else:
        print("No trades executed")
    
    # Get PnL data
    pnl = results['pnl']
    if not pnl.empty:
        print(f"\nPnL Summary:")
        print(f"  Total PnL: ${pnl['total_pnl'].iloc[-1]:.2f}")
        print(f"  Realized PnL: ${pnl['realized_pnl'].iloc[-1]:.2f}")
        print(f"  Unrealized PnL: ${pnl['unrealized_pnl'].iloc[-1]:.2f}")
        print(f"  Max drawdown: {pnl['max_drawdown'].iloc[-1]:.2%}")
        print(f"  Current drawdown: {pnl['current_drawdown'].iloc[-1]:.2%}")
    else:
        print("No PnL data available")
    
    return trades, pnl


def create_visualizations(trades, pnl):
    """Create visualizations."""
    if pnl.empty:
        print("No PnL data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total PnL
    axes[0, 0].plot(pnl.index, pnl['total_pnl'])
    axes[0, 0].set_title('Total PnL Over Time')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].grid(True)
    
    # Realized vs Unrealized PnL
    axes[0, 1].plot(pnl.index, pnl['realized_pnl'], label='Realized')
    axes[0, 1].plot(pnl.index, pnl['unrealized_pnl'], label='Unrealized')
    axes[0, 1].set_title('Realized vs Unrealized PnL')
    axes[0, 1].set_ylabel('PnL ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Drawdown
    axes[1, 0].plot(pnl.index, pnl['current_drawdown'] * 100)
    axes[1, 0].set_title('Current Drawdown')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].grid(True)
    
    # Trade distribution
    if not trades.empty:
        trade_prices = trades['price']
        axes[1, 1].hist(trade_prices, bins=20, alpha=0.7)
        axes[1, 1].set_title('Trade Price Distribution')
        axes[1, 1].set_xlabel('Price')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def save_results(results, trades, pnl):
    """Save results to files."""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    if not trades.empty:
        trades.to_csv(output_dir / 'trades.csv')
        print(f"Trades saved to {output_dir / 'trades.csv'}")
    
    if not pnl.empty:
        pnl.to_csv(output_dir / 'pnl.csv')
        print(f"PnL data saved to {output_dir / 'pnl.csv'}")
    
    print("All results saved successfully!")


def main():
    """Main function."""
    print("Flashback Basic Backtest Example")
    print("=" * 40)
    
    # Generate sample data
    print("1. Generating sample data...")
    generate_sample_data()
    
    # Create configuration
    print("2. Creating configuration...")
    config = create_config()
    
    # Run backtest
    print("3. Running backtest...")
    engine, results = run_backtest(config)
    
    # Analyze results
    print("4. Analyzing results...")
    trades, pnl = analyze_results(engine, results)
    
    # Create visualizations
    print("5. Creating visualizations...")
    create_visualizations(trades, pnl)
    
    # Save results
    print("6. Saving results...")
    save_results(results, trades, pnl)
    
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main()
