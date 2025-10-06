"""Strategy Comparison Example

This script demonstrates how to compare different trading strategies using the flashback engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from flashback.core.engine import BacktestEngine
from flashback.strategy.mean_reversion import MeanReversionStrategy
from flashback.strategy.momentum import MomentumStrategy


def generate_sample_data():
    """Generate sample market data."""
    timestamps = pd.date_range('2024-01-01', periods=5000, freq='1ms')
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
    print(f"Generated {len(df)} records")
    return df


def define_strategies():
    """Define strategy configurations."""
    base_config = {
        "data": {
            "source": "sample_data.parquet",
            "symbols": ["AAPL"]
        },
        "risk": {
            "max_position": 1000,
            "max_drawdown": 0.05
        },
        "execution": {
            "fees": {"commission_per_share": 0.001}
        }
    }
    
    strategies = {
        "Mean Reversion": {
            **base_config,
            "strategy": {
                "name": "mean_reversion",
                "params": {"lookback": 100, "threshold": 0.001, "position_size": 100}
            }
        },
        "Momentum": {
            **base_config,
            "strategy": {
                "name": "momentum",
                "params": {"short_ma_period": 10, "long_ma_period": 50, "imbalance_threshold": 0.3, "position_size": 100}
            }
        }
    }
    
    return strategies


def run_backtests(strategies):
    """Run backtests for all strategies."""
    results = {}
    
    for name, config in strategies.items():
        print(f"Running {name} strategy...")
        engine = BacktestEngine(config)
        results[name] = engine.run()
        print(f"  Events: {results[name]['statistics']['events_processed']:,}")
        print(f"  Orders: {results[name]['statistics']['orders_placed']:,}")
        print(f"  Fills: {results[name]['statistics']['fills_generated']:,}")
        print()
    
    return results


def compare_results(results):
    """Compare strategy results."""
    comparison_data = []
    
    for name, result in results.items():
        stats = result['statistics']
        pnl = result['pnl']
        trades = result['trades']
        
        comparison_data.append({
            'Strategy': name,
            'Events': stats['events_processed'],
            'Orders': stats['orders_placed'],
            'Fills': stats['fills_generated'],
            'Total PnL': pnl['total_pnl'].iloc[-1] if not pnl.empty else 0,
            'Max Drawdown': pnl['max_drawdown'].iloc[-1] if not pnl.empty else 0,
            'Total Trades': len(trades) if not trades.empty else 0,
            'Total Volume': trades['quantity'].sum() if not trades.empty else 0,
            'Total Commission': trades['commission'].sum() if not trades.empty else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Strategy Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    return comparison_df


def create_visualizations(results):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (name, result) in enumerate(results.items()):
        pnl = result['pnl']
        if not pnl.empty:
            axes[0, 0].plot(pnl.index, pnl['total_pnl'], label=name, linewidth=2)
            axes[0, 1].plot(pnl.index, pnl['current_drawdown'] * 100, label=name, linewidth=2)
    
    axes[0, 0].set_title('Total PnL Comparison')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Drawdown Comparison')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Trade analysis
    for i, (name, result) in enumerate(results.items()):
        trades = result['trades']
        if not trades.empty:
            axes[1, 0].hist(trades['price'], alpha=0.7, label=name, bins=20)
            axes[1, 1].hist(trades['quantity'], alpha=0.7, label=name, bins=20)
    
    axes[1, 0].set_title('Trade Price Distribution')
    axes[1, 0].set_xlabel('Price')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Trade Size Distribution')
    axes[1, 1].set_xlabel('Quantity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    print("Flashback Strategy Comparison Example")
    print("=" * 40)
    
    # Generate sample data
    print("1. Generating sample data...")
    generate_sample_data()
    
    # Define strategies
    print("2. Defining strategies...")
    strategies = define_strategies()
    
    # Run backtests
    print("3. Running backtests...")
    results = run_backtests(strategies)
    
    # Compare results
    print("4. Comparing results...")
    comparison_df = compare_results(results)
    
    # Create visualizations
    print("5. Creating visualizations...")
    create_visualizations(results)
    
    print("Strategy comparison completed!")


if __name__ == "__main__":
    main()
