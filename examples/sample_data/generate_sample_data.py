"""Generate sample data for testing and examples."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_tick_data(symbol: str, start_date: str, end_date: str, 
                      freq: str = '1ms', num_ticks: int = 10000) -> pd.DataFrame:
    """Generate sample tick data."""
    timestamps = pd.date_range(start_date, end_date, freq=freq)
    
    # Limit to requested number of ticks
    if len(timestamps) > num_ticks:
        timestamps = timestamps[:num_ticks]
    
    data = []
    
    for i, ts in enumerate(timestamps):
        # Generate realistic price movement
        base_price = 150.0 + np.sin(i * 0.01) * 2.0 + np.random.normal(0, 0.1)
        spread = 0.01 + np.random.uniform(0, 0.005)
        
        # Bid
        data.append({
            'ts': ts,
            'symbol': symbol,
            'side': 'B',
            'price': base_price - spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
        
        # Ask
        data.append({
            'ts': ts + pd.Timedelta(microseconds=100),
            'symbol': symbol,
            'side': 'S',
            'price': base_price + spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
    
    return pd.DataFrame(data)


def generate_l2_data(symbol: str, start_date: str, end_date: str, 
                    freq: str = '100ms', num_snapshots: int = 1000) -> pd.DataFrame:
    """Generate sample L2 order book data."""
    timestamps = pd.date_range(start_date, end_date, freq=freq)
    
    # Limit to requested number of snapshots
    if len(timestamps) > num_snapshots:
        timestamps = timestamps[:num_snapshots]
    
    data = []
    
    for i, ts in enumerate(timestamps):
        # Generate realistic price movement
        base_price = 150.0 + np.sin(i * 0.01) * 2.0 + np.random.normal(0, 0.1)
        spread = 0.01 + np.random.uniform(0, 0.005)
        
        # Generate bid levels
        for level in range(5):
            price = base_price - spread/2 - level * 0.01
            size = np.random.randint(100, 1000) * (5 - level)
            
            data.append({
                'ts': ts,
                'symbol': symbol,
                'side': 'B',
                'price': price,
                'size': size,
                'event_type': 'L2_BID'
            })
        
        # Generate ask levels
        for level in range(5):
            price = base_price + spread/2 + level * 0.01
            size = np.random.randint(100, 1000) * (5 - level)
            
            data.append({
                'ts': ts,
                'symbol': symbol,
                'side': 'S',
                'price': price,
                'size': size,
                'event_type': 'L2_ASK'
            })
    
    return pd.DataFrame(data)


def generate_imbalance_data(symbol: str, start_date: str, end_date: str, 
                           freq: str = '1s', num_periods: int = 1000) -> pd.DataFrame:
    """Generate sample imbalance data."""
    timestamps = pd.date_range(start_date, end_date, freq=freq)
    
    # Limit to requested number of periods
    if len(timestamps) > num_periods:
        timestamps = timestamps[:num_periods]
    
    data = []
    
    for i, ts in enumerate(timestamps):
        # Generate imbalance
        imbalance = np.random.normal(0, 0.3)
        
        data.append({
            'ts': ts,
            'symbol': symbol,
            'side': 'I',
            'price': imbalance,
            'size': 0,
            'event_type': 'IMBALANCE'
        })
    
    return pd.DataFrame(data)


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description='Generate sample data for flashback')
    parser.add_argument('--symbol', default='AAPL', help='Symbol to generate data for')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', default='2024-01-31', help='End date')
    parser.add_argument('--output-dir', default='examples/sample_data', help='Output directory')
    parser.add_argument('--data-type', choices=['tick', 'l2', 'imbalance'], 
                       default='tick', help='Type of data to generate')
    parser.add_argument('--num-records', type=int, default=10000, 
                       help='Number of records to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data based on type
    if args.data_type == 'tick':
        data = generate_tick_data(args.symbol, args.start_date, args.end_date, 
                                 num_ticks=args.num_records)
        output_file = output_dir / f'{args.symbol.lower()}_tick_data.parquet'
    elif args.data_type == 'l2':
        data = generate_l2_data(args.symbol, args.start_date, args.end_date, 
                               num_snapshots=args.num_records)
        output_file = output_dir / f'{args.symbol.lower()}_l2_data.parquet'
    elif args.data_type == 'imbalance':
        data = generate_imbalance_data(args.symbol, args.start_date, args.end_date, 
                                      num_periods=args.num_records)
        output_file = output_dir / f'{args.symbol.lower()}_imbalance_data.parquet'
    
    # Save data
    data.to_parquet(output_file)
    print(f"Generated {len(data)} records and saved to {output_file}")
    
    # Display sample
    print("\nSample data:")
    print(data.head(10))
    
    # Display statistics
    print(f"\nData statistics:")
    print(f"Total records: {len(data)}")
    print(f"Date range: {data['ts'].min()} to {data['ts'].max()}")
    print(f"Price range: {data['price'].min():.2f} to {data['price'].max():.2f}")
    print(f"Size range: {data['size'].min()} to {data['size'].max()}")


if __name__ == '__main__':
    main()
