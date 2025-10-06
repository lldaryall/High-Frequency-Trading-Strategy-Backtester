"""Generate test fixtures for data validation."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_trade_fixture():
    """Generate valid trade data fixture."""
    # Generate timestamps in nanoseconds (strictly increasing)
    base_time = 1704067200000000000  # 2024-01-01 00:00:00 in nanoseconds
    timestamps = [base_time + i * 1000000 for i in range(100)]  # 1ms intervals
    
    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'ts': ts,
            'symbol': 'AAPL',
            'price': 150.0 + np.sin(i * 0.1) * 2.0,
            'size': np.random.randint(100, 1000),
            'aggressor_side': 'B' if i % 2 == 0 else 'S'
        })
    
    df = pd.DataFrame(data)
    df.to_parquet('trade_fixture.parquet')
    print(f"Generated trade fixture with {len(df)} records")
    return df


def generate_book_update_fixture():
    """Generate valid book update data fixture."""
    # Generate timestamps in nanoseconds (strictly increasing)
    base_time = 1704067200000000000  # 2024-01-01 00:00:00 in nanoseconds
    timestamps = [base_time + i * 1000000 for i in range(100)]  # 1ms intervals
    
    data = []
    for i, ts in enumerate(timestamps):
        base_price = 150.0 + np.sin(i * 0.1) * 2.0
        spread = 0.01
        
        data.append({
            'ts': ts,
            'symbol': 'AAPL',
            'bid_px': base_price - spread/2,
            'bid_sz': np.random.randint(100, 1000),
            'ask_px': base_price + spread/2,
            'ask_sz': np.random.randint(100, 1000),
            'level': 1
        })
    
    df = pd.DataFrame(data)
    df.to_parquet('book_update_fixture.parquet')
    print(f"Generated book update fixture with {len(df)} records")
    return df


def generate_bad_data_fixture():
    """Generate bad data fixture with various validation errors."""
    data = []
    
    # Valid row
    data.append({
        'ts': 1704067200000000000,
        'symbol': 'AAPL',
        'price': 150.0,
        'size': 100,
        'aggressor_side': 'B'
    })
    
    # Non-monotonic timestamp
    data.append({
        'ts': 1704067200000000000 - 1000000,  # Earlier timestamp
        'symbol': 'AAPL',
        'price': 151.0,
        'size': 200,
        'aggressor_side': 'S'
    })
    
    # Invalid price (negative)
    data.append({
        'ts': 1704067200000000001,
        'symbol': 'AAPL',
        'price': -150.0,
        'size': 100,
        'aggressor_side': 'B'
    })
    
    # Invalid size (zero)
    data.append({
        'ts': 1704067200000000002,
        'symbol': 'AAPL',
        'price': 152.0,
        'size': 0,
        'aggressor_side': 'S'
    })
    
    # Invalid aggressor_side
    data.append({
        'ts': 1704067200000000003,
        'symbol': 'AAPL',
        'price': 153.0,
        'size': 100,
        'aggressor_side': 'X'  # Invalid side
    })
    
    # Invalid timestamp (too early)
    data.append({
        'ts': 946684800000000000 - 1,  # Before 2000
        'symbol': 'AAPL',
        'price': 154.0,
        'size': 100,
        'aggressor_side': 'B'
    })
    
    # NaN price
    data.append({
        'ts': 1704067200000000004,
        'symbol': 'AAPL',
        'price': np.nan,
        'size': 100,
        'aggressor_side': 'S'
    })
    
    df = pd.DataFrame(data)
    df.to_parquet('bad_data_fixture.parquet')
    print(f"Generated bad data fixture with {len(df)} records")
    return df


def generate_book_update_bad_fixture():
    """Generate bad book update data fixture."""
    data = []
    
    # Valid row
    data.append({
        'ts': 1704067200000000000,
        'symbol': 'AAPL',
        'bid_px': 149.99,
        'bid_sz': 100,
        'ask_px': 150.01,
        'ask_sz': 100,
        'level': 1
    })
    
    # Invalid level (not 1)
    data.append({
        'ts': 1704067200000000001,
        'symbol': 'AAPL',
        'bid_px': 149.98,
        'bid_sz': 200,
        'ask_px': 150.02,
        'ask_sz': 200,
        'level': 2
    })
    
    # Bid price >= ask price
    data.append({
        'ts': 1704067200000000002,
        'symbol': 'AAPL',
        'bid_px': 150.01,
        'bid_sz': 100,
        'ask_px': 150.00,  # Ask <= bid
        'ask_sz': 100,
        'level': 1
    })
    
    # Negative bid price
    data.append({
        'ts': 1704067200000000003,
        'symbol': 'AAPL',
        'bid_px': -150.0,
        'bid_sz': 100,
        'ask_px': 150.01,
        'ask_sz': 100,
        'level': 1
    })
    
    # Negative size
    data.append({
        'ts': 1704067200000000004,
        'symbol': 'AAPL',
        'bid_px': 149.99,
        'bid_sz': -100,
        'ask_px': 150.01,
        'ask_sz': 100,
        'level': 1
    })
    
    df = pd.DataFrame(data)
    df.to_parquet('book_update_bad_fixture.parquet')
    print(f"Generated bad book update fixture with {len(df)} records")
    return df


def main():
    """Generate all fixtures."""
    print("Generating test fixtures...")
    
    # Change to the sample_data directory
    sample_data_dir = Path(__file__).parent
    os.chdir(sample_data_dir)
    
    # Generate fixtures
    generate_trade_fixture()
    generate_book_update_fixture()
    generate_bad_data_fixture()
    generate_book_update_bad_fixture()
    
    print("All fixtures generated successfully!")


if __name__ == "__main__":
    import os
    main()
