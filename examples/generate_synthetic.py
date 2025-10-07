#!/usr/bin/env python3
"""
Synthetic market data generator for Flashback HFT backtesting engine.

Generates realistic L1 order book and trade data with:
- Microprice process with bursts and imbalance
- Configurable event count and parameters
- Parquet output for high performance
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
from datetime import datetime, timedelta


class MicropriceProcess:
    """Generates realistic microprice movements with bursts and imbalance."""
    
    def __init__(self, seed: int = 42):
        """Initialize the microprice process."""
        self.rng = np.random.RandomState(seed)
        self.current_price = 150.0
        self.current_spread = 0.01
        self.volatility = 0.02
        self.mean_reversion = 0.1
        self.burst_probability = 0.05
        self.burst_intensity = 2.0
        
    def generate_price_movement(self) -> Tuple[float, float, float]:
        """Generate next price movement with bursts and imbalance."""
        # Base random walk with mean reversion
        drift = -self.mean_reversion * (self.current_price - 150.0) / 150.0
        shock = self.rng.normal(0, self.volatility)
        
        # Burst component
        if self.rng.random() < self.burst_probability:
            burst = self.rng.normal(0, self.burst_intensity * self.volatility)
            shock += burst
        
        # Update price
        price_change = drift + shock
        self.current_price *= (1 + price_change)
        
        # Generate imbalance (affects spread)
        imbalance = self.rng.normal(0, 0.3)  # -1 to 1 range
        spread_adjustment = abs(imbalance) * 0.005
        self.current_spread = max(0.005, 0.01 + spread_adjustment)
        
        # Calculate bid/ask
        half_spread = self.current_spread / 2
        bid = self.current_price - half_spread
        ask = self.current_price + half_spread
        
        return bid, ask, imbalance


class OrderBookSimulator:
    """Simulates L1 order book with realistic dynamics."""
    
    def __init__(self, symbol: str, seed: int = 42):
        """Initialize order book simulator."""
        self.symbol = symbol
        self.rng = np.random.RandomState(seed)
        self.microprice = MicropriceProcess(seed)
        
        # Order book state
        self.bid_price = 150.0
        self.ask_price = 150.01
        self.bid_size = 1000
        self.ask_size = 1000
        
        # Trade generation parameters
        self.trade_probability = 0.1
        self.min_trade_size = 10
        self.max_trade_size = 500
        
    def generate_events(self, num_events: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate book updates and trades."""
        book_events = []
        trade_events = []
        
        # Start time (nanoseconds since epoch)
        start_time = int(pd.Timestamp('2024-01-01 09:30:00').value)
        current_time = start_time
        
        for i in range(num_events):
            # Generate new microprice
            bid, ask, imbalance = self.microprice.generate_price_movement()
            
            # Update order book
            self.bid_price = bid
            self.ask_price = ask
            
            # Generate realistic sizes with imbalance
            base_bid_size = 1000
            base_ask_size = 1000
            
            # Imbalance affects sizes
            if imbalance > 0:  # More buying pressure
                self.bid_size = int(base_bid_size * (1 + abs(imbalance) * 0.5))
                self.ask_size = int(base_ask_size * (1 - abs(imbalance) * 0.3))
            else:  # More selling pressure
                self.bid_size = int(base_bid_size * (1 - abs(imbalance) * 0.3))
                self.ask_size = int(base_ask_size * (1 + abs(imbalance) * 0.5))
            
            # Ensure minimum sizes
            self.bid_size = max(100, self.bid_size)
            self.ask_size = max(100, self.ask_size)
            
            # Generate book update event
            book_event = {
                'timestamp': current_time,
                'symbol': self.symbol,
                'bid_px': round(self.bid_price, 2),
                'bid_sz': self.bid_size,
                'ask_px': round(self.ask_price, 2),
                'ask_sz': self.ask_size,
                'level': 1,
                'event_type': 'L2_UPDATE'
            }
            book_events.append(book_event)
            
            # Generate trade event with probability
            if self.rng.random() < self.trade_probability:
                # Trade size based on imbalance
                if imbalance > 0.2:  # Strong buying pressure
                    side = 'B'  # Buy (aggressor)
                    trade_size = self.rng.randint(self.min_trade_size, self.max_trade_size)
                elif imbalance < -0.2:  # Strong selling pressure
                    side = 'S'  # Sell (aggressor)
                    trade_size = self.rng.randint(self.min_trade_size, self.max_trade_size)
                else:  # Neutral
                    side = self.rng.choice(['B', 'S'])
                    trade_size = self.rng.randint(self.min_trade_size, self.max_trade_size // 2)
                
                # Trade price (at bid/ask with some slippage)
                if side == 'B':
                    trade_price = self.ask_price + self.rng.normal(0, 0.001)
                else:
                    trade_price = self.bid_price - self.rng.normal(0, 0.001)
                
                trade_event = {
                    'timestamp': current_time + self.rng.randint(1000, 10000),  # Slight delay
                    'symbol': self.symbol,
                    'price': round(trade_price, 2),
                    'size': trade_size,
                    'aggressor_side': side,
                    'event_type': 'TICK'
                }
                trade_events.append(trade_event)
            
            # Advance time (microseconds between events)
            current_time += self.rng.randint(1000, 10000)
        
        return book_events, trade_events


def generate_synthetic_data(
    symbol: str = "AAPL",
    num_events: int = 1000000,
    seed: int = 42,
    output_dir: str = "examples/sample_data"
) -> Tuple[str, str]:
    """Generate synthetic market data and save to Parquet files.
    
    Args:
        symbol: Trading symbol
        num_events: Number of book update events to generate
        seed: Random seed for reproducibility
        output_dir: Output directory for Parquet files
        
    Returns:
        Tuple of (book_file_path, trade_file_path)
    """
    print(f"Generating {num_events:,} events for {symbol}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    simulator = OrderBookSimulator(symbol, seed)
    book_events, trade_events = simulator.generate_events(num_events)
    
    print(f"Generated {len(book_events):,} book updates and {len(trade_events):,} trades")
    
    # Convert to DataFrames
    book_df = pd.DataFrame(book_events)
    trade_df = pd.DataFrame(trade_events)
    
    # Sort by timestamp
    book_df = book_df.sort_values('timestamp').reset_index(drop=True)
    trade_df = trade_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to Parquet
    book_file = output_path / f"{symbol.lower()}_l1.parquet"
    trade_file = output_path / f"{symbol.lower()}_trades.parquet"
    
    book_df.to_parquet(book_file, index=False)
    trade_df.to_parquet(trade_file, index=False)
    
    print(f"Saved book data to: {book_file}")
    print(f"Saved trade data to: {trade_file}")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Book updates: {len(book_df):,}")
    print(f"Trades: {len(trade_df):,}")
    print(f"Time range: {pd.to_datetime(book_df['timestamp'].min(), unit='ns')} to {pd.to_datetime(book_df['timestamp'].max(), unit='ns')}")
    print(f"Price range: ${book_df['bid_px'].min():.2f} - ${book_df['ask_px'].max():.2f}")
    print(f"Average spread: ${(book_df['ask_px'] - book_df['bid_px']).mean():.4f}")
    
    return str(book_file), str(trade_file)


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic market data for Flashback")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--events", type=int, default=1000000, help="Number of events to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="examples/sample_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate data
    book_file, trade_file = generate_synthetic_data(
        symbol=args.symbol,
        num_events=args.events,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\n Synthetic data generation complete!")
    print(f" Book data: {book_file}")
    print(f" Trade data: {trade_file}")


if __name__ == "__main__":
    main()