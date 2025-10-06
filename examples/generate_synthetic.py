#!/usr/bin/env python3
"""
Synthetic market data generator for flashback HFT backtesting engine.

Generates realistic L1 order book and trade data with:
- Microprice process with mean reversion
- Burst events (volatility spikes)
- Order flow imbalance
- Configurable number of events (~1M default)
- Parquet output format
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current market state for simulation."""
    timestamp: int
    microprice: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    imbalance: float
    volatility: float
    burst_active: bool
    burst_remaining: int


class SyntheticDataGenerator:
    """Generates synthetic L1 order book and trade data."""
    
    def __init__(
        self,
        symbol: str = "AAPL",
        initial_price: float = 150.0,
        tick_size: float = 0.01,
        min_spread: float = 0.01,
        max_spread: float = 0.10,
        base_volatility: float = 0.02,
        burst_probability: float = 0.001,
        burst_duration: int = 100,
        burst_volatility_multiplier: float = 5.0,
        mean_reversion_speed: float = 0.1,
        imbalance_persistence: float = 0.8,
        trade_probability: float = 0.3,
        min_trade_size: int = 100,
        max_trade_size: int = 10000,
        seed: int = 42
    ):
        """Initialize the synthetic data generator.
        
        Args:
            symbol: Trading symbol
            initial_price: Starting price
            tick_size: Minimum price increment
            min_spread: Minimum bid-ask spread
            max_spread: Maximum bid-ask spread
            base_volatility: Base volatility level
            burst_probability: Probability of burst event per event
            burst_duration: Duration of burst events (in events)
            burst_volatility_multiplier: Volatility multiplier during bursts
            mean_reversion_speed: Speed of mean reversion
            imbalance_persistence: Persistence of order flow imbalance
            trade_probability: Probability of trade per event
            min_trade_size: Minimum trade size
            max_trade_size: Maximum trade size
            seed: Random seed for reproducibility
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.base_volatility = base_volatility
        self.burst_probability = burst_probability
        self.burst_duration = burst_duration
        self.burst_volatility_multiplier = burst_volatility_multiplier
        self.mean_reversion_speed = mean_reversion_speed
        self.imbalance_persistence = imbalance_persistence
        self.trade_probability = trade_probability
        self.min_trade_size = min_trade_size
        self.max_trade_size = max_trade_size
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize market state
        self.state = MarketState(
            timestamp=0,
            microprice=initial_price,
            bid_price=initial_price - min_spread / 2,
            ask_price=initial_price + min_spread / 2,
            bid_size=1000,
            ask_size=1000,
            imbalance=0.0,
            volatility=base_volatility,
            burst_active=False,
            burst_remaining=0
        )
        
        # Event storage
        self.book_events: List[Dict[str, Any]] = []
        self.trade_events: List[Dict[str, Any]] = []
        
    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self.tick_size) * self.tick_size
    
    def _update_imbalance(self) -> None:
        """Update order flow imbalance with persistence."""
        # Add random component
        imbalance_change = np.random.normal(0, 0.1)
        
        # Apply persistence
        self.state.imbalance = (
            self.state.imbalance * self.imbalance_persistence + 
            imbalance_change * (1 - self.imbalance_persistence)
        )
        
        # Clamp to [-1, 1]
        self.state.imbalance = np.clip(self.state.imbalance, -1, 1)
    
    def _update_volatility(self) -> None:
        """Update volatility, handling burst events."""
        if self.state.burst_active:
            self.state.volatility = self.base_volatility * self.burst_volatility_multiplier
            self.state.burst_remaining -= 1
            if self.state.burst_remaining <= 0:
                self.state.burst_active = False
        else:
            # Check for burst event
            if np.random.random() < self.burst_probability:
                self.state.burst_active = True
                self.state.burst_remaining = self.burst_duration
                self.state.volatility = self.base_volatility * self.burst_volatility_multiplier
            else:
                # Mean revert volatility
                self.state.volatility = (
                    self.state.volatility * 0.99 + 
                    self.base_volatility * 0.01
                )
    
    def _update_microprice(self) -> None:
        """Update microprice with mean reversion and volatility."""
        # Mean reversion component
        mean_reversion = -self.mean_reversion_speed * (self.state.microprice - 150.0)
        
        # Random walk component
        random_component = np.random.normal(0, self.state.volatility)
        
        # Update microprice
        self.state.microprice += mean_reversion + random_component
        
        # Ensure positive price
        self.state.microprice = max(self.state.microprice, 1.0)
    
    def _update_spread(self) -> None:
        """Update bid-ask spread based on imbalance and volatility."""
        # Base spread
        base_spread = self.min_spread
        
        # Volatility component
        vol_component = self.state.volatility * 0.1
        
        # Imbalance component (wider spread when imbalanced)
        imbalance_component = abs(self.state.imbalance) * 0.05
        
        # Calculate target spread
        target_spread = base_spread + vol_component + imbalance_component
        target_spread = np.clip(target_spread, self.min_spread, self.max_spread)
        
        # Update spread with some persistence
        current_spread = self.state.ask_price - self.state.bid_price
        new_spread = current_spread * 0.7 + target_spread * 0.3
        
        # Update prices
        mid_price = self.state.microprice
        self.state.bid_price = self._round_to_tick(mid_price - new_spread / 2)
        self.state.ask_price = self._round_to_tick(mid_price + new_spread / 2)
        
        # Ensure minimum spread
        if self.state.ask_price - self.state.bid_price < self.min_spread:
            self.state.ask_price = self.state.bid_price + self.min_spread
    
    def _update_sizes(self) -> None:
        """Update bid and ask sizes based on imbalance."""
        base_size = 1000
        
        # Imbalance affects sizes
        if self.state.imbalance > 0:
            # More buy pressure -> larger ask size, smaller bid size
            self.state.ask_size = int(base_size * (1 + abs(self.state.imbalance)))
            self.state.bid_size = int(base_size * (1 - abs(self.state.imbalance) * 0.5))
        else:
            # More sell pressure -> larger bid size, smaller ask size
            self.state.bid_size = int(base_size * (1 + abs(self.state.imbalance)))
            self.state.ask_size = int(base_size * (1 - abs(self.state.imbalance) * 0.5))
        
        # Add some randomness
        self.state.bid_size = max(100, int(self.state.bid_size * np.random.uniform(0.5, 1.5)))
        self.state.ask_size = max(100, int(self.state.ask_size * np.random.uniform(0.5, 1.5)))
    
    def _generate_trade(self) -> Dict[str, Any]:
        """Generate a trade event."""
        # Determine trade side based on imbalance
        if self.state.imbalance > 0.2:
            side = "BUY"
        elif self.state.imbalance < -0.2:
            side = "SELL"
        else:
            side = random.choice(["BUY", "SELL"])
        
        # Determine trade price
        if side == "BUY":
            # Buy at ask price with some slippage
            price = self.state.ask_price + np.random.normal(0, self.tick_size * 0.5)
        else:
            # Sell at bid price with some slippage
            price = self.state.bid_price + np.random.normal(0, self.tick_size * 0.5)
        
        price = self._round_to_tick(price)
        
        # Determine trade size
        size = np.random.randint(self.min_trade_size, self.max_trade_size + 1)
        
        # Update imbalance after trade
        if side == "BUY":
            self.state.imbalance += 0.1
        else:
            self.state.imbalance -= 0.1
        
        self.state.imbalance = np.clip(self.state.imbalance, -1, 1)
        
        return {
            "timestamp": self.state.timestamp,
            "symbol": self.symbol,
            "side": side,
            "price": price,
            "size": size,
            "event_type": "TRADE",
            "bid_price": None,
            "bid_size": None,
            "ask_price": None,
            "ask_size": None
        }
    
    def _generate_book_update(self) -> Dict[str, Any]:
        """Generate a book update event."""
        return {
            "timestamp": self.state.timestamp,
            "symbol": self.symbol,
            "bid_price": self.state.bid_price,
            "bid_size": self.state.bid_size,
            "ask_price": self.state.ask_price,
            "ask_size": self.state.ask_size,
            "event_type": "BOOK_UPDATE",
            "side": None,
            "price": None,
            "size": None
        }
    
    def generate_events(self, num_events: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic market data events.
        
        Args:
            num_events: Number of events to generate
            
        Returns:
            Tuple of (book_events_df, trade_events_df)
        """
        logger.info(f"Generating {num_events:,} synthetic market events...")
        
        # Clear previous events
        self.book_events.clear()
        self.trade_events.clear()
        
        # Generate events
        for i in range(num_events):
            if i % 100000 == 0:
                logger.info(f"Generated {i:,} events...")
            
            # Update timestamp (nanoseconds)
            self.state.timestamp = 1000000000000000000 + i * 1000000  # 1ms intervals
            
            # Update market state
            self._update_imbalance()
            self._update_volatility()
            self._update_microprice()
            self._update_spread()
            self._update_sizes()
            
            # Generate book update
            book_event = self._generate_book_update()
            self.book_events.append(book_event)
            
            # Generate trade with probability
            if np.random.random() < self.trade_probability:
                trade_event = self._generate_trade()
                self.trade_events.append(trade_event)
        
        logger.info(f"Generated {len(self.book_events):,} book events and {len(self.trade_events):,} trade events")
        
        # Convert to DataFrames
        book_df = pd.DataFrame(self.book_events)
        trade_df = pd.DataFrame(self.trade_events)
        
        # Ensure consistent data types
        for df in [book_df, trade_df]:
            # Convert None to NaN for numeric columns
            numeric_cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size', 'price', 'size']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert None to empty string for string columns
            string_cols = ['side']
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('')
        
        return book_df, trade_df
    
    def save_to_parquet(self, book_df: pd.DataFrame, trade_df: pd.DataFrame, output_dir: Path) -> None:
        """Save data to Parquet files.
        
        Args:
            book_df: Book events DataFrame
            trade_df: Trade events DataFrame
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save book events
        book_file = output_dir / f"{self.symbol.lower()}_book.parquet"
        book_df.to_parquet(book_file, index=False)
        logger.info(f"Saved book events to {book_file}")
        
        # Save trade events
        trade_file = output_dir / f"{self.symbol.lower()}_trades.parquet"
        trade_df.to_parquet(trade_file, index=False)
        logger.info(f"Saved trade events to {trade_file}")
        
        # Create L1 data in the same format as existing sample data (trade events only)
        # Filter out NaN values and keep only valid trade events
        l1_df = trade_df.dropna(subset=['side', 'price', 'size']).copy()
        l1_df = l1_df[['timestamp', 'symbol', 'side', 'price', 'size', 'event_type']]
        l1_df = l1_df.sort_values("timestamp")
        
        # Save L1 data (compatible with existing config)
        l1_file = output_dir / f"{self.symbol.lower()}_l1.parquet"
        l1_df.to_parquet(l1_file, index=False)
        logger.info(f"Saved L1 data to {l1_file}")
        
        # Also save as CSV for compatibility
        l1_csv_file = output_dir / f"{self.symbol.lower()}_l1.csv"
        l1_df.to_csv(l1_csv_file, index=False)
        logger.info(f"Saved L1 data to {l1_csv_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate synthetic market data")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--events", type=int, default=1000000, help="Number of events to generate")
    parser.add_argument("--output-dir", default="examples/sample_data", help="Output directory")
    parser.add_argument("--initial-price", type=float, default=150.0, help="Initial price")
    parser.add_argument("--volatility", type=float, default=0.02, help="Base volatility")
    parser.add_argument("--burst-prob", type=float, default=0.001, help="Burst probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(
        symbol=args.symbol,
        initial_price=args.initial_price,
        base_volatility=args.volatility,
        burst_probability=args.burst_prob,
        seed=args.seed
    )
    
    # Generate events
    book_df, trade_df = generator.generate_events(args.events)
    
    # Save to Parquet
    output_dir = Path(args.output_dir)
    generator.save_to_parquet(book_df, trade_df, output_dir)
    
    # Print summary
    print(f"\nSynthetic data generation complete!")
    print(f"Symbol: {args.symbol}")
    print(f"Events generated: {args.events:,}")
    print(f"Book events: {len(book_df):,}")
    print(f"Trade events: {len(trade_df):,}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {args.symbol.lower()}_book.parquet")
    print(f"  - {args.symbol.lower()}_trades.parquet")
    print(f"  - {args.symbol.lower()}_l1.parquet")


if __name__ == "__main__":
    main()
