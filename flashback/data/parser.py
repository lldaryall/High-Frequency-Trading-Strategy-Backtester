"""Data parsers for different market data formats."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from ..core.events import MarketDataEvent


class DataParser(ABC):
    """Base class for market data parsers."""
    
    @abstractmethod
    def parse(self, data: pd.DataFrame) -> List[MarketDataEvent]:
        """Parse raw data into market data events."""
        pass


class TickDataParser(DataParser):
    """Parser for tick-by-tick market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize tick data parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def parse(self, data: pd.DataFrame) -> List[MarketDataEvent]:
        """
        Parse tick data into market data events.
        
        Args:
            data: DataFrame with columns: ts, symbol, side, price, size, event_type
            
        Returns:
            List of MarketDataEvent objects
        """
        events = []
        
        for _, row in data.iterrows():
            event = MarketDataEvent(
                timestamp=row['ts'],
                symbol=row['symbol'],
                side=row['side'],
                price=float(row['price']),
                size=int(row['size']),
                event_type_str=row['event_type']
            )
            events.append(event)
            
        return events


class L2DataParser(DataParser):
    """Parser for Level 2 order book data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize L2 data parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.max_levels = config.get("max_levels", 10)
        
    def parse(self, data: pd.DataFrame) -> List[MarketDataEvent]:
        """
        Parse L2 order book data into market data events.
        
        Args:
            data: DataFrame with L2 data structure
            
        Returns:
            List of MarketDataEvent objects
        """
        events = []
        
        # Group by timestamp and symbol to process order book snapshots
        grouped = data.groupby(['ts', 'symbol'])
        
        for (timestamp, symbol), group in grouped:
            # Process bid levels
            bid_levels = group[group['side'] == 'B'].sort_values('price', ascending=False)
            for _, row in bid_levels.head(self.max_levels).iterrows():
                event = MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    side='B',
                    price=float(row['price']),
                    size=int(row['size']),
                    event_type_str='L2_BID'
                )
                events.append(event)
                
            # Process ask levels
            ask_levels = group[group['side'] == 'S'].sort_values('price', ascending=True)
            for _, row in ask_levels.head(self.max_levels).iterrows():
                event = MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    side='S',
                    price=float(row['price']),
                    size=int(row['size']),
                    event_type_str='L2_ASK'
                )
                events.append(event)
                
        return events


class ImbalanceParser(DataParser):
    """Parser for order book imbalance data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize imbalance parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.lookback_window = config.get("lookback_window", 100)
        
    def parse(self, data: pd.DataFrame) -> List[MarketDataEvent]:
        """
        Parse data and calculate order book imbalance.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            List of MarketDataEvent objects with imbalance information
        """
        events = []
        
        # Group by symbol
        for symbol, group in data.groupby('symbol'):
            group = group.sort_values('ts')
            
            # Calculate rolling imbalance
            group['bid_volume'] = group[group['side'] == 'B']['size'].rolling(
                window=self.lookback_window, min_periods=1
            ).sum()
            group['ask_volume'] = group[group['side'] == 'S']['size'].rolling(
                window=self.lookback_window, min_periods=1
            ).sum()
            
            group['imbalance'] = (group['bid_volume'] - group['ask_volume']) / (
                group['bid_volume'] + group['ask_volume']
            )
            
            # Create imbalance events
            for _, row in group.iterrows():
                if not pd.isna(row['imbalance']):
                    event = MarketDataEvent(
                        timestamp=row['ts'],
                        symbol=symbol,
                        side='I',  # Imbalance
                        price=float(row['imbalance']),
                        size=0,
                        event_type_str='IMBALANCE'
                    )
                    events.append(event)
                    
        return events


class VolumeProfileParser(DataParser):
    """Parser for volume profile data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize volume profile parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.price_bins = config.get("price_bins", 50)
        self.time_window = config.get("time_window", "1H")
        
    def parse(self, data: pd.DataFrame) -> List[MarketDataEvent]:
        """
        Parse data and calculate volume profile.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            List of MarketDataEvent objects with volume profile information
        """
        events = []
        
        # Group by symbol and time window
        data['time_window'] = data['ts'].dt.floor(self.time_window)
        
        for symbol, group in data.groupby(['symbol', 'time_window']):
            symbol_name, time_window = symbol
            
            # Calculate volume at each price level
            price_volume = group.groupby('price')['size'].sum().reset_index()
            
            # Create price bins
            price_min, price_max = price_volume['price'].min(), price_volume['price'].max()
            price_bins = np.linspace(price_min, price_max, self.price_bins + 1)
            
            for i in range(len(price_bins) - 1):
                bin_start, bin_end = price_bins[i], price_bins[i + 1]
                bin_volume = price_volume[
                    (price_volume['price'] >= bin_start) & 
                    (price_volume['price'] < bin_end)
                ]['size'].sum()
                
                if bin_volume > 0:
                    event = MarketDataEvent(
                        timestamp=time_window,
                        symbol=symbol_name,
                        side='V',  # Volume
                        price=float((bin_start + bin_end) / 2),
                        size=int(bin_volume),
                        event_type_str='VOLUME_PROFILE'
                    )
                    events.append(event)
                    
        return events
