"""Data loader for market data files."""

import os
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import validate_events as _validate_events, detect_event_type
from ..utils.logger import get_logger


class DataLoader:
    """Loads market data from various file formats."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary with data source settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        self.source = config.get("source", "")
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.symbols = config.get("symbols", [])
        
    def load(self) -> pd.DataFrame:
        """
        Load market data from the configured source.
        
        Returns:
            DataFrame with market data containing columns: ts, symbol, side, price, size, event_type
        """
        if not self.source:
            raise ValueError("No data source specified")
            
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Data file not found: {self.source}")
            
        # Determine file format
        file_ext = os.path.splitext(self.source)[1].lower()
        
        if file_ext == '.parquet':
            return self._load_parquet()
        elif file_ext == '.csv':
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
    def _load_parquet(self) -> pd.DataFrame:
        """Load data from Parquet file."""
        self.logger.info(f"Loading Parquet data from {self.source}")
        
        # Read with PyArrow for better performance
        table = pq.read_table(self.source)
        df = table.to_pandas()
        
        # Convert timestamp column
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
            
        # Filter by date range if specified
        if self.start_date or self.end_date:
            df = self._filter_by_date(df)
            
        # Filter by symbols if specified
        if self.symbols:
            df = df[df['symbol'].isin(self.symbols)]
            
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df)} records")
        return df
        
    def _load_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        self.logger.info(f"Loading CSV data from {self.source}")
        
        # Read CSV with optimized settings
        df = pd.read_csv(
            self.source,
            parse_dates=['ts'],
            dtype={
                'symbol': 'category',
                'side': 'category', 
                'price': 'float64',
                'size': 'int64',
                'event_type': 'category'
            }
        )
        
        # Filter by date range if specified
        if self.start_date or self.end_date:
            df = self._filter_by_date(df)
            
        # Filter by symbols if specified
        if self.symbols:
            df = df[df['symbol'].isin(self.symbols)]
            
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df)} records")
        return df
        
    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if self.start_date:
            start_ts = pd.to_datetime(self.start_date)
            df = df[df['ts'] >= start_ts]
            
        if self.end_date:
            end_ts = pd.to_datetime(self.end_date)
            df = df[df['ts'] <= end_ts]
            
        return df
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the loaded data has the required format.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['ts', 'symbol', 'side', 'price', 'size', 'event_type']
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            self.logger.error("Timestamp column must be datetime type")
            return False
            
        if not pd.api.types.is_numeric_dtype(df['price']):
            self.logger.error("Price column must be numeric")
            return False
            
        if not pd.api.types.is_numeric_dtype(df['size']):
            self.logger.error("Size column must be numeric")
            return False
            
        # Check for required values
        if df['side'].nunique() != 2 or set(df['side'].unique()) != {'B', 'S'}:
            self.logger.error("Side column must contain only 'B' and 'S' values")
            return False
            
        # Check for negative prices or sizes
        if (df['price'] <= 0).any():
            self.logger.error("Price values must be positive")
            return False
            
        if (df['size'] <= 0).any():
            self.logger.error("Size values must be positive")
            return False
            
        self.logger.info("Data validation passed")
        return True
        
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data statistics
        """
        info = {
            "total_records": len(df),
            "start_time": df['ts'].min(),
            "end_time": df['ts'].max(),
            "duration": df['ts'].max() - df['ts'].min(),
            "symbols": df['symbol'].unique().tolist(),
            "event_types": df['event_type'].unique().tolist(),
            "price_range": {
                "min": df['price'].min(),
                "max": df['price'].max(),
                "mean": df['price'].mean(),
            },
            "size_range": {
                "min": df['size'].min(),
                "max": df['size'].max(),
                "mean": df['size'].mean(),
            },
        }
        
        return info


def read_parquet(path: str) -> pd.DataFrame:
    """
    Read Parquet file and return DataFrame.
    
    Args:
        path: Path to Parquet file
        
    Returns:
        DataFrame with data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading Parquet file {path}: {e}")


def read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV file and return DataFrame.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file {path}: {e}")


def validate_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Validate events in DataFrame and return cleaned data with warnings.
    
    Args:
        df: DataFrame containing events
        
    Returns:
        Tuple of (cleaned_dataframe, list_of_warnings)
    """
    return _validate_events(df)
