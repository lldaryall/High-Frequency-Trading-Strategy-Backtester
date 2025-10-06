"""Pydantic data schemas for strict event validation."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd
import numpy as np


class Trade(BaseModel):
    """Trade event schema with strict validation."""
    
    ts: int = Field(..., description="Timestamp in nanoseconds")
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., gt=0, description="Trade price (must be positive)")
    size: int = Field(..., gt=0, description="Trade size (must be positive)")
    aggressor_side: Literal['B', 'S'] = Field(..., description="Aggressor side: 'B' for buy, 'S' for sell")
    
    @field_validator('ts')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable (after 2000, before 2100)."""
        if v < 946684800000000000:  # 2000-01-01 in nanoseconds
            raise ValueError("Timestamp too early (before 2000)")
        if v > 4102444800000000000:  # 2100-01-01 in nanoseconds
            raise ValueError("Timestamp too late (after 2100)")
        return v
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        """Validate price is finite and reasonable."""
        if not np.isfinite(v):
            raise ValueError("Price must be finite")
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        """Validate size is positive integer."""
        if v <= 0:
            raise ValueError("Size must be positive")
        return v
    
    model_config = {"validate_assignment": True, "extra": "forbid"}


class BookUpdate(BaseModel):
    """Order book update event schema with strict validation."""
    
    ts: int = Field(..., description="Timestamp in nanoseconds")
    symbol: str = Field(..., description="Trading symbol")
    bid_px: Optional[float] = Field(None, gt=0, description="Bid price (must be positive if provided)")
    bid_sz: Optional[int] = Field(None, ge=0, description="Bid size (must be non-negative if provided)")
    ask_px: Optional[float] = Field(None, gt=0, description="Ask price (must be positive if provided)")
    ask_sz: Optional[int] = Field(None, ge=0, description="Ask size (must be non-negative if provided)")
    level: int = Field(1, eq=1, description="Book level (must be 1)")
    
    @field_validator('ts')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable."""
        if v < 946684800000000000:  # 2000-01-01 in nanoseconds
            raise ValueError("Timestamp too early (before 2000)")
        if v > 4102444800000000000:  # 2100-01-01 in nanoseconds
            raise ValueError("Timestamp too late (after 2100)")
        return v
    
    @field_validator('bid_px', 'ask_px')
    @classmethod
    def validate_prices(cls, v):
        """Validate prices are finite and positive."""
        if v is not None:
            if not np.isfinite(v):
                raise ValueError("Price must be finite")
            if v <= 0:
                raise ValueError("Price must be positive")
        return v
    
    @field_validator('bid_sz', 'ask_sz')
    @classmethod
    def validate_sizes(cls, v):
        """Validate sizes are non-negative integers."""
        if v is not None and v < 0:
            raise ValueError("Size must be non-negative")
        return v
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Validate level is exactly 1."""
        if v != 1:
            raise ValueError("Level must be 1")
        return v
    
    @model_validator(mode='after')
    def validate_bid_ask_consistency(self):
        """Validate bid/ask price consistency."""
        if self.bid_px is not None and self.ask_px is not None and self.bid_px >= self.ask_px:
            raise ValueError("Bid price must be less than ask price")
        return self
    
    model_config = {"validate_assignment": True, "extra": "forbid"}


def validate_trade_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single trade row.
    
    Args:
        row: Pandas Series representing a trade row
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    try:
        # Convert to dict and validate
        trade_dict = row.to_dict()
        Trade(**trade_dict)
        return True, warnings
    except Exception as e:
        warnings.append(f"Trade validation failed: {str(e)}")
        return False, warnings


def validate_book_update_row(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single book update row.
    
    Args:
        row: Pandas Series representing a book update row
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    try:
        # Convert to dict and validate
        book_dict = row.to_dict()
        BookUpdate(**book_dict)
        return True, warnings
    except Exception as e:
        warnings.append(f"Book update validation failed: {str(e)}")
        return False, warnings


def detect_event_type(df: pd.DataFrame) -> str:
    """
    Detect event type from DataFrame columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Event type: 'trade', 'book_update', or 'unknown'
    """
    required_trade_cols = {'ts', 'symbol', 'price', 'size', 'aggressor_side'}
    required_book_cols = {'ts', 'symbol', 'bid_px', 'bid_sz', 'ask_px', 'ask_sz', 'level'}
    
    df_cols = set(df.columns)
    
    if required_trade_cols.issubset(df_cols):
        return 'trade'
    elif required_book_cols.issubset(df_cols):
        return 'book_update'
    else:
        return 'unknown'


def validate_events(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate events in DataFrame and return cleaned data with warnings.
    
    Args:
        df: DataFrame containing events
        
    Returns:
        Tuple of (cleaned_dataframe, list_of_warnings)
    """
    warnings = []
    
    if df.empty:
        warnings.append("DataFrame is empty")
        return df, warnings
    
    # Detect event type
    event_type = detect_event_type(df)
    if event_type == 'unknown':
        warnings.append("Unknown event type - cannot validate")
        return df, warnings
    
    # Validate timestamp monotonicity
    if 'ts' in df.columns:
        # Sort by timestamp first
        df_sorted = df.sort_values('ts').reset_index(drop=True)
        
        # Check for non-monotonic timestamps
        non_monotonic = df_sorted['ts'].diff() < 0
        if non_monotonic.any():
            non_monotonic_count = non_monotonic.sum()
            warnings.append(f"Found {non_monotonic_count} non-monotonic timestamps - dropping rows")
            
            # Keep only monotonic rows
            df_sorted = df_sorted[~non_monotonic].reset_index(drop=True)
        
        df = df_sorted
    
    # Validate individual rows
    valid_rows = []
    row_warnings = []
    
    for idx, row in df.iterrows():
        if event_type == 'trade':
            is_valid, row_warn = validate_trade_row(row)
        elif event_type == 'book_update':
            is_valid, row_warn = validate_book_update_row(row)
        else:
            is_valid, row_warn = False, ["Unknown event type"]
        
        if is_valid:
            valid_rows.append(idx)
        else:
            row_warnings.extend([f"Row {idx}: {w}" for w in row_warn])
    
    # Add row validation warnings
    warnings.extend(row_warnings)
    
    # Filter to valid rows
    if valid_rows:
        cleaned_df = df.iloc[valid_rows].reset_index(drop=True)
    else:
        cleaned_df = pd.DataFrame(columns=df.columns)
        warnings.append("No valid rows found after validation")
    
    return cleaned_df, warnings
