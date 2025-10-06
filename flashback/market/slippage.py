"""
Slippage modeling for flashback HFT backtesting engine.

This module provides configurable slippage models including adverse-selection
models tied to order book imbalance.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .orders import OrderSide, OrderBookSnapshot


@dataclass
class SlippageConfig:
    """Configuration for slippage modeling."""
    base_slippage_bps: float = 0.5  # Base slippage in basis points
    adverse_selection_factor: float = 1.0  # Multiplier for adverse selection
    imbalance_threshold: float = 0.3  # Threshold for significant imbalance
    max_slippage_bps: float = 10.0  # Maximum slippage in basis points
    min_slippage_bps: float = 0.0  # Minimum slippage in basis points


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self,
        order_side: OrderSide,
        order_size: int,
        book_snapshot: OrderBookSnapshot,
        price: float
    ) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order_side: Side of the order (BUY/SELL)
            order_size: Size of the order
            book_snapshot: Current order book snapshot
            price: Intended execution price
            
        Returns:
            Slippage amount in price units
        """
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed slippage model with constant slippage."""
    
    def __init__(self, slippage_bps: float = 0.5):
        """
        Initialize fixed slippage model.
        
        Args:
            slippage_bps: Fixed slippage in basis points
        """
        self.slippage_bps = slippage_bps
    
    def calculate_slippage(
        self,
        order_side: OrderSide,
        order_size: int,
        book_snapshot: OrderBookSnapshot,
        price: float
    ) -> float:
        """Calculate fixed slippage."""
        return price * (self.slippage_bps / 10000.0)


class ImbalanceSlippageModel(SlippageModel):
    """
    Slippage model based on order book imbalance and adverse selection.
    
    This model calculates slippage based on:
    1. Base slippage (configurable)
    2. Order book imbalance (adverse selection)
    3. Order size impact
    """
    
    def __init__(self, config: SlippageConfig):
        """
        Initialize imbalance-based slippage model.
        
        Args:
            config: Slippage configuration
        """
        self.config = config
    
    def calculate_slippage(
        self,
        order_side: OrderSide,
        order_size: int,
        book_snapshot: OrderBookSnapshot,
        price: float
    ) -> float:
        """
        Calculate slippage based on order book imbalance.
        
        Args:
            order_side: Side of the order (BUY/SELL)
            order_size: Size of the order
            book_snapshot: Current order book snapshot
            price: Intended execution price
            
        Returns:
            Slippage amount in price units
        """
        # Calculate order book imbalance
        imbalance = self._calculate_imbalance(book_snapshot)
        
        # Calculate adverse selection factor
        adverse_selection = self._calculate_adverse_selection(
            order_side, imbalance
        )
        
        # Calculate size impact
        size_impact = self._calculate_size_impact(order_size, book_snapshot)
        
        # Combine factors
        total_slippage_bps = (
            self.config.base_slippage_bps +
            adverse_selection +
            size_impact
        )
        
        # Apply bounds
        total_slippage_bps = max(
            self.config.min_slippage_bps,
            min(total_slippage_bps, self.config.max_slippage_bps)
        )
        
        return price * (total_slippage_bps / 10000.0)
    
    def _calculate_imbalance(self, book_snapshot: OrderBookSnapshot) -> float:
        """
        Calculate order book imbalance.
        
        Args:
            book_snapshot: Order book snapshot
            
        Returns:
            Imbalance ratio (-1 to 1, negative = ask heavy, positive = bid heavy)
        """
        if not book_snapshot.bids or not book_snapshot.asks:
            return 0.0
        
        # Calculate total bid and ask volume
        bid_volume = sum(level.total_qty for level in book_snapshot.bids)
        ask_volume = sum(level.total_qty for level in book_snapshot.asks)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Imbalance = (bid_volume - ask_volume) / total_volume
        return (bid_volume - ask_volume) / total_volume
    
    def _calculate_adverse_selection(
        self,
        order_side: OrderSide,
        imbalance: float
    ) -> float:
        """
        Calculate adverse selection component.
        
        Args:
            order_side: Side of the order
            imbalance: Order book imbalance
            
        Returns:
            Adverse selection in basis points
        """
        # Adverse selection occurs when trading against the imbalance
        if order_side == OrderSide.BUY and imbalance < -self.config.imbalance_threshold:
            # Buying when ask-heavy (adverse selection)
            return abs(imbalance) * self.config.adverse_selection_factor * 10.0
        elif order_side == OrderSide.SELL and imbalance > self.config.imbalance_threshold:
            # Selling when bid-heavy (adverse selection)
            return abs(imbalance) * self.config.adverse_selection_factor * 10.0
        else:
            # No adverse selection
            return 0.0
    
    def _calculate_size_impact(
        self,
        order_size: int,
        book_snapshot: OrderBookSnapshot
    ) -> float:
        """
        Calculate size impact component.
        
        Args:
            order_size: Size of the order
            book_snapshot: Order book snapshot
            
        Returns:
            Size impact in basis points
        """
        if not book_snapshot.bids or not book_snapshot.asks:
            return 0.0
        
        # Calculate average book depth
        bid_depth = sum(level.total_qty for level in book_snapshot.bids)
        ask_depth = sum(level.total_qty for level in book_snapshot.asks)
        avg_depth = (bid_depth + ask_depth) / 2.0
        
        if avg_depth == 0:
            return 0.0
        
        # Size impact increases with order size relative to book depth
        size_ratio = order_size / avg_depth
        return min(size_ratio * 5.0, 5.0)  # Cap at 5 bps


class AdaptiveSlippageModel(SlippageModel):
    """
    Adaptive slippage model that learns from historical data.
    
    This model adjusts slippage based on recent market conditions
    and order execution patterns.
    """
    
    def __init__(self, config: SlippageConfig, lookback_period: int = 100):
        """
        Initialize adaptive slippage model.
        
        Args:
            config: Base slippage configuration
            lookback_period: Number of recent orders to consider
        """
        self.config = config
        self.lookback_period = lookback_period
        self.recent_slippages: List[float] = []
        self.recent_imbalances: List[float] = []
    
    def calculate_slippage(
        self,
        order_side: OrderSide,
        order_size: int,
        book_snapshot: OrderBookSnapshot,
        price: float
    ) -> float:
        """
        Calculate adaptive slippage.
        
        Args:
            order_side: Side of the order (BUY/SELL)
            order_size: Size of the order
            book_snapshot: Current order book snapshot
            price: Intended execution price
            
        Returns:
            Slippage amount in price units
        """
        # Calculate base slippage using imbalance model
        base_model = ImbalanceSlippageModel(self.config)
        base_slippage = base_model.calculate_slippage(
            order_side, order_size, book_snapshot, price
        )
        
        # Calculate current imbalance
        imbalance = self._calculate_imbalance(book_snapshot)
        
        # Update historical data
        self.recent_imbalances.append(imbalance)
        if len(self.recent_imbalances) > self.lookback_period:
            self.recent_imbalances.pop(0)
        
        # Calculate adaptive adjustment
        adaptive_factor = self._calculate_adaptive_factor()
        
        # Apply adaptive adjustment
        total_slippage = base_slippage * adaptive_factor
        
        # Update recent slippages for learning
        self.recent_slippages.append(total_slippage)
        if len(self.recent_slippages) > self.lookback_period:
            self.recent_slippages.pop(0)
        
        return total_slippage
    
    def _calculate_imbalance(self, book_snapshot: OrderBookSnapshot) -> float:
        """Calculate order book imbalance."""
        if not book_snapshot.bids or not book_snapshot.asks:
            return 0.0
        
        bid_volume = sum(level.total_qty for level in book_snapshot.bids)
        ask_volume = sum(level.total_qty for level in book_snapshot.asks)
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def _calculate_adaptive_factor(self) -> float:
        """Calculate adaptive adjustment factor."""
        if len(self.recent_slippages) < 10:
            return 1.0
        
        # Calculate recent volatility in slippage
        recent_slippages = np.array(self.recent_slippages[-20:])
        volatility = np.std(recent_slippages)
        
        # Calculate recent imbalance volatility
        recent_imbalances = np.array(self.recent_imbalances[-20:])
        imbalance_volatility = np.std(recent_imbalances)
        
        # Adaptive factor based on market volatility
        if imbalance_volatility > 0.5:  # High imbalance volatility
            return 1.2  # Increase slippage
        elif imbalance_volatility < 0.1:  # Low imbalance volatility
            return 0.8  # Decrease slippage
        else:
            return 1.0  # No adjustment


def create_slippage_model(
    model_type: str = "imbalance",
    config: Optional[SlippageConfig] = None
) -> SlippageModel:
    """
    Factory function to create slippage models.
    
    Args:
        model_type: Type of slippage model ("fixed", "imbalance", "adaptive")
        config: Slippage configuration (uses default if None)
        
    Returns:
        Slippage model instance
    """
    if config is None:
        config = SlippageConfig()
    
    if model_type == "fixed":
        return FixedSlippageModel(config.base_slippage_bps)
    elif model_type == "imbalance":
        return ImbalanceSlippageModel(config)
    elif model_type == "adaptive":
        return AdaptiveSlippageModel(config)
    else:
        raise ValueError(f"Unknown slippage model type: {model_type}")
