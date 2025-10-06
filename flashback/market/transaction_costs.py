"""
Transaction costs modeling for flashback HFT backtesting engine.

This module provides configurable transaction cost models including
maker/taker schedules and per-share costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .orders import OrderSide, OrderType, TimeInForce


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs."""
    maker_fee_bps: float = 0.0  # Maker fee in basis points
    taker_fee_bps: float = 0.5  # Taker fee in basis points
    per_share_cost: float = 0.0  # Per-share cost in dollars
    min_fee: float = 0.0  # Minimum fee per trade
    max_fee: float = 1000.0  # Maximum fee per trade
    exchange_fees: Dict[str, Dict[str, float]] = None  # Exchange-specific fees


@dataclass
class FeeTier:
    """Fee tier for maker/taker schedules."""
    min_volume: float = 0.0  # Minimum monthly volume
    max_volume: float = float('inf')  # Maximum monthly volume
    maker_fee_bps: float = 0.0  # Maker fee in basis points
    taker_fee_bps: float = 0.5  # Taker fee in basis points


@dataclass
class TransactionCosts:
    """Transaction cost breakdown."""
    maker_taker_fee: float = 0.0
    per_share_cost: float = 0.0
    total_cost: float = 0.0
    fee_type: str = "unknown"  # "maker" or "taker"


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""
    
    @abstractmethod
    def calculate_costs(
        self,
        order_side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: float,
        is_maker: bool = False,
        monthly_volume: float = 0.0
    ) -> TransactionCosts:
        """
        Calculate transaction costs for an order.
        
        Args:
            order_side: Side of the order (BUY/SELL)
            order_type: Type of the order (MARKET/LIMIT/etc.)
            quantity: Number of shares
            price: Execution price
            is_maker: Whether the order is a maker
            monthly_volume: Monthly trading volume for tier calculation
            
        Returns:
            Transaction cost breakdown
        """
        pass


class SimpleTransactionCostModel(TransactionCostModel):
    """Simple transaction cost model with fixed maker/taker fees."""
    
    def __init__(self, config: TransactionCostConfig):
        """
        Initialize simple transaction cost model.
        
        Args:
            config: Transaction cost configuration
        """
        self.config = config
    
    def calculate_costs(
        self,
        order_side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: float,
        is_maker: bool = False,
        monthly_volume: float = 0.0
    ) -> TransactionCosts:
        """Calculate simple transaction costs."""
        # Determine fee type and rate
        if is_maker:
            fee_bps = self.config.maker_fee_bps
            fee_type = "maker"
        else:
            fee_bps = self.config.taker_fee_bps
            fee_type = "taker"
        
        # Calculate maker/taker fee
        notional_value = quantity * price
        maker_taker_fee = notional_value * (fee_bps / 10000.0)
        
        # Calculate per-share cost
        per_share_cost = quantity * self.config.per_share_cost
        
        # Calculate total cost
        total_cost = maker_taker_fee + per_share_cost
        
        # Apply min/max fee bounds
        total_cost = max(
            self.config.min_fee,
            min(total_cost, self.config.max_fee)
        )
        
        return TransactionCosts(
            maker_taker_fee=maker_taker_fee,
            per_share_cost=per_share_cost,
            total_cost=total_cost,
            fee_type=fee_type
        )


class TieredTransactionCostModel(TransactionCostModel):
    """Tiered transaction cost model with volume-based fee schedules."""
    
    def __init__(self, config: TransactionCostConfig, fee_tiers: List[FeeTier]):
        """
        Initialize tiered transaction cost model.
        
        Args:
            config: Base transaction cost configuration
            fee_tiers: List of fee tiers sorted by volume
        """
        self.config = config
        self.fee_tiers = sorted(fee_tiers, key=lambda x: x.min_volume)
    
    def calculate_costs(
        self,
        order_side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: float,
        is_maker: bool = False,
        monthly_volume: float = 0.0
    ) -> TransactionCosts:
        """Calculate tiered transaction costs."""
        # Find appropriate fee tier
        fee_tier = self._find_fee_tier(monthly_volume)
        
        # Determine fee type and rate
        if is_maker:
            fee_bps = fee_tier.maker_fee_bps
            fee_type = "maker"
        else:
            fee_bps = fee_tier.taker_fee_bps
            fee_type = "taker"
        
        # Calculate maker/taker fee
        notional_value = quantity * price
        maker_taker_fee = notional_value * (fee_bps / 10000.0)
        
        # Calculate per-share cost
        per_share_cost = quantity * self.config.per_share_cost
        
        # Calculate total cost
        total_cost = maker_taker_fee + per_share_cost
        
        # Apply min/max fee bounds
        total_cost = max(
            self.config.min_fee,
            min(total_cost, self.config.max_fee)
        )
        
        return TransactionCosts(
            maker_taker_fee=maker_taker_fee,
            per_share_cost=per_share_cost,
            total_cost=total_cost,
            fee_type=fee_type
        )
    
    def _find_fee_tier(self, monthly_volume: float) -> FeeTier:
        """Find the appropriate fee tier for given monthly volume."""
        for tier in reversed(self.fee_tiers):  # Start from highest volume
            if monthly_volume >= tier.min_volume:
                return tier
        
        # Return the lowest tier if no match
        return self.fee_tiers[0]


class ExchangeTransactionCostModel(TransactionCostModel):
    """Exchange-specific transaction cost model."""
    
    def __init__(self, config: TransactionCostConfig):
        """
        Initialize exchange-specific transaction cost model.
        
        Args:
            config: Transaction cost configuration with exchange fees
        """
        self.config = config
        self.exchange_fees = config.exchange_fees or {}
    
    def calculate_costs(
        self,
        order_side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: float,
        is_maker: bool = False,
        monthly_volume: float = 0.0,
        exchange: str = "default"
    ) -> TransactionCosts:
        """Calculate exchange-specific transaction costs."""
        # Get exchange-specific fees
        exchange_config = self.exchange_fees.get(exchange, {})
        
        # Determine fee type and rate
        if is_maker:
            fee_bps = exchange_config.get('maker_fee_bps', self.config.maker_fee_bps)
            fee_type = "maker"
        else:
            fee_bps = exchange_config.get('taker_fee_bps', self.config.taker_fee_bps)
            fee_type = "taker"
        
        # Calculate maker/taker fee
        notional_value = quantity * price
        maker_taker_fee = notional_value * (fee_bps / 10000.0)
        
        # Calculate per-share cost
        per_share_cost = quantity * self.config.per_share_cost
        
        # Add exchange-specific per-share costs
        exchange_per_share = exchange_config.get('per_share_cost', 0.0)
        per_share_cost += quantity * exchange_per_share
        
        # Calculate total cost
        total_cost = maker_taker_fee + per_share_cost
        
        # Apply min/max fee bounds
        min_fee = exchange_config.get('min_fee', self.config.min_fee)
        max_fee = exchange_config.get('max_fee', self.config.max_fee)
        total_cost = max(min_fee, min(total_cost, max_fee))
        
        return TransactionCosts(
            maker_taker_fee=maker_taker_fee,
            per_share_cost=per_share_cost,
            total_cost=total_cost,
            fee_type=fee_type
        )


def create_transaction_cost_model(
    model_type: str = "simple",
    config: Optional[TransactionCostConfig] = None,
    fee_tiers: Optional[List[FeeTier]] = None
) -> TransactionCostModel:
    """
    Factory function to create transaction cost models.
    
    Args:
        model_type: Type of cost model ("simple", "tiered", "exchange")
        config: Transaction cost configuration
        fee_tiers: Fee tiers for tiered model
        
    Returns:
        Transaction cost model instance
    """
    if config is None:
        config = TransactionCostConfig()
    
    if model_type == "simple":
        return SimpleTransactionCostModel(config)
    elif model_type == "tiered":
        if fee_tiers is None:
            # Default fee tiers
            fee_tiers = [
                FeeTier(min_volume=0, max_volume=1000000, maker_fee_bps=0.0, taker_fee_bps=0.5),
                FeeTier(min_volume=1000000, max_volume=10000000, maker_fee_bps=0.0, taker_fee_bps=0.3),
                FeeTier(min_volume=10000000, max_volume=float('inf'), maker_fee_bps=0.0, taker_fee_bps=0.1),
            ]
        return TieredTransactionCostModel(config, fee_tiers)
    elif model_type == "exchange":
        return ExchangeTransactionCostModel(config)
    else:
        raise ValueError(f"Unknown transaction cost model type: {model_type}")


def calculate_maker_taker_status(
    order_type: OrderType,
    time_in_force: TimeInForce,
    order_price: float,
    market_price: float,
    order_side: OrderSide
) -> bool:
    """
    Determine if an order is a maker or taker.
    
    Args:
        order_type: Type of the order
        time_in_force: Time in force
        order_price: Order price
        market_price: Current market price
        order_side: Side of the order
        
    Returns:
        True if maker, False if taker
    """
    # Market orders are always takers
    if order_type == OrderType.MARKET:
        return False
    
    # IOC and FOK orders are typically takers
    if time_in_force in [TimeInForce.IOC, TimeInForce.FOK]:
        return False
    
    # Limit orders are makers if they don't cross the spread
    if order_side == OrderSide.BUY:
        return order_price < market_price
    else:  # SELL
        return order_price > market_price
