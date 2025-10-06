"""Fee models for trading costs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import math

from .orders import Fill, OrderSide


@dataclass
class FeeConfig:
    """Configuration for fee calculation."""
    maker_bps: float = 0.0  # Maker fee in basis points
    taker_bps: float = 0.0  # Taker fee in basis points
    maker_per_share: float = 0.0  # Maker fee per share
    taker_per_share: float = 0.0  # Taker fee per share
    min_fee: float = 0.0  # Minimum fee
    max_fee: Optional[float] = None  # Maximum fee (None for no limit)
    round_to_cents: bool = True  # Round to nearest cent


class FeeModel(ABC):
    """Abstract base class for fee models."""
    
    @abstractmethod
    def calculate_fee(self, fill: Fill) -> float:
        """
        Calculate the fee for a fill.
        
        Args:
            fill: Fill to calculate fee for
            
        Returns:
            Fee amount
        """
        pass
    
    @abstractmethod
    def get_fee_breakdown(self, fill: Fill) -> Dict[str, float]:
        """
        Get detailed fee breakdown.
        
        Args:
            fill: Fill to calculate fee for
            
        Returns:
            Dictionary with fee components
        """
        pass


class PerTradeFeeModel(FeeModel):
    """Simple per-trade fee model."""
    
    def __init__(self, fee_per_trade: float = 0.0):
        """
        Initialize per-trade fee model.
        
        Args:
            fee_per_trade: Fixed fee per trade
        """
        self.fee_per_trade = fee_per_trade
    
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate fixed fee per trade."""
        return self.fee_per_trade
    
    def get_fee_breakdown(self, fill: Fill) -> Dict[str, float]:
        """Get fee breakdown."""
        return {
            "per_trade": self.fee_per_trade,
            "total": self.fee_per_trade
        }


class BasisPointsFeeModel(FeeModel):
    """Basis points fee model with maker/taker differentiation."""
    
    def __init__(self, config: FeeConfig):
        """
        Initialize basis points fee model.
        
        Args:
            config: Fee configuration
        """
        self.config = config
    
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate fee based on notional value and maker/taker status."""
        notional = fill.get_notional()
        
        # Determine fee rate based on maker/taker status
        if fill.is_maker():
            bps_rate = self.config.maker_bps
            per_share_rate = self.config.maker_per_share
        else:
            bps_rate = self.config.taker_bps
            per_share_rate = self.config.taker_per_share
        
        # Calculate basis points fee
        bps_fee = notional * (bps_rate / 10000.0)
        
        # Calculate per-share fee
        per_share_fee = fill.quantity * per_share_rate
        
        # Total fee
        total_fee = bps_fee + per_share_fee
        
        # Apply minimum fee
        total_fee = max(total_fee, self.config.min_fee)
        
        # Apply maximum fee
        if self.config.max_fee is not None:
            total_fee = min(total_fee, self.config.max_fee)
        
        # Round to cents if configured
        if self.config.round_to_cents:
            total_fee = round(total_fee, 2)
        
        return total_fee
    
    def get_fee_breakdown(self, fill: Fill) -> Dict[str, float]:
        """Get detailed fee breakdown."""
        notional = fill.get_notional()
        
        # Determine fee rate based on maker/taker status
        if fill.is_maker():
            bps_rate = self.config.maker_bps
            per_share_rate = self.config.maker_per_share
            fee_type = "maker"
        else:
            bps_rate = self.config.taker_bps
            per_share_rate = self.config.taker_per_share
            fee_type = "taker"
        
        # Calculate basis points fee
        bps_fee = notional * (bps_rate / 10000.0)
        
        # Calculate per-share fee
        per_share_fee = fill.quantity * per_share_rate
        
        # Total before min/max
        base_fee = bps_fee + per_share_fee
        
        # Apply minimum fee
        min_adjustment = max(0, self.config.min_fee - base_fee)
        total_fee = base_fee + min_adjustment
        
        # Apply maximum fee
        max_adjustment = 0
        if self.config.max_fee is not None:
            max_adjustment = min(0, self.config.max_fee - total_fee)
            total_fee = total_fee + max_adjustment
        
        # Round to cents if configured
        if self.config.round_to_cents:
            total_fee = round(total_fee, 2)
            bps_fee = round(bps_fee, 2)
            per_share_fee = round(per_share_fee, 2)
            min_adjustment = round(min_adjustment, 2)
            max_adjustment = round(max_adjustment, 2)
        
        return {
            "notional": notional,
            "fee_type": fee_type,
            "bps_rate": bps_rate,
            "bps_fee": bps_fee,
            "per_share_rate": per_share_rate,
            "per_share_fee": per_share_fee,
            "base_fee": base_fee,
            "min_adjustment": min_adjustment,
            "max_adjustment": max_adjustment,
            "total": total_fee
        }


class TieredFeeModel(FeeModel):
    """Tiered fee model based on volume or notional."""
    
    def __init__(self, tiers: list[tuple[float, FeeConfig]], volume_based: bool = True):
        """
        Initialize tiered fee model.
        
        Args:
            tiers: List of (threshold, FeeConfig) tuples sorted by threshold
            volume_based: If True, use volume for tiering; if False, use notional
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.volume_based = volume_based
        self.volume_tracker: Dict[str, int] = {}  # symbol -> volume
        self.notional_tracker: Dict[str, float] = {}  # symbol -> notional
    
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate fee based on current tier."""
        # Update tracking
        if self.volume_based:
            self.volume_tracker[fill.symbol] = self.volume_tracker.get(fill.symbol, 0) + fill.quantity
            current_value = self.volume_tracker[fill.symbol]
        else:
            self.notional_tracker[fill.symbol] = self.notional_tracker.get(fill.symbol, 0.0) + fill.get_notional()
            current_value = self.notional_tracker[fill.symbol]
        
        # Find appropriate tier
        config = self.tiers[0][1]  # Default to first tier
        for threshold, tier_config in self.tiers:
            if current_value >= threshold:
                config = tier_config
            else:
                break
        
        # Calculate fee using the tier's config
        model = BasisPointsFeeModel(config)
        return model.calculate_fee(fill)
    
    def get_fee_breakdown(self, fill: Fill) -> Dict[str, float]:
        """Get fee breakdown with tier information."""
        # Don't update tracking here - just get current value
        if self.volume_based:
            current_value = self.volume_tracker.get(fill.symbol, 0)
        else:
            current_value = self.notional_tracker.get(fill.symbol, 0.0)
        
        # Find appropriate tier
        config = self.tiers[0][1]  # Default to first tier
        tier_threshold = self.tiers[0][0]
        for threshold, tier_config in self.tiers:
            if current_value >= threshold:
                config = tier_config
                tier_threshold = threshold
            else:
                break
        
        # Calculate fee using the tier's config
        model = BasisPointsFeeModel(config)
        breakdown = model.get_fee_breakdown(fill)
        breakdown["tier_threshold"] = tier_threshold
        breakdown["current_value"] = current_value
        breakdown["tier_based_on"] = "volume" if self.volume_based else "notional"
        
        return breakdown
    
    def reset_tracking(self, symbol: Optional[str] = None) -> None:
        """Reset volume/notional tracking."""
        if symbol is None:
            self.volume_tracker.clear()
            self.notional_tracker.clear()
        else:
            self.volume_tracker.pop(symbol, None)
            self.notional_tracker.pop(symbol, None)


class ExchangeFeeModel(FeeModel):
    """Exchange-specific fee model with complex rules."""
    
    def __init__(self, exchange: str, config: Dict[str, FeeConfig]):
        """
        Initialize exchange-specific fee model.
        
        Args:
            exchange: Exchange name
            config: Dictionary mapping symbol patterns to FeeConfig
        """
        self.exchange = exchange
        self.config = config
        self.default_config = FeeConfig()  # Default config
    
    def _get_config_for_symbol(self, symbol: str) -> FeeConfig:
        """Get fee config for a symbol."""
        # Simple pattern matching (can be extended)
        exchange_config = self.config.get(self.exchange, {})
        for pattern, config in exchange_config.items():
            if pattern == "*" or symbol.startswith(pattern):
                return config
        return self.default_config
    
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate fee using symbol-specific config."""
        config = self._get_config_for_symbol(fill.symbol)
        model = BasisPointsFeeModel(config)
        return model.calculate_fee(fill)
    
    def get_fee_breakdown(self, fill: Fill) -> Dict[str, float]:
        """Get fee breakdown with exchange info."""
        config = self._get_config_for_symbol(fill.symbol)
        model = BasisPointsFeeModel(config)
        breakdown = model.get_fee_breakdown(fill)
        breakdown["exchange"] = self.exchange
        breakdown["symbol_pattern"] = self._get_pattern_for_symbol(fill.symbol)
        return breakdown
    
    def _get_pattern_for_symbol(self, symbol: str) -> str:
        """Get the pattern that matches the symbol."""
        exchange_config = self.config.get(self.exchange, {})
        for pattern in exchange_config.keys():
            if pattern == "*" or symbol.startswith(pattern):
                return pattern
        return "default"


def create_standard_fee_model(
    maker_bps: float = 0.0,
    taker_bps: float = 0.0,
    maker_per_share: float = 0.0,
    taker_per_share: float = 0.0,
    min_fee: float = 0.0,
    max_fee: Optional[float] = None
) -> BasisPointsFeeModel:
    """
    Create a standard basis points fee model.
    
    Args:
        maker_bps: Maker fee in basis points
        taker_bps: Taker fee in basis points
        maker_per_share: Maker fee per share
        taker_per_share: Taker fee per share
        min_fee: Minimum fee
        max_fee: Maximum fee
        
    Returns:
        Configured BasisPointsFeeModel
    """
    config = FeeConfig(
        maker_bps=maker_bps,
        taker_bps=taker_bps,
        maker_per_share=maker_per_share,
        taker_per_share=taker_per_share,
        min_fee=min_fee,
        max_fee=max_fee
    )
    return BasisPointsFeeModel(config)


def create_tiered_volume_fee_model(
    tiers: list[tuple[int, float, float]]  # (volume_threshold, maker_bps, taker_bps)
) -> TieredFeeModel:
    """
    Create a tiered fee model based on volume.
    
    Args:
        tiers: List of (volume_threshold, maker_bps, taker_bps) tuples
        
    Returns:
        Configured TieredFeeModel
    """
    fee_tiers = []
    for volume, maker_bps, taker_bps in tiers:
        config = FeeConfig(maker_bps=maker_bps, taker_bps=taker_bps)
        fee_tiers.append((volume, config))
    
    return TieredFeeModel(fee_tiers, volume_based=True)


def create_exchange_fee_model(exchange: str) -> ExchangeFeeModel:
    """
    Create an exchange-specific fee model with common configurations.
    
    Args:
        exchange: Exchange name
        
    Returns:
        Configured ExchangeFeeModel
    """
    configs = {
        "NASDAQ": {
            "*": FeeConfig(maker_bps=0.0, taker_bps=0.0),  # No fees for simplicity
        },
        "NYSE": {
            "*": FeeConfig(maker_bps=0.0, taker_bps=0.0),
        },
        "BATS": {
            "*": FeeConfig(maker_bps=0.0, taker_bps=0.0),
        },
        "IEX": {
            "*": FeeConfig(maker_bps=0.0, taker_bps=0.0),
        }
    }
    
    return ExchangeFeeModel(exchange, configs.get(exchange, {"*": FeeConfig()}))