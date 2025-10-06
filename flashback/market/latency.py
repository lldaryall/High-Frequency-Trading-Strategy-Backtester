"""Latency models for order processing delays."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import random
import numpy as np
from enum import Enum

from .orders import Order, OrderType, OrderSide


class LatencyType(Enum):
    """Types of latency to model."""
    SUBMISSION = "submission"  # Order submission latency
    CANCELLATION = "cancellation"  # Order cancellation latency
    MARKET_DATA = "market_data"  # Market data processing latency
    EXECUTION = "execution"  # Order execution latency


@dataclass
class LatencyConfig:
    """Configuration for latency models."""
    constant_ns: int = 0  # Constant latency in nanoseconds
    mean_ns: int = 0  # Mean of random component
    std_ns: int = 0  # Standard deviation of random component
    min_ns: int = 0  # Minimum latency
    max_ns: Optional[int] = None  # Maximum latency (None for no limit)
    distribution: str = "normal"  # Distribution type: "normal", "exponential", "uniform"
    seed: Optional[int] = None  # Random seed for reproducibility


class LatencyModel(ABC):
    """Abstract base class for latency models."""
    
    @abstractmethod
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """
        Calculate latency for an order.
        
        Args:
            order: Order to calculate latency for
            latency_type: Type of latency to calculate
            
        Returns:
            Latency in nanoseconds
        """
        pass
    
    @abstractmethod
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """
        Get detailed latency breakdown.
        
        Args:
            order: Order to calculate latency for
            latency_type: Type of latency to calculate
            
        Returns:
            Dictionary with latency components
        """
        pass


class FixedLatencyModel(LatencyModel):
    """Fixed latency model with constant delay."""
    
    def __init__(self, latency_ns: int = 0):
        """
        Initialize fixed latency model.
        
        Args:
            latency_ns: Fixed latency in nanoseconds
        """
        self.latency_ns = latency_ns
    
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """Calculate fixed latency."""
        return self.latency_ns
    
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """Get latency breakdown."""
        return {
            "constant": self.latency_ns,
            "random": 0,
            "total": self.latency_ns,
            "latency_type": latency_type.value
        }


class ConstantLatencyModel(LatencyModel):
    """Constant latency model with different delays per latency type."""
    
    def __init__(self, latencies: Dict[LatencyType, int]):
        """
        Initialize constant latency model.
        
        Args:
            latencies: Dictionary mapping latency types to delays in nanoseconds
        """
        self.latencies = latencies
    
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """Calculate constant latency for the given type."""
        return self.latencies.get(latency_type, 0)
    
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """Get latency breakdown."""
        latency = self.latencies.get(latency_type, 0)
        return {
            "constant": latency,
            "random": 0,
            "total": latency,
            "latency_type": latency_type.value
        }


class RandomLatencyModel(LatencyModel):
    """Random latency model with configurable distribution."""
    
    def __init__(self, config: LatencyConfig):
        """
        Initialize random latency model.
        
        Args:
            config: Latency configuration
        """
        self.config = config
        self.rng = random.Random()
        self._call_count = 0
        
        # Initialize numpy random state if using numpy distributions
        if config.distribution in ["normal", "exponential"]:
            np.random.seed(config.seed)
    
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """Calculate random latency."""
        # Constant component
        total_latency = self.config.constant_ns
        
        # Random component - use a deterministic seed based on order properties
        # This ensures the same order always gets the same latency
        if self.config.seed is not None:
            # Create a deterministic seed based on order properties and call count
            order_seed = hash((order.order_id, order.timestamp, order.symbol, order.quantity)) % (2**32)
            # For deterministic replay, don't include model instance ID
            combined_seed = (self.config.seed + order_seed + self._call_count) % (2**32)
            
            if self.config.distribution in ["normal", "exponential"]:
                np.random.seed(combined_seed)
            else:
                self.rng.seed(combined_seed)
            
            self._call_count += 1
        
        if self.config.distribution == "normal":
            random_latency = np.random.normal(self.config.mean_ns, self.config.std_ns)
        elif self.config.distribution == "exponential":
            random_latency = np.random.exponential(self.config.mean_ns)
        elif self.config.distribution == "uniform":
            min_val = max(0, self.config.mean_ns - self.config.std_ns)
            max_val = self.config.mean_ns + self.config.std_ns
            random_latency = self.rng.uniform(min_val, max_val)
        else:
            random_latency = 0
        
        # Ensure random component doesn't make total less than constant
        random_component = max(0, int(random_latency))
        total_latency += random_component
        
        # Apply bounds
        total_latency = max(total_latency, self.config.min_ns)
        if self.config.max_ns is not None:
            total_latency = min(total_latency, self.config.max_ns)
        
        return total_latency
    
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """Get latency breakdown."""
        constant = self.config.constant_ns
        
        # Calculate random component (same as in calculate_latency)
        if self.config.seed is not None:
            # Use the same seeding logic as calculate_latency
            order_seed = hash((order.order_id, order.timestamp, order.symbol, order.quantity)) % (2**32)
            # For deterministic replay, don't include model instance ID
            combined_seed = (self.config.seed + order_seed + self._call_count) % (2**32)
            
            if self.config.distribution in ["normal", "exponential"]:
                np.random.seed(combined_seed)
            else:
                self.rng.seed(combined_seed)
        
        if self.config.distribution == "normal":
            random_latency = np.random.normal(self.config.mean_ns, self.config.std_ns)
        elif self.config.distribution == "exponential":
            random_latency = np.random.exponential(self.config.mean_ns)
        elif self.config.distribution == "uniform":
            min_val = max(0, self.config.mean_ns - self.config.std_ns)
            max_val = self.config.mean_ns + self.config.std_ns
            random_latency = self.rng.uniform(min_val, max_val)
        else:
            random_latency = 0
        
        # Ensure random component doesn't make total less than constant
        random_component = max(0, int(random_latency))
        total = constant + random_component
        
        # Apply bounds
        total = max(total, self.config.min_ns)
        if self.config.max_ns is not None:
            total = min(total, self.config.max_ns)
        
        return {
            "constant": constant,
            "random": random_component,
            "total": total,
            "latency_type": latency_type.value,
            "distribution": self.config.distribution,
            "mean": self.config.mean_ns,
            "std": self.config.std_ns,
            "min": self.config.min_ns,
            "max": self.config.max_ns
        }


class AdaptiveLatencyModel(LatencyModel):
    """Adaptive latency model that adjusts based on order characteristics."""
    
    def __init__(self, base_config: LatencyConfig, multipliers: Dict[str, float] = None):
        """
        Initialize adaptive latency model.
        
        Args:
            base_config: Base latency configuration
            multipliers: Multipliers for different order characteristics
        """
        self.base_config = base_config
        self.multipliers = multipliers or {
            "market_order": 1.0,
            "limit_order": 1.0,
            "large_order": 1.5,  # Orders > 1000 shares
            "small_order": 0.8,  # Orders < 100 shares
            "ioc_order": 0.9,    # IOC orders
            "fok_order": 0.9,    # FOK orders
            "day_order": 1.1,    # Day orders
        }
        self.rng = random.Random(base_config.seed)
        
        if base_config.distribution in ["normal", "exponential"]:
            np.random.seed(base_config.seed)
    
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """Calculate adaptive latency."""
        # Start with base latency
        base_latency = self.base_config.constant_ns
        
        # Apply multipliers based on order characteristics
        multiplier = 1.0
        
        if order.is_market():
            multiplier *= self.multipliers.get("market_order", 1.0)
        elif order.is_limit():
            multiplier *= self.multipliers.get("limit_order", 1.0)
        
        if order.quantity > 1000:
            multiplier *= self.multipliers.get("large_order", 1.0)
        elif order.quantity < 100:
            multiplier *= self.multipliers.get("small_order", 1.0)
        
        if order.is_ioc():
            multiplier *= self.multipliers.get("ioc_order", 1.0)
        elif order.is_fok():
            multiplier *= self.multipliers.get("fok_order", 1.0)
        elif order.is_day_order():
            multiplier *= self.multipliers.get("day_order", 1.0)
        
        # Apply multiplier to constant component
        total_latency = int(base_latency * multiplier)
        
        # Add random component
        if self.base_config.distribution == "normal":
            random_latency = np.random.normal(
                self.base_config.mean_ns * multiplier,
                self.base_config.std_ns * multiplier
            )
        elif self.base_config.distribution == "exponential":
            random_latency = np.random.exponential(self.base_config.mean_ns * multiplier)
        elif self.base_config.distribution == "uniform":
            min_val = max(0, (self.base_config.mean_ns - self.base_config.std_ns) * multiplier)
            max_val = (self.base_config.mean_ns + self.base_config.std_ns) * multiplier
            random_latency = self.rng.uniform(min_val, max_val)
        else:
            random_latency = 0
        
        total_latency += int(random_latency)
        
        # Apply bounds
        total_latency = max(total_latency, self.base_config.min_ns)
        if self.base_config.max_ns is not None:
            total_latency = min(total_latency, self.base_config.max_ns)
        
        return total_latency
    
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """Get detailed latency breakdown."""
        # Calculate multiplier
        multiplier = 1.0
        applied_multipliers = {}
        
        if order.is_market():
            mult = self.multipliers.get("market_order", 1.0)
            multiplier *= mult
            applied_multipliers["market_order"] = mult
        elif order.is_limit():
            mult = self.multipliers.get("limit_order", 1.0)
            multiplier *= mult
            applied_multipliers["limit_order"] = mult
        
        if order.quantity > 1000:
            mult = self.multipliers.get("large_order", 1.0)
            multiplier *= mult
            applied_multipliers["large_order"] = mult
        elif order.quantity < 100:
            mult = self.multipliers.get("small_order", 1.0)
            multiplier *= mult
            applied_multipliers["small_order"] = mult
        
        if order.is_ioc():
            mult = self.multipliers.get("ioc_order", 1.0)
            multiplier *= mult
            applied_multipliers["ioc_order"] = mult
        elif order.is_fok():
            mult = self.multipliers.get("fok_order", 1.0)
            multiplier *= mult
            applied_multipliers["fok_order"] = mult
        elif order.is_day_order():
            mult = self.multipliers.get("day_order", 1.0)
            multiplier *= mult
            applied_multipliers["day_order"] = mult
        
        # Calculate components
        constant = int(self.base_config.constant_ns * multiplier)
        
        if self.base_config.distribution == "normal":
            random_latency = np.random.normal(
                self.base_config.mean_ns * multiplier,
                self.base_config.std_ns * multiplier
            )
        elif self.base_config.distribution == "exponential":
            random_latency = np.random.exponential(self.base_config.mean_ns * multiplier)
        elif self.base_config.distribution == "uniform":
            min_val = max(0, (self.base_config.mean_ns - self.base_config.std_ns) * multiplier)
            max_val = (self.base_config.mean_ns + self.base_config.std_ns) * multiplier
            random_latency = self.rng.uniform(min_val, max_val)
        else:
            random_latency = 0
        
        random_component = int(random_latency)
        total = constant + random_component
        
        # Apply bounds
        total = max(total, self.base_config.min_ns)
        if self.base_config.max_ns is not None:
            total = min(total, self.base_config.max_ns)
        
        return {
            "constant": constant,
            "random": random_component,
            "total": total,
            "latency_type": latency_type.value,
            "multiplier": multiplier,
            "applied_multipliers": applied_multipliers,
            "distribution": self.base_config.distribution,
            "base_mean": self.base_config.mean_ns,
            "base_std": self.base_config.std_ns,
            "min": self.base_config.min_ns,
            "max": self.base_config.max_ns
        }


class NetworkLatencyModel(LatencyModel):
    """Network latency model with distance-based delays."""
    
    def __init__(self, base_latency_ns: int, distance_multiplier: float = 0.001):
        """
        Initialize network latency model.
        
        Args:
            base_latency_ns: Base latency in nanoseconds
            distance_multiplier: Multiplier for distance-based latency
        """
        self.base_latency_ns = base_latency_ns
        self.distance_multiplier = distance_multiplier
        self.rng = random.Random()
    
    def calculate_latency(self, order: Order, latency_type: LatencyType) -> int:
        """Calculate network latency."""
        # Base latency
        latency = self.base_latency_ns
        
        # Add distance-based latency (simplified model)
        # In reality, this would be based on actual network topology
        distance_latency = int(order.quantity * self.distance_multiplier)
        latency += distance_latency
        
        # Add some random jitter (but ensure total doesn't go below base)
        jitter = self.rng.randint(-1000, 1000)  # ±1 microsecond
        latency += jitter
        
        return max(latency, self.base_latency_ns)  # Ensure at least base latency
    
    def get_latency_breakdown(self, order: Order, latency_type: LatencyType) -> Dict[str, Any]:
        """Get latency breakdown."""
        base = self.base_latency_ns
        distance = int(order.quantity * self.distance_multiplier)
        jitter = self.rng.randint(-1000, 1000)
        total = base + distance + jitter
        
        return {
            "base": base,
            "distance": distance,
            "jitter": jitter,
            "total": max(total, 0),
            "latency_type": latency_type.value,
            "distance_multiplier": self.distance_multiplier
        }


def create_standard_latency_model(
    constant_ns: int = 1000,
    mean_ns: int = 500,
    std_ns: int = 200,
    distribution: str = "normal",
    seed: Optional[int] = None
) -> RandomLatencyModel:
    """
    Create a standard latency model.
    
    Args:
        constant_ns: Constant latency in nanoseconds
        mean_ns: Mean of random component
        std_ns: Standard deviation of random component
        distribution: Distribution type
        seed: Random seed
        
    Returns:
        Configured RandomLatencyModel
    """
    config = LatencyConfig(
        constant_ns=constant_ns,
        mean_ns=mean_ns,
        std_ns=std_ns,
        distribution=distribution,
        seed=seed
    )
    return RandomLatencyModel(config)


def create_hft_latency_model(seed: Optional[int] = None) -> AdaptiveLatencyModel:
    """
    Create an HFT-optimized latency model.
    
    Args:
        seed: Random seed
        
    Returns:
        Configured AdaptiveLatencyModel
    """
    config = LatencyConfig(
        constant_ns=100,  # Very low base latency
        mean_ns=50,       # Low random component
        std_ns=20,        # Low variance
        min_ns=50,        # Minimum 50ns
        max_ns=500,       # Maximum 500ns
        distribution="normal",
        seed=seed
    )
    
    multipliers = {
        "market_order": 0.8,   # Market orders are fastest
        "limit_order": 1.0,    # Limit orders normal
        "large_order": 1.2,    # Large orders slightly slower
        "small_order": 0.9,    # Small orders slightly faster
        "ioc_order": 0.7,      # IOC orders very fast
        "fok_order": 0.7,      # FOK orders very fast
        "day_order": 1.1,      # Day orders slightly slower
    }
    
    return AdaptiveLatencyModel(config, multipliers)


def create_retail_latency_model(seed: Optional[int] = None) -> RandomLatencyModel:
    """
    Create a retail trading latency model.
    
    Args:
        seed: Random seed
        
    Returns:
        Configured RandomLatencyModel
    """
    config = LatencyConfig(
        constant_ns=10000,  # 10 microseconds base
        mean_ns=5000,       # 5 microseconds random
        std_ns=2000,        # 2 microseconds std
        min_ns=5000,        # Minimum 5 microseconds
        max_ns=50000,       # Maximum 50 microseconds
        distribution="normal",
        seed=seed
    )
    return RandomLatencyModel(config)