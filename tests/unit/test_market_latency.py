"""Unit tests for market latency functionality."""

import pytest
import numpy as np
from flashback.market.latency import (
    LatencyConfig, LatencyType, LatencyModel, FixedLatencyModel,
    ConstantLatencyModel, RandomLatencyModel, AdaptiveLatencyModel,
    NetworkLatencyModel, create_standard_latency_model,
    create_hft_latency_model, create_retail_latency_model
)
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce


class TestLatencyConfig:
    """Test LatencyConfig dataclass."""
    
    def test_latency_config_creation(self):
        """Test basic latency config creation."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            min_ns=100,
            max_ns=2000,
            distribution="normal",
            seed=42
        )
        
        assert config.constant_ns == 1000
        assert config.mean_ns == 500
        assert config.std_ns == 200
        assert config.min_ns == 100
        assert config.max_ns == 2000
        assert config.distribution == "normal"
        assert config.seed == 42


class TestFixedLatencyModel:
    """Test FixedLatencyModel functionality."""
    
    def test_fixed_latency_calculation(self):
        """Test fixed latency calculation."""
        model = FixedLatencyModel(latency_ns=1000)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency == 1000
        
        breakdown = model.get_latency_breakdown(order, LatencyType.SUBMISSION)
        assert breakdown["constant"] == 1000
        assert breakdown["random"] == 0
        assert breakdown["total"] == 1000
        assert breakdown["latency_type"] == "submission"


class TestConstantLatencyModel:
    """Test ConstantLatencyModel functionality."""
    
    def test_constant_latency_calculation(self):
        """Test constant latency calculation for different types."""
        latencies = {
            LatencyType.SUBMISSION: 1000,
            LatencyType.CANCELLATION: 500,
            LatencyType.MARKET_DATA: 200,
            LatencyType.EXECUTION: 300
        }
        model = ConstantLatencyModel(latencies)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Test different latency types
        submission_latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert submission_latency == 1000
        
        cancellation_latency = model.calculate_latency(order, LatencyType.CANCELLATION)
        assert cancellation_latency == 500
        
        market_data_latency = model.calculate_latency(order, LatencyType.MARKET_DATA)
        assert market_data_latency == 200
        
        execution_latency = model.calculate_latency(order, LatencyType.EXECUTION)
        assert execution_latency == 300
    
    def test_unknown_latency_type(self):
        """Test unknown latency type returns 0."""
        latencies = {LatencyType.SUBMISSION: 1000}
        model = ConstantLatencyModel(latencies)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Unknown type should return 0
        latency = model.calculate_latency(order, LatencyType.CANCELLATION)
        assert latency == 0


class TestRandomLatencyModel:
    """Test RandomLatencyModel functionality."""
    
    def test_normal_distribution_latency(self):
        """Test normal distribution latency calculation."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        model = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Test multiple calculations with same seed
        latencies = []
        for _ in range(10):
            latency = model.calculate_latency(order, LatencyType.SUBMISSION)
            latencies.append(latency)
        
        # All should be different due to random component
        assert len(set(latencies)) > 1
        
        # All should be >= constant_ns
        assert all(latency >= 1000 for latency in latencies)
    
    def test_exponential_distribution_latency(self):
        """Test exponential distribution latency calculation."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="exponential",
            seed=42
        )
        model = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency >= 1000  # Should be >= constant_ns
    
    def test_uniform_distribution_latency(self):
        """Test uniform distribution latency calculation."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="uniform",
            seed=42
        )
        model = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency >= 1000  # Should be >= constant_ns
    
    def test_min_max_bounds(self):
        """Test min/max bounds application."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            min_ns=1500,  # Higher than constant + mean
            max_ns=2000,
            distribution="normal",
            seed=42
        )
        model = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Test multiple calculations
        for _ in range(100):
            latency = model.calculate_latency(order, LatencyType.SUBMISSION)
            assert 1500 <= latency <= 2000
    
    def test_deterministic_with_seed(self):
        """Test that latency is deterministic with same seed."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        
        # Create two models with same seed
        model1 = RandomLatencyModel(config)
        model2 = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Should produce same results
        latency1 = model1.calculate_latency(order, LatencyType.SUBMISSION)
        latency2 = model2.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency1 == latency2


class TestAdaptiveLatencyModel:
    """Test AdaptiveLatencyModel functionality."""
    
    def test_adaptive_latency_calculation(self):
        """Test adaptive latency calculation based on order characteristics."""
        base_config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        
        multipliers = {
            "market_order": 0.8,
            "limit_order": 1.0,
            "large_order": 1.5,
            "small_order": 0.9,
            "ioc_order": 0.7,
            "fok_order": 0.7,
            "day_order": 1.1,
        }
        
        model = AdaptiveLatencyModel(base_config, multipliers)
        
        # Test market order (should be faster)
        market_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.MARKET)
        market_latency = model.calculate_latency(market_order, LatencyType.SUBMISSION)
        
        # Test limit order (should be normal)
        limit_order = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        limit_latency = model.calculate_latency(limit_order, LatencyType.SUBMISSION)
        
        # Market order should generally be faster (but not guaranteed due to randomness)
        # We'll just check that the breakdown shows the correct multiplier
        breakdown = model.get_latency_breakdown(market_order, LatencyType.SUBMISSION)
        assert abs(breakdown["multiplier"] - 0.8) < 0.1
        assert "market_order" in breakdown["applied_multipliers"]
    
    def test_large_order_multiplier(self):
        """Test large order multiplier application."""
        base_config = LatencyConfig(constant_ns=1000, mean_ns=0, std_ns=0, seed=42)
        multipliers = {"large_order": 2.0}
        model = AdaptiveLatencyModel(base_config, multipliers)
        
        # Large order (>1000 shares)
        large_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 2000, TimeInForce.DAY, OrderType.LIMIT)
        breakdown = model.get_latency_breakdown(large_order, LatencyType.SUBMISSION)
        
        assert breakdown["multiplier"] == 2.0
        assert "large_order" in breakdown["applied_multipliers"]
        assert breakdown["applied_multipliers"]["large_order"] == 2.0
    
    def test_small_order_multiplier(self):
        """Test small order multiplier application."""
        base_config = LatencyConfig(constant_ns=1000, mean_ns=0, std_ns=0, seed=42)
        multipliers = {"small_order": 0.5}
        model = AdaptiveLatencyModel(base_config, multipliers)
        
        # Small order (<100 shares)
        small_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 50, TimeInForce.DAY, OrderType.LIMIT)
        breakdown = model.get_latency_breakdown(small_order, LatencyType.SUBMISSION)
        
        assert breakdown["multiplier"] == 0.5
        assert "small_order" in breakdown["applied_multipliers"]
    
    def test_ioc_fok_multipliers(self):
        """Test IOC and FOK order multipliers."""
        base_config = LatencyConfig(constant_ns=1000, mean_ns=0, std_ns=0, seed=42)
        multipliers = {"ioc_order": 0.6, "fok_order": 0.7}
        model = AdaptiveLatencyModel(base_config, multipliers)
        
        # IOC order
        ioc_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.IOC, OrderType.LIMIT)
        ioc_breakdown = model.get_latency_breakdown(ioc_order, LatencyType.SUBMISSION)
        assert ioc_breakdown["multiplier"] == 0.6
        assert "ioc_order" in ioc_breakdown["applied_multipliers"]
        
        # FOK order
        fok_order = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.FOK, OrderType.LIMIT)
        fok_breakdown = model.get_latency_breakdown(fok_order, LatencyType.SUBMISSION)
        assert fok_breakdown["multiplier"] == 0.7
        assert "fok_order" in fok_breakdown["applied_multipliers"]
    
    def test_multiple_multipliers(self):
        """Test multiple multipliers applied together."""
        base_config = LatencyConfig(constant_ns=1000, mean_ns=0, std_ns=0, seed=42)
        multipliers = {
            "market_order": 0.8,
            "large_order": 1.5,
            "ioc_order": 0.7,
        }
        model = AdaptiveLatencyModel(base_config, multipliers)
        
        # Market IOC order with large quantity
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 2000, TimeInForce.IOC, OrderType.MARKET)
        breakdown = model.get_latency_breakdown(order, LatencyType.SUBMISSION)
        
        expected_multiplier = 0.8 * 1.5 * 0.7  # 0.84
        assert abs(breakdown["multiplier"] - expected_multiplier) < 0.001
        assert "market_order" in breakdown["applied_multipliers"]
        assert "large_order" in breakdown["applied_multipliers"]
        assert "ioc_order" in breakdown["applied_multipliers"]


class TestNetworkLatencyModel:
    """Test NetworkLatencyModel functionality."""
    
    def test_network_latency_calculation(self):
        """Test network latency calculation."""
        model = NetworkLatencyModel(base_latency_ns=1000, distance_multiplier=0.001)
        
        # Small order
        small_order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        small_latency = model.calculate_latency(small_order, LatencyType.SUBMISSION)
        
        # Large order
        large_order = Order("2", 1000, "AAPL", OrderSide.BUY, 150.0, 10000, TimeInForce.DAY, OrderType.LIMIT)
        large_latency = model.calculate_latency(large_order, LatencyType.SUBMISSION)
        
        # Large order should generally have higher latency (but not guaranteed due to jitter)
        # Check that the distance component is higher for large order
        small_breakdown = model.get_latency_breakdown(small_order, LatencyType.SUBMISSION)
        large_breakdown = model.get_latency_breakdown(large_order, LatencyType.SUBMISSION)
        assert large_breakdown["distance"] > small_breakdown["distance"]
        
        # Check breakdown
        breakdown = model.get_latency_breakdown(large_order, LatencyType.SUBMISSION)
        assert breakdown["base"] == 1000
        assert breakdown["distance"] == 10  # 10000 * 0.001
        assert breakdown["distance_multiplier"] == 0.001
    
    def test_jitter_component(self):
        """Test that jitter is applied."""
        model = NetworkLatencyModel(base_latency_ns=1000, distance_multiplier=0.0)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Test multiple calculations
        latencies = []
        for _ in range(100):
            latency = model.calculate_latency(order, LatencyType.SUBMISSION)
            latencies.append(latency)
        
        # Should have variation due to jitter
        assert len(set(latencies)) > 1
        
        # All should be >= base latency
        assert all(latency >= 1000 for latency in latencies)


class TestFactoryFunctions:
    """Test factory functions for creating latency models."""
    
    def test_create_standard_latency_model(self):
        """Test creating standard latency model."""
        model = create_standard_latency_model(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        
        assert isinstance(model, RandomLatencyModel)
        assert model.config.constant_ns == 1000
        assert model.config.mean_ns == 500
        assert model.config.std_ns == 200
        assert model.config.distribution == "normal"
        assert model.config.seed == 42
    
    def test_create_hft_latency_model(self):
        """Test creating HFT latency model."""
        model = create_hft_latency_model(seed=42)
        
        assert isinstance(model, AdaptiveLatencyModel)
        assert model.base_config.constant_ns == 100
        assert model.base_config.mean_ns == 50
        assert model.base_config.std_ns == 20
        assert model.base_config.min_ns == 50
        assert model.base_config.max_ns == 500
        assert "market_order" in model.multipliers
        assert "ioc_order" in model.multipliers
        assert model.multipliers["market_order"] == 0.8
        assert model.multipliers["ioc_order"] == 0.7
    
    def test_create_retail_latency_model(self):
        """Test creating retail latency model."""
        model = create_retail_latency_model(seed=42)
        
        assert isinstance(model, RandomLatencyModel)
        assert model.config.constant_ns == 10000  # 10 microseconds
        assert model.config.mean_ns == 5000      # 5 microseconds
        assert model.config.std_ns == 2000       # 2 microseconds
        assert model.config.min_ns == 5000       # 5 microseconds
        assert model.config.max_ns == 50000      # 50 microseconds


class TestLatencyModelIntegration:
    """Test latency model integration scenarios."""
    
    def test_deterministic_replay_with_seed(self):
        """Test that latency models provide deterministic replay with seeded RNG."""
        config = LatencyConfig(
            constant_ns=1000,
            mean_ns=500,
            std_ns=200,
            distribution="normal",
            seed=42
        )
        
        # Create two models with same seed
        model1 = RandomLatencyModel(config)
        model2 = RandomLatencyModel(config)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        # Should produce identical results
        latency1 = model1.calculate_latency(order, LatencyType.SUBMISSION)
        latency2 = model2.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency1 == latency2
        
        breakdown1 = model1.get_latency_breakdown(order, LatencyType.SUBMISSION)
        breakdown2 = model2.get_latency_breakdown(order, LatencyType.SUBMISSION)
        assert breakdown1["random"] == breakdown2["random"]
    
    def test_latency_type_consistency(self):
        """Test that latency type is consistent in breakdown."""
        model = FixedLatencyModel(1000)
        
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        for latency_type in LatencyType:
            breakdown = model.get_latency_breakdown(order, latency_type)
            assert breakdown["latency_type"] == latency_type.value
    
    def test_edge_cases(self):
        """Test latency model edge cases."""
        # Zero latency
        model = FixedLatencyModel(0)
        order = Order("1", 1000, "AAPL", OrderSide.BUY, 150.0, 100, TimeInForce.DAY, OrderType.LIMIT)
        
        latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency == 0
        
        # Very high latency
        model = FixedLatencyModel(1000000)  # 1ms
        latency = model.calculate_latency(order, LatencyType.SUBMISSION)
        assert latency == 1000000


if __name__ == "__main__":
    pytest.main([__file__])
