"""Unit tests for market fees functionality."""

import pytest
from flashback.market.fees import (
    FeeConfig, FeeModel, PerTradeFeeModel, BasisPointsFeeModel,
    TieredFeeModel, ExchangeFeeModel, create_standard_fee_model,
    create_tiered_volume_fee_model, create_exchange_fee_model
)
from flashback.market.orders import Fill, OrderSide


class TestFeeConfig:
    """Test FeeConfig dataclass."""
    
    def test_fee_config_creation(self):
        """Test basic fee config creation."""
        config = FeeConfig(
            maker_bps=1.0,
            taker_bps=2.0,
            maker_per_share=0.001,
            taker_per_share=0.002,
            min_fee=0.01,
            max_fee=10.0
        )
        
        assert config.maker_bps == 1.0
        assert config.taker_bps == 2.0
        assert config.maker_per_share == 0.001
        assert config.taker_per_share == 0.002
        assert config.min_fee == 0.01
        assert config.max_fee == 10.0
        assert config.round_to_cents is True


class TestPerTradeFeeModel:
    """Test PerTradeFeeModel functionality."""
    
    def test_per_trade_fee_calculation(self):
        """Test per-trade fee calculation."""
        model = PerTradeFeeModel(fee_per_trade=0.5)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100)
        
        fee = model.calculate_fee(fill)
        assert fee == 0.5
        
        breakdown = model.get_fee_breakdown(fill)
        assert breakdown["per_trade"] == 0.5
        assert breakdown["total"] == 0.5


class TestBasisPointsFeeModel:
    """Test BasisPointsFeeModel functionality."""
    
    def test_maker_fee_calculation(self):
        """Test maker fee calculation."""
        config = FeeConfig(maker_bps=1.0, maker_per_share=0.001)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        fee = model.calculate_fee(fill)
        expected = (15000.0 * 0.0001) + (100 * 0.001)  # 1.5 + 0.1 = 1.6
        assert fee == expected
        
        breakdown = model.get_fee_breakdown(fill)
        assert breakdown["fee_type"] == "maker"
        assert breakdown["bps_rate"] == 1.0
        assert breakdown["per_share_rate"] == 0.001
    
    def test_taker_fee_calculation(self):
        """Test taker fee calculation."""
        config = FeeConfig(taker_bps=2.0, taker_per_share=0.002)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="TAKER")
        
        fee = model.calculate_fee(fill)
        expected = (15000.0 * 0.0002) + (100 * 0.002)  # 3.0 + 0.2 = 3.2
        assert fee == expected
        
        breakdown = model.get_fee_breakdown(fill)
        assert breakdown["fee_type"] == "taker"
        assert breakdown["bps_rate"] == 2.0
        assert breakdown["per_share_rate"] == 0.002
    
    def test_minimum_fee(self):
        """Test minimum fee application."""
        config = FeeConfig(maker_bps=0.1, min_fee=1.0)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        fee = model.calculate_fee(fill)
        assert fee == 1.0  # Minimum fee applied
        
        breakdown = model.get_fee_breakdown(fill)
        assert breakdown["min_adjustment"] > 0
    
    def test_maximum_fee(self):
        """Test maximum fee application."""
        config = FeeConfig(maker_bps=10.0, max_fee=5.0)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        fee = model.calculate_fee(fill)
        assert fee == 5.0  # Maximum fee applied
        
        breakdown = model.get_fee_breakdown(fill)
        assert breakdown["max_adjustment"] < 0
    
    def test_rounding_to_cents(self):
        """Test rounding to cents."""
        config = FeeConfig(maker_bps=1.0, round_to_cents=True)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        fee = model.calculate_fee(fill)
        assert fee == round(fee, 2)  # Should be rounded to 2 decimal places


class TestTieredFeeModel:
    """Test TieredFeeModel functionality."""
    
    def test_volume_based_tiering(self):
        """Test volume-based tiering."""
        tiers = [
            (0, FeeConfig(maker_bps=2.0, taker_bps=3.0)),      # 0-999 shares
            (1000, FeeConfig(maker_bps=1.5, taker_bps=2.5)),   # 1000-4999 shares
            (5000, FeeConfig(maker_bps=1.0, taker_bps=2.0)),   # 5000+ shares
        ]
        model = TieredFeeModel(tiers, volume_based=True)
        
        # First fill - should use tier 0 (0-999 shares)
        fill1 = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 500, maker_taker="MAKER")
        fee1 = model.calculate_fee(fill1)
        breakdown1 = model.get_fee_breakdown(fill1)
        assert breakdown1["tier_threshold"] == 0  # First tier (0-999)
        assert breakdown1["current_value"] == 500
        
        # Second fill - should use tier 1
        fill2 = Fill("3", "4", 2000, "AAPL", OrderSide.BUY, 150.0, 600, maker_taker="MAKER")
        fee2 = model.calculate_fee(fill2)
        breakdown2 = model.get_fee_breakdown(fill2)
        assert breakdown2["tier_threshold"] == 1000
        assert breakdown2["current_value"] == 1100
        
        # Third fill - should use tier 2
        fill3 = Fill("5", "6", 3000, "AAPL", OrderSide.BUY, 150.0, 5000, maker_taker="MAKER")
        fee3 = model.calculate_fee(fill3)
        breakdown3 = model.get_fee_breakdown(fill3)
        assert breakdown3["tier_threshold"] == 5000
        assert breakdown3["current_value"] == 6100
    
    def test_notional_based_tiering(self):
        """Test notional-based tiering."""
        tiers = [
            (0, FeeConfig(maker_bps=2.0, taker_bps=3.0)),           # 0-9999 notional
            (10000, FeeConfig(maker_bps=1.5, taker_bps=2.5)),       # 10000-49999 notional
            (50000, FeeConfig(maker_bps=1.0, taker_bps=2.0)),       # 50000+ notional
        ]
        model = TieredFeeModel(tiers, volume_based=False)
        
        # First fill - should use tier 0
        fill1 = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 50, maker_taker="MAKER")
        breakdown1 = model.get_fee_breakdown(fill1)
        assert breakdown1["tier_threshold"] == 0
        assert breakdown1["tier_based_on"] == "notional"
    
    def test_reset_tracking(self):
        """Test resetting volume/notional tracking."""
        tiers = [(0, FeeConfig(maker_bps=2.0, taker_bps=3.0))]
        model = TieredFeeModel(tiers, volume_based=True)
        
        # Add some volume
        fill1 = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 500, maker_taker="MAKER")
        model.calculate_fee(fill1)
        
        assert model.volume_tracker["AAPL"] == 500
        
        # Reset tracking
        model.reset_tracking("AAPL")
        assert "AAPL" not in model.volume_tracker
        
        # Reset all
        model.reset_tracking()
        assert len(model.volume_tracker) == 0


class TestExchangeFeeModel:
    """Test ExchangeFeeModel functionality."""
    
    def test_exchange_specific_fees(self):
        """Test exchange-specific fee calculation."""
        configs = {
            "NASDAQ": {
                "AAPL": FeeConfig(maker_bps=0.0, taker_bps=0.0),
                "*": FeeConfig(maker_bps=1.0, taker_bps=2.0),
            },
            "NYSE": {
                "*": FeeConfig(maker_bps=0.5, taker_bps=1.5),
            }
        }
        model = ExchangeFeeModel("NASDAQ", configs)
        
        # AAPL should use specific config
        fill1 = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        breakdown1 = model.get_fee_breakdown(fill1)
        assert breakdown1["exchange"] == "NASDAQ"
        assert breakdown1["symbol_pattern"] == "AAPL"
        assert breakdown1["bps_rate"] == 0.0
        
        # MSFT should use wildcard config
        fill2 = Fill("3", "4", 2000, "MSFT", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        breakdown2 = model.get_fee_breakdown(fill2)
        assert breakdown2["symbol_pattern"] == "*"
        assert breakdown2["bps_rate"] == 1.0


class TestFactoryFunctions:
    """Test factory functions for creating fee models."""
    
    def test_create_standard_fee_model(self):
        """Test creating standard fee model."""
        model = create_standard_fee_model(
            maker_bps=1.0,
            taker_bps=2.0,
            maker_per_share=0.001,
            taker_per_share=0.002,
            min_fee=0.01,
            max_fee=10.0
        )
        
        assert isinstance(model, BasisPointsFeeModel)
        assert model.config.maker_bps == 1.0
        assert model.config.taker_bps == 2.0
        assert model.config.maker_per_share == 0.001
        assert model.config.taker_per_share == 0.002
        assert model.config.min_fee == 0.01
        assert model.config.max_fee == 10.0
    
    def test_create_tiered_volume_fee_model(self):
        """Test creating tiered volume fee model."""
        tiers = [
            (0, 2.0, 3.0),      # 0-999 shares: 2bps maker, 3bps taker
            (1000, 1.5, 2.5),   # 1000-4999 shares: 1.5bps maker, 2.5bps taker
            (5000, 1.0, 2.0),   # 5000+ shares: 1bps maker, 2bps taker
        ]
        model = create_tiered_volume_fee_model(tiers)
        
        assert isinstance(model, TieredFeeModel)
        assert model.volume_based is True
        assert len(model.tiers) == 3
        assert model.tiers[0][0] == 0  # First threshold
        assert model.tiers[0][1].maker_bps == 2.0  # First tier maker rate
    
    def test_create_exchange_fee_model(self):
        """Test creating exchange fee model."""
        model = create_exchange_fee_model("NASDAQ")
        
        assert isinstance(model, ExchangeFeeModel)
        assert model.exchange == "NASDAQ"
        assert "*" in model.config  # Should have wildcard config


class TestFeeModelIntegration:
    """Test fee model integration scenarios."""
    
    def test_deterministic_fees_with_seed(self):
        """Test that fees are deterministic with seeded RNG."""
        # This test would be relevant if we had random components in fees
        # For now, all our fee models are deterministic
        config = FeeConfig(maker_bps=1.0, taker_bps=2.0)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        
        # Calculate fee multiple times
        fee1 = model.calculate_fee(fill)
        fee2 = model.calculate_fee(fill)
        fee3 = model.calculate_fee(fill)
        
        # Should be identical
        assert fee1 == fee2 == fee3
    
    def test_fee_calculation_edge_cases(self):
        """Test fee calculation edge cases."""
        config = FeeConfig(maker_bps=1.0, maker_per_share=0.001, min_fee=0.01)
        model = BasisPointsFeeModel(config)
        
        # Very small notional
        fill1 = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 1.0, 1, maker_taker="MAKER")
        fee1 = model.calculate_fee(fill1)
        assert fee1 == 0.01  # Should hit minimum fee
        
        # Very large notional
        fill2 = Fill("3", "4", 2000, "AAPL", OrderSide.BUY, 1000.0, 10000, maker_taker="MAKER")
        fee2 = model.calculate_fee(fill2)
        expected = (10000000.0 * 0.0001) + (10000 * 0.001)  # 1000 + 10 = 1010
        assert fee2 == expected
    
    def test_maker_taker_differentiation(self):
        """Test that maker and taker fees are calculated differently."""
        config = FeeConfig(maker_bps=1.0, taker_bps=2.0, maker_per_share=0.001, taker_per_share=0.002)
        model = BasisPointsFeeModel(config)
        
        fill = Fill("1", "2", 1000, "AAPL", OrderSide.BUY, 150.0, 100, maker_taker="MAKER")
        maker_fee = model.calculate_fee(fill)
        
        fill.maker_taker = "TAKER"
        taker_fee = model.calculate_fee(fill)
        
        assert taker_fee > maker_fee  # Taker should pay more


if __name__ == "__main__":
    pytest.main([__file__])
