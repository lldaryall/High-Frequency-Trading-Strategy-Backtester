"""
Unit tests for transaction costs modeling.
"""

import pytest

from flashback.market.transaction_costs import (
    TransactionCostConfig,
    FeeTier,
    TransactionCosts,
    SimpleTransactionCostModel,
    TieredTransactionCostModel,
    ExchangeTransactionCostModel,
    create_transaction_cost_model,
    calculate_maker_taker_status
)
from flashback.market.orders import OrderSide, OrderType, TimeInForce


class TestTransactionCostConfig:
    """Test TransactionCostConfig dataclass."""
    
    def test_transaction_cost_config_creation(self):
        """Test transaction cost config creation."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.5,
            per_share_cost=0.001,
            min_fee=0.01,
            max_fee=100.0
        )
        
        assert config.maker_fee_bps == 0.0
        assert config.taker_fee_bps == 0.5
        assert config.per_share_cost == 0.001
        assert config.min_fee == 0.01
        assert config.max_fee == 100.0


class TestFeeTier:
    """Test FeeTier dataclass."""
    
    def test_fee_tier_creation(self):
        """Test fee tier creation."""
        tier = FeeTier(
            min_volume=1000000,
            max_volume=10000000,
            maker_fee_bps=0.0,
            taker_fee_bps=0.3
        )
        
        assert tier.min_volume == 1000000
        assert tier.max_volume == 10000000
        assert tier.maker_fee_bps == 0.0
        assert tier.taker_fee_bps == 0.3


class TestTransactionCosts:
    """Test TransactionCosts dataclass."""
    
    def test_transaction_costs_creation(self):
        """Test transaction costs creation."""
        costs = TransactionCosts(
            maker_taker_fee=1.0,
            per_share_cost=0.5,
            total_cost=1.5,
            fee_type="taker"
        )
        
        assert costs.maker_taker_fee == 1.0
        assert costs.per_share_cost == 0.5
        assert costs.total_cost == 1.5
        assert costs.fee_type == "taker"


class TestSimpleTransactionCostModel:
    """Test SimpleTransactionCostModel class."""
    
    def test_simple_model_creation(self):
        """Test simple transaction cost model creation."""
        config = TransactionCostConfig()
        model = SimpleTransactionCostModel(config)
        assert model.config == config
    
    def test_maker_fee_calculation(self):
        """Test maker fee calculation."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.5,
            per_share_cost=0.0
        )
        model = SimpleTransactionCostModel(config)
        
        costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 100, 100.0, is_maker=True
        )
        
        assert costs.maker_taker_fee == 0.0
        assert costs.fee_type == "maker"
        assert costs.total_cost == 0.0
    
    def test_taker_fee_calculation(self):
        """Test taker fee calculation."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.5,
            per_share_cost=0.0
        )
        model = SimpleTransactionCostModel(config)
        
        costs = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False
        )
        
        expected_fee = 100.0 * 100.0 * (0.5 / 10000.0)  # 0.5 bps
        assert abs(costs.maker_taker_fee - expected_fee) < 1e-10
        assert costs.fee_type == "taker"
        assert costs.total_cost == expected_fee
    
    def test_per_share_cost_calculation(self):
        """Test per-share cost calculation."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            per_share_cost=0.001
        )
        model = SimpleTransactionCostModel(config)
        
        costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 100, 100.0, is_maker=True
        )
        
        expected_per_share = 100 * 0.001
        assert abs(costs.per_share_cost - expected_per_share) < 1e-10
        assert costs.total_cost == expected_per_share
    
    def test_min_max_fee_bounds(self):
        """Test min/max fee bounds."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            per_share_cost=0.0,
            min_fee=1.0,
            max_fee=10.0
        )
        model = SimpleTransactionCostModel(config)
        
        # Test minimum fee
        costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 1, 1.0, is_maker=True
        )
        assert costs.total_cost == 1.0  # Should be min_fee
        
        # Test maximum fee (would need very large order)
        config.max_fee = 0.01
        config.min_fee = 0.0  # Set min_fee to 0 to test max_fee
        config.taker_fee_bps = 1.0  # Add some fee to exceed max_fee
        model = SimpleTransactionCostModel(config)
        costs = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 1000000, 100.0, is_maker=False  # Use taker to get fees
        )
        assert costs.total_cost == 0.01  # Should be max_fee


class TestTieredTransactionCostModel:
    """Test TieredTransactionCostModel class."""
    
    def test_tiered_model_creation(self):
        """Test tiered transaction cost model creation."""
        config = TransactionCostConfig()
        tiers = [
            FeeTier(min_volume=0, max_volume=1000000, maker_fee_bps=0.0, taker_fee_bps=0.5),
            FeeTier(min_volume=1000000, max_volume=10000000, maker_fee_bps=0.0, taker_fee_bps=0.3),
            FeeTier(min_volume=10000000, max_volume=float('inf'), maker_fee_bps=0.0, taker_fee_bps=0.1),
        ]
        model = TieredTransactionCostModel(config, tiers)
        assert model.config == config
        assert len(model.fee_tiers) == 3
    
    def test_fee_tier_selection(self):
        """Test fee tier selection based on volume."""
        config = TransactionCostConfig()
        tiers = [
            FeeTier(min_volume=0, max_volume=1000000, maker_fee_bps=0.0, taker_fee_bps=0.5),
            FeeTier(min_volume=1000000, max_volume=10000000, maker_fee_bps=0.0, taker_fee_bps=0.3),
            FeeTier(min_volume=10000000, max_volume=float('inf'), maker_fee_bps=0.0, taker_fee_bps=0.1),
        ]
        model = TieredTransactionCostModel(config, tiers)
        
        # Low volume - should get highest fee
        tier = model._find_fee_tier(500000)
        assert tier.taker_fee_bps == 0.5
        
        # Medium volume - should get medium fee
        tier = model._find_fee_tier(5000000)
        assert tier.taker_fee_bps == 0.3
        
        # High volume - should get lowest fee
        tier = model._find_fee_tier(15000000)
        assert tier.taker_fee_bps == 0.1
    
    def test_tiered_cost_calculation(self):
        """Test tiered cost calculation."""
        config = TransactionCostConfig()
        tiers = [
            FeeTier(min_volume=0, max_volume=1000000, maker_fee_bps=0.0, taker_fee_bps=0.5),
            FeeTier(min_volume=1000000, max_volume=10000000, maker_fee_bps=0.0, taker_fee_bps=0.3),
        ]
        model = TieredTransactionCostModel(config, tiers)
        
        # Low volume trade
        costs_low = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, monthly_volume=500000
        )
        
        # High volume trade
        costs_high = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, monthly_volume=5000000
        )
        
        # High volume should have lower fees
        assert costs_high.maker_taker_fee < costs_low.maker_taker_fee


class TestExchangeTransactionCostModel:
    """Test ExchangeTransactionCostModel class."""
    
    def test_exchange_model_creation(self):
        """Test exchange transaction cost model creation."""
        config = TransactionCostConfig(
            exchange_fees={
                "NASDAQ": {"maker_fee_bps": 0.0, "taker_fee_bps": 0.3},
                "NYSE": {"maker_fee_bps": 0.0, "taker_fee_bps": 0.4}
            }
        )
        model = ExchangeTransactionCostModel(config)
        assert model.config == config
        assert "NASDAQ" in model.exchange_fees
        assert "NYSE" in model.exchange_fees
    
    def test_exchange_specific_fees(self):
        """Test exchange-specific fee calculation."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.5,
            exchange_fees={
                "NASDAQ": {"maker_fee_bps": 0.0, "taker_fee_bps": 0.3},
                "NYSE": {"maker_fee_bps": 0.0, "taker_fee_bps": 0.4}
            }
        )
        model = ExchangeTransactionCostModel(config)
        
        # NASDAQ trade
        costs_nasdaq = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, exchange="NASDAQ"
        )
        
        # NYSE trade
        costs_nyse = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, exchange="NYSE"
        )
        
        # Default exchange trade
        costs_default = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, exchange="DEFAULT"
        )
        
        # NASDAQ should have lowest fees
        assert costs_nasdaq.maker_taker_fee < costs_nyse.maker_taker_fee
        assert costs_nasdaq.maker_taker_fee < costs_default.maker_taker_fee
        
        # NYSE should have medium fees
        assert costs_nyse.maker_taker_fee < costs_default.maker_taker_fee


class TestMakerTakerStatus:
    """Test maker/taker status calculation."""
    
    def test_market_order_is_taker(self):
        """Test that market orders are always takers."""
        is_maker = calculate_maker_taker_status(
            OrderType.MARKET, TimeInForce.DAY, 100.0, 100.0, OrderSide.BUY
        )
        assert not is_maker
    
    def test_ioc_order_is_taker(self):
        """Test that IOC orders are takers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.IOC, 100.0, 100.0, OrderSide.BUY
        )
        assert not is_maker
    
    def test_fok_order_is_taker(self):
        """Test that FOK orders are takers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.FOK, 100.0, 100.0, OrderSide.BUY
        )
        assert not is_maker
    
    def test_limit_order_maker_when_below_market(self):
        """Test that limit buy orders below market are makers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.DAY, 99.0, 100.0, OrderSide.BUY
        )
        assert is_maker
    
    def test_limit_order_taker_when_above_market(self):
        """Test that limit buy orders above market are takers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.DAY, 101.0, 100.0, OrderSide.BUY
        )
        assert not is_maker
    
    def test_limit_sell_order_maker_when_above_market(self):
        """Test that limit sell orders above market are makers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.DAY, 101.0, 100.0, OrderSide.SELL
        )
        assert is_maker
    
    def test_limit_sell_order_taker_when_below_market(self):
        """Test that limit sell orders below market are takers."""
        is_maker = calculate_maker_taker_status(
            OrderType.LIMIT, TimeInForce.DAY, 99.0, 100.0, OrderSide.SELL
        )
        assert not is_maker


class TestTransactionCostFactory:
    """Test transaction cost model factory function."""
    
    def test_create_simple_model(self):
        """Test creating simple transaction cost model."""
        model = create_transaction_cost_model("simple")
        assert isinstance(model, SimpleTransactionCostModel)
    
    def test_create_tiered_model(self):
        """Test creating tiered transaction cost model."""
        model = create_transaction_cost_model("tiered")
        assert isinstance(model, TieredTransactionCostModel)
    
    def test_create_exchange_model(self):
        """Test creating exchange transaction cost model."""
        model = create_transaction_cost_model("exchange")
        assert isinstance(model, ExchangeTransactionCostModel)
    
    def test_create_unknown_model(self):
        """Test creating unknown model type."""
        with pytest.raises(ValueError, match="Unknown transaction cost model type"):
            create_transaction_cost_model("unknown")


class TestTransactionCostIntegration:
    """Test transaction cost integration and PnL impact."""
    
    def test_costs_reduce_pnl(self):
        """Test that transaction costs reduce PnL compared to no-cost baseline."""
        # No cost baseline
        no_cost_model = SimpleTransactionCostModel(
            TransactionCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.0,
                per_share_cost=0.0
            )
        )
        
        # With costs
        cost_model = SimpleTransactionCostModel(
            TransactionCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.5,
                per_share_cost=0.001
            )
        )
        
        # Simulate a trade
        order_side = OrderSide.BUY
        order_type = OrderType.MARKET
        quantity = 100
        price = 100.0
        
        # Calculate costs
        no_costs = no_cost_model.calculate_costs(
            order_side, order_type, quantity, price, is_maker=False
        )
        with_costs = cost_model.calculate_costs(
            order_side, order_type, quantity, price, is_maker=False
        )
        
        # With costs should have higher total cost
        assert with_costs.total_cost > no_costs.total_cost
        
        # PnL impact should be negative (reduces PnL)
        pnl_impact = with_costs.total_cost - no_costs.total_cost
        assert pnl_impact > 0  # Higher costs reduce PnL
    
    def test_maker_vs_taker_cost_difference(self):
        """Test that maker and taker have different costs."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.5,
            per_share_cost=0.0
        )
        model = SimpleTransactionCostModel(config)
        
        # Maker order
        maker_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 100, 100.0, is_maker=True
        )
        
        # Taker order
        taker_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False
        )
        
        # Taker should have higher costs
        assert taker_costs.total_cost > maker_costs.total_cost
        assert taker_costs.fee_type == "taker"
        assert maker_costs.fee_type == "maker"
    
    def test_per_share_cost_scales_with_quantity(self):
        """Test that per-share costs scale with quantity."""
        config = TransactionCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            per_share_cost=0.001
        )
        model = SimpleTransactionCostModel(config)
        
        # Small quantity
        small_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 100, 100.0, is_maker=True
        )
        
        # Large quantity
        large_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.LIMIT, 1000, 100.0, is_maker=True
        )
        
        # Large quantity should have 10x the per-share cost
        assert large_costs.per_share_cost == 10 * small_costs.per_share_cost
        assert large_costs.total_cost == 10 * small_costs.total_cost
    
    def test_tiered_costs_encourage_volume(self):
        """Test that tiered costs encourage higher volume trading."""
        config = TransactionCostConfig()
        tiers = [
            FeeTier(min_volume=0, max_volume=1000000, maker_fee_bps=0.0, taker_fee_bps=0.5),
            FeeTier(min_volume=1000000, max_volume=10000000, maker_fee_bps=0.0, taker_fee_bps=0.3),
            FeeTier(min_volume=10000000, max_volume=float('inf'), maker_fee_bps=0.0, taker_fee_bps=0.1),
        ]
        model = TieredTransactionCostModel(config, tiers)
        
        # Low volume trader
        low_volume_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, monthly_volume=500000
        )
        
        # High volume trader
        high_volume_costs = model.calculate_costs(
            OrderSide.BUY, OrderType.MARKET, 100, 100.0, is_maker=False, monthly_volume=15000000
        )
        
        # High volume trader should have significantly lower costs
        cost_reduction = (low_volume_costs.maker_taker_fee - high_volume_costs.maker_taker_fee) / low_volume_costs.maker_taker_fee
        assert cost_reduction > 0.5  # At least 50% reduction
