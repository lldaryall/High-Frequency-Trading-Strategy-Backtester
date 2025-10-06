"""
Unit tests for slippage modeling.
"""

import pytest
import numpy as np

from flashback.market.slippage import (
    SlippageConfig,
    FixedSlippageModel,
    ImbalanceSlippageModel,
    AdaptiveSlippageModel,
    create_slippage_model
)
from flashback.market.orders import OrderSide, OrderBookLevel, OrderBookSnapshot


class TestSlippageConfig:
    """Test SlippageConfig dataclass."""
    
    def test_slippage_config_creation(self):
        """Test slippage config creation."""
        config = SlippageConfig(
            base_slippage_bps=1.0,
            adverse_selection_factor=1.5,
            imbalance_threshold=0.4,
            max_slippage_bps=5.0,
            min_slippage_bps=0.1
        )
        
        assert config.base_slippage_bps == 1.0
        assert config.adverse_selection_factor == 1.5
        assert config.imbalance_threshold == 0.4
        assert config.max_slippage_bps == 5.0
        assert config.min_slippage_bps == 0.1


class TestFixedSlippageModel:
    """Test FixedSlippageModel class."""
    
    def test_fixed_slippage_creation(self):
        """Test fixed slippage model creation."""
        model = FixedSlippageModel(slippage_bps=2.0)
        assert model.slippage_bps == 2.0
    
    def test_fixed_slippage_calculation(self):
        """Test fixed slippage calculation."""
        model = FixedSlippageModel(slippage_bps=1.0)
        
        # Create mock order book
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        # Test buy order
        slippage = model.calculate_slippage(
            OrderSide.BUY, 100, book, 100.0
        )
        expected_slippage = 100.0 * (1.0 / 10000.0)  # 1 bps
        assert abs(slippage - expected_slippage) < 1e-10
        
        # Test sell order
        slippage = model.calculate_slippage(
            OrderSide.SELL, 100, book, 101.0
        )
        expected_slippage = 101.0 * (1.0 / 10000.0)  # 1 bps
        assert abs(slippage - expected_slippage) < 1e-10


class TestImbalanceSlippageModel:
    """Test ImbalanceSlippageModel class."""
    
    def test_imbalance_slippage_creation(self):
        """Test imbalance slippage model creation."""
        config = SlippageConfig()
        model = ImbalanceSlippageModel(config)
        assert model.config == config
    
    def test_imbalance_calculation(self):
        """Test order book imbalance calculation."""
        config = SlippageConfig()
        model = ImbalanceSlippageModel(config)
        
        # Balanced book
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        imbalance = model._calculate_imbalance(book)
        assert abs(imbalance) < 1e-10  # Should be balanced
        
        # Bid-heavy book
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=2000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        imbalance = model._calculate_imbalance(book)
        assert imbalance > 0  # Should be positive (bid-heavy)
        
        # Ask-heavy book
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=2000, order_count=1)]
        )
        
        imbalance = model._calculate_imbalance(book)
        assert imbalance < 0  # Should be negative (ask-heavy)
    
    def test_adverse_selection_calculation(self):
        """Test adverse selection calculation."""
        config = SlippageConfig(imbalance_threshold=0.3, adverse_selection_factor=2.0)
        model = ImbalanceSlippageModel(config)
        
        # No adverse selection - buying when bid-heavy
        adverse_selection = model._calculate_adverse_selection(OrderSide.BUY, 0.5)
        assert adverse_selection == 0.0
        
        # Adverse selection - buying when ask-heavy
        adverse_selection = model._calculate_adverse_selection(OrderSide.BUY, -0.5)
        assert adverse_selection > 0
        
        # Adverse selection - selling when bid-heavy
        adverse_selection = model._calculate_adverse_selection(OrderSide.SELL, 0.5)
        assert adverse_selection > 0
        
        # No adverse selection - selling when ask-heavy
        adverse_selection = model._calculate_adverse_selection(OrderSide.SELL, -0.5)
        assert adverse_selection == 0.0
    
    def test_size_impact_calculation(self):
        """Test size impact calculation."""
        config = SlippageConfig()
        model = ImbalanceSlippageModel(config)
        
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        # Small order relative to book depth
        size_impact = model._calculate_size_impact(100, book)
        assert size_impact >= 0
        
        # Large order relative to book depth
        size_impact = model._calculate_size_impact(2000, book)
        assert size_impact > 0
        assert size_impact <= 5.0  # Should be capped at 5 bps
    
    def test_slippage_calculation_with_imbalance(self):
        """Test complete slippage calculation with imbalance."""
        config = SlippageConfig(
            base_slippage_bps=1.0,
            adverse_selection_factor=2.0,
            imbalance_threshold=0.3
        )
        model = ImbalanceSlippageModel(config)
        
        # Ask-heavy book (adverse selection for buy orders)
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=500, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1500, order_count=1)]
        )
        
        slippage = model.calculate_slippage(
            OrderSide.BUY, 100, book, 100.0
        )
        
        # Should have base slippage + adverse selection
        assert slippage > 100.0 * (1.0 / 10000.0)  # More than just base slippage
    
    def test_slippage_bounds(self):
        """Test slippage bounds enforcement."""
        config = SlippageConfig(
            base_slippage_bps=1.0,
            max_slippage_bps=2.0,
            min_slippage_bps=0.5
        )
        model = ImbalanceSlippageModel(config)
        
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        slippage = model.calculate_slippage(
            OrderSide.BUY, 100, book, 100.0
        )
        
        min_slippage = 100.0 * (0.5 / 10000.0)
        max_slippage = 100.0 * (2.0 / 10000.0)
        
        assert min_slippage <= slippage <= max_slippage


class TestAdaptiveSlippageModel:
    """Test AdaptiveSlippageModel class."""
    
    def test_adaptive_slippage_creation(self):
        """Test adaptive slippage model creation."""
        config = SlippageConfig()
        model = AdaptiveSlippageModel(config, lookback_period=50)
        assert model.config == config
        assert model.lookback_period == 50
    
    def test_adaptive_factor_calculation(self):
        """Test adaptive factor calculation."""
        config = SlippageConfig()
        model = AdaptiveSlippageModel(config, lookback_period=10)
        
        # With no historical data, should return 1.0
        factor = model._calculate_adaptive_factor()
        assert factor == 1.0
        
        # Add some historical data
        model.recent_slippages = [0.001, 0.002, 0.0015, 0.003, 0.0025] * 5
        model.recent_imbalances = [0.1, 0.2, 0.15, 0.3, 0.25] * 5
        
        factor = model._calculate_adaptive_factor()
        assert isinstance(factor, float)
        assert factor > 0
    
    def test_adaptive_slippage_calculation(self):
        """Test adaptive slippage calculation."""
        config = SlippageConfig()
        model = AdaptiveSlippageModel(config)
        
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        # First calculation should be similar to base model
        slippage1 = model.calculate_slippage(
            OrderSide.BUY, 100, book, 100.0
        )
        
        # Second calculation should update historical data
        slippage2 = model.calculate_slippage(
            OrderSide.BUY, 100, book, 100.0
        )
        
        assert len(model.recent_slippages) == 2
        assert len(model.recent_imbalances) == 2


class TestSlippageModelFactory:
    """Test slippage model factory function."""
    
    def test_create_fixed_model(self):
        """Test creating fixed slippage model."""
        model = create_slippage_model("fixed", SlippageConfig(base_slippage_bps=2.0))
        assert isinstance(model, FixedSlippageModel)
        assert model.slippage_bps == 2.0
    
    def test_create_imbalance_model(self):
        """Test creating imbalance slippage model."""
        config = SlippageConfig(base_slippage_bps=1.0)
        model = create_slippage_model("imbalance", config)
        assert isinstance(model, ImbalanceSlippageModel)
        assert model.config == config
    
    def test_create_adaptive_model(self):
        """Test creating adaptive slippage model."""
        config = SlippageConfig(base_slippage_bps=1.0)
        model = create_slippage_model("adaptive", config)
        assert isinstance(model, AdaptiveSlippageModel)
        assert model.config == config
    
    def test_create_model_with_default_config(self):
        """Test creating model with default config."""
        model = create_slippage_model("fixed")
        assert isinstance(model, FixedSlippageModel)
    
    def test_create_unknown_model(self):
        """Test creating unknown model type."""
        with pytest.raises(ValueError, match="Unknown slippage model type"):
            create_slippage_model("unknown")


class TestSlippageIntegration:
    """Test slippage model integration."""
    
    def test_slippage_reduces_pnl(self):
        """Test that slippage reduces PnL compared to no-cost baseline."""
        # Create a scenario where we can measure PnL impact
        
        # No slippage baseline
        no_slippage_model = FixedSlippageModel(slippage_bps=0.0)
        
        # With slippage
        slippage_model = FixedSlippageModel(slippage_bps=5.0)
        
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        # Simulate a trade
        order_side = OrderSide.BUY
        order_size = 100
        price = 100.0
        
        # Calculate slippage
        no_slippage = no_slippage_model.calculate_slippage(
            order_side, order_size, book, price
        )
        with_slippage = slippage_model.calculate_slippage(
            order_side, order_size, book, price
        )
        
        # Slippage should reduce effective price for buy orders
        effective_price_no_slippage = price + no_slippage
        effective_price_with_slippage = price + with_slippage
        
        assert effective_price_with_slippage > effective_price_no_slippage
        
        # Calculate PnL impact
        pnl_impact = (with_slippage - no_slippage) * order_size
        assert pnl_impact > 0  # Should reduce PnL
    
    def test_imbalance_adverse_selection_impact(self):
        """Test that adverse selection increases slippage appropriately."""
        config = SlippageConfig(
            base_slippage_bps=1.0,
            adverse_selection_factor=2.0,
            imbalance_threshold=0.3
        )
        model = ImbalanceSlippageModel(config)
        
        # Balanced book
        balanced_book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=1000, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1000, order_count=1)]
        )
        
        # Imbalanced book (ask-heavy)
        imbalanced_book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=1000,
            bids=[OrderBookLevel(price=100.0, total_qty=500, order_count=1)],
            asks=[OrderBookLevel(price=101.0, total_qty=1500, order_count=1)]
        )
        
        # Buy order in balanced book
        balanced_slippage = model.calculate_slippage(
            OrderSide.BUY, 100, balanced_book, 100.0
        )
        
        # Buy order in imbalanced book (adverse selection)
        imbalanced_slippage = model.calculate_slippage(
            OrderSide.BUY, 100, imbalanced_book, 100.0
        )
        
        # Imbalanced book should have higher slippage due to adverse selection
        assert imbalanced_slippage > balanced_slippage
        
        # The difference should be significant
        slippage_difference = imbalanced_slippage - balanced_slippage
        assert slippage_difference > 0.001  # At least 0.1 bps difference
