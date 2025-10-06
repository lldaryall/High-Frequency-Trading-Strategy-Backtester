"""
Unit tests for mean reversion strategy.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from flashback.strategy.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from flashback.core.events import MarketDataEvent, FillEvent, EventType
from flashback.market.orders import OrderSide


class TestMeanReversionConfig:
    """Test mean reversion configuration."""
    
    def test_config_creation(self):
        """Test configuration creation with default values."""
        config = MeanReversionConfig(
            strategy_id="test_strategy",
            symbol="AAPL"
        )
        
        assert config.strategy_id == "test_strategy"
        assert config.symbol == "AAPL"
        assert config.lookback_period == 20
        assert config.z_score_threshold == 2.0
        assert config.exit_z_score == 0.5
        assert config.position_size == 100
        assert config.min_price_history == 30
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = MeanReversionConfig(
            strategy_id="test_strategy",
            symbol="AAPL",
            lookback_period=10,
            z_score_threshold=1.5,
            exit_z_score=0.3
        )
        assert config.exit_z_score < config.z_score_threshold
        
        # Invalid config - exit threshold >= entry threshold
        with pytest.raises(ValueError, match="exit_z_score must be less than z_score_threshold"):
            MeanReversionConfig(
                strategy_id="test_strategy",
                symbol="AAPL",
                z_score_threshold=1.0,
                exit_z_score=1.5
            )


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = MeanReversionConfig(
            strategy_id="test_strategy",
            symbol="AAPL"
        )
        strategy = MeanReversionStrategy(config)
        
        assert strategy.strategy_id == "test_strategy"
        assert strategy.symbol == "AAPL"
        assert strategy.current_position == 0
        assert strategy.entry_price is None
        assert strategy.rolling_mean is None
        assert strategy.rolling_std is None
        assert strategy.current_z_score is None
    
    def test_strategy_state_management(self):
        """Test strategy state management."""
        config = MeanReversionConfig(strategy_id="test", symbol="AAPL")
        strategy = MeanReversionStrategy(config)
        
        # Initial state
        assert not strategy.is_active()
        
        # Start strategy
        strategy.start()
        assert strategy.is_active()
        
        # Pause strategy
        strategy.pause()
        assert not strategy.is_active()
        
        # Resume strategy
        strategy.resume()
        assert strategy.is_active()
        
        # Stop strategy
        strategy.stop()
        assert not strategy.is_active()
    
    def test_create_order_intent(self):
        """Test order intent creation."""
        config = MeanReversionConfig(strategy_id="test", symbol="AAPL")
        strategy = MeanReversionStrategy(config)
        strategy.last_update_time = 1000
        
        # Create buy order
        order = strategy._create_order_intent(
            side=OrderSide.BUY,
            price=150.0,
            quantity=100
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.price == 150.0
        assert order.quantity == 100
        assert order.strategy_id == "test"
        assert order.intent_id.startswith("test_")
        
        # Check order counter
        assert strategy.total_orders == 1
    
    def test_update_rolling_statistics(self):
        """Test rolling statistics calculation."""
        config = MeanReversionConfig(
            strategy_id="test",
            symbol="AAPL",
            lookback_period=5
        )
        strategy = MeanReversionStrategy(config)
        
        # Add price history
        strategy.price_history = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        
        # Update statistics
        strategy._update_rolling_statistics()
        
        # Check rolling mean and std
        expected_mean = np.mean([105, 106, 107, 108, 109])
        expected_std = np.std([105, 106, 107, 108, 109])
        
        assert abs(strategy.rolling_mean - expected_mean) < 1e-10
        assert abs(strategy.rolling_std - expected_std) < 1e-10
        
        # Check z-score
        expected_z_score = (109 - expected_mean) / expected_std
        assert abs(strategy.current_z_score - expected_z_score) < 1e-10
    
    def test_generate_signals_no_position(self):
        """Test signal generation when no position."""
        config = MeanReversionConfig(
            strategy_id="test",
            symbol="AAPL",
            z_score_threshold=2.0,
            exit_z_score=0.5
        )
        strategy = MeanReversionStrategy(config)
        strategy.current_position = 0
        
        # Test long signal (negative z-score)
        strategy.current_z_score = -2.5
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 100
        
        # Test short signal (positive z-score)
        strategy.current_z_score = 2.5
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 100
        
        # Test no signal (z-score within threshold)
        strategy.current_z_score = 1.0
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 0
    
    def test_generate_signals_with_position(self):
        """Test signal generation when position exists."""
        config = MeanReversionConfig(
            strategy_id="test",
            symbol="AAPL",
            z_score_threshold=2.0,
            exit_z_score=0.5
        )
        strategy = MeanReversionStrategy(config)
        
        # Test exit long position
        strategy.current_position = 100
        strategy.current_z_score = -0.3  # Within exit threshold
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 100
        
        # Test exit short position
        strategy.current_position = -100
        strategy.current_z_score = 0.3  # Within exit threshold
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 100
    
    def test_on_trade_position_update(self):
        """Test position update on trade execution."""
        config = MeanReversionConfig(strategy_id="test", symbol="AAPL")
        strategy = MeanReversionStrategy(config)
        strategy.start()  # Start the strategy
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            commission=0.5
        )
        
        # Process trade
        orders = strategy.on_trade(fill_event)
        
        # Check position update
        assert strategy.current_position == 100
        assert strategy.entry_price == 150.0
        assert strategy.entry_timestamp == 1000
        assert strategy.total_fills == 1
        assert len(orders) == 0  # No new orders generated
    
    def test_on_trade_position_close(self):
        """Test position closing on trade execution."""
        config = MeanReversionConfig(strategy_id="test", symbol="AAPL")
        strategy = MeanReversionStrategy(config)
        strategy.start()  # Start the strategy
        
        # Set up existing position
        strategy.current_position = 100
        strategy.entry_price = 150.0
        strategy.entry_timestamp = 1000
        
        # Create fill event that closes position
        fill_event = FillEvent(
            timestamp=pd.Timestamp(2000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_2",
            symbol="AAPL",
            side="SELL",
            quantity=100,
            price=155.0,
            commission=0.5
        )
        
        # Process trade
        orders = strategy.on_trade(fill_event)
        
        # Check position closed
        assert strategy.current_position == 0
        assert strategy.entry_price is None
        assert strategy.entry_timestamp is None
        assert strategy.total_trades == 1
        assert strategy.total_pnl == 500.0  # (155 - 150) * 100
        assert strategy.winning_trades == 1
    
    def test_stop_loss_condition(self):
        """Test stop loss condition."""
        config = MeanReversionConfig(
            strategy_id="test",
            symbol="AAPL",
            stop_loss_pct=0.02
        )
        strategy = MeanReversionStrategy(config)
        
        # Set up long position
        strategy.current_position = 100
        strategy.entry_price = 150.0
        
        # Test stop loss triggered
        current_price = 147.0  # 2% below entry
        assert strategy._should_stop_loss(current_price)
        
        # Test stop loss not triggered
        current_price = 148.0  # Less than 2% below entry
        assert not strategy._should_stop_loss(current_price)
        
        # Test short position stop loss
        strategy.current_position = -100
        current_price = 153.0  # 2% above entry
        assert strategy._should_stop_loss(current_price)
    
    def test_synthetic_data_trading_round_trip(self):
        """Test complete trading round trip with synthetic data."""
        config = MeanReversionConfig(
            strategy_id="test",
            symbol="AAPL",
            lookback_period=5,
            z_score_threshold=1.5,
            exit_z_score=0.3,
            position_size=50,
            min_price_history=10
        )
        strategy = MeanReversionStrategy(config)
        strategy.start()
        
        # Create synthetic price data with mean reversion pattern
        base_price = 150.0
        prices = []
        
        # Create a pattern: normal prices, then extreme low, then recovery
        for i in range(20):
            if i < 10:
                # Normal prices around base
                price = base_price + np.random.normal(0, 0.5)
            elif i < 15:
                # Extreme low prices (should trigger long signal)
                price = base_price - 3.0 + np.random.normal(0, 0.2)
            else:
                # Recovery (should trigger exit)
                price = base_price - 0.5 + np.random.normal(0, 0.2)
            prices.append(price)
        
        # Process price data
        orders_generated = []
        fills_processed = []
        
        for i, price in enumerate(prices):
            # Create market data event
            book_update = MarketDataEvent(
                timestamp=pd.Timestamp(1000 + i * 1000, unit='ns'),
                event_type=EventType.MARKET_DATA,
                data={
                    'mid_price': price,
                    'volume': 1000
                }
            )
            
            # Process bar
            orders = strategy.on_bar(book_update)
            orders_generated.extend(orders)
            
            # Simulate immediate fills for market orders
            for order in orders:
                if order.order_type.value == "LIMIT":  # Convert to market for testing
                    fill_event = FillEvent(
                        timestamp=pd.Timestamp(1000 + i * 1000 + 100, unit='ns'),
                        event_type=EventType.FILL,
                        order_id=order.intent_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        price=price,
                        commission=0.5
                    )
                    fills_processed.append(fill_event)
                    strategy.on_trade(fill_event)
        
        # Verify that we had at least one entry and exit
        assert len(orders_generated) >= 2, f"Expected at least 2 orders, got {len(orders_generated)}"
        assert len(fills_processed) >= 2, f"Expected at least 2 fills, got {len(fills_processed)}"
        
        # Verify strategy statistics
        stats = strategy.get_statistics()
        assert stats['total_trades'] >= 1, f"Expected at least 1 trade, got {stats['total_trades']}"
        assert stats['current_position'] == 0, f"Expected position to be closed, got {stats['current_position']}"
        
        print(f"Strategy completed {stats['total_trades']} trades with P&L: {stats['total_pnl']}")
    
    def test_get_statistics(self):
        """Test statistics generation."""
        config = MeanReversionConfig(strategy_id="test", symbol="AAPL")
        strategy = MeanReversionStrategy(config)
        
        # Add some test data
        strategy.current_position = 100
        strategy.entry_price = 150.0
        strategy.rolling_mean = 148.0
        strategy.rolling_std = 2.0
        strategy.current_z_score = 1.0
        strategy.total_trades = 5
        strategy.winning_trades = 3
        strategy.losing_trades = 2
        strategy.total_pnl = 1000.0
        
        stats = strategy.get_statistics()
        
        assert stats['strategy_id'] == "test"
        assert stats['symbol'] == "AAPL"
        assert stats['current_position'] == 100
        assert stats['entry_price'] == 150.0
        assert stats['rolling_mean'] == 148.0
        assert stats['rolling_std'] == 2.0
        assert stats['current_z_score'] == 1.0
        assert stats['total_trades'] == 5
        assert stats['winning_trades'] == 3
        assert stats['losing_trades'] == 2
        assert stats['total_pnl'] == 1000.0
        assert stats['win_rate'] == 0.6
        assert stats['avg_pnl_per_trade'] == 200.0
