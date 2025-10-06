"""
Unit tests for momentum imbalance strategy.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from flashback.strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
from flashback.core.events import MarketDataEvent, FillEvent, EventType
from flashback.market.orders import OrderSide


class TestMomentumImbalanceConfig:
    """Test momentum imbalance configuration."""
    
    def test_config_creation(self):
        """Test configuration creation with default values."""
        config = MomentumImbalanceConfig(
            strategy_id="test_strategy",
            symbol="AAPL"
        )
        
        assert config.strategy_id == "test_strategy"
        assert config.symbol == "AAPL"
        assert config.imbalance_lookback == 10
        assert config.imbalance_threshold == 0.6
        assert config.short_ema_period == 5
        assert config.long_ema_period == 20
        assert config.position_size == 100
        assert config.min_price_history == 50
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = MomentumImbalanceConfig(
            strategy_id="test_strategy",
            symbol="AAPL",
            short_ema_period=5,
            long_ema_period=20,
            stop_loss_pct=0.01,
            take_profit_pct=0.02
        )
        assert config.long_ema_period > config.short_ema_period
        assert config.take_profit_pct > config.stop_loss_pct
        
        # Invalid config - long EMA <= short EMA
        with pytest.raises(ValueError, match="long_ema_period must be greater than short_ema_period"):
            MomentumImbalanceConfig(
                strategy_id="test_strategy",
                symbol="AAPL",
                short_ema_period=10,
                long_ema_period=5
            )
        
        # Invalid config - take profit <= stop loss
        with pytest.raises(ValueError, match="take_profit_pct must be greater than stop_loss_pct"):
            MomentumImbalanceConfig(
                strategy_id="test_strategy",
                symbol="AAPL",
                stop_loss_pct=0.02,
                take_profit_pct=0.01
            )


class TestMomentumImbalanceStrategy:
    """Test momentum imbalance strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = MomentumImbalanceConfig(
            strategy_id="test_strategy",
            symbol="AAPL"
        )
        strategy = MomentumImbalanceStrategy(config)
        
        assert strategy.strategy_id == "test_strategy"
        assert strategy.symbol == "AAPL"
        assert strategy.current_position == 0
        assert strategy.entry_price is None
        assert strategy.current_imbalance is None
        assert strategy.short_ema is None
        assert strategy.long_ema is None
        assert strategy.last_ema_signal is None
        assert strategy.last_imbalance_signal is None
    
    def test_strategy_state_management(self):
        """Test strategy state management."""
        config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
        strategy = MomentumImbalanceStrategy(config)
        
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
    
    def test_update_order_flow_imbalance(self):
        """Test order flow imbalance calculation."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            imbalance_lookback=5
        )
        strategy = MomentumImbalanceStrategy(config)
        
        # Add volume data
        volumes = [100, 200, 150, 300, 250, 400, 350, 500, 450, 600]
        
        for volume in volumes:
            strategy._update_order_flow_imbalance(volume)
        
        # Check that imbalance is calculated
        assert strategy.current_imbalance is not None
        assert -1.0 <= strategy.current_imbalance <= 1.0
        
        # Check history length is maintained
        assert len(strategy.buy_volume_history) <= config.imbalance_lookback * 2
    
    def test_update_emas(self):
        """Test EMA calculation."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            short_ema_period=3,
            long_ema_period=5
        )
        strategy = MomentumImbalanceStrategy(config)
        
        # Test EMA initialization
        strategy._update_emas(100.0)
        assert strategy.short_ema == 100.0
        assert strategy.long_ema == 100.0
        
        # Test EMA updates
        prices = [101.0, 102.0, 103.0, 104.0, 105.0]
        for price in prices:
            strategy._update_emas(price)
        
        # EMAs should be updated
        assert strategy.short_ema != 100.0
        assert strategy.long_ema != 100.0
        assert strategy.short_ema > strategy.long_ema  # Short EMA should be higher for rising prices
    
    def test_get_ema_signal(self):
        """Test EMA signal generation."""
        config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
        strategy = MomentumImbalanceStrategy(config)
        
        # No signal when EMAs are not set
        assert strategy._get_ema_signal() is None
        
        # Set EMAs
        strategy.short_ema = 105.0
        strategy.long_ema = 100.0
        assert strategy._get_ema_signal() == 'bullish'
        
        strategy.short_ema = 95.0
        strategy.long_ema = 100.0
        assert strategy._get_ema_signal() == 'bearish'
        
        strategy.short_ema = 100.0
        strategy.long_ema = 100.0
        assert strategy._get_ema_signal() is None
    
    def test_get_imbalance_signal(self):
        """Test imbalance signal generation."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            imbalance_threshold=0.5
        )
        strategy = MomentumImbalanceStrategy(config)
        
        # No signal when imbalance is not set
        assert strategy._get_imbalance_signal() is None
        
        # Set imbalance
        strategy.current_imbalance = 0.7
        assert strategy._get_imbalance_signal() == 'bullish'
        
        strategy.current_imbalance = -0.7
        assert strategy._get_imbalance_signal() == 'bearish'
        
        strategy.current_imbalance = 0.3
        assert strategy._get_imbalance_signal() is None
    
    def test_generate_signals_no_position(self):
        """Test signal generation when no position."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            imbalance_threshold=0.5
        )
        strategy = MomentumImbalanceStrategy(config)
        strategy.current_position = 0
        
        # Set up bullish signals
        strategy.short_ema = 105.0
        strategy.long_ema = 100.0
        strategy.current_imbalance = 0.7
        strategy.last_ema_signal = None  # Simulate crossover
        
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 100
        
        # Set up bearish signals
        strategy.short_ema = 95.0
        strategy.long_ema = 100.0
        strategy.current_imbalance = -0.7
        strategy.last_ema_signal = None  # Simulate crossover
        
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 100
    
    def test_generate_signals_with_position(self):
        """Test signal generation when position exists."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            imbalance_threshold=0.5
        )
        strategy = MomentumImbalanceStrategy(config)
        
        # Test exit long position on bearish signals
        strategy.current_position = 100
        strategy.short_ema = 95.0
        strategy.long_ema = 100.0
        strategy.current_imbalance = -0.7
        
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 100
        
        # Test exit short position on bullish signals
        strategy.current_position = -100
        strategy.short_ema = 105.0
        strategy.long_ema = 100.0
        strategy.current_imbalance = 0.7
        
        orders = strategy._generate_signals(150.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 100
    
    def test_check_exit_conditions(self):
        """Test stop loss and take profit conditions."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            stop_loss_pct=0.02,
            take_profit_pct=0.03
        )
        strategy = MomentumImbalanceStrategy(config)
        
        # Set up long position
        strategy.current_position = 100
        strategy.entry_price = 150.0
        
        # Test stop loss
        orders = strategy._check_exit_conditions(147.0)  # 2% below entry
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        
        # Test take profit
        orders = strategy._check_exit_conditions(154.5)  # 3% above entry
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        
        # Test no exit
        orders = strategy._check_exit_conditions(152.0)  # Between stop and take profit
        assert len(orders) == 0
        
        # Test short position
        strategy.current_position = -100
        
        # Test stop loss
        orders = strategy._check_exit_conditions(153.0)  # 2% above entry
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        
        # Test take profit
        orders = strategy._check_exit_conditions(145.5)  # 3% below entry
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
    
    def test_on_trade_position_update(self):
        """Test position update on trade execution."""
        config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
        strategy = MomentumImbalanceStrategy(config)
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
        config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
        strategy = MomentumImbalanceStrategy(config)
        strategy.start()  # Start the strategy
        
        # Set up existing position
        strategy.current_position = 100
        strategy.entry_price = 150.0
        strategy.entry_timestamp = 1000
        strategy.position.current_position = 100
        strategy.position.entry_price = 150.0
        strategy.position.entry_timestamp = 1000
        
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
    
    def test_synthetic_data_trading_round_trip(self):
        """Test complete trading round trip with synthetic data."""
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol="AAPL",
            imbalance_lookback=5,
            imbalance_threshold=0.4,
            short_ema_period=3,
            long_ema_period=8,
            position_size=50,
            min_price_history=20
        )
        strategy = MomentumImbalanceStrategy(config)
        strategy.start()
        
        # Create synthetic price data with momentum pattern
        base_price = 150.0
        prices = []
        volumes = []
        
        # Create a pattern: sideways, then strong uptrend, then reversal
        for i in range(30):
            if i < 10:
                # Sideways movement
                price = base_price + np.random.normal(0, 0.5)
                volume = 1000
            elif i < 20:
                # Strong uptrend (should trigger long signal)
                price = base_price + (i - 10) * 0.5 + np.random.normal(0, 0.2)
                volume = 2000  # Higher volume
            else:
                # Reversal (should trigger exit)
                price = base_price + 5.0 - (i - 20) * 0.3 + np.random.normal(0, 0.2)
                volume = 1500
            prices.append(price)
            volumes.append(volume)
        
        # Process price data
        orders_generated = []
        fills_processed = []
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            # Create market data event
            book_update = MarketDataEvent(
                timestamp=pd.Timestamp(1000 + i * 1000, unit='ns'),
                event_type=EventType.MARKET_DATA,
                symbol="AAPL",
                side="B",  # Bid side
                price=price,
                size=volume,
                event_type_str="TICK",
                data={
                    'mid_price': price,
                    'volume': volume
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
        config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
        strategy = MomentumImbalanceStrategy(config)
        
        # Add some test data
        strategy.current_position = 100
        strategy.entry_price = 150.0
        strategy.current_imbalance = 0.7
        strategy.short_ema = 155.0
        strategy.long_ema = 150.0
        strategy.total_trades = 8
        strategy.winning_trades = 5
        strategy.losing_trades = 3
        strategy.total_pnl = 2000.0
        
        stats = strategy.get_statistics()
        
        assert stats['strategy_id'] == "test"
        assert stats['symbol'] == "AAPL"
        assert stats['current_position'] == 100
        assert stats['entry_price'] == 150.0
        assert stats['current_imbalance'] == 0.7
        assert stats['short_ema'] == 155.0
        assert stats['long_ema'] == 150.0
        assert stats['ema_signal'] == 'bullish'
        assert stats['imbalance_signal'] == 'bullish'
        assert stats['total_trades'] == 8
        assert stats['winning_trades'] == 5
        assert stats['losing_trades'] == 3
        assert stats['total_pnl'] == 2000.0
        assert stats['win_rate'] == 0.625
        assert stats['avg_pnl_per_trade'] == 250.0
