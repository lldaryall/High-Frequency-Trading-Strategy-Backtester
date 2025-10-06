"""
Unit tests for portfolio risk management system.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from flashback.risk.portfolio import (
    PortfolioRiskManager,
    Position,
    RiskLimit,
    RiskLimitType,
    PortfolioSnapshot
)
from flashback.core.events import FillEvent, EventType
from flashback.market.orders import OrderSide, OrderType, TimeInForce


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test position creation."""
        pos = Position(symbol="AAPL")
        assert pos.symbol == "AAPL"
        assert pos.quantity == 0
        assert pos.avg_price == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short
    
    def test_position_properties(self):
        """Test position properties."""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            last_price=155.0
        )
        
        # Calculate unrealized PnL
        pos.unrealized_pnl = pos.calculate_unrealized_pnl()
        
        assert pos.market_value == 15500.0  # 100 * 155.0
        assert pos.total_pnl == 500.0  # (155 - 150) * 100
        assert not pos.is_flat
        assert pos.is_long
        assert not pos.is_short
    
    def test_short_position_properties(self):
        """Test short position properties."""
        pos = Position(
            symbol="AAPL",
            quantity=-100,
            avg_price=150.0,
            last_price=145.0
        )
        
        # Calculate unrealized PnL
        pos.unrealized_pnl = pos.calculate_unrealized_pnl()
        
        assert pos.market_value == -14500.0  # -100 * 145.0
        assert pos.total_pnl == 500.0  # (150 - 145) * 100
        assert not pos.is_flat
        assert not pos.is_long
        assert pos.is_short


class TestRiskLimit:
    """Test RiskLimit dataclass."""
    
    def test_risk_limit_creation(self):
        """Test risk limit creation."""
        limit = RiskLimit(
            limit_type=RiskLimitType.MAX_GROSS_EXPOSURE,
            value=1000000.0
        )
        
        assert limit.limit_type == RiskLimitType.MAX_GROSS_EXPOSURE
        assert limit.value == 1000000.0
        assert limit.symbol is None
        assert limit.enabled
        assert limit.auto_flatten


class TestPortfolioRiskManager:
    """Test PortfolioRiskManager class."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        assert portfolio.initial_cash == 1000000.0
        assert portfolio.cash == 1000000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.risk_limits) == 0
        assert portfolio.daily_pnl == 0.0
        assert portfolio.total_fees_paid == 0.0
    
    def test_update_position_buy(self):
        """Test updating position with buy order."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Create fill event for buy order
        fill_event = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        
        portfolio.update_position(fill_event, current_price=150.0)
        
        # Check position
        assert "AAPL" in portfolio.positions
        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 100
        assert pos.avg_price == 150.0
        assert pos.unrealized_pnl == 0.0  # No price change
        assert pos.total_fees == 1.0
        
        # Check cash
        expected_cash = 1000000.0 - (150.0 * 100 + 1.0)
        assert portfolio.cash == expected_cash
    
    def test_update_position_sell(self):
        """Test updating position with sell order."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # First buy some shares
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Then sell some shares
        sell_fill = FillEvent(
            timestamp=pd.Timestamp(2000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_2",
            symbol="AAPL",
            side=OrderSide.SELL.value,
            quantity=50,
            price=155.0,
            commission=0.5
        )
        portfolio.update_position(sell_fill, current_price=155.0)
        
        # Check position
        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 50  # 100 - 50
        assert pos.avg_price == 150.0  # Unchanged
        assert pos.realized_pnl == 250.0  # (155 - 150) * 50
        assert pos.unrealized_pnl == 250.0  # (155 - 150) * 50 remaining
        assert pos.total_fees == 1.5  # 1.0 + 0.5
    
    def test_position_flip_long_to_short(self):
        """Test position flipping from long to short."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Buy 100 shares
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Sell 150 shares (flip to short)
        sell_fill = FillEvent(
            timestamp=pd.Timestamp(2000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_2",
            symbol="AAPL",
            side=OrderSide.SELL.value,
            quantity=150,
            price=155.0,
            commission=1.5
        )
        portfolio.update_position(sell_fill, current_price=155.0)
        
        # Check position
        pos = portfolio.positions["AAPL"]
        assert pos.quantity == -50  # 100 - 150
        assert pos.avg_price == 155.0  # New average for short position
        assert pos.realized_pnl == 500.0  # (155 - 150) * 100
        assert pos.unrealized_pnl == 0.0  # No price change from 155
    
    def test_position_flip_short_to_long(self):
        """Test position flipping from short to long."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Sell 100 shares (short)
        sell_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.SELL.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(sell_fill, current_price=150.0)
        
        # Buy 150 shares (flip to long)
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(2000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_2",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=150,
            price=145.0,
            commission=1.5
        )
        portfolio.update_position(buy_fill, current_price=145.0)
        
        # Check position
        pos = portfolio.positions["AAPL"]
        assert pos.quantity == 50  # -100 + 150
        assert pos.avg_price == 145.0  # New average for long position
        assert pos.realized_pnl == 500.0  # (150 - 145) * 100
        assert pos.unrealized_pnl == 0.0  # No price change from 145
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Buy 100 shares at 150
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Update price to 155
        portfolio.update_market_prices({"AAPL": 155.0}, timestamp=2000)
        
        # Check unrealized PnL
        pos = portfolio.positions["AAPL"]
        assert pos.unrealized_pnl == 500.0  # (155 - 150) * 100
        assert pos.total_pnl == 500.0  # 0 realized + 500 unrealized
    
    def test_short_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation for short position."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Sell 100 shares at 150 (short)
        sell_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.SELL.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(sell_fill, current_price=150.0)
        
        # Update price to 145
        portfolio.update_market_prices({"AAPL": 145.0}, timestamp=2000)
        
        # Check unrealized PnL
        pos = portfolio.positions["AAPL"]
        assert pos.unrealized_pnl == 500.0  # (150 - 145) * 100
        assert pos.total_pnl == 500.0  # 0 realized + 500 unrealized
    
    def test_risk_limits_max_gross_exposure(self):
        """Test max gross exposure risk limit."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Add risk limit
        risk_limit = RiskLimit(
            limit_type=RiskLimitType.MAX_GROSS_EXPOSURE,
            value=50000.0
        )
        portfolio.risk_limits = [risk_limit]
        
        # Buy shares worth 60000 (exceeds limit)
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=400,  # 400 * 150 = 60000
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Check risk limits
        breached_limits = portfolio.check_risk_limits(1000)
        assert RiskLimitType.MAX_GROSS_EXPOSURE in breached_limits
    
    def test_risk_limits_max_position_per_symbol(self):
        """Test max position per symbol risk limit."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Add risk limit
        risk_limit = RiskLimit(
            limit_type=RiskLimitType.MAX_POSITION_PER_SYMBOL,
            value=100,
            symbol="AAPL"
        )
        portfolio.risk_limits = [risk_limit]
        
        # Buy 150 shares (exceeds limit)
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=150,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Check risk limits
        breached_limits = portfolio.check_risk_limits(1000)
        assert RiskLimitType.MAX_POSITION_PER_SYMBOL in breached_limits
    
    def test_risk_limits_daily_loss_limit(self):
        """Test daily loss limit."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Add risk limit
        risk_limit = RiskLimit(
            limit_type=RiskLimitType.DAILY_LOSS_LIMIT,
            value=10000.0
        )
        portfolio.risk_limits = [risk_limit]
        
        # Simulate daily loss
        portfolio.daily_pnl = -15000.0
        
        # Check risk limits
        breached_limits = portfolio.check_risk_limits(1000)
        assert RiskLimitType.DAILY_LOSS_LIMIT in breached_limits
    
    def test_get_flatten_orders(self):
        """Test generation of flatten orders."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Add risk limit
        risk_limit = RiskLimit(
            limit_type=RiskLimitType.MAX_GROSS_EXPOSURE,
            value=10000.0  # Lower limit to trigger breach
        )
        portfolio.risk_limits = [risk_limit]
        
        # Create positions
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Check for breaches and get flatten orders
        breached_limits = portfolio.check_risk_limits(1000)
        flatten_orders = portfolio.get_flatten_orders(breached_limits)
        
        assert len(flatten_orders) == 1
        order = flatten_orders[0]
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.strategy_id == "RISK_MANAGER"
    
    def test_portfolio_snapshot(self):
        """Test portfolio snapshot creation."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Create a position
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Update price
        portfolio.update_market_prices({"AAPL": 155.0}, timestamp=2000)
        
        # Take snapshot
        snapshot = portfolio.take_snapshot(2000)
        
        assert snapshot.timestamp == 2000
        assert snapshot.cash == portfolio.cash
        assert snapshot.total_market_value == 15500.0  # 100 * 155
        assert snapshot.total_unrealized_pnl == 500.0  # (155 - 150) * 100
        assert snapshot.gross_exposure == 15500.0
        assert snapshot.net_exposure == 15500.0
        assert "AAPL" in snapshot.positions
    
    def test_multiple_symbols(self):
        """Test portfolio with multiple symbols."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Buy AAPL
        aapl_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(aapl_fill, current_price=150.0)
        
        # Buy MSFT
        msft_fill = FillEvent(
            timestamp=pd.Timestamp(2000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_2",
            symbol="MSFT",
            side=OrderSide.BUY.value,
            quantity=50,
            price=300.0,
            commission=1.5
        )
        portfolio.update_position(msft_fill, current_price=300.0)
        
        # Check positions
        assert len(portfolio.positions) == 2
        assert "AAPL" in portfolio.positions
        assert "MSFT" in portfolio.positions
        
        # Check statistics
        stats = portfolio.get_statistics()
        assert stats["num_positions"] == 2
        assert "AAPL" in stats["positions"]
        assert "MSFT" in stats["positions"]
    
    def test_fees_and_slippage_tracking(self):
        """Test fees and slippage tracking."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Create fill with commission
        fill_event = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=5.0
        )
        
        portfolio.update_position(fill_event, current_price=150.0)
        
        # Check fees
        assert portfolio.total_fees_paid == 5.0
        assert portfolio.positions["AAPL"].total_fees == 5.0
    
    def test_reset_daily_pnl(self):
        """Test daily PnL reset."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Simulate some trading
        portfolio.cash = 950000.0
        portfolio.daily_pnl = -50000.0
        
        # Reset daily PnL
        portfolio.reset_daily_pnl()
        
        assert portfolio.daily_pnl == 0.0
        assert portfolio.daily_start_cash == 950000.0
    
    def test_get_statistics(self):
        """Test portfolio statistics generation."""
        portfolio = PortfolioRiskManager(initial_cash=1000000.0)
        
        # Create a position
        buy_fill = FillEvent(
            timestamp=pd.Timestamp(1000, unit='ns'),
            event_type=EventType.FILL,
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY.value,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        portfolio.update_position(buy_fill, current_price=150.0)
        
        # Update price
        portfolio.update_market_prices({"AAPL": 155.0}, timestamp=2000)
        
        # Get statistics
        stats = portfolio.get_statistics()
        
        assert "cash" in stats
        assert "total_pnl" in stats
        assert "gross_exposure" in stats
        assert "net_exposure" in stats
        assert "total_fees_paid" in stats
        assert "daily_pnl" in stats
        assert "num_positions" in stats
        assert "positions" in stats
        
        # Check specific values
        assert stats["total_pnl"] == 500.0  # (155 - 150) * 100
        assert stats["gross_exposure"] == 15500.0  # 100 * 155
        assert stats["num_positions"] == 1
        assert "AAPL" in stats["positions"]


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test portfolio snapshot creation."""
        positions = {"AAPL": Position(symbol="AAPL", quantity=100, avg_price=150.0)}
        
        snapshot = PortfolioSnapshot(
            timestamp=1000,
            cash=900000.0,
            total_market_value=15000.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            gross_exposure=15000.0,
            net_exposure=15000.0,
            positions=positions
        )
        
        assert snapshot.timestamp == 1000
        assert snapshot.cash == 900000.0
        assert snapshot.total_market_value == 15000.0
        assert len(snapshot.positions) == 1
        assert "AAPL" in snapshot.positions
