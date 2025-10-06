"""
Portfolio risk management system for flashback HFT backtesting engine.

This module provides position tracking, PnL calculations, and risk limit enforcement
for the high-frequency trading backtesting system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pandas as pd
from enum import Enum

from ..core.events import FillEvent, OrderEvent
from ..market.orders import OrderSide, OrderType, TimeInForce
from ..exec.router import NewOrder, CancelOrder


class RiskLimitType(Enum):
    """Types of risk limits."""
    MAX_GROSS_EXPOSURE = "max_gross_exposure"
    MAX_POSITION_PER_SYMBOL = "max_position_per_symbol"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_LEVERAGE = "max_leverage"
    MAX_CORRELATION = "max_correlation"


@dataclass
class Position:
    """Represents a position in a single symbol."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    last_price: Optional[float] = None
    last_update_time: Optional[int] = None
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        if self.last_price is None:
            return 0.0
        return self.quantity * self.last_price
    
    @property
    def total_pnl(self) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL based on current price."""
        if self.last_price is None or self.quantity == 0:
            return 0.0
        
        if self.is_long:
            return (self.last_price - self.avg_price) * self.quantity
        else:
            return (self.avg_price - self.last_price) * abs(self.quantity)
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (zero quantity)."""
        return self.quantity == 0
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0


@dataclass
class RiskLimit:
    """Represents a risk limit configuration."""
    limit_type: RiskLimitType
    value: float
    symbol: Optional[str] = None  # None for portfolio-wide limits
    enabled: bool = True
    auto_flatten: bool = True  # Whether to auto-flatten when breached


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: int
    cash: float
    total_market_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    gross_exposure: float
    net_exposure: float
    positions: Dict[str, Position]
    risk_breaches: List[RiskLimitType] = field(default_factory=list)


class PortfolioRiskManager:
    """
    Portfolio risk management system.
    
    Tracks positions per symbol, cash, realized/unrealized PnL,
    and enforces risk limits with auto-flattening capabilities.
    """
    
    def __init__(
        self,
        initial_cash: float = 1000000.0,
        risk_limits: Optional[List[RiskLimit]] = None
    ):
        """
        Initialize portfolio risk manager.
        
        Args:
            initial_cash: Starting cash balance
            risk_limits: List of risk limits to enforce
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.risk_limits = risk_limits or []
        self.daily_pnl = 0.0
        self.daily_start_cash = initial_cash
        self.snapshots: List[PortfolioSnapshot] = []
        
        # Fee and slippage tracking
        self.total_fees_paid = 0.0
        self.total_slippage = 0.0
        
        # Risk breach tracking
        self.risk_breaches: List[Tuple[int, RiskLimitType, str]] = []
        
    def update_position(
        self,
        fill_event: FillEvent,
        current_price: Optional[float] = None
    ) -> None:
        """
        Update position based on fill event.
        
        Args:
            fill_event: Fill event containing trade details
            current_price: Current market price for unrealized PnL calculation
        """
        symbol = fill_event.symbol
        side = fill_event.side
        quantity = fill_event.quantity
        price = fill_event.price
        commission = fill_event.commission
        timestamp = fill_event.timestamp.value
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Update position quantity and average price
        if side == OrderSide.BUY.value:
            # Buying - increase position
            if position.quantity >= 0:
                # Adding to long position or going long
                new_quantity = position.quantity + quantity
                if new_quantity > 0:
                    # Calculate new average price
                    position.avg_price = (
                        (position.avg_price * position.quantity + price * quantity) / new_quantity
                    )
                position.quantity = new_quantity
            else:
                # Covering short position
                if quantity <= abs(position.quantity):
                    # Partial or full cover
                    position.quantity += quantity
                    # Realize PnL on covered portion
                    pnl = (position.avg_price - price) * quantity
                    position.realized_pnl += pnl
                else:
                    # Over-covering, going long
                    covered_qty = abs(position.quantity)
                    remaining_qty = quantity - covered_qty
                    # Realize PnL on covered portion
                    pnl = (position.avg_price - price) * covered_qty
                    position.realized_pnl += pnl
                    # Set new long position
                    position.quantity = remaining_qty
                    position.avg_price = price
        else:
            # Selling - decrease position
            if position.quantity <= 0:
                # Adding to short position or going short
                new_quantity = position.quantity - quantity
                if new_quantity < 0:
                    # Calculate new average price
                    position.avg_price = (
                        (position.avg_price * abs(position.quantity) + price * quantity) / abs(new_quantity)
                    )
                position.quantity = new_quantity
            else:
                # Reducing long position
                if quantity <= position.quantity:
                    # Partial or full reduction
                    position.quantity -= quantity
                    # Realize PnL on reduced portion
                    pnl = (price - position.avg_price) * quantity
                    position.realized_pnl += pnl
                else:
                    # Over-selling, going short
                    reduced_qty = position.quantity
                    remaining_qty = quantity - reduced_qty
                    # Realize PnL on reduced portion
                    pnl = (price - position.avg_price) * reduced_qty
                    position.realized_pnl += pnl
                    # Set new short position
                    position.quantity = -remaining_qty
                    position.avg_price = price
        
        # Update fees and timestamps
        position.total_fees += commission
        self.total_fees_paid += commission
        position.last_price = current_price or price
        position.last_update_time = timestamp
        
        # Update cash (simplified - assumes all trades are cash-settled)
        if side == OrderSide.BUY.value:
            self.cash -= (price * quantity + commission)
        else:
            self.cash += (price * quantity - commission)
        
        # Update daily PnL
        self.daily_pnl = self.cash - self.daily_start_cash
        
        # Update unrealized PnL if current price is available
        if current_price is not None:
            self._update_unrealized_pnl(symbol, current_price)
    
    def _update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized PnL for a position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.last_price = current_price
        position.unrealized_pnl = position.calculate_unrealized_pnl()
    
    def update_market_prices(self, prices: Dict[str, float], timestamp: int) -> None:
        """
        Update market prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> current price
            timestamp: Current timestamp
        """
        for symbol, price in prices.items():
            self._update_unrealized_pnl(symbol, price)
            if symbol in self.positions:
                self.positions[symbol].last_update_time = timestamp
    
    def check_risk_limits(self, timestamp: int) -> List[RiskLimitType]:
        """
        Check all risk limits and return breached limits.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of breached risk limit types
        """
        breached_limits = []
        
        for limit in self.risk_limits:
            if not limit.enabled:
                continue
                
            if limit.limit_type == RiskLimitType.MAX_GROSS_EXPOSURE:
                if self.gross_exposure > limit.value:
                    breached_limits.append(limit.limit_type)
                    self.risk_breaches.append((timestamp, limit.limit_type, f"Gross exposure {self.gross_exposure} exceeds limit {limit.value}"))
            
            elif limit.limit_type == RiskLimitType.MAX_POSITION_PER_SYMBOL:
                if limit.symbol and limit.symbol in self.positions:
                    position = self.positions[limit.symbol]
                    if abs(position.quantity) > limit.value:
                        breached_limits.append(limit.limit_type)
                        self.risk_breaches.append((timestamp, limit.limit_type, f"Position {position.quantity} in {limit.symbol} exceeds limit {limit.value}"))
            
            elif limit.limit_type == RiskLimitType.DAILY_LOSS_LIMIT:
                if self.daily_pnl < -limit.value:
                    breached_limits.append(limit.limit_type)
                    self.risk_breaches.append((timestamp, limit.limit_type, f"Daily PnL {self.daily_pnl} exceeds loss limit {limit.value}"))
        
        return breached_limits
    
    def get_flatten_orders(self, breached_limits: List[RiskLimitType]) -> List[NewOrder]:
        """
        Generate orders to flatten positions when risk limits are breached.
        
        Args:
            breached_limits: List of breached risk limit types
            
        Returns:
            List of flatten orders
        """
        flatten_orders = []
        
        for limit_type in breached_limits:
            if limit_type == RiskLimitType.MAX_GROSS_EXPOSURE:
                # Flatten all positions
                for symbol, position in self.positions.items():
                    if not position.is_flat:
                        order = self._create_flatten_order(symbol, position)
                        if order:
                            flatten_orders.append(order)
            
            elif limit_type == RiskLimitType.MAX_POSITION_PER_SYMBOL:
                # Flatten specific symbol positions
                for limit in self.risk_limits:
                    if (limit.limit_type == limit_type and 
                        limit.symbol and 
                        limit.symbol in self.positions):
                        position = self.positions[limit.symbol]
                        if not position.is_flat:
                            order = self._create_flatten_order(limit.symbol, position)
                            if order:
                                flatten_orders.append(order)
            
            elif limit_type == RiskLimitType.DAILY_LOSS_LIMIT:
                # Flatten all positions
                for symbol, position in self.positions.items():
                    if not position.is_flat:
                        order = self._create_flatten_order(symbol, position)
                        if order:
                            flatten_orders.append(order)
        
        return flatten_orders
    
    def _create_flatten_order(self, symbol: str, position: Position) -> Optional[NewOrder]:
        """Create a flatten order for a position."""
        if position.is_flat:
            return None
        
        side = OrderSide.SELL if position.is_long else OrderSide.BUY
        quantity = abs(position.quantity)
        
        return NewOrder(
            intent_id=f"FLATTEN_{symbol}_{position.last_update_time}",
            timestamp=position.last_update_time or 0,
            symbol=symbol,
            side=side,
            price=0.0,  # Market order
            quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="RISK_MANAGER"
        )
    
    def take_snapshot(self, timestamp: int) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Portfolio snapshot
        """
        # Calculate portfolio metrics
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_pnl = total_unrealized_pnl + total_realized_pnl
        
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        net_exposure = sum(pos.market_value for pos in self.positions.values())
        
        # Check for risk breaches
        breached_limits = self.check_risk_limits(timestamp)
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_pnl,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            positions=self.positions.copy(),
            risk_breaches=breached_limits
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def reset_daily_pnl(self) -> None:
        """Reset daily PnL tracking (call at start of each day)."""
        self.daily_pnl = 0.0
        self.daily_start_cash = self.cash
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        return sum(pos.total_pnl for pos in self.positions.values())
    
    def get_gross_exposure(self) -> float:
        """Get gross exposure (sum of absolute position values)."""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_net_exposure(self) -> float:
        """Get net exposure (sum of position values)."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def gross_exposure(self) -> float:
        """Get gross exposure (sum of absolute position values)."""
        return self.get_gross_exposure()
    
    @property
    def net_exposure(self) -> float:
        """Get net exposure (sum of position values)."""
        return self.get_net_exposure()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get portfolio statistics."""
        total_pnl = self.get_total_pnl()
        gross_exposure = self.get_gross_exposure()
        net_exposure = self.get_net_exposure()
        
        return {
            "cash": self.cash,
            "total_pnl": total_pnl,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "total_fees_paid": self.total_fees_paid,
            "total_slippage": self.total_slippage,
            "daily_pnl": self.daily_pnl,
            "num_positions": len([p for p in self.positions.values() if not p.is_flat]),
            "num_risk_breaches": len(self.risk_breaches),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_pnl": pos.total_pnl,
                    "total_fees": pos.total_fees
                }
                for symbol, pos in self.positions.items()
                if not pos.is_flat
            }
        }
