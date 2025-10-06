"""Main risk management system."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .positions import PositionManager
from .limits import RiskLimits
from ..core.events import OrderEvent, FillEvent
from ..utils.logger import get_logger


class RiskManager:
    """Main risk management system."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.position_manager = PositionManager(config.get("positions", {}))
        self.risk_limits = RiskLimits(config.get("limits", {}))
        
        # PnL tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.pnl_history: List[Dict[str, Any]] = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Trade tracking
        self.trades: List[Dict[str, Any]] = []
        self.win_trades = 0
        self.loss_trades = 0
        
    def check_order(self, order: OrderEvent) -> bool:
        """
        Check if an order passes risk limits.
        
        Args:
            order: Order to check
            
        Returns:
            True if order passes risk checks
        """
        # Check position limits
        if not self.position_manager.check_position_limit(order):
            self.logger.warning(f"Order {order.order_id} rejected: Position limit exceeded")
            return False
            
        # Check risk limits
        if not self.risk_limits.check_order(order, self.position_manager.get_positions()):
            self.logger.warning(f"Order {order.order_id} rejected: Risk limit exceeded")
            return False
            
        # Check drawdown limits
        if self.current_drawdown > self.risk_limits.max_drawdown:
            self.logger.warning(f"Order {order.order_id} rejected: Max drawdown exceeded")
            return False
            
        return True
        
    def update_position(self, fill: FillEvent) -> None:
        """
        Update position after a fill.
        
        Args:
            fill: Fill event
        """
        # Update position manager
        self.position_manager.update_position(fill)
        
        # Calculate PnL impact
        pnl_impact = self._calculate_pnl_impact(fill)
        self.total_pnl += pnl_impact
        
        # Update realized/unrealized PnL
        if self.position_manager.is_position_closed(fill.symbol):
            self.realized_pnl += pnl_impact
        else:
            self.unrealized_pnl += pnl_impact
            
        # Update PnL history
        self._update_pnl_history(fill)
        
        # Update risk metrics
        self._update_risk_metrics()
        
        # Track trade
        self._track_trade(fill)
        
    def _calculate_pnl_impact(self, fill: FillEvent) -> float:
        """Calculate PnL impact of a fill."""
        symbol = fill.symbol
        current_position = self.position_manager.get_position(symbol)
        
        if fill.side == 'B':  # Buy
            # Increase position, decrease cash
            return -fill.price * fill.quantity
        else:  # Sell
            # Decrease position, increase cash
            return fill.price * fill.quantity
            
    def _update_pnl_history(self, fill: FillEvent) -> None:
        """Update PnL history."""
        pnl_entry = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "pnl_impact": self._calculate_pnl_impact(fill),
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }
        self.pnl_history.append(pnl_entry)
        
    def _update_risk_metrics(self) -> None:
        """Update risk metrics."""
        # Update peak equity
        if self.total_pnl > self.peak_equity:
            self.peak_equity = self.total_pnl
            
        # Update drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.total_pnl) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
    def _track_trade(self, fill: FillEvent) -> None:
        """Track trade for statistics."""
        trade = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "pnl": self._calculate_pnl_impact(fill),
        }
        self.trades.append(trade)
        
        # Update win/loss counts
        if self._calculate_pnl_impact(fill) > 0:
            self.win_trades += 1
        else:
            self.loss_trades += 1
            
    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        return self.position_manager.get_positions()
        
    def get_position(self, symbol: str) -> int:
        """Get position for a specific symbol."""
        return self.position_manager.get_position(symbol)
        
    def get_pnl(self) -> Dict[str, float]:
        """Get PnL summary."""
        return {
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
        }
        
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics."""
        total_trades = len(self.trades)
        win_rate = self.win_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate average win/loss
        if self.trades:
            wins = [t["pnl"] for t in self.trades if t["pnl"] > 0]
            losses = [t["pnl"] for t in self.trades if t["pnl"] < 0]
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0
            
        return {
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "total_trades": total_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
        }
        
    def get_pnl_series(self) -> pd.DataFrame:
        """Get PnL time series."""
        if not self.pnl_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.pnl_history)
        df.set_index("timestamp", inplace=True)
        return df
        
    def get_trade_blotter(self) -> pd.DataFrame:
        """Get trade blotter."""
        if not self.trades:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.trades)
        df.set_index("timestamp", inplace=True)
        return df
        
    def reset(self) -> None:
        """Reset risk manager state."""
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.pnl_history.clear()
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.trades.clear()
        self.win_trades = 0
        self.loss_trades = 0
        
        # Reset components
        self.position_manager.reset()
        self.risk_limits.reset()
