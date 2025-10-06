"""Position management system."""

from typing import Any, Dict, List, Optional
import pandas as pd

from ..core.events import FillEvent
from ..utils.logger import get_logger


class PositionManager:
    """Manages trading positions and PnL."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize position manager.
        
        Args:
            config: Position management configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Position tracking
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.avg_prices: Dict[str, float] = {}  # symbol -> average price
        self.position_values: Dict[str, float] = {}  # symbol -> total value
        
        # Position limits
        self.max_position = config.get("max_position", 1000)
        self.max_position_per_symbol = config.get("max_position_per_symbol", 1000)
        
        # Position history
        self.position_history: List[Dict[str, Any]] = []
        
    def update_position(self, fill: FillEvent) -> None:
        """
        Update position after a fill.
        
        Args:
            fill: Fill event
        """
        symbol = fill.symbol
        quantity = fill.quantity if fill.side == 'B' else -fill.quantity
        
        # Initialize position if new symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0
            self.avg_prices[symbol] = 0.0
            self.position_values[symbol] = 0.0
            
        # Update position
        old_position = self.positions[symbol]
        new_position = old_position + quantity
        
        # Update average price
        if new_position != 0:
            if old_position == 0:
                # New position
                self.avg_prices[symbol] = fill.price
            elif (old_position > 0 and quantity > 0) or (old_position < 0 and quantity < 0):
                # Adding to position
                total_value = self.position_values[symbol] + (fill.price * quantity)
                self.avg_prices[symbol] = total_value / new_position
            else:
                # Reducing position
                self.avg_prices[symbol] = self.avg_prices[symbol]  # Keep same average price
                
        # Update position value
        self.position_values[symbol] = new_position * self.avg_prices[symbol]
        self.positions[symbol] = new_position
        
        # Record position change
        self._record_position_change(symbol, old_position, new_position, fill)
        
    def _record_position_change(self, symbol: str, old_position: int, 
                              new_position: int, fill: FillEvent) -> None:
        """Record position change in history."""
        change = {
            "timestamp": fill.timestamp,
            "symbol": symbol,
            "old_position": old_position,
            "new_position": new_position,
            "change": new_position - old_position,
            "price": fill.price,
            "side": fill.side,
            "quantity": fill.quantity,
        }
        self.position_history.append(change)
        
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0)
        
    def get_positions(self) -> Dict[str, int]:
        """Get all current positions."""
        return self.positions.copy()
        
    def get_average_price(self, symbol: str) -> float:
        """Get average price for a symbol."""
        return self.avg_prices.get(symbol, 0.0)
        
    def get_position_value(self, symbol: str) -> float:
        """Get position value for a symbol."""
        return self.position_values.get(symbol, 0.0)
        
    def get_total_position_value(self) -> float:
        """Get total position value across all symbols."""
        return sum(self.position_values.values())
        
    def is_position_closed(self, symbol: str) -> bool:
        """Check if position is closed."""
        return self.positions.get(symbol, 0) == 0
        
    def check_position_limit(self, order) -> bool:
        """
        Check if order would exceed position limits.
        
        Args:
            order: Order to check
            
        Returns:
            True if order passes position limits
        """
        symbol = order.symbol
        current_position = self.get_position(symbol)
        
        # Calculate new position after order
        quantity = order.quantity if order.side == 'B' else -order.quantity
        new_position = current_position + quantity
        
        # Check per-symbol limit
        if abs(new_position) > self.max_position_per_symbol:
            self.logger.warning(f"Position limit exceeded for {symbol}: {new_position}")
            return False
            
        # Check total position limit
        total_position = sum(abs(pos) for pos in self.positions.values())
        if total_position + abs(quantity) > self.max_position:
            self.logger.warning(f"Total position limit exceeded: {total_position + abs(quantity)}")
            return False
            
        return True
        
    def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary."""
        total_long = sum(pos for pos in self.positions.values() if pos > 0)
        total_short = sum(abs(pos) for pos in self.positions.values() if pos < 0)
        total_value = self.get_total_position_value()
        
        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_position": total_long - total_short,
            "total_value": total_value,
            "symbols": list(self.positions.keys()),
            "position_count": len([p for p in self.positions.values() if p != 0]),
        }
        
    def get_position_history(self) -> pd.DataFrame:
        """Get position history as DataFrame."""
        if not self.position_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.position_history)
        df.set_index("timestamp", inplace=True)
        return df
        
    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate unrealized PnL for all positions.
        
        Args:
            current_prices: Current market prices for each symbol
            
        Returns:
            Dictionary of unrealized PnL by symbol
        """
        unrealized_pnl = {}
        
        for symbol, position in self.positions.items():
            if position != 0 and symbol in current_prices:
                avg_price = self.avg_prices[symbol]
                current_price = current_prices[symbol]
                
                if position > 0:  # Long position
                    unrealized_pnl[symbol] = (current_price - avg_price) * position
                else:  # Short position
                    unrealized_pnl[symbol] = (avg_price - current_price) * abs(position)
                    
        return unrealized_pnl
        
    def reset(self) -> None:
        """Reset position manager."""
        self.positions.clear()
        self.avg_prices.clear()
        self.position_values.clear()
        self.position_history.clear()
