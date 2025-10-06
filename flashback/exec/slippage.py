"""Slippage modeling for realistic execution simulation."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from ..market.orders import Order, OrderSide, OrderType
from ..core.events import FillEvent


class SlippageModel:
    """Model execution slippage for realistic backtesting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize slippage model.
        
        Args:
            config: Slippage configuration
        """
        self.config = config or {}
        
        # Slippage parameters
        self.base_slippage_bps = self.config.get("base_slippage_bps", 1.0)  # 1 basis point
        self.volatility_multiplier = self.config.get("volatility_multiplier", 2.0)
        self.volume_impact = self.config.get("volume_impact", 0.5)
        self.market_impact = self.config.get("market_impact", 0.1)
        
        # Order type slippage
        self.order_type_slippage = {
            OrderType.MARKET: 1.0,
            OrderType.LIMIT: 0.5,
            OrderType.IOC: 0.8,
            OrderType.FOK: 1.2,
        }
        
        # Market conditions
        self.volatility_window = self.config.get("volatility_window", 20)
        self.volume_window = self.config.get("volume_window", 10)
        
    def calculate_slippage(self, order: Order, market_conditions: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order: Order to calculate slippage for
            market_conditions: Optional market conditions
            
        Returns:
            Slippage in basis points
        """
        if market_conditions is None:
            market_conditions = {}
            
        # Base slippage
        slippage = self.base_slippage_bps
        
        # Order type adjustment
        order_type_multiplier = self.order_type_slippage.get(order.order_type, 1.0)
        slippage *= order_type_multiplier
        
        # Volatility adjustment
        volatility = market_conditions.get("volatility", 0.0)
        if volatility > 0:
            volatility_adjustment = 1 + (volatility * self.volatility_multiplier)
            slippage *= volatility_adjustment
            
        # Volume impact
        order_size = order.quantity
        avg_volume = market_conditions.get("avg_volume", 1000)
        if avg_volume > 0:
            volume_ratio = order_size / avg_volume
            volume_adjustment = 1 + (volume_ratio * self.volume_impact)
            slippage *= volume_adjustment
            
        # Market impact
        spread = market_conditions.get("spread", 0.0)
        if spread > 0:
            spread_adjustment = 1 + (spread * self.market_impact)
            slippage *= spread_adjustment
            
        # Random component
        random_factor = np.random.normal(1.0, 0.1)
        slippage *= max(0.1, random_factor)  # Ensure positive slippage
        
        return slippage
        
    def apply_slippage(self, order: Order, fill_price: float, 
                      market_conditions: Optional[Dict[str, Any]] = None) -> float:
        """
        Apply slippage to fill price.
        
        Args:
            order: Original order
            fill_price: Original fill price
            market_conditions: Optional market conditions
            
        Returns:
            Adjusted fill price
        """
        slippage_bps = self.calculate_slippage(order, market_conditions)
        
        # Convert basis points to price adjustment
        slippage_factor = slippage_bps / 10000
        
        # Apply slippage based on order side
        if order.side == OrderSide.BUY:
            # Buy orders get worse prices (higher)
            adjusted_price = fill_price * (1 + slippage_factor)
        else:
            # Sell orders get worse prices (lower)
            adjusted_price = fill_price * (1 - slippage_factor)
            
        return adjusted_price
        
    def calculate_volume_impact(self, order: Order, order_book_depth: Dict[str, Any]) -> float:
        """
        Calculate volume impact on price.
        
        Args:
            order: Order to calculate impact for
            order_book_depth: Order book depth information
            
        Returns:
            Price impact factor
        """
        if order.side == OrderSide.BUY:
            # Calculate impact on ask side
            ask_levels = order_book_depth.get("asks", [])
            if not ask_levels:
                return 1.0
                
            # Calculate cumulative volume and price impact
            cumulative_volume = 0
            cumulative_impact = 0
            
            for price, volume in ask_levels:
                if cumulative_volume >= order.quantity:
                    break
                    
                volume_to_fill = min(volume, order.quantity - cumulative_volume)
                cumulative_volume += volume_to_fill
                
                # Price impact is proportional to volume filled at this level
                impact = volume_to_fill / order.quantity
                cumulative_impact += impact * (price / ask_levels[0][0] - 1)
                
        else:
            # Calculate impact on bid side
            bid_levels = order_book_depth.get("bids", [])
            if not bid_levels:
                return 1.0
                
            # Calculate cumulative volume and price impact
            cumulative_volume = 0
            cumulative_impact = 0
            
            for price, volume in bid_levels:
                if cumulative_volume >= order.quantity:
                    break
                    
                volume_to_fill = min(volume, order.quantity - cumulative_volume)
                cumulative_volume += volume_to_fill
                
                # Price impact is proportional to volume filled at this level
                impact = volume_to_fill / order.quantity
                cumulative_impact += impact * (1 - price / bid_levels[0][0])
                
        return 1 + cumulative_impact
        
    def calculate_market_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """
        Calculate market impact based on order size and market conditions.
        
        Args:
            order: Order to calculate impact for
            market_data: Market data including volume, volatility, etc.
            
        Returns:
            Market impact factor
        """
        # Get market data
        daily_volume = market_data.get("daily_volume", 1000000)
        volatility = market_data.get("volatility", 0.02)
        spread = market_data.get("spread", 0.001)
        
        # Calculate order size as percentage of daily volume
        size_ratio = order.quantity / daily_volume
        
        # Market impact is proportional to size ratio and volatility
        impact = size_ratio * volatility * 100  # Convert to basis points
        
        # Adjust for spread
        impact *= (1 + spread * 10)
        
        return impact
        
    def calculate_timing_slippage(self, order: Order, market_conditions: Dict[str, Any]) -> float:
        """
        Calculate slippage due to timing and market conditions.
        
        Args:
            order: Order to calculate slippage for
            market_conditions: Current market conditions
            
        Returns:
            Timing slippage in basis points
        """
        # Market volatility
        volatility = market_conditions.get("volatility", 0.02)
        
        # Time of day impact (higher volatility during market open/close)
        hour = order.timestamp.hour if order.timestamp else 9
        if hour in [9, 10, 15, 16]:  # Market open/close
            time_multiplier = 1.5
        else:
            time_multiplier = 1.0
            
        # News impact
        news_impact = market_conditions.get("news_impact", 0.0)
        
        # Calculate timing slippage
        timing_slippage = volatility * time_multiplier * 100  # Convert to basis points
        timing_slippage += news_impact * 10  # News impact in basis points
        
        return timing_slippage
        
    def get_slippage_summary(self, orders: List[Order], fills: List[FillEvent]) -> Dict[str, Any]:
        """
        Get slippage summary for a set of orders and fills.
        
        Args:
            orders: List of orders
            fills: List of fills
            
        Returns:
            Dictionary with slippage statistics
        """
        if not orders or not fills:
            return {}
            
        # Calculate slippage for each fill
        slippages = []
        for fill in fills:
            # Find corresponding order
            order = next((o for o in orders if o.order_id == fill.order_id), None)
            if order and order.price:
                slippage = abs(fill.price - order.price) / order.price
                slippages.append(slippage)
                
        if not slippages:
            return {}
            
        return {
            "avg_slippage": np.mean(slippages),
            "median_slippage": np.median(slippages),
            "max_slippage": np.max(slippages),
            "min_slippage": np.min(slippages),
            "std_slippage": np.std(slippages),
            "p95_slippage": np.percentile(slippages, 95),
            "p99_slippage": np.percentile(slippages, 99),
        }
