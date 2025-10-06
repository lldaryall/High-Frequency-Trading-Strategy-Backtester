"""Order execution engine."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..core.events import OrderEvent, FillEvent, CancelEvent, RejectEvent
from ..market.orders import Order, OrderSide, OrderType
from ..market.book import MatchingEngine
from ..market.fees import FeeModel, create_standard_fee_model
from ..market.latency import LatencyModel, create_standard_latency_model
from ..utils.logger import get_logger


class OrderExecutor:
    """High-performance order execution engine."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize order executor.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.matching_engine = MatchingEngine(config.get("matching", {}))
        self.fee_calculator = FeeCalculator(config.get("fees", {}))
        self.latency_model = LatencyModel(config.get("latency", {}))
        
        # Execution queue
        self.execution_queue: List[Order] = []
        self.priority_queue: List[Tuple[float, Order]] = []  # (priority, order)
        
        # Execution state
        self.is_running = False
        self.execution_thread = None
        
        # Performance tracking
        self.execution_metrics = {
            "orders_executed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_volume": 0,
            "total_fees": 0.0,
            "avg_execution_time_us": 0.0,
        }
        
    def submit_order(self, order: Order, priority: float = 1.0) -> str:
        """
        Submit an order for execution.
        
        Args:
            order: Order to execute
            priority: Execution priority (higher = more urgent)
            
        Returns:
            Order ID
        """
        # Add to priority queue
        self.priority_queue.append((priority, order))
        
        # Sort by priority (descending)
        self.priority_queue.sort(key=lambda x: x[0], reverse=True)
        
        self.logger.info(f"Order {order.order_id} submitted with priority {priority}")
        return order.order_id
        
    def execute_orders(self, order_books: Dict[str, Any]) -> List[FillEvent]:
        """
        Execute all pending orders.
        
        Args:
            order_books: Available order books
            
        Returns:
            List of fill events
        """
        fills = []
        
        # Process priority queue
        while self.priority_queue:
            priority, order = self.priority_queue.pop(0)
            
            try:
                # Execute order
                order_fills = self._execute_single_order(order, order_books)
                fills.extend(order_fills)
                
                # Update metrics
                self.execution_metrics["orders_executed"] += 1
                if order.is_filled:
                    self.execution_metrics["orders_filled"] += 1
                elif order.status == OrderStatus.CANCELLED:
                    self.execution_metrics["orders_cancelled"] += 1
                elif order.status == OrderStatus.REJECTED:
                    self.execution_metrics["orders_rejected"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error executing order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED
                self.execution_metrics["orders_rejected"] += 1
                
        return fills
        
    def _execute_single_order(self, order: Order, order_books: Dict[str, Any]) -> List[FillEvent]:
        """Execute a single order."""
        if order.symbol not in order_books:
            self.logger.warning(f"No order book for symbol {order.symbol}")
            order.status = OrderStatus.REJECTED
            return []
            
        order_book = order_books[order.symbol]
        
        # Record start time
        start_time = pd.Timestamp.now()
        
        # Execute order
        fills = self.matching_engine.match_order(order, order_book)
        
        # Process fills
        processed_fills = []
        for fill in fills:
            # Calculate fees
            fees = self.fee_calculator.calculate_fees(fill)
            fill.commission = fees["total"]
            
            # Calculate latency
            latency = self.latency_model.calculate_latency(order)
            fill.latency_us = latency
            
            # Update metrics
            self.execution_metrics["total_volume"] += fill.quantity
            self.execution_metrics["total_fees"] += fill.commission
            
            processed_fills.append(fill)
            
        # Calculate execution time
        execution_time = (pd.Timestamp.now() - start_time).total_seconds() * 1e6  # microseconds
        self._update_avg_execution_time(execution_time)
        
        return processed_fills
        
    def _update_avg_execution_time(self, execution_time: float) -> None:
        """Update average execution time."""
        current_avg = self.execution_metrics["avg_execution_time_us"]
        orders_executed = self.execution_metrics["orders_executed"]
        
        if orders_executed == 0:
            new_avg = execution_time
        else:
            new_avg = (current_avg * (orders_executed - 1) + execution_time) / orders_executed
            
        self.execution_metrics["avg_execution_time_us"] = new_avg
        
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled
        """
        # Find order in priority queue
        for i, (priority, order) in enumerate(self.priority_queue):
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                del self.priority_queue[i]
                self.execution_metrics["orders_cancelled"] += 1
                return True
                
        return False
        
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        total_orders = self.execution_metrics["orders_executed"]
        
        if total_orders == 0:
            return self.execution_metrics.copy()
            
        # Calculate additional metrics
        fill_rate = self.execution_metrics["orders_filled"] / total_orders
        cancel_rate = self.execution_metrics["orders_cancelled"] / total_orders
        reject_rate = self.execution_metrics["orders_rejected"] / total_orders
        
        # Calculate average trade size
        if self.execution_metrics["orders_filled"] > 0:
            avg_trade_size = self.execution_metrics["total_volume"] / self.execution_metrics["orders_filled"]
        else:
            avg_trade_size = 0.0
            
        # Calculate average fees
        if self.execution_metrics["orders_filled"] > 0:
            avg_fees = self.execution_metrics["total_fees"] / self.execution_metrics["orders_filled"]
        else:
            avg_fees = 0.0
            
        metrics = self.execution_metrics.copy()
        metrics.update({
            "fill_rate": fill_rate,
            "cancel_rate": cancel_rate,
            "reject_rate": reject_rate,
            "avg_trade_size": avg_trade_size,
            "avg_fees": avg_fees,
        })
        
        return metrics
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get execution queue status."""
        return {
            "queue_length": len(self.priority_queue),
            "is_running": self.is_running,
            "orders_in_queue": [order.order_id for _, order in self.priority_queue],
        }
        
    def clear_queue(self) -> None:
        """Clear execution queue."""
        self.priority_queue.clear()
        
    def reset(self) -> None:
        """Reset executor state."""
        self.clear_queue()
        self.is_running = False
        
        # Reset metrics
        for key in self.execution_metrics:
            self.execution_metrics[key] = 0
