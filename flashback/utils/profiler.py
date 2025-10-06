"""Profiling utilities for performance analysis."""

import cProfile
import pstats
import io
import time
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager
from pathlib import Path


class Profiler:
    """Simple profiler wrapper around cProfile."""
    
    def __init__(self, output_dir: str = "profiles"):
        """Initialize profiler.
        
        Args:
            output_dir: Directory to save profile results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = cProfile.Profile()
    
    @contextmanager
    def profile(self, name: str = "profile"):
        """Context manager for profiling code blocks.
        
        Args:
            name: Name for the profile output file
            
        Yields:
            Profiler instance
        """
        self.profiler.enable()
        start_time = time.time()
        
        try:
            yield self
        finally:
            self.profiler.disable()
            end_time = time.time()
            
            # Save profile results
            profile_file = self.output_dir / f"{name}.prof"
            self.profiler.dump_stats(str(profile_file))
            
            # Generate human-readable report
            report_file = self.output_dir / f"{name}_report.txt"
            self._generate_report(report_file, end_time - start_time)
            
            print(f"Profile saved to {profile_file}")
            print(f"Report saved to {report_file}")
    
    def _generate_report(self, output_file: Path, duration: float):
        """Generate human-readable profile report.
        
        Args:
            output_file: Output file path
            duration: Total duration in seconds
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        with open(output_file, 'w') as f:
            f.write(f"Profile Report - Duration: {duration:.3f}s\n")
            f.write("=" * 50 + "\n\n")
            f.write(s.getvalue())
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a single function call.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        with self.profile(func.__name__):
            return func(*args, **kwargs)


def profile_backtest(config_path: str, output_name: str = "backtest_profile"):
    """Profile a complete backtest run.
    
    Args:
        config_path: Path to backtest configuration
        output_name: Name for profile output
    """
    from flashback.config import load_config
    from flashback.cli.runner import BacktestRunner
    
    profiler = Profiler()
    
    with profiler.profile(output_name):
        config = load_config(config_path)
        runner = BacktestRunner(config)
        results = runner.run()
    
    return results


def profile_matching_engine(symbol: str = "AAPL", num_orders: int = 10000):
    """Profile the matching engine with synthetic orders.
    
    Args:
        symbol: Trading symbol
        num_orders: Number of orders to process
    """
    from flashback.market.book import MatchingEngine
    from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
    import numpy as np
    
    profiler = Profiler()
    
    def run_matching_test():
        # Create matching engine
        engine = MatchingEngine(symbol)
        
        # Generate synthetic orders
        orders = []
        for i in range(num_orders):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            price = 150.0 + np.random.normal(0, 0.1)
            quantity = np.random.randint(100, 1000)
            
            order = Order(
                order_id=f"order_{i}",
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                time_in_force=TimeInForce.DAY,
                timestamp=1000000000000000000 + i * 1000
            )
            orders.append(order)
        
        # Process orders
        for order in orders:
            engine.add_order(order)
        
        return engine
    
    with profiler.profile("matching_engine"):
        engine = run_matching_test()
    
    return engine


def profile_strategy_execution(symbol: str = "AAPL", num_events: int = 10000):
    """Profile strategy execution with synthetic market data.
    
    Args:
        symbol: Trading symbol
        num_events: Number of market events to process
    """
    from flashback.strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
    from flashback.core.events import MarketDataEvent, EventType
    import numpy as np
    
    profiler = Profiler()
    
    def run_strategy_test():
        # Create strategy
        config = MomentumImbalanceConfig(
            strategy_id="test",
            symbol=symbol,
            enabled=True,
            max_position=1000,
            max_order_size=100,
            risk_limits={},
            short_ema_period=5,
            long_ema_period=10,
            imbalance_threshold=0.6,
            take_profit_pct=0.02,
            stop_loss_pct=0.01
        )
        strategy = MomentumImbalanceStrategy(config)
        strategy.start()
        
        # Generate synthetic market data
        base_price = 150.0
        for i in range(num_events):
            price = base_price + np.random.normal(0, 0.1)
            volume = np.random.randint(100, 1000)
            
            event = MarketDataEvent(
                timestamp=1000000000000000000 + i * 1000,
                symbol=symbol,
                side="BUY",  # Not used for market data
                price=price,
                size=volume,
                event_type_str="TRADE",
                event_type=EventType.MARKET_DATA,
                data={
                    "mid_price": price,
                    "volume": volume,
                    "bid_price": price - 0.01,
                    "ask_price": price + 0.01
                }
            )
            
            strategy.on_bar(event)
        
        strategy.stop()
        return strategy
    
    with profiler.profile("strategy_execution"):
        strategy = run_strategy_test()
    
    return strategy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Flashback components")
    parser.add_argument("--component", choices=["backtest", "matching", "strategy"], 
                       default="backtest", help="Component to profile")
    parser.add_argument("--config", default="config/backtest.yaml", 
                       help="Backtest config file (for backtest profiling)")
    parser.add_argument("--orders", type=int, default=10000, 
                       help="Number of orders (for matching profiling)")
    parser.add_argument("--events", type=int, default=10000, 
                       help="Number of events (for strategy profiling)")
    
    args = parser.parse_args()
    
    if args.component == "backtest":
        profile_backtest(args.config)
    elif args.component == "matching":
        profile_matching_engine(num_orders=args.orders)
    elif args.component == "strategy":
        profile_strategy_execution(num_events=args.events)