"""
Performance profiling utilities for Flashback HFT backtesting engine.

This module provides profiling tools to identify performance bottlenecks
and optimize hot paths in the backtesting engine.
"""

import cProfile
import pstats
import io
import time
from typing import Dict, Any, Callable, Optional
from pathlib import Path
import pandas as pd
import numpy as np


class PerformanceProfiler:
    """Performance profiler for identifying hot paths."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        """Initialize the profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiles = {}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a single function call."""
        print(f"Profiling function: {func.__name__}")
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the function
        start_time = time.time()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        end_time = time.time()
        
        # Get statistics
        stats = self._get_profile_stats(profiler)
        stats['execution_time'] = end_time - start_time
        stats['function_name'] = func.__name__
        
        # Save profile
        profile_file = self.output_dir / f"{func.__name__}_profile.prof"
        profiler.dump_stats(str(profile_file))
        
        self.profiles[func.__name__] = stats
        return stats
    
    def profile_matching_engine(self, num_orders: int = 10000) -> Dict[str, Any]:
        """Profile the matching engine with synthetic orders."""
        from flashback.market.book import MatchingEngine
        from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
        import pandas as pd
        
        def run_matching_test():
            engine = MatchingEngine('TEST')
            orders = []
            
            # Generate synthetic orders
            np.random.seed(42)
            for i in range(num_orders):
                side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL
                price = 150.0 + np.random.normal(0, 1.0)
                quantity = np.random.randint(10, 1000)
                
                order = Order(
                    order_id=f'order_{i}',
                    timestamp=pd.Timestamp.now().value + i * 1000,
                    symbol='TEST',
                    side=side,
                    price=price,
                    quantity=quantity,
                    time_in_force=TimeInForce.DAY,
                    order_type=OrderType.LIMIT
                )
                orders.append(order)
            
            # Process orders
            total_fills = 0
            for order in orders:
                fills = engine.add_order(order)
                total_fills += len(fills)
            
            return total_fills
        
        return self.profile_function(run_matching_test)
    
    def profile_strategy_calculations(self, num_events: int = 50000) -> Dict[str, Any]:
        """Profile strategy calculation hot paths."""
        from flashback.strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
        from flashback.core.events import MarketDataEvent, EventType
        import pandas as pd
        
        def run_strategy_test():
            config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
            strategy = MomentumImbalanceStrategy(config)
            
            # Generate synthetic market data
            np.random.seed(42)
            for i in range(num_events):
                price = 150.0 + np.random.normal(0, 0.1)
                size = np.random.randint(100, 1000)
                side = 'B' if np.random.random() < 0.5 else 'S'
                
                event = MarketDataEvent(
                    symbol='AAPL',
                    side=side,
                    price=price,
                    size=size,
                    event_type_str='TICK',
                    event_type=EventType.MARKET_DATA
                )
                
                # Process event
                strategy.on_bar(event)
            
            return strategy.get_statistics()
        
        return self.profile_function(run_strategy_test)
    
    def profile_ema_calculation(self, num_values: int = 100000) -> Dict[str, Any]:
        """Profile EMA calculation performance."""
        from flashback.strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
        
        def run_ema_test():
            config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
            strategy = MomentumImbalanceStrategy(config)
            
            # Generate synthetic price data
            np.random.seed(42)
            prices = 150.0 + np.cumsum(np.random.normal(0, 0.01, num_values))
            
            # Calculate EMAs
            for price in prices:
                strategy._update_emas(price)
            
            return strategy.get_statistics()
        
        return self.profile_function(run_ema_test)
    
    def profile_imbalance_calculation(self, num_trades: int = 100000) -> Dict[str, Any]:
        """Profile order flow imbalance calculation."""
        from flashback.strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
        from flashback.core.events import MarketDataEvent, EventType
        
        def run_imbalance_test():
            config = MomentumImbalanceConfig(strategy_id="test", symbol="AAPL")
            strategy = MomentumImbalanceStrategy(config)
            
            # Generate synthetic trade data
            np.random.seed(42)
            for i in range(num_trades):
                price = 150.0 + np.random.normal(0, 0.1)
                size = np.random.randint(100, 1000)
                side = 'B' if np.random.random() < 0.5 else 'S'
                
                event = MarketDataEvent(
                    symbol='AAPL',
                    side=side,
                    price=price,
                    size=size,
                    event_type_str='TICK',
                    event_type=EventType.MARKET_DATA
                )
                
                # Update imbalance
                strategy._update_order_flow_imbalance(event)
            
            return strategy.get_statistics()
        
        return self.profile_function(run_imbalance_test)
    
    def _get_profile_stats(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Extract statistics from profiler."""
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        stats_text = s.getvalue()
        
        # Parse key statistics
        lines = stats_text.split('\n')
        total_calls = 0
        total_time = 0.0
        hot_functions = []
        
        for line in lines[5:25]:  # Skip header lines
            if line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        calls = int(parts[0])
                        total_time_val = float(parts[3])
                        function_name = ' '.join(parts[5:]) if len(parts) > 5 else parts[4]
                        
                        total_calls += calls
                        total_time += total_time_val
                        
                        hot_functions.append({
                            'calls': calls,
                            'total_time': total_time_val,
                            'function': function_name
                        })
                    except (ValueError, IndexError):
                        continue
        
        return {
            'total_calls': total_calls,
            'total_time': total_time,
            'hot_functions': hot_functions,
            'full_stats': stats_text
        }
    
    def compare_cython_vs_python(self, num_orders: int = 10000) -> Dict[str, Any]:
        """Compare Cython vs Python performance."""
        print("Comparing Cython vs Python performance...")
        
        # Test Python implementation
        python_stats = self.profile_matching_engine(num_orders)
        
        # Test Cython implementation (if available)
        cython_stats = None
        try:
            from flashback.market._match import CythonMatchingEngine
            
            def run_cython_test():
                engine = CythonMatchingEngine()
                
                # Generate synthetic orders
                np.random.seed(42)
                for i in range(num_orders):
                    side = 1 if np.random.random() < 0.5 else -1
                    price = 150.0 + np.random.normal(0, 1.0)
                    quantity = np.random.randint(10, 1000)
                    
                    fills = engine.add_order(
                        order_id=i,
                        timestamp=1000000000 + i * 1000,
                        side='BUY' if side == 1 else 'SELL',
                        price=price,
                        quantity=quantity
                    )
                
                return engine.get_statistics()
            
            cython_stats = self.profile_function(run_cython_test)
            
        except ImportError:
            print("Cython extension not available. Install with: python setup_cython.py build_ext --inplace")
        
        return {
            'python': python_stats,
            'cython': cython_stats,
            'speedup': (python_stats['execution_time'] / cython_stats['execution_time']) 
                      if cython_stats else None
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive profiling report."""
        report_lines = [
            "# Flashback HFT Backtesting Engine - Performance Profile Report",
            "",
            f"Generated: {pd.Timestamp.now()}",
            "",
            "## Summary",
            ""
        ]
        
        if self.profiles:
            total_time = sum(stats.get('execution_time', 0) for stats in self.profiles.values())
            report_lines.append(f"Total execution time: {total_time:.3f} seconds")
            report_lines.append(f"Number of profiled functions: {len(self.profiles)}")
            report_lines.append("")
        
        # Add individual function reports
        for func_name, stats in self.profiles.items():
            report_lines.extend([
                f"## {func_name}",
                f"Execution time: {stats.get('execution_time', 0):.3f} seconds",
                f"Total calls: {stats.get('total_calls', 0):,}",
                f"Total time: {stats.get('total_time', 0):.3f} seconds",
                "",
                "### Hot Functions:",
                ""
            ])
            
            for func in stats.get('hot_functions', [])[:10]:
                report_lines.append(
                    f"- {func['function']}: {func['calls']:,} calls, "
                    f"{func['total_time']:.3f}s"
                )
            
            report_lines.extend(["", "---", ""])
        
        # Save report
        report_file = self.output_dir / "performance_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Performance report saved to: {report_file}")
        return str(report_file)


def run_comprehensive_profile():
    """Run comprehensive performance profiling."""
    profiler = PerformanceProfiler()
    
    print("🔍 Starting comprehensive performance profiling...")
    
    # Profile key components
    print("\n1. Profiling matching engine...")
    profiler.profile_matching_engine(10000)
    
    print("\n2. Profiling strategy calculations...")
    profiler.profile_strategy_calculations(50000)
    
    print("\n3. Profiling EMA calculations...")
    profiler.profile_ema_calculation(100000)
    
    print("\n4. Profiling imbalance calculations...")
    profiler.profile_imbalance_calculation(100000)
    
    print("\n5. Comparing Cython vs Python...")
    profiler.compare_cython_vs_python(10000)
    
    print("\n6. Generating report...")
    report_file = profiler.generate_report()
    
    print(f"\n✅ Profiling complete! Report saved to: {report_file}")


if __name__ == "__main__":
    run_comprehensive_profile()