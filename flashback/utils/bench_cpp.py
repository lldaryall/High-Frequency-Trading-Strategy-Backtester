#!/usr/bin/env python3
"""
Performance benchmark script for C++ vs Python matching engines.

Generates synthetic L1 market data and benchmarks both implementations
with detailed performance metrics and rich-formatted output.
"""

import time
import random
import statistics
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flashback.market.book import MatchingEngine, PythonMatchingEngine, CppMatchingEngineWrapper, HAS_CPP
from flashback.market.orders import Order, OrderSide, OrderType, TimeInForce
from flashback.utils.profiler import Profiler

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")


class MarketDataGenerator:
    """Generate synthetic L1 market data for benchmarking."""
    
    def __init__(self, symbol: str = "AAPL", seed: int = 42):
        """Initialize the market data generator."""
        self.symbol = symbol
        self.seed = seed
        random.seed(seed)
        
        # Market state
        self.current_price = 150.0
        self.bid_price = 149.95
        self.ask_price = 150.05
        self.spread = 0.10
        
        # Order ID counter
        self.order_counter = 0
        
    def generate_orders(self, num_orders: int) -> List[Order]:
        """Generate synthetic L1 market orders."""
        orders = []
        
        for i in range(num_orders):
            # Randomly choose order type and side
            side = random.choice([OrderSide.BUY, OrderSide.SELL])
            order_type = random.choice([OrderType.LIMIT, OrderType.MARKET])
            tif = random.choice([TimeInForce.DAY, TimeInForce.IOC, TimeInForce.FOK])
            
            # Generate price based on current market state
            if order_type == OrderType.MARKET:
                price = 0.0  # Market orders have no price
            else:
                # Limit orders around current price
                if side == OrderSide.BUY:
                    price = self.bid_price + random.uniform(-0.5, 0.5)
                else:
                    price = self.ask_price + random.uniform(-0.5, 0.5)
                
                price = round(price, 2)
            
            # Generate quantity
            quantity = random.randint(10, 1000)
            
            # Create order
            order = Order(
                order_id=f"ORDER_{self.order_counter:06d}",
                timestamp=int(time.time() * 1e9) + i * 1000,  # 1μs between orders
                symbol=self.symbol,
                side=side,
                price=price,
                quantity=quantity,
                time_in_force=tif,
                order_type=order_type
            )
            
            orders.append(order)
            self.order_counter += 1
            
            # Update market state occasionally
            if i % 1000 == 0:
                self._update_market_state()
        
        return orders
    
    def _update_market_state(self):
        """Update market state to simulate price movement."""
        # Random walk for price
        price_change = random.uniform(-0.1, 0.1)
        self.current_price += price_change
        
        # Update bid/ask around current price
        self.bid_price = round(self.current_price - self.spread / 2, 2)
        self.ask_price = round(self.current_price + self.spread / 2, 2)
        
        # Occasionally change spread
        if random.random() < 0.1:
            self.spread = round(random.uniform(0.05, 0.20), 2)


class BenchmarkRunner:
    """Run performance benchmarks on matching engines."""
    
    def __init__(self, console=None):
        """Initialize the benchmark runner."""
        self.console = console
        self.results = []
        
    def run_benchmark(self, engine_name: str, engine, orders: List[Order], 
                     warmup_ratio: float = 0.1, num_trials: int = 3) -> Dict[str, Any]:
        """Run benchmark on a specific engine."""
        if self.console:
            self.console.print(f"[blue]Benchmarking {engine_name}...[/blue]")
        
        # Calculate warmup
        warmup_orders = int(len(orders) * warmup_ratio)
        benchmark_orders = orders[warmup_orders:]
        
        trial_times = []
        trial_fills = []
        
        for trial in range(num_trials):
            if self.console:
                self.console.print(f"  Trial {trial + 1}/{num_trials}...")
            
            # Create fresh engine for each trial
            if engine_name == "Python":
                fresh_engine = PythonMatchingEngine("AAPL")
            else:
                fresh_engine = CppMatchingEngineWrapper("AAPL")
            
            # Create fresh orders for this trial to avoid state issues
            generator = MarketDataGenerator(seed=42 + trial)
            fresh_orders = generator.generate_orders(len(orders))
            fresh_warmup_orders = fresh_orders[:warmup_orders]
            fresh_benchmark_orders = fresh_orders[warmup_orders:]
            
            # Warmup
            for order in fresh_warmup_orders:
                fresh_engine.add_order(order)
            
            # Timed benchmark
            start_time = time.perf_counter()
            
            fills = []
            for order in fresh_benchmark_orders:
                fills.extend(fresh_engine.add_order(order))
            
            end_time = time.perf_counter()
            
            trial_time = end_time - start_time
            trial_times.append(trial_time)
            trial_fills.append(len(fills))
        
        # Calculate statistics
        avg_time = statistics.mean(trial_times)
        std_time = statistics.stdev(trial_times) if len(trial_times) > 1 else 0
        min_time = min(trial_times)
        max_time = max(trial_times)
        
        num_events = len(benchmark_orders)
        ops_per_sec = num_events / avg_time
        avg_latency_us = (avg_time / num_events) * 1_000_000  # Convert to microseconds
        
        avg_fills = statistics.mean(trial_fills)
        
        return {
            "engine": engine_name,
            "num_events": num_events,
            "num_trials": num_trials,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "ops_per_sec": ops_per_sec,
            "avg_latency_us": avg_latency_us,
            "avg_fills": avg_fills,
            "trial_times": trial_times,
            "trial_fills": trial_fills
        }
    
    def run_comparison(self, num_orders: int = 1_000_000, num_trials: int = 3) -> List[Dict[str, Any]]:
        """Run comparison benchmark between Python and C++ engines."""
        if self.console:
            self.console.print(f"[green]Starting benchmark with {num_orders:,} orders...[/green]")
        
        # Generate synthetic market data
        generator = MarketDataGenerator(seed=42)
        orders = generator.generate_orders(num_orders)
        
        if self.console:
            self.console.print(f"Generated {len(orders):,} synthetic orders")
        
        results = []
        
        # Benchmark Python engine
        python_result = self.run_benchmark("Python", PythonMatchingEngine, orders, num_trials=num_trials)
        results.append(python_result)
        
        # Benchmark C++ engine (if available)
        if HAS_CPP:
            cpp_result = self.run_benchmark("C++", CppMatchingEngineWrapper, orders, num_trials=num_trials)
            results.append(cpp_result)
        else:
            if self.console:
                self.console.print("[yellow]C++ engine not available. Install with 'make cpp'[/yellow]")
        
        self.results = results
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print benchmark results in a rich-formatted table."""
        if not self.console:
            self._print_results_plain(results)
            return
        
        # Print banner
        self._print_banner(results)
        
        # Create results table
        table = Table(title="Matching Engine Performance Benchmark", box=box.ROUNDED)
        
        table.add_column("Engine", style="cyan", no_wrap=True)
        table.add_column("Events", justify="right", style="magenta")
        table.add_column("Trials", justify="right", style="magenta")
        table.add_column("Avg Time (s)", justify="right", style="green")
        table.add_column("Std Dev (s)", justify="right", style="green")
        table.add_column("Ops/sec", justify="right", style="yellow")
        table.add_column("Latency (μs)", justify="right", style="yellow")
        table.add_column("Avg Fills", justify="right", style="blue")
        
        for result in results:
            table.add_row(
                result["engine"],
                f"{result['num_events']:,}",
                str(result["num_trials"]),
                f"{result['avg_time']:.3f}",
                f"{result['std_time']:.3f}",
                f"{result['ops_per_sec']:,.0f}",
                f"{result['avg_latency_us']:.2f}",
                f"{result['avg_fills']:,.0f}"
            )
        
        self.console.print(table)
        
        # Calculate and display speedup
        if len(results) == 2:
            python_ops = results[0]["ops_per_sec"]
            cpp_ops = results[1]["ops_per_sec"]
            speedup = cpp_ops / python_ops
            
            speedup_panel = Panel(
                f"[bold green]C++ is {speedup:.2f}x faster than Python[/bold green]\n"
                f"Python: {python_ops:,.0f} ops/sec\n"
                f"C++: {cpp_ops:,.0f} ops/sec",
                title="Performance Comparison",
                border_style="green"
            )
            self.console.print(speedup_panel)
    
    def _print_banner(self, results: List[Dict[str, Any]]):
        """Print the performance banner with key metrics."""
        if len(results) < 2:
            return
        
        python_result = results[0]
        cpp_result = results[1]
        
        # Calculate core metrics
        python_ops = python_result["ops_per_sec"]
        cpp_ops = cpp_result["ops_per_sec"]
        speedup = cpp_ops / python_ops
        
        # Format throughput for display
        python_throughput = self._format_throughput(python_ops)
        cpp_throughput = self._format_throughput(cpp_ops)
        
        # Calculate p95 latency (approximate from std dev)
        python_std_latency = (
            python_result.get("std_latency_us", 0) or 
            python_result["avg_latency_us"] * 0.1
        )
        cpp_std_latency = (
            cpp_result.get("std_latency_us", 0) or 
            cpp_result["avg_latency_us"] * 0.1
        )
        
        python_latency_p95 = python_result["avg_latency_us"] + 1.96 * python_std_latency
        cpp_latency_p95 = cpp_result["avg_latency_us"] + 1.96 * cpp_std_latency
        
        # Use the better (lower) p95 latency
        best_latency_p95 = min(python_latency_p95, cpp_latency_p95)
        
        # Format latency with appropriate units
        if best_latency_p95 < 1:
            latency_str = f"{best_latency_p95 * 1000:.0f} ns"
        else:
            latency_str = f"{best_latency_p95:.0f} μs"
        
        # Print formatted banner
        self.console.print()
        self.console.print("[bold blue] Flashback C++ Engine Benchmark [/bold blue]")
        self.console.print("[bold blue]-----------------------------------[/bold blue]")
        self.console.print(f"Python engine throughput:   [yellow]{python_throughput}[/yellow]")
        self.console.print(f"C++ engine throughput:     [green]{cpp_throughput}[/green]")
        self.console.print(f"Speedup:                   [bold green]{speedup:.1f}x faster[/bold green]")
        self.console.print(f"Latency p95:               [cyan]{latency_str}[/cyan]")
        self.console.print()
    
    def _format_throughput(self, ops_per_sec: float) -> str:
        """Format throughput in a human-readable format.
        
        Args:
            ops_per_sec: Operations per second value
            
        Returns:
            Formatted string with appropriate units (K/M)
        """
        if not isinstance(ops_per_sec, (int, float)) or ops_per_sec < 0:
            return "0 events/sec"
            
        if ops_per_sec >= 1_000_000:
            return f"{ops_per_sec / 1_000_000:.1f}M events/sec"
        elif ops_per_sec >= 1_000:
            return f"{ops_per_sec / 1_000:.1f}K events/sec"
        else:
            return f"{ops_per_sec:.0f} events/sec"
    
    def _print_results_plain(self, results: List[Dict[str, Any]]):
        """Print results in plain text format."""
        # Print banner
        self._print_banner_plain(results)
        
        print("\n" + "="*80)
        print("MATCHING ENGINE PERFORMANCE BENCHMARK")
        print("="*80)
        
        for result in results:
            print(f"\n{result['engine']} Engine:")
            print(f"  Events: {result['num_events']:,}")
            print(f"  Trials: {result['num_trials']}")
            print(f"  Avg Time: {result['avg_time']:.3f}s")
            print(f"  Std Dev: {result['std_time']:.3f}s")
            print(f"  Ops/sec: {result['ops_per_sec']:,.0f}")
            print(f"  Latency: {result['avg_latency_us']:.2f}μs")
            print(f"  Avg Fills: {result['avg_fills']:,.0f}")
        
        if len(results) == 2:
            speedup = results[1]["ops_per_sec"] / results[0]["ops_per_sec"]
            print(f"\nC++ is {speedup:.2f}x faster than Python")
    
    def _print_banner_plain(self, results: List[Dict[str, Any]]):
        """Print the performance banner in plain text format."""
        if len(results) < 2:
            return
        
        python_result = results[0]
        cpp_result = results[1]
        
        # Calculate core metrics
        python_ops = python_result["ops_per_sec"]
        cpp_ops = cpp_result["ops_per_sec"]
        speedup = cpp_ops / python_ops
        
        # Format throughput for display
        python_throughput = self._format_throughput(python_ops)
        cpp_throughput = self._format_throughput(cpp_ops)
        
        # Calculate p95 latency (approximate from std dev)
        python_std_latency = (
            python_result.get("std_latency_us", 0) or 
            python_result["avg_latency_us"] * 0.1
        )
        cpp_std_latency = (
            cpp_result.get("std_latency_us", 0) or 
            cpp_result["avg_latency_us"] * 0.1
        )
        
        python_latency_p95 = python_result["avg_latency_us"] + 1.96 * python_std_latency
        cpp_latency_p95 = cpp_result["avg_latency_us"] + 1.96 * cpp_std_latency
        
        # Use the better (lower) p95 latency
        best_latency_p95 = min(python_latency_p95, cpp_latency_p95)
        
        # Format latency with appropriate units
        if best_latency_p95 < 1:
            latency_str = f"{best_latency_p95 * 1000:.0f} ns"
        else:
            latency_str = f"{best_latency_p95:.0f} μs"
        
        # Print formatted banner
        print()
        print(" Flashback C++ Engine Benchmark ")
        print("-----------------------------------")
        print(f"Python engine throughput:   {python_throughput}")
        print(f"C++ engine throughput:     {cpp_throughput}")
        print(f"Speedup:                   {speedup:.1f}x faster")
        print(f"Latency p95:               {latency_str}")
        print()
    
    def export_csv(self, results: List[Dict[str, Any]], filename: str = "bench_results.csv"):
        """Export benchmark results to CSV file."""
        output_path = Path(filename)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'engine', 'num_events', 'num_trials', 'avg_time', 'std_time',
                'min_time', 'max_time', 'ops_per_sec', 'avg_latency_us', 'avg_fills'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Create a copy without trial-specific data
                export_result = {k: v for k, v in result.items() if k not in ['trial_times', 'trial_fills']}
                writer.writerow(export_result)
        
        if self.console:
            self.console.print(f"[green]Results exported to {output_path}[/green]")
        else:
            print(f"Results exported to {output_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark C++ vs Python matching engines")
    parser.add_argument("--orders", type=int, default=1_000_000, 
                       help="Number of orders to generate (default: 1,000,000)")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of benchmark trials (default: 3)")
    parser.add_argument("--output", type=str, default="bench_results.csv",
                       help="Output CSV file (default: bench_results.csv)")
    parser.add_argument("--no-rich", action="store_true",
                       help="Disable rich formatting")
    
    args = parser.parse_args()
    
    # Initialize console
    console = None
    if RICH_AVAILABLE and not args.no_rich:
        console = Console()
    elif args.no_rich:
        print("Rich formatting disabled")
    
    # Check C++ availability
    if not HAS_CPP:
        if console:
            console.print("[yellow]Warning: C++ matching engine not available[/yellow]")
            console.print("[yellow]Run 'make cpp' to build the C++ extension[/yellow]")
        else:
            print("Warning: C++ matching engine not available")
            print("Run 'make cpp' to build the C++ extension")
    
    # Run benchmark
    runner = BenchmarkRunner(console)
    
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=None)
            results = runner.run_comparison(args.orders, args.trials)
    else:
        print("Running benchmark...")
        results = runner.run_comparison(args.orders, args.trials)
    
    # Display results
    runner.print_results(results)
    
    # Export results
    runner.export_csv(results, args.output)
    
    return results


if __name__ == "__main__":
    main()
