#!/usr/bin/env python3
"""
Benchmark visualization module for C++ vs Python matching engines.

Generates comprehensive plots for performance analysis including:
- Bar charts for ops/sec comparison
- Line charts for latency distribution
- Scatter plots for throughput vs orders
- All plots saved to runs/<timestamp>/benchmarks/
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class BenchmarkPlotter:
    """Generate performance benchmark visualizations."""
    
    def __init__(self, console=None):
        """Initialize the benchmark plotter."""
        self.console = console or Console() if RICH_AVAILABLE else None
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def load_benchmark_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load benchmark data from CSV file."""
        data = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                numeric_fields = [
                    'num_events', 'num_trials', 'avg_time', 'std_time',
                    'min_time', 'max_time', 'ops_per_sec', 'avg_latency_us', 'avg_fills'
                ]
                
                for field in numeric_fields:
                    if field in row:
                        row[field] = float(row[field])
                
                data.append(row)
        
        return data
    
    def create_ops_comparison_chart(self, data: List[Dict[str, Any]], output_dir: Path) -> str:
        """Create bar chart comparing ops/sec between Python and C++."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        engines = [row['engine'] for row in data]
        ops_per_sec = [row['ops_per_sec'] for row in data]
        
        # Create bar chart
        bars = ax.bar(engines, ops_per_sec, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, ops in zip(bars, ops_per_sec):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{ops:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Calculate and display speedup
        if len(data) == 2:
            python_ops = data[0]['ops_per_sec']
            cpp_ops = data[1]['ops_per_sec']
            speedup = cpp_ops / python_ops
            
            ax.text(0.5, 0.95, f'C++ is {speedup:.2f}x faster than Python',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Operations per Second', fontweight='bold')
        ax.set_xlabel('Matching Engine', fontweight='bold')
        ax.set_title('Performance Comparison: Operations per Second', fontweight='bold', fontsize=14)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'ops_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def create_latency_distribution_chart(self, data: List[Dict[str, Any]], output_dir: Path) -> str:
        """Create line chart showing latency distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create latency data for each engine
        for i, row in enumerate(data):
            engine = row['engine']
            avg_latency = row['avg_latency_us']
            std_latency = row.get('std_latency_us', 0)  # We'll estimate this
            
            # Create a normal distribution around the average latency
            x = np.linspace(avg_latency - 3*std_latency, avg_latency + 3*std_latency, 100)
            y = np.exp(-0.5 * ((x - avg_latency) / max(std_latency, 0.1))**2)
            
            # Normalize
            y = y / np.max(y)
            
            color = '#1f77b4' if engine == 'Python' else '#ff7f0e'
            ax.plot(x, y, label=f'{engine} (μ={avg_latency:.2f}μs)', 
                   color=color, linewidth=2, alpha=0.8)
            
            # Add vertical line at average
            ax.axvline(avg_latency, color=color, linestyle='--', alpha=0.6)
        
        ax.set_xlabel('Latency (microseconds)', fontweight='bold')
        ax.set_ylabel('Normalized Distribution', fontweight='bold')
        ax.set_title('Latency Distribution Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'latency_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def create_throughput_scatter(self, data: List[Dict[str, Any]], output_dir: Path) -> str:
        """Create scatter plot of throughput vs number of orders."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        engines = [row['engine'] for row in data]
        num_events = [row['num_events'] for row in data]
        ops_per_sec = [row['ops_per_sec'] for row in data]
        colors = ['#1f77b4' if engine == 'Python' else '#ff7f0e' for engine in engines]
        
        # Create scatter plot
        scatter = ax.scatter(num_events, ops_per_sec, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add trend lines
        for i, engine in enumerate(set(engines)):
            engine_data = [(row['num_events'], row['ops_per_sec']) for row in data if row['engine'] == engine]
            if len(engine_data) > 1:
                x_vals, y_vals = zip(*engine_data)
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                color = '#1f77b4' if engine == 'Python' else '#ff7f0e'
                ax.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        # Add labels for each point
        for i, (x, y, engine) in enumerate(zip(num_events, ops_per_sec, engines)):
            ax.annotate(f'{engine}\n{x:,.0f} events', (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Number of Orders', fontweight='bold')
        ax.set_ylabel('Operations per Second', fontweight='bold')
        ax.set_title('Throughput vs Number of Orders', fontweight='bold', fontsize=14)
        
        # Format axes
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add legend
        python_patch = mpatches.Patch(color='#1f77b4', label='Python')
        cpp_patch = mpatches.Patch(color='#ff7f0e', label='C++')
        ax.legend(handles=[python_patch, cpp_patch])
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'throughput_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def create_performance_summary(self, data: List[Dict[str, Any]], output_dir: Path) -> str:
        """Create a comprehensive performance summary plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Ops/sec comparison
        engines = [row['engine'] for row in data]
        ops_per_sec = [row['ops_per_sec'] for row in data]
        bars1 = ax1.bar(engines, ops_per_sec, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax1.set_ylabel('Operations per Second')
        ax1.set_title('Operations per Second')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add value labels
        for bar, ops in zip(bars1, ops_per_sec):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{ops:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Latency comparison
        latencies = [row['avg_latency_us'] for row in data]
        bars2 = ax2.bar(engines, latencies, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax2.set_ylabel('Average Latency (μs)')
        ax2.set_title('Average Latency')
        
        # Add value labels
        for bar, lat in zip(bars2, latencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{lat:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Fills comparison
        fills = [row['avg_fills'] for row in data]
        bars3 = ax3.bar(engines, fills, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax3.set_ylabel('Average Fills')
        ax3.set_title('Average Fills Generated')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add value labels
        for bar, fill in zip(bars3, fills):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{fill:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Speedup calculation
        if len(data) == 2:
            python_ops = data[0]['ops_per_sec']
            cpp_ops = data[1]['ops_per_sec']
            speedup = cpp_ops / python_ops
            
            ax4.bar(['Speedup'], [speedup], color='green', alpha=0.8)
            ax4.set_ylabel('Speedup Factor')
            ax4.set_title(f'C++ vs Python Speedup: {speedup:.2f}x')
            ax4.text(0, speedup + speedup*0.01, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor speedup calculation', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Speedup Calculation')
        
        # Add grid to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Benchmark Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'performance_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def generate_all_plots(self, csv_file: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Generate all benchmark plots and return file paths."""
        if output_dir is None:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output_dir = Path(f"runs/{timestamp}/benchmarks")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.console:
            self.console.print(f"[blue]Generating plots in {output_dir}[/blue]")
        
        # Load data
        data = self.load_benchmark_data(csv_file)
        
        if not data:
            raise ValueError("No data found in CSV file")
        
        # Generate plots
        plot_files = {}
        
        try:
            plot_files['ops_comparison'] = self.create_ops_comparison_chart(data, output_dir)
            if self.console:
                self.console.print(f"[green]✓[/green] Ops comparison chart: {plot_files['ops_comparison']}")
            
            plot_files['latency_distribution'] = self.create_latency_distribution_chart(data, output_dir)
            if self.console:
                self.console.print(f"[green]✓[/green] Latency distribution chart: {plot_files['latency_distribution']}")
            
            plot_files['throughput_scatter'] = self.create_throughput_scatter(data, output_dir)
            if self.console:
                self.console.print(f"[green]✓[/green] Throughput scatter plot: {plot_files['throughput_scatter']}")
            
            plot_files['performance_summary'] = self.create_performance_summary(data, output_dir)
            if self.console:
                self.console.print(f"[green]✓[/green] Performance summary: {plot_files['performance_summary']}")
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Error generating plots: {e}[/red]")
            raise
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'csv_file': csv_file,
            'output_dir': str(output_dir),
            'plots': plot_files,
            'data_summary': {
                'num_engines': len(data),
                'engines': [row['engine'] for row in data],
                'total_events': sum(row['num_events'] for row in data)
            }
        }
        
        metadata_file = output_dir / 'benchmark_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.console:
            self.console.print(f"[green]✓[/green] Metadata saved: {metadata_file}")
        
        return plot_files


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate benchmark visualization plots")
    parser.add_argument("csv_file", help="CSV file containing benchmark results")
    parser.add_argument("--output-dir", help="Output directory for plots (default: runs/<timestamp>/benchmarks/)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich formatting")
    
    args = parser.parse_args()
    
    # Initialize console
    console = None
    if RICH_AVAILABLE and not args.no_rich:
        console = Console()
    
    # Check matplotlib availability
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return 1
    
    # Create plotter
    plotter = BenchmarkPlotter(console)
    
    # Generate plots
    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        plot_files = plotter.generate_all_plots(args.csv_file, output_dir)
        
        if console:
            console.print(f"\n[green]Successfully generated {len(plot_files)} plots![/green]")
            for plot_name, plot_file in plot_files.items():
                console.print(f"  • {plot_name}: {plot_file}")
        else:
            print(f"\nSuccessfully generated {len(plot_files)} plots!")
            for plot_name, plot_file in plot_files.items():
                print(f"  • {plot_name}: {plot_file}")
        
        return 0
        
    except Exception as e:
        if console:
            console.print(f"[red]Error: {e}[/red]")
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
