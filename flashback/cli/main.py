"""Main CLI entry point for flashback backtesting engine."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from ..config import load_config
from .runner import BacktestRunner
from .sweeper import LatencySweeper
from .pack import create_run_bundle

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_command(args) -> int:
    """Run a single backtest."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Create output directory
        output_dir = Path(config.report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run backtest
        runner = BacktestRunner(config)
        result = runner.run()
        
        # Save results
        runner.save_results(result, output_dir)
        
        logger.info(f"Backtest completed successfully. Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def sweep_command(args) -> int:
    """Run latency sensitivity sweep."""
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Parse latency values
        latency_values = [int(x.strip()) for x in args.latency.split(',')]
        
        # Create output directory
        output_dir = Path(config.report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run sweep
        sweeper = LatencySweeper(config)
        results = sweeper.run_sweep(latency_values)
        
        # Save results
        sweeper.save_results(results, output_dir)
        
        logger.info(f"Latency sweep completed successfully. Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Latency sweep failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def pack_command(args) -> int:
    """Pack a backtest run directory."""
    try:
        # Setup logging
        setup_logging(args.verbose)
        
        # Pack the run
        output_path = create_run_bundle(str(args.run), str(args.output) if args.output else None)
        
        logger.info(f"Successfully packed run to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Packing failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="flashback",
        description="High-Frequency Trading Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashback run --config config/backtest.yaml
  flashback sweep --config config/backtest.yaml --latency 100000,250000,500000
  flashback pack --run runs/2025-10-06T00-00-00
  flashback run --config config/backtest.yaml --verbose
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single backtest"
    )
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    run_parser.set_defaults(func=run_command)
    
    # Sweep command
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run latency sensitivity sweep"
    )
    sweep_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    sweep_parser.add_argument(
        "--latency",
        type=str,
        required=True,
        help="Comma-separated list of latency values in nanoseconds"
    )
    sweep_parser.set_defaults(func=sweep_command)
    
    # Pack command
    pack_parser = subparsers.add_parser(
        "pack",
        help="Pack a backtest run directory into a zip file"
    )
    pack_parser.add_argument(
        "--run", "-r",
        type=Path,
        required=True,
        help="Path to run directory (e.g., runs/2025-10-06T00-00-00)"
    )
    pack_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output zip file path (default: run_directory.zip)"
    )
    pack_parser.set_defaults(func=pack_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    return args.func(args)


def cli():
    """CLI entry point for the flashback command."""
    return main()


if __name__ == "__main__":
    sys.exit(main())