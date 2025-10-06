#!/usr/bin/env bash
set -euo pipefail

# Flashback HFT Backtesting Engine - Demo Script
# This script demonstrates the complete workflow from setup to results

echo "🚀 Flashback HFT Backtesting Engine Demo"
echo "========================================"

# Setup environment
echo "📦 Setting up environment..."
make setup

# Run quality checks (skip for demo performance)
echo "🔍 Skipping quality checks for demo performance..."
# make lint && make typecheck

# Generate synthetic data
echo "📊 Generating synthetic market data..."
python examples/generate_synthetic.py --seed 7 --events 200000

# Run single backtest
echo "⚡ Running single backtest..."
flashback run --config config/backtest.yaml

# Run latency sensitivity sweep
echo "🔄 Running latency sensitivity sweep..."
flashback sweep --config config/backtest.yaml --latency 100000,250000,500000

# Pack the latest run
echo "📦 Packing latest run results..."
LATEST_RUN=$(ls -dt runs/* | head -n1)
if [ -n "$LATEST_RUN" ]; then
    flashback pack --run "$LATEST_RUN"
    echo "✅ Packed run: $LATEST_RUN"
else
    echo "❌ No runs found to pack"
    exit 1
fi

echo ""
echo "🎉 Demo completed successfully!"
echo "📁 Results available in: $LATEST_RUN"
echo "📊 Check the generated plots and performance metrics"
