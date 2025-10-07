#!/usr/bin/env bash
set -euo pipefail

# Flashback HFT Backtesting Engine - Demo Script
# This script demonstrates the complete pipeline from data generation to results packaging
# Target runtime: < 2 minutes on 200k events

echo "🚀 Flashback HFT Backtesting Engine - Demo"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to measure execution time
measure_time() {
    local start_time=$(date +%s.%N)
    "$@"
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    echo "$duration"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if ! command_exists make; then
    print_error "Make is required but not installed"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$python_version < 3.11" | bc -l) -eq 1 ]]; then
    print_error "Python 3.11+ is required, found $python_version"
    exit 1
fi

print_success "Prerequisites check passed"

# Step 1: Setup
print_status "Step 1: Setting up environment..."
setup_start=$(date +%s.%N)

if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -e .
pip install -q -e ".[dev]"

setup_duration=$(echo "$(date +%s.%N) - $setup_start" | bc -l)
print_success "Setup completed in $(printf "%.2f" $setup_duration)s"

# Step 2: Code Quality Checks
print_status "Step 2: Running code quality checks..."

# Lint check
print_status "Running linting..."
if make lint >/dev/null 2>&1; then
    print_success "Linting passed"
else
    print_warning "Linting failed, continuing anyway..."
fi

# Type check
print_status "Running type checking..."
if make typecheck >/dev/null 2>&1; then
    print_success "Type checking passed"
else
    print_warning "Type checking failed, continuing anyway..."
fi

# Step 3: Generate Synthetic Data
print_status "Step 3: Generating synthetic market data..."
data_start=$(date +%s.%N)

# Generate 200k events for demo
python examples/generate_synthetic.py \
    --seed 7 \
    --events 200000 \
    --output-dir examples/sample_data

data_duration=$(echo "$(date +%s.%N) - $data_start" | bc -l)
print_success "Data generation completed in $(printf "%.2f" $data_duration)s"

# Step 4: Run Backtest
print_status "Step 4: Running backtest with momentum strategy..."
backtest_start=$(date +%s.%N)

# Create a demo config
cat > config/demo_backtest.yaml << EOF
name: "Demo Backtest"
description: "Flashback HFT Demo with Momentum Strategy"

data:
  path: "examples/sample_data/aapl_l1.parquet"
  kind: "book"
  symbol: "AAPL"

strategy:
  name: "momentum_imbalance"
  symbol: "AAPL"
  enabled: true
  max_position: 1000
  max_order_size: 100
  params:
    short_ema_period: 5
    long_ema_period: 20
    imbalance_threshold: 0.3
    take_profit_pct: 0.02
    stop_loss_pct: 0.01
    min_trade_size: 10
    max_trade_size: 100

execution:
  fees:
    maker_bps: 0.0
    taker_bps: 0.5
    per_share: 0.0
  latency:
    model: "normal"
    mean_ns: 500000
    std_ns: 100000
    seed: 42

risk:
  max_gross: 100000
  max_pos_per_symbol: 1000
  max_daily_loss: -2000

report:
  output_dir: "runs/demo_run"
  format: "both"
  plots: true
  detailed_trades: true
  performance_metrics: true
EOF

# Run the backtest
flashback run --config config/demo_backtest.yaml

backtest_duration=$(echo "$(date +%s.%N) - $backtest_start" | bc -l)
print_success "Backtest completed in $(printf "%.2f" $backtest_duration)s"

# Step 5: Latency Sensitivity Analysis
print_status "Step 5: Running latency sensitivity analysis..."
sweep_start=$(date +%s.%N)

flashback sweep \
    --config config/demo_backtest.yaml \
    --latency 100000,250000,500000,1000000

sweep_duration=$(echo "$(date +%s.%N) - $sweep_start" | bc -l)
print_success "Latency sweep completed in $(printf "%.2f" $sweep_duration)s"

# Step 6: Package Results
print_status "Step 6: Packaging results..."
pack_start=$(date +%s.%N)

# Find the latest run directory
latest_run=$(ls -dt runs/*/ | head -n1 | sed 's|/$||')
if [ -n "$latest_run" ]; then
    flashback pack --run "$latest_run"
    print_success "Results packaged successfully"
else
    print_warning "No run directory found to package"
fi

pack_duration=$(echo "$(date +%s.%N) - $pack_start" | bc -l)
print_success "Packaging completed in $(printf "%.2f" $pack_duration)s"

# Step 7: Display Results Summary
print_status "Step 7: Results Summary"
echo "=========================="

# Find the latest run directory
latest_run=$(ls -dt runs/*/ | head -n1 | sed 's|/$||')
if [ -n "$latest_run" ]; then
    echo "📁 Run Directory: $latest_run"
    
    # Display performance metrics if available
    if [ -f "$latest_run/performance.json" ]; then
        echo ""
        echo "📊 Performance Metrics:"
        python3 -c "
import json
import sys
try:
    with open('$latest_run/performance.json', 'r') as f:
        metrics = json.load(f)
    
    print(f'   Total Return: {metrics.get(\"total_return\", 0):.2%}')
    print(f'   Sharpe Ratio: {metrics.get(\"sharpe_ratio\", 0):.2f}')
    print(f'   Max Drawdown: {metrics.get(\"max_drawdown\", 0):.2%}')
    print(f'   Total Trades: {metrics.get(\"total_trades\", 0)}')
    print(f'   Hit Rate: {metrics.get(\"hit_rate\", 0):.2%}')
    print(f'   Turnover: {metrics.get(\"turnover\", 0):.2f}')
except Exception as e:
    print(f'   Error reading metrics: {e}')
"
    fi
    
    # List generated files
    echo ""
    echo "📄 Generated Files:"
    ls -la "$latest_run" | grep -E '\.(json|csv|png|parquet|yaml)$' | awk '{print "   " $9 " (" $5 " bytes)"}'
    
    # Check for plots
    plot_count=$(find "$latest_run" -name "*.png" | wc -l)
    echo "   📈 Generated $plot_count visualization plots"
    
    # Check for latency sweep results
    if [ -f "$latest_run/latency_sweep.csv" ]; then
        echo "   🔄 Latency sensitivity analysis completed"
    fi
fi

# Calculate total execution time
total_duration=$(echo "$(date +%s.%N) - $setup_start" | bc -l)

echo ""
echo "🎉 Demo completed successfully!"
echo "⏱️  Total execution time: $(printf "%.2f" $total_duration)s"

# Performance check
if (( $(echo "$total_duration < 120" | bc -l) )); then
    print_success "✅ Performance target met: < 2 minutes"
else
    print_warning "⚠️  Performance target missed: > 2 minutes"
fi

echo ""
echo "📚 Next Steps:"
echo "   1. Explore the generated plots in $latest_run"
echo "   2. Analyze performance metrics in performance.json"
echo "   3. Review trade data in trades.csv"
echo "   4. Try different strategy parameters"
echo "   5. Run with larger datasets for more comprehensive analysis"
echo ""
echo "🔗 Documentation: docs/architecture.md"
echo "📖 Examples: examples/notebooks/01_quickstart.ipynb"
echo ""

# Optional: Open results directory (macOS)
if command_exists open && [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open results directory? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$latest_run"
    fi
fi

print_success "Demo completed successfully! 🚀"