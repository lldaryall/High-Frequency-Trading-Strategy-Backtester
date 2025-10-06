#!/usr/bin/env bash
set -euo pipefail

# CI Benchmark Script
# Runs performance benchmarks and generates plots if C++ build succeeds

echo "🚀 Starting CI Benchmark Process"
echo "================================"

# Check if we're in a CI environment
if [ "${CI:-false}" = "true" ]; then
    echo "📋 Running in CI environment"
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
else
    echo "💻 Running locally"
fi

# Function to check if C++ extension is available
check_cpp_extension() {
    echo "🔍 Checking C++ extension availability..."
    if python -c "import flashback.market._match_cpp; print('C++ extension available')" 2>/dev/null; then
        echo "✅ C++ extension is available"
        return 0
    else
        echo "❌ C++ extension not available"
        return 1
    fi
}

# Function to run benchmarks
run_benchmarks() {
    echo "📊 Running performance benchmarks..."
    
    # Run benchmark with different order counts for comprehensive analysis
    local orders=(100000 500000 1000000)
    local results_dir="ci_benchmark_results"
    
    mkdir -p "$results_dir"
    
    for orders_count in "${orders[@]}"; do
        echo "  Running benchmark with $orders_count orders..."
        python flashback/utils/bench_cpp.py \
            --orders "$orders_count" \
            --trials 3 \
            --output "$results_dir/bench_${orders_count}.csv"
    done
    
    # Generate plots for the largest benchmark
    echo "📈 Generating plots..."
    python flashback/metrics/bench_plots.py \
        "$results_dir/bench_1000000.csv" \
        --output-dir "$results_dir/plots"
    
    echo "✅ Benchmarks completed successfully"
    return 0
}

# Function to generate summary report
generate_summary() {
    local results_dir="ci_benchmark_results"
    local summary_file="$results_dir/summary.md"
    
    echo "📝 Generating summary report..."
    
    cat > "$summary_file" << EOF
# CI Benchmark Results

**Timestamp:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")
**Environment:** ${CI:-"Local"}

## C++ Extension Status
EOF

    if check_cpp_extension; then
        echo "✅ **Available** - C++ matching engine is built and functional" >> "$summary_file"
    else
        echo "❌ **Not Available** - Falling back to Python implementation" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

## Benchmark Results

### Performance Summary
EOF

    # Add performance data from CSV files
    for csv_file in "$results_dir"/bench_*.csv; do
        if [ -f "$csv_file" ]; then
            local orders=$(basename "$csv_file" .csv | sed 's/bench_//')
            echo "#### $orders Orders" >> "$summary_file"
            echo "" >> "$summary_file"
            echo '| Engine | Ops/sec | Latency (μs) | Avg Fills |' >> "$summary_file"
            echo '|--------|---------|--------------|----------|' >> "$summary_file"
            
            # Parse CSV and add rows
            tail -n +2 "$csv_file" | while IFS=',' read -r engine num_events num_trials avg_time std_time min_time max_time ops_per_sec avg_latency_us avg_fills; do
                printf "| %s | %s | %s | %s |\n" "$engine" "$ops_per_sec" "$avg_latency_us" "$avg_fills" >> "$summary_file"
            done
            echo "" >> "$summary_file"
        fi
    done

    cat >> "$summary_file" << EOF

## Generated Plots

- **Ops Comparison**: Bar chart comparing operations per second
- **Latency Distribution**: Line chart showing latency distribution
- **Throughput Scatter**: Scatter plot of throughput vs orders
- **Performance Summary**: Comprehensive 4-panel summary

## Files Generated

EOF

    find "$results_dir" -type f -name "*.png" -o -name "*.csv" -o -name "*.json" | sort >> "$summary_file"
    
    echo "📄 Summary report saved to: $summary_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    # Check C++ extension
    if check_cpp_extension; then
        echo "🎯 C++ extension available - running full benchmarks"
        run_benchmarks
    else
        echo "⚠️  C++ extension not available - running Python-only benchmarks"
        # Still run benchmarks but only with Python
        python flashback/utils/bench_cpp.py --orders 100000 --trials 1 --no-rich
    fi
    
    # Generate summary
    generate_summary
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "🎉 CI Benchmark Process Completed"
    echo "⏱️  Total time: ${duration}s"
    echo "📁 Results saved in: ci_benchmark_results/"
    
    # List generated files
    echo ""
    echo "📋 Generated files:"
    find ci_benchmark_results -type f | sort | sed 's/^/  - /'
}

# Run main function
main "$@"
