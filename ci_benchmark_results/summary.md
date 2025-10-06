# CI Benchmark Results

**Timestamp:** 2025-10-06T19:03:22Z
**Environment:** Local

## C++ Extension Status
✅ **Available** - C++ matching engine is built and functional

## Benchmark Results

### Performance Summary
#### 100000 Orders

| Engine | Ops/sec | Latency (μs) | Avg Fills |
|--------|---------|--------------|----------|
| Python | 108139.75444728753 | 9.247293052505013 | 95642 |
| C++ | 156426.34918512046 | 6.392784880612184 | 71072.66666666667 |

#### 1000000 Orders

| Engine | Ops/sec | Latency (μs) | Avg Fills |
|--------|---------|--------------|----------|
| Python | 103522.9849487056 | 9.659690555633494 | 953122.6666666666 |
| C++ | 141508.94199575315 | 7.066691234466378 | 710506 |

#### 500000 Orders

| Engine | Ops/sec | Latency (μs) | Avg Fills |
|--------|---------|--------------|----------|
| Python | 108454.97694398109 | 9.220415956720156 | 476966 |
| C++ | 165147.50163564688 | 6.0551930249979105 | 356503.3333333333 |


## Generated Plots

- **Ops Comparison**: Bar chart comparing operations per second
- **Latency Distribution**: Line chart showing latency distribution
- **Throughput Scatter**: Scatter plot of throughput vs orders
- **Performance Summary**: Comprehensive 4-panel summary

## Files Generated

ci_benchmark_results/bench_100000.csv
ci_benchmark_results/bench_1000000.csv
ci_benchmark_results/bench_500000.csv
ci_benchmark_results/plots/benchmark_metadata.json
ci_benchmark_results/plots/latency_distribution.png
ci_benchmark_results/plots/ops_comparison.png
ci_benchmark_results/plots/performance_summary.png
ci_benchmark_results/plots/throughput_scatter.png
