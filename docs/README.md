# Flashback HFT Backtesting Engine - Documentation

This directory contains comprehensive documentation for the Flashback High-Frequency Trading Backtesting Engine.

## Documentation Structure

### [Architecture](architecture.md)
Comprehensive technical documentation covering:
- System architecture and component design
- Event loop and processing flow
- Order matching engine implementation
- Risk management system
- Latency and cost modeling
- Market microstructure assumptions
- Performance characteristics
- Limitations and disclaimers

## Quick Start

1. **Installation**: See the main README.md for installation instructions
2. **Configuration**: Create a YAML configuration file (see examples/)
3. **Running**: Use the CLI commands to run backtests
4. **Analysis**: Review generated performance reports and plots

## Key Concepts

### Event-Driven Architecture
The engine uses a discrete-event simulation approach with a priority queue to process market events in chronological order.

### Order Matching
Implements price-time priority matching with support for partial fills and multiple order types.

### Risk Management
Real-time position tracking with configurable risk limits and automatic position flattening.

### Latency Modeling
Multiple latency models to simulate realistic execution delays and their impact on strategy performance.

### Cost Integration
Transaction costs, slippage, and fees are integrated into fill generation for realistic performance measurement.

## Important Limitations

 **This is a backtesting engine, not a production trading system**

- Uses synthetic data that may not reflect real market conditions
- Simplified models may not capture all market nuances
- Not suitable for real money trading
- Intended for research and education purposes only

## Getting Help

- Review the architecture documentation for technical details
- Check the examples/ directory for usage examples
- Run the test suite to understand expected behavior
- Refer to the source code for implementation details

## Contributing

When contributing to the codebase:
1. Follow the existing code style and patterns
2. Add appropriate tests for new functionality
3. Update documentation for any architectural changes
4. Ensure all tests pass before submitting changes

---

*For more information, see the main project README.md and source code documentation.*
