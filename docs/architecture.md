# Flashback HFT Backtesting Engine - Architecture Documentation

## Overview

Flashback is a high-frequency trading (HFT) strategy backtesting engine designed for microsecond-precision simulation of trading strategies with realistic market microstructure modeling. The engine provides a comprehensive framework for developing, testing, and analyzing HFT strategies with detailed performance metrics and risk management.

## System Architecture

### Core Components

The Flashback engine is built around an event-driven architecture with the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│   Event Loop    │───▶│   Strategies    │
│   Generator     │    │   Engine        │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Order Router  │◀───│   Matching      │───▶│   Risk          │
│   & Blotter     │    │   Engine        │    │   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Performance   │◀───│   Fill Events   │───▶│   Reporting     │
│   Analyzer      │    │   & Metrics     │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1. Event Loop Engine

The event loop is the heart of the backtesting engine, processing events in strict chronological order:

**Event Types:**
- `MarketDataEvent`: L1/L2 order book updates, trade ticks
- `OrderEvent`: Strategy-generated order intents
- `FillEvent`: Order executions with price, quantity, and timing
- `CancelEvent`: Order cancellations
- `RejectEvent`: Order rejections due to risk limits
- `TimerEvent`: Time-based strategy triggers

**Processing Flow:**
1. **Event Ingestion**: Market data events are loaded from Parquet files
2. **Event Scheduling**: Events are sorted by timestamp (nanosecond precision)
3. **Event Processing**: Each event is processed in order:
   - Market data events trigger strategy logic
   - Order events are routed to matching engines
   - Fill events update positions and trigger risk checks
4. **State Updates**: Portfolio positions, PnL, and risk metrics are updated

**Key Features:**
- Nanosecond timestamp precision
- Deterministic event processing
- Memory-efficient streaming of large datasets
- Support for multiple symbols and strategies

### 2. Matching Engine

The matching engine implements price-time priority order matching with realistic market microstructure:

**Core Algorithm:**
- **Price-Time Priority**: Orders are matched first by price, then by time
- **Partial Fills**: Large orders can be filled across multiple price levels
- **Order Types**: Market, Limit, IOC (Immediate-or-Cancel), FOK (Fill-or-Kill)
- **Time-in-Force**: DAY, IOC, FOK with automatic expiration

**Implementation Options:**
- **Python Engine**: Pure Python implementation for development and testing
- **C++ Engine**: High-performance C++ implementation via pybind11 (8-12× speedup)
- **Cython Engine**: Optimized Cython implementation for hot paths

**Matching Logic:**
```python
def match_order(order):
    if order.side == BUY:
        # Match against ask levels (ascending price)
        for ask_level in sorted_ask_levels:
            if order.price >= ask_level.price:
                fill_quantity = min(order.remaining_qty, ask_level.qty)
                create_fill(order, ask_level, fill_quantity)
                update_order_book(ask_level, fill_quantity)
    else:
        # Match against bid levels (descending price)
        for bid_level in sorted_bid_levels:
            if order.price <= bid_level.price:
                fill_quantity = min(order.remaining_qty, bid_level.qty)
                create_fill(order, bid_level, fill_quantity)
                update_order_book(bid_level, fill_quantity)
```

**Performance Characteristics:**
- **Throughput**: 10M+ events/second with C++ engine
- **Latency**: Sub-microsecond order matching
- **Memory**: Optimized data structures with minimal allocations

### 3. Strategy Engine

The strategy engine executes trading strategies based on market events:

**Strategy Interface:**
```python
class Strategy:
    def on_bar(self, market_data: MarketDataEvent) -> List[NewOrder]:
        """Process market data and generate order intents."""
        pass
    
    def on_trade(self, fill: FillEvent) -> List[NewOrder]:
        """Process trade executions and update positions."""
        pass
    
    def on_timer(self, timer: TimerEvent) -> List[NewOrder]:
        """Process time-based events."""
        pass
```

**Built-in Strategies:**

**Momentum Imbalance Strategy:**
- Uses dual EMA (5-period, 20-period) for trend detection
- Calculates order flow imbalance from trade data
- Enters positions when momentum and imbalance align
- Exits on profit targets or stop losses

**Mean Reversion Strategy:**
- Calculates rolling z-score of price vs. moving average
- Enters positions when price deviates significantly from mean
- Exits when price returns to mean or hits stop loss

**Strategy Features:**
- Position sizing with risk limits
- Stop loss and take profit levels
- Maximum position and order size limits
- Real-time PnL tracking

### 4. Order Router & Blotter

The order router handles strategy intents and manages order lifecycle:

**Order Processing Flow:**
1. **Intent Processing**: Strategy generates order intents
2. **Latency Simulation**: Orders are scheduled with realistic latency
3. **Risk Checks**: Pre-trade risk validation
4. **Order Submission**: Orders are sent to matching engines
5. **Fill Processing**: Fill events update order states
6. **Blotter Updates**: Order status and fills are tracked

**Latency Models:**
- **Normal Distribution**: Configurable mean and standard deviation
- **Exponential Distribution**: Realistic latency tail behavior
- **Fixed Latency**: Deterministic for testing
- **Custom Models**: User-defined latency distributions

**Blotter Features:**
- Real-time order status tracking
- Fill aggregation and partial fill handling
- Order history and audit trail
- Performance metrics per order

### 5. Risk Management

The risk management system enforces position and exposure limits:

**Risk Limits:**
- **Gross Exposure**: Maximum total position value
- **Net Exposure**: Maximum net position value
- **Position Limits**: Maximum position per symbol
- **Daily Loss Limits**: Maximum daily loss threshold
- **Drawdown Limits**: Maximum portfolio drawdown

**Risk Checks:**
- **Pre-trade**: Validate orders before submission
- **Real-time**: Monitor positions during execution
- **Post-trade**: Update risk metrics after fills

**Auto-flattening:**
- Automatic position closure when limits are breached
- Emergency stop mechanisms
- Risk alert notifications

### 6. Performance Analytics

The performance analyzer calculates comprehensive trading metrics:

**Return Metrics:**
- Total return and annualized return
- Sharpe ratio and Sortino ratio
- Maximum drawdown and recovery time
- Calmar ratio and Sterling ratio

**Risk Metrics:**
- Volatility (realized and annualized)
- Value at Risk (VaR) at multiple confidence levels
- Expected Shortfall (Conditional VaR)
- Skewness and kurtosis of returns

**Trading Metrics:**
- Hit rate and win/loss ratio
- Average win/loss amounts
- Profit factor and expectancy
- Turnover and holding period analysis

**Latency Metrics:**
- Order-to-fill latency statistics
- Queue time and processing time
- Latency percentiles and distributions
- Latency sensitivity analysis

## Market Microstructure Assumptions

### Order Book Model

**L1 Data Only:**
- Best bid and ask prices with sizes
- No full depth-of-book reconstruction
- Simplified order book dynamics

**Price-Time Priority:**
- Orders matched by price first, then time
- No hidden orders or dark pools
- No order routing or smart order routing

**Market Impact:**
- Linear market impact model
- No permanent vs. temporary impact distinction
- Simplified price discovery mechanism

### Latency Modeling

**Latency Components:**
- **Network Latency**: Order transmission time
- **Processing Latency**: Exchange processing time
- **Queue Latency**: Order book queue time
- **Tick-to-Trade**: Market data to order latency

**Latency Distributions:**
- Normal distribution for typical latencies
- Exponential tail for extreme latencies
- Configurable parameters for different environments

### Fee and Cost Models

**Commission Structure:**
- Maker/taker fee differentiation
- Per-share and per-trade fees
- Tiered fee structures

**Slippage Models:**
- Fixed slippage per trade
- Imbalance-based slippage
- Adaptive slippage based on market conditions

**Transaction Costs:**
- Bid-ask spread costs
- Market impact costs
- Timing costs

## Limitations and Caveats

### Data Limitations

**Synthetic Data:**
- Generated data may not reflect real market dynamics
- Limited to L1 order book data
- No news events or fundamental data

**Historical Data:**
- Survivorship bias in historical datasets
- Limited to available data providers
- No real-time data feed simulation

### Model Limitations

**Market Impact:**
- Simplified linear impact model
- No consideration of order size effects
- Limited market depth modeling

**Latency Modeling:**
- Static latency distributions
- No network congestion effects
- Simplified queue modeling

**Risk Management:**
- No correlation modeling between positions
- Limited to single-asset strategies
- No portfolio-level risk metrics

### Performance Limitations

**Computational:**
- Memory usage scales with dataset size
- Single-threaded event processing
- Limited parallelization

**Accuracy:**
- Nanosecond precision may exceed real-world accuracy
- Simplified market microstructure
- No consideration of market maker behavior

## References and Further Reading

### Academic References

1. **Market Microstructure Theory:**
   - O'Hara, M. (1995). "Market Microstructure Theory"
   - Hasbrouck, J. (2007). "Empirical Market Microstructure"

2. **High-Frequency Trading:**
   - Aldridge, I. (2013). "High-Frequency Trading: A Practical Guide"
   - Easley, D., et al. (2012). "The Microstructure of the 'Flash Crash'"

3. **Latency and Performance:**
   - Budish, E., et al. (2015). "The High-Frequency Trading Arms Race"
   - Menkveld, A. J. (2016). "The Economics of High-Frequency Trading"

### Technical References

1. **Order Matching Algorithms:**
   - Harris, L. (2003). "Trading and Exchanges: Market Microstructure for Practitioners"
   - Biais, B., et al. (2005). "Market Microstructure: A Survey"

2. **Risk Management:**
   - Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk"
   - McNeil, A. J., et al. (2015). "Quantitative Risk Management"

3. **Performance Measurement:**
   - Sharpe, W. F. (1966). "Mutual Fund Performance"
   - Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework"

## Implementation Notes

### Development Guidelines

**Code Organization:**
- Modular design with clear interfaces
- Comprehensive unit and integration tests
- Type hints and documentation
- Performance profiling and optimization

**Testing Strategy:**
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance benchmarks for critical paths
- Bit-for-bit tests for Cython/C++ implementations

**Deployment Considerations:**
- Docker containerization for consistent environments
- CI/CD pipeline with automated testing
- Performance monitoring and alerting
- Scalability planning for large datasets

### Future Enhancements

**Planned Features:**
- Multi-asset portfolio strategies
- Advanced order types (iceberg, hidden)
- Real-time data feed integration
- Machine learning strategy framework

**Performance Improvements:**
- Multi-threaded event processing
- GPU acceleration for calculations
- Distributed computing support
- Memory-mapped file I/O

**Risk Management:**
- Portfolio-level risk metrics
- Correlation modeling
- Stress testing framework
- Real-time risk monitoring

---

*This documentation is part of the Flashback HFT Backtesting Engine. For technical support or feature requests, please refer to the project repository.*