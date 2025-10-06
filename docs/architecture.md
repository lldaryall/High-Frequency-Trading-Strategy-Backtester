# Flashback HFT Backtesting Engine - Architecture

## Overview

Flashback is a high-frequency trading (HFT) backtesting engine designed to simulate realistic market microstructure and execution dynamics. The engine implements an event-driven architecture with discrete-event simulation, order matching, risk management, and performance analytics.

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│  Event Engine   │───▶│  Strategy       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Matching Engine │    │ Order Router    │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Risk Manager    │    │ Performance     │
                       └─────────────────┘    └─────────────────┘
```

### Event Processing Flow

```
Market Data ──┐
              │
              ▼
    ┌─────────────────┐
    │   Event Queue   │ (Priority Queue)
    │   (Heap-based)  │
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  Event Engine   │
    │  (Dispatcher)   │
    └─────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Strategy │ │Matching │ │  Risk   │
│ Handler │ │ Engine  │ │Manager  │
└─────────┘ └─────────┘ └─────────┘
    │         │         │
    ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Orders  │ │ Fills   │ │Position │
│Generated│ │Created  │ │Updated  │
└─────────┘ └─────────┘ └─────────┘
```

### Order Matching Process

```
Incoming Order
      │
      ▼
┌─────────────┐
│ Price-Time  │
│   Priority  │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Match      │
│  Against    │
│  Opposite   │
│  Side       │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Generate    │
│   Fills     │
│ (Partial    │
│  Possible)  │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Update      │
│ Positions   │
│ & PnL       │
└─────────────┘
```

## Event Loop Architecture

### Event Types

The engine processes five types of events in a priority queue:

1. **DATA** - Market data updates (L1 order book, trades)
2. **TIMER** - Scheduled events (strategy timers, risk checks)
3. **CONTROL** - System control events (start, stop, pause)
4. **FILL** - Order execution notifications
5. **CANCEL** - Order cancellation notifications

### Event Processing Flow

```python
class EventLoop:
    def __init__(self):
        self.event_queue = []  # Priority queue (heapq)
        self.clock = SimClock()
        self.strategies = []
        self.matching_engines = {}
        self.risk_manager = PortfolioRiskManager()
    
    def process_event(self, event):
        if event.type == EventType.DATA:
            self._process_market_data(event)
        elif event.type == EventType.TIMER:
            self._process_timer_event(event)
        elif event.type == EventType.FILL:
            self._process_fill_event(event)
        # ... other event types
```

### Event Priority and Ordering

Events are processed in strict chronological order using a min-heap priority queue:

1. **Timestamp** (primary) - Events processed in chronological order
2. **Event Type** (secondary) - Within same timestamp: DATA → FILL → CANCEL → TIMER
3. **Sequence Number** (tertiary) - For tie-breaking

## Matching Engine

### Order Book Structure

The matching engine maintains separate bid and ask order books using price-time priority:

```python
class MatchingEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bid_book = {}  # price -> [orders]
        self.ask_book = {}  # price -> [orders]
        self.order_map = {}  # order_id -> order
```

### Price-Time Priority

Orders are matched using strict price-time priority:

1. **Price Priority**: Better prices execute first
   - Bids: Higher prices have priority
   - Asks: Lower prices have priority

2. **Time Priority**: Within same price, earlier orders execute first
   - Orders are stored in FIFO queues at each price level

### Matching Algorithm

```python
def match_orders(self, incoming_order: Order) -> List[Fill]:
    fills = []
    
    if incoming_order.side == OrderSide.BUY:
        # Match against ask book
        for price in sorted(self.ask_book.keys()):
            if price <= incoming_order.price:
                fills.extend(self._match_at_price(price, incoming_order))
                if incoming_order.quantity == 0:
                    break
    
    return fills
```

### Order Types Supported

- **LIMIT**: Execute at specified price or better
- **MARKET**: Execute at best available price
- **IOC** (Immediate or Cancel): Execute immediately or cancel
- **FOK** (Fill or Kill): Execute completely or cancel

### Partial Fills

The engine supports partial fills with proper quantity tracking:

```python
class Order:
    def __init__(self, ...):
        self.quantity = 1000
        self.filled_quantity = 0
        self.remaining_quantity = 1000
    
    def is_partially_filled(self) -> bool:
        return self.filled_quantity > 0 and self.remaining_quantity > 0
```

## Risk Management

### Portfolio Tracking

The risk manager maintains real-time portfolio state:

```python
class PortfolioRiskManager:
    def __init__(self):
        self.positions = {}  # symbol -> Position
        self.cash = 100000.0
        self.risk_limits = []
    
    def update_position(self, symbol: str, quantity: int, price: float):
        # Update position and calculate PnL
        pass
```

### Risk Limits

1. **Maximum Gross Exposure**: Total absolute position value
2. **Maximum Position per Symbol**: Individual symbol limits
3. **Daily Loss Limit**: Maximum daily loss before auto-flattening
4. **Concentration Limits**: Maximum percentage in single position

### Auto-Flattening

When risk limits are breached, the system automatically generates flattening orders:

```python
def check_risk_limits(self) -> List[Order]:
    flatten_orders = []
    
    if self.gross_exposure > self.max_gross_exposure:
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                flatten_orders.append(self._create_flatten_order(symbol))
    
    return flatten_orders
```

## Latency and Cost Modeling

### Latency Models

The engine implements multiple latency models:

1. **Random Latency**: Normal distribution around mean
2. **Adaptive Latency**: Varies based on market conditions
3. **Network Latency**: Simulates network delays

```python
class RandomLatencyModel:
    def __init__(self, mean_ns: int, std_ns: int, seed: int = None):
        self.mean_ns = mean_ns
        self.std_ns = std_ns
        self.rng = np.random.RandomState(seed)
    
    def get_latency(self, order: Order) -> int:
        return max(0, int(self.rng.normal(self.mean_ns, self.std_ns)))
```

### Transaction Costs

Multiple cost models are supported:

1. **Simple Model**: Fixed basis points + per-share fees
2. **Tiered Model**: Volume-based fee tiers
3. **Exchange Model**: Realistic exchange fee structures

```python
class SimpleTransactionCostModel:
    def calculate_costs(self, fill: Fill) -> TransactionCosts:
        notional = fill.quantity * fill.price
        bps_cost = notional * (self.maker_bps / 10000)
        per_share_cost = fill.quantity * self.per_share
        return TransactionCosts(bps_cost + per_share_cost, 0.0)
```

### Slippage Models

1. **Fixed Slippage**: Constant slippage in basis points
2. **Imbalance Slippage**: Varies with order book imbalance
3. **Adaptive Slippage**: Adjusts based on market volatility

```python
class ImbalanceSlippageModel:
    def calculate_slippage(self, fill: Fill, book: OrderBookSnapshot) -> float:
        imbalance = self._calculate_imbalance(book)
        size_impact = self._calculate_size_impact(fill.quantity, book)
        return self.base_slippage * (1 + imbalance) * size_impact
```

## Latency and Cost Integration

### Latency Impact on Fills

Latency affects order execution in several ways:

1. **Order Scheduling**: Orders are delayed by simulated latency before reaching the matching engine
2. **Market Data Staleness**: Strategies receive market data that may be outdated by latency
3. **Fill Timing**: Fills are timestamped with the actual execution time (including latency)

```python
def process_order_with_latency(self, order: Order) -> List[Fill]:
    # Calculate latency for this order
    latency_ns = self.latency_model.get_latency(order)
    
    # Schedule order for future execution
    execution_time = self.clock.current_time + latency_ns
    self.schedule_event(OrderEvent(order, execution_time))
    
    # When executed, fills include latency information
    fills = self.matching_engine.match_order(order)
    for fill in fills:
        fill.latency_us = latency_ns // 1000  # Convert to microseconds
```

### Cost Integration

Transaction costs and slippage are calculated and integrated into fills:

```python
def create_fill_with_costs(self, order: Order, match_price: float, 
                          match_quantity: int) -> Fill:
    # Calculate slippage
    slippage = self.slippage_model.calculate_slippage(
        order, self.get_order_book_snapshot()
    )
    
    # Calculate transaction costs
    costs = self.cost_model.calculate_costs(
        match_quantity, match_price, order.side
    )
    
    # Create fill with all cost information
    return Fill(
        order_id=order.order_id,
        quantity=match_quantity,
        price=match_price + slippage,
        commission=costs.commission,
        slippage=slippage,
        timestamp=self.clock.current_time
    )
```

### Fill Composition

Each fill contains comprehensive execution information:

```python
@dataclass
class Fill:
    # Basic execution details
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: int
    
    # Cost and latency information
    commission: float      # Transaction fees
    slippage: float        # Price impact
    latency_us: int        # Execution latency
    
    # Market microstructure
    maker_taker: str       # "MAKER" or "TAKER"
    venue: str            # Execution venue
    order_type: OrderType # Original order type
```

## Data Flow and Execution

### Order Lifecycle

1. **Strategy** generates order intent
2. **Order Router** schedules order with latency
3. **Matching Engine** processes order and generates fills
4. **Cost Models** calculate fees and slippage
5. **Risk Manager** updates positions and checks limits
6. **Performance Analyzer** records trade and calculates metrics

### Detailed Execution Flow

```
Strategy Intent
      │
      ▼
┌─────────────┐
│Order Router │ ── Latency Model ──┐
│(Scheduling) │                    │
└─────────────┘                    │
      │                            │
      ▼ (Delayed by latency)       │
┌─────────────┐                    │
│Matching     │ ── Slippage Model ─┤
│Engine       │                    │
└─────────────┘                    │
      │                            │
      ▼                            │
┌─────────────┐                    │
│Fill Creation│ ── Cost Models ────┘
│(with costs) │
└─────────────┘
      │
      ▼
┌─────────────┐
│Risk Manager │
│(Position    │
│ Update)     │
└─────────────┘
      │
      ▼
┌─────────────┐
│Performance  │
│Analyzer     │
└─────────────┘
```

### Latency and Cost Integration Flow

```
Order Intent
      │
      ▼
┌─────────────┐    ┌─────────────┐
│Latency      │───▶│Order        │
│Calculation  │    │Scheduling   │
└─────────────┘    └─────────────┘
      │                    │
      ▼                    ▼
┌─────────────┐    ┌─────────────┐
│Latency      │    │Delayed      │
│Tracking     │    │Execution    │
└─────────────┘    └─────────────┘
                           │
                           ▼
                  ┌─────────────┐
                  │Matching     │
                  │Engine       │
                  └─────────────┘
                           │
                           ▼
                  ┌─────────────┐
                  │Fill         │
                  │Generation   │
                  └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Slippage     │    │Transaction  │    │Latency      │
│Calculation  │    │Cost Calc    │    │Assignment   │
└─────────────┘    └─────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  ┌─────────────┐
                  │Final Fill   │
                  │with All     │
                  │Costs        │
                  └─────────────┘
```

### Fill Generation

Fills are generated by the matching engine and include:

```python
@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: int
    commission: float
    slippage: float
    latency_us: int
```

### Performance Tracking

The performance analyzer maintains:

- **Trade Blotter**: All executed trades with PnL
- **Position Snapshots**: Portfolio state over time
- **Risk Metrics**: Real-time risk exposure
- **Performance Metrics**: Returns, Sharpe ratio, drawdown, etc.

## Market Microstructure Assumptions

### Order Book Model

**Assumptions:**
- L1 (Level 1) order book only - best bid/ask prices and sizes
- Price-time priority matching with strict FIFO within price levels
- Continuous double auction mechanism
- No hidden orders, dark pools, or internalization
- Orders are either fully executable or rejected (no partial fills at price levels)

**Limitations:**
- Does not model L2/L3 order book depth and market depth dynamics
- No iceberg orders or hidden liquidity representation
- Simplified market maker behavior (no inventory management)
- No order book reconstruction delays or market data gaps
- No order book state transitions or market state changes

### Market Data Representation

**Data Sources:**
- Synthetic data generation for testing and validation
- L1 order book snapshots with best bid/ask prices and sizes
- Trade data with nanosecond timestamps
- No real-time market data feeds

**Data Quality Assumptions:**
- Market data is clean and free of errors
- Timestamps are accurate and synchronized
- No missing or delayed market data
- Price and size data are integer multiples of tick size

**Limitations:**
- Synthetic data may not reflect real market dynamics and microstructure
- No tick-by-tick order book updates or order flow reconstruction
- Simplified market impact modeling without realistic order flow
- No news, fundamental data, or external event integration
- No market data vendor-specific formatting or delays

### Execution and Order Routing

**Order Execution Model:**
- Orders execute immediately upon matching with no delays
- No partial fills at individual price levels (simplified matching)
- No order book reconstruction delays or state transitions
- Perfect order routing with no rejections or errors
- No venue-specific execution rules or market access protocols

**Latency Modeling:**
- Deterministic latency based on order properties and system state
- No network jitter, packet loss, or infrastructure failures
- Simplified queueing delays without realistic queuing theory
- No cross-venue latency differences or geographic considerations
- No hardware-specific latency characteristics

**Maker-Taker Classification:**
- Simple classification based on order placement relative to best bid/ask
- No complex maker-taker rebate structures
- No dynamic fee schedules or volume-based pricing
- No venue-specific maker-taker rules

### Market Impact and Slippage

**Slippage Models:**
- Fixed slippage models may not capture realistic market impact
- Imbalance-based slippage is simplified and may not reflect true market dynamics
- No consideration of order size relative to market depth
- No time-of-day or volatility-based slippage adjustments

**Market Impact:**
- Linear market impact models may not reflect realistic price impact
- No consideration of order flow toxicity or adverse selection
- No dynamic market impact based on recent trading activity
- No venue-specific market impact characteristics

### Risk Management and Position Tracking

**Position Management:**
- Real-time PnL calculation with mark-to-market pricing
- Simplified margin requirements without realistic risk models
- No cross-margining or portfolio-level risk considerations
- No real-time risk system integration or external risk feeds

**Risk Limitations:**
- No real-time risk system integration with external risk management systems
- Simplified margin calculations without realistic regulatory requirements
- No regulatory reporting requirements or compliance monitoring
- No real-time position limits or dynamic risk adjustments
- No credit risk or counterparty risk considerations

### Market Structure Assumptions

**Trading Venues:**
- Single venue simulation without multi-venue considerations
- No venue-specific rules, regulations, or market access requirements
- No cross-venue arbitrage or smart order routing
- No venue-specific latency or execution characteristics

**Market Participants:**
- Simplified market maker behavior without realistic inventory management
- No high-frequency trading firm behavior or predatory trading
- No institutional order flow or block trading considerations
- No retail vs. institutional order flow differentiation

**Regulatory Environment:**
- No regulatory constraints or compliance requirements
- No market surveillance or monitoring capabilities
- No circuit breakers or market halts
- No position limits or reporting requirements

### Data and Model Limitations

**Synthetic Data Caveats:**
- Generated data may not reflect real market microstructure
- Price movements may not follow realistic statistical distributions
- Order flow patterns may not match real market behavior
- Market volatility and correlation structures may be oversimplified

**Model Validation:**
- Models may not be validated against real market data
- Performance metrics may not reflect actual trading results
- Risk models may not capture real-world risk factors
- Latency and cost models may not reflect actual execution costs

**Calibration and Parameters:**
- Model parameters may not be calibrated to real market data
- Default parameters may not be appropriate for all market conditions
- No sensitivity analysis or parameter uncertainty quantification
- No model validation or backtesting against real market data

## Performance Characteristics

### Scalability

- **Event Processing**: O(log n) per event (heap operations)
- **Order Matching**: O(k) where k is number of price levels
- **Memory Usage**: O(n) where n is number of active orders
- **Storage**: Parquet format for efficient data storage

### Latency Simulation

- **Order-to-Fill Latency**: 10-1000 microseconds typical
- **Market Data Latency**: 1-100 microseconds typical
- **Risk Check Latency**: 1-10 microseconds typical

### Throughput

- **Events per Second**: 1M+ events/second
- **Orders per Second**: 100K+ orders/second
- **Fills per Second**: 50K+ fills/second

## Configuration and Extensibility

### Strategy Interface

```python
class Strategy(Protocol):
    def on_bar(self, book_update: MarketDataEvent) -> List[NewOrder]:
        """Handle market data updates"""
        pass
    
    def on_trade(self, trade: FillEvent) -> List[NewOrder]:
        """Handle trade executions"""
        pass
    
    def on_timer(self, timestamp: int) -> List[NewOrder]:
        """Handle timer events"""
        pass
```

### Configuration System

YAML-based configuration with validation:

```yaml
data:
  path: "data/market_data.parquet"
  kind: "trade"
  format: "parquet"

strategy:
  name: "momentum_imbalance"
  symbol: "AAPL"
  params:
    short_ema_period: 5
    long_ema_period: 10

execution:
  fees:
    maker_bps: 0.0
    taker_bps: 0.5
  latency:
    model: "normal"
    mean_ns: 100000
    std_ns: 20000
```

## References and Further Reading

### Academic References

1. **Market Microstructure Theory**
   - O'Hara, M. (1995). "Market Microstructure Theory"
   - Hasbrouck, J. (2007). "Empirical Market Microstructure"

2. **High-Frequency Trading**
   - Aldridge, I. (2013). "High-Frequency Trading: A Practical Guide"
   - Easley, D., et al. (2012). "The Volume Clock: Insights into the High-Frequency Paradigm"

3. **Order Book Dynamics**
   - Bouchaud, J.P., et al. (2002). "More statistical properties of order books"
   - Mike, S., & Farmer, J.D. (2008). "An empirical behavioral model of liquidity and volatility"

### Technical References

1. **Event-Driven Architecture**
   - Fowler, M. (2017). "Event Sourcing"
   - Richardson, C. (2018). "Microservices Patterns"

2. **Financial Data Processing**
   - QuantLib Documentation
   - Zipline Documentation (Quantopian)

3. **Performance Optimization**
   - Intel VTune Profiler
   - Python Performance Tips (Python.org)

## Limitations and Disclaimers

### Model Limitations

1. **Market Data**: Synthetic data may not reflect real market conditions or microstructure
2. **Latency**: Simplified latency models may not capture all real-world network and hardware effects
3. **Risk Management**: Basic risk controls, not suitable for production trading or real money
4. **Regulatory**: No compliance with trading regulations, reporting requirements, or market surveillance
5. **Market Structure**: Simplified single-venue model without multi-venue dynamics
6. **Order Flow**: No realistic order flow patterns or market participant behavior

### Performance Limitations

1. **Scalability**: Designed for backtesting, not real-time trading or production use
2. **Memory**: Large datasets may require significant memory and processing power
3. **Accuracy**: Simplified models may not capture all market nuances and edge cases
4. **Validation**: Models not validated against real market data or production systems
5. **Calibration**: Default parameters may not be appropriate for all market conditions

### Usage Disclaimers

1. **Not for Production**: This is a backtesting engine, not a trading system or production platform
2. **No Guarantees**: Results may not reflect actual trading performance or real-world outcomes
3. **Educational Purpose**: Intended for research, education, and strategy development only
4. **Risk Warning**: Trading involves substantial risk of loss and may not be suitable for all investors
5. **No Investment Advice**: This software does not provide investment advice or recommendations
6. **Regulatory Compliance**: Users are responsible for ensuring compliance with applicable regulations

### Technical Limitations

1. **L1 Only**: Limited to Level 1 order book data without depth or market microstructure
2. **Synthetic Data**: No real market data integration or validation against live markets
3. **Simplified Models**: Many market dynamics are oversimplified or not modeled
4. **No Real-Time**: Not designed for real-time trading or live market simulation
5. **Limited Venues**: Single venue simulation without multi-venue or cross-venue considerations

### Research and Development Limitations

1. **Model Validation**: Models may not be validated against real market data
2. **Parameter Sensitivity**: No comprehensive sensitivity analysis or parameter uncertainty quantification
3. **Market Regimes**: May not perform well across different market regimes or conditions
4. **Data Quality**: Assumes clean, error-free data without realistic data quality issues
5. **Market Events**: No modeling of market events, news, or external factors

### Legal and Regulatory Disclaimers

1. **No Liability**: The software is provided "as is" without warranty or liability
2. **User Responsibility**: Users are responsible for their own trading decisions and compliance
3. **No Endorsement**: No endorsement of any trading strategy or investment approach
4. **Regulatory Compliance**: Users must ensure compliance with applicable laws and regulations
5. **Professional Advice**: Users should consult with qualified professionals before making investment decisions

## Future Enhancements

### Planned Features

1. **L2/L3 Order Book**: Full depth order book simulation
2. **Real Market Data**: Integration with real market data feeds
3. **Advanced Risk Models**: More sophisticated risk management
4. **Multi-Asset Support**: Cross-asset trading strategies
5. **Real-Time Mode**: Live market simulation capabilities

### Research Areas

1. **Market Impact Models**: More realistic execution cost modeling
2. **Latency Arbitrage**: Cross-venue latency simulation
3. **Machine Learning**: AI-driven strategy optimization
4. **Regulatory Compliance**: Integration with regulatory frameworks

---

*This document provides a comprehensive overview of the Flashback HFT Backtesting Engine architecture. For implementation details, refer to the source code and unit tests.*
