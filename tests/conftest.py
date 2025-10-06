"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from flashback.core.events import MarketDataEvent, OrderEvent, FillEvent
from flashback.market.orders import Order, OrderSide, OrderType


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='1ms')
    data = []
    
    for i, ts in enumerate(timestamps):
        # Generate bid/ask data
        base_price = 150.0 + np.sin(i * 0.01) * 2.0
        spread = 0.01
        
        # Bid
        data.append({
            'ts': ts,
            'symbol': 'AAPL',
            'side': 'B',
            'price': base_price - spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
        
        # Ask
        data.append({
            'ts': ts + pd.Timedelta(microseconds=100),
            'symbol': 'AAPL',
            'side': 'S',
            'price': base_price + spread/2,
            'size': np.random.randint(100, 1000),
            'event_type': 'TICK'
        })
        
    return pd.DataFrame(data)


@pytest.fixture
def sample_orders():
    """Sample orders for testing."""
    orders = []
    
    # Buy order
    orders.append(OrderEvent(
        timestamp=pd.Timestamp('2024-01-01 09:30:00'),
        order_id='order_1',
        symbol='AAPL',
        side='B',
        order_type='LIMIT',
        quantity=100,
        price=150.0
    ))
    
    # Sell order
    orders.append(OrderEvent(
        timestamp=pd.Timestamp('2024-01-01 09:30:01'),
        order_id='order_2',
        symbol='AAPL',
        side='S',
        order_type='LIMIT',
        quantity=50,
        price=151.0
    ))
    
    # Market order
    orders.append(OrderEvent(
        timestamp=pd.Timestamp('2024-01-01 09:30:02'),
        order_id='order_3',
        symbol='AAPL',
        side='B',
        order_type='MARKET',
        quantity=200
    ))
    
    return orders


@pytest.fixture
def sample_fills():
    """Sample fills for testing."""
    fills = []
    
    fills.append(FillEvent(
        timestamp=pd.Timestamp('2024-01-01 09:30:00.001'),
        order_id='order_1',
        symbol='AAPL',
        side='B',
        quantity=100,
        price=150.0,
        commission=0.1,
        latency_us=50
    ))
    
    fills.append(FillEvent(
        timestamp=pd.Timestamp('2024-01-01 09:30:01.001'),
        order_id='order_2',
        symbol='AAPL',
        side='S',
        quantity=50,
        price=151.0,
        commission=0.05,
        latency_us=75
    ))
    
    return fills


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "source": "data/sample_data.parquet",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "symbols": ["AAPL"]
        },
        "strategy": {
            "name": "mean_reversion",
            "params": {
                "lookback": 100,
                "threshold": 0.001,
                "position_size": 100
            }
        },
        "risk": {
            "max_position": 1000,
            "max_drawdown": 0.05,
            "max_trade_size": 100
        },
        "execution": {
            "fees": {
                "commission_per_share": 0.001,
                "slippage_bps": 1.0
            },
            "latency": {
                "base_latency_us": 100,
                "jitter_us": 10
            }
        },
        "output": {
            "directory": "output",
            "trade_blotter": "trades.csv",
            "performance": "performance.json"
        }
    }


@pytest.fixture
def sample_order_book():
    """Sample order book for testing."""
    from flashback.market.book import MatchingEngine
    
    ob = OrderBook('AAPL')
    
    # Add some orders
    order1 = Order(
        order_id='order_1',
        symbol='AAPL',
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=150.0
    )
    
    order2 = Order(
        order_id='order_2',
        symbol='AAPL',
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50,
        price=151.0
    )
    
    ob.add_order(order1)
    ob.add_order(order2)
    
    return ob


@pytest.fixture
def sample_pnl_data():
    """Sample PnL data for testing."""
    timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
    pnl_values = np.cumsum(np.random.normal(0, 10, 100))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'total_pnl': pnl_values,
        'realized_pnl': pnl_values * 0.8,
        'unrealized_pnl': pnl_values * 0.2
    }).set_index('timestamp')


@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    timestamps = pd.date_range('2024-01-01', periods=50, freq='1H')
    
    trades = []
    for i, ts in enumerate(timestamps):
        trades.append({
            'timestamp': ts,
            'symbol': 'AAPL',
            'side': 'B' if i % 2 == 0 else 'S',
            'quantity': np.random.randint(10, 100),
            'price': 150.0 + np.random.normal(0, 1),
            'commission': np.random.uniform(0.01, 0.1),
            'latency_us': np.random.randint(50, 200)
        })
        
    return pd.DataFrame(trades).set_index('timestamp')
