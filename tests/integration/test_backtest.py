"""Integration tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from flashback.core.engine import BacktestEngine
from flashback.data.loader import DataLoader
from flashback.strategy.mean_reversion import MeanReversionStrategy
from flashback.strategy.momentum import MomentumStrategy


class TestBacktestIntegration:
    """Integration tests for the backtesting engine."""
    
    def test_full_backtest_mean_reversion(self, sample_config):
        """Test full backtest with mean reversion strategy."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data
            timestamps = pd.date_range('2024-01-01', periods=1000, freq='1ms')
            data = []
            
            for i, ts in enumerate(timestamps):
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
                
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            # Update config to use temporary file
            sample_config['data']['source'] = tmp_file.name
            
            try:
                # Run backtest
                engine = BacktestEngine(sample_config)
                results = engine.run()
                
                # Verify results
                assert 'statistics' in results
                assert 'positions' in results
                assert 'pnl' in results
                assert 'trades' in results
                assert 'order_books' in results
                
                # Check statistics
                stats = results['statistics']
                assert stats['events_processed'] > 0
                
                # Check that we have some market data
                assert 'AAPL' in results['order_books']
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    def test_full_backtest_momentum(self, sample_config):
        """Test full backtest with momentum strategy."""
        # Update config for momentum strategy
        sample_config['strategy'] = {
            'name': 'momentum',
            'params': {
                'short_ma_period': 10,
                'long_ma_period': 50,
                'imbalance_threshold': 0.3,
                'position_size': 100
            }
        }
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data
            timestamps = pd.date_range('2024-01-01', periods=1000, freq='1ms')
            data = []
            
            for i, ts in enumerate(timestamps):
                base_price = 150.0 + i * 0.01  # Trending price
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
                
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            # Update config to use temporary file
            sample_config['data']['source'] = tmp_file.name
            
            try:
                # Run backtest
                engine = BacktestEngine(sample_config)
                results = engine.run()
                
                # Verify results
                assert 'statistics' in results
                assert 'positions' in results
                assert 'pnl' in results
                assert 'trades' in results
                assert 'order_books' in results
                
                # Check statistics
                stats = results['statistics']
                assert stats['events_processed'] > 0
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    def test_data_loader_integration(self, sample_config):
        """Test data loader integration."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data
            timestamps = pd.date_range('2024-01-01', periods=100, freq='1s')
            data = []
            
            for i, ts in enumerate(timestamps):
                base_price = 150.0 + np.sin(i * 0.1) * 2.0
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
                
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            try:
                # Test data loader
                data_config = sample_config['data'].copy()
                data_config['source'] = tmp_file.name
                
                loader = DataLoader(data_config)
                loaded_data = loader.load()
                
                # Verify data
                assert len(loaded_data) == 200  # 100 timestamps * 2 (bid/ask)
                assert 'ts' in loaded_data.columns
                assert 'symbol' in loaded_data.columns
                assert 'side' in loaded_data.columns
                assert 'price' in loaded_data.columns
                assert 'size' in loaded_data.columns
                assert 'event_type' in loaded_data.columns
                
                # Verify data validation
                assert loader.validate_data(loaded_data)
                
                # Verify data info
                info = loader.get_data_info(loaded_data)
                assert info['total_records'] == 200
                assert 'AAPL' in info['symbols']
                assert 'TICK' in info['event_types']
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    def test_strategy_integration(self):
        """Test strategy integration."""
        # Test mean reversion strategy
        config = {
            "lookback": 100,
            "threshold": 0.001,
            "position_size": 100
        }
        
        strategy = MeanReversionStrategy(config)
        strategy.initialize(pd.Timestamp('2024-01-01 09:30:00'))
        
        # Add some market data
        for i in range(150):
            event = MarketDataEvent(
                timestamp=pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i),
                symbol='AAPL',
                side='B' if i % 2 == 0 else 'S',
                price=150.0 + np.sin(i * 0.01) * 2.0,
                size=100,
                event_type_str='TICK'
            )
            strategy.on_market_data(event, event.timestamp)
            
        # Check that strategy has processed data
        assert len(strategy.last_mid_prices['AAPL']) > 0
        
        # Test momentum strategy
        config = {
            "short_ma_period": 10,
            "long_ma_period": 50,
            "imbalance_threshold": 0.3,
            "position_size": 100
        }
        
        strategy = MomentumStrategy(config)
        strategy.initialize(pd.Timestamp('2024-01-01 09:30:00'))
        
        # Add some market data
        for i in range(60):
            event = MarketDataEvent(
                timestamp=pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i),
                symbol='AAPL',
                side='B' if i % 2 == 0 else 'S',
                price=150.0 + i * 0.01,
                size=100,
                event_type_str='TICK'
            )
            strategy.on_market_data(event, event.timestamp)
            
        # Check that strategy has processed data
        assert len(strategy.price_history['AAPL']) > 0
        
    def test_risk_management_integration(self, sample_config):
        """Test risk management integration."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data
            timestamps = pd.date_range('2024-01-01', periods=100, freq='1s')
            data = []
            
            for i, ts in enumerate(timestamps):
                base_price = 150.0 + np.sin(i * 0.1) * 2.0
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
                
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            # Update config to use temporary file
            sample_config['data']['source'] = tmp_file.name
            
            try:
                # Run backtest
                engine = BacktestEngine(sample_config)
                results = engine.run()
                
                # Check risk management
                assert 'positions' in results
                assert 'pnl' in results
                
                # Check that risk manager is working
                positions = results['positions']
                pnl = results['pnl']
                
                assert isinstance(positions, dict)
                assert isinstance(pnl, dict)
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    def test_performance_metrics_integration(self, sample_config):
        """Test performance metrics integration."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data
            timestamps = pd.date_range('2024-01-01', periods=100, freq='1s')
            data = []
            
            for i, ts in enumerate(timestamps):
                base_price = 150.0 + np.sin(i * 0.1) * 2.0
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
                
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            # Update config to use temporary file
            sample_config['data']['source'] = tmp_file.name
            
            try:
                # Run backtest
                engine = BacktestEngine(sample_config)
                results = engine.run()
                
                # Check performance metrics
                metrics = engine.get_performance_metrics()
                
                assert isinstance(metrics, dict)
                # Note: metrics might be empty if no trades were executed
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    def test_error_handling(self, sample_config):
        """Test error handling in backtest."""
        # Test with invalid data source
        sample_config['data']['source'] = 'nonexistent_file.parquet'
        
        engine = BacktestEngine(sample_config)
        
        with pytest.raises(FileNotFoundError):
            engine.run()
            
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with missing required fields
        invalid_config = {
            "data": {
                "source": "data/sample_data.parquet"
            }
            # Missing strategy, risk, execution, output
        }
        
        with pytest.raises(ValueError):
            engine = BacktestEngine(invalid_config)
            
    def test_multiple_symbols(self, sample_config):
        """Test backtest with multiple symbols."""
        # Update config for multiple symbols
        sample_config['data']['symbols'] = ['AAPL', 'MSFT']
        
        # Create temporary data file with multiple symbols
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Generate sample data for multiple symbols
            timestamps = pd.date_range('2024-01-01', periods=100, freq='1s')
            data = []
            
            for symbol in ['AAPL', 'MSFT']:
                for i, ts in enumerate(timestamps):
                    base_price = 150.0 if symbol == 'AAPL' else 200.0
                    base_price += np.sin(i * 0.1) * 2.0
                    spread = 0.01
                    
                    # Bid
                    data.append({
                        'ts': ts,
                        'symbol': symbol,
                        'side': 'B',
                        'price': base_price - spread/2,
                        'size': np.random.randint(100, 1000),
                        'event_type': 'TICK'
                    })
                    
                    # Ask
                    data.append({
                        'ts': ts + pd.Timedelta(microseconds=100),
                        'symbol': symbol,
                        'side': 'S',
                        'price': base_price + spread/2,
                        'size': np.random.randint(100, 1000),
                        'event_type': 'TICK'
                    })
                    
            df = pd.DataFrame(data)
            df.to_parquet(tmp_file.name)
            
            # Update config to use temporary file
            sample_config['data']['source'] = tmp_file.name
            
            try:
                # Run backtest
                engine = BacktestEngine(sample_config)
                results = engine.run()
                
                # Verify results
                assert 'order_books' in results
                assert 'AAPL' in results['order_books']
                assert 'MSFT' in results['order_books']
                
            finally:
                # Clean up
                os.unlink(tmp_file.name)
