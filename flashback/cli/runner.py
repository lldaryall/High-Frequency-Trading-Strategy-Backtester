"""Backtest runner for executing single backtests."""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import json

from ..config import BacktestConfig
from ..core.engine import BacktestEngine
from ..core.clock import SimClock
from ..core.events import EventType
from ..market.book import MatchingEngine
from ..market.latency import create_standard_latency_model
from ..market.fees import create_standard_fee_model
from ..market.slippage import create_slippage_model
from ..market.transaction_costs import create_transaction_cost_model
from ..strategy.base import BaseStrategy
from ..strategy.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from ..strategy.momentum_imbalance import MomentumImbalanceStrategy, MomentumImbalanceConfig
from ..risk.portfolio import PortfolioRiskManager
from ..metrics.performance import PerformanceAnalyzer
from ..exec.router import OrderRouter
from ..reporting.reporter import BacktestReporter

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Runs a single backtest with the given configuration."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtest runner."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.clock = SimClock(start_time=1000000000000000000)  # Start from first data timestamp
        self.engine = BacktestEngine({})  # Empty config for now
        self.order_books: Dict[str, MatchingEngine] = {}
        self.router = OrderRouter({})  # Empty config for now
        self.risk_manager = PortfolioRiskManager(
            initial_cash=1000000.0,  # Default initial capital
            risk_limits={
                "max_gross_exposure": config.risk.max_gross,
                "max_position_per_symbol": config.risk.max_pos_per_symbol,
                "max_daily_loss": config.risk.max_daily_loss
            }
        )
        
        # Initialize market models
        self._setup_market_models()
        
        # Initialize strategy
        self.strategy = self._create_strategy()
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Results storage
        self.snapshots: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.latencies: List[float] = []
    
    def _setup_market_models(self):
        """Setup market models from configuration."""
        # Latency model
        latency_config = self.config.execution.latency
        self.latency_model = create_standard_latency_model(
            constant_ns=latency_config["mean_ns"],
            mean_ns=latency_config["mean_ns"],
            std_ns=latency_config["std_ns"],
            seed=latency_config.get("seed", 42)
        )
        
        # Fee model
        fee_config = self.config.execution.fees
        self.fee_model = create_standard_fee_model(
            maker_bps=fee_config["maker_bps"],
            taker_bps=fee_config["taker_bps"],
            maker_per_share=fee_config["per_share"],
            taker_per_share=fee_config["per_share"]
        )
        
        # Slippage model (if configured)
        if self.config.execution.slippage:
            self.slippage_model = create_slippage_model(
                model_type=self.config.execution.slippage.get("model", "fixed"),
                config=self.config.execution.slippage
            )
        else:
            self.slippage_model = None
        
        # Transaction cost model (if configured)
        if self.config.execution.transaction_costs:
            self.transaction_cost_model = create_transaction_cost_model(
                model_type=self.config.execution.transaction_costs.get("model", "simple"),
                config=self.config.execution.transaction_costs
            )
        else:
            self.transaction_cost_model = None
    
    def _create_strategy(self) -> BaseStrategy:
        """Create strategy instance from configuration."""
        strategy_name = self.config.strategy.name
        strategy_params = self.config.strategy.params
        
        if strategy_name == "mean_reversion":
            config = MeanReversionConfig(
                strategy_id="mean_reversion_1",
                symbol=self.config.strategy.symbol or "AAPL",
                enabled=self.config.strategy.enabled,
                max_position=self.config.strategy.max_position,
                max_order_size=self.config.strategy.max_order_size,
                risk_limits={},
                **strategy_params
            )
            return MeanReversionStrategy(config)
        
        elif strategy_name == "momentum_imbalance":
            config = MomentumImbalanceConfig(
                strategy_id="momentum_imbalance_1",
                symbol=self.config.strategy.symbol or "AAPL",
                enabled=self.config.strategy.enabled,
                max_position=self.config.strategy.max_position,
                max_order_size=self.config.strategy.max_order_size,
                risk_limits={},
                **strategy_params
            )
            return MomentumImbalanceStrategy(config)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _load_market_data(self) -> List[Dict[str, Any]]:
        """Load market data from configured source."""
        data_path = Path(self.config.data.path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data based on file type
        if data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Convert to events
        events = []
        for _, row in df.iterrows():
            event = {
                "timestamp": int(row.get("timestamp", 0)),
                "symbol": row.get("symbol", "AAPL"),
                "side": row.get("side", "BUY"),
                "price": float(row.get("price", 0.0)),
                "size": int(row.get("size", 0)),
                "event_type": row.get("event_type", "TRADE")
            }
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        
        self.logger.info(f"Loaded {len(events)} market data events")
        return events
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest."""
        self.logger.info("Starting backtest...")
        
        # Load market data
        market_data = self._load_market_data()
        
        # Initialize strategy
        self.strategy.start()
        
        # Process events
        for event_data in market_data:
            self._process_market_event(event_data)
        
        # Finalize strategy
        self.strategy.stop()
        
        # Add snapshots and trades to analyzer
        for snapshot in self.snapshots:
            from ..risk.portfolio import PortfolioSnapshot
            # Filter out None positions
            positions = {k: v for k, v in snapshot["positions"].items() if v is not None}
            ps = PortfolioSnapshot(
                timestamp=snapshot["timestamp"],
                cash=snapshot["cash"],
                positions=positions,
                total_market_value=snapshot["portfolio_value"] - snapshot["cash"],
                total_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                gross_exposure=0.0,
                net_exposure=0.0
            )
            self.performance_analyzer.snapshots.append(ps)
        
        for trade in self.trades:
            from ..metrics.performance import TradeRecord
            tr = TradeRecord(
                trade_id=trade.get("trade_id", "unknown"),
                symbol=trade.get("symbol", "unknown"),
                side=trade.get("side", "unknown"),
                quantity=trade.get("quantity", 0),
                price=trade.get("price", 0.0),
                timestamp=trade.get("timestamp", 0),
                pnl=trade.get("pnl", 0.0),
                commission=trade.get("commission", 0.0),
                latency_us=trade.get("latency_us", 0)
            )
            self.performance_analyzer.trades.append(tr)
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics()
        
        # Create reporter and generate comprehensive report
        reporter = BacktestReporter(self.config)
        reporter.generate_report(
            trades=self.trades,
            positions=self.snapshots,
            blotter=pd.DataFrame([{
                'order_id': order_state.order.order_id,
                'strategy_id': order_state.strategy_id,
                'symbol': order_state.order.symbol,
                'side': order_state.order.side.value,
                'order_type': order_state.order.order_type.value,
                'quantity': order_state.order.quantity,
                'price': order_state.order.price,
                'status': order_state.order.status.value,
                'submitted_at': order_state.submitted_at,
                'filled_at': order_state.filled_at,
                'cancelled_at': order_state.cancelled_at,
                'total_filled_qty': order_state.total_filled_qty,
                'avg_fill_price': order_state.avg_fill_price
            } for order_state in self.router.blotter.orders.values()]),
            performance_metrics=performance_metrics
        )
        
        # Compile results
        results = {
            "config": self.config.to_dict(),
            "performance": performance_metrics.to_dict(),
            "snapshots": self.snapshots,
            "trades": self.trades,
            "latencies": self.latencies,
            "summary": {
                "total_events": len(market_data),
                "total_trades": len(self.trades),
                "total_snapshots": len(self.snapshots),
                "strategy": self.config.strategy.name,
                "symbol": self.config.strategy.symbol or "AAPL"
            },
            "run_directory": str(reporter.output_dir)
        }
        
        self.logger.info(f"Backtest completed successfully. Results saved to {reporter.output_dir}")
        return results
    
    def _create_run_directory(self) -> Path:
        """Create timestamped run directory."""
        from datetime import datetime
        
        # Create runs directory if it doesn't exist
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = runs_dir / timestamp
        run_dir.mkdir(exist_ok=True)
        
        return run_dir
    
    def _save_config_yaml(self, run_dir: Path):
        """Save resolved configuration as YAML."""
        import yaml
        
        config_file = run_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved configuration to {config_file}")
    
    def _save_performance_files(self, run_dir: Path, performance_metrics):
        """Save performance metrics as JSON and CSV."""
        import json
        import pandas as pd
        
        # Save as JSON
        perf_json = run_dir / "performance.json"
        with open(perf_json, 'w') as f:
            json.dump(performance_metrics.to_dict(), f, indent=2)
        
        # Save as CSV
        perf_csv = run_dir / "performance.csv"
        perf_df = pd.DataFrame([performance_metrics.to_dict()])
        perf_df.to_csv(perf_csv, index=False)
        
        self.logger.info(f"Saved performance metrics to {perf_json} and {perf_csv}")
    
    def _save_plots(self, run_dir: Path, performance_metrics):
        """Generate and save all required plots."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # Set style
        plt.style.use('default')
        
        # 1. Equity Curve
        if self.snapshots:
            self._plot_equity_curve(run_dir)
        
        # 2. Drawdown Curve
        if self.snapshots:
            self._plot_drawdown_curve(run_dir)
        
        # 3. Trade PnL Histogram
        if self.trades:
            self._plot_trade_pnl_histogram(run_dir)
        
        self.logger.info(f"Saved plots to {run_dir}")
    
    def _plot_equity_curve(self, run_dir: Path):
        """Plot equity curve."""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not self.snapshots:
            return
        
        # Convert snapshots to DataFrame
        snapshots_df = pd.DataFrame(self.snapshots)
        snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'], unit='ns')
        
        plt.figure(figsize=(12, 6))
        plt.plot(snapshots_df['timestamp'], snapshots_df['portfolio_value'], linewidth=2)
        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        equity_file = run_dir / "equity_curve.png"
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_curve(self, run_dir: Path):
        """Plot drawdown curve."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        if not self.snapshots:
            return
        
        # Convert snapshots to DataFrame
        snapshots_df = pd.DataFrame(self.snapshots)
        snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'], unit='ns')
        
        # Calculate running maximum and drawdown
        snapshots_df['running_max'] = snapshots_df['portfolio_value'].cummax()
        snapshots_df['drawdown'] = (snapshots_df['portfolio_value'] - snapshots_df['running_max']) / snapshots_df['running_max'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(snapshots_df['timestamp'], snapshots_df['drawdown'], 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(snapshots_df['timestamp'], snapshots_df['drawdown'], color='red', linewidth=1)
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        drawdown_file = run_dir / "drawdown.png"
        plt.savefig(drawdown_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_pnl_histogram(self, run_dir: Path):
        """Plot trade PnL histogram."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        if not self.trades:
            return
        
        # Extract PnL values
        pnl_values = [trade.get('pnl', 0) for trade in self.trades if trade.get('pnl') is not None]
        
        if not pnl_values:
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(pnl_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        plt.title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Trade PnL ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        hist_file = run_dir / "trade_pnl_hist.png"
        plt.savefig(hist_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_blotter_parquet(self, run_dir: Path):
        """Save trade blotter as Parquet."""
        import pandas as pd
        
        if not self.trades:
            # Create empty blotter
            blotter_df = pd.DataFrame(columns=[
                'trade_id', 'symbol', 'side', 'quantity', 'price', 'timestamp', 
                'pnl', 'commission', 'latency_us'
            ])
        else:
            blotter_df = pd.DataFrame(self.trades)
        
        blotter_file = run_dir / "blotter.parquet"
        blotter_df.to_parquet(blotter_file, index=False)
        
        self.logger.info(f"Saved trade blotter to {blotter_file}")
    
    def _save_positions_parquet(self, run_dir: Path):
        """Save position snapshots as Parquet."""
        import pandas as pd
        
        if not self.snapshots:
            # Create empty positions
            positions_df = pd.DataFrame(columns=[
                'timestamp', 'portfolio_value', 'cash', 'positions', 
                'unrealized_pnl', 'realized_pnl'
            ])
        else:
            positions_df = pd.DataFrame(self.snapshots)
            # Convert timestamp to datetime for better readability
            positions_df['timestamp'] = pd.to_datetime(positions_df['timestamp'], unit='ns')
        
        positions_file = run_dir / "positions.parquet"
        positions_df.to_parquet(positions_file, index=False)
        
        self.logger.info(f"Saved position snapshots to {positions_file}")
    
    def _process_market_event(self, event_data: Dict[str, Any]):
        """Process a single market event."""
        timestamp = event_data["timestamp"]
        symbol = event_data["symbol"]
        
        # Update clock
        self.clock.advance_to(timestamp)
        
        # Create order book if needed
        if symbol not in self.order_books:
            self.order_books[symbol] = MatchingEngine(symbol)
        
        # Process the event through the strategy
        if event_data["event_type"] == "TRADE":
            # Create FillEvent for trade
            from ..core.events import FillEvent
            fill_event = FillEvent(
                event_type=EventType.FILL,
                timestamp=pd.Timestamp(timestamp, unit='ns'),
                order_id=f"order_{timestamp}",
                symbol=symbol,
                side=event_data["side"],
                price=event_data["price"],
                quantity=event_data["size"],
                commission=0.0,
                latency_us=0
            )
            self.strategy.on_trade(fill_event)
        else:
            # Create MarketDataEvent for book updates
            from ..core.events import MarketDataEvent
            market_event = MarketDataEvent(
                event_type=EventType.MARKET_DATA,
                timestamp=pd.Timestamp(timestamp, unit='ns'),
                symbol=symbol,
                side=event_data["side"],
                price=event_data["price"],
                size=event_data["size"],
                event_type_str=event_data["event_type"],
                data={"mid_price": event_data["price"], "volume": event_data["size"]}
            )
            self.strategy.on_bar(market_event)
        
        # Process any pending orders
        self._process_pending_orders()
        
        # Update risk manager
        snapshot = self.order_books[symbol].get_snapshot(timestamp)
        if snapshot.bids and snapshot.asks:
            mid_price = (snapshot.bids[0].price + snapshot.asks[0].price) / 2
            self.risk_manager.update_market_prices({symbol: mid_price}, timestamp)
        
        # Record snapshot
        snapshot = {
            "timestamp": timestamp,
            "symbol": symbol,
            "portfolio_value": self.risk_manager.cash + self.risk_manager.get_total_pnl(),
            "cash": self.risk_manager.cash,
            "positions": {symbol: self.risk_manager.get_position(symbol) if self.risk_manager.get_position(symbol) else None}
        }
        self.snapshots.append(snapshot)
    
    def _process_pending_orders(self):
        """Process any pending orders from the strategy."""
        # This would integrate with the order router and matching engine
        # For now, we'll just log that orders would be processed
        pass
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save backtest results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance metrics
        if self.config.report.format in ["json", "both"]:
            with open(output_dir / "performance.json", "w") as f:
                json.dump(results["performance"], f, indent=2)
        
        if self.config.report.format in ["csv", "both"]:
            # Convert performance metrics to CSV
            perf_df = pd.DataFrame([results["performance"]])
            perf_df.to_csv(output_dir / "performance.csv", index=False)
        
        # Save detailed results
        if self.config.report.detailed_trades:
            with open(output_dir / "detailed_results.json", "w") as f:
                json.dump(results, f, indent=2)
        
        # Save snapshots
        if self.snapshots:
            snapshots_df = pd.DataFrame(self.snapshots)
            snapshots_df.to_csv(output_dir / "snapshots.csv", index=False)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(output_dir / "trades.csv", index=False)
        
        # Generate plots if requested
        if self.config.report.plots:
            self._generate_plots(results, output_dir)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate performance plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Equity curve
            if self.snapshots:
                snapshots_df = pd.DataFrame(self.snapshots)
                plt.figure(figsize=(12, 6))
                plt.plot(snapshots_df["timestamp"], snapshots_df["portfolio_value"])
                plt.title("Portfolio Value Over Time")
                plt.xlabel("Timestamp")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                plt.savefig(output_dir / "equity_curve.png")
                plt.close()
            
            # Trade PnL histogram
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                if "pnl" in trades_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.hist(trades_df["pnl"], bins=50, alpha=0.7)
                    plt.title("Trade PnL Distribution")
                    plt.xlabel("PnL")
                    plt.ylabel("Frequency")
                    plt.grid(True)
                    plt.savefig(output_dir / "trade_pnl_histogram.png")
                    plt.close()
            
            self.logger.info("Plots generated successfully")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
