"""Configuration dataclasses for flashback backtesting engine."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


@dataclass
class DataConfig:
    """Data source configuration."""
    path: str
    kind: str  # "book" or "trade"
    symbol: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def __post_init__(self):
        """Validate data configuration."""
        if self.kind not in ["book", "trade"]:
            raise ValueError(f"Data kind must be 'book' or 'trade', got '{self.kind}'")
        
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str  # "momentum_imbalance" or "mean_reversion"
    params: Dict[str, Any]
    symbol: Optional[str] = None
    enabled: bool = True
    max_position: int = 1000
    max_order_size: int = 100
    
    def __post_init__(self):
        """Validate strategy configuration."""
        if self.name not in ["momentum_imbalance", "mean_reversion"]:
            raise ValueError(f"Unknown strategy: {self.name}")


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    fees: Dict[str, float]
    latency: Dict[str, Any]
    slippage: Optional[Dict[str, Any]] = None
    transaction_costs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate execution configuration."""
        required_fee_keys = ["maker_bps", "taker_bps", "per_share"]
        for key in required_fee_keys:
            if key not in self.fees:
                raise ValueError(f"Missing required fee parameter: {key}")
        
        required_latency_keys = ["model", "mean_ns", "std_ns"]
        for key in required_latency_keys:
            if key not in self.latency:
                raise ValueError(f"Missing required latency parameter: {key}")


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_gross: float
    max_pos_per_symbol: int
    max_daily_loss: float
    max_drawdown: Optional[float] = None
    max_leverage: Optional[float] = None
    
    def __post_init__(self):
        """Validate risk configuration."""
        if self.max_gross <= 0:
            raise ValueError("max_gross must be positive")
        if self.max_pos_per_symbol <= 0:
            raise ValueError("max_pos_per_symbol must be positive")
        if self.max_daily_loss >= 0:
            raise ValueError("max_daily_loss must be negative")


@dataclass
class ReportConfig:
    """Reporting configuration."""
    output_dir: str
    format: str = "json"  # "json", "csv", "both"
    plots: bool = True
    detailed_trades: bool = True
    performance_metrics: bool = True
    
    def __post_init__(self):
        """Validate report configuration."""
        if self.format not in ["json", "csv", "both"]:
            raise ValueError(f"Report format must be 'json', 'csv', or 'both', got '{self.format}'")


@dataclass
class BacktestConfig:
    """Complete backtest configuration."""
    data: DataConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    risk: RiskConfig
    report: ReportConfig
    name: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BacktestConfig":
        """Create BacktestConfig from dictionary."""
        return cls(
            data=DataConfig(**config_dict["data"]),
            strategy=StrategyConfig(**config_dict["strategy"]),
            execution=ExecutionConfig(**config_dict["execution"]),
            risk=RiskConfig(**config_dict["risk"]),
            report=ReportConfig(**config_dict["report"]),
            name=config_dict.get("name"),
            description=config_dict.get("description")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BacktestConfig to dictionary."""
        return {
            "data": {
                "path": self.data.path,
                "kind": self.data.kind,
                "symbol": self.data.symbol,
                "start_time": self.data.start_time,
                "end_time": self.data.end_time
            },
            "strategy": {
                "name": self.strategy.name,
                "params": self.strategy.params,
                "symbol": self.strategy.symbol,
                "enabled": self.strategy.enabled,
                "max_position": self.strategy.max_position,
                "max_order_size": self.strategy.max_order_size
            },
            "execution": {
                "fees": self.execution.fees,
                "latency": self.execution.latency,
                "slippage": self.execution.slippage,
                "transaction_costs": self.execution.transaction_costs
            },
            "risk": {
                "max_gross": self.risk.max_gross,
                "max_pos_per_symbol": self.risk.max_pos_per_symbol,
                "max_daily_loss": self.risk.max_daily_loss,
                "max_drawdown": self.risk.max_drawdown,
                "max_leverage": self.risk.max_leverage
            },
            "report": {
                "output_dir": self.report.output_dir,
                "format": self.report.format,
                "plots": self.report.plots,
                "detailed_trades": self.report.detailed_trades,
                "performance_metrics": self.report.performance_metrics
            },
            "name": self.name,
            "description": self.description
        }
