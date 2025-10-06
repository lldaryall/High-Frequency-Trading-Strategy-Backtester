"""Configuration parser for YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .config import BacktestConfig, DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, ReportConfig

logger = logging.getLogger(__name__)


class ConfigParser:
    """Parser for YAML configuration files."""
    
    def __init__(self):
        """Initialize the configuration parser."""
        self.logger = logging.getLogger(__name__)
    
    def load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_path.suffix.lower() in ['.yaml', '.yml']:
            raise ValueError(f"Configuration file must be YAML format, got: {config_path.suffix}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if not isinstance(config_dict, dict):
                raise ValueError("Configuration file must contain a dictionary")
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config_dict
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    def parse_config(self, config_dict: Dict[str, Any]) -> BacktestConfig:
        """Parse configuration dictionary into BacktestConfig."""
        try:
            # Validate required sections
            required_sections = ["data", "strategy", "execution", "risk", "report"]
            for section in required_sections:
                if section not in config_dict:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Parse each section
            data_config = self._parse_data_config(config_dict["data"])
            strategy_config = self._parse_strategy_config(config_dict["strategy"])
            execution_config = self._parse_execution_config(config_dict["execution"])
            risk_config = self._parse_risk_config(config_dict["risk"])
            report_config = self._parse_report_config(config_dict["report"])
            
            # Create main config
            backtest_config = BacktestConfig(
                data=data_config,
                strategy=strategy_config,
                execution=execution_config,
                risk=risk_config,
                report=report_config,
                name=config_dict.get("name"),
                description=config_dict.get("description")
            )
            
            self.logger.info("Successfully parsed configuration")
            return backtest_config
            
        except Exception as e:
            raise ValueError(f"Error parsing configuration: {e}")
    
    def _parse_data_config(self, data_dict: Dict[str, Any]) -> DataConfig:
        """Parse data configuration section."""
        return DataConfig(
            path=data_dict["path"],
            kind=data_dict["kind"],
            symbol=data_dict.get("symbol"),
            start_time=data_dict.get("start_time"),
            end_time=data_dict.get("end_time")
        )
    
    def _parse_strategy_config(self, strategy_dict: Dict[str, Any]) -> StrategyConfig:
        """Parse strategy configuration section."""
        return StrategyConfig(
            name=strategy_dict["name"],
            params=strategy_dict["params"],
            symbol=strategy_dict.get("symbol"),
            enabled=strategy_dict.get("enabled", True),
            max_position=strategy_dict.get("max_position", 1000),
            max_order_size=strategy_dict.get("max_order_size", 100)
        )
    
    def _parse_execution_config(self, execution_dict: Dict[str, Any]) -> ExecutionConfig:
        """Parse execution configuration section."""
        return ExecutionConfig(
            fees=execution_dict["fees"],
            latency=execution_dict["latency"],
            slippage=execution_dict.get("slippage"),
            transaction_costs=execution_dict.get("transaction_costs")
        )
    
    def _parse_risk_config(self, risk_dict: Dict[str, Any]) -> RiskConfig:
        """Parse risk configuration section."""
        return RiskConfig(
            max_gross=risk_dict["max_gross"],
            max_pos_per_symbol=risk_dict["max_pos_per_symbol"],
            max_daily_loss=risk_dict["max_daily_loss"],
            max_drawdown=risk_dict.get("max_drawdown"),
            max_leverage=risk_dict.get("max_leverage")
        )
    
    def _parse_report_config(self, report_dict: Dict[str, Any]) -> ReportConfig:
        """Parse report configuration section."""
        return ReportConfig(
            output_dir=report_dict["output_dir"],
            format=report_dict.get("format", "json"),
            plots=report_dict.get("plots", True),
            detailed_trades=report_dict.get("detailed_trades", True),
            performance_metrics=report_dict.get("performance_metrics", True)
        )
    
    def validate_config(self, config: BacktestConfig) -> bool:
        """Validate configuration for completeness and consistency."""
        try:
            # Validate data file exists
            if not Path(config.data.path).exists():
                raise ValueError(f"Data file does not exist: {config.data.path}")
            
            # Validate strategy parameters
            if config.strategy.name == "momentum_imbalance":
                required_params = ["short_ema_period", "long_ema_period", "imbalance_threshold"]
                for param in required_params:
                    if param not in config.strategy.params:
                        raise ValueError(f"Missing required parameter for momentum_imbalance strategy: {param}")
            
            elif config.strategy.name == "mean_reversion":
                required_params = ["lookback_period", "z_score_threshold", "exit_z_score"]
                for param in required_params:
                    if param not in config.strategy.params:
                        raise ValueError(f"Missing required parameter for mean_reversion strategy: {param}")
            
            # Validate execution parameters
            if config.execution.fees["maker_bps"] < 0 or config.execution.fees["taker_bps"] < 0:
                raise ValueError("Fee rates cannot be negative")
            
            if config.execution.latency["mean_ns"] <= 0 or config.execution.latency["std_ns"] < 0:
                raise ValueError("Latency parameters must be positive")
            
            # Validate risk parameters
            if config.risk.max_gross <= 0:
                raise ValueError("max_gross must be positive")
            
            if config.risk.max_pos_per_symbol <= 0:
                raise ValueError("max_pos_per_symbol must be positive")
            
            if config.risk.max_daily_loss >= 0:
                raise ValueError("max_daily_loss must be negative")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise


def load_config(config_path: str) -> BacktestConfig:
    """Load and parse configuration from YAML file."""
    parser = ConfigParser()
    config_dict = parser.load_yaml(config_path)
    config = parser.parse_config(config_dict)
    parser.validate_config(config)
    return config


def validate_config(config: BacktestConfig) -> bool:
    """Validate a BacktestConfig object."""
    parser = ConfigParser()
    return parser.validate_config(config)
