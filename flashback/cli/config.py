"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import jsonschema

from ..utils.logger import get_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger = get_logger(__name__)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger = get_logger(__name__)
    
    # Define configuration schema
    schema = {
        "type": "object",
        "required": ["data", "strategy", "risk", "execution", "output"],
        "properties": {
            "data": {
                "type": "object",
                "required": ["source"],
                "properties": {
                    "source": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                }
            },
            "strategy": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object"}
                }
            },
            "risk": {
                "type": "object",
                "properties": {
                    "max_position": {"type": "number"},
                    "max_drawdown": {"type": "number"},
                    "max_trade_size": {"type": "number"},
                    "max_position_per_symbol": {"type": "number"},
                    "max_total_exposure": {"type": "number"},
                }
            },
            "execution": {
                "type": "object",
                "properties": {
                    "fees": {"type": "object"},
                    "latency": {"type": "object"},
                    "matching": {"type": "object"},
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "trade_blotter": {"type": "string"},
                    "performance": {"type": "string"},
                    "plots": {"type": "string"},
                }
            }
        }
    }
    
    try:
        jsonschema.validate(config, schema)
        logger.info("Configuration validation passed")
        
    except jsonschema.ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e.message}")
    except Exception as e:
        raise ValueError(f"Error validating configuration: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def load_config_with_overrides(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration with optional overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional override values
        
    Returns:
        Merged configuration
    """
    config = load_config(config_path)
    
    if overrides:
        config = merge_configs(config, overrides)
        
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    logger = get_logger(__name__)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        raise ValueError(f"Error saving configuration: {e}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
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
            "max_trade_size": 100,
            "max_position_per_symbol": 1000,
            "max_total_exposure": 10000
        },
        "execution": {
            "fees": {
                "commission_per_share": 0.001,
                "slippage_bps": 1.0,
                "exchange_fee_bps": 0.0,
                "sec_fee_rate": 0.0000229
            },
            "latency": {
                "base_latency_us": 100,
                "network_latency_us": 50,
                "processing_latency_us": 30,
                "exchange_latency_us": 20,
                "jitter_us": 10
            },
            "matching": {
                "partial_fills": True,
                "min_tick_size": 0.01
            }
        },
        "output": {
            "directory": "output",
            "trade_blotter": "trades.csv",
            "performance": "performance.json",
            "plots": "plots/"
        }
    }


def validate_data_source(config: Dict[str, Any]) -> None:
    """
    Validate data source configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If data source is invalid
    """
    data_config = config.get("data", {})
    source = data_config.get("source")
    
    if not source:
        raise ValueError("Data source not specified")
        
    if not os.path.exists(source):
        raise ValueError(f"Data source file not found: {source}")
        
    # Check file extension
    file_ext = os.path.splitext(source)[1].lower()
    if file_ext not in ['.parquet', '.csv']:
        raise ValueError(f"Unsupported file format: {file_ext}")


def validate_strategy_config(config: Dict[str, Any]) -> None:
    """
    Validate strategy configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If strategy configuration is invalid
    """
    strategy_config = config.get("strategy", {})
    strategy_name = strategy_config.get("name")
    
    if not strategy_name:
        raise ValueError("Strategy name not specified")
        
    # Check if strategy is supported
    supported_strategies = ["mean_reversion", "momentum"]
    if strategy_name not in supported_strategies:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
        
    # Validate strategy parameters
    params = strategy_config.get("params", {})
    if strategy_name == "mean_reversion":
        required_params = ["lookback", "threshold", "position_size"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter for mean_reversion strategy: {param}")
                
    elif strategy_name == "momentum":
        required_params = ["short_ma_period", "long_ma_period", "imbalance_threshold"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter for momentum strategy: {param}")


def validate_risk_config(config: Dict[str, Any]) -> None:
    """
    Validate risk configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If risk configuration is invalid
    """
    risk_config = config.get("risk", {})
    
    # Validate position limits
    max_position = risk_config.get("max_position", 0)
    if max_position <= 0:
        raise ValueError("max_position must be positive")
        
    max_drawdown = risk_config.get("max_drawdown", 0)
    if max_drawdown <= 0 or max_drawdown > 1:
        raise ValueError("max_drawdown must be between 0 and 1")
        
    max_trade_size = risk_config.get("max_trade_size", 0)
    if max_trade_size <= 0:
        raise ValueError("max_trade_size must be positive")
