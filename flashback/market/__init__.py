"""Market simulation components."""

from .orders import Order, Fill, Cancel, OrderSide, OrderType, TimeInForce, OrderBookLevel, OrderBookSnapshot
from .book import MatchingEngine
from .fees import FeeModel, BasisPointsFeeModel, create_standard_fee_model
from .latency import LatencyModel, create_standard_latency_model
from .slippage import (
    SlippageModel,
    FixedSlippageModel,
    ImbalanceSlippageModel,
    AdaptiveSlippageModel,
    SlippageConfig,
    create_slippage_model
)
from .transaction_costs import (
    TransactionCostModel,
    SimpleTransactionCostModel,
    TieredTransactionCostModel,
    ExchangeTransactionCostModel,
    TransactionCostConfig,
    FeeTier,
    TransactionCosts,
    create_transaction_cost_model,
    calculate_maker_taker_status
)

# C++ matching engine (optional)
try:
    from ._match_cpp_wrapper import (
        CppMatchEngine,
        CppFill,
        create_cpp_matching_engine,
        CPP_AVAILABLE
    )
    CPP_MATCHING_AVAILABLE = True
except ImportError:
    CppMatchEngine = None
    CppFill = None
    create_cpp_matching_engine = None
    CPP_AVAILABLE = False
    CPP_MATCHING_AVAILABLE = False

__all__ = [
    "Order",
    "Fill", 
    "Cancel",
    "OrderSide", 
    "OrderType",
    "TimeInForce",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "MatchingEngine",
    "FeeModel",
    "BasisPointsFeeModel",
    "create_standard_fee_model",
    "LatencyModel",
    "create_standard_latency_model",
    # Slippage
    "SlippageModel", "FixedSlippageModel", "ImbalanceSlippageModel", "AdaptiveSlippageModel",
    "SlippageConfig", "create_slippage_model",
    # Transaction Costs
    "TransactionCostModel", "SimpleTransactionCostModel", "TieredTransactionCostModel",
    "ExchangeTransactionCostModel", "TransactionCostConfig", "FeeTier", "TransactionCosts",
    "create_transaction_cost_model", "calculate_maker_taker_status",
    # C++ Matching Engine (optional)
    "CppMatchEngine", "CppFill", "create_cpp_matching_engine", "CPP_AVAILABLE", "CPP_MATCHING_AVAILABLE"
]
