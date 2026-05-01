"""Weekly zero-shot LLM paper-trading workflow."""

from quantitative_trading.paper_trading.config import (
    PaperTradingConfig,
    get_paper_trading_config,
)
from quantitative_trading.paper_trading.orchestrator import (
    WeeklyPaperTrader,
    current_trade_week,
)

__all__ = [
    "PaperTradingConfig",
    "WeeklyPaperTrader",
    "current_trade_week",
    "get_paper_trading_config",
]
