"""Deterministic guardrails for LLM-generated trade decisions."""

from __future__ import annotations

from datetime import date

from quantitative_trading.paper_trading.models import AccountSnapshot, OrderPlan, WeeklyDecision


class GuardrailError(ValueError):
    """Raised when a decision violates deterministic trading constraints."""


class TradeGuardrails:
    """Validate decisions before any paper order is submitted."""

    def __init__(self, *, reserve_cash_usd: float = 5.0) -> None:
        if reserve_cash_usd < 0:
            raise ValueError("reserve_cash_usd must be non-negative.")
        self.reserve_cash_usd = reserve_cash_usd

    def build_order_plan(
        self,
        *,
        decision: WeeklyDecision,
        account: AccountSnapshot,
        allowed_symbols: set[str],
        trade_week: date,
    ) -> OrderPlan:
        """Validate a decision and convert it into a budget-bounded order plan."""
        ticker = decision.ticker.upper()
        if decision.trade_week != trade_week:
            raise GuardrailError("decision trade_week does not match requested trade week.")
        if ticker not in allowed_symbols:
            raise GuardrailError(f"{ticker} is not in the allowed S&P 500 candidate universe.")
        if account.currency != "USD":
            raise GuardrailError(f"Alpaca account currency must be USD, got {account.currency!r}.")
        if account.trading_blocked:
            raise GuardrailError("Alpaca account is trading-blocked.")
        if account.buying_power <= self.reserve_cash_usd:
            raise GuardrailError("Insufficient buying power after reserve cash.")
        if not 0 <= decision.confidence <= 1:
            raise GuardrailError("Decision confidence must be between 0 and 1.")

        return OrderPlan(
            trade_week=trade_week,
            symbol=ticker,
            notional_usd=round(account.buying_power - self.reserve_cash_usd, 2),
            side="buy",
            time_in_force="day",
        )
