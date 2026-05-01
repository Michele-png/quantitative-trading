"""Tests for paper-trading guardrails."""

from __future__ import annotations

from datetime import date

import pytest

from quantitative_trading.paper_trading.guardrails import GuardrailError, TradeGuardrails
from quantitative_trading.paper_trading.models import AccountSnapshot, WeeklyDecision


def _account() -> AccountSnapshot:
    return AccountSnapshot(
        currency="USD",
        cash=1_000,
        buying_power=1_000,
        portfolio_value=1_000,
        trading_blocked=False,
    )


def _decision(ticker: str = "AAPL") -> WeeklyDecision:
    return WeeklyDecision(
        trade_week=date(2026, 5, 4),
        ticker=ticker,
        confidence=0.8,
        thesis="Test thesis.",
        risks=["Test risk."],
        market_summary="Test market.",
        raw_response={},
        model="claude-test",
    )


class TestTradeGuardrails:
    """Tests for deterministic pre-order checks."""

    def test_builds_budget_bounded_order_plan(self) -> None:
        guardrails = TradeGuardrails(reserve_cash_usd=5)

        plan = guardrails.build_order_plan(
            decision=_decision(),
            account=_account(),
            allowed_symbols={"AAPL", "MSFT"},
            trade_week=date(2026, 5, 4),
        )

        assert plan.symbol == "AAPL"
        assert plan.notional_usd == pytest.approx(995)

    def test_rejects_symbol_outside_universe(self) -> None:
        guardrails = TradeGuardrails()

        with pytest.raises(GuardrailError, match="allowed"):
            guardrails.build_order_plan(
                decision=_decision("TSLA"),
                account=_account(),
                allowed_symbols={"AAPL"},
                trade_week=date(2026, 5, 4),
            )

    def test_rejects_blocked_account(self) -> None:
        guardrails = TradeGuardrails()
        account = AccountSnapshot(
            currency="USD",
            cash=1_000,
            buying_power=1_000,
            portfolio_value=1_000,
            trading_blocked=True,
        )

        with pytest.raises(GuardrailError, match="trading-blocked"):
            guardrails.build_order_plan(
                decision=_decision(),
                account=account,
                allowed_symbols={"AAPL"},
                trade_week=date(2026, 5, 4),
            )
