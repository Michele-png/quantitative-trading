"""Tests for paper-trading orchestration helpers."""

from __future__ import annotations

import pytest

from quantitative_trading.paper_trading.models import AccountSnapshot
from quantitative_trading.paper_trading.orchestrator import budget_capped_account


class TestBudgetCappedAccount:
    """Tests for EUR budget-based order sizing."""

    def test_caps_alpaca_buying_power_to_initial_eur_budget(self) -> None:
        account = AccountSnapshot(
            currency="USD",
            cash=200_000,
            buying_power=200_000,
            portfolio_value=200_000,
            trading_blocked=False,
        )

        capped = budget_capped_account(
            account=account,
            initial_budget_eur=10_000,
            eur_usd_rate=1.2,
        )

        assert capped.buying_power == pytest.approx(12_000)

    def test_does_not_raise_buying_power_above_broker_limit(self) -> None:
        account = AccountSnapshot(
            currency="USD",
            cash=5_000,
            buying_power=5_000,
            portfolio_value=5_000,
            trading_blocked=False,
        )

        capped = budget_capped_account(
            account=account,
            initial_budget_eur=10_000,
            eur_usd_rate=1.2,
        )

        assert capped.buying_power == pytest.approx(5_000)
