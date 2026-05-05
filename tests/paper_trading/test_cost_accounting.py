"""Tests for paper-trading transaction cost estimates."""

from __future__ import annotations

import pytest

from quantitative_trading.paper_trading.cost_accounting import (
    TransactionCostConfig,
    TransactionCostEstimator,
)


class TestTransactionCostEstimator:
    """Tests for realistic paper-trading cost estimates."""

    def test_buy_cost_has_spread_and_slippage_only_by_default(self) -> None:
        estimator = TransactionCostEstimator(
            TransactionCostConfig(spread_bps=5, slippage_bps=5)
        )

        result = estimator.estimate(side="buy", notional_usd=10_000, shares=50)

        assert result.spread_cost_usd == pytest.approx(5.0)
        assert result.slippage_cost_usd == pytest.approx(5.0)
        assert result.regulatory_fees_usd == pytest.approx(0.0)
        assert result.total_usd == pytest.approx(10.0)

    def test_sell_cost_includes_regulatory_fees(self) -> None:
        estimator = TransactionCostEstimator(
            TransactionCostConfig(spread_bps=0, slippage_bps=0)
        )

        result = estimator.estimate(side="sell", notional_usd=100_000, shares=1_000)

        expected_sec_fee = 100_000 * 27.80 / 1_000_000
        expected_taf = 1_000 * 0.000166
        assert result.regulatory_fees_usd == pytest.approx(expected_sec_fee + expected_taf)

    def test_negative_notional_raises(self) -> None:
        estimator = TransactionCostEstimator()

        with pytest.raises(ValueError, match="notional_usd"):
            estimator.estimate(side="buy", notional_usd=-1, shares=1)
