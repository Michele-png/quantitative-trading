"""Tests for simplified Italian capital-gains tax accounting."""

from __future__ import annotations

import pytest

from quantitative_trading.paper_trading.tax import ItalianCapitalGainsTaxCalculator, TaxState


class TestItalianCapitalGainsTaxCalculator:
    """Tests for EUR realized-gain and tax calculations."""

    def test_positive_gain_is_taxed_after_costs(self) -> None:
        calculator = ItalianCapitalGainsTaxCalculator(tax_rate=0.26)

        result = calculator.close_lot(
            buy_notional_usd=1_000,
            sell_notional_usd=1_200,
            buy_fx_eur_usd=1.10,
            sell_fx_eur_usd=1.20,
            total_costs_usd=10,
        )

        expected_gain = (1_200 / 1.20) - (1_000 / 1.10) - (10 / 1.15)
        assert result.realized_gain_eur == pytest.approx(expected_gain)
        assert result.tax_due_eur == pytest.approx(expected_gain * 0.26)

    def test_loss_increases_loss_carryforward(self) -> None:
        calculator = ItalianCapitalGainsTaxCalculator(tax_rate=0.26)

        result = calculator.close_lot(
            buy_notional_usd=1_000,
            sell_notional_usd=900,
            buy_fx_eur_usd=1.0,
            sell_fx_eur_usd=1.0,
            total_costs_usd=0,
        )

        assert result.taxable_gain_eur == 0
        assert result.tax_due_eur == 0
        assert result.loss_carryforward_eur == pytest.approx(100)

    def test_loss_carryforward_offsets_future_gain(self) -> None:
        calculator = ItalianCapitalGainsTaxCalculator(tax_rate=0.26)

        result = calculator.close_lot(
            buy_notional_usd=1_000,
            sell_notional_usd=1_200,
            buy_fx_eur_usd=1.0,
            sell_fx_eur_usd=1.0,
            total_costs_usd=0,
            state=TaxState(loss_carryforward_eur=50),
        )

        assert result.taxable_gain_eur == pytest.approx(150)
        assert result.tax_due_eur == pytest.approx(39)
