"""Italian-style tax accounting for realized paper-trading gains."""

from __future__ import annotations

from dataclasses import dataclass

from quantitative_trading.paper_trading.models import TaxLotResult


@dataclass(frozen=True)
class TaxState:
    """Running tax state for the experiment."""

    loss_carryforward_eur: float = 0.0

    def __post_init__(self) -> None:
        if self.loss_carryforward_eur < 0:
            raise ValueError("loss_carryforward_eur must be non-negative.")


class ItalianCapitalGainsTaxCalculator:
    """Compute realized capital-gains tax in EUR.

    This is a simplified experiment model, not tax advice. It taxes realized
    positive gains at the configured rate and tracks realized losses as an
    offset against future gains.
    """

    def __init__(self, tax_rate: float) -> None:
        if not 0 <= tax_rate <= 1:
            raise ValueError("tax_rate must be between 0 and 1.")
        self.tax_rate = tax_rate

    def close_lot(
        self,
        *,
        buy_notional_usd: float,
        sell_notional_usd: float,
        buy_fx_eur_usd: float,
        sell_fx_eur_usd: float,
        total_costs_usd: float,
        state: TaxState | None = None,
    ) -> TaxLotResult:
        """Compute tax for one closed weekly trade.

        Args:
            buy_notional_usd: Gross USD amount spent at entry.
            sell_notional_usd: Gross USD proceeds at exit.
            buy_fx_eur_usd: USD per EUR at entry.
            sell_fx_eur_usd: USD per EUR at exit.
            total_costs_usd: Estimated trade costs paid across entry and exit.
            state: Existing tax state with loss carryforward.

        Returns:
            Realized gain and tax due in EUR.
        """
        for name, value in (
            ("buy_notional_usd", buy_notional_usd),
            ("sell_notional_usd", sell_notional_usd),
            ("buy_fx_eur_usd", buy_fx_eur_usd),
            ("sell_fx_eur_usd", sell_fx_eur_usd),
            ("total_costs_usd", total_costs_usd),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if buy_fx_eur_usd == 0 or sell_fx_eur_usd == 0:
            raise ValueError("FX rates must be positive.")

        current_state = state or TaxState()
        cost_basis_eur = buy_notional_usd / buy_fx_eur_usd
        proceeds_eur = sell_notional_usd / sell_fx_eur_usd
        average_fx = (buy_fx_eur_usd + sell_fx_eur_usd) / 2
        costs_eur = total_costs_usd / average_fx
        realized_gain = proceeds_eur - cost_basis_eur - costs_eur

        if realized_gain >= 0:
            offset = min(realized_gain, current_state.loss_carryforward_eur)
            taxable_gain = realized_gain - offset
            loss_carryforward = current_state.loss_carryforward_eur - offset
        else:
            taxable_gain = 0.0
            loss_carryforward = current_state.loss_carryforward_eur + abs(realized_gain)

        return TaxLotResult(
            proceeds_eur=proceeds_eur,
            cost_basis_eur=cost_basis_eur,
            costs_eur=costs_eur,
            realized_gain_eur=realized_gain,
            taxable_gain_eur=taxable_gain,
            tax_due_eur=taxable_gain * self.tax_rate,
            loss_carryforward_eur=loss_carryforward,
        )
