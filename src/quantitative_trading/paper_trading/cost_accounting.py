"""Transaction-cost estimates for paper trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quantitative_trading.paper_trading.models import TransactionCostEstimate


@dataclass(frozen=True)
class TransactionCostConfig:
    """Configurable assumptions for realistic paper-trading frictions.

    Defaults are conservative approximations for liquid US equities. They are
    intentionally explicit because paper brokers often fill at optimistic prices.
    """

    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    broker_fee_usd: float = 0.0
    sec_fee_per_million_usd: float = 27.80
    finra_taf_per_share_usd: float = 0.000166
    finra_taf_cap_usd: float = 8.30

    def __post_init__(self) -> None:
        for name, value in (
            ("spread_bps", self.spread_bps),
            ("slippage_bps", self.slippage_bps),
            ("broker_fee_usd", self.broker_fee_usd),
            ("sec_fee_per_million_usd", self.sec_fee_per_million_usd),
            ("finra_taf_per_share_usd", self.finra_taf_per_share_usd),
            ("finra_taf_cap_usd", self.finra_taf_cap_usd),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


class TransactionCostEstimator:
    """Estimate costs to subtract from paper-trading P&L."""

    def __init__(self, config: TransactionCostConfig | None = None) -> None:
        self.config = config or TransactionCostConfig()

    def estimate(
        self,
        *,
        side: Literal["buy", "sell"],
        notional_usd: float,
        shares: float,
    ) -> TransactionCostEstimate:
        """Estimate transaction costs for one buy or sell order.

        Args:
            side: Order side. Regulatory fees apply only on sells.
            notional_usd: Gross order value.
            shares: Approximate number of shares.

        Returns:
            Cost estimate broken out by source.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if notional_usd < 0:
            raise ValueError("notional_usd must be non-negative.")
        if shares < 0:
            raise ValueError("shares must be non-negative.")

        spread_cost = notional_usd * self.config.spread_bps / 10_000
        slippage_cost = notional_usd * self.config.slippage_bps / 10_000
        regulatory_fees = 0.0
        if side == "sell":
            sec_fee = notional_usd * self.config.sec_fee_per_million_usd / 1_000_000
            taf_fee = min(
                shares * self.config.finra_taf_per_share_usd,
                self.config.finra_taf_cap_usd,
            )
            regulatory_fees = sec_fee + taf_fee

        return TransactionCostEstimate(
            side=side,
            notional_usd=notional_usd,
            shares=shares,
            spread_cost_usd=spread_cost,
            slippage_cost_usd=slippage_cost,
            regulatory_fees_usd=regulatory_fees,
            broker_fees_usd=self.config.broker_fee_usd,
        )
