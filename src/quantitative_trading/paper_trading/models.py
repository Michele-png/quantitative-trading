"""Shared data models for the weekly paper-trading workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class CandidateSnapshot:
    """One stock candidate shown to the zero-shot LLM decision maker."""

    symbol: str
    name: str
    price: float
    previous_close: float | None
    day_return: float | None
    five_day_return: float | None
    dollar_volume: float
    news_headlines: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serialisable representation for prompting."""
        return asdict(self)


@dataclass(frozen=True)
class MarketContext:
    """Market snapshot used to make one weekly trading decision."""

    trade_week: date
    generated_at: datetime
    universe_size: int
    candidates: list[CandidateSnapshot]
    market_summary: dict[str, Any]
    data_notes: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the context in a stable structure for the zero-shot prompt."""
        return {
            "trade_week": self.trade_week.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "universe_size": self.universe_size,
            "market_summary": self.market_summary,
            "data_notes": self.data_notes,
            "candidates": [candidate.to_prompt_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class WeeklyDecision:
    """Structured zero-shot LLM decision for one weekly trade."""

    trade_week: date
    ticker: str
    confidence: float
    thesis: str
    risks: list[str]
    market_summary: str
    raw_response: dict[str, Any]
    model: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class AccountSnapshot:
    """Subset of Alpaca account state needed by the strategy."""

    currency: str
    cash: float
    buying_power: float
    portfolio_value: float
    trading_blocked: bool


@dataclass(frozen=True)
class OrderPlan:
    """Validated order plan before optional paper execution."""

    trade_week: date
    symbol: str
    notional_usd: float
    side: Literal["buy"]
    time_in_force: Literal["day"]


@dataclass(frozen=True)
class OrderResult:
    """Minimal order metadata returned by Alpaca after submission."""

    order_id: str
    symbol: str
    side: str
    status: str
    submitted_at: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class TransactionCostEstimate:
    """Estimated costs that Alpaca paper fills do not realistically model."""

    side: Literal["buy", "sell"]
    notional_usd: float
    shares: float
    spread_cost_usd: float
    slippage_cost_usd: float
    regulatory_fees_usd: float
    broker_fees_usd: float

    @property
    def total_usd(self) -> float:
        """Total estimated transaction cost in USD."""
        return (
            self.spread_cost_usd
            + self.slippage_cost_usd
            + self.regulatory_fees_usd
            + self.broker_fees_usd
        )


@dataclass(frozen=True)
class TaxLotResult:
    """Italian-style realized gain tax calculation for one closed weekly lot."""

    proceeds_eur: float
    cost_basis_eur: float
    costs_eur: float
    realized_gain_eur: float
    taxable_gain_eur: float
    tax_due_eur: float
    loss_carryforward_eur: float


@dataclass(frozen=True)
class WeeklyRunResult:
    """Top-level result emitted by a dry-run or execute run."""

    trade_week: date
    dry_run: bool
    decision: WeeklyDecision
    order_plan: OrderPlan
    account: AccountSnapshot
    fx_rate_eur_usd: float
    estimated_buy_costs: TransactionCostEstimate
    order_result: OrderResult | None
    ledger_path: str
