"""Forward-return labels for the historical dataset.

Given a trade date `T`, compute the annualized total return over the next
`label_horizon_years` (default 5). A buy is labeled "passes" iff the forward
CAGR is >= `target_cagr` (default 0.15 — Phil Town's stated 15%/yr goal).

PIT discipline:
    The label uses prices in (T, T + label_horizon_years]. The agent's view
    only uses prices <= T. The two are conceptually separate — the agent never
    sees the label inputs.

Delisting handling:
    See `prices.PriceClient.forward_total_return_cagr` for the exact rule.
    The label is computed from the last available Adj Close, annualized over
    the nominal horizon, and `delisted_before_horizon` is recorded so callers
    can filter or weight as they prefer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from quantitative_trading.data.prices import PriceClient


DEFAULT_LABEL_HORIZON_YEARS = 5
DEFAULT_TARGET_CAGR = 0.15


@dataclass(frozen=True)
class LabelResult:
    ticker: str
    trade_date: date
    label_end_date: date
    label_horizon_years: int
    target_cagr: float

    forward_cagr: float | None
    label_passes: bool
    delisted_before_horizon: bool
    error: str | None = None


def compute_label(
    ticker: str,
    trade_date: date,
    price_client: PriceClient,
    *,
    label_horizon_years: int = DEFAULT_LABEL_HORIZON_YEARS,
    target_cagr: float = DEFAULT_TARGET_CAGR,
) -> LabelResult:
    """Compute the forward label for a single (ticker, trade_date)."""
    end_date = trade_date + timedelta(days=int(365.25 * label_horizon_years))

    cagr = price_client.forward_total_return_cagr(ticker, trade_date, end_date)
    if cagr is None:
        return LabelResult(
            ticker=ticker.upper(), trade_date=trade_date, label_end_date=end_date,
            label_horizon_years=label_horizon_years, target_cagr=target_cagr,
            forward_cagr=None, label_passes=False,
            delisted_before_horizon=False,
            error="No price data available for label computation",
        )

    delisted = price_client.was_delisted_before(ticker, end_date)
    return LabelResult(
        ticker=ticker.upper(), trade_date=trade_date, label_end_date=end_date,
        label_horizon_years=label_horizon_years, target_cagr=target_cagr,
        forward_cagr=cagr,
        label_passes=cagr >= target_cagr,
        delisted_before_horizon=delisted,
        error=None,
    )
