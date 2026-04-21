"""Sticker Price, Margin of Safety, and Payback Time (Phil Town's valuation core).

Phil Town's Sticker Price formula:

    Future EPS    = current_EPS × (1 + FGR)^10
    Future Price  = Future EPS × Future PE
    Sticker Price = Future Price / (1 + required_return)^10
                  = current_EPS × Future PE × ((1 + FGR) / (1 + r))^10

Where:
    * current_EPS = most recent annual diluted EPS (normalized for stock splits
      to today's share basis, so it can be compared to yfinance prices).
    * FGR (Future Growth Rate) = min(historical 10y EPS CAGR, analyst estimate,
      0.15). Without analyst estimates we use min(historical, 0.15). The 15%
      cap is Phil Town's recommendation: don't extrapolate growth above what's
      realistically sustainable for 10 years.
    * Future PE = min(historical_average_PE, 2 × FGR_pct). Without historical
      PE data we use 2 × FGR_pct. With FGR capped at 15%, this gives a max PE
      of 30 — a reasonable ceiling.
    * required_return = 0.15 (Phil Town's target: 15% annual return).

    Margin of Safety price = Sticker Price / 2

    BUY iff current_market_price <= Margin_of_Safety_price.

Payback Time:

    Find the smallest N such that
        Σ (current_EPS × (1 + FGR)^k) for k=1..N  >=  current_market_price
    Phil Town wants N <= 8 years for a good investment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import PointInTimeFacts
from quantitative_trading.data.prices import PriceClient


# Phil Town's defaults.
DEFAULT_REQUIRED_RETURN = 0.15
DEFAULT_FGR_CAP = 0.15
DEFAULT_FUTURE_PE_CAP = 30.0
DEFAULT_HORIZON_YEARS = 10
DEFAULT_PAYBACK_THRESHOLD = 8.0


@dataclass(frozen=True)
class StickerPriceResult:
    """Result of the Sticker Price + Margin of Safety analysis."""

    ticker: str
    as_of: date
    fiscal_year_used: int | None

    eps_today_basis: float | None  # diluted EPS, normalized to current share basis
    historical_growth_rate: float | None  # observed EPS CAGR (uncapped)
    future_growth_rate: float | None  # capped FGR used in the formula
    future_pe: float | None
    future_eps: float | None
    sticker_price: float | None
    margin_of_safety_price: float | None

    current_price: float | None  # yfinance Close at as_of (today's basis)
    margin_of_safety_passes: bool

    rationale: str


@dataclass(frozen=True)
class PaybackTimeResult:
    """Result of the Payback Time analysis."""

    ticker: str
    as_of: date
    payback_years: int | None
    threshold_years: float
    passes: bool
    rationale: str


def compute_sticker_price(
    *,
    eps_today_basis: float,
    historical_growth_rate: float,
    required_return: float = DEFAULT_REQUIRED_RETURN,
    fgr_cap: float = DEFAULT_FGR_CAP,
    future_pe_cap: float = DEFAULT_FUTURE_PE_CAP,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    historical_avg_pe: float | None = None,
) -> tuple[float, float, float, float]:
    """Pure function: compute (FGR, Future PE, Future EPS, Sticker) given inputs.

    Mirrors Phil Town's exact formula. No data fetching here — just the math.
    """
    # Cap historical growth at the FGR cap (Phil Town: don't extrapolate above ~15%).
    fgr = max(0.0, min(historical_growth_rate, fgr_cap))

    # Future PE = 2 × FGR%, capped by future_pe_cap and (if available) historical avg PE.
    pe_from_growth = 2.0 * fgr * 100.0
    pe_candidates = [pe_from_growth, future_pe_cap]
    if historical_avg_pe is not None and historical_avg_pe > 0:
        pe_candidates.append(historical_avg_pe)
    future_pe = min(pe_candidates)

    future_eps = eps_today_basis * (1.0 + fgr) ** horizon_years
    future_price = future_eps * future_pe
    sticker = future_price / (1.0 + required_return) ** horizon_years
    return fgr, future_pe, future_eps, sticker


def compute_payback_years(
    *,
    eps_today_basis: float,
    growth_rate: float,
    current_price: float,
    max_years: int = 50,
) -> int | None:
    """Smallest N such that cumulative EPS over years 1..N >= current_price."""
    if eps_today_basis <= 0 or current_price <= 0:
        return None
    cumulative = 0.0
    for year in range(1, max_years + 1):
        cumulative += eps_today_basis * (1.0 + growth_rate) ** year
        if cumulative >= current_price:
            return year
    return None


class StickerPriceCalculator:
    """Sticker Price, Margin of Safety, Payback Time at a point in time."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        price_client: PriceClient,
    ) -> None:
        self._edgar = edgar_client
        self._prices = price_client

    def evaluate(
        self,
        ticker: str,
        as_of: date,
        historical_eps_growth: float | None,
        *,
        required_return: float = DEFAULT_REQUIRED_RETURN,
        fgr_cap: float = DEFAULT_FGR_CAP,
        future_pe_cap: float = DEFAULT_FUTURE_PE_CAP,
        horizon_years: int = DEFAULT_HORIZON_YEARS,
        payback_threshold: float = DEFAULT_PAYBACK_THRESHOLD,
    ) -> tuple[StickerPriceResult, PaybackTimeResult]:
        """Compute Sticker Price + MoS + Payback Time at `as_of`.

        `historical_eps_growth` is supplied by the caller (typically the Big 5
        EPS growth metric). If None, the analysis cannot proceed and an
        all-fail result is returned.
        """
        cik = self._edgar.get_cik(ticker)
        facts = self._edgar.get_company_facts(cik)
        pit = PointInTimeFacts(facts)

        latest_fy = pit.latest_fiscal_year_with_data("eps_diluted", as_of)
        if latest_fy is None:
            return self._all_fail(
                ticker, as_of, "No EPS data available before as_of."
            )

        eps_fv = pit.get_annual("eps_diluted", latest_fy, as_of)
        if eps_fv is None:
            return self._all_fail(
                ticker, as_of, f"EPS missing for FY{latest_fy}."
            )

        # Normalize EPS to today's share basis (yfinance prices are post-all-splits).
        split_factor = self._prices.split_factor_since(ticker, eps_fv.filed)
        eps_today = eps_fv.val / split_factor if split_factor else eps_fv.val

        if historical_eps_growth is None:
            return self._all_fail(
                ticker, as_of,
                "No historical EPS growth rate provided (Big 5 EPS Growth must be computable).",
                fiscal_year_used=latest_fy, eps_today_basis=eps_today,
            )

        current_price = self._prices.get_close_at(ticker, as_of)
        if current_price is None:
            return self._all_fail(
                ticker, as_of, "No price data at or near as_of.",
                fiscal_year_used=latest_fy, eps_today_basis=eps_today,
            )

        fgr, future_pe, future_eps, sticker = compute_sticker_price(
            eps_today_basis=eps_today,
            historical_growth_rate=historical_eps_growth,
            required_return=required_return,
            fgr_cap=fgr_cap,
            future_pe_cap=future_pe_cap,
            horizon_years=horizon_years,
        )
        mos_price = sticker / 2.0
        mos_passes = current_price <= mos_price

        sticker_result = StickerPriceResult(
            ticker=ticker.upper(),
            as_of=as_of,
            fiscal_year_used=latest_fy,
            eps_today_basis=eps_today,
            historical_growth_rate=historical_eps_growth,
            future_growth_rate=fgr,
            future_pe=future_pe,
            future_eps=future_eps,
            sticker_price=sticker,
            margin_of_safety_price=mos_price,
            current_price=current_price,
            margin_of_safety_passes=mos_passes,
            rationale=(
                f"FY{latest_fy} EPS=${eps_today:.2f} (today basis); "
                f"hist growth={historical_eps_growth * 100:.1f}%, "
                f"capped FGR={fgr * 100:.1f}%, future PE={future_pe:.1f}; "
                f"Sticker=${sticker:.2f}, MoS=${mos_price:.2f}; "
                f"price=${current_price:.2f} → {'BUY' if mos_passes else 'no'}"
            ),
        )

        payback_years = compute_payback_years(
            eps_today_basis=eps_today,
            growth_rate=fgr,
            current_price=current_price,
        )
        payback_passes = payback_years is not None and payback_years <= payback_threshold
        payback_result = PaybackTimeResult(
            ticker=ticker.upper(),
            as_of=as_of,
            payback_years=payback_years,
            threshold_years=payback_threshold,
            passes=payback_passes,
            rationale=(
                f"At EPS=${eps_today:.2f} growing {fgr * 100:.1f}%/yr, "
                f"cumulative EPS recoups ${current_price:.2f} in "
                f"{payback_years if payback_years is not None else '>50'} years "
                f"(threshold ≤ {payback_threshold:.0f})"
            ),
        )

        return sticker_result, payback_result

    def _all_fail(
        self,
        ticker: str,
        as_of: date,
        reason: str,
        *,
        fiscal_year_used: int | None = None,
        eps_today_basis: float | None = None,
    ) -> tuple[StickerPriceResult, PaybackTimeResult]:
        sticker = StickerPriceResult(
            ticker=ticker.upper(), as_of=as_of, fiscal_year_used=fiscal_year_used,
            eps_today_basis=eps_today_basis,
            historical_growth_rate=None, future_growth_rate=None, future_pe=None,
            future_eps=None, sticker_price=None, margin_of_safety_price=None,
            current_price=None, margin_of_safety_passes=False, rationale=reason,
        )
        payback = PaybackTimeResult(
            ticker=ticker.upper(), as_of=as_of, payback_years=None,
            threshold_years=DEFAULT_PAYBACK_THRESHOLD,
            passes=False, rationale=reason,
        )
        return sticker, payback
