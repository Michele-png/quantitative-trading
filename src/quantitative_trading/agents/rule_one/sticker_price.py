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

Implementation status (matches the documented method)
-----------------------------------------------------
* **Historical EPS CAGR** — supplied by the caller from
  ``BigFiveAnalyzer`` (which now walks SEC → SEC NI/shares → yfinance).
* **Historical average PE** — computed in ``StickerPriceCalculator`` from
  the price client's adjusted Close at each FY end and the same
  split-adjusted EPS series the Big 5 uses. Falls back to ``None`` (in
  which case the FGR-implied cap binds) if fewer than ``min_pe_years``
  fiscal years can be priced.
* **Analyst growth estimate** — pluggable provider. The default
  ``NullAnalystProvider`` returns ``None`` so behaviour is unchanged
  unless the caller wires in a paid data source. Whether or not an
  estimate was used is reported on ``StickerPriceResult.inputs_used``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import PointInTimeFacts
from quantitative_trading.data.prices import PriceClient


log = logging.getLogger(__name__)


# Phil Town's defaults.
DEFAULT_REQUIRED_RETURN = 0.15
DEFAULT_FGR_CAP = 0.15
DEFAULT_FUTURE_PE_CAP = 30.0
DEFAULT_HORIZON_YEARS = 10
DEFAULT_PAYBACK_THRESHOLD = 8.0
DEFAULT_PE_LOOKBACK_YEARS = 5
DEFAULT_PE_MIN_YEARS = 3
# Sensitivity sweep: ±2.5pp and ±5pp around the base growth rate. The
# dashboard renders this as "what if EPS grows at 5% / 7.5% / 10% /
# 12.5% / 15%?" so the user can see how robust the sticker price is.
DEFAULT_FGR_SENSITIVITY_DELTAS = (-0.05, -0.025, 0.0, 0.025, 0.05)
DEFAULT_RR_SENSITIVITY_DELTAS = (-0.05, -0.025, 0.0, 0.025, 0.05)


# Provenance labels for the inputs that drive the sticker computation.
INPUT_USED = "used"
INPUT_NOT_AVAILABLE = "not_available"
INPUT_DISABLED = "disabled"


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

    # Optional inputs to the documented method. ``None`` means the input
    # was not available; the FGR-from-growth ceiling and the historical
    # CAGR end up driving the result alone in that case.
    historical_avg_pe: float | None = None
    """Mean of FY-end price / EPS over up to ``DEFAULT_PE_LOOKBACK_YEARS``
    fiscal years. Used as one of the candidates in the future-PE cap."""

    analyst_growth_estimate: float | None = None
    """Forward EPS growth estimate from an external provider. Defaults
    to ``None`` (no provider). When present, the FGR is capped by
    ``min(historical, analyst, fgr_cap)``."""

    inputs_used: dict[str, str] = field(default_factory=dict)
    """One entry per input: ``{"historical_growth": "used",
    "historical_avg_pe": "not_available", "analyst_growth": "disabled"}``.
    Lets the dashboard say "analyst estimate not used" instead of
    silently dropping the input."""

    pe_history: list[dict[str, Any]] = field(default_factory=list)
    """Per-fiscal-year PE records used to compute the historical
    average. Each record: ``{"fiscal_year", "fy_end", "price",
    "eps_today_basis", "pe"}``. Empty when no PE history could be
    assembled (typically because EPS or price data is missing)."""

    sensitivity: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Per-input sensitivity sweep — see
    ``compute_sticker_sensitivity`` for the entry shape. Lets the
    dashboard show "sticker @ 5% growth = $X / @ 15% growth = $Y" so
    a user can see how brittle the headline number is."""


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


class AnalystEstimateProvider(Protocol):
    """Strategy for fetching forward EPS-growth estimates.

    The default ``NullAnalystProvider`` returns ``None`` for every
    ticker, which keeps behaviour identical to the historical-only path.
    A future paid integration (FactSet, Refinitiv, ZACKS, FMP Premium)
    would implement ``get_eps_growth_5y`` to return the consensus 5-year
    growth rate as a decimal fraction (e.g. ``0.12`` for 12%/yr).

    ``credentials_available`` lets the caller distinguish "no provider
    wired up" from "wired up but no estimate for this ticker".
    """

    def credentials_available(self) -> bool:
        ...

    def get_eps_growth_5y(self, ticker: str, as_of: date) -> float | None:
        ...


class NullAnalystProvider:
    """No-op provider — returns ``None`` for every ticker.

    This is the default so the screening pipeline never reaches out to
    a paid API unless the caller explicitly wires one in. The Phase 3
    plan calls out a hard requirement to "verify credentials before
    using any external analyst-estimate API"; the simplest way to
    enforce that is to have the no-op provider be the default.
    """

    def credentials_available(self) -> bool:
        return False

    def get_eps_growth_5y(self, ticker: str, as_of: date) -> float | None:  # noqa: ARG002
        return None


def compute_historical_avg_pe(
    eps_today_basis_series: dict[int, float | None],
    fiscal_year_ends: dict[int, date | None],
    ticker: str,
    price_client: PriceClient,
    *,
    n_years: int = DEFAULT_PE_LOOKBACK_YEARS,
    min_years: int = DEFAULT_PE_MIN_YEARS,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Average historical P/E over the most-recent ``n_years`` fiscal years.

    For each FY ``y`` we read the closing price at ``fy_end_y`` and
    compute ``PE_y = (close_y / split_factor_since(close_y)) /
    eps_today_basis_y``. Both numerator and denominator are in today's
    share basis, so the ratio is comparable across multi-decade splits.

    Returns ``(avg_pe, per_year_records)``. ``avg_pe`` is ``None`` when
    fewer than ``min_years`` populated, positive PE values are
    available — the dashboard treats that as "historical PE not used"
    rather than silently dropping the input.
    """
    fys_sorted = sorted(eps_today_basis_series.keys(), reverse=True)[:n_years]
    pe_records: list[dict[str, Any]] = []
    pe_vals: list[float] = []
    for fy in sorted(fys_sorted):
        eps = eps_today_basis_series.get(fy)
        end = fiscal_year_ends.get(fy)
        if eps is None or eps <= 0 or end is None:
            continue
        price = price_client.get_close_at(ticker, end)
        if price is None or price <= 0:
            continue
        split_factor = price_client.split_factor_since(ticker, end) or 1.0
        price_today_basis = price / split_factor
        pe = price_today_basis / eps
        if pe <= 0:
            continue
        pe_vals.append(pe)
        pe_records.append({
            "fiscal_year": fy,
            "fy_end": end.isoformat(),
            "close_price": price,
            "split_factor_since": split_factor,
            "price_today_basis": price_today_basis,
            "eps_today_basis": eps,
            "pe": pe,
        })
    if len(pe_vals) < min_years:
        return None, pe_records
    return sum(pe_vals) / len(pe_vals), pe_records


def compute_sticker_sensitivity(
    *,
    eps_today_basis: float,
    historical_growth_rate: float,
    historical_avg_pe: float | None,
    required_return: float,
    fgr_cap: float,
    future_pe_cap: float,
    horizon_years: int,
    current_price: float,
    fgr_deltas: tuple[float, ...] = DEFAULT_FGR_SENSITIVITY_DELTAS,
    rr_deltas: tuple[float, ...] = DEFAULT_RR_SENSITIVITY_DELTAS,
) -> dict[str, list[dict[str, Any]]]:
    """Sweep FGR and required-return ±5pp and report sticker / MoS / MoS%.

    Each entry of the returned lists has shape::

        {"input": float,         # the input value swept
         "sticker": float,       # implied sticker price
         "mos_price": float,     # sticker / 2
         "implied_mos_pct": float | None,  # (mos - cur) / cur
        }

    Useful for the dashboard to render small "sticker vs growth" tables
    next to the headline so the reader can see how robust the headline
    is to the FGR assumption.
    """
    out: dict[str, list[dict[str, Any]]] = {
        "future_growth_rate": [],
        "required_return": [],
    }

    def _row(*, input_value: float, sticker: float) -> dict[str, Any]:
        mos_price = sticker / 2.0
        if current_price > 0:
            implied = (mos_price - current_price) / current_price
        else:
            implied = None
        return {
            "input": input_value,
            "sticker": sticker,
            "mos_price": mos_price,
            "implied_mos_pct": implied,
        }

    for delta in fgr_deltas:
        fgr_test = max(0.0, min(historical_growth_rate + delta, fgr_cap))
        _, _, _, sticker = compute_sticker_price(
            eps_today_basis=eps_today_basis,
            historical_growth_rate=fgr_test,
            required_return=required_return,
            fgr_cap=fgr_cap,
            future_pe_cap=future_pe_cap,
            horizon_years=horizon_years,
            historical_avg_pe=historical_avg_pe,
        )
        out["future_growth_rate"].append(_row(input_value=fgr_test, sticker=sticker))

    for delta in rr_deltas:
        rr_test = required_return + delta
        if rr_test <= 0:
            continue
        _, _, _, sticker = compute_sticker_price(
            eps_today_basis=eps_today_basis,
            historical_growth_rate=historical_growth_rate,
            required_return=rr_test,
            fgr_cap=fgr_cap,
            future_pe_cap=future_pe_cap,
            horizon_years=horizon_years,
            historical_avg_pe=historical_avg_pe,
        )
        out["required_return"].append(_row(input_value=rr_test, sticker=sticker))

    return out


class StickerPriceCalculator:
    """Sticker Price, Margin of Safety, Payback Time at a point in time."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        price_client: PriceClient,
        *,
        analyst_provider: AnalystEstimateProvider | None = None,
    ) -> None:
        self._edgar = edgar_client
        self._prices = price_client
        self._analyst = analyst_provider or NullAnalystProvider()

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
        eps_history: dict[int, float | None] | None = None,
        fiscal_year_ends: dict[int, date | None] | None = None,
    ) -> tuple[StickerPriceResult, PaybackTimeResult]:
        """Compute Sticker Price + MoS + Payback Time at ``as_of``.

        Optional inputs:

        * ``historical_eps_growth`` — observed long-run EPS CAGR in
          today's share basis. Typically supplied by the Big 5 layer
          (which already walks SEC → SEC NI/shares → yfinance), so the
          two stay in sync.
        * ``eps_history`` / ``fiscal_year_ends`` — per-FY EPS values and
          period-end dates. When present, the calculator computes a
          historical average PE and uses it as one of the candidates for
          the future-PE cap. When missing, the cap falls back to the
          FGR-implied PE alone (the previous behaviour).

        Whether each documented input was actually used is reported on
        ``StickerPriceResult.inputs_used`` so the dashboard can show
        "analyst growth = not available" instead of silently dropping
        the input.
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

        # ---- Optional: historical average PE ----------------------------
        historical_avg_pe: float | None = None
        pe_history: list[dict[str, Any]] = []
        if eps_history and fiscal_year_ends:
            historical_avg_pe, pe_history = compute_historical_avg_pe(
                eps_today_basis_series=eps_history,
                fiscal_year_ends=fiscal_year_ends,
                ticker=ticker,
                price_client=self._prices,
            )

        # ---- Optional: analyst forward growth estimate ------------------
        analyst_growth: float | None = None
        if self._analyst.credentials_available():
            try:
                analyst_growth = self._analyst.get_eps_growth_5y(ticker, as_of)
            except Exception as exc:  # noqa: BLE001 - external provider
                log.warning(
                    "Analyst-estimate provider failed for %s: %s", ticker, exc,
                )
                analyst_growth = None

        # The documented FGR rule: cap by min(historical, analyst,
        # fgr_cap). Without an analyst estimate we fall back to
        # min(historical, fgr_cap) — matches the original code path.
        if analyst_growth is not None:
            growth_for_formula = min(historical_eps_growth, analyst_growth)
        else:
            growth_for_formula = historical_eps_growth

        fgr, future_pe, future_eps, sticker = compute_sticker_price(
            eps_today_basis=eps_today,
            historical_growth_rate=growth_for_formula,
            required_return=required_return,
            fgr_cap=fgr_cap,
            future_pe_cap=future_pe_cap,
            horizon_years=horizon_years,
            historical_avg_pe=historical_avg_pe,
        )
        mos_price = sticker / 2.0
        mos_passes = current_price <= mos_price

        inputs_used = {
            "historical_growth": INPUT_USED,
            "historical_avg_pe": (
                INPUT_USED if historical_avg_pe is not None else INPUT_NOT_AVAILABLE
            ),
            "analyst_growth": (
                INPUT_USED
                if analyst_growth is not None
                else (
                    INPUT_DISABLED
                    if not self._analyst.credentials_available()
                    else INPUT_NOT_AVAILABLE
                )
            ),
        }

        sensitivity = compute_sticker_sensitivity(
            eps_today_basis=eps_today,
            historical_growth_rate=growth_for_formula,
            historical_avg_pe=historical_avg_pe,
            required_return=required_return,
            fgr_cap=fgr_cap,
            future_pe_cap=future_pe_cap,
            horizon_years=horizon_years,
            current_price=current_price,
        )

        rationale_parts = [
            f"FY{latest_fy} EPS=${eps_today:.2f} (today basis)",
            f"hist growth={historical_eps_growth * 100:.1f}%",
        ]
        if analyst_growth is not None:
            rationale_parts.append(f"analyst growth={analyst_growth * 100:.1f}%")
        rationale_parts.append(f"capped FGR={fgr * 100:.1f}%")
        if historical_avg_pe is not None:
            rationale_parts.append(
                f"avg PE={historical_avg_pe:.1f} (used in cap)"
            )
        rationale_parts.append(f"future PE={future_pe:.1f}")
        rationale_parts.append(
            f"Sticker=${sticker:.2f}, MoS=${mos_price:.2f}; "
            f"price=${current_price:.2f} → {'BUY' if mos_passes else 'no'}"
        )
        rationale = "; ".join(rationale_parts)

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
            rationale=rationale,
            historical_avg_pe=historical_avg_pe,
            analyst_growth_estimate=analyst_growth,
            inputs_used=inputs_used,
            pe_history=pe_history,
            sensitivity=sensitivity,
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
            historical_avg_pe=None, analyst_growth_estimate=None,
            inputs_used={
                "historical_growth": INPUT_NOT_AVAILABLE,
                "historical_avg_pe": INPUT_NOT_AVAILABLE,
                "analyst_growth": INPUT_NOT_AVAILABLE,
            },
            pe_history=[],
            sensitivity={},
        )
        payback = PaybackTimeResult(
            ticker=ticker.upper(), as_of=as_of, payback_years=None,
            threshold_years=DEFAULT_PAYBACK_THRESHOLD,
            passes=False, rationale=reason,
        )
        return sticker, payback
