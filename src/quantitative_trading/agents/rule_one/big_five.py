"""Big Five Numbers analyzer (Phil Town's quantitative core).

Per Phil Town's Rule One framework, a buy candidate must have:
    1. ROIC ≥ 10% per year for the last 10 years (and "holding steady or rising").
       ROIC := Net Income / (Equity + Long-term Debt). NB: this is Town's
       definition, which is closer to ROCE than the textbook ROIC. We use his
       formulation for fidelity to the framework.
    2. Sales (Revenue) growth ≥ 10% per year (10-year CAGR).
    3. EPS growth ≥ 10% per year (10-year CAGR). EPS is normalized for stock
       splits using the price client's split history (SEC reports per-share
       values in the share basis as of the filing date).
    4. Equity (Book Value) growth ≥ 10% per year (10-year CAGR).
    5. Operating Cash Flow growth ≥ 10% per year (10-year CAGR).

A side-check, not part of the Big 5 but mentioned in the same chapter:
    * Current Ratio (current_assets / current_liabilities) ≥ 2.0.

This module is pure analysis: it consumes the PIT data layer and emits a
structured result. No external I/O beyond what the EDGAR/Price clients do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import FactValue, PointInTimeFacts
from quantitative_trading.data.prices import PriceClient


# Phil Town's universal Big 5 threshold.
DEFAULT_THRESHOLD = 0.10
DEFAULT_N_YEARS = 10
LIQUIDITY_THRESHOLD = 2.0


@dataclass(frozen=True)
class MetricResult:
    """One Big-5 (or side-check) metric with its computed value and provenance."""

    name: str
    value: float | None  # decimal fraction, e.g. 0.15 for 15%, or None if not computable
    threshold: float
    passes: bool
    rationale: str
    series: dict[int, float | None] = field(default_factory=dict)
    """Per-fiscal-year values used to compute this metric."""


@dataclass(frozen=True)
class BigFiveResult:
    """Aggregate result of the Big 5 analysis at a point in time."""

    ticker: str
    as_of: date
    latest_fiscal_year: int | None
    n_years_required: int

    roic: MetricResult
    sales_growth: MetricResult
    eps_growth: MetricResult
    equity_growth: MetricResult
    ocf_growth: MetricResult

    current_ratio: MetricResult  # side check, not part of Big 5

    @property
    def all_pass(self) -> bool:
        return all(
            m.passes
            for m in (
                self.roic,
                self.sales_growth,
                self.eps_growth,
                self.equity_growth,
                self.ocf_growth,
            )
        )

    def summary(self) -> str:
        lines = [f"BigFive({self.ticker} as of {self.as_of}, latest FY={self.latest_fiscal_year}):"]
        for m in (
            self.roic,
            self.sales_growth,
            self.eps_growth,
            self.equity_growth,
            self.ocf_growth,
            self.current_ratio,
        ):
            v = "n/a" if m.value is None else f"{m.value * 100:6.2f}%"
            tag = "OK" if m.passes else "FAIL"
            lines.append(f"  {m.name:>20s}: {v}  threshold={m.threshold * 100:.0f}%  {tag}")
        lines.append(f"  ALL FIVE PASS: {self.all_pass}")
        return "\n".join(lines)


def _cagr(first: float, last: float, n_periods: int) -> float | None:
    """Annualized growth from `first` to `last` over `n_periods` years."""
    if n_periods <= 0:
        return None
    if first <= 0 or last <= 0:
        # Negative or zero starting equity / earnings makes CAGR ill-defined.
        # For backtest purposes this fails the check (which is the right call:
        # a company with negative book value 10 years ago is not Rule One-able).
        return None
    return (last / first) ** (1.0 / n_periods) - 1.0


def _series_to_dict(series: dict[int, FactValue | None]) -> dict[int, float | None]:
    return {fy: (fv.val if fv is not None else None) for fy, fv in series.items()}


def _adjust_eps_for_splits(
    eps_series: dict[int, FactValue | None],
    ticker: str,
    price_client: PriceClient,
) -> dict[int, float | None]:
    """Normalize each EPS value to current-share-basis using yfinance split history.

    SEC reports EPS in the share basis as of the filing date; over a 10-year
    window the basis can shift across multiple stock splits. To compute a
    meaningful EPS CAGR we put every value into a single basis (today's).
    """
    out: dict[int, float | None] = {}
    for fy, fv in eps_series.items():
        if fv is None:
            out[fy] = None
            continue
        factor = price_client.split_factor_since(ticker, fv.filed)
        out[fy] = fv.val / factor if factor else fv.val
    return out


class BigFiveAnalyzer:
    """Compute the Big 5 numbers + liquidity at a point in time."""

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
        *,
        n_years: int = DEFAULT_N_YEARS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> BigFiveResult:
        cik = self._edgar.get_cik(ticker)
        facts = self._edgar.get_company_facts(cik)
        pit = PointInTimeFacts(facts)

        latest_fy = pit.latest_fiscal_year_with_data("revenue", as_of)
        if latest_fy is None:
            return self._all_fail(ticker, as_of, latest_fy=None,
                                  n_years=n_years, threshold=threshold,
                                  reason="No fiscal year revenue data available before as_of.")

        # Pull all Big 5 series.
        revenue = pit.get_annual_series("revenue", latest_fy, n_years, as_of)
        net_income = pit.get_annual_series("net_income", latest_fy, n_years, as_of)
        eps = pit.get_annual_series("eps_diluted", latest_fy, n_years, as_of)
        equity = pit.get_annual_series("stockholders_equity", latest_fy, n_years, as_of)
        ocf = pit.get_annual_series("operating_cash_flow", latest_fy, n_years, as_of)
        ltdebt = pit.get_annual_series("long_term_debt", latest_fy, n_years, as_of)

        roic = self._roic(net_income, equity, ltdebt, threshold)
        sales = self._growth("Sales Growth", revenue, n_years, threshold)
        equity_g = self._growth("Equity Growth", equity, n_years, threshold)
        ocf_g = self._growth("OCF Growth", ocf, n_years, threshold)

        eps_adj = _adjust_eps_for_splits(eps, ticker, self._prices)
        eps_g = self._growth_from_dict("EPS Growth", eps_adj, n_years, threshold)

        liquidity = self._current_ratio(pit, latest_fy, as_of)

        return BigFiveResult(
            ticker=ticker.upper(),
            as_of=as_of,
            latest_fiscal_year=latest_fy,
            n_years_required=n_years,
            roic=roic,
            sales_growth=sales,
            eps_growth=eps_g,
            equity_growth=equity_g,
            ocf_growth=ocf_g,
            current_ratio=liquidity,
        )

    # ------------------------------------------------------------------ ROIC

    def _roic(
        self,
        net_income: dict[int, FactValue | None],
        equity: dict[int, FactValue | None],
        ltdebt: dict[int, FactValue | None],
        threshold: float,
    ) -> MetricResult:
        per_year: dict[int, float | None] = {}
        for fy in net_income:
            ni = net_income[fy]
            eq = equity[fy]
            dt = ltdebt.get(fy)
            if ni is None or eq is None or eq.val <= 0:
                per_year[fy] = None
                continue
            denom = eq.val + (dt.val if dt is not None else 0.0)
            if denom <= 0:
                per_year[fy] = None
                continue
            per_year[fy] = ni.val / denom

        valid = [v for v in per_year.values() if v is not None]
        if not valid:
            return MetricResult(
                name="ROIC", value=None, threshold=threshold, passes=False,
                rationale="No years with valid ROIC data.", series=per_year,
            )

        avg = sum(valid) / len(valid)
        most_recent_fy = max(per_year)
        most_recent = per_year[most_recent_fy]

        # Phil Town: ROIC ≥ 10% per year, "holding steady or rising".
        # Operationalization: average ≥ threshold AND most-recent ≥ threshold.
        passes = (
            len(valid) >= len(per_year) // 2  # at least half the years have data
            and avg >= threshold
            and most_recent is not None
            and most_recent >= threshold
        )
        rat = (
            f"avg ROIC over {len(valid)} years = {avg * 100:.2f}%; "
            f"most recent FY{most_recent_fy} = "
            f"{most_recent * 100:.2f}%" if most_recent is not None else "n/a"
        )
        return MetricResult(
            name="ROIC", value=avg, threshold=threshold, passes=passes,
            rationale=rat, series=per_year,
        )

    # ---------------------------------------------------------- Growth rates

    def _growth(
        self,
        name: str,
        series: dict[int, FactValue | None],
        n_years: int,
        threshold: float,
    ) -> MetricResult:
        vals = _series_to_dict(series)
        return self._growth_from_dict(name, vals, n_years, threshold)

    def _growth_from_dict(
        self,
        name: str,
        series: dict[int, float | None],
        n_years: int,
        threshold: float,
        min_years: int = 5,
    ) -> MetricResult:
        """CAGR over the longest available contiguous-ish window in `series`.

        Phil Town wants 10 years; SEC XBRL only became mandatory in 2009-2011,
        so for backtests in the early 2010s a strict 10-year requirement
        excludes nearly everything. We instead require at least `min_years`
        (default 5 — the shortest window over which CAGR is meaningful) and
        compute CAGR from the earliest available year to the latest available
        year. The relaxation is surfaced in the rationale string so callers
        can audit which checks rested on partial history.
        """
        if not series:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale="No fiscal years available.", series={},
            )

        present_pairs = [(fy, v) for fy, v in sorted(series.items()) if v is not None]
        if len(present_pairs) < min_years:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale=(
                    f"Only {len(present_pairs)} years with data; need ≥ {min_years}."
                ),
                series=series,
            )

        first_fy, first_val = present_pairs[0]
        last_fy, last_val = present_pairs[-1]
        n_periods = last_fy - first_fy
        if n_periods <= 0:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale=f"Need >1 year of data (got first=last=FY{first_fy}).",
                series=series,
            )

        cagr = _cagr(first_val, last_val, n_periods)
        if cagr is None or math.isnan(cagr) or math.isinf(cagr):
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale=(
                    f"CAGR ill-defined for FY{first_fy}={first_val:.2f} -> "
                    f"FY{last_fy}={last_val:.2f}"
                ),
                series=series,
            )

        n_present = len(present_pairs)
        passes = cagr >= threshold
        relaxed_note = "" if n_present >= n_years else (
            f" (NOTE: only {n_present}y data, want {n_years}y)"
        )
        rat = (
            f"{n_periods}y CAGR FY{first_fy}->{last_fy}: "
            f"{first_val:,.2f} -> {last_val:,.2f} = {cagr * 100:.2f}%"
            f"{relaxed_note}"
        )
        return MetricResult(
            name=name, value=cagr, threshold=threshold, passes=passes,
            rationale=rat, series=series,
        )

    # ------------------------------------------------------------ Liquidity

    def _current_ratio(
        self,
        pit: PointInTimeFacts,
        fiscal_year: int,
        as_of: date,
    ) -> MetricResult:
        ca = pit.get_annual("current_assets", fiscal_year, as_of)
        cl = pit.get_annual("current_liabilities", fiscal_year, as_of)
        if ca is None or cl is None or cl.val <= 0:
            return MetricResult(
                name="Current Ratio", value=None, threshold=LIQUIDITY_THRESHOLD,
                passes=False,
                rationale="Current assets / liabilities not reported for latest FY.",
            )
        ratio = ca.val / cl.val
        passes = ratio >= LIQUIDITY_THRESHOLD
        rat = (
            f"FY{fiscal_year}: current assets ${ca.val / 1e6:,.0f}M / "
            f"current liabilities ${cl.val / 1e6:,.0f}M = {ratio:.2f}x"
        )
        return MetricResult(
            name="Current Ratio", value=ratio, threshold=LIQUIDITY_THRESHOLD,
            passes=passes, rationale=rat,
        )

    # ----------------------------------------------------------------- Misc

    def _all_fail(
        self,
        ticker: str,
        as_of: date,
        *,
        latest_fy: int | None,
        n_years: int,
        threshold: float,
        reason: str,
    ) -> BigFiveResult:
        empty = MetricResult("", None, threshold, False, reason)
        return BigFiveResult(
            ticker=ticker.upper(),
            as_of=as_of,
            latest_fiscal_year=latest_fy,
            n_years_required=n_years,
            roic=MetricResult("ROIC", None, threshold, False, reason),
            sales_growth=MetricResult("Sales Growth", None, threshold, False, reason),
            eps_growth=MetricResult("EPS Growth", None, threshold, False, reason),
            equity_growth=MetricResult("Equity Growth", None, threshold, False, reason),
            ocf_growth=MetricResult("OCF Growth", None, threshold, False, reason),
            current_ratio=MetricResult(
                "Current Ratio", None, LIQUIDITY_THRESHOLD, False, reason
            ),
        )
