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

Decision grade
--------------
``MetricResult.passes`` is now strict: a check is only considered to pass
when the full ``n_years`` window has been observed. With less than the
target window we still surface the underlying CAGR / ratio in
``MetricResult.value`` (so the dashboard can show "would have been X%
over the available history") but ``passes`` stays False — Phil Town's
rule explicitly wants 10 years of evidence and we no longer let a short
record claim a clean pass.

EPS extraction fallbacks
------------------------
SEC ``companyfacts`` is the canonical EPS source, but several large
issuers (Visa is the canonical example) simply do not tag
``EarningsPerShareDiluted`` in machine-readable XBRL. The analyzer now
walks an explicit fallback chain:

    1. SEC XBRL EPS concepts (``EarningsPerShareDiluted``, etc.).
    2. Derived from ``NetIncomeLoss / WeightedAverageNumberOfDilutedSharesOutstanding``
       — also from SEC XBRL but scoped to two more reliable concepts.
    3. yfinance income statement (``Diluted EPS`` row) for issuers
       whose XBRL is incomplete.

The chosen source is recorded on ``MetricResult.data_source`` so the
dashboard evidence can label "EPS computed from yfinance" or "EPS
unavailable from any source" instead of silently scoring NO DATA.

This module is pure analysis: it consumes the PIT data layer and emits a
structured result. No external I/O beyond what the EDGAR/Price clients do.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import FactValue, PointInTimeFacts
from quantitative_trading.data.prices import PriceClient


log = logging.getLogger(__name__)


# Phil Town's universal Big 5 threshold.
DEFAULT_THRESHOLD = 0.10
DEFAULT_N_YEARS = 10
LIQUIDITY_THRESHOLD = 2.0
# Below this many populated years we don't even compute a CAGR — the
# value would be too noisy to display. Between this floor and ``n_years``
# we compute the CAGR (relaxed window) but ``passes`` stays False.
MIN_YEARS_FOR_VALUE = 5


# Data-source provenance labels used on ``MetricResult.data_source``.
# Free-form strings keep the schema flexible; the dashboard maps them to
# friendly labels in the evidence body.
SOURCE_SEC_XBRL = "sec_xbrl"
SOURCE_SEC_NI_OVER_SHARES = "sec_ni_over_shares"
SOURCE_YFINANCE = "yfinance"
SOURCE_UNAVAILABLE = "unavailable"


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
    decision_grade: bool = False
    """True iff the underlying data window is fully populated AND the
    pass criterion was met. When False the metric should be treated as
    informational only — a relaxed-window CAGR or a value computed from
    a fallback source. The dashboard uses this to surface PARTIAL DATA
    badges instead of FAIL when the gap is data, not signal."""
    data_source: str = SOURCE_SEC_XBRL
    """Provenance for the underlying series. Defaults to SEC XBRL —
    overridden when EPS or any other concept is filled from the
    fallback chain (NI/shares derivation, yfinance, etc.)."""


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


def _eps_from_ni_and_shares(
    ni_series: dict[int, FactValue | None],
    sh_series: dict[int, FactValue | None],
    ticker: str,
    price_client: PriceClient,
) -> dict[int, float | None]:
    """Derive EPS per FY from net income and weighted-average diluted shares.

    Both series come from the same SEC XBRL filing so they share the
    period-end share basis; the result is normalised to today's basis
    using ``split_factor_since`` against the share-count filing date.
    """
    out: dict[int, float | None] = {}
    for fy in sh_series:
        ni = ni_series.get(fy)
        sh = sh_series.get(fy)
        if ni is None or sh is None or sh.val <= 0:
            out[fy] = None
            continue
        factor = price_client.split_factor_since(ticker, sh.filed) or 1.0
        # Normalise to today's basis: shares as filed × cumulative split
        # factor since filing == today's equivalent share count.
        out[fy] = ni.val / (sh.val * factor)
    return out


def _yfinance_eps_series(
    ticker: str,
    latest_fy: int,
    n_years: int,
) -> dict[int, float | None]:
    """Pull a per-FY diluted-EPS series from yfinance income statement.

    yfinance values are already split-adjusted to today's basis, so no
    further normalisation is required. yfinance typically exposes only
    the last 4 fiscal years; missing slots stay ``None`` and the caller
    treats this as a partial-data result.
    """
    try:
        import yfinance as yf  # heavy import — defer until fallback hit
    except ImportError:  # pragma: no cover - yfinance is a hard dep
        return {}

    try:
        t = yf.Ticker(ticker)
        df = t.income_stmt
    except Exception as exc:  # noqa: BLE001 — broad: any yfinance error
        log.warning("yfinance EPS fallback failed for %s: %s", ticker, exc)
        return {}

    if df is None or getattr(df, "empty", True):
        return {}

    eps_row = None
    for label in ("Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"):
        if label in df.index:
            eps_row = df.loc[label]
            break
    if eps_row is None:
        return {}

    out: dict[int, float | None] = {}
    for col, val in eps_row.items():
        # yfinance columns are pandas Timestamps representing the period
        # end. Map to the filer's reported fiscal year — for nearly every
        # US issuer that's the calendar year of the period end (Visa's
        # FY2025 ends 2025-09-30, Apple's FY2024 ends 2024-09-28, etc.).
        try:
            fy = int(col.year)
        except AttributeError:
            continue
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        out[fy] = v

    # Only keep years inside the requested window so downstream loops
    # don't have to filter again.
    window = set(range(latest_fy - n_years + 1, latest_fy + 1))
    return {fy: v for fy, v in out.items() if fy in window}


def _collect_eps_series(
    pit: PointInTimeFacts,
    ticker: str,
    latest_fy: int,
    n_years: int,
    as_of: date,
    price_client: PriceClient,
) -> tuple[dict[int, float | None], str]:
    """Resolve a per-FY diluted-EPS series with explicit fallback chain.

    Returns ``(series_in_today_basis, data_source)``. The series always
    spans ``[latest_fy - n_years + 1, latest_fy]`` with ``None`` for
    fiscal years that no source could populate; ``data_source`` records
    which leg of the chain produced the values so the dashboard can show
    that EPS came from yfinance instead of SEC XBRL.
    """
    window_years = list(range(latest_fy - n_years + 1, latest_fy + 1))

    # Primary: SEC XBRL EPS concept, normalised for splits.
    eps_facts = pit.get_annual_series("eps_diluted", latest_fy, n_years, as_of)
    primary = _adjust_eps_for_splits(eps_facts, ticker, price_client)
    if any(v is not None for v in primary.values()):
        return primary, SOURCE_SEC_XBRL

    # Fallback 1: derive from net income / diluted weighted-average shares
    # (both from SEC XBRL but tagged separately, so issuers like Visa
    # that omit standardised EPS but still report NI and shares are
    # covered).
    ni_series = pit.get_annual_series("net_income", latest_fy, n_years, as_of)
    sh_series = pit.get_annual_series(
        "weighted_avg_shares_diluted", latest_fy, n_years, as_of,
    )
    derived = _eps_from_ni_and_shares(ni_series, sh_series, ticker, price_client)
    full = {fy: derived.get(fy) for fy in window_years}
    if any(v is not None for v in full.values()):
        return full, SOURCE_SEC_NI_OVER_SHARES

    # Fallback 2: yfinance income statement. yfinance only exposes a
    # handful of recent years, so this almost always produces a
    # partial-data window — but it lets sticker price + the latest-year
    # EPS keep working for issuers (e.g. Visa) where SEC XBRL is empty.
    yf_series = _yfinance_eps_series(ticker, latest_fy, n_years)
    if any(v is not None for v in yf_series.values()):
        return {fy: yf_series.get(fy) for fy in window_years}, SOURCE_YFINANCE

    return {fy: None for fy in window_years}, SOURCE_UNAVAILABLE


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
        pit_facts: PointInTimeFacts | None = None,
    ) -> BigFiveResult:
        # Allow the caller (e.g. ``RuleOneAgent.evaluate``) to amortise the
        # company-facts JSON parse + PIT construction across multiple
        # analyzers for the same (ticker, as_of). Fall back to the historical
        # path when no shared PIT is supplied so single-analyzer callers and
        # tests still work unchanged.
        if pit_facts is not None:
            pit = pit_facts
        else:
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

        roic = self._roic(net_income, equity, ltdebt, threshold, n_years=n_years)
        sales = self._growth("Sales Growth", revenue, n_years, threshold)
        equity_g = self._growth("Equity Growth", equity, n_years, threshold)
        ocf_g = self._growth("OCF Growth", ocf, n_years, threshold)

        # EPS is brittle in SEC XBRL — many issuers (Visa, certain
        # foreign filers) skip the standardised concept entirely.
        # ``_collect_eps_series`` walks the SEC → SEC-derived → yfinance
        # fallback chain and tags the source so the evidence can show
        # where the numbers came from.
        eps_adj, eps_source = _collect_eps_series(
            pit=pit, ticker=ticker, latest_fy=latest_fy, n_years=n_years,
            as_of=as_of, price_client=self._prices,
        )
        eps_g = self._growth_from_dict(
            "EPS Growth", eps_adj, n_years, threshold,
            data_source=eps_source,
        )

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
        *,
        n_years: int,
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
                decision_grade=False,
            )

        avg = sum(valid) / len(valid)
        most_recent_fy = max(per_year)
        most_recent = per_year[most_recent_fy]

        # Phil Town: ROIC ≥ 10% per year, "holding steady or rising".
        # Decision-grade rule: every year of the target window populated,
        # average ≥ threshold, AND most-recent ≥ threshold. Below the
        # full window we report the average for transparency but do NOT
        # let the metric "pass" — Phil Town's rule explicitly wants 10
        # years of evidence.
        full_window = len(valid) >= n_years
        signal_pass = (
            avg >= threshold
            and most_recent is not None
            and most_recent >= threshold
        )
        passes = full_window and signal_pass
        if most_recent is None:
            rat = (
                f"avg ROIC over {len(valid)} years = {avg * 100:.2f}%; "
                "most recent FY n/a"
            )
        else:
            rat = (
                f"avg ROIC over {len(valid)} years = {avg * 100:.2f}%; "
                f"most recent FY{most_recent_fy} = "
                f"{most_recent * 100:.2f}%"
            )
            if not full_window:
                rat += (
                    f" (NOTE: only {len(valid)}/{n_years} years populated; "
                    "decision-grade pass requires the full window)"
                )
            elif signal_pass is False:
                # Full window but failing — make it explicit which leg
                # of the rule was violated for the dashboard.
                if avg < threshold:
                    rat += " — fails: average below threshold"
                if most_recent < threshold:
                    rat += " — fails: most-recent FY below threshold"
        return MetricResult(
            name="ROIC", value=avg, threshold=threshold, passes=passes,
            rationale=rat, series=per_year,
            decision_grade=passes,
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
        min_years_for_value: int = MIN_YEARS_FOR_VALUE,
        data_source: str = SOURCE_SEC_XBRL,
    ) -> MetricResult:
        """First-to-last CAGR over the populated subset of ``series``.

        Two distinct decisions live here:

        1. **Should we expose a value at all?** The CAGR is computed when
           at least ``min_years_for_value`` (default 5) populated years are
           available — fewer years make the rate too noisy to be useful
           even as a hint. With less data we return ``value=None`` and
           the metric reads as NO DATA.
        2. **Does the metric pass?** ``passes=True`` requires *both* the
           CAGR ≥ ``threshold`` *and* the full ``n_years`` window
           populated. With fewer years the CAGR is still surfaced (so
           the dashboard can show "would have been X% over 7y"), but
           ``passes`` stays False. This was the previous "relaxed
           5-year pass" loophole; Phil Town's rule explicitly wants the
           full 10-year window so we no longer let a short record claim
           a clean pass.
        """
        if not series:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale="No fiscal years available.", series={},
                decision_grade=False, data_source=data_source,
            )

        present_pairs = [(fy, v) for fy, v in sorted(series.items()) if v is not None]
        if len(present_pairs) < min_years_for_value:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale=(
                    f"Only {len(present_pairs)} years with data; "
                    f"need ≥ {min_years_for_value} to even compute a CAGR."
                ),
                series=series,
                decision_grade=False, data_source=data_source,
            )

        first_fy, first_val = present_pairs[0]
        last_fy, last_val = present_pairs[-1]
        n_periods = last_fy - first_fy
        if n_periods <= 0:
            return MetricResult(
                name=name, value=None, threshold=threshold, passes=False,
                rationale=f"Need >1 year of data (got first=last=FY{first_fy}).",
                series=series,
                decision_grade=False, data_source=data_source,
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
                decision_grade=False, data_source=data_source,
            )

        n_present = len(present_pairs)
        full_window = n_present >= n_years
        signal_pass = cagr >= threshold
        passes = full_window and signal_pass

        if not full_window:
            relaxed_note = (
                f" (NOTE: only {n_present}/{n_years}y populated — "
                "decision-grade pass requires the full window)"
            )
        else:
            relaxed_note = ""
        rat = (
            f"{n_periods}y CAGR FY{first_fy}->{last_fy}: "
            f"{first_val:,.2f} -> {last_val:,.2f} = {cagr * 100:.2f}%"
            f"{relaxed_note}"
        )
        return MetricResult(
            name=name, value=cagr, threshold=threshold, passes=passes,
            rationale=rat, series=series,
            decision_grade=passes, data_source=data_source,
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
                decision_grade=False,
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
            decision_grade=passes,
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
        return BigFiveResult(
            ticker=ticker.upper(),
            as_of=as_of,
            latest_fiscal_year=latest_fy,
            n_years_required=n_years,
            roic=MetricResult(
                "ROIC", None, threshold, False, reason,
                decision_grade=False,
            ),
            sales_growth=MetricResult(
                "Sales Growth", None, threshold, False, reason,
                decision_grade=False,
            ),
            eps_growth=MetricResult(
                "EPS Growth", None, threshold, False, reason,
                decision_grade=False, data_source=SOURCE_UNAVAILABLE,
            ),
            equity_growth=MetricResult(
                "Equity Growth", None, threshold, False, reason,
                decision_grade=False,
            ),
            ocf_growth=MetricResult(
                "OCF Growth", None, threshold, False, reason,
                decision_grade=False,
            ),
            current_ratio=MetricResult(
                "Current Ratio", None, LIQUIDITY_THRESHOLD, False, reason,
                decision_grade=False,
            ),
        )
