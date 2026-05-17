"""Extra Phil Town quantitative checks beyond the Big 5.

These complement the Big 5 + Sticker Price + Payback Time and live alongside
them in the agent's output. Per the design plan they are emitted as **soft
flags** at the agent level — the academic backtest's ``is_buy_full`` keeps
its 9-check definition for reproducibility, and the screening orchestrator
applies its own stricter hard-gate on top.

The three checks:

1. **Debt-Payoff Years** — ``LT_Debt_latest / FCF_latest``. FCF is computed
   here as ``OCF - CapEx`` from PIT facts. Pass iff ``≤ 3`` years (Phil Town:
   "Can the company pay off all its long-term debt with its current free cash
   flow in 3 years or less?").

2. **Dilution** — 10-year CAGR of diluted weighted-average shares outstanding:
       Green ≤ 0%/yr (buybacks)
       Yellow within ±2%/yr (flat)
       Red > +2%/yr (active dilution)
   Pass iff Green or Yellow.

3. **Dividend Quality** — implements Phil Town's logic tree:
       * If ROIC ≥ 15% and dividends are zero → PASS (high-ROIC compounder
         shouldn't be paying dividends).
       * Else compute payout ratio (Dividends / FCF):
            < 60% → PASS, 60–80% → FLAG, > 80% → FAIL.
       * Cross-check **debt-funded dividend**: ``payout_ratio > 100%`` OR
         (``LT_Debt`` rising YoY for 3+ consecutive years AND dividends paid).
       * Cross-check **yield trap**: ``dividend_yield > 7%`` AND any of Big 5
         trending down (latest YoY < 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from quantitative_trading.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import FactValue, PointInTimeFacts
from quantitative_trading.data.prices import PriceClient


# Phil Town defaults for the three new checks.
DEBT_PAYOFF_THRESHOLD_YEARS = 3.0
DILUTION_GREEN_MAX = 0.0       # any non-positive CAGR is good (buybacks)
DILUTION_YELLOW_MAX = 0.02     # up to +2%/yr is acceptable
DIVIDEND_HIGH_ROIC_THRESHOLD = 0.15
DIVIDEND_PAYOUT_PASS_MAX = 0.60
DIVIDEND_PAYOUT_FLAG_MAX = 0.80
DIVIDEND_PAYOUT_DEBT_FUNDED_MIN = 1.00
YIELD_TRAP_MIN = 0.07
DEBT_RISING_YEARS = 3


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DividendQualityDetails:
    """Per-rule reasoning carried alongside the aggregate dividend pass/fail."""

    high_roic_compounder: bool
    pays_dividend: bool
    payout_ratio: float | None
    payout_band: str  # "pass" | "flag" | "fail" | "n/a"
    debt_funded_dividend: bool
    yield_trap: bool
    dividend_yield: float | None
    dividend_growth_pct: float | None  # YoY of dividends_per_share, latest


@dataclass(frozen=True)
class QuantExtrasResult:
    """Aggregate result of the three Phil Town extra checks at one date."""

    ticker: str
    as_of: date
    fiscal_year: int | None

    debt_payoff: MetricResult
    dilution: MetricResult
    dividend_quality: MetricResult
    dividend_details: DividendQualityDetails

    @property
    def all_pass(self) -> bool:
        return (
            self.debt_payoff.passes
            and self.dilution.passes
            and self.dividend_quality.passes
        )

    @property
    def per_check(self) -> dict[str, bool]:
        return {
            "debt_payoff": self.debt_payoff.passes,
            "dilution": self.dilution.passes,
            "dividend_quality": self.dividend_quality.passes,
        }

    def summary(self) -> str:
        lines = [f"QuantExtras({self.ticker} as of {self.as_of}, FY={self.fiscal_year}):"]
        for m in (self.debt_payoff, self.dilution, self.dividend_quality):
            v = "n/a" if m.value is None else f"{m.value:.3f}"
            tag = "OK" if m.passes else "FAIL"
            lines.append(f"  {m.name:>22s}: value={v}  threshold={m.threshold}  {tag}")
        d = self.dividend_details
        lines.append(
            f"  dividend details: pays={d.pays_dividend} payout_band={d.payout_band} "
            f"debt_funded={d.debt_funded_dividend} yield_trap={d.yield_trap}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _val(fv: FactValue | None) -> float | None:
    return fv.val if fv is not None else None


def _last_present(series: dict[int, FactValue | None]) -> tuple[int, float] | None:
    """Return (fy, val) for the most recent non-null entry in a PIT series."""
    candidates = [(fy, fv.val) for fy, fv in sorted(series.items()) if fv is not None]
    return candidates[-1] if candidates else None


def _yoy_change(series: dict[int, FactValue | None]) -> float | None:
    """Latest year-over-year change (decimal) for a PIT series, or None."""
    pairs = [(fy, fv.val) for fy, fv in sorted(series.items()) if fv is not None]
    if len(pairs) < 2:
        return None
    (_, prev), (_, curr) = pairs[-2], pairs[-1]
    if prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _consecutive_rising_years(series: dict[int, FactValue | None]) -> int:
    """How many of the most recent years saw a strict YoY increase."""
    pairs = [fv.val for _, fv in sorted(series.items()) if fv is not None]
    count = 0
    for i in range(len(pairs) - 1, 0, -1):
        if pairs[i] > pairs[i - 1]:
            count += 1
        else:
            break
    return count


def _safe_cagr(first: float, last: float, n_periods: int) -> float | None:
    """CAGR that handles share counts (always positive) — returns signed CAGR."""
    if n_periods <= 0 or first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / n_periods) - 1.0


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------


class QuantExtrasAnalyzer:
    """Compute the three extra Phil Town checks at a point in time.

    All metrics are emitted as ``MetricResult`` to mirror ``BigFiveAnalyzer``'s
    output shape, so downstream code (dataset builder, screening orchestrator)
    can treat them uniformly.
    """

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
        big_five: BigFiveResult | None = None,
        n_years: int = 10,
        pit_facts: PointInTimeFacts | None = None,
    ) -> QuantExtrasResult:
        # See ``BigFiveAnalyzer.evaluate``: the shared ``pit_facts`` is
        # threaded down from the agent so a single (ticker, as_of) only
        # parses the SEC companyfacts JSON once across all analyzers.
        if pit_facts is not None:
            pit = pit_facts
        else:
            cik = self._edgar.get_cik(ticker)
            facts = self._edgar.get_company_facts(cik)
            pit = PointInTimeFacts(facts)

        latest_fy = pit.latest_fiscal_year_with_data("revenue", as_of)
        if latest_fy is None:
            return self._all_unable(ticker, as_of, fy=None,
                                    reason="No fiscal-year revenue data before as_of.")

        # Pull all needed series.
        ocf = pit.get_annual_series("operating_cash_flow", latest_fy, n_years, as_of)
        capex = pit.get_annual_series("capex", latest_fy, n_years, as_of)
        ltdebt = pit.get_annual_series("long_term_debt", latest_fy, n_years, as_of)
        shares = pit.get_annual_series("shares_outstanding", latest_fy, n_years, as_of)
        dividends_cash = pit.get_annual_series(
            "dividends_paid_cash", latest_fy, n_years, as_of
        )
        dividends_per_share = pit.get_annual_series(
            "dividends_per_share", latest_fy, n_years, as_of
        )

        debt_payoff = self._debt_payoff(ocf, capex, ltdebt)
        dilution = self._dilution(shares)
        dividend_quality, dividend_details = self._dividend_quality(
            ticker=ticker, as_of=as_of, big_five=big_five,
            ocf=ocf, capex=capex, ltdebt=ltdebt,
            dividends_cash=dividends_cash,
            dividends_per_share=dividends_per_share,
            shares=shares,
        )

        return QuantExtrasResult(
            ticker=ticker.upper(), as_of=as_of, fiscal_year=latest_fy,
            debt_payoff=debt_payoff, dilution=dilution,
            dividend_quality=dividend_quality,
            dividend_details=dividend_details,
        )

    # ----------------------------------------------------------- Debt-Payoff

    def _debt_payoff(
        self,
        ocf_series: dict[int, FactValue | None],
        capex_series: dict[int, FactValue | None],
        ltdebt_series: dict[int, FactValue | None],
    ) -> MetricResult:
        ocf_pair = _last_present(ocf_series)
        ltdebt_pair = _last_present(ltdebt_series)
        if ocf_pair is None or ltdebt_pair is None:
            return MetricResult(
                name="Debt Payoff (yrs)", value=None,
                threshold=DEBT_PAYOFF_THRESHOLD_YEARS, passes=False,
                rationale="OCF or long-term debt unavailable for latest FY.",
            )
        latest_fy, ocf_val = ocf_pair
        # CapEx is reported as a positive cash outflow on SEC filings. Subtract it.
        capex_val = _val(capex_series.get(latest_fy)) or 0.0
        fcf = ocf_val - capex_val
        ltdebt_val = ltdebt_pair[1]

        if ltdebt_val <= 0:
            return MetricResult(
                name="Debt Payoff (yrs)", value=0.0,
                threshold=DEBT_PAYOFF_THRESHOLD_YEARS, passes=True,
                rationale=f"FY{latest_fy}: no long-term debt — trivially pays off in 0 years.",
                series={fy: (fv.val if fv is not None else None) for fy, fv in ltdebt_series.items()},
            )
        if fcf <= 0:
            return MetricResult(
                name="Debt Payoff (yrs)", value=None,
                threshold=DEBT_PAYOFF_THRESHOLD_YEARS, passes=False,
                rationale=(
                    f"FY{latest_fy}: FCF non-positive (${fcf/1e6:,.1f}M) — "
                    f"cannot service ${ltdebt_val/1e6:,.0f}M of LT debt."
                ),
            )
        years = ltdebt_val / fcf
        passes = years <= DEBT_PAYOFF_THRESHOLD_YEARS
        return MetricResult(
            name="Debt Payoff (yrs)", value=years,
            threshold=DEBT_PAYOFF_THRESHOLD_YEARS, passes=passes,
            rationale=(
                f"FY{latest_fy}: LT debt ${ltdebt_val/1e6:,.0f}M / "
                f"FCF ${fcf/1e6:,.0f}M = {years:.2f} years "
                f"(threshold ≤ {DEBT_PAYOFF_THRESHOLD_YEARS:.0f})"
            ),
            series={fy: (fv.val if fv is not None else None)
                    for fy, fv in ltdebt_series.items()},
        )

    # --------------------------------------------------------------- Dilution

    def _dilution(
        self,
        shares_series: dict[int, FactValue | None],
    ) -> MetricResult:
        present = [(fy, fv.val) for fy, fv in sorted(shares_series.items()) if fv is not None]
        if len(present) < 2:
            return MetricResult(
                name="Dilution CAGR", value=None,
                threshold=DILUTION_YELLOW_MAX, passes=False,
                rationale=f"Need ≥2 years of share-count data (got {len(present)}).",
                series={fy: (fv.val if fv is not None else None)
                        for fy, fv in shares_series.items()},
            )
        first_fy, first_val = present[0]
        last_fy, last_val = present[-1]
        cagr = _safe_cagr(first_val, last_val, last_fy - first_fy)
        if cagr is None:
            return MetricResult(
                name="Dilution CAGR", value=None,
                threshold=DILUTION_YELLOW_MAX, passes=False,
                rationale=(
                    f"CAGR ill-defined for FY{first_fy}={first_val:,.0f} -> "
                    f"FY{last_fy}={last_val:,.0f}"
                ),
            )
        if cagr <= DILUTION_GREEN_MAX:
            band = "GREEN (buybacks)"
        elif cagr <= DILUTION_YELLOW_MAX:
            band = "YELLOW (flat)"
        else:
            band = "RED (active dilution)"
        passes = cagr <= DILUTION_YELLOW_MAX
        return MetricResult(
            name="Dilution CAGR", value=cagr,
            threshold=DILUTION_YELLOW_MAX, passes=passes,
            rationale=(
                f"Shares FY{first_fy}={first_val/1e6:,.1f}M -> "
                f"FY{last_fy}={last_val/1e6:,.1f}M, "
                f"CAGR={cagr*100:.2f}%/yr — {band}"
            ),
            series={fy: (fv.val if fv is not None else None)
                    for fy, fv in shares_series.items()},
        )

    # ------------------------------------------------------- Dividend Quality

    def _dividend_quality(
        self,
        *,
        ticker: str,
        as_of: date,
        big_five: BigFiveResult | None,
        ocf: dict[int, FactValue | None],
        capex: dict[int, FactValue | None],
        ltdebt: dict[int, FactValue | None],
        dividends_cash: dict[int, FactValue | None],
        dividends_per_share: dict[int, FactValue | None],
        shares: dict[int, FactValue | None],
    ) -> tuple[MetricResult, DividendQualityDetails]:

        ocf_pair = _last_present(ocf)
        if ocf_pair is None:
            details = DividendQualityDetails(
                high_roic_compounder=False, pays_dividend=False,
                payout_ratio=None, payout_band="n/a",
                debt_funded_dividend=False, yield_trap=False,
                dividend_yield=None, dividend_growth_pct=None,
            )
            return (MetricResult(
                name="Dividend Quality", value=None,
                threshold=DIVIDEND_PAYOUT_PASS_MAX, passes=False,
                rationale="OCF unavailable; cannot evaluate dividend quality.",
            ), details)
        latest_fy, ocf_val = ocf_pair
        capex_val = _val(capex.get(latest_fy)) or 0.0
        fcf = ocf_val - capex_val

        div_cash = _val(dividends_cash.get(latest_fy)) or 0.0
        pays_dividend = div_cash > 0
        # Payout ratio uses absolute dividends paid against FCF (negative FCF
        # makes the ratio meaningless — caught by the band logic).
        if not pays_dividend:
            payout_ratio: float | None = 0.0 if fcf > 0 else None
        elif fcf <= 0:
            payout_ratio = None  # ratio undefined; caller treats as red flag
        else:
            payout_ratio = abs(div_cash) / fcf

        # Phil Town's high-ROIC carve-out.
        roic = big_five.roic.value if big_five is not None else None
        high_roic = roic is not None and roic >= DIVIDEND_HIGH_ROIC_THRESHOLD

        # Debt-funded check: LT debt rising 3+ consecutive years AND dividend paid.
        debt_streak = _consecutive_rising_years(ltdebt)
        debt_funded = pays_dividend and (
            (payout_ratio is not None and payout_ratio >= DIVIDEND_PAYOUT_DEBT_FUNDED_MIN)
            or debt_streak >= DEBT_RISING_YEARS
        )

        # Yield-trap check: requires a current price + dividends-per-share.
        dps_val = _val(dividends_per_share.get(latest_fy))
        current_price = self._prices.get_close_at(ticker, as_of)
        dividend_yield = None
        if dps_val is not None and current_price and current_price > 0:
            # split-adjust historic per-share figures so the yield matches
            # today's price basis.
            split_factor = self._prices.split_factor_since(
                ticker, dividends_per_share[latest_fy].filed
                if dividends_per_share.get(latest_fy) is not None else as_of
            ) or 1.0
            dividend_yield = (dps_val / split_factor) / current_price

        big5_trending_down = self._big_five_trending_down(big_five)
        yield_trap = (
            dividend_yield is not None
            and dividend_yield > YIELD_TRAP_MIN
            and big5_trending_down
        )

        # Dividend growth %.
        div_growth = _yoy_change(dividends_per_share)

        # Banding.
        if not pays_dividend and high_roic:
            band, base_pass, rationale = (
                "pass",
                True,
                f"FY{latest_fy}: ROIC {roic*100:.1f}% ≥ 15% and no dividend — "
                "high-ROIC compounder; reinvesting is correct.",
            )
        elif not pays_dividend:
            band, base_pass, rationale = (
                "pass",
                True,
                f"FY{latest_fy}: no dividend (ROIC={roic*100:.1f}%)" if roic is not None
                else f"FY{latest_fy}: no dividend.",
            )
        elif payout_ratio is None:
            band, base_pass, rationale = (
                "fail",
                False,
                f"FY{latest_fy}: dividend ${abs(div_cash)/1e6:,.0f}M paid against "
                f"non-positive FCF ${fcf/1e6:,.0f}M — payout undefined.",
            )
        elif payout_ratio < DIVIDEND_PAYOUT_PASS_MAX:
            band, base_pass, rationale = (
                "pass",
                True,
                f"FY{latest_fy}: payout ratio {payout_ratio*100:.0f}% (< 60%); dividend safe.",
            )
        elif payout_ratio < DIVIDEND_PAYOUT_FLAG_MAX:
            band, base_pass, rationale = (
                "flag",
                True,  # Phil Town: 60-80% is "flag", we still pass with a warning
                f"FY{latest_fy}: payout ratio {payout_ratio*100:.0f}% (60-80%) — "
                "tight; little margin for error.",
            )
        else:
            band, base_pass, rationale = (
                "fail",
                False,
                f"FY{latest_fy}: payout ratio {payout_ratio*100:.0f}% (> 80%) — "
                "dividend eats too much FCF.",
            )

        # Apply red-flag overrides.
        passes = base_pass and not debt_funded and not yield_trap
        if debt_funded:
            rationale += (
                f" RED FLAG: debt-funded dividend "
                f"(LT debt rising {debt_streak}y in a row"
                + (f", payout {payout_ratio*100:.0f}%>=100%" if payout_ratio
                   and payout_ratio >= 1.0 else "")
                + ")."
            )
        if yield_trap:
            rationale += (
                f" RED FLAG: yield trap (yield {dividend_yield*100:.1f}% > 7% "
                "with Big 5 trending down)."
            )

        details = DividendQualityDetails(
            high_roic_compounder=high_roic,
            pays_dividend=pays_dividend,
            payout_ratio=payout_ratio,
            payout_band=band,
            debt_funded_dividend=debt_funded,
            yield_trap=yield_trap,
            dividend_yield=dividend_yield,
            dividend_growth_pct=div_growth,
        )
        result = MetricResult(
            name="Dividend Quality",
            value=payout_ratio,
            threshold=DIVIDEND_PAYOUT_PASS_MAX,
            passes=passes,
            rationale=rationale,
        )
        return result, details

    @staticmethod
    def _big_five_trending_down(big_five: BigFiveResult | None) -> bool:
        """Heuristic: any of the Big 5 most-recent YoY < 0."""
        if big_five is None:
            return False
        for metric in (
            big_five.roic, big_five.sales_growth, big_five.eps_growth,
            big_five.equity_growth, big_five.ocf_growth,
        ):
            series = metric.series
            if not series:
                continue
            present = [v for _, v in sorted(series.items()) if v is not None]
            if len(present) < 2:
                continue
            if present[-1] < present[-2]:
                return True
        return False

    # ----------------------------------------------------------------- Misc

    def _all_unable(
        self,
        ticker: str,
        as_of: date,
        *,
        fy: int | None,
        reason: str,
    ) -> QuantExtrasResult:
        return QuantExtrasResult(
            ticker=ticker.upper(), as_of=as_of, fiscal_year=fy,
            debt_payoff=MetricResult("Debt Payoff (yrs)", None,
                                     DEBT_PAYOFF_THRESHOLD_YEARS, False, reason),
            dilution=MetricResult("Dilution CAGR", None,
                                  DILUTION_YELLOW_MAX, False, reason),
            dividend_quality=MetricResult("Dividend Quality", None,
                                          DIVIDEND_PAYOUT_PASS_MAX, False, reason),
            dividend_details=DividendQualityDetails(
                high_roic_compounder=False, pays_dividend=False,
                payout_ratio=None, payout_band="n/a",
                debt_funded_dividend=False, yield_trap=False,
                dividend_yield=None, dividend_growth_pct=None,
            ),
        )
