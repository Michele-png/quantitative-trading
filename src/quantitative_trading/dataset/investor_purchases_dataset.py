"""Investor-purchase audit dataset builder.

Produces the audit row per `(investor, cusip, period_of_report)` tuple
discovered by `purchase_detection.detect_new_positions_with_reinit_check`.
Each row carries:

* The 7 Rule One quant booleans (Big 5 + MoS + Payback) at T_eval,
  computed by reusing the existing `RuleOneAgent.evaluate(..., include_llm=False)`.
* `non_evaluable_reason` (one of `young_company` / `financial` / `holdco` /
  `foreign_no_data` / `cusip_unresolved` / `cik_unknown` / `null`).
* The §3 lookback stratum fields (`lookback_completeness`, `lookback_strategy`,
  `effective_lookback_quarters`).
* Provenance (CIK / accession / filing_date / position size at first appearance).

T_eval convention
-----------------
Per audit plan section 3 PRIMARY rule: `T_eval = end of quarter Q-1` (the last date by which
the buy decision must have been made). The end-of-Q "generous" sensitivity
is supported via the `t_eval_mode` parameter.

Realized-return computation
---------------------------
Holding period and realized CAGR are computed in a SEPARATE pass via
`enrich_with_realized_returns` since they require walking the same investor's
13F history forward to find the exit quarter. Keeping it separate lets the
fast-path scoring run independently.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantitative_trading.agents.rule_one.agent import RuleOneAgent
from quantitative_trading.data.edgar import EdgarClient, EdgarError
from quantitative_trading.data.prices import PriceClient
from quantitative_trading.investors.cusip_resolver import (
    CusipResolution,
    CusipResolver,
)
from quantitative_trading.investors.investor_universe import Investor
from quantitative_trading.investors.purchase_detection import (
    NewPosition,
    detect_new_positions_with_reinit_check,
)
from quantitative_trading.investors.thirteen_f import ThirteenFClient

log = logging.getLogger(__name__)


# Phil Town's Big 5 horizon. <10y of XBRL history at T_eval -> young_company.
BIG_FIVE_HORIZON_YEARS = 10

# SIC range for Finance, Insurance & Real Estate (excluded — ROIC ill-defined).
FINANCIAL_SIC_MIN = 6000
FINANCIAL_SIC_MAX = 6999

# Hardcoded list of known holdcos / investment vehicles whose Big 5 are
# undefined (Berkshire's book value reflects mark-to-market of marketable
# securities, not operating ROIC). Add more as discovered.
KNOWN_HOLDCO_CIKS: frozenset[int] = frozenset({
    1067983,  # Berkshire Hathaway Inc
    62996,    # Markel Group Inc
    915389,   # Loews Corp
    63060,    # Leucadia/Jefferies Financial Group
})


@dataclass(frozen=True)
class AuditRow:
    """One row of the investor-purchase audit dataset."""

    # Identifiers / provenance
    investor_short_id: str
    cusip: str
    ticker: str | None
    cik: int | None
    name_of_issuer: str
    security_type: str | None
    sic_code: int | None

    # 13F event
    period_of_report: date
    filing_date: date
    filing_cik: int            # which CIK reported the buy (for multi-CIK investors)
    accession_number: str
    shares_initial: int
    value_usd_initial: float

    # T_eval choice
    t_eval: date
    t_eval_mode: str           # "q_minus_1" | "q_end"

    # §3 lookback stratification
    lookback_completeness: str
    lookback_strategy: str | None
    effective_lookback_quarters: int

    # Evaluability + reason for exclusion
    non_evaluable_reason: str | None    # None == evaluable
    years_of_history_available: int | None

    # The 7 Rule One booleans (None if non-evaluable)
    pass_roic: bool | None
    pass_sales_growth: bool | None
    pass_eps_growth: bool | None
    pass_equity_growth: bool | None
    pass_ocf_growth: bool | None
    pass_margin_of_safety: bool | None
    pass_payback_time: bool | None

    # Derived
    n_criteria_passed: int | None       # 0-7 if evaluable, else None
    big5_pass: bool | None
    all_seven_pass: bool | None

    # Realized-return (filled by enrich_with_realized_returns, NaN initially)
    holding_period_quarters: int | None
    realized_cagr_to_exit: float | None
    is_right_censored: bool | None


# ----------------------------------------------------------------- T_eval


def t_eval_for(period_of_report: date, mode: str) -> date:
    """Return the agent-evaluation date for a given quarter and mode.

    * `q_minus_1` (PRIMARY, audit plan section 3): the last day of the quarter BEFORE
      the one in which the position first appeared. This is the latest date
      at which the buy decision must have been made.
    * `q_end` (sensitivity): the quarter-end of the discovering quarter. The
      most generous to the investor — they get any 10-K filed mid-quarter.
    """
    if mode == "q_end":
        return period_of_report
    if mode == "q_minus_1":
        # End of the previous quarter (the day before the start of this one).
        first_of_q = date(period_of_report.year, period_of_report.month, 1)
        return first_of_q - timedelta(days=1)
    raise ValueError(f"Unknown t_eval mode: {mode!r}")


# ----------------------------------------------------------------- Evaluability


def get_sic_for_cik(edgar: EdgarClient, cik: int) -> int | None:
    """Fetch the SIC code from SEC submissions for a CIK."""
    try:
        subs = edgar.get_submissions(cik)
    except EdgarError:
        return None
    try:
        return int(subs.get("sic", "").strip()) if subs.get("sic") else None
    except (ValueError, AttributeError):
        return None


def classify_non_evaluable(
    *,
    resolution: CusipResolution | None,
    sic_code: int | None,
    cik: int | None,
) -> str | None:
    """Return the non_evaluable_reason or None if the position is evaluable.

    Evaluable means: resolved to a US Common Stock ticker with a SEC CIK,
    not in a financials SIC range, not a known holdco. The young-company
    check happens later (requires fetching XBRL facts, so deferred).
    """
    if resolution is None or not resolution.is_resolved:
        return "cusip_unresolved"
    if not resolution.is_evaluable_security_type:
        # ADRs, preferred, warrants, etc.
        return "foreign_no_data" if resolution.security_type == "ADR" else "other_security_type"
    if cik is None:
        return "cik_unknown"
    if cik in KNOWN_HOLDCO_CIKS:
        return "holdco"
    if sic_code is not None and FINANCIAL_SIC_MIN <= sic_code <= FINANCIAL_SIC_MAX:
        return "financial"
    return None


# ----------------------------------------------------------------- Scoring


def _score_with_agent(
    agent: RuleOneAgent,
    ticker: str,
    t_eval: date,
) -> tuple[dict[str, bool] | None, int | None]:
    """Run the Rule One agent on (ticker, t_eval), return (booleans, years_history).

    Returns (None, n_years) if the company has insufficient history to
    compute Big 5 (young_company case).
    """
    try:
        result = agent.evaluate(ticker, t_eval, include_llm=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Agent failed for %s @ %s: %s", ticker, t_eval, exc)
        return None, None

    bf = result.big_five
    # Years of history available is the count of distinct fiscal years where
    # the revenue series had a value (proxy for company-history depth).
    series = bf.sales_growth.series or {}
    n_years_with_data = sum(1 for v in series.values() if v is not None)

    # If the agent itself found no fiscal year, the result will be all-fail
    # with a `latest_fiscal_year=None` -- that signals "company didn't exist
    # at T_eval as a public filer".
    if bf.latest_fiscal_year is None:
        return None, n_years_with_data

    return result.quant_check_results, n_years_with_data


# ----------------------------------------------------------------- Builder


def build_audit_dataset(
    investors: Iterable[Investor],
    *,
    window_start: date,
    window_end: date,
    t_eval_mode: str = "q_minus_1",
    edgar_client: EdgarClient | None = None,
    price_client: PriceClient | None = None,
    cusip_resolver: CusipResolver | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Build the full audit dataset for the given investors and window.

    Steps:
        1. Per-investor: detect new positions across the window (uses
           `detect_new_positions_with_reinit_check`).
        2. Bulk-resolve all unique CUSIPs to (ticker, CIK, security_type)
           via the dual-source CUSIP resolver.
        3. For each (resolved, evaluable) position: run the Rule One agent
           at T_eval and extract the 7 quant booleans.
        4. For each non-evaluable position: record the reason.
        5. Emit one `AuditRow` per `NewPosition` (regardless of evaluability).
        6. Write CSV if `output_csv` is provided.

    The returned DataFrame contains all rows — re-initiations and
    incomplete-lookback included. Consumers filter by `lookback_completeness`
    and `non_evaluable_reason` for the headline analyses.
    """
    edgar = edgar_client or EdgarClient()
    prices = price_client or PriceClient()
    resolver = cusip_resolver or CusipResolver(edgar)
    tf = ThirteenFClient(edgar)
    agent = RuleOneAgent(edgar, prices, anthropic_client=None)

    # Step 1: detect all new positions per investor.
    all_positions: list[NewPosition] = []
    for inv in investors:
        log.info("Detecting new positions for %s", inv.short_id)
        positions = detect_new_positions_with_reinit_check(
            inv, tf, window_start=window_start, window_end=window_end,
        )
        all_positions.extend(positions)
    log.info("Total NewPosition rows: %d", len(all_positions))

    # Step 2: bulk-resolve CUSIPs (with issuer-name hints for SEC fallback).
    unique_cusips = sorted({p.cusip for p in all_positions})
    issuer_hints: dict[str, str] = {}
    for p in all_positions:
        issuer_hints.setdefault(p.cusip, p.name_of_issuer)
    log.info("Bulk-resolving %d unique CUSIPs", len(unique_cusips))
    resolutions = resolver.bulk_resolve(unique_cusips, issuer_name_hints=issuer_hints)

    # Step 3-5: build rows.
    sic_cache: dict[int, int | None] = {}
    rows: list[AuditRow] = []

    for i, np in enumerate(all_positions):
        if (i + 1) % 50 == 0:
            log.info("Scoring position %d / %d", i + 1, len(all_positions))

        res = resolutions.get(np.cusip)
        ticker = res.ticker if res else None
        cik = res.cik if res else None
        sic = sic_cache.get(cik) if cik is not None else None
        if cik is not None and cik not in sic_cache:
            sic = get_sic_for_cik(edgar, cik)
            sic_cache[cik] = sic

        non_eval = classify_non_evaluable(resolution=res, sic_code=sic, cik=cik)

        t_eval = t_eval_for(np.period_of_report, t_eval_mode)

        # Run the agent only if the position passes the structural evaluability
        # check AND lookback_completeness == "clean" (we don't waste cycles
        # scoring re-initiations that are excluded from the headline).
        booleans: dict[str, bool] | None = None
        years_history: int | None = None
        if non_eval is None and np.lookback_completeness == "clean" and ticker is not None:
            booleans, years_history = _score_with_agent(agent, ticker, t_eval)
            if booleans is None:
                # Agent couldn't compute -> young_company or similar
                non_eval = "young_company" if (years_history or 0) < BIG_FIVE_HORIZON_YEARS \
                    else "agent_failed"
            elif (years_history or 0) < BIG_FIVE_HORIZON_YEARS:
                # Has data but <10 years -> route to young_company per audit plan section 5.
                non_eval = "young_company"

        # Derived counts.
        if booleans is not None and non_eval is None:
            n_pass = sum(1 for v in booleans.values() if v)
            big5_keys = ("roic", "sales_growth", "eps_growth", "equity_growth", "ocf_growth")
            big5_pass = all(booleans[k] for k in big5_keys)
            all7 = all(booleans.values())
        else:
            n_pass = None
            big5_pass = None
            all7 = None

        rows.append(AuditRow(
            investor_short_id=np.investor_short_id,
            cusip=np.cusip,
            ticker=ticker,
            cik=cik,
            name_of_issuer=np.name_of_issuer,
            security_type=res.security_type if res else None,
            sic_code=sic,
            period_of_report=np.period_of_report,
            filing_date=np.filing_date,
            filing_cik=np.cik,
            accession_number=np.accession_number,
            shares_initial=np.shares,
            value_usd_initial=np.value_usd,
            t_eval=t_eval,
            t_eval_mode=t_eval_mode,
            lookback_completeness=np.lookback_completeness,
            lookback_strategy=np.lookback_strategy,
            effective_lookback_quarters=np.effective_lookback_quarters,
            non_evaluable_reason=non_eval,
            years_of_history_available=years_history,
            pass_roic=booleans["roic"] if booleans else None,
            pass_sales_growth=booleans["sales_growth"] if booleans else None,
            pass_eps_growth=booleans["eps_growth"] if booleans else None,
            pass_equity_growth=booleans["equity_growth"] if booleans else None,
            pass_ocf_growth=booleans["ocf_growth"] if booleans else None,
            pass_margin_of_safety=booleans["margin_of_safety"] if booleans else None,
            pass_payback_time=booleans["payback_time"] if booleans else None,
            n_criteria_passed=n_pass,
            big5_pass=big5_pass,
            all_seven_pass=all7,
            holding_period_quarters=None,
            realized_cagr_to_exit=None,
            is_right_censored=None,
        ))

    df = pd.DataFrame([asdict(r) for r in rows])
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Wrote %d rows to %s", len(df), output_csv)

    return df


# ----------------------------------------------------------------- Realized returns


def enrich_with_realized_returns(
    df: pd.DataFrame,
    investors: Iterable[Investor],
    *,
    window_end: date,
    edgar_client: EdgarClient | None = None,
    price_client: PriceClient | None = None,
) -> pd.DataFrame:
    """Add `holding_period_quarters`, `realized_cagr_to_exit`, `is_right_censored`.

    For each row, we walk the investor's 13F history forward from the
    appearance quarter and find the FIRST quarter where the CUSIP no longer
    appears in any of their 13Fs (across all CIKs, latest-filed per quarter).
    If still held at `window_end`, the row is right-censored.

    Realized CAGR uses `PriceClient.forward_total_return_cagr(ticker, t_eval, exit_date)`.
    """
    edgar = edgar_client or EdgarClient()
    prices = price_client or PriceClient()
    tf = ThirteenFClient(edgar)

    # Build per-investor quarterly-holdings index.
    investor_quarterly_cusips: dict[str, dict[date, set[str]]] = {}
    for inv in investors:
        cusip_by_quarter: dict[date, set[str]] = {}
        for rec in inv.cik_history:
            for f in tf.fetch_all_filings(rec.cik):
                cusip_by_quarter.setdefault(f.period_of_report, set()).update(
                    h.cusip for h in f.holdings
                )
        investor_quarterly_cusips[inv.short_id] = cusip_by_quarter

    holdings: list[int | None] = []
    cagrs: list[float | None] = []
    censored: list[bool | None] = []

    for _, row in df.iterrows():
        inv_qty = investor_quarterly_cusips.get(row["investor_short_id"], {})
        sorted_q = sorted(inv_qty.keys())
        appearance_q = row["period_of_report"]
        if isinstance(appearance_q, str):
            appearance_q = date.fromisoformat(appearance_q)

        # Walk forward from appearance_q.
        exit_q: date | None = None
        for q in sorted_q:
            if q <= appearance_q:
                continue
            if row["cusip"] not in inv_qty[q]:
                exit_q = q
                break

        is_censored = exit_q is None
        # If censored, use the latest quarter the investor reported (or window_end).
        end_for_return = exit_q if exit_q is not None else (sorted_q[-1] if sorted_q else window_end)

        # Holding period in quarters from appearance to exit (or end).
        delta_months = (end_for_return.year - appearance_q.year) * 12 + (
            end_for_return.month - appearance_q.month
        )
        holding_q = max(0, delta_months // 3)

        # Realized CAGR (only if ticker is known).
        ticker = row.get("ticker")
        cagr: float | None = None
        if ticker and not pd.isna(ticker):
            t_eval = row["t_eval"]
            if isinstance(t_eval, str):
                t_eval = date.fromisoformat(t_eval)
            try:
                cagr = prices.forward_total_return_cagr(ticker, t_eval, end_for_return)
            except Exception as exc:  # noqa: BLE001
                log.debug("CAGR failed for %s: %s", ticker, exc)
                cagr = None

        holdings.append(holding_q)
        cagrs.append(cagr)
        censored.append(is_censored)

    df = df.copy()
    df["holding_period_quarters"] = holdings
    df["realized_cagr_to_exit"] = cagrs
    df["is_right_censored"] = censored
    return df
