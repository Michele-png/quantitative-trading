"""Sector/date-matched S&P 500 controls for the elite-vs-control comparison.

For each evaluated elite buy `(investor, ticker_i, T_eval_i)` we draw `K`
random control tickers from:

    S&P 500 constituents at T_eval_i
    intersect same SIC 2-digit sector
    intersect evaluable (>=10y XBRL history, non-financial, non-holdco)

and score each control with the same Rule One agent at the same T_eval. This
produces the matched-control dataset that the §7.A headline test compares
against, and the §6.1 baseline that turns descriptive elite pass-rates into
elite-vs-baseline effect sizes.

Per audit plan section 6: the SAME exclusion rules apply to controls (financials,
holdcos, ADRs, young companies). Without sector matching, the control would
systematically favour industries where Big 5 is mechanically easy (consumer
staples, software) and punish industries where it's hard (banks, capital-
intensive cyclicals).
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantitative_trading.agents.rule_one.agent import RuleOneAgent
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.prices import PriceClient
from quantitative_trading.data.universe import SP500Universe
from quantitative_trading.dataset.investor_purchases_dataset import (
    BIG_FIVE_HORIZON_YEARS,
    FINANCIAL_SIC_MAX,
    FINANCIAL_SIC_MIN,
    KNOWN_HOLDCO_CIKS,
    _score_with_agent,
    get_sic_for_cik,
)

log = logging.getLogger(__name__)


# Default number of controls drawn per elite buy. Audit plan §6.
DEFAULT_K = 10

# Window around the elite buy from which to consider control candidates.
# Use exact T_eval (no slack) — we want strict date matching for the CMH
# stratification by (sector, quarter).


@dataclass(frozen=True)
class ControlRow:
    """One scored control observation matched to an elite buy."""

    elite_investor_short_id: str
    elite_cusip: str
    elite_ticker: str
    elite_period_of_report: date

    # The control's identifiers + match keys
    control_ticker: str
    control_cik: int
    control_sic_2digit: int

    # T_eval is shared with the elite buy
    t_eval: date

    # Booleans (None if scoring failed)
    pass_roic: bool | None
    pass_sales_growth: bool | None
    pass_eps_growth: bool | None
    pass_equity_growth: bool | None
    pass_ocf_growth: bool | None
    pass_margin_of_safety: bool | None
    pass_payback_time: bool | None
    n_criteria_passed: int | None
    big5_pass: bool | None
    all_seven_pass: bool | None
    years_of_history_available: int | None


def _is_evaluable_candidate(
    edgar: EdgarClient, ticker: str, sic_cache: dict[int, int | None]
) -> tuple[bool, int | None, int | None]:
    """Return (is_evaluable, sic_code, cik) for a control candidate.

    A candidate is evaluable iff it:
      * has a CIK in SEC company_tickers (must be a US public filer),
      * is not in a financials SIC 6000-6999 sector,
      * is not in the known-holdcos set.

    The "10y of history" check happens at scoring time — we let those candidates
    through here and only exclude them later when the agent reports
    `years_of_history_available < 10`.
    """
    try:
        cik = edgar.get_cik(ticker)
    except Exception:  # noqa: BLE001
        return False, None, None
    if cik in KNOWN_HOLDCO_CIKS:
        return False, None, cik
    sic = sic_cache.get(cik)
    if sic is None:
        sic = get_sic_for_cik(edgar, cik)
        sic_cache[cik] = sic
    if sic is not None and FINANCIAL_SIC_MIN <= sic <= FINANCIAL_SIC_MAX:
        return False, sic, cik
    return True, sic, cik


def sample_controls(
    elite_buys_df: pd.DataFrame,
    *,
    k_per_buy: int = DEFAULT_K,
    seed: int = 20260501,
    edgar_client: EdgarClient | None = None,
    price_client: PriceClient | None = None,
    universe: SP500Universe | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Draw `k_per_buy` sector/date-matched S&P 500 controls per evaluable elite buy.

    `elite_buys_df` is the output of `build_audit_dataset`; we filter it to
    `lookback_completeness == "clean" AND non_evaluable_reason is null` and
    sample controls only against that subset.

    Random seed `seed` is mixed with each elite-buy CUSIP+date so the sampling
    is reproducible and per-buy.

    Returns the long-format control DataFrame (one row per (elite_buy, control)).
    """
    edgar = edgar_client or EdgarClient()
    prices = price_client or PriceClient()
    sp500 = universe or SP500Universe()
    agent = RuleOneAgent(edgar, prices, anthropic_client=None)

    elite_evaluable = elite_buys_df[
        (elite_buys_df["lookback_completeness"] == "clean")
        & (elite_buys_df["non_evaluable_reason"].isna())
    ].copy()

    log.info("Sampling %d controls per elite buy x %d evaluable buys = %d controls",
             k_per_buy, len(elite_evaluable), k_per_buy * len(elite_evaluable))

    rng = random.Random(seed)
    sic_cache: dict[int, int | None] = {}
    rows: list[ControlRow] = []

    # Cache S&P 500 candidates per (T_eval, sector) so multiple elite buys
    # in the same stratum reuse the candidate pool.
    candidates_cache: dict[tuple[date, int], list[tuple[str, int, int]]] = {}

    for i, buy in elite_evaluable.iterrows():
        elite_ticker = buy["ticker"]
        elite_cusip = buy["cusip"]
        elite_period = (
            buy["period_of_report"]
            if isinstance(buy["period_of_report"], date)
            else date.fromisoformat(str(buy["period_of_report"]))
        )
        t_eval = (
            buy["t_eval"]
            if isinstance(buy["t_eval"], date)
            else date.fromisoformat(str(buy["t_eval"]))
        )
        elite_sic = buy.get("sic_code")
        if elite_sic is None or pd.isna(elite_sic):
            log.debug("Skipping elite buy without SIC: %s", elite_ticker)
            continue
        elite_sic = int(elite_sic)
        sic2 = elite_sic // 100  # SIC 2-digit prefix

        # Build the candidate pool for (T_eval, sic2) if not cached.
        cache_key = (t_eval, sic2)
        if cache_key not in candidates_cache:
            sp500_members = sp500.get_members(t_eval)
            candidates: list[tuple[str, int, int]] = []
            for ticker in sorted(sp500_members):
                if ticker == elite_ticker:
                    continue
                evaluable, sic, cik = _is_evaluable_candidate(edgar, ticker, sic_cache)
                if not evaluable or sic is None or cik is None:
                    continue
                if sic // 100 != sic2:
                    continue
                candidates.append((ticker, cik, sic))
            candidates_cache[cache_key] = candidates
            log.debug("S&P500 candidates for (%s, SIC %d): %d",
                      t_eval, sic2, len(candidates))

        candidates = candidates_cache[cache_key]
        if not candidates:
            log.warning("No matched candidates for %s @ %s SIC %d",
                        elite_ticker, t_eval, sic2)
            continue

        # Sample K controls deterministically per elite buy.
        local_rng = random.Random(f"{seed}|{elite_cusip}|{t_eval.isoformat()}")
        sample_size = min(k_per_buy, len(candidates))
        sampled = local_rng.sample(candidates, sample_size)

        for control_ticker, control_cik, control_sic in sampled:
            booleans, years_history = _score_with_agent(agent, control_ticker, t_eval)

            # If the control has < 10y history, mark booleans as None
            # (it would be `young_company` in the elite analogy).
            if booleans is None or (years_history or 0) < BIG_FIVE_HORIZON_YEARS:
                rows.append(ControlRow(
                    elite_investor_short_id=buy["investor_short_id"],
                    elite_cusip=elite_cusip,
                    elite_ticker=elite_ticker,
                    elite_period_of_report=elite_period,
                    control_ticker=control_ticker,
                    control_cik=control_cik,
                    control_sic_2digit=sic2,
                    t_eval=t_eval,
                    pass_roic=None, pass_sales_growth=None, pass_eps_growth=None,
                    pass_equity_growth=None, pass_ocf_growth=None,
                    pass_margin_of_safety=None, pass_payback_time=None,
                    n_criteria_passed=None, big5_pass=None, all_seven_pass=None,
                    years_of_history_available=years_history,
                ))
                continue

            n_pass = sum(1 for v in booleans.values() if v)
            big5_keys = ("roic", "sales_growth", "eps_growth", "equity_growth", "ocf_growth")
            big5_pass = all(booleans[k] for k in big5_keys)
            all7 = all(booleans.values())
            rows.append(ControlRow(
                elite_investor_short_id=buy["investor_short_id"],
                elite_cusip=elite_cusip,
                elite_ticker=elite_ticker,
                elite_period_of_report=elite_period,
                control_ticker=control_ticker,
                control_cik=control_cik,
                control_sic_2digit=sic2,
                t_eval=t_eval,
                pass_roic=booleans["roic"],
                pass_sales_growth=booleans["sales_growth"],
                pass_eps_growth=booleans["eps_growth"],
                pass_equity_growth=booleans["equity_growth"],
                pass_ocf_growth=booleans["ocf_growth"],
                pass_margin_of_safety=booleans["margin_of_safety"],
                pass_payback_time=booleans["payback_time"],
                n_criteria_passed=n_pass,
                big5_pass=big5_pass,
                all_seven_pass=all7,
                years_of_history_available=years_history,
            ))

        if (i + 1) % 10 == 0:
            log.info("Processed elite buy %d / %d", i + 1, len(elite_evaluable))

    df = pd.DataFrame([asdict(r) for r in rows])
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Wrote %d control rows to %s", len(df), output_csv)
    return df
