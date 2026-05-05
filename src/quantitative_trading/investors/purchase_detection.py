"""Detect first-ever new positions in an investor's 13F history.

Per audit plan section 3 (revised after feedback B): the lookback for "is this a
brand-new position?" uses the FULL available filing history for that investor
(uncapped, not min(12, ...) as in the earlier design). This eliminates the
~5% misclassification rate of pre-2013 re-initiations as new buys for
long-history filers (Akre, Harris, Russo, Berkowitz, Weitz, etc.).

Per-purchase classification
---------------------------
Each (investor, ticker, quarter-of-first-appearance) tuple is tagged with:

* `lookback_completeness`:
    - `clean` -- effective_lookback_quarters >= 12 AND ticker absent in all
                 prior available filings. Enters the headline test (audit plan section 7.A).
    - `incomplete_lookback` -- effective_lookback_quarters < 12 (insufficient
                               data to call the buy "first ever").
    - `re_initiation` -- ticker present at some prior quarter within the
                         available history (true re-buy, not a new position).

* `lookback_strategy` (only for `clean` rows):
    - `full_filing_history` -- effective_lookback_quarters >= 40 (>=10y of
                               pre-T_eval data, matching the Big 5 horizon).
    - `truncated_to_first_filing` -- 12 <= effective_lookback_quarters < 40.
                                     Lookback uses everything available but
                                     is not a full 10y.

The §7.E sensitivity 2 re-runs the headline on `full_filing_history` rows
only; this stratum field makes that subset query trivial.

Multi-CIK handling
------------------
Investors who reorganized their filing entity (Pabrai, Greenberg) have
multiple CIKs in `investor_universe.py`. This module merges 13F filings
across all CIKs for an investor before computing first-ever-appearance.

Round-trips and re-initiations
------------------------------
A ticker that was held in some prior quarter (by ANY of the investor's CIKs),
disappeared, and reappeared is `re_initiation` -- excluded from the headline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from quantitative_trading.investors.investor_universe import Investor
from quantitative_trading.investors.thirteen_f import (
    ThirteenFClient,
    ThirteenFFiling,
)

log = logging.getLogger(__name__)


# Audit plan section 3 thresholds.
MIN_CLEAN_LOOKBACK_QUARTERS = 12       # 3y of prior 13F history required for `clean`
FULL_HISTORY_THRESHOLD_QUARTERS = 40   # 10y of pre-T_eval data for `full_filing_history`


@dataclass(frozen=True)
class NewPosition:
    """A first-ever appearance of a CUSIP in an investor's merged 13F history."""

    investor_short_id: str
    cusip: str
    name_of_issuer: str

    # The quarter Q in which the position first appeared.
    period_of_report: date              # quarter-end of the discovering 13F
    filing_date: date                   # actual SEC filing date (Q + ~45d)
    cik: int                            # which CIK reported the new position
    accession_number: str               # provenance: which filing

    # The position's size at first appearance.
    shares: int
    value_usd: float

    # The audit-plan section 3 classification.
    lookback_completeness: str          # "clean" | "incomplete_lookback" | "re_initiation"
    lookback_strategy: str | None       # "full_filing_history" | "truncated_to_first_filing" | None
    effective_lookback_quarters: int    # T_eval - first_ever_filing_quarter, in quarters

    @property
    def is_clean(self) -> bool:
        return self.lookback_completeness == "clean"

    @property
    def is_full_history(self) -> bool:
        return self.lookback_strategy == "full_filing_history"


# ----------------------------------------------------------------- Helpers


def _quarters_between(start: date, end: date) -> int:
    """Number of full quarters from `start` to `end` (rounded down)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months // 3


def _filing_realm(filings: list[ThirteenFFiling]) -> dict[date, list[ThirteenFFiling]]:
    """Group filings by `period_of_report`. There can be amendments per quarter."""
    out: dict[date, list[ThirteenFFiling]] = {}
    for f in filings:
        out.setdefault(f.period_of_report, []).append(f)
    return out


def _consolidated_holdings_for_quarter(
    quarter_filings: list[ThirteenFFiling],
) -> dict[str, tuple[ThirteenFFiling, int, float, str]]:
    """Reduce one quarter's filings (potentially HR + HR/A) into CUSIP -> last value.

    For each CUSIP, the LATEST-filed entry wins (amendments supersede the
    original). Returns CUSIP -> (filing, shares, value_usd, name_of_issuer).
    """
    # Sort so that amendments (filed later) overwrite originals.
    quarter_filings_sorted = sorted(quarter_filings, key=lambda f: f.filing_date)
    out: dict[str, tuple[ThirteenFFiling, int, float, str]] = {}
    for f in quarter_filings_sorted:
        for h in f.holdings:
            out[h.cusip] = (f, h.shares, h.value_usd, h.name_of_issuer)
    return out


# ----------------------------------------------------------------- Detection


def detect_new_positions(
    investor: Investor,
    tf_client: ThirteenFClient,
    *,
    window_start: date,
    window_end: date,
) -> list[NewPosition]:
    """Detect all new positions for an investor in [window_start, window_end].

    Workflow:
        1. Fetch all 13F filings across ALL of the investor's CIKs.
        2. Merge by quarter (handle amendments by latest-wins).
        3. Walk quarters in chronological order; track the cumulative set of
           CUSIPs ever seen in any prior quarter.
        4. For each quarter Q in the analysis window, find CUSIPs present
           in Q that are NOT in the cumulative-prior-set.
        5. Classify each by lookback_completeness and lookback_strategy.
    """
    # Step 1: gather across all CIKs.
    all_filings: list[ThirteenFFiling] = []
    for rec in investor.cik_history:
        log.info("Fetching 13F history for %s CIK %d", investor.short_id, rec.cik)
        all_filings.extend(tf_client.fetch_all_filings(rec.cik))

    if not all_filings:
        log.warning("No 13F filings found for %s", investor.short_id)
        return []

    # Step 2: group by quarter, oldest first.
    by_quarter = _filing_realm(all_filings)
    sorted_quarters = sorted(by_quarter.keys())

    investor_first_filing = investor.first_ever_filing_date

    # Step 3-4: walk quarters tracking ever-seen CUSIPs.
    ever_seen: set[str] = set()
    new_positions: list[NewPosition] = []

    for q in sorted_quarters:
        consolidated = _consolidated_holdings_for_quarter(by_quarter[q])

        # Identify new CUSIPs (relative to all prior quarters in this investor's history)
        new_cusips = set(consolidated.keys()) - ever_seen

        if window_start <= q <= window_end:
            for cusip in sorted(new_cusips):
                filing, shares, value_usd, issuer = consolidated[cusip]

                # Audit plan section 3: effective_lookback uses uncapped first-ever-filing.
                # T_eval anchor for lookback comparison: the first day of quarter Q
                # (i.e., we want "12+ quarters of prior data" before Q starts).
                quarter_start = date(q.year, q.month, 1)
                eff_lookback = _quarters_between(investor_first_filing, quarter_start)

                if eff_lookback < MIN_CLEAN_LOOKBACK_QUARTERS:
                    classification = "incomplete_lookback"
                    strategy: str | None = None
                else:
                    classification = "clean"
                    strategy = (
                        "full_filing_history"
                        if eff_lookback >= FULL_HISTORY_THRESHOLD_QUARTERS
                        else "truncated_to_first_filing"
                    )

                new_positions.append(
                    NewPosition(
                        investor_short_id=investor.short_id,
                        cusip=cusip,
                        name_of_issuer=issuer,
                        period_of_report=q,
                        filing_date=filing.filing_date,
                        cik=filing.cik,
                        accession_number=filing.accession_number,
                        shares=shares,
                        value_usd=value_usd,
                        lookback_completeness=classification,
                        lookback_strategy=strategy,
                        effective_lookback_quarters=eff_lookback,
                    )
                )

        # Update the ever-seen set with this quarter's holdings.
        ever_seen.update(consolidated.keys())

    return new_positions


def detect_new_positions_with_reinit_check(
    investor: Investor,
    tf_client: ThirteenFClient,
    *,
    window_start: date,
    window_end: date,
) -> list[NewPosition]:
    """Same as detect_new_positions but also catches re-initiations.

    `detect_new_positions` already excludes positions held in any prior
    quarter (those are filtered by the `ever_seen` set). However, a position
    that EXITED the 13F and then RE-ENTERED is currently classified as a
    re-buy of the same security and is therefore correctly EXCLUDED from
    the new-positions output (because it was already in `ever_seen`).

    To produce a separate `re_initiation` stratum (per audit plan section 3 and section 11
    deliverable: a separate row count per investor), use this function which
    also yields the re-initiation events as `lookback_completeness="re_initiation"`.
    """
    all_filings: list[ThirteenFFiling] = []
    for rec in investor.cik_history:
        all_filings.extend(tf_client.fetch_all_filings(rec.cik))
    if not all_filings:
        return []

    by_quarter = _filing_realm(all_filings)
    sorted_quarters = sorted(by_quarter.keys())
    investor_first_filing = investor.first_ever_filing_date

    ever_seen: set[str] = set()           # CUSIPs ever held
    last_seen: set[str] = set()           # CUSIPs in the most recent quarter
    out: list[NewPosition] = []

    for q in sorted_quarters:
        consolidated = _consolidated_holdings_for_quarter(by_quarter[q])
        this_quarter_cusips = set(consolidated.keys())
        new_to_this_quarter = this_quarter_cusips - last_seen

        for cusip in sorted(new_to_this_quarter):
            if not (window_start <= q <= window_end):
                continue
            filing, shares, value_usd, issuer = consolidated[cusip]
            quarter_start = date(q.year, q.month, 1)
            eff_lookback = _quarters_between(investor_first_filing, quarter_start)

            is_reinit = cusip in ever_seen

            if is_reinit:
                classification = "re_initiation"
                strategy: str | None = None
            elif eff_lookback < MIN_CLEAN_LOOKBACK_QUARTERS:
                classification = "incomplete_lookback"
                strategy = None
            else:
                classification = "clean"
                strategy = (
                    "full_filing_history"
                    if eff_lookback >= FULL_HISTORY_THRESHOLD_QUARTERS
                    else "truncated_to_first_filing"
                )

            out.append(
                NewPosition(
                    investor_short_id=investor.short_id,
                    cusip=cusip,
                    name_of_issuer=issuer,
                    period_of_report=q,
                    filing_date=filing.filing_date,
                    cik=filing.cik,
                    accession_number=filing.accession_number,
                    shares=shares,
                    value_usd=value_usd,
                    lookback_completeness=classification,
                    lookback_strategy=strategy,
                    effective_lookback_quarters=eff_lookback,
                )
            )

        ever_seen.update(this_quarter_cusips)
        last_seen = this_quarter_cusips

    return out
