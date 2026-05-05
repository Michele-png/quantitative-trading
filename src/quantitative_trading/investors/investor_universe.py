"""Static catalog of the 10 audited value investors with verified CIKs.

Each `Investor` carries:
    * `short_id` — stable machine identifier used in CSV outputs and joins.
    * `display_name` — human-readable label for plots and reports.
    * `cik_history` — one or more `CikRecord` entries. Most investors have a
      single CIK, but some (Pabrai, Greenberg) reorganized their filing entity
      mid-history. For those, BOTH CIKs are listed and treated as one investor
      for lookback / first-ever-appearance purposes.
    * `concentration_profile` — qualitative tag used by the §7.E "original-5"
      sensitivity and the §13.2 framing notes.
    * `notes` — caveats per §2 of the audit plan (ADR-heavy, financials-heavy,
      multi-PM, etc.).

All CIKs and first-13F dates were verified against EDGAR via
`scripts.discover_investor_ciks` on 2026-05-01.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CikRecord:
    """A single CIK, plus the date range during which the investor filed under it.

    `effective_until` is the earliest date *after which* this CIK was no longer
    the active filing entity (i.e., the next CIK takes over). None means
    currently-active. Date ranges are non-overlapping for a single investor
    (verified clean handoffs in `scripts/discover_investor_ciks.py`).
    """

    cik: int
    legal_entity_name: str
    first_filing_date: date  # date of earliest 13F-HR under this CIK
    effective_until: date | None = None


@dataclass(frozen=True)
class Investor:
    """One of the 10 audited value investors."""

    short_id: str
    display_name: str
    cik_history: tuple[CikRecord, ...]
    concentration_profile: str  # "very_concentrated" | "concentrated" | "diversified"
    is_original_five: bool      # used by §7.E sensitivity 1
    notes: str

    @property
    def all_ciks(self) -> tuple[int, ...]:
        return tuple(r.cik for r in self.cik_history)

    @property
    def first_ever_filing_date(self) -> date:
        """Earliest 13F filing date across all CIKs for this investor.

        This is the anchor used by `purchase_detection` to compute
        `effective_lookback_quarters = T_eval - first_ever_filing_date`.
        """
        return min(r.first_filing_date for r in self.cik_history)


# ----------------------------------------------------------------- Catalog

INVESTORS: tuple[Investor, ...] = (
    Investor(
        short_id="munger_djco",
        display_name="Charlie Munger / Daily Journal Corp",
        cik_history=(
            CikRecord(
                cik=783412,
                legal_entity_name="Daily Journal Corp",
                first_filing_date=date(2014, 2, 11),
            ),
        ),
        concentration_profile="very_concentrated",
        is_original_five=True,
        notes=(
            "Munger died Nov 2023; sample ends there. ~5 holdings. DJCO did not "
            "file 13F until 2014-Q1, so all in-window buys are "
            "truncated_to_first_filing until ~2024-Q1."
        ),
    ),
    Investor(
        short_id="pabrai",
        display_name="Mohnish Pabrai (Personal + Dalal Street LLC)",
        cik_history=(
            CikRecord(
                cik=1173334,
                legal_entity_name="Pabrai Mohnish",
                first_filing_date=date(2005, 2, 15),
                effective_until=date(2012, 2, 14),  # last filing under this CIK
            ),
            CikRecord(
                cik=1549575,
                legal_entity_name="Dalal Street, LLC",
                first_filing_date=date(2012, 5, 11),
            ),
        ),
        concentration_profile="concentrated",
        is_original_five=True,
        notes=(
            "Two CIKs: personal filings 2005-2012, then Dalal Street LLC "
            "from 2012-05. Clean handoff (3-month gap). Concentrated, "
            "value/special-sit. Effective first-ever filing 2005 once merged."
        ),
    ),
    Investor(
        short_id="li_lu",
        display_name="Li Lu / Himalaya Capital Management",
        cik_history=(
            CikRecord(
                cik=1709323,
                legal_entity_name="Himalaya Capital Management LLC",
                first_filing_date=date(2017, 6, 14),
            ),
        ),
        concentration_profile="very_concentrated",
        is_original_five=True,
        notes=(
            "Himalaya only began 13F filings 2017-Q2. With headline window "
            "starting 2017-Q1, Li Lu has zero clean lookback for early-window "
            "buys; reaches `clean` status only ~2020-Q2 (3y after first filing). "
            "Heavy ADR exposure (BABA, MU). Threshold-crossing risk in 2017-2018."
        ),
    ),
    Investor(
        short_id="akre",
        display_name="Chuck Akre / Akre Capital Management",
        cik_history=(
            CikRecord(
                cik=1112520,
                legal_entity_name="Akre Capital Management LLC",
                first_filing_date=date(2001, 8, 9),
            ),
        ),
        concentration_profile="concentrated",
        is_original_five=True,
        notes=(
            "24+ years of filing history. Most active of the original 5; "
            "~30 holdings. Always full_filing_history throughout 2017-2024 window."
        ),
    ),
    Investor(
        short_id="spier",
        display_name="Guy Spier / Aquamarine Capital Management",
        cik_history=(
            CikRecord(
                cik=1404599,
                legal_entity_name="Aquamarine Capital Management, LLC",
                first_filing_date=date(2015, 2, 27),
            ),
        ),
        concentration_profile="concentrated",
        is_original_five=True,
        notes=(
            "First 13F 2015-Q1 (later than initially assumed). 2017-Q1 buys "
            "have 8 quarters lookback (incomplete). Spier reaches `clean` from "
            "2018-Q1+. Always truncated_to_first_filing in window."
        ),
    ),
    Investor(
        short_id="nygren_harris",
        display_name="Bill Nygren / Harris Associates LP (Oakmark Funds)",
        cik_history=(
            CikRecord(
                cik=813917,
                legal_entity_name="Harris Associates L P",
                first_filing_date=date(1999, 5, 12),
            ),
        ),
        concentration_profile="diversified",
        is_original_five=False,
        notes=(
            "Harris files ONE 13F covering all Oakmark funds (~$100B AUM, "
            "~100 holdings). The 'Nygren' attribution is an approximation; "
            "this slot is empirically 'Harris Associates as a value house' — "
            "decisions span multiple PMs across multiple sleeves. 26 years "
            "of history."
        ),
    ),
    Investor(
        short_id="russo",
        display_name="Tom Russo / Gardner Russo & Quinn",
        cik_history=(
            CikRecord(
                cik=860643,
                legal_entity_name="Gardner Russo & Quinn LLC",
                first_filing_date=date(1999, 6, 8),
            ),
        ),
        concentration_profile="concentrated",
        is_original_five=False,
        notes=(
            "Firm rebranded over time (Gardner Russo & Gardner -> & Quaker -> "
            "& Quinn) but kept CIK 860643. Heavy in European consumer-staples "
            "ADRs (Nestle, Heineken, Pernod) which fall in the non-evaluable "
            "bucket per audit plan section 5. Will contribute fewer evaluable rows than "
            "headline AUM suggests. 26 years of history."
        ),
    ),
    Investor(
        short_id="berkowitz_fairholme",
        display_name="Bruce Berkowitz / Fairholme Capital Management",
        cik_history=(
            CikRecord(
                cik=1056831,
                legal_entity_name="Fairholme Capital Management LLC",
                first_filing_date=date(1999, 5, 13),
            ),
        ),
        concentration_profile="concentrated",
        is_original_five=False,
        notes=(
            "Last decade dominated by financials (BAC, AIG) and holdcos "
            "(JOE, Sears Holdings) - most positions land in non-evaluable "
            "per audit plan section 5. Few evaluable rows expected. 26 years of history."
        ),
    ),
    Investor(
        short_id="weitz",
        display_name="Wally Weitz / Weitz Investment Management",
        cik_history=(
            CikRecord(
                cik=883965,
                legal_entity_name="Weitz Investment Management, Inc.",
                first_filing_date=date(1999, 5, 7),
            ),
        ),
        concentration_profile="diversified",
        is_original_five=False,
        notes=(
            "Clean filer, ~30-40 holdings, classic value/quality. Likely "
            "the cleanest contributor of the 5 additions. 26 years of history."
        ),
    ),
    Investor(
        short_id="greenberg_brave_warrior",
        display_name="Glenn Greenberg / Brave Warrior (Capital Inc + Advisors LLC)",
        cik_history=(
            CikRecord(
                cik=789920,
                legal_entity_name="Brave Warrior Capital, Inc.",
                first_filing_date=date(1999, 5, 17),
                effective_until=date(2012, 5, 15),
            ),
            CikRecord(
                cik=1553733,
                legal_entity_name="Brave Warrior Advisors, LLC",
                first_filing_date=date(2012, 7, 9),
            ),
        ),
        concentration_profile="very_concentrated",
        is_original_five=False,
        notes=(
            "Two CIKs from a 2012 entity reorganization (Capital Inc -> "
            "Advisors LLC). Clean ~2-month handoff. ~10 holdings, deep-value "
            "in the Munger/Klarman mold. (Picked over David Rolfe/Wedgewood, "
            "who has drifted growth-tilted; trivially swappable.) Effective "
            "first-ever filing 1999 once merged."
        ),
    ),
)


# ----------------------------------------------------------------- Helpers

def get_investor(short_id: str) -> Investor:
    for inv in INVESTORS:
        if inv.short_id == short_id:
            return inv
    raise KeyError(f"No investor with short_id {short_id!r}; "
                   f"available: {[i.short_id for i in INVESTORS]}")


def all_ciks() -> dict[int, str]:
    """Flat map of {cik -> short_id} across all investors. For 13F download."""
    out: dict[int, str] = {}
    for inv in INVESTORS:
        for rec in inv.cik_history:
            out[rec.cik] = inv.short_id
    return out


def original_five() -> tuple[Investor, ...]:
    """The original-5 subset used by audit plan section 7.E sensitivity 1."""
    return tuple(i for i in INVESTORS if i.is_original_five)
