"""Point-in-time query layer over SEC EDGAR XBRL company facts.

Semantics
---------
For (concept, fiscal_year, as_of), the value as it would have appeared to an
investor on `as_of` is the value reported in the *latest* filing whose `filed`
date is <= `as_of` for the company's official fiscal year `fiscal_year`. This
naturally includes restatements that were public before `as_of` and excludes
those made after.

How a fiscal year is identified
-------------------------------
The XBRL `fy` field on each fact is the fiscal year of the *FILING* that
contains the fact. A 10-K with `fy=2017` may contain comparative data for
prior fiscal years tagged with the same `fy=2017`. Naively filtering on
`fy == Y` therefore mixes "current year" with "comparative" values. Naively
filtering on `end.year == Y` fails for companies with 52/53-week fiscal years
that cross calendar boundaries (e.g., JNJ's FY2016 ends Jan 1, 2017).

The robust algorithm:
    1. Look at all entries with SEC `fy == Y` (i.e., from the company's FY-Y
       10-K). Within those, the "current year" entries are those with the
       latest `end` date — that latest end is the company's official FY-Y
       year-end date.
    2. Across all 10-K/10-K/A filings, find every entry whose `end` matches
       that exact FY-Y year-end date. This collects the original FY-Y value
       plus any restated comparatives in FY-(Y+1), FY-(Y+2), … filings.
    3. Filter by `filed <= as_of`. Among the survivors, return the entry with
       the latest `filed` (most recent restatement visible at `as_of`).
    4. For flow concepts, also require ~365-day period length to exclude
       quarterly subperiods that share the same `end` date.

Concept categorization
----------------------
* Flow concepts (income statement, cash flow): have `start` and `end`.
  Annual values require ~365-day periods (not quarters, not 5-year tables).
* Snapshot concepts (balance sheet): have `end` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class FactValue:
    """A single XBRL fact value, with provenance."""

    concept: str
    end: date
    val: float
    accn: str
    fy_filing: int  # fiscal year of the FILING containing this entry
    fp: str  # "FY", "Q1", "Q2", "Q3", "Q4"
    form: str  # "10-K", "10-Q", "10-K/A", ...
    filed: date
    unit: str  # "USD", "shares", "USD/shares", ...
    start: date | None = None  # None for balance-sheet snapshots


# Ordered concept fallbacks per Big 5 input. First-match-wins per fiscal year.
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
        "IncomeLossFromContinuingOperationsPerBasicShare",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "shares_outstanding_dei": ["EntityCommonStockSharesOutstanding"],
    "weighted_avg_shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
}


# Flow vs. snapshot — flow concepts require a ~365-day period to be considered annual.
_FLOW_GROUPS: set[str] = {
    "revenue",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "operating_cash_flow",
    "capex",
    "weighted_avg_shares_diluted",
}


# Per-concept expected unit. None means accept any.
_UNIT_FOR: dict[str, str] = {
    "revenue": "USD",
    "net_income": "USD",
    "eps_basic": "USD/shares",
    "eps_diluted": "USD/shares",
    "stockholders_equity": "USD",
    "long_term_debt": "USD",
    "operating_cash_flow": "USD",
    "capex": "USD",
    "current_assets": "USD",
    "current_liabilities": "USD",
    "total_assets": "USD",
    "shares_outstanding_dei": "shares",
    "weighted_avg_shares_diluted": "shares",
}


# Which XBRL taxonomy each concept lives in.
_TAXONOMY_FOR: dict[str, str] = {
    "shares_outstanding_dei": "dei",
}


# Period-length tolerance for "annual" flow concepts (covers 52/53-week years).
_ANNUAL_DAYS_MIN = 350
_ANNUAL_DAYS_MAX = 380


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _is_10k(form: str) -> bool:
    return form.startswith("10-K")


class PointInTimeFacts:
    """Wraps a company's raw XBRL facts and answers PIT queries."""

    def __init__(self, facts_payload: dict[str, Any]) -> None:
        self._raw = facts_payload
        self._cik = facts_payload.get("cik")
        self._entity = facts_payload.get("entityName", "")
        self._fy_end_cache: dict[int, date | None] = {}

    @property
    def entity_name(self) -> str:
        return self._entity

    @property
    def cik(self) -> int | None:
        return self._cik

    def _iter_concept_entries(
        self, concept: str, taxonomy: str = "us-gaap"
    ) -> list[tuple[str, dict[str, Any]]]:
        bucket = (
            self._raw.get("facts", {})
            .get(taxonomy, {})
            .get(concept, {})
            .get("units", {})
        )
        return [
            (unit_key, entry)
            for unit_key, entries in bucket.items()
            for entry in entries
        ]

    def _build_value(
        self, *, concept: str, unit: str, entry: dict[str, Any]
    ) -> FactValue:
        return FactValue(
            concept=concept,
            end=_parse_date(entry["end"]),
            start=_parse_date(entry["start"]) if "start" in entry else None,
            val=float(entry["val"]),
            accn=entry["accn"],
            fy_filing=int(entry["fy"]),
            fp=str(entry["fp"]),
            form=str(entry["form"]),
            filed=_parse_date(entry["filed"]),
            unit=unit,
        )

    def fiscal_year_end(self, fiscal_year: int) -> date | None:
        """Return the company's official period-end date for fiscal year `fiscal_year`.

        The FY-Y 10-K is identified by SEC `fy == fiscal_year`. The year-end
        date is the latest `end` among us-gaap FY entries in those filings,
        with the additional constraint that flow entries must have a ~365-day
        period (so partial-period or 5-year-anchor data can't pollute the max).

        DEI taxonomy is excluded because entries like
        EntityCommonStockSharesOutstanding use the cover-page-as-of date
        (typically a week or two after the actual fiscal year end), not the
        fiscal year end itself.
        """
        if fiscal_year in self._fy_end_cache:
            return self._fy_end_cache[fiscal_year]

        latest_end: date | None = None
        for _concept, payload in self._raw.get("facts", {}).get("us-gaap", {}).items():
            for _unit_key, entries in payload.get("units", {}).items():
                for entry in entries:
                    if entry.get("fp") != "FY":
                        continue
                    if not _is_10k(entry.get("form", "")):
                        continue
                    if int(entry.get("fy", -1)) != fiscal_year:
                        continue
                    end = _parse_date(entry["end"])
                    start_str = entry.get("start")
                    if start_str is not None:
                        # Flow entry: must be a ~365-day period to anchor the FY end.
                        if not (
                            _ANNUAL_DAYS_MIN
                            <= (end - _parse_date(start_str)).days
                            <= _ANNUAL_DAYS_MAX
                        ):
                            continue
                    if latest_end is None or end > latest_end:
                        latest_end = end

        self._fy_end_cache[fiscal_year] = latest_end
        return latest_end

    def get_annual(
        self,
        concept_group: str,
        fiscal_year: int,
        as_of: date,
    ) -> FactValue | None:
        """Return the FY value for the concept group, as visible at `as_of`.

        See module docstring for the algorithm. In short:
            1. Resolve the company's official FY-Y year-end date.
            2. Collect entries whose `end` matches that date (across all 10-K
               and 10-K/A filings — including restated comparatives).
            3. Apply unit, period-length, form, and `filed <= as_of` filters.
            4. Return the latest-filed survivor.
        """
        if concept_group not in CONCEPTS:
            raise ValueError(f"Unknown concept group: {concept_group!r}")

        fy_end = self.fiscal_year_end(fiscal_year)
        if fy_end is None:
            return None

        taxonomy = _TAXONOMY_FOR.get(concept_group, "us-gaap")
        expected_unit = _UNIT_FOR.get(concept_group)
        is_flow = concept_group in _FLOW_GROUPS

        for concept in CONCEPTS[concept_group]:
            candidates: list[FactValue] = []
            for unit_key, entry in self._iter_concept_entries(concept, taxonomy):
                if expected_unit and unit_key != expected_unit:
                    continue
                if entry.get("fp") != "FY":
                    continue
                form = entry.get("form", "")
                if not _is_10k(form):
                    continue
                end = _parse_date(entry["end"])
                if end != fy_end:
                    continue
                if is_flow:
                    start_str = entry.get("start")
                    if start_str is None:
                        continue
                    period_days = (end - _parse_date(start_str)).days
                    if not (_ANNUAL_DAYS_MIN <= period_days <= _ANNUAL_DAYS_MAX):
                        continue
                filed = _parse_date(entry["filed"])
                if filed > as_of:
                    continue
                candidates.append(
                    self._build_value(concept=concept, unit=unit_key, entry=entry)
                )
            if candidates:
                return max(candidates, key=lambda fv: fv.filed)

        return None

    def get_annual_series(
        self,
        concept_group: str,
        last_fiscal_year: int,
        n_years: int,
        as_of: date,
    ) -> dict[int, FactValue | None]:
        """Return {fiscal_year -> FactValue or None} for n_years up to last_fiscal_year."""
        return {
            fy: self.get_annual(concept_group, fy, as_of)
            for fy in range(last_fiscal_year - n_years + 1, last_fiscal_year + 1)
        }

    def latest_fiscal_year_with_data(
        self,
        concept_group: str,
        as_of: date,
    ) -> int | None:
        """Find the most recent fiscal year for which the concept has been filed by as_of.

        Walks fy backwards from a generous upper bound; returns the first year
        with data. Bounded by the latest `fy` value seen anywhere in the facts.
        """
        if concept_group not in CONCEPTS:
            raise ValueError(f"Unknown concept group: {concept_group!r}")
        max_fy = self._max_fy_seen(as_of)
        if max_fy is None:
            return None
        for fy in range(max_fy, max_fy - 30, -1):
            if self.get_annual(concept_group, fy, as_of) is not None:
                return fy
        return None

    def _max_fy_seen(self, as_of: date) -> int | None:
        out: int | None = None
        for taxonomy in ("us-gaap", "dei"):
            for _c, payload in self._raw.get("facts", {}).get(taxonomy, {}).items():
                for _u, entries in payload.get("units", {}).items():
                    for entry in entries:
                        if entry.get("fp") != "FY":
                            continue
                        if not _is_10k(entry.get("form", "")):
                            continue
                        if _parse_date(entry["filed"]) > as_of:
                            continue
                        fy = int(entry.get("fy", -1))
                        if fy > 0 and (out is None or fy > out):
                            out = fy
        return out
