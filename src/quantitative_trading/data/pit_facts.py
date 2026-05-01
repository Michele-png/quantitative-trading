"""Point-in-time parsing for SEC company-facts payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
}


@dataclass(frozen=True)
class FactValue:
    """One SEC company-facts observation."""

    concept: str
    end: date
    start: date | None
    val: float
    accn: str
    fy_filing: int
    fp: str
    form: str
    filed: date
    unit: str


class PointInTimeFacts:
    """Read annual facts from an SEC company-facts payload without look-ahead."""

    def __init__(self, facts_payload: dict[str, Any]) -> None:
        self.facts_payload = facts_payload

    def latest_fiscal_year_with_data(self, alias: str, as_of: date) -> int | None:
        """Return the latest fiscal year with an annual fact filed by ``as_of``."""
        years = [
            fact.fy_filing
            for fact in self._facts_for_alias(alias)
            if fact.filed <= as_of and self._is_annual(fact)
        ]
        return max(years) if years else None

    def fiscal_year_end(self, fiscal_year: int) -> date | None:
        """Return the reported fiscal-year-end date for any fact in ``fiscal_year``."""
        ends = [
            fact.end
            for concept in CONCEPT_ALIASES
            for fact in self._facts_for_alias(concept)
            if fact.fy_filing == fiscal_year and self._is_annual(fact)
        ]
        return max(ends) if ends else None

    def get_annual(self, alias: str, fiscal_year: int, as_of: date) -> FactValue | None:
        """Return the latest annual fact for an alias/FY filed by ``as_of``."""
        candidates = [
            fact
            for fact in self._facts_for_alias(alias)
            if fact.fy_filing == fiscal_year and fact.filed <= as_of and self._is_annual(fact)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda fact: fact.filed)

    def get_annual_series(
        self,
        alias: str,
        latest_fiscal_year: int,
        n_years: int,
        as_of: date,
    ) -> dict[int, FactValue | None]:
        """Return annual facts from oldest to latest fiscal year."""
        start_year = latest_fiscal_year - n_years + 1
        return {
            fiscal_year: self.get_annual(alias, fiscal_year, as_of)
            for fiscal_year in range(start_year, latest_fiscal_year + 1)
        }

    def _facts_for_alias(self, alias: str) -> list[FactValue]:
        concepts = CONCEPT_ALIASES.get(alias, (alias,))
        facts = []
        us_gaap = self.facts_payload.get("facts", {}).get("us-gaap", {})
        for concept in concepts:
            concept_payload = us_gaap.get(concept)
            if not concept_payload:
                continue
            for unit, entries in concept_payload.get("units", {}).items():
                for entry in entries:
                    parsed = self._parse_fact(concept=concept, unit=unit, entry=entry)
                    if parsed is not None:
                        facts.append(parsed)
        return facts

    @staticmethod
    def _parse_fact(concept: str, unit: str, entry: dict[str, Any]) -> FactValue | None:
        try:
            return FactValue(
                concept=concept,
                end=date.fromisoformat(str(entry["end"])),
                start=date.fromisoformat(str(entry["start"])) if entry.get("start") else None,
                val=float(entry["val"]),
                accn=str(entry.get("accn", "")),
                fy_filing=int(entry["fy"]),
                fp=str(entry.get("fp", "")),
                form=str(entry.get("form", "")),
                filed=date.fromisoformat(str(entry["filed"])),
                unit=unit,
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_annual(fact: FactValue) -> bool:
        return fact.fp == "FY" or fact.form == "10-K"
