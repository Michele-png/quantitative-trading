"""One-time discovery script: query EDGAR for the 10 audited investors' CIKs.

Run once to verify the static `INVESTORS` list in `investor_universe.py`.
Re-running is idempotent (cached EDGAR responses) but is not part of the
runtime pipeline.

Usage:
    python -m scripts.discover_investor_ciks
"""

from __future__ import annotations

import logging
from datetime import date

from quantitative_trading.config import init_env
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.investors.cik_discovery import search_filers_by_name


# Distinctive name prefixes for EDGAR's "starts-with" search semantics.
# Tuned to disambiguate from same-prefix entities (e.g., "Akre" vs "Akre Capital").
QUERIES: list[tuple[str, str]] = [
    ("munger_djco", "Daily Journal"),
    ("pabrai", "Dalal Street"),
    ("li_lu", "Himalaya Capital"),
    ("akre", "Akre Capital"),
    ("spier", "Aquamarine Capital"),
    ("nygren_harris", "Harris Associates"),
    ("russo", "Gardner Russo"),
    ("berkowitz_fairholme", "Fairholme Capital"),
    ("weitz", "Weitz Investment"),
    ("greenberg_brave_warrior", "Brave Warrior"),
]


def first_thirteen_f_filing_date(edgar: EdgarClient, cik: int) -> date | None:
    """Return the date of the entity's earliest 13F-HR filing, or None."""
    try:
        filings = edgar.list_filings(cik, forms=("13F-HR",))
    except Exception as exc:  # noqa: BLE001
        logging.warning("list_filings failed for CIK %d: %s", cik, exc)
        return None
    if not filings:
        return None
    earliest = min(filings, key=lambda f: f["filingDate"])
    return date.fromisoformat(earliest["filingDate"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_env()
    edgar = EdgarClient()

    print(f"{'short_id':>30s}  {'CIK':>10s}  {'first 13F':>11s}  entity_name")
    print("-" * 100)
    for short_id, query in QUERIES:
        matches = search_filers_by_name(edgar, query)
        if not matches:
            print(f"{short_id:>30s}  {'NONE':>10s}  {'-':>11s}  (no matches for {query!r})")
            continue
        # Top-3 candidates with their first 13F date
        for i, m in enumerate(matches[:3]):
            first_date = first_thirteen_f_filing_date(edgar, m.cik)
            first_str = first_date.isoformat() if first_date else "n/a"
            marker = "  <--" if i == 0 else "     "
            print(
                f"{(short_id if i == 0 else ''):>30s}  "
                f"{m.cik:>10d}  {first_str:>11s}  {m.entity_name}{marker}"
            )
        print()


if __name__ == "__main__":
    main()
