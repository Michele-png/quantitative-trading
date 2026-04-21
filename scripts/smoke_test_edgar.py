"""End-to-end PIT smoke test against live SEC EDGAR.

Run this any time you change the EDGAR / PIT layer to verify it still produces
correct numbers for known cases. Hits the SEC API (cached after first run).

Usage:
    python scripts/smoke_test_edgar.py
"""

from __future__ import annotations

import sys
from datetime import date

from quantitative_trading.config import init_env
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import PointInTimeFacts


CASES: list[tuple[str, int, date, dict[str, float], str]] = [
    (
        "AAPL", 2014, date(2016, 1, 1),
        {
            "revenue": 182_795,
            "net_income": 39_510,
            "stockholders_equity": 111_547,
            "operating_cash_flow": 59_713,
        },
        "Apple FY2014, fiscal year ends late September",
    ),
    (
        "MSFT", 2018, date(2020, 1, 1),
        {
            "revenue": 110_360,
            "net_income": 16_571,
            "operating_cash_flow": 43_884,
        },
        "Microsoft FY2018, fiscal year ends June 30",
    ),
    (
        "GOOGL", 2018, date(2020, 1, 1),
        {
            "revenue": 136_819,
            "net_income": 30_736,
            "operating_cash_flow": 47_971,
        },
        "Google FY2018, calendar year (Dec 31)",
    ),
    (
        "JNJ", 2017, date(2019, 1, 1),
        {"net_income": 1_300},
        "JNJ FY2017 (ends Dec 31, 2017) — should NOT be confused with "
        "JNJ FY2016 ending Jan 1, 2017 (52/53-week calendar)",
    ),
    (
        "JNJ", 2016, date(2018, 1, 1),
        {"net_income": 16_540},
        "JNJ FY2016 (ends Jan 1, 2017) — the 52/53-week case",
    ),
]


def main() -> int:
    init_env()
    client = EdgarClient()

    failures = 0
    for ticker, year, as_of, expected, note in CASES:
        facts = client.get_company_facts(client.get_cik(ticker))
        pit = PointInTimeFacts(facts)
        fy_end = pit.fiscal_year_end(year)
        print(f"=== {ticker} FY{year} (as_of={as_of}) | fy_end={fy_end} ===")
        print(f"    {note}")
        for grp, expected_m in expected.items():
            fv = pit.get_annual(grp, year, as_of=as_of)
            if fv is None:
                print(f"  {grp}: NO DATA  (expected ${expected_m:,}M)  FAIL")
                failures += 1
                continue
            actual_m = fv.val / 1e6
            tol = 0.10 if abs(expected_m) < 5_000 else 0.02
            ok = abs(actual_m - expected_m) / max(abs(expected_m), 1) < tol
            tag = "OK  " if ok else "FAIL"
            print(
                f"  {grp}: ${actual_m:>10,.0f}M  expected ${expected_m:>10,}M  "
                f"{tag}  ({fv.concept})"
            )
            if not ok:
                failures += 1
        print()

    if failures:
        print(f"FAILED: {failures} mismatch(es).")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
