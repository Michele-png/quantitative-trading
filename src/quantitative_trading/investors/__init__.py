"""Value-investor 13F audit package.

Implements the experimental design in
`docs/value_investor_criteria_audit.md` (or the project plan): for each of
N concentrated-to-diversified value investors, fetch their 13F-HR filings
from SEC EDGAR, detect first-ever new positions, and score them against
Phil Town's Rule One bar at the time of the buy decision.

Submodules
----------
investor_universe : The static catalog of audited investors with CIKs.
thirteen_f        : 13F-HR downloader and informationtable.xml parser.
cusip_resolver    : Dual-source CUSIP -> ticker resolution as of a date.
purchase_detection: Q vs first-ever-appearance delta with lookback strata.
"""
