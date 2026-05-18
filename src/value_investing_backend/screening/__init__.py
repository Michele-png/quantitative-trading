"""Dashboard-facing screening service.

Given a list of tickers and an as-of date, emit ``ScreenedRecord`` rows that
the external dashboard backend (separate repo) can persist to its database.

Two top-level entry points:

    * ``screen_tickers`` — initial screen. Hard-gates on every Phil Town
      criterion EXCEPT Margin of Safety. Used when a user adds firms to the
      dashboard for the first time.
    * ``refresh_records`` — weekly refresh. Same as above plus the MoS as a
      continuous percentage so the dashboard can show how close current price
      is to intrinsic value.

Both functions are pure: they take inputs, return records. The DB and cron
live in a separate repo.
"""

from value_investing_backend.screening.evidence import (
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
)
from value_investing_backend.screening.evidence import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_NO_DATA,
    STATUS_PARTIAL_DATA,
    STATUS_PASS,
    build_big_five_evidence,
    build_management_evidence,
)
from value_investing_backend.screening.orchestrator import (
    ScreeningOrchestrator,
    refresh_records,
    screen_tickers,
)
from value_investing_backend.screening.records import (
    HardGatePolicy,
    ScreenedRecord,
)

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "HardGatePolicy",
    "STATUS_ERROR",
    "STATUS_FAIL",
    "STATUS_NO_DATA",
    "STATUS_PARTIAL_DATA",
    "STATUS_PASS",
    "ScreenedRecord",
    "ScreeningOrchestrator",
    "build_big_five_evidence",
    "build_management_evidence",
    "refresh_records",
    "screen_tickers",
]
