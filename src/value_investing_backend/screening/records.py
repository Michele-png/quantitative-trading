"""Dashboard-ready record schema for screened firms.

Designed to be the row-shape an external service can persist to its DB. A
ScreenedRecord is a flat dataclass so it serializes trivially via ``asdict``.
The hard-gating policy lives here too, kept separate from the agent so the
academic backtest's ``is_buy_full`` semantics in
``agents/rule_one/agent.py`` stay frozen for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class HardGatePolicy:
    """Which agent checks must pass for a firm to enter the dashboard.

    Mirrors the user's stated screening criteria:

        Big 5 (5 checks) + moat + meaning + management + extras (debt-payoff
        + dilution + dividend quality)

    Margin of Safety is intentionally NOT a gate; it is reported as a
    continuous percentage on the weekly refresh so the dashboard can show
    how far current price is from intrinsic value.

    Use ``HardGatePolicy.default()`` for the standard policy. Tests override
    individual fields to verify the hard-gating logic.
    """

    require_roic: bool = True
    require_sales_growth: bool = True
    require_eps_growth: bool = True
    require_equity_growth: bool = True
    require_ocf_growth: bool = True
    require_meaning: bool = True
    require_moat: bool = True
    require_management: bool = True
    require_debt_payoff: bool = True
    require_dilution: bool = True
    require_dividend_quality: bool = True
    # Soft / informational — not part of the screen.
    require_payback_time: bool = False
    require_current_ratio: bool = False

    @classmethod
    def default(cls) -> HardGatePolicy:
        return cls()


@dataclass(frozen=True)
class ScreenedRecord:
    """One firm's evaluation flattened for dashboard ingestion.

    Schema is intentionally flat (no nested dicts of dicts) so a downstream
    ORM / SQL writer can map fields 1:1 to columns.
    """

    # Identity
    ticker: str
    as_of: date
    fiscal_year: int | None
    model: str

    # Big 5
    check_roic: bool | None
    check_sales_growth: bool | None
    check_eps_growth: bool | None
    check_equity_growth: bool | None
    check_ocf_growth: bool | None
    value_roic: float | None
    value_sales_growth: float | None
    value_eps_growth: float | None
    value_equity_growth: float | None
    value_ocf_growth: float | None
    value_current_ratio: float | None

    # Phil Town extras (soft flags at agent level, hard-gated here)
    check_debt_payoff: bool | None
    check_dilution: bool | None
    check_dividend_quality: bool | None
    value_debt_payoff_years: float | None
    value_dilution_cagr: float | None
    value_dividend_payout_ratio: float | None
    value_dividend_yield: float | None
    dividend_payout_band: str | None
    dividend_debt_funded: bool | None
    dividend_yield_trap: bool | None

    # LLM Meaning + Moat
    check_meaning: bool | None
    check_moat: bool | None
    moat_type: str | None
    rationale_meaning: str | None
    rationale_moat: str | None

    # Management sub-checks (multi-doc pipeline)
    check_management: bool | None
    check_mgmt_blame: bool | None
    check_mgmt_long_short: bool | None
    check_mgmt_clarity: bool | None
    check_mgmt_compensation: bool | None
    check_mgmt_insider: bool | None
    check_mgmt_capital_allocation: bool | None
    value_mgmt_clarity_score: float | None
    value_mgmt_long_short_ratio: float | None
    value_mgmt_insider_net_usd: float | None
    value_mgmt_capital_allocation_score: float | None
    rationale_management: str | None

    # Decision
    screen_passes: bool
    failed_gates: list[str] = field(default_factory=list)

    # Margin of Safety — only filled by ``refresh_records``
    sticker_price: float | None = None
    margin_of_safety_price: float | None = None
    current_price: float | None = None
    mos_percent: float | None = None
    """``(margin_of_safety_price - current_price) / current_price``.

    Positive = current price is below the safety price (we like it).
    Negative = current price is above the safety price (we don't).
    """

    # Provenance
    error: str | None = None
    """Short error message if evaluation failed; otherwise None."""

    # Evidence blobs — JSON-serializable dicts that let the dashboard show
    # "see evidence" drill-downs without forcing a wide-column schema.
    # Shapes are documented in ``screening/evidence.py``. Default to empty
    # dicts so error rows and tests that build ``ScreenedRecord`` directly
    # remain valid; orchestrator-built records always populate them.
    big_five_evidence: dict[str, Any] = field(default_factory=dict)
    """Per-Big-5-check yearly series, calculation method, and data-window status.

    Schema:
        ``{"schema_version": 1, "as_of": "...", "latest_fiscal_year": int|None,
        "n_years_target": 10, "checks": {<key>: {...}}, "current_ratio": {...}}``
    """

    management_evidence: dict[str, Any] = field(default_factory=dict)
    """Per-subcheck evidence + source coverage for the Management pillar.

    Schema:
        ``{"schema_version": 1, "as_of": "...", "fiscal_year": int|None,
        "model": str, "cached": bool, "bundle_hash": str|None,
        "source_coverage": {...}, "subchecks": {<key>: {...}}}``
    """
