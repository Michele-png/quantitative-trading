"""Evidence builders for ``ScreenedRecord``.

The dashboard needs to show, for each Big 5 check and each Management
sub-check, *why* the result is what it is — yearly data points, source
documents available, missing-source explanations, etc. The screening
backend already computes that information internally; this module turns it
into a JSON-friendly shape that lives alongside the flat boolean columns
on ``ScreenedRecord``.

Two top-level builders:

    * ``build_big_five_evidence`` — yearly series, calculation method,
      relaxed-window flag for each Big 5 metric.
    * ``build_management_evidence`` — per-subcheck status + evidence +
      top-level source coverage for the multi-document pipeline.

Status vocabulary (used as ``status`` in every check/subcheck dict):

    * ``"pass"``   — assessment was made and the gate passed.
    * ``"fail"``   — assessment was made and the gate failed.
    * ``"no_data"``— assessment could NOT be made because the required
                     source data was unavailable. NOT the same as fail.
    * ``"partial_data"`` — assessment was made on degraded source coverage
                     (e.g. a sub-check that benefits from two sources only
                     had one). The dashboard should warn rather than treat
                     it as a clean pass/fail. Reserved for Management;
                     Big 5 uses ``relaxed_window`` instead so the underlying
                     pass/fail signal stays visible.
    * ``"error"``  — runtime exception during evaluation (e.g. LLM 429,
                     network error, JSON parse failure).

The dashboard maps these directly to colored badges so users can tell a
real negative signal apart from missing data.
"""

from __future__ import annotations

from typing import Any

from value_investing_backend.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from value_investing_backend.agents.rule_one.four_ms_llm import FourMsResult
from value_investing_backend.agents.rule_one.management_llm import (
    EVIDENCE_SCHEMA_VERSION,
    OUTCOME_ERROR,
    OUTCOME_HARD_FAIL,
    OUTCOME_NEUTRAL,
    OUTCOME_NO_DATA,
    OUTCOME_PARTIAL_DATA,
    OUTCOME_PASS,
    SHAREHOLDER_LETTER_FROM_ARCHIVE,
    SHAREHOLDER_LETTER_FROM_BUSINESS_DESCRIPTION,
    SHAREHOLDER_LETTER_FROM_DEF14A,
    ManagementResult,
    SourceCoverage,
    SubCheck,
)

# Bumped to 2 when management evidence grew the explicit ``outcome``
# field per sub-check and the ``decision`` aggregate block. Big 5
# evidence still uses the legacy schema_version=1 shape; the dashboard
# tolerates a mixed-version blob during the rollout.
SCHEMA_VERSION = 1

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NO_DATA = "no_data"
STATUS_PARTIAL_DATA = "partial_data"
STATUS_NEUTRAL = "neutral"
STATUS_ERROR = "error"

# The shareholder_letter coverage slot carries a ``from`` tag whose
# value identifies the source. Only def14a and archive-backed
# shareholder_letter docs are treated as the "real" letter material;
# everything else (10-K Item 1 fallback, annual-report container
# without an extracted letter section) is degraded for LongShort and
# Clarity even though the slot reports ``available=true``.
_REAL_SHAREHOLDER_LETTER_PREFIXES: tuple[str, ...] = (
    SHAREHOLDER_LETTER_FROM_DEF14A,
    f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:shareholder_letter",
    f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:proxy_letter",
)
_DEGRADED_SHAREHOLDER_LETTER_PREFIXES: tuple[str, ...] = (
    SHAREHOLDER_LETTER_FROM_BUSINESS_DESCRIPTION,
    "10-k_item1_fallback",
    f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:annual_report",
    f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:ten_k_item1_fallback",
    f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:ir_material",
)


def _shareholder_letter_quality(coverage: dict[str, Any] | None) -> str:
    """Return ``"real" | "degraded" | "missing"`` for the letter slot.

    "Real" means a CEO-style shareholder letter (def14a proxy letter or
    archive-validated shareholder_letter / proxy_letter doc). "Degraded"
    means the slot is filled, but the underlying source is Item 1
    Business Description or a broad annual-report container without an
    extracted letter section; LongShort and Clarity should treat that
    as ``partial_data``. "Missing" means the slot is empty.
    """
    if not coverage:
        return "missing"
    letter = coverage.get("shareholder_letter")
    if not isinstance(letter, dict) or not letter.get("available"):
        return "missing"
    source = (letter.get("from") or "").lower()
    if not source:
        # Slot reports ``available`` but the bundler didn't tag the
        # source — treat as degraded so the dashboard doesn't silently
        # upgrade unknown provenance to a clean letter signal.
        return "degraded"
    if any(source.startswith(p.lower()) for p in _REAL_SHAREHOLDER_LETTER_PREFIXES):
        return "real"
    if any(source.startswith(p.lower()) for p in _DEGRADED_SHAREHOLDER_LETTER_PREFIXES):
        return "degraded"
    return "degraded"


# --------------------------------------------------------------------------
# Big 5
# --------------------------------------------------------------------------


# Maps each Big 5 metric key to the calculation method label the dashboard
# should display. ROIC uses an "average + most-recent floor" gate, the
# growth metrics use first-to-last CAGR over whatever fiscal years the SEC
# XBRL layer surfaces (relaxed to a 5y minimum when 10y are not available).
_BIG_FIVE_METHODS: dict[str, str] = {
    "roic": "avg_and_most_recent_floor",
    "sales_growth": "cagr_first_to_last",
    "eps_growth": "cagr_first_to_last_split_adjusted",
    "equity_growth": "cagr_first_to_last",
    "ocf_growth": "cagr_first_to_last",
    "current_ratio": "latest_fy_ratio",
}

_BIG_FIVE_LABELS: dict[str, str] = {
    "roic": "ROIC",
    "sales_growth": "Sales growth",
    "eps_growth": "EPS growth",
    "equity_growth": "Equity growth",
    "ocf_growth": "OCF growth",
    "current_ratio": "Current ratio",
}

# Per-metric units. ``value_unit`` describes the headline metric value
# (e.g. ROIC average is a decimal fraction); ``yearly_unit`` describes the
# underlying per-fiscal-year series (e.g. EPS yearly values are dollars per
# share, revenue yearly values are dollars). The dashboard uses these to
# format both columns correctly — the previous single ``unit`` field caused
# raw EPS/dollar values to be rendered as percentages.
_BIG_FIVE_VALUE_UNITS: dict[str, str] = {
    "roic": "fraction",
    "sales_growth": "fraction",
    "eps_growth": "fraction",
    "equity_growth": "fraction",
    "ocf_growth": "fraction",
    "current_ratio": "ratio_decimal",
}

_BIG_FIVE_YEARLY_UNITS: dict[str, str] = {
    "roic": "fraction",
    "sales_growth": "currency",
    "eps_growth": "currency_per_share",
    "equity_growth": "currency",
    "ocf_growth": "currency",
    "current_ratio": "ratio_decimal",
}


def _metric_status(metric: MetricResult, *, has_full_window: bool) -> str:
    """Pass / fail / partial_data / no_data status from a Big 5 ``MetricResult``.

    ``has_full_window`` is True when the metric had every fiscal year of
    its target window populated. When the metric value is computed but
    the window is short (relaxed-window CAGR) or the data came from a
    fallback source, the dashboard should treat it as ``partial_data``
    rather than ``fail`` — the underlying signal might or might not
    cross Phil Town's bar, but we don't have enough evidence to call it.
    """
    if metric.value is None:
        return STATUS_NO_DATA
    if metric.passes:
        return STATUS_PASS
    if not has_full_window:
        return STATUS_PARTIAL_DATA
    return STATUS_FAIL


def _yearly_points(series: dict[int, float | None]) -> list[dict[str, Any]]:
    """Sort ``{fy: val}`` into ``[{fiscal_year, value}]`` for stable JSON."""
    return [
        {"fiscal_year": int(fy), "value": (None if v is None else float(v))}
        for fy, v in sorted(series.items())
    ]


def _metric_evidence(
    *,
    key: str,
    metric: MetricResult,
    n_years_target: int,
) -> dict[str, Any]:
    yearly = _yearly_points(metric.series)
    years_with_data = [
        p["fiscal_year"] for p in yearly if p["value"] is not None
    ]
    missing = [
        p["fiscal_year"] for p in yearly if p["value"] is None
    ]
    years_used = len(years_with_data)
    # The current-ratio side check looks only at the latest FY so the
    # "relaxed window" concept doesn't apply.
    is_window_metric = key != "current_ratio"
    has_full_window = (not is_window_metric) or years_used >= n_years_target
    relaxed = (
        is_window_metric and years_used > 0 and years_used < n_years_target
    )
    interior_gaps = _interior_gaps(years_with_data, missing)

    value_unit = _BIG_FIVE_VALUE_UNITS.get(key, "fraction")
    yearly_unit = _BIG_FIVE_YEARLY_UNITS.get(key, value_unit)
    out: dict[str, Any] = {
        "label": _BIG_FIVE_LABELS.get(key, metric.name or key),
        "status": _metric_status(metric, has_full_window=has_full_window),
        "value": (None if metric.value is None else float(metric.value)),
        "threshold": float(metric.threshold),
        "passes": bool(metric.passes),
        "decision_grade": bool(getattr(metric, "decision_grade", metric.passes)),
        "data_source": getattr(metric, "data_source", "sec_xbrl"),
        "rationale": metric.rationale,
        "calculation_method": _BIG_FIVE_METHODS.get(key, "unknown"),
        # ``unit`` is preserved for backwards compatibility with older
        # dashboard builds; new code should prefer ``value_unit`` and
        # ``yearly_unit`` so headline ratios and per-year underlyings can
        # be formatted differently (e.g. ROIC% headline + raw EPS yearly).
        "unit": value_unit,
        "value_unit": value_unit,
        "yearly_unit": yearly_unit,
        "yearly": yearly,
        "years_used": years_used,
        "years_with_data": years_with_data,
        "missing_fiscal_years": missing,
        "interior_gaps": interior_gaps,
        "n_years_target": n_years_target,
        "relaxed_window": relaxed,
    }
    if relaxed:
        out["relaxed_reason"] = (
            f"Only {years_used} of the target {n_years_target} fiscal "
            "years have usable XBRL data. Phil Town's rule explicitly "
            "wants a full 10-year window; we still surface the CAGR / "
            "ROIC computed over the available history, but the gate "
            "stays open (PARTIAL DATA) until the missing years catch up."
        )
    elif interior_gaps:
        out["relaxed_reason"] = (
            f"Window has interior gaps ({', '.join(map(str, interior_gaps))}) "
            "— treat the headline value as approximate."
        )
    return out


def _interior_gaps(
    years_with_data: list[int],
    missing_years: list[int],
) -> list[int]:
    """Return the missing years that fall *inside* the populated window.

    A gap at the start of the window (typical for young issuers / late
    XBRL adoption) is not an "interior" gap — it just means the company
    hasn't been around for the full 10y. An interior gap, by contrast,
    suggests a tagging issue or a restatement that dropped a single
    year, and the dashboard should warn the user separately.
    """
    if not years_with_data or not missing_years:
        return []
    earliest_present = min(years_with_data)
    return sorted(fy for fy in missing_years if fy > earliest_present)


def build_big_five_evidence(big_five: BigFiveResult | None) -> dict[str, Any]:
    """Serialize a ``BigFiveResult`` into the dashboard evidence blob."""
    if big_five is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "as_of": None,
            "latest_fiscal_year": None,
            "n_years_target": 10,
            "checks": {},
            "current_ratio": None,
        }

    n_years_target = big_five.n_years_required
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": big_five.as_of.isoformat() if big_five.as_of else None,
        "latest_fiscal_year": big_five.latest_fiscal_year,
        "n_years_target": n_years_target,
        "checks": {
            "roic": _metric_evidence(
                key="roic", metric=big_five.roic,
                n_years_target=n_years_target,
            ),
            "sales_growth": _metric_evidence(
                key="sales_growth", metric=big_five.sales_growth,
                n_years_target=n_years_target,
            ),
            "eps_growth": _metric_evidence(
                key="eps_growth", metric=big_five.eps_growth,
                n_years_target=n_years_target,
            ),
            "equity_growth": _metric_evidence(
                key="equity_growth", metric=big_five.equity_growth,
                n_years_target=n_years_target,
            ),
            "ocf_growth": _metric_evidence(
                key="ocf_growth", metric=big_five.ocf_growth,
                n_years_target=n_years_target,
            ),
        },
        "current_ratio": _metric_evidence(
            key="current_ratio", metric=big_five.current_ratio,
            n_years_target=n_years_target,
        ),
    }


# --------------------------------------------------------------------------
# Management
# --------------------------------------------------------------------------


# Per-subcheck source requirements. ``any_of_required=True`` means that
# any one source in ``required_sources`` is enough to compute the check;
# missing the rest only downgrades to ``partial_data``. Otherwise, every
# listed source must be present or the check is ``no_data``.
#
# ``shareholder_letter_must_be_real`` means the sub-check is downgraded
# to ``partial_data`` when the shareholder-letter slot is filled by
# Item 1 / annual-report-container fallback rather than a real letter.
# This stops broad regulatory disclosure text from being scored as if
# it were a Buffett/Bezos owner letter.
_SUBCHECK_SPECS: dict[str, dict[str, Any]] = {
    "blame": {
        "label": "Blame test",
        "required_sources": ["transcripts"],
        "any_of_required": False,
    },
    "long_short": {
        "label": "Long-term focus",
        "required_sources": ["transcripts", "shareholder_letter"],
        "any_of_required": True,
        "shareholder_letter_must_be_real": True,
    },
    "clarity": {
        "label": "Plain-English clarity",
        "required_sources": ["shareholder_letter", "mda"],
        "any_of_required": True,
        "shareholder_letter_must_be_real": True,
    },
    "compensation": {
        "label": "Compensation alignment",
        "required_sources": ["def14a_compensation"],
        "any_of_required": False,
    },
    "insider": {
        "label": "Insider alignment",
        "required_sources": ["form4"],
        "any_of_required": False,
    },
    "capital_allocation": {
        "label": "Capital allocation",
        # The check needs at least one qualitative source (MD&A or
        # shareholder letter) to evaluate the LLM side; ``any_of`` keeps
        # the gate workable when one source is missing. MD&A counts as
        # primary signal here, so the Item 1 fallback is acceptable.
        "required_sources": ["mda", "shareholder_letter"],
        "any_of_required": True,
    },
}


def _status_from_outcome(outcome: str) -> str:
    """Map a ``SubCheckOutcome`` to the dashboard's status vocabulary.

    Sub-check ``outcome`` is the source of truth on new runs. Older
    cached results without an explicit outcome fall through to the
    legacy pass/fail derivation in ``_derive_subcheck_status``.
    """
    if outcome == OUTCOME_PASS:
        return STATUS_PASS
    if outcome == OUTCOME_HARD_FAIL:
        return STATUS_FAIL
    if outcome == OUTCOME_NEUTRAL:
        return STATUS_NEUTRAL
    if outcome == OUTCOME_PARTIAL_DATA:
        return STATUS_PARTIAL_DATA
    if outcome == OUTCOME_NO_DATA:
        return STATUS_NO_DATA
    if outcome == OUTCOME_ERROR:
        return STATUS_ERROR
    return STATUS_NO_DATA


def _coverage_lookup(
    coverage: dict[str, Any] | None, source: str,
) -> bool:
    if not coverage:
        return False
    info = coverage.get(source)
    if not isinstance(info, dict):
        return False
    return bool(info.get("available", False))


def _derive_subcheck_status(
    *,
    sub: SubCheck,
    spec: dict[str, Any],
    coverage: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    """Return ``(status, missing_sources, degraded_sources)`` for a sub-check.

    Resolution order:

    1. Operational ``error`` outcome (or legacy ``error`` flag in
       details) → ``STATUS_ERROR``.
    2. Explicit ``outcome`` field on the SubCheck wins when set: it is
       the source of truth on new runs, and ``no_data`` / ``hard_fail``
       beat any source-coverage-derived status.
    3. Source coverage gates: if a required source is missing the
       status degrades to ``no_data``; if only a degraded source is
       available (e.g. Item 1 fallback for the shareholder letter slot)
       the status degrades to ``partial_data`` even when the underlying
       LLM signal would have passed.
    4. Fall back to pass/fail from the legacy ``passes`` boolean.

    Coverage is optional so this stays back-compatible with cached
    Management results that pre-date the ``SourceCoverage`` field.
    """
    legacy_error = bool(sub.details.get("error"))
    explicit_outcome = sub.outcome

    if legacy_error or explicit_outcome == OUTCOME_ERROR:
        return STATUS_ERROR, [], []

    required = spec.get("required_sources", []) or []
    if coverage is None:
        if explicit_outcome is not None:
            return _status_from_outcome(explicit_outcome), [], []
        return (STATUS_PASS if sub.passes else STATUS_FAIL), [], []

    available = [s for s in required if _coverage_lookup(coverage, s)]
    missing = [s for s in required if not _coverage_lookup(coverage, s)]

    degraded: list[str] = []
    letter_quality = _shareholder_letter_quality(coverage)
    if (
        spec.get("shareholder_letter_must_be_real", False)
        and "shareholder_letter" in available
        and letter_quality == "degraded"
    ):
        degraded.append("shareholder_letter")

    if spec.get("any_of_required", False):
        # Sources that count as "really available" — degraded sources
        # are present but downgraded. If every available source is
        # degraded, treat the slot as partial rather than missing.
        non_degraded = [s for s in available if s not in degraded]
        if not available:
            return STATUS_NO_DATA, missing, degraded
        if missing or not non_degraded:
            base_status = STATUS_PARTIAL_DATA
            if explicit_outcome in {OUTCOME_HARD_FAIL, OUTCOME_NEUTRAL}:
                return _status_from_outcome(explicit_outcome), missing, degraded
            return base_status, missing, degraded
    elif missing:
        return STATUS_NO_DATA, missing, degraded

    if explicit_outcome is not None:
        status = _status_from_outcome(explicit_outcome)
        if degraded and status == STATUS_PASS:
            status = STATUS_PARTIAL_DATA
        return status, missing, degraded

    base_status = STATUS_PASS if sub.passes else STATUS_FAIL
    if degraded and base_status == STATUS_PASS:
        base_status = STATUS_PARTIAL_DATA
    return base_status, missing, degraded


def _subcheck_evidence(
    *,
    key: str,
    sub: SubCheck,
    coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = _SUBCHECK_SPECS.get(key, {"label": sub.name, "required_sources": []})
    status, missing, degraded = _derive_subcheck_status(
        sub=sub, spec=spec, coverage=coverage,
    )
    out: dict[str, Any] = {
        "label": spec.get("label", sub.name),
        "status": status,
        "outcome": sub.resolved_outcome,
        "passes": bool(sub.passes),
        "score": (None if sub.score is None else float(sub.score)),
        "rationale": sub.rationale,
        "evidence": dict(sub.details or {}),
        "required_sources": list(spec.get("required_sources", [])),
        "missing_sources": missing,
        "degraded_sources": degraded,
    }
    if status == STATUS_ERROR:
        out["error_type"] = sub.details.get("exception_type")
    return out


def _coverage_to_dict(coverage: SourceCoverage | None) -> dict[str, Any] | None:
    """Convert ``SourceCoverage`` (or ``None``) to a plain JSON dict.

    The shareholder_letter slot grows a ``quality`` field
    (``"real" | "degraded" | "missing"``) so the dashboard does not have
    to re-derive provenance semantics. ``available=true`` plus
    ``quality="degraded"`` means: we have *some* letter text, but it is
    Item 1 or annual-report container material, not a CEO-to-shareholder
    letter — LongShort / Clarity will mark themselves ``partial_data``
    even when the LLM signal is positive.
    """
    if coverage is None:
        return None
    out = {
        "transcripts": {
            "available": coverage.transcripts_available,
            "count": int(coverage.transcripts_count),
            "expected": int(coverage.transcripts_expected),
        },
        "def14a_compensation": {"available": coverage.def14a_compensation_available},
        "def14a_letter": {"available": coverage.def14a_letter_available},
        "shareholder_letter": {
            "available": coverage.shareholder_letter_available,
            "from": coverage.shareholder_letter_source,
        },
        "mda": {"available": coverage.mda_available},
        "form4": {
            "available": coverage.form4_available,
            "n_transactions": coverage.form4_n_transactions,
        },
        "source_documents": coverage.source_documents,
    }
    out["shareholder_letter"]["quality"] = _shareholder_letter_quality(out)
    return out


def _empty_management_evidence(
    *,
    fm: FourMsResult | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "as_of": None,
        "fiscal_year": None,
        "model": fm.model if fm is not None else "",
        "cached": False,
        "bundle_hash": None,
        "source_coverage": None,
        "subchecks": {},
        "decision": {
            "outcome": "no_data",
            "hard_failures": [],
            "clean_passes": [],
            "neutral_subchecks": [],
            "partial_subchecks": [],
            "no_data_subchecks": [],
            "error_subchecks": [],
            "usable_evidence_count": 0,
            "total_subchecks": 0,
            "rationale": reason or "Management pipeline did not run.",
        },
        "reason": reason,
    }


def _decision_to_dict(management: ManagementResult) -> dict[str, Any]:
    """Serialize the aggregate ``ManagementDecision`` into JSON shape."""
    d = management.decision()
    return {
        "outcome": d.outcome,
        "hard_failures": list(d.hard_failures),
        "clean_passes": list(d.clean_passes),
        "neutral_subchecks": list(d.neutral_subchecks),
        "partial_subchecks": list(d.partial_subchecks),
        "no_data_subchecks": list(d.no_data_subchecks),
        "error_subchecks": list(d.error_subchecks),
        "usable_evidence_count": int(d.usable_evidence_count),
        "total_subchecks": int(d.total_subchecks),
        "rationale": d.rationale,
    }


def build_management_evidence(
    *,
    management: ManagementResult | None,
    four_ms: FourMsResult | None,
) -> dict[str, Any]:
    """Serialize a ``ManagementResult`` into the dashboard evidence blob.

    When the management pipeline did not run at all (no LLM client, or the
    LLM call raised before producing a ``ManagementResult``), the function
    returns a stub so the dashboard can render a clear "not evaluated"
    state instead of crashing on a missing key.
    """
    if management is None:
        return _empty_management_evidence(
            fm=four_ms,
            reason="Management pipeline did not run for this evaluation.",
        )

    coverage_dict = _coverage_to_dict(management.coverage)

    subchecks: dict[str, Any] = {
        "blame": _subcheck_evidence(
            key="blame", sub=management.blame, coverage=coverage_dict,
        ),
        "long_short": _subcheck_evidence(
            key="long_short", sub=management.long_short, coverage=coverage_dict,
        ),
        "clarity": _subcheck_evidence(
            key="clarity", sub=management.clarity, coverage=coverage_dict,
        ),
        "compensation": _subcheck_evidence(
            key="compensation", sub=management.compensation, coverage=coverage_dict,
        ),
        "insider": _subcheck_evidence(
            key="insider", sub=management.insider, coverage=coverage_dict,
        ),
    }
    # Capital allocation is optional on legacy cache entries (entries
    # written before the sub-check existed); skip the slot rather than
    # emit a confusing "no_data" placeholder when no SubCheck is present.
    if management.capital_allocation is not None:
        subchecks["capital_allocation"] = _subcheck_evidence(
            key="capital_allocation",
            sub=management.capital_allocation,
            coverage=coverage_dict,
        )

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "as_of": management.as_of.isoformat() if management.as_of else None,
        "fiscal_year": management.fiscal_year,
        "model": management.model,
        "cached": bool(management.cached),
        "bundle_hash": management.bundle_hash,
        "source_coverage": coverage_dict,
        "subchecks": subchecks,
        "decision": _decision_to_dict(management),
    }


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NO_DATA",
    "STATUS_PARTIAL_DATA",
    "STATUS_NEUTRAL",
    "STATUS_ERROR",
    "build_big_five_evidence",
    "build_management_evidence",
]
