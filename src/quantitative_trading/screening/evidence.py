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

from quantitative_trading.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from quantitative_trading.agents.rule_one.four_ms_llm import FourMsResult
from quantitative_trading.agents.rule_one.management_llm import (
    ManagementResult,
    SourceCoverage,
    SubCheck,
)

SCHEMA_VERSION = 1

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NO_DATA = "no_data"
STATUS_PARTIAL_DATA = "partial_data"
STATUS_ERROR = "error"


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
    },
    "clarity": {
        "label": "Plain-English clarity",
        "required_sources": ["shareholder_letter", "mda"],
        "any_of_required": True,
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
        # the gate workable when one source is missing.
        "required_sources": ["mda", "shareholder_letter"],
        "any_of_required": True,
    },
}


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
) -> tuple[str, list[str]]:
    """Return ``(status, missing_sources)`` for a Management sub-check.

    Coverage is optional so this stays back-compatible with cached
    Management results that pre-date the ``SourceCoverage`` field. Without
    coverage we can only fall back to pass/fail (or ``error`` when the
    sub-check exception flag is set).
    """
    if sub.details.get("error"):
        return STATUS_ERROR, []

    if coverage is None:
        return STATUS_PASS if sub.passes else STATUS_FAIL, []

    required = spec.get("required_sources", []) or []
    available = [s for s in required if _coverage_lookup(coverage, s)]
    missing = [s for s in required if not _coverage_lookup(coverage, s)]

    if spec.get("any_of_required", False):
        if not available:
            return STATUS_NO_DATA, missing
        if missing:
            return STATUS_PARTIAL_DATA, missing
    elif missing:
        return STATUS_NO_DATA, missing

    return (STATUS_PASS if sub.passes else STATUS_FAIL), missing


def _subcheck_evidence(
    *,
    key: str,
    sub: SubCheck,
    coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = _SUBCHECK_SPECS.get(key, {"label": sub.name, "required_sources": []})
    status, missing = _derive_subcheck_status(sub=sub, spec=spec, coverage=coverage)
    out: dict[str, Any] = {
        "label": spec.get("label", sub.name),
        "status": status,
        "passes": bool(sub.passes),
        "score": (None if sub.score is None else float(sub.score)),
        "rationale": sub.rationale,
        "evidence": dict(sub.details or {}),
        "required_sources": list(spec.get("required_sources", [])),
        "missing_sources": missing,
    }
    if status == STATUS_ERROR:
        out["error_type"] = sub.details.get("exception_type")
    return out


def _coverage_to_dict(coverage: SourceCoverage | None) -> dict[str, Any] | None:
    """Convert ``SourceCoverage`` (or ``None``) to a plain JSON dict."""
    if coverage is None:
        return None
    return {
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


def _empty_management_evidence(
    *,
    fm: FourMsResult | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": None,
        "fiscal_year": None,
        "model": fm.model if fm is not None else "",
        "cached": False,
        "bundle_hash": None,
        "source_coverage": None,
        "subchecks": {},
        "reason": reason,
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
        "schema_version": SCHEMA_VERSION,
        "as_of": management.as_of.isoformat() if management.as_of else None,
        "fiscal_year": management.fiscal_year,
        "model": management.model,
        "cached": bool(management.cached),
        "bundle_hash": management.bundle_hash,
        "source_coverage": coverage_dict,
        "subchecks": subchecks,
    }


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NO_DATA",
    "STATUS_PARTIAL_DATA",
    "STATUS_ERROR",
    "build_big_five_evidence",
    "build_management_evidence",
]
