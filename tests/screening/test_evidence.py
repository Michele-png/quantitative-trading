"""Tests for ``screening/evidence.py`` JSON builders."""

from __future__ import annotations

from datetime import date

from value_investing_backend.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from value_investing_backend.agents.rule_one.four_ms_llm import FourMsResult, MCheck
from value_investing_backend.agents.rule_one.management_llm import (
    ManagementResult,
    SourceCoverage,
    SubCheck,
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

# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _full_series(start_fy: int = 2014, n: int = 10, val: float = 100.0) -> dict[int, float]:
    return {start_fy + i: val * (1 + 0.10) ** i for i in range(n)}


def _partial_series(present_years: int, start_fy: int = 2017) -> dict[int, float | None]:
    """Series with `present_years` years of data filling the latest slots,
    older years recorded as None to mirror what ``PointInTimeFacts`` returns
    for a young issuer."""
    out: dict[int, float | None] = {}
    target_n = 10
    earliest = start_fy + target_n - 1 - (target_n - 1)
    for offset in range(target_n):
        fy = earliest + offset
        if offset >= target_n - present_years:
            out[fy] = 100.0 * (1 + 0.12) ** (offset - (target_n - present_years))
        else:
            out[fy] = None
    return out


def _make_big_five(
    *,
    roic_value: float | None = 0.18,
    sales_series: dict[int, float | None] | None = None,
    eps_series: dict[int, float | None] | None = None,
    sales_passes: bool = True,
) -> BigFiveResult:
    sales = sales_series if sales_series is not None else _full_series()
    eps = eps_series if eps_series is not None else _full_series(val=2.0)
    return BigFiveResult(
        ticker="FAKE", as_of=date(2024, 1, 1), latest_fiscal_year=2023,
        n_years_required=10,
        roic=MetricResult(
            "ROIC", roic_value, 0.10,
            roic_value is not None and roic_value >= 0.10,
            "ok",
            series={2014 + i: 0.18 for i in range(10)},
        ),
        sales_growth=MetricResult(
            "Sales Growth", 0.12 if sales_passes else 0.05, 0.10,
            sales_passes, "ok", series=sales,
        ),
        eps_growth=MetricResult(
            "EPS Growth", 0.15, 0.10, True, "ok", series=eps,
        ),
        equity_growth=MetricResult(
            "Equity Growth", 0.13, 0.10, True, "ok", series=_full_series(val=50.0),
        ),
        ocf_growth=MetricResult(
            "OCF Growth", 0.14, 0.10, True, "ok", series=_full_series(val=80.0),
        ),
        current_ratio=MetricResult(
            "Current Ratio", 2.5, 2.0, True, "ok",
        ),
    )


# --------------------------------------------------------------------------
# Big 5 evidence
# --------------------------------------------------------------------------


class TestBigFiveEvidence:
    def test_full_window_marks_each_check_pass(self) -> None:
        ev = build_big_five_evidence(_make_big_five())
        assert ev["schema_version"] == 1
        assert ev["latest_fiscal_year"] == 2023
        assert ev["n_years_target"] == 10
        for key in ("roic", "sales_growth", "eps_growth",
                    "equity_growth", "ocf_growth"):
            check = ev["checks"][key]
            assert check["status"] == STATUS_PASS
            assert check["years_used"] == 10
            assert check["relaxed_window"] is False
            assert check["missing_fiscal_years"] == []
            assert "relaxed_reason" not in check

    def test_relaxed_window_marks_partial_data(self) -> None:
        """A relaxed-window metric is no longer a clean PASS — even when
        the underlying CAGR is well above the threshold the dashboard
        should show PARTIAL DATA so the user knows the gate is open
        because evidence is missing, not because the company is failing.
        """
        partial = _partial_series(present_years=6)
        big_five = _make_big_five(sales_series=partial, sales_passes=False)
        ev = build_big_five_evidence(big_five)
        sales = ev["checks"]["sales_growth"]
        assert sales["status"] == STATUS_PARTIAL_DATA
        assert sales["relaxed_window"] is True
        assert sales["years_used"] == 6
        assert len(sales["missing_fiscal_years"]) == 4
        assert "Phil Town" in sales["relaxed_reason"]
        assert "10-year" in sales["relaxed_reason"]
        # Decision-grade is False whenever the window is not full.
        assert sales["decision_grade"] is False

    def test_no_data_status_when_metric_value_is_none(self) -> None:
        big_five = _make_big_five(roic_value=None)
        ev = build_big_five_evidence(big_five)
        assert ev["checks"]["roic"]["status"] == STATUS_NO_DATA
        assert ev["checks"]["roic"]["value"] is None

    def test_fail_status_when_metric_below_threshold(self) -> None:
        ev = build_big_five_evidence(_make_big_five(sales_passes=False))
        assert ev["checks"]["sales_growth"]["status"] == STATUS_FAIL

    def test_yearly_points_are_sorted_and_typed(self) -> None:
        ev = build_big_five_evidence(_make_big_five())
        yearly = ev["checks"]["roic"]["yearly"]
        assert yearly == sorted(yearly, key=lambda p: p["fiscal_year"])
        assert all(isinstance(p["fiscal_year"], int) for p in yearly)
        assert all(isinstance(p["value"], float) for p in yearly)

    def test_calculation_method_distinguishes_roic_from_growth(self) -> None:
        ev = build_big_five_evidence(_make_big_five())
        assert ev["checks"]["roic"]["calculation_method"] == "avg_and_most_recent_floor"
        assert ev["checks"]["sales_growth"]["calculation_method"] == "cagr_first_to_last"
        assert ev["checks"]["eps_growth"]["calculation_method"] == (
            "cagr_first_to_last_split_adjusted"
        )

    def test_none_big_five_returns_stub(self) -> None:
        ev = build_big_five_evidence(None)
        assert ev["checks"] == {}
        assert ev["latest_fiscal_year"] is None


# --------------------------------------------------------------------------
# Management evidence
# --------------------------------------------------------------------------


def _full_coverage(*, transcripts: int = 8) -> SourceCoverage:
    return SourceCoverage(
        transcripts_available=transcripts > 0,
        transcripts_count=transcripts,
        transcripts_expected=8,
        def14a_compensation_available=True,
        def14a_letter_available=True,
        shareholder_letter_available=True,
        shareholder_letter_source="def14a",
        mda_available=True,
        form4_available=True,
        form4_n_transactions=12,
    )


def _make_management(
    *,
    coverage: SourceCoverage | None = None,
    blame_passes: bool = True,
    blame_details: dict | None = None,
    long_short_passes: bool = True,
    clarity_passes: bool = True,
    compensation_passes: bool = True,
    insider_passes: bool = True,
    insider_details: dict | None = None,
    compensation_details: dict | None = None,
) -> ManagementResult:
    return ManagementResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year=2023,
        bundle_hash="abc", model="claude-opus-4-7", cached=False,
        blame=SubCheck("Blame", blame_passes, 1.0, "ok",
                       details=blame_details or {}),
        long_short=SubCheck("LongShort", long_short_passes, 5.0, "ok",
                            details={"ratio": 5.0}),
        clarity=SubCheck("Clarity", clarity_passes, 8.0, "ok"),
        compensation=SubCheck("Compensation", compensation_passes, None,
                              "ok", details=compensation_details or {}),
        insider=SubCheck("Insider", insider_passes,
                         100_000.0 if insider_passes else -50_000.0, "ok",
                         details=insider_details or {
                             "n_transactions": 12,
                             "net_open_market_value_usd": 100_000.0,
                         }),
        coverage=coverage,
    )


def _stub_four_ms() -> FourMsResult:
    return FourMsResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year=2023,
        accession="A1", model="claude-opus-4-7",
        meaning=MCheck("Meaning", True, "ok"),
        moat=MCheck("Moat", True, "ok", details={"moat_type": "brand"}),
        management=MCheck("Management", True, "ok"),
        cached=False, raw_response={},
    )


class TestManagementEvidence:
    def test_full_coverage_reports_all_pass(self) -> None:
        ev = build_management_evidence(
            management=_make_management(coverage=_full_coverage()),
            four_ms=_stub_four_ms(),
        )
        # Management evidence schema bumped to v2 when the explicit
        # ``outcome`` field and aggregate ``decision`` block shipped.
        assert ev["schema_version"] == 2
        for key in ("blame", "long_short", "clarity",
                    "compensation", "insider"):
            assert ev["subchecks"][key]["status"] == STATUS_PASS
            assert ev["subchecks"][key]["missing_sources"] == []
        cov = ev["source_coverage"]
        assert cov["transcripts"]["available"] is True
        assert cov["transcripts"]["count"] == 8
        assert cov["form4"]["available"] is True
        # Real DEF 14A letter source → ``quality="real"``; the dashboard
        # uses this to avoid down-grading LongShort / Clarity.
        assert cov["shareholder_letter"]["quality"] == "real"
        # Aggregate decision block accompanies every fresh evidence blob.
        assert ev["decision"]["outcome"] == "pass"
        assert ev["decision"]["usable_evidence_count"] >= 3

    def test_no_transcripts_marks_blame_no_data(self) -> None:
        coverage = _full_coverage(transcripts=0)
        ev = build_management_evidence(
            management=_make_management(
                coverage=coverage, blame_passes=False,
            ),
            four_ms=_stub_four_ms(),
        )
        blame = ev["subchecks"]["blame"]
        assert blame["status"] == STATUS_NO_DATA
        assert "transcripts" in blame["missing_sources"]
        # And underlying boolean still preserved for hard-gating audits.
        assert blame["passes"] is False

    def test_only_one_of_two_sources_marks_partial(self) -> None:
        # Long-short benefits from BOTH transcripts and a shareholder
        # letter; with only transcripts available it should be partial.
        coverage = SourceCoverage(
            transcripts_available=True, transcripts_count=8,
            transcripts_expected=8,
            def14a_compensation_available=True,
            def14a_letter_available=False,
            shareholder_letter_available=False,
            shareholder_letter_source=None,
            mda_available=True,
            form4_available=True, form4_n_transactions=4,
        )
        ev = build_management_evidence(
            management=_make_management(coverage=coverage),
            four_ms=_stub_four_ms(),
        )
        ls = ev["subchecks"]["long_short"]
        assert ls["status"] == STATUS_PARTIAL_DATA
        assert "shareholder_letter" in ls["missing_sources"]

    def test_safe_eval_error_marks_subcheck_error(self) -> None:
        ev = build_management_evidence(
            management=_make_management(
                coverage=_full_coverage(),
                compensation_passes=False,
                compensation_details={"error": True,
                                      "exception_type": "RuntimeError"},
            ),
            four_ms=_stub_four_ms(),
        )
        comp = ev["subchecks"]["compensation"]
        assert comp["status"] == STATUS_ERROR
        assert comp["error_type"] == "RuntimeError"

    def test_form4_failure_marks_insider_no_data(self) -> None:
        coverage = SourceCoverage(
            transcripts_available=True, transcripts_count=8,
            transcripts_expected=8,
            def14a_compensation_available=True,
            def14a_letter_available=True,
            shareholder_letter_available=True,
            shareholder_letter_source="def14a",
            mda_available=True,
            form4_available=False, form4_n_transactions=None,
        )
        ev = build_management_evidence(
            management=_make_management(
                coverage=coverage, insider_passes=False,
                insider_details={},
            ),
            four_ms=_stub_four_ms(),
        )
        insider = ev["subchecks"]["insider"]
        assert insider["status"] == STATUS_NO_DATA
        assert insider["missing_sources"] == ["form4"]

    def test_missing_coverage_falls_back_to_passfail(self) -> None:
        # Pre-coverage cached entries — no SourceCoverage attached.
        ev = build_management_evidence(
            management=_make_management(coverage=None, blame_passes=False),
            four_ms=_stub_four_ms(),
        )
        assert ev["source_coverage"] is None
        assert ev["subchecks"]["blame"]["status"] == STATUS_FAIL
        assert ev["subchecks"]["clarity"]["status"] == STATUS_PASS

    def test_none_management_returns_stub(self) -> None:
        ev = build_management_evidence(management=None, four_ms=None)
        assert ev["subchecks"] == {}
        assert ev["source_coverage"] is None
        assert "did not run" in ev["reason"]
        # Stub still carries an aggregate decision block so the
        # dashboard never has to guess: explicit ``no_data``.
        assert ev["decision"]["outcome"] == "no_data"


class TestManagementEvidenceFallbackLetter:
    """The shareholder-letter slot must surface fallback provenance so the
    dashboard's LongShort / Clarity downgrade behaviour is consistent."""

    def _fallback_coverage(self, *, source: str) -> SourceCoverage:
        return SourceCoverage(
            transcripts_available=True, transcripts_count=8,
            transcripts_expected=8,
            def14a_compensation_available=True,
            def14a_letter_available=False,
            shareholder_letter_available=True,
            shareholder_letter_source=source,
            mda_available=True,
            form4_available=True, form4_n_transactions=4,
        )

    def test_item1_fallback_letter_marks_long_short_partial(self) -> None:
        """Item 1 fallback in the shareholder-letter slot must downgrade
        LongShort to ``partial_data`` even when the underlying signal
        passes — Item 1 is regulatory product copy, not a CEO letter."""
        coverage = self._fallback_coverage(
            source="10-k_item1_fallback_business_description",
        )
        ev = build_management_evidence(
            management=_make_management(
                coverage=coverage, long_short_passes=True,
            ),
            four_ms=_stub_four_ms(),
        )
        assert ev["source_coverage"]["shareholder_letter"]["quality"] == "degraded"
        ls = ev["subchecks"]["long_short"]
        assert ls["status"] == STATUS_PARTIAL_DATA
        assert "shareholder_letter" in ls["degraded_sources"]

    def test_annual_report_archive_marks_clarity_partial(self) -> None:
        """An archived ``annual_report`` filling the shareholder-letter
        slot must also be treated as degraded — the annual report is a
        container, not a real CEO letter section."""
        coverage = self._fallback_coverage(source="archive:annual_report")
        ev = build_management_evidence(
            management=_make_management(
                coverage=coverage, clarity_passes=True,
            ),
            four_ms=_stub_four_ms(),
        )
        assert ev["source_coverage"]["shareholder_letter"]["quality"] == "degraded"
        clarity = ev["subchecks"]["clarity"]
        assert clarity["status"] == STATUS_PARTIAL_DATA
        assert "shareholder_letter" in clarity["degraded_sources"]

    def test_real_def14a_letter_keeps_clean_pass(self) -> None:
        """A genuine DEF 14A letter must not be downgraded — the
        ``quality`` tag stays ``real`` and the sub-check passes cleanly."""
        coverage = self._fallback_coverage(source="def14a")
        ev = build_management_evidence(
            management=_make_management(
                coverage=coverage, long_short_passes=True,
            ),
            four_ms=_stub_four_ms(),
        )
        assert ev["source_coverage"]["shareholder_letter"]["quality"] == "real"
        assert ev["subchecks"]["long_short"]["status"] == STATUS_PASS
