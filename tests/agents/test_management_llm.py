"""Tests for the multi-document Management evaluator pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.agents.rule_one.llm_client import LlmCallResult, LlmClient
from quantitative_trading.agents.rule_one.management_llm import (
    BlameEvaluator,
    ClarityEvaluator,
    CompensationEvaluator,
    DocumentBundle,
    DocumentBundler,
    InsiderAlignmentEvaluator,
    LongShortEvaluator,
    ManagementAnalyzer,
    SubCheck,
)
from quantitative_trading.config import get_config
from quantitative_trading.data.transcripts import EarningsTranscript


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _bundle(
    *,
    ticker: str = "FAKE",
    fy: int = 2023,
    transcripts: list[EarningsTranscript] | None = None,
    mda: str = "MD&A text " * 200,
    def14a_comp: str = "Compensation Discussion and Analysis text " * 100,
    def14a_letter: str = "",
    letter: str = "Shareholder letter text. " * 200,
) -> DocumentBundle:
    return DocumentBundle(
        ticker=ticker, as_of=date(2024, 1, 1), fiscal_year=fy,
        accession_10k="10K-1", accession_def14a="14A-1",
        mda_text=mda,
        def14a_compensation_text=def14a_comp,
        def14a_letter_text=def14a_letter,
        shareholder_letter_text=letter,
        transcripts=transcripts or [],
    )


def _make_llm(payload: dict, *, dry_run: bool = False) -> MagicMock:
    """Mock LlmClient that always returns ``payload`` from ``.call``."""
    llm = MagicMock(spec=LlmClient)
    llm.model = "claude-opus-4-7"
    llm.thinking_budget_tokens = 32_000
    llm.dry_run = dry_run
    llm.truncate.side_effect = lambda s: s[:50_000]
    llm.call.return_value = LlmCallResult(
        payload=payload, estimated_input_tokens=100,
        estimated_output_tokens=100, estimated_cost_usd=0.01, dry_run=dry_run,
    )
    return llm


def _transcript(year: int = 2023, q: int = 1) -> EarningsTranscript:
    return EarningsTranscript(
        ticker="FAKE", fiscal_year=year, fiscal_quarter=q,
        call_date=date(year, q * 3, 25), source="fmp",
        text=f"Transcript Q{q} {year}: We made progress on our long-term strategy. " * 50,
    )


# --------------------------------------------------------------------------
# BlameEvaluator
# --------------------------------------------------------------------------


def test_blame_passes_when_takes_responsibility_and_few_scapegoats() -> None:
    llm = _make_llm({
        "takes_responsibility": True, "scapegoat_count": 1,
        "rationale": "CEO owned the inventory miss directly.",
        "supporting_quotes": ["We made a mistake."],
    })
    sub = BlameEvaluator(llm).evaluate(_bundle(transcripts=[_transcript()]))
    assert sub.passes
    assert sub.score == 1.0


def test_blame_fails_when_too_many_scapegoats() -> None:
    llm = _make_llm({
        "takes_responsibility": False, "scapegoat_count": 5,
        "rationale": "Repeatedly blamed macro conditions.",
    })
    sub = BlameEvaluator(llm).evaluate(_bundle(transcripts=[_transcript()]))
    assert not sub.passes


def test_blame_fails_with_no_transcripts_and_skips_llm() -> None:
    llm = _make_llm({"takes_responsibility": True, "scapegoat_count": 0,
                      "rationale": ""})
    sub = BlameEvaluator(llm).evaluate(_bundle(transcripts=[]))
    assert not sub.passes
    llm.call.assert_not_called()


# --------------------------------------------------------------------------
# LongShortEvaluator
# --------------------------------------------------------------------------


def test_long_short_passes_when_dominant_long() -> None:
    llm = _make_llm({
        "short_term_mentions": 2, "long_term_mentions": 18, "ratio": 9.0,
        "dominant_orientation": "long", "rationale": "Decade-long vision.",
    })
    sub = LongShortEvaluator(llm).evaluate(_bundle(transcripts=[_transcript()]))
    assert sub.passes
    assert sub.score == 9.0


def test_long_short_fails_when_dominant_short_or_mixed() -> None:
    for orientation in ("short", "mixed"):
        llm = _make_llm({
            "short_term_mentions": 20, "long_term_mentions": 2, "ratio": 0.1,
            "dominant_orientation": orientation, "rationale": "Quarterly focus.",
        })
        sub = LongShortEvaluator(llm).evaluate(
            _bundle(transcripts=[_transcript()])
        )
        assert not sub.passes, orientation


# --------------------------------------------------------------------------
# ClarityEvaluator
# --------------------------------------------------------------------------


def test_clarity_passes_when_score_at_or_above_seven() -> None:
    llm = _make_llm({"clarity_score": 8, "rationale": "Plain English throughout.",
                      "jargon_examples": [], "plain_english_examples": []})
    sub = ClarityEvaluator(llm).evaluate(_bundle())
    assert sub.passes
    assert sub.score == 8.0


def test_clarity_fails_below_seven() -> None:
    llm = _make_llm({"clarity_score": 4, "rationale": "Heavy jargon."})
    sub = ClarityEvaluator(llm).evaluate(_bundle())
    assert not sub.passes


def test_clarity_fails_when_no_text_available() -> None:
    llm = _make_llm({"clarity_score": 10, "rationale": ""})
    sub = ClarityEvaluator(llm).evaluate(
        _bundle(letter="", mda="")
    )
    assert not sub.passes
    llm.call.assert_not_called()


# --------------------------------------------------------------------------
# CompensationEvaluator
# --------------------------------------------------------------------------


def test_compensation_passes_when_aligned() -> None:
    llm = _make_llm({
        "metrics": ["ROIC", "EPS"],
        "shareholder_aligned_metrics": ["ROIC", "EPS"],
        "empire_building_metrics": [],
        "aligned_with_shareholders": True,
        "rationale": "Bonuses driven by ROIC.",
    })
    sub = CompensationEvaluator(llm).evaluate(_bundle())
    assert sub.passes


def test_compensation_fails_when_only_revenue_and_ebitda() -> None:
    llm = _make_llm({
        "metrics": ["Revenue Growth", "Adjusted EBITDA"],
        "shareholder_aligned_metrics": [],
        "empire_building_metrics": ["Revenue Growth", "Adjusted EBITDA"],
        "aligned_with_shareholders": False,
        "rationale": "Pure empire-building.",
    })
    sub = CompensationEvaluator(llm).evaluate(_bundle())
    assert not sub.passes


def test_compensation_fails_with_no_def14a_text() -> None:
    llm = _make_llm({})
    sub = CompensationEvaluator(llm).evaluate(_bundle(def14a_comp=""))
    assert not sub.passes
    llm.call.assert_not_called()


# --------------------------------------------------------------------------
# InsiderAlignmentEvaluator (deterministic)
# --------------------------------------------------------------------------


def test_insider_evaluator_calls_summarizer_via_edgar() -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.list_filings.return_value = []  # no insider trades found
    sub = InsiderAlignmentEvaluator(edgar).evaluate("FAKE", date(2024, 1, 1))
    # No transactions → net 0 → passes (no large recent sells either)
    assert sub.passes
    assert sub.details["n_transactions"] == 0


def test_insider_evaluator_handles_missing_ticker() -> None:
    edgar = MagicMock()
    edgar.get_cik.side_effect = KeyError("nope")
    sub = InsiderAlignmentEvaluator(edgar).evaluate("BADTICK", date(2024, 1, 1))
    assert not sub.passes
    assert "No CIK" in sub.rationale


# --------------------------------------------------------------------------
# Aggregator + caching
# --------------------------------------------------------------------------


def _passing_payload(eval_name: str) -> dict:
    return {
        "blame": {
            "takes_responsibility": True, "scapegoat_count": 0,
            "rationale": "ok",
        },
        "horizon": {
            "short_term_mentions": 1, "long_term_mentions": 10, "ratio": 10.0,
            "dominant_orientation": "long", "rationale": "ok",
        },
        "clarity": {"clarity_score": 9, "rationale": "ok"},
        "compensation": {
            "metrics": ["ROIC"], "aligned_with_shareholders": True,
            "rationale": "ok",
        },
    }[eval_name]


def _smart_llm() -> MagicMock:
    """LLM mock that returns the right payload depending on the tool called."""
    llm = MagicMock(spec=LlmClient)
    llm.model = "claude-opus-4-7"
    llm.thinking_budget_tokens = 32_000
    llm.dry_run = False
    llm.truncate.side_effect = lambda s: s[:50_000]

    def fake_call(*, system_prompt, user_prompt, tool, dry_run_payload=None):
        name = tool["name"]
        mapping = {
            "submit_blame_assessment": _passing_payload("blame"),
            "submit_horizon_assessment": _passing_payload("horizon"),
            "submit_clarity_assessment": _passing_payload("clarity"),
            "submit_compensation_assessment": _passing_payload("compensation"),
        }
        return LlmCallResult(
            payload=mapping[name],
            estimated_input_tokens=10, estimated_output_tokens=10,
            estimated_cost_usd=0.0, dry_run=False,
        )

    llm.call.side_effect = fake_call
    return llm


def test_management_analyzer_aggregates_and_caches(tmp_path: Path) -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.list_filings.return_value = []  # insider check passes (no trades)

    llm = _smart_llm()
    bundler = MagicMock(spec=DocumentBundler)
    bundler.build.return_value = _bundle(transcripts=[_transcript()])

    analyzer = ManagementAnalyzer(
        edgar_client=edgar, llm_client=llm,
        document_bundler=bundler, cache_dir=tmp_path / "mgmt_cache",
    )
    r1 = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))
    r2 = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))

    assert r1.passes
    assert r1.blame.passes and r1.long_short.passes
    assert r1.clarity.passes and r1.compensation.passes
    assert r1.insider.passes
    assert not r1.cached
    assert r2.cached
    assert llm.call.call_count == 4  # one per LLM evaluator


def test_management_analyzer_per_check_dict_lists_all_five() -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.list_filings.return_value = []
    llm = _smart_llm()
    bundler = MagicMock(spec=DocumentBundler)
    bundler.build.return_value = _bundle(transcripts=[_transcript()])
    analyzer = ManagementAnalyzer(
        edgar_client=edgar, llm_client=llm, document_bundler=bundler,
    )
    result = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))
    assert set(result.per_check.keys()) == {
        "blame", "long_short", "clarity", "compensation", "insider"
    }


def test_management_passes_only_when_all_five_pass() -> None:
    """Replace one sub-check with a failing payload and confirm aggregator fails."""
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.list_filings.return_value = []
    llm = _smart_llm()
    # Override compensation to fail.
    original_side = llm.call.side_effect

    def failing_compensation(*, system_prompt, user_prompt, tool, dry_run_payload=None):
        if tool["name"] == "submit_compensation_assessment":
            return LlmCallResult(
                payload={"metrics": ["Revenue"], "aligned_with_shareholders": False,
                         "rationale": "no"},
                estimated_input_tokens=1, estimated_output_tokens=1,
                estimated_cost_usd=0.0, dry_run=False,
            )
        return original_side(system_prompt=system_prompt,
                              user_prompt=user_prompt, tool=tool)

    llm.call.side_effect = failing_compensation
    bundler = MagicMock(spec=DocumentBundler)
    bundler.build.return_value = _bundle(transcripts=[_transcript()])
    analyzer = ManagementAnalyzer(
        edgar_client=edgar, llm_client=llm, document_bundler=bundler,
    )
    result = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))
    assert not result.passes
    assert not result.compensation.passes
    assert result.blame.passes  # other sub-checks still pass


def test_management_one_subcheck_exception_does_not_zero_result(tmp_path: Path) -> None:
    """If a single sub-check evaluator raises (e.g. LLM streaming error,
    no parseable JSON, transient 429), the aggregator must still return a
    ManagementResult with the OTHER sub-checks intact and the failed one
    surfaced as ``passes=False`` with the exception in its rationale."""
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.list_filings.return_value = []  # insider check passes
    llm = _smart_llm()
    original_side = llm.call.side_effect

    def raising_compensation(*, system_prompt, user_prompt, tool, dry_run_payload=None):
        if tool["name"] == "submit_compensation_assessment":
            raise RuntimeError(
                "Anthropic returned no tool_use block and no parseable JSON in text"
            )
        return original_side(system_prompt=system_prompt,
                              user_prompt=user_prompt, tool=tool)

    llm.call.side_effect = raising_compensation
    bundler = MagicMock(spec=DocumentBundler)
    bundler.build.return_value = _bundle(transcripts=[_transcript()])
    analyzer = ManagementAnalyzer(
        edgar_client=edgar, llm_client=llm, document_bundler=bundler,
        cache_dir=tmp_path / "mgmt_cache_safe",
    )
    result = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))

    # Other sub-checks remain truthful.
    assert result.blame.passes
    assert result.long_short.passes
    assert result.clarity.passes
    assert result.insider.passes
    # Compensation degraded gracefully — exists, fails, has the exception
    # message in its rationale.
    assert not result.compensation.passes
    assert "no tool_use" in result.compensation.rationale
    assert result.compensation.details.get("error") is True
    # Aggregate still fails (because compensation is required for the AND).
    assert not result.passes


# --------------------------------------------------------------------------
# DocumentBundler.hash determinism
# --------------------------------------------------------------------------


def test_bundle_hash_changes_when_text_changes() -> None:
    b1 = _bundle(mda="A")
    b2 = _bundle(mda="B")
    assert b1.hash() != b2.hash()


def test_bundle_hash_stable_for_same_inputs() -> None:
    b1 = _bundle(mda="A", transcripts=[_transcript()])
    b2 = _bundle(mda="A", transcripts=[_transcript()])
    assert b1.hash() == b2.hash()
