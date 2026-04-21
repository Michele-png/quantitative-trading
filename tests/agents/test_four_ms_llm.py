"""Unit tests for the LLM-driven 4Ms analyzer.

Mocks the Anthropic API and the EDGAR client so tests are hermetic.
Verifies prompt construction, section extraction, caching behavior, and
result parsing — without making any real API calls.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.agents.rule_one.four_ms_llm import (
    FourMsAnalyzer,
    _coerce_to_dict,
    extract_10k_sections,
    find_pit_10k,
    html_to_text,
)
from quantitative_trading.config import get_config


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


@pytest.fixture
def fake_10k_html() -> str:
    """A minimal HTML 10-K fixture with TOC + body for Items 1, 1A, 7."""
    return """<html><body>
<h1>FAKECO INC. — Annual Report on Form 10-K</h1>
<h2>Table of Contents</h2>
<ul>
  <li>Item 1. Business</li>
  <li>Item 1A. Risk Factors</li>
  <li>Item 2. Properties</li>
  <li>Item 7. Management's Discussion and Analysis</li>
  <li>Item 7A. Quantitative and Qualitative Disclosures</li>
</ul>

<h2>Item 1. Business</h2>
<p>FakeCo makes widgets. Our brand is recognized worldwide. We hold patents
on our widget-making process and have durable competitive advantages.</p>

<h2>Item 1A. Risk Factors</h2>
<p>We face risks from competition, regulation, and macroeconomic conditions.</p>

<h2>Item 2. Properties</h2>
<p>We own factories in Ohio.</p>

<h2>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>Revenue grew 15% year over year. Management focuses on long-term value
creation through prudent capital allocation.</p>

<h2>Item 7A. Quantitative and Qualitative Disclosures</h2>
<p>We hedge against currency risk.</p>

</body></html>"""


@pytest.fixture
def mock_anthropic_response_pass_all() -> MagicMock:
    """Mock Anthropic Messages.create response for a 'pass all 3 Ms' result."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {
        "meaning": {
            "passes": True,
            "rationale": "Widget-making is a clear, durable business in our circle of competence.",
        },
        "moat": {
            "passes": True,
            "moat_type": "brand",
            "rationale": "Worldwide brand recognition + patents indicate durable competitive advantage.",
        },
        "management": {
            "passes": True,
            "rationale": "MD&A indicates focus on long-term value and prudent capital allocation.",
            "red_flags": [],
        },
    }
    response = MagicMock()
    response.content = [block]
    return response


# --------------------------------------------------------------------------
# html_to_text and section extraction
# --------------------------------------------------------------------------


def test_html_to_text_strips_tags(fake_10k_html: str) -> None:
    text = html_to_text(fake_10k_html)
    assert "FakeCo makes widgets" in text
    assert "<p>" not in text
    assert "<html>" not in text


def test_extract_10k_sections_finds_all_three(fake_10k_html: str) -> None:
    text = html_to_text(fake_10k_html)
    sections = extract_10k_sections(text)
    assert "FakeCo makes widgets" in sections["item_1_business"]
    assert "Risk Factors" in sections["item_1a_risk_factors"]
    assert "competition, regulation" in sections["item_1a_risk_factors"]
    assert "Revenue grew 15%" in sections["item_7_mda"]


def test_extract_10k_sections_skips_table_of_contents(fake_10k_html: str) -> None:
    """The TOC mentions all items but has no body text. Body sections are
    selected (last occurrence of each item header)."""
    text = html_to_text(fake_10k_html)
    sections = extract_10k_sections(text)
    item_1 = sections["item_1_business"]
    # Body version contains the actual paragraph; TOC just mentions item names.
    assert "FakeCo makes widgets" in item_1
    # The TOC region should NOT be the dominant content of the section.
    assert len(item_1) > 30


def test_extract_returns_empty_dict_when_no_items() -> None:
    sections = extract_10k_sections("This document has no Items at all.")
    assert sections["item_1_business"] == ""
    assert sections["item_1a_risk_factors"] == ""
    assert sections["item_7_mda"] == ""


# --------------------------------------------------------------------------
# find_pit_10k
# --------------------------------------------------------------------------


def test_find_pit_10k_returns_latest_before_as_of() -> None:
    pit = MagicMock()
    pit.fiscal_year_end.side_effect = lambda fy: {
        2018: date(2018, 12, 31),
        2019: date(2019, 12, 31),
        2020: date(2020, 12, 31),
    }.get(fy)
    edgar = MagicMock()
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-19", "filingDate": "2020-02-15",
            "reportDate": "2019-12-31", "form": "10-K",
            "primaryDocument": "fy19.htm", "cik": 999,
        },
        {
            "accessionNumber": "0001-20", "filingDate": "2021-02-15",
            "reportDate": "2020-12-31", "form": "10-K",
            "primaryDocument": "fy20.htm", "cik": 999,
        },
        {
            "accessionNumber": "0001-18", "filingDate": "2019-02-15",
            "reportDate": "2018-12-31", "form": "10-K",
            "primaryDocument": "fy18.htm", "cik": 999,
        },
    ]
    result = find_pit_10k(pit, edgar, cik=999, as_of=date(2020, 6, 1))
    assert result is not None
    assert result["accessionNumber"] == "0001-19"
    assert result["fiscal_year"] == 2019


def test_find_pit_10k_returns_none_when_no_filings_before_as_of() -> None:
    pit = MagicMock()
    edgar = MagicMock()
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-25", "filingDate": "2025-02-15",
            "reportDate": "2024-12-31", "form": "10-K",
            "primaryDocument": "fy24.htm", "cik": 999,
        },
    ]
    assert find_pit_10k(pit, edgar, cik=999, as_of=date(2020, 1, 1)) is None


# --------------------------------------------------------------------------
# FourMsAnalyzer end-to-end (mocked)
# --------------------------------------------------------------------------


def test_evaluate_makes_one_llm_call_and_caches(
    fake_10k_html: str,
    mock_anthropic_response_pass_all: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {"us-gaap": {}}, "cik": 999}
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-23-1", "filingDate": "2023-02-15",
            "reportDate": "2022-12-31", "form": "10-K",
            "primaryDocument": "fy22.htm", "cik": 999,
        },
    ]
    edgar.fetch_filing_document.return_value = fake_10k_html

    anthropic = MagicMock()
    anthropic.messages.create.return_value = mock_anthropic_response_pass_all

    analyzer = FourMsAnalyzer(
        edgar_client=edgar,
        anthropic_client=anthropic,
        cache_dir=tmp_path / "llm_cache",
    )

    result1 = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))
    result2 = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))

    assert result1.all_pass
    assert result1.meaning.passes and result1.moat.passes and result1.management.passes
    assert result1.moat.details["moat_type"] == "brand"
    assert not result1.cached
    assert result2.cached
    # Anthropic called exactly once across both evaluate() calls
    assert anthropic.messages.create.call_count == 1


def test_evaluate_returns_unable_when_no_10k_before_as_of(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {}, "cik": 999}
    edgar.list_filings.return_value = []  # no filings at all
    anthropic = MagicMock()

    analyzer = FourMsAnalyzer(
        edgar_client=edgar, anthropic_client=anthropic,
        cache_dir=tmp_path / "llm_cache",
    )
    result = analyzer.evaluate("FAKE", as_of=date(2020, 1, 1))
    assert not result.all_pass
    assert result.fiscal_year is None
    assert "No 10-K filed before" in result.meaning.rationale
    anthropic.messages.create.assert_not_called()


def test_ticker_masked_uses_different_cache_key(
    fake_10k_html: str,
    mock_anthropic_response_pass_all: MagicMock,
    tmp_path: Path,
) -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {}, "cik": 999}
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-23-1", "filingDate": "2023-02-15",
            "reportDate": "2022-12-31", "form": "10-K",
            "primaryDocument": "fy22.htm", "cik": 999,
        },
    ]
    edgar.fetch_filing_document.return_value = fake_10k_html
    anthropic = MagicMock()
    anthropic.messages.create.return_value = mock_anthropic_response_pass_all

    analyzer = FourMsAnalyzer(
        edgar_client=edgar, anthropic_client=anthropic,
        cache_dir=tmp_path / "llm_cache",
    )

    analyzer.evaluate("FAKE", as_of=date(2024, 1, 1), ticker_masked=False)
    analyzer.evaluate("FAKE", as_of=date(2024, 1, 1), ticker_masked=True)

    # Two distinct cache entries (different masking) → two LLM calls.
    assert anthropic.messages.create.call_count == 2
    cached_files = list((tmp_path / "llm_cache").glob("*.json"))
    assert len(cached_files) == 2


def test_coerce_to_dict_passes_through_dicts() -> None:
    d = {"passes": True, "rationale": "ok"}
    assert _coerce_to_dict(d) is d


def test_coerce_to_dict_parses_json_encoded_strings() -> None:
    """Claude occasionally returns a nested field as a JSON-encoded string."""
    s = '{"passes": false, "rationale": "bad", "red_flags": ["a", "b"]}'
    out = _coerce_to_dict(s)
    assert out["passes"] is False
    assert out["rationale"] == "bad"
    assert out["red_flags"] == ["a", "b"]


def test_coerce_to_dict_falls_back_for_plain_strings() -> None:
    """A non-JSON string becomes a fail with the string as rationale."""
    out = _coerce_to_dict("just some prose, not json")
    assert out["passes"] is False
    assert out["rationale"] == "just some prose, not json"


def test_coerce_to_dict_handles_none_and_other_types() -> None:
    assert _coerce_to_dict(None) == {}
    assert _coerce_to_dict(42) == {}


def test_coerce_to_dict_tolerates_trailing_brace_in_json() -> None:
    """The real bug we hit: Claude appended an extra '}' after a valid object."""
    s = '{"passes": false, "rationale": "bad"}\n}'
    out = _coerce_to_dict(s)
    assert out["passes"] is False
    assert out["rationale"] == "bad"


def test_coerce_to_dict_regex_fallback_for_unrepairable_json() -> None:
    """Last-resort: extract passes and rationale via regex even from broken JSON."""
    s = 'random preamble "passes": true, more junk "rationale": "good stuff" trailing'
    out = _coerce_to_dict(s)
    assert out["passes"] is True
    assert out["rationale"] == "good stuff"


def test_evaluate_handles_string_field_in_response(
    fake_10k_html: str,
    tmp_path: Path,
) -> None:
    """If the LLM returns 'management' as a JSON string (Claude misbehavior),
    the analyzer should still produce a valid result rather than crash."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {
        "meaning": {"passes": True, "rationale": "ok"},
        "moat": {"passes": True, "moat_type": "brand", "rationale": "ok"},
        "management": '{"passes": false, "rationale": "red flags exist"}',  # STRING
    }
    response = MagicMock()
    response.content = [block]

    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {}, "cik": 999}
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-23-1", "filingDate": "2023-02-15",
            "reportDate": "2022-12-31", "form": "10-K",
            "primaryDocument": "fy22.htm", "cik": 999,
        },
    ]
    edgar.fetch_filing_document.return_value = fake_10k_html
    anthropic = MagicMock()
    anthropic.messages.create.return_value = response

    analyzer = FourMsAnalyzer(
        edgar_client=edgar, anthropic_client=anthropic,
        cache_dir=tmp_path / "llm_cache",
    )
    result = analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))

    assert result.meaning.passes is True
    assert result.moat.passes is True
    # The string-encoded management was correctly decoded.
    assert result.management.passes is False
    assert "red flags" in result.management.rationale


def test_cache_payload_is_valid_json(
    fake_10k_html: str,
    mock_anthropic_response_pass_all: MagicMock,
    tmp_path: Path,
) -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {}, "cik": 999}
    edgar.list_filings.return_value = [
        {
            "accessionNumber": "0001-23-1", "filingDate": "2023-02-15",
            "reportDate": "2022-12-31", "form": "10-K",
            "primaryDocument": "fy22.htm", "cik": 999,
        },
    ]
    edgar.fetch_filing_document.return_value = fake_10k_html
    anthropic = MagicMock()
    anthropic.messages.create.return_value = mock_anthropic_response_pass_all

    analyzer = FourMsAnalyzer(
        edgar_client=edgar, anthropic_client=anthropic,
        cache_dir=tmp_path / "llm_cache",
    )
    analyzer.evaluate("FAKE", as_of=date(2024, 1, 1))

    cached_files = list((tmp_path / "llm_cache").glob("*.json"))
    assert len(cached_files) == 1
    payload = json.loads(cached_files[0].read_text())
    assert "meaning" in payload
    assert "moat" in payload
    assert "management" in payload
