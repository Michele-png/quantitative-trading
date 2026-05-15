"""Tests for ``evals.management_quality.confidence``."""

from __future__ import annotations

import pytest

from quantitative_trading.evals.management_quality.confidence import (
    ConfidenceWeights,
    compute_management_confidence,
)


def _full_evidence() -> dict:
    """A management evidence blob with full source coverage and rich
    LLM evidence — should land near the top of the confidence range."""
    return {
        "schema_version": 1,
        "as_of": "2024-01-01",
        "fiscal_year": 2023,
        "model": "claude-opus-4-7",
        "cached": False,
        "bundle_hash": "abc",
        "source_coverage": {
            "transcripts": {"available": True, "count": 8, "expected": 8},
            "def14a_compensation": {"available": True},
            "def14a_letter": {"available": True},
            "shareholder_letter": {"available": True, "from": "def14a"},
            "mda": {"available": True},
            "form4": {"available": True, "n_transactions": 12},
        },
        "subchecks": {
            "blame": {
                "label": "Blame", "status": "pass", "passes": True,
                "score": 0.0, "rationale": "ok",
                "evidence": {
                    "supporting_quotes": [
                        "I made a mistake on inventory forecasting.",
                        "That was on me.",
                        "I underestimated the integration timeline.",
                        "I owe shareholders a clearer roadmap.",
                        "We rushed the launch and that's on me.",
                    ],
                    "scapegoat_count": 0,
                    "takes_responsibility": True,
                },
                "required_sources": ["transcripts"], "missing_sources": [],
            },
            "long_short": {
                "label": "LongShort", "status": "pass", "passes": True,
                "score": 5.0, "rationale": "ok",
                "evidence": {
                    "long_term_mentions": 12, "short_term_mentions": 1,
                    "ratio": 12.0, "dominant_orientation": "long",
                },
                "required_sources": ["transcripts", "shareholder_letter"],
                "missing_sources": [],
            },
            "clarity": {
                "label": "Clarity", "status": "pass", "passes": True,
                "score": 9.0, "rationale": "ok",
                "evidence": {
                    "plain_english_examples": [
                        "We sell two things: software, and trust.",
                        "Our growth came from existing customers buying more.",
                        "We earned $4 per share last year.",
                        "We bought back $3B of stock at an average $42.",
                        "We will pay our debt down before raising the dividend.",
                    ],
                    "jargon_examples": [],
                },
                "required_sources": ["shareholder_letter", "mda"],
                "missing_sources": [],
            },
            "compensation": {
                "label": "Compensation", "status": "pass", "passes": True,
                "score": None, "rationale": "ok",
                "evidence": {
                    "metrics": ["ROIC", "EPS growth", "FCF / share"],
                    "shareholder_aligned_metrics": [
                        "ROIC", "EPS growth", "FCF / share",
                    ],
                    "empire_building_metrics": [],
                    "aligned_with_shareholders": True,
                },
                "required_sources": ["def14a_compensation"],
                "missing_sources": [],
            },
            "insider": {
                "label": "Insider", "status": "pass", "passes": True,
                "score": 1_000_000.0, "rationale": "ok",
                "evidence": {
                    "n_transactions": 8,
                    "net_open_market_value_usd": 1_000_000.0,
                },
                "required_sources": ["form4"], "missing_sources": [],
            },
            "capital_allocation": {
                "label": "CapitalAllocation", "status": "pass", "passes": True,
                "score": 9.0, "rationale": "ok",
                "evidence": {
                    "stated_priorities": ["reinvestment", "buybacks"],
                    "discipline_score": 9,
                    "capital_misallocation_flags": [],
                },
                "required_sources": ["mda", "shareholder_letter"],
                "missing_sources": [],
            },
        },
    }


def _degraded_evidence() -> dict:
    """Half the sources missing, sub-checks degraded to NO_DATA."""
    base = _full_evidence()
    base["source_coverage"]["transcripts"]["available"] = False
    base["source_coverage"]["transcripts"]["count"] = 0
    base["source_coverage"]["def14a_compensation"]["available"] = False
    base["source_coverage"]["form4"]["available"] = False
    base["subchecks"]["blame"]["status"] = "no_data"
    base["subchecks"]["blame"]["missing_sources"] = ["transcripts"]
    base["subchecks"]["compensation"]["status"] = "no_data"
    base["subchecks"]["compensation"]["missing_sources"] = ["def14a_compensation"]
    base["subchecks"]["insider"]["status"] = "no_data"
    base["subchecks"]["insider"]["missing_sources"] = ["form4"]
    return base


def test_full_evidence_scores_near_top() -> None:
    conf = compute_management_confidence(_full_evidence())
    assert conf.overall > 0.85
    assert conf.source_coverage == 1.0
    assert conf.evidence_richness is not None and conf.evidence_richness >= 0.7
    assert conf.rubric_margin is not None and conf.rubric_margin > 0.0
    assert "source_coverage" in conf.components_used
    assert "evidence_richness" in conf.components_used
    assert "rubric_margin" in conf.components_used


def test_degraded_evidence_scores_in_the_middle() -> None:
    conf = compute_management_confidence(_degraded_evidence())
    # Three of six sub-checks are missing their primary source.
    assert conf.source_coverage is not None
    assert conf.source_coverage < 0.9
    assert conf.overall < 0.85
    # Rubric margin still computable from clarity + capital_allocation.
    assert conf.rubric_margin is not None


def test_empty_evidence_returns_zero_overall() -> None:
    conf = compute_management_confidence(None)
    assert conf.overall == 0.0
    assert conf.components_used == []


def test_explicit_calibration_and_agreement_lift_overall() -> None:
    """Plumb calibration / agreement values through and confirm they
    contribute to the overall score even though the auto components
    don't compute them."""
    base = compute_management_confidence(_full_evidence())
    boosted = compute_management_confidence(
        _full_evidence(),
        agreement=1.0, calibration=1.0,
    )
    assert "agreement" in boosted.components_used
    assert "calibration" in boosted.components_used
    assert boosted.overall >= base.overall


def test_custom_weights_change_overall() -> None:
    """Switching the weights should shift the overall score in the
    direction of the heavily-weighted component."""
    ev = _full_evidence()
    coverage_weighted = compute_management_confidence(
        ev, weights=ConfidenceWeights(
            source_coverage=1.0, evidence_richness=0.0,
            rubric_margin=0.0, agreement=0.0, calibration=0.0,
        ),
    )
    margin_weighted = compute_management_confidence(
        ev, weights=ConfidenceWeights(
            source_coverage=0.0, evidence_richness=0.0,
            rubric_margin=1.0, agreement=0.0, calibration=0.0,
        ),
    )
    assert coverage_weighted.overall == pytest.approx(1.0, abs=1e-9)
    # Margin is < 1 because the per-subcheck margins are normalised.
    assert margin_weighted.overall < 1.0


def test_handles_subcheck_with_no_required_sources() -> None:
    """A sub-check with an empty ``required_sources`` array should be
    skipped from the source-coverage average rather than count as 0/0."""
    ev = _full_evidence()
    ev["subchecks"]["blame"]["required_sources"] = []
    ev["subchecks"]["blame"]["missing_sources"] = []
    conf = compute_management_confidence(ev)
    # Blame is excluded from the average — coverage still 1.0 for the
    # other 5 sub-checks.
    assert conf.source_coverage == 1.0


def test_rubric_margin_handles_score_below_threshold() -> None:
    """A clarity score of 4/10 (well below the pass floor of 7)
    contributes a meaningful margin in the failing direction."""
    ev = _full_evidence()
    ev["subchecks"]["clarity"]["score"] = 4.0
    ev["subchecks"]["clarity"]["passes"] = False
    ev["subchecks"]["clarity"]["status"] = "fail"
    conf = compute_management_confidence(ev)
    assert conf.rubric_margin is not None
    # We still get a non-zero margin (4 is 3pp away from the 7 threshold).
    assert conf.rubric_margin > 0.1
