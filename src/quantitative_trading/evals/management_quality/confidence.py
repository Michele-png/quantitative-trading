"""Confidence scoring for the multi-document Management pipeline.

The screening backend already emits a rich ``management_evidence`` blob
(per-subcheck status, source coverage, cited quotes, LLM scores). What
it doesn't yet emit is a single number that says "how much should the
reader trust this evaluation?". This module fills the gap by
collapsing the evidence blob into a ``ManagementConfidence`` dataclass
with five named components:

    * **Source coverage** — fraction of required documents present per
      sub-check, averaged across sub-checks.
    * **Evidence richness** — average number of cited quotes / examples
      / metrics per sub-check, normalised by an empirical maximum.
    * **Rubric margin** — how far each LLM score sits from its pass
      threshold; wide margins → confident pass/fail.
    * **Agreement** — placeholder for multi-run / multi-model
      agreement. Returns ``None`` until each evaluation is run twice
      (Phase 6 future work).
    * **Calibration** — placeholder for empirical precision / recall on
      the labelled validation set. Returns ``None`` until the rubric
      labels are populated.

The components are weighted into an ``overall`` score by
``ConfidenceWeights``; the default weights are documented inline.

Design notes
------------
* The function takes a *plain dict* (``ManagementEvidence``-shaped) so
  it can be applied at the dashboard, the screening orchestrator, or
  an offline eval without coupling to the dataclasses in
  ``management_llm.py``.
* All components return ``None`` when the underlying data is missing
  rather than guessing — the dashboard treats ``None`` as "data not
  yet wired up" instead of "no confidence".
* The function never mutates the input. Idempotent, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Empirical caps used to normalise raw counts onto the [0, 1] range.
# Tuned against the prompts in ``management_llm.py`` — each tool schema
# typically asks for up to 5 supporting quotes and up to 5 jargon /
# plain-English examples. Sub-checks that don't have a quote-list slot
# (insider, capital_allocation) fall back to a "details length" heuristic.
EVIDENCE_RICHNESS_TARGET = 5

# Per-LLM-subcheck pass threshold. Mirrors the constants in
# ``management_llm.py`` — the rubric margin is computed as how far the
# LLM score is from this threshold.
LLM_THRESHOLDS: dict[str, float] = {
    "blame": 0.0,           # Blame uses scapegoat_count not 1..10
    "long_short": 0.0,      # LongShort uses dominant_orientation not score
    "clarity": 7.0,         # 1..10 scale
    "compensation": 0.0,    # Compensation is boolean, no margin
    "insider": 0.0,         # Insider uses USD net not 1..10
    "capital_allocation": 7.0,  # 1..10 scale
}


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights applied to each component when computing the overall score.

    Defaults emphasise source coverage and rubric margin: a 3/5 score
    on incomplete sources is much less trustworthy than a 4/5 score on
    full sources, even if the headline is the same. Agreement and
    calibration default to 0 because they're placeholders today —
    raise them when the upstream data lands.
    """

    source_coverage: float = 0.4
    evidence_richness: float = 0.2
    rubric_margin: float = 0.3
    agreement: float = 0.05
    calibration: float = 0.05


DEFAULT_CONFIDENCE_WEIGHTS = ConfidenceWeights()


@dataclass(frozen=True)
class ManagementConfidence:
    """Per-evaluation confidence breakdown."""

    overall: float
    """Weighted average of the populated components, in [0, 1]."""

    source_coverage: float | None
    evidence_richness: float | None
    rubric_margin: float | None
    agreement: float | None
    calibration: float | None
    components_used: list[str] = field(default_factory=list)
    """Names of the components that contributed to ``overall``. Useful
    when only some components are populated and the dashboard needs to
    label the score honestly ("3 of 5 components populated")."""


def compute_management_confidence(
    evidence: dict[str, Any] | None,
    *,
    weights: ConfidenceWeights = DEFAULT_CONFIDENCE_WEIGHTS,
    agreement: float | None = None,
    calibration: float | None = None,
) -> ManagementConfidence:
    """Reduce a ``management_evidence`` blob to a confidence breakdown.

    ``agreement`` and ``calibration`` are passed in by the caller so
    multi-run agreement and validation-set calibration can be computed
    offline and injected without re-deriving them here.
    """
    if not evidence or not isinstance(evidence, dict):
        return _empty_confidence()
    subchecks = evidence.get("subchecks") or {}
    coverage = evidence.get("source_coverage") or {}

    source_coverage = _compute_source_coverage(subchecks)
    evidence_richness = _compute_evidence_richness(subchecks)
    rubric_margin = _compute_rubric_margin(subchecks)

    components: dict[str, tuple[float, float]] = {}
    if source_coverage is not None:
        components["source_coverage"] = (source_coverage, weights.source_coverage)
    if evidence_richness is not None:
        components["evidence_richness"] = (
            evidence_richness, weights.evidence_richness,
        )
    if rubric_margin is not None:
        components["rubric_margin"] = (rubric_margin, weights.rubric_margin)
    if agreement is not None:
        components["agreement"] = (agreement, weights.agreement)
    if calibration is not None:
        components["calibration"] = (calibration, weights.calibration)

    if not components:
        return _empty_confidence()

    total_weight = sum(w for _, w in components.values())
    overall = sum(v * w for v, w in components.values()) / total_weight

    return ManagementConfidence(
        overall=_clip(overall),
        source_coverage=source_coverage,
        evidence_richness=evidence_richness,
        rubric_margin=rubric_margin,
        agreement=agreement,
        calibration=calibration,
        components_used=sorted(components.keys()),
    )


# --------------------------------------------------------------------------
# Component-level helpers
# --------------------------------------------------------------------------


def _compute_source_coverage(subchecks: dict[str, Any]) -> float | None:
    """Average fraction of required sources available per sub-check.

    Uses the ``required_sources`` and ``missing_sources`` arrays the
    evidence builder already populates. A sub-check with all required
    sources present scores 1.0; one with half missing scores 0.5; one
    with no required sources (rare) is skipped — counting it would
    bias the average upward.
    """
    if not subchecks:
        return None
    fractions: list[float] = []
    for sub in subchecks.values():
        if not isinstance(sub, dict):
            continue
        required = sub.get("required_sources") or []
        missing = sub.get("missing_sources") or []
        if not required:
            continue
        present = max(0, len(required) - len(missing))
        fractions.append(present / len(required))
    if not fractions:
        return None
    return sum(fractions) / len(fractions)


def _compute_evidence_richness(subchecks: dict[str, Any]) -> float | None:
    """Average count of cited evidence items per sub-check, capped at 1.0.

    For sub-checks with quote-list slots (blame, long_short, clarity,
    capital_allocation) we count those entries. For sub-checks without
    explicit quotes we count the populated keys in ``details`` as a
    proxy for "how much structured evidence the LLM produced".
    """
    if not subchecks:
        return None
    norms: list[float] = []
    for key, sub in subchecks.items():
        if not isinstance(sub, dict):
            continue
        details = sub.get("evidence") or {}
        # Count the most-informative slot for each subcheck.
        if key == "blame":
            n = len(details.get("supporting_quotes") or [])
        elif key == "long_short":
            n = (
                int(details.get("long_term_mentions") or 0)
                + int(details.get("short_term_mentions") or 0)
            )
        elif key == "clarity":
            n = (
                len(details.get("plain_english_examples") or [])
                + len(details.get("jargon_examples") or [])
            )
        elif key == "compensation":
            n = (
                len(details.get("metrics") or [])
                + len(details.get("shareholder_aligned_metrics") or [])
                + len(details.get("empire_building_metrics") or [])
            )
        elif key == "capital_allocation":
            n = (
                len(details.get("stated_priorities") or [])
                + len(details.get("capital_misallocation_flags") or [])
            )
        elif key == "insider":
            # Form 4 transactions are the natural unit.
            n = int(details.get("n_transactions") or 0)
        else:
            n = sum(1 for v in details.values() if v not in (None, "", [], {}))
        norms.append(min(1.0, n / EVIDENCE_RICHNESS_TARGET))
    if not norms:
        return None
    return sum(norms) / len(norms)


def _compute_rubric_margin(subchecks: dict[str, Any]) -> float | None:
    """Average normalised distance of each LLM score from its threshold.

    Only LLM-driven sub-checks with a numeric ``score`` participate
    (clarity, capital_allocation). The margin is normalised by the
    maximum possible distance (10 - threshold for the pass side, or
    threshold itself for the fail side) so all sub-checks contribute
    on the same scale.
    """
    if not subchecks:
        return None
    margins: list[float] = []
    for key, sub in subchecks.items():
        if not isinstance(sub, dict):
            continue
        threshold = LLM_THRESHOLDS.get(key)
        if not threshold:
            continue
        score = sub.get("score")
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f >= threshold:
            margin = (score_f - threshold) / max(1e-6, 10.0 - threshold)
        else:
            margin = (threshold - score_f) / max(1e-6, threshold)
        margins.append(min(1.0, max(0.0, margin)))
    if not margins:
        return None
    return sum(margins) / len(margins)


def _empty_confidence() -> ManagementConfidence:
    return ManagementConfidence(
        overall=0.0,
        source_coverage=None,
        evidence_richness=None,
        rubric_margin=None,
        agreement=None,
        calibration=None,
        components_used=[],
    )


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))
