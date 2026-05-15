"""Management Quality validation set scaffolding (Phase 6).

See ``README.md`` in this directory for the methodology and rationale.
The submodules are deliberately decoupled from the screening
orchestrator so the calibration loop can iterate without touching the
hot path:

    * ``confidence`` — pure functions that turn an evidence blob into a
      confidence score (tested under ``tests/evals/test_confidence.py``).
    * ``build_dataset`` — CLI that generates / refreshes the weak
      positive + matched-control ticker list.
    * ``run_eval`` — CLI that scores the current Management pipeline
      against the human-labelled ``labels.jsonl``.
"""

from quantitative_trading.evals.management_quality.confidence import (
    DEFAULT_CONFIDENCE_WEIGHTS,
    ConfidenceWeights,
    ManagementConfidence,
    compute_management_confidence,
)

__all__ = [
    "ConfidenceWeights",
    "DEFAULT_CONFIDENCE_WEIGHTS",
    "ManagementConfidence",
    "compute_management_confidence",
]
