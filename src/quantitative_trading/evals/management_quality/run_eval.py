"""Run the Management pipeline against the labelled validation set.

Reads ``labels.jsonl``, runs ``RuleOneAgent.evaluate`` for each ticker
at the labelled ``as_of``, and compares the predicted ``passes`` flag
plus the per-sub-check pass flags against the human rubric. Emits:

    * a JSON report under ``evals/management_quality/results/<ts>.json``
    * a printed precision / recall summary per sub-check
    * an aggregate confusion matrix on the overall pass

Cost
----
This is the only evaluation in the project that hits the real LLM —
each ticker triggers 5 Anthropic calls (blame, long-short, clarity,
compensation, capital_allocation). Run sparingly: the 8-row seed set
costs ~$2-4 in Opus 4.7 with the standard prompt.

Usage
-----
::

    python -m quantitative_trading.evals.management_quality.run_eval \\
        --labels evals/management_quality/labels.jsonl \\
        --max-tickers 5

``--max-tickers`` makes it cheap to iterate. ``--dry-run`` skips the
LLM and the run produces an empty results file — useful to verify
the wiring without spending API credits.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import click

from quantitative_trading.evals.management_quality.confidence import (
    compute_management_confidence,
)


log = logging.getLogger("evals.management_quality.run_eval")


@click.command()
@click.option(
    "--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).parent / "labels.jsonl",
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path),
    default=Path(__file__).parent / "results",
)
@click.option("--max-tickers", type=int, default=None)
@click.option("--dry-run", is_flag=True)
def main(
    *,
    labels: Path,
    out_dir: Path,
    max_tickers: int | None,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    rows = _load_labels(labels)
    if max_tickers is not None:
        rows = rows[:max_tickers]
    log.info("Loaded %d labelled rows from %s", len(rows), labels)

    runner = _load_management_analyzer(dry_run=dry_run)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for row in rows:
        ticker = row["ticker"]
        as_of = date.fromisoformat(row["as_of"])
        log.info("Evaluating %s @ %s", ticker, as_of)
        try:
            result = runner(ticker, as_of)
        except Exception as exc:  # noqa: BLE001 - validation must not crash
            log.exception("evaluate(%s @ %s) failed", ticker, as_of)
            results.append({
                "ticker": ticker, "as_of": row["as_of"],
                "error": str(exc), "label": row.get("rubric"),
            })
            continue
        results.append(_build_result_row(row, result))

    metrics = _aggregate_metrics(results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(results),
        "metrics": metrics,
        "rows": results,
    }
    out_path = out_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", out_path)

    _print_summary(metrics)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _load_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Dropping malformed JSONL line: %r", line)
    return rows


def _load_management_analyzer(*, dry_run: bool):
    """Lazy-import the agent so a dry-run doesn't require API keys."""
    if dry_run:
        def stub(ticker: str, as_of: date):  # noqa: ARG001 - stub
            return None
        return stub

    # Real wiring is intentionally minimal — this script is meant to be
    # run by hand, not from CI, so we accept the import cost up-front.
    from anthropic import Anthropic  # noqa: F401  - imported for side-effects
    from quantitative_trading.agents.rule_one.agent import RuleOneAgent
    from quantitative_trading.config import get_config
    from quantitative_trading.data.edgar import EdgarClient
    from quantitative_trading.data.prices import PriceClient

    cfg = get_config()
    client = Anthropic(api_key=cfg.anthropic_api_key)
    agent = RuleOneAgent(EdgarClient(), PriceClient(), anthropic_client=client)

    def runner(ticker: str, as_of: date):
        return agent.evaluate(
            ticker, as_of,
            include_llm=True, include_extras=True, include_management=True,
        )
    return runner


def _build_result_row(row: dict[str, Any], result: Any) -> dict[str, Any]:
    if result is None:
        return {**row, "predicted": None}
    mgmt = getattr(result, "management", None)
    evidence_blob = (
        result.management_evidence  # type: ignore[attr-defined]
        if hasattr(result, "management_evidence") else None
    )
    confidence = compute_management_confidence(evidence_blob)
    return {
        "ticker": row["ticker"],
        "as_of": row["as_of"],
        "label": row.get("rubric"),
        "weak_positive": row.get("weak_positive"),
        "predicted_pass": (
            mgmt.passes if mgmt is not None else None
        ),
        "predicted_per_check": (
            mgmt.per_check if mgmt is not None else None
        ),
        "confidence": {
            "overall": confidence.overall,
            "source_coverage": confidence.source_coverage,
            "evidence_richness": confidence.evidence_richness,
            "rubric_margin": confidence.rubric_margin,
            "components_used": confidence.components_used,
        },
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision / recall against the human ``pass_overall``.

    Rows without a populated rubric are excluded from precision /
    recall (no ground truth) but still counted in the overall row count.
    """
    labelled = [
        r for r in rows
        if isinstance(r.get("label"), dict)
        and r["label"].get("pass_overall") is not None
        and r.get("predicted_pass") is not None
    ]
    if not labelled:
        return {
            "n_labelled": 0,
            "note": (
                "Validation set has no rubric scores yet — metrics "
                "are not computable. See README.md for how to "
                "populate ``labels.jsonl``."
            ),
        }
    tp = sum(
        1 for r in labelled
        if r["label"]["pass_overall"] and r["predicted_pass"]
    )
    fp = sum(
        1 for r in labelled
        if not r["label"]["pass_overall"] and r["predicted_pass"]
    )
    fn = sum(
        1 for r in labelled
        if r["label"]["pass_overall"] and not r["predicted_pass"]
    )
    tn = sum(
        1 for r in labelled
        if not r["label"]["pass_overall"] and not r["predicted_pass"]
    )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "n_labelled": len(labelled),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def _print_summary(metrics: dict[str, Any]) -> None:
    if metrics.get("n_labelled", 0) == 0:
        click.echo("No rubric labels yet — nothing to evaluate.")
        click.echo(metrics.get("note", ""))
        return
    click.echo(
        f"n={metrics['n_labelled']} | "
        f"precision={metrics['precision']:.2f} | "
        f"recall={metrics['recall']:.2f} | "
        f"accuracy={metrics['accuracy']:.2f}",
    )
    click.echo(
        f"confusion: TP={metrics['true_positive']} "
        f"FP={metrics['false_positive']} "
        f"FN={metrics['false_negative']} "
        f"TN={metrics['true_negative']}",
    )


if __name__ == "__main__":
    main()
