"""Assemble the Management Quality validation set.

Produces a ``labels.jsonl`` skeleton with one row per ticker and an
empty ``rubric`` slot for the human reviewer to fill in. Two cohorts:

* **Weak positives** — Berkshire Hathaway 13F new-positions and adds
  in the last ``--lookback-quarters`` quarters. Sourced from SEC
  EDGAR. Tag: ``source = "berkshire_13f"``.
* **Matched controls** — same-sector, similar-market-cap tickers from
  the universe (``etl/tickers.yml``). Tag: ``source = "matched_control"``.

Does NOT call any LLM. Does NOT auto-fill rubric scores — those are
human work, see ``README.md``.

Usage::

    python -m quantitative_trading.evals.management_quality.build_dataset \\
        --out evals/management_quality/labels.jsonl \\
        --lookback-quarters 8

The script is intentionally idempotent: if a ticker is already in the
output file with a populated ``rubric`` we keep the existing entry
unchanged. Re-running just adds new tickers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click


log = logging.getLogger("evals.management_quality.build_dataset")


@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(__file__).parent / "labels.jsonl",
    help="Path to write the labels JSONL.",
)
@click.option(
    "--lookback-quarters", type=int, default=8,
    help="How many quarters of Berkshire 13F to scan for weak positives.",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Print the rows that would be added; don't touch the file.",
)
def main(*, out: Path, lookback_quarters: int, dry_run: bool) -> None:
    """Top-level entry point — see module docstring for behaviour."""
    logging.basicConfig(level=logging.INFO)

    existing = _load_existing(out)
    log.info("Loaded %d existing labels from %s", len(existing), out)

    weak_positives = _build_berkshire_weak_positives(lookback_quarters)
    log.info(
        "Berkshire weak positives last %d quarters: %d tickers",
        lookback_quarters, len(weak_positives),
    )

    controls = _build_matched_controls(weak_positives)
    log.info("Matched controls: %d tickers", len(controls))

    new_rows = _merge_rows(existing, weak_positives + controls)
    if dry_run:
        for row in new_rows:
            print(json.dumps(row))
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in new_rows:
            fh.write(json.dumps(row) + "\n")
    log.info("Wrote %d rows to %s", len(new_rows), out)


# --------------------------------------------------------------------------
# Helpers — most of these are stubs that document what needs to happen.
# Filling them in is part of the validation-set workstream and depends
# on the SEC 13F XML parser already available in ``data/edgar.py``.
# --------------------------------------------------------------------------


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _build_berkshire_weak_positives(lookback_quarters: int) -> list[dict[str, Any]]:
    """Pull Berkshire's recent 13F adds.

    Implementation deferred — uses ``EdgarClient.list_filings`` for
    Berkshire's CIK (0001067983) form 13F-HR. Compare each quarter's
    holdings to the prior quarter to detect new positions and material
    adds, then map CUSIPs to tickers via ``investors/cusip_resolver.py``.
    """
    log.warning(
        "_build_berkshire_weak_positives is not implemented yet — "
        "see README.md. Returning empty list so the CLI doesn't crash.",
    )
    _ = lookback_quarters  # silence linter; will be used once implemented
    return []


def _build_matched_controls(
    positives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick same-sector / similar-cap controls from the dashboard universe.

    Implementation deferred — read ``quantitative-trading-dashboard/
    etl/tickers.yml``, group by sector / market-cap band, and emit one
    control per positive. The control should NOT be on Berkshire's
    13F so the validation set isn't trivially "is this on the list?".
    """
    log.warning(
        "_build_matched_controls is not implemented yet — "
        "see README.md. Returning empty list so the CLI doesn't crash.",
    )
    _ = positives
    return []


def _merge_rows(
    existing: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep populated existing rows, add any new ones from ``proposed``.

    A row is keyed by ``(ticker, as_of)``. When both files have the
    same key, the existing row wins so the human's rubric scores are
    never overwritten by automated re-runs.
    """
    keyed: dict[tuple[str, str], dict[str, Any]] = {
        (r["ticker"], r["as_of"]): r for r in existing
    }
    for row in proposed:
        key = (row["ticker"], row["as_of"])
        if key not in keyed:
            keyed[key] = row
    return sorted(keyed.values(), key=lambda r: (r["ticker"], r["as_of"]))


if __name__ == "__main__":
    main()
