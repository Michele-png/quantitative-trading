# Management Quality Validation Set

Phase 6 scaffold for tuning and calibrating the multi-document Management
pipeline (`agents/rule_one/management_llm.py`).

## Why a validation set is needed

Today every Management sub-check produces a binary `passes` flag without
any sense of the model's certainty. We don't know whether a 7/10 clarity
score actually maps to "this CEO writes clearly" or whether the LLM is
guessing. Until we have ground-truth labels we can't:

* tune thresholds (e.g. `clarity >= 7` vs `>= 8`),
* report confidence intervals on the gate output,
* spot regressions from prompt changes,
* compare candidate models (Sonnet vs Opus, prompt v1 vs v2).

This directory holds the artifacts needed to do all four.

## Why Buffett/Town purchases alone are NOT the ground truth

The user proposed using "stocks Buffett or Phil Town purchased over the
past N years" as positive labels for management quality. That signal is
useful but **not sufficient on its own**:

1. A purchase decision encodes price, moat, expected return, portfolio
   sizing, liquidity, tax impact, and mandate constraints — not just
   management quality. Berkshire bought AAPL when management was great
   *and* the multiple was reasonable. Either factor flipping can flip
   the purchase decision.
2. Berkshire holds positions in companies where management quality is
   merely "good enough" rather than "exemplary" (think IBM, KHC).
3. The set is small, public-equity-only, and biased toward consumer /
   financial / industrial sectors.
4. Non-purchases are not "bad management" — Buffett doesn't buy
   semiconductors regardless of management.

Treat purchases as **weak positives** that anchor one end of the
distribution; treat human-rubric labels as the ground truth for
training-style decisions.

## Dataset design

Two complementary sources of labels:

### A. Weak positives — Buffett / Phil Town purchases

* Berkshire 13F: new positions and material adds in the last 8 quarters.
  Source: SEC EDGAR Form 13F-HR (CIK 0001067983).
* Phil Town public picks: Rule One Investing podcast, blog, and Town's
  public portfolio disclosures. Tag with `source = "phil_town"`.
* Match each weak positive against the same-sector, similar-market-cap
  set so the validation task isn't trivially "is this a famous stock?".

### B. Strong labels — explicit human rubric

For each ticker on the list, a human reviewer fills in a rubric in
`labels.jsonl` with one score per dimension:

| Field | Range | Meaning |
|-------|-------|---------|
| `candor` | 0–10 | Earnings-call accountability, post-mistake honesty |
| `capital_allocation` | 0–10 | Discipline in reinvestment / M&A / buybacks / dividends |
| `owner_orientation` | 0–10 | Per-share metrics, long-horizon framing, founder/operator alignment |
| `compensation_alignment` | 0–10 | CEO comp tied to per-share value vs empire-building |
| `insider_alignment` | 0–10 | Meaningful insider ownership / open-market buying |
| `source_quality` | 0–10 | How complete and recent the source documents were |

`pass_overall` is a derived boolean — `True` iff every dimension ≥ 6.

## Files

* `labels.jsonl` — one ticker per line. Seed entries cover a handful of
  Buffett purchases (AAPL, KO, AXP) and matched controls. Extend by
  hand; do **not** auto-generate the human-rubric scores.
* `build_dataset.py` — fetches Berkshire 13F and assembles the weak
  positive list with matched controls. Does not produce rubric scores —
  it only emits ticker + metadata rows the human can fill in.
* `confidence.py` — computes per-evaluation confidence scores from the
  evidence blob. Tested under `tests/evals/test_confidence.py`. Keeps
  the methodology decoupled from the screening orchestrator so it can
  be re-applied at any point in the pipeline.
* `run_eval.py` — CLI: runs the management analyzer over `labels.jsonl`,
  compares predicted `pass_overall` against the human label, and emits
  per-dimension precision/recall + an aggregate confusion matrix.

## Confidence score

`confidence.py` exposes `compute_management_confidence(evidence)` which
returns a `ManagementConfidence` dataclass with:

* `source_coverage` — fraction of required documents present per
  sub-check (averaged).
* `evidence_richness` — average number of cited quotes / examples per
  sub-check, normalised by an empirical maximum.
* `rubric_margin` — distance of each LLM score from its pass threshold,
  normalised. Wider margins → more confident pass/fail.
* `agreement` — placeholder for multi-run agreement (None until we
  start running each evaluation twice).
* `calibration` — placeholder for empirical precision/recall (None
  until the validation set is labelled).
* `overall` — weighted average of the above; the dashboard surfaces
  this next to the management score so the reader knows when a 4/5
  score is "barely scraped" vs "clear consensus".

The placeholders are explicit so future work can light them up without
touching every call-site — see `confidence.py` for the wiring.

## Workflow

1. Run `python build_dataset.py` to generate / update the ticker list.
2. Hand-label the new rows in `labels.jsonl`.
3. Run `python run_eval.py` to score the current pipeline against the
   labels. Outputs go under `evals/management_quality/results/`.
4. Iterate on prompts / thresholds; re-run.

The eval is not part of CI by default — it makes real LLM calls and is
slow. Treat it as an offline calibration loop, run before each prompt
or model change.
