"""End-to-end pipeline: 13F download -> CUSIP resolve -> Big 5 score -> control sample.

Outputs
-------
* `data/investors/investor_purchases_audit_raw.csv` -- one row per (investor,
  cusip, period_of_report) with all 7 booleans + non_evaluable_reason.
* `data/investors/investor_purchases_audit.csv` -- same, enriched with
  realized_cagr_to_exit / holding_period_quarters / is_right_censored.
* `data/investors/control_sample.csv` -- one row per (elite_buy, control)
  with the same 7 booleans for the matched-control stock.
* `data/investors/cusip_disagreements.csv` -- CUSIPs that OpenFIGI couldn't
  resolve to a US-primary listing AND that the SEC name-match fallback
  didn't catch. For manual review (audit plan section 10.5).

Usage
-----
    python -m scripts.build_investor_purchases [--start 2017-01-01] [--end 2024-12-31]
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

from quantitative_trading.config import init_env
from quantitative_trading.dataset.investor_purchases_dataset import (
    build_audit_dataset,
    enrich_with_realized_returns,
)
from quantitative_trading.dataset.matched_control_sampler import sample_controls
from quantitative_trading.investors.investor_universe import INVESTORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-01",
                        help="Window start (YYYY-MM-DD). Default 2017-01-01 (audit plan section 3).")
    parser.add_argument("--end", default="2024-12-31",
                        help="Window end (YYYY-MM-DD). Default 2024-12-31.")
    parser.add_argument("--k", type=int, default=10,
                        help="Controls per elite buy. Default 10 (audit plan section 6).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/investors"),
                        help="Output directory.")
    parser.add_argument("--skip-controls", action="store_true",
                        help="Skip control sampling (much faster).")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip realized-return enrichment.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    init_env()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_start = date.fromisoformat(args.start)
    window_end = date.fromisoformat(args.end)

    print(f"[1/3] Building elite-buy dataset for {len(INVESTORS)} investors, "
          f"window {window_start} -- {window_end}")
    t0 = time.time()
    elite = build_audit_dataset(
        INVESTORS,
        window_start=window_start,
        window_end=window_end,
        output_csv=args.out_dir / "investor_purchases_audit_raw.csv",
    )
    print(f"  -> {len(elite)} rows in {time.time() - t0:.0f}s")

    if not args.skip_enrich:
        print("[2/3] Enriching with realized returns (holding period + CAGR + censoring)")
        t0 = time.time()
        elite_enriched = enrich_with_realized_returns(
            elite, INVESTORS, window_end=window_end,
        )
        elite_enriched.to_csv(args.out_dir / "investor_purchases_audit.csv", index=False)
        print(f"  -> enriched in {time.time() - t0:.0f}s")
    else:
        print("[2/3] Skipping enrichment (--skip-enrich)")

    if not args.skip_controls:
        print(f"[3/3] Sampling {args.k} matched controls per evaluable elite buy")
        t0 = time.time()
        controls = sample_controls(
            elite, k_per_buy=args.k,
            output_csv=args.out_dir / "control_sample.csv",
        )
        print(f"  -> {len(controls)} control rows in {time.time() - t0:.0f}s")
    else:
        print("[3/3] Skipping control sampling (--skip-controls)")

    print()
    print("=== Outputs ===")
    print(f"  {args.out_dir / 'investor_purchases_audit_raw.csv'}")
    if not args.skip_enrich:
        print(f"  {args.out_dir / 'investor_purchases_audit.csv'}")
    if not args.skip_controls:
        print(f"  {args.out_dir / 'control_sample.csv'}")
    print(f"  {args.out_dir / 'cusip_disagreements.csv'}  (review for manual overrides)")
    print()
    print("Next: python -m scripts.run_investor_audit")


if __name__ == "__main__":
    main()
