"""Run the backtest ablation on a built dataset.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --dataset data/dataset/dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from quantitative_trading.backtest.engine import run_backtest
from quantitative_trading.config import init_env


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=None,
                   help="Path to dataset.parquet (default: cfg.dataset_dir/dataset.parquet)")
    p.add_argument("--no-spy", action="store_true",
                   help="Skip the SPY benchmark column (faster, no extra yfinance fetch)")
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--target-cagr", type=float, default=0.15)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    init_env()

    report = run_backtest(
        dataset_path=args.dataset,
        add_spy=not args.no_spy,
        label_horizon_years=args.label_horizon,
        target_cagr=args.target_cagr,
    )

    print("\n=== CLASSIFICATION METRICS ===")
    print(report.classification.to_string(index=False))
    print("\n=== PORTFOLIO METRICS ===")
    print(report.portfolio.to_string(index=False))
    print(f"\nBase rate (P(label_passes)): {report.base_rate:.3f}  "
          f"on {report.n_eligible} eligible rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
