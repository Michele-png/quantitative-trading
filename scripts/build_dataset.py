"""Build the historical (ticker, trade_date) dataset.

Examples:
    # Tiny smoke test: 10 tickers, 4 trade dates, no LLM (free)
    python scripts/build_dataset.py --start 2018 --end 2018 --sample 10 --no-llm

    # Full run: all S&P 500 history members, 10 years, with LLM
    python scripts/build_dataset.py --start 2012 --end 2021

    # Resume after interruption (chunks already on disk are skipped)
    python scripts/build_dataset.py --start 2012 --end 2021 --resume
"""

from __future__ import annotations

import argparse
import logging
import sys

from quantitative_trading.config import init_env
from quantitative_trading.dataset.builder import build_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2012, help="Start year (inclusive)")
    p.add_argument("--end", type=int, default=2021, help="End year (inclusive)")
    p.add_argument(
        "--sample", type=int, default=None,
        help="If set, only process the first N tickers per trade date (for testing)",
    )
    p.add_argument(
        "--no-llm", action="store_true",
        help="Skip the LLM 4Ms (free; useful for quick iteration)",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip trade-date chunks that already exist on disk (default ON)",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force re-run all chunks (overwrites existing)",
    )
    p.add_argument(
        "--label-horizon", type=int, default=5,
        help="Forward CAGR horizon in years (default 5)",
    )
    p.add_argument(
        "--target-cagr", type=float, default=0.15,
        help="Label threshold (default 0.15 = Phil Town's 15%/yr target)",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
    )
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    init_env()

    out = build_dataset(
        start_year=args.start,
        end_year=args.end,
        include_llm=not args.no_llm,
        sample_size=args.sample,
        skip_existing=args.resume,
        label_horizon_years=args.label_horizon,
        target_cagr=args.target_cagr,
    )
    print(f"\nDataset written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
