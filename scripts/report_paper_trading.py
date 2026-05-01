"""Print recent paper-trading ledger events."""

from __future__ import annotations

import argparse
import json
import sys

from quantitative_trading.config import init_env
from quantitative_trading.paper_trading.config import get_paper_trading_config
from quantitative_trading.paper_trading.ledger import PaperTradingLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    init_env()
    cfg = get_paper_trading_config()
    ledger = PaperTradingLedger(cfg.ledger_path)
    for event in ledger.latest_events(limit=args.limit):
        print(json.dumps(event, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
