"""Run the weekly zero-shot LLM paper-trading workflow.

Examples:
    python scripts/run_weekly_paper_trade.py --dry-run
    python scripts/run_weekly_paper_trade.py --execute
    python scripts/run_weekly_paper_trade.py --dry-run --symbols AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from quantitative_trading.config import init_env
from quantitative_trading.paper_trading import WeeklyPaperTrader, get_paper_trading_config


def _parse_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a decision without placing orders.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Close old positions and submit the paper order.",
    )
    parser.add_argument("--trade-week", help="Monday trade-week date, e.g. 2026-05-04.")
    parser.add_argument("--symbols", help="Comma-separated ticker subset for smoke tests.")
    parser.add_argument(
        "--allow-closed-market",
        action="store_true",
        help="Allow execute outside market hours.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    init_env()
    cfg = get_paper_trading_config()
    trader = WeeklyPaperTrader(config=cfg)
    result = trader.run(
        dry_run=not args.execute,
        trade_week=_parse_date(args.trade_week),
        symbols=_parse_symbols(args.symbols),
        require_market_open=not args.allow_closed_market,
    )

    print(f"trade_week={result.trade_week.isoformat()} dry_run={result.dry_run}")
    print(f"decision={result.decision.ticker} confidence={result.decision.confidence:.2f}")
    print(f"thesis={result.decision.thesis}")
    print(f"risks={'; '.join(result.decision.risks)}")
    print(f"notional_usd={result.order_plan.notional_usd:.2f}")
    print(f"estimated_buy_costs_usd={result.estimated_buy_costs.total_usd:.2f}")
    print(f"eur_usd={result.fx_rate_eur_usd:.4f}")
    if result.order_result is not None:
        print(f"order_id={result.order_result.order_id} status={result.order_result.status}")
    print(f"ledger={result.ledger_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
