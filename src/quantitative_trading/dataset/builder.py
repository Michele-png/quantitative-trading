"""Historical dataset builder.

For each (ticker, trade_date) in the universe × calendar grid, run the agent
and the label computation, then write a row to a parquet file.

Resumability: the builder writes one parquet *per chunk* under
`dataset_dir/chunks/`, keyed by trade_date. Re-running skips chunks that
already exist. To force a full rebuild, delete the chunk files.

Final consolidation is done by `consolidate()`, which merges all chunks into
a single `dataset_dir/dataset.parquet`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm

from quantitative_trading.agents.rule_one.agent import AgentResult, RuleOneAgent
from quantitative_trading.config import get_config
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.prices import PriceClient
from quantitative_trading.data.universe import SP500Universe
from quantitative_trading.dataset.labels import (
    DEFAULT_LABEL_HORIZON_YEARS,
    DEFAULT_TARGET_CAGR,
    LabelResult,
    compute_label,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------- Trade dates


def generate_trade_dates(
    start_year: int,
    end_year: int,
    months: tuple[int, ...] = (3, 6, 9, 12),
    day: int = 15,
) -> list[date]:
    """Quarterly trade-date grid: e.g., 15th of Mar/Jun/Sep/Dec each year.

    Defaults pick the 15th — well after the typical 10-Q filing deadline (40
    days for large accelerated filers) so the agent's PIT data is fresh.
    """
    out: list[date] = []
    for year in range(start_year, end_year + 1):
        for month in months:
            out.append(date(year, month, day))
    return out


# --------------------------------------------------------------------- Row


def _row_from_result(
    result: AgentResult,
    label: LabelResult,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    b5 = result.big_five
    sp = result.sticker
    pb = result.payback
    fm = result.four_ms

    row: dict[str, Any] = {
        "ticker": result.ticker,
        "trade_date": pd.Timestamp(result.as_of),
        "fiscal_year": b5.latest_fiscal_year,

        # Quant check booleans
        "check_roic": b5.roic.passes,
        "check_sales_growth": b5.sales_growth.passes,
        "check_eps_growth": b5.eps_growth.passes,
        "check_equity_growth": b5.equity_growth.passes,
        "check_ocf_growth": b5.ocf_growth.passes,
        "check_margin_of_safety": sp.margin_of_safety_passes,
        "check_payback_time": pb.passes,
        "check_current_ratio": b5.current_ratio.passes,

        # Quant raw values
        "value_roic": b5.roic.value,
        "value_sales_growth": b5.sales_growth.value,
        "value_eps_growth": b5.eps_growth.value,
        "value_equity_growth": b5.equity_growth.value,
        "value_ocf_growth": b5.ocf_growth.value,
        "value_eps_today": sp.eps_today_basis,
        "value_future_growth_rate": sp.future_growth_rate,
        "value_future_pe": sp.future_pe,
        "value_sticker_price": sp.sticker_price,
        "value_mos_price": sp.margin_of_safety_price,
        "value_current_price": sp.current_price,
        "value_payback_years": pb.payback_years,
        "value_current_ratio": b5.current_ratio.value,

        # LLM checks
        "check_meaning": fm.meaning.passes if fm is not None else None,
        "check_moat": fm.moat.passes if fm is not None else None,
        "check_management": fm.management.passes if fm is not None else None,
        "moat_type": (
            fm.moat.details.get("moat_type") if fm is not None else None
        ),
        "llm_cached": fm.cached if fm is not None else None,
        "llm_skipped": fm is None,
        "llm_accession": fm.accession if fm is not None else None,
        "llm_model": fm.model if fm is not None else None,
        "llm_meaning_rationale": fm.meaning.rationale if fm is not None else None,
        "llm_moat_rationale": fm.moat.rationale if fm is not None else None,
        "llm_management_rationale": fm.management.rationale if fm is not None else None,

        # Decision derivations
        "decision_quant_pass": result.quant_pass,
        "decision_llm_pass": result.llm_pass,
        "decision_full_buy": result.is_buy_full,
        "decision_quant_only_buy": result.is_buy_quant_only,

        # Label
        "label_horizon_years": label.label_horizon_years,
        "label_target_cagr": label.target_cagr,
        "forward_cagr": label.forward_cagr,
        "label_passes": label.label_passes,
        "delisted_before_horizon": label.delisted_before_horizon,
        "label_error": label.error,

        # Agent provenance
        "agent_error": error,
        "evaluated_at": pd.Timestamp.now(),
    }
    return row


def _error_row(ticker: str, trade_date: date, error: str) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "trade_date": pd.Timestamp(trade_date),
        "fiscal_year": None,
        "decision_full_buy": False,
        "decision_quant_only_buy": False,
        "agent_error": error,
        "evaluated_at": pd.Timestamp.now(),
    }


# --------------------------------------------------------------- Chunk runner


def _chunk_path(chunks_dir: Path, trade_date: date) -> Path:
    return chunks_dir / f"chunk_{trade_date.isoformat()}.parquet"


def build_dataset(
    *,
    start_year: int = 2012,
    end_year: int = 2021,
    months: tuple[int, ...] = (3, 6, 9, 12),
    day: int = 15,
    label_horizon_years: int = DEFAULT_LABEL_HORIZON_YEARS,
    target_cagr: float = DEFAULT_TARGET_CAGR,
    include_llm: bool = True,
    sample_size: int | None = None,
    skip_existing: bool = True,
    output_dir: Path | None = None,
) -> Path:
    """Build the historical dataset and return the path to the consolidated parquet.

    Args:
        start_year, end_year: inclusive trade-date year range.
        months, day: trade dates are (year, month, day) for each year × month.
        label_horizon_years: forward horizon for the CAGR label.
        target_cagr: label threshold (Phil Town's 15% target).
        include_llm: whether to run the LLM 4Ms (the expensive part).
        sample_size: if set, randomly sample N (ticker, date) pairs (for testing).
        skip_existing: skip trade-date chunks already on disk (resumable).
        output_dir: defaults to cfg.dataset_dir.
    """
    cfg = get_config()
    output_dir = output_dir or cfg.dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    edgar = EdgarClient()
    prices = PriceClient()
    universe = SP500Universe()
    anthropic = Anthropic(api_key=cfg.anthropic_api_key) if include_llm else None
    agent = RuleOneAgent(edgar, prices, anthropic)

    trade_dates = generate_trade_dates(start_year, end_year, months, day)

    log.info(
        "Building dataset: %d trade dates, include_llm=%s, sample_size=%s",
        len(trade_dates), include_llm, sample_size,
    )

    for trade_date in trade_dates:
        chunk_path = _chunk_path(chunks_dir, trade_date)
        if skip_existing and chunk_path.exists():
            log.info("Skipping %s (chunk exists)", trade_date)
            continue

        members = sorted(universe.get_members(trade_date))
        if sample_size:
            members = members[:sample_size]
        log.info("Processing %s — %d tickers", trade_date, len(members))

        rows: list[dict[str, Any]] = []
        for ticker in tqdm(members, desc=str(trade_date), leave=False):
            try:
                result = agent.evaluate(
                    ticker, trade_date, include_llm=include_llm
                )
                label = compute_label(
                    ticker, trade_date, prices,
                    label_horizon_years=label_horizon_years,
                    target_cagr=target_cagr,
                )
                rows.append(_row_from_result(result, label))
            except Exception as exc:  # noqa: BLE001
                log.warning("Agent failed for %s @ %s: %s", ticker, trade_date, exc)
                rows.append(_error_row(ticker, trade_date, str(exc)))

        df = pd.DataFrame(rows)
        df.to_parquet(chunk_path, index=False)
        log.info(
            "Saved chunk %s — %d rows, %d full-buys",
            chunk_path.name, len(df),
            int(df.get("decision_full_buy", pd.Series([])).sum()),
        )

    return consolidate(output_dir)


def consolidate(output_dir: Path | None = None) -> Path:
    """Merge all chunk parquets into a single dataset.parquet."""
    cfg = get_config()
    output_dir = output_dir or cfg.dataset_dir
    chunks_dir = output_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")
    frames = [pd.read_parquet(p) for p in chunk_files]
    df = pd.concat(frames, ignore_index=True)
    out = output_dir / "dataset.parquet"
    df.to_parquet(out, index=False)
    log.info("Consolidated dataset: %s rows, %s columns -> %s",
             len(df), len(df.columns), out)
    return out
