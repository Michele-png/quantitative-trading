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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
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


# Cap on concurrent ``agent.evaluate`` calls per chunk. Each evaluate may make
# one LLM 4Ms request (the cache absorbs repeats keyed by ``(ticker,
# fiscal_year)``), so this also caps simultaneous Anthropic requests at 8.
MAX_AGENT_WORKERS = 8


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
    qx = result.quant_extras
    mg = result.management

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

        # Phil Town extras (soft flags)
        "check_debt_payoff": qx.debt_payoff.passes if qx is not None else None,
        "check_dilution": qx.dilution.passes if qx is not None else None,
        "check_dividend_quality": qx.dividend_quality.passes if qx is not None else None,
        "value_debt_payoff_years": qx.debt_payoff.value if qx is not None else None,
        "value_dilution_cagr": qx.dilution.value if qx is not None else None,
        "value_dividend_payout_ratio": (
            qx.dividend_details.payout_ratio if qx is not None else None
        ),
        "value_dividend_yield": (
            qx.dividend_details.dividend_yield if qx is not None else None
        ),
        "dividend_payout_band": (
            qx.dividend_details.payout_band if qx is not None else None
        ),
        "dividend_debt_funded": (
            qx.dividend_details.debt_funded_dividend if qx is not None else None
        ),
        "dividend_yield_trap": (
            qx.dividend_details.yield_trap if qx is not None else None
        ),

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

        # Management sub-checks (multi-document pipeline)
        "check_mgmt_blame": mg.blame.passes if mg is not None else None,
        "check_mgmt_long_short": mg.long_short.passes if mg is not None else None,
        "check_mgmt_clarity": mg.clarity.passes if mg is not None else None,
        "check_mgmt_compensation": mg.compensation.passes if mg is not None else None,
        "check_mgmt_insider": mg.insider.passes if mg is not None else None,
        "value_mgmt_clarity_score": mg.clarity.score if mg is not None else None,
        "value_mgmt_long_short_ratio": (
            mg.long_short.details.get("ratio") if mg is not None else None
        ),
        "value_mgmt_insider_net_usd": (
            mg.insider.details.get("net_open_market_value_usd")
            if mg is not None else None
        ),
        "mgmt_bundle_hash": mg.bundle_hash if mg is not None else None,
        "mgmt_cached": mg.cached if mg is not None else None,

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

        # Single-threaded warm-up of the per-ticker SEC companyfacts cache.
        # The agent's parallel ``evaluate`` workers below all read the same
        # on-disk JSON; warming it sequentially first avoids two threads
        # racing to fetch + write the same file.
        _prewarm_company_facts(edgar, members)

        rows = _evaluate_chunk_parallel(
            agent=agent,
            prices=prices,
            members=members,
            trade_date=trade_date,
            include_llm=include_llm,
            label_horizon_years=label_horizon_years,
            target_cagr=target_cagr,
        )

        # Deterministic row order — sort by (ticker, trade_date). trade_date
        # is constant within a chunk but we still include it so the same
        # ordering convention applies when the consolidated parquet is
        # produced. ``stable`` keeps multiple rows for the same key in their
        # original collection order.
        df = pd.DataFrame(rows).sort_values(
            ["ticker", "trade_date"], kind="stable",
        ).reset_index(drop=True)
        df.to_parquet(chunk_path, index=False)
        log.info(
            "Saved chunk %s — %d rows, %d full-buys",
            chunk_path.name, len(df),
            int(df.get("decision_full_buy", pd.Series([])).sum()),
        )

    return consolidate(output_dir)


def _prewarm_company_facts(edgar: EdgarClient, tickers: list[str]) -> None:
    """Sequentially prefetch SEC companyfacts for every ticker in the chunk.

    The disk cache is then warm for all parallel workers, so the
    ``ThreadPoolExecutor`` never has two threads racing to write the same
    JSON file. Failures are logged at ``debug`` because the per-evaluate
    fallback inside ``agent.evaluate`` will report them at the right log
    level if they actually matter for that ticker.
    """
    for ticker in tickers:
        try:
            cik = edgar.get_cik(ticker)
            edgar.get_company_facts(cik)
        except Exception as exc:  # noqa: BLE001
            log.debug("companyfacts prewarm failed for %s: %s", ticker, exc)


def _evaluate_one(
    *,
    agent: RuleOneAgent,
    prices: PriceClient,
    ticker: str,
    trade_date: date,
    include_llm: bool,
    label_horizon_years: int,
    target_cagr: float,
) -> dict[str, Any]:
    """Run the agent + label for one (ticker, trade_date) pair."""
    try:
        result = agent.evaluate(
            ticker, trade_date, include_llm=include_llm,
        )
        label = compute_label(
            ticker, trade_date, prices,
            label_horizon_years=label_horizon_years,
            target_cagr=target_cagr,
        )
        return _row_from_result(result, label)
    except Exception as exc:  # noqa: BLE001
        log.warning("Agent failed for %s @ %s: %s", ticker, trade_date, exc)
        return _error_row(ticker, trade_date, str(exc))


def _evaluate_chunk_parallel(
    *,
    agent: RuleOneAgent,
    prices: PriceClient,
    members: list[str],
    trade_date: date,
    include_llm: bool,
    label_horizon_years: int,
    target_cagr: float,
) -> list[dict[str, Any]]:
    """Run ``agent.evaluate`` for every ticker in ``members`` with a bounded
    ``ThreadPoolExecutor``. ``agent.evaluate`` is sync, and the LLM 4Ms client
    is I/O-bound, so a thread pool is the natural fit — no asyncio rewrite
    needed. The pool is capped at ``MAX_AGENT_WORKERS`` (8) per the audit.
    """
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_AGENT_WORKERS) as ex:
        futures = {
            ex.submit(
                _evaluate_one,
                agent=agent,
                prices=prices,
                ticker=ticker,
                trade_date=trade_date,
                include_llm=include_llm,
                label_horizon_years=label_horizon_years,
                target_cagr=target_cagr,
            ): ticker
            for ticker in members
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=str(trade_date),
            leave=False,
        ):
            rows.append(fut.result())
    return rows


def consolidate(output_dir: Path | None = None) -> Path:
    """Merge all chunk parquets into a single dataset.parquet.

    Reads the chunk directory as a ``pyarrow.dataset`` so we stream the
    files in a single Arrow scan instead of materialising one DataFrame
    per chunk and ``pd.concat``-ing them at the end — the latter
    duplicates memory and was the dominant cost on full-sized
    consolidations. The final pandas-side reorder preserves the column
    order from the first chunk so downstream consumers see the same
    schema as before.
    """
    cfg = get_config()
    output_dir = output_dir or cfg.dataset_dir
    chunks_dir = output_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")

    # Preserve the original chunk-0 column order — pyarrow may union the
    # schemas in a different order if individual chunks were written with
    # slightly different column sequences (older chunks predate newer
    # columns). We re-project to chunk_0_cols to keep downstream consumers
    # (notebooks, the backtest) seeing the historical layout.
    first_table = pq.read_table(chunk_files[0])
    chunk_0_cols: list[str] = list(first_table.column_names)

    ds = pa_dataset.dataset(
        [str(p) for p in chunk_files], format="parquet",
    )
    table = ds.to_table()
    union_cols = list(table.column_names)
    # Re-project: keep chunk_0 ordering for the columns we know, append any
    # extras at the end. ``pa.Table.select`` is a zero-copy column slice.
    ordered_cols = chunk_0_cols + [c for c in union_cols if c not in chunk_0_cols]
    if ordered_cols != union_cols:
        table = table.select(ordered_cols)
    df = table.to_pandas()
    # Ensure even pandas-side column ordering matches the chunk-0 source.
    if list(df.columns) != ordered_cols:
        df = df[ordered_cols]
    out = output_dir / "dataset.parquet"
    df.to_parquet(out, index=False)
    log.info("Consolidated dataset: %s rows, %s columns -> %s",
             len(df), len(df.columns), out)
    return out
