"""Backtest engine: run the ablation across the dataset and produce metrics + reports.

Three variants are evaluated on the same (ticker, trade_date) grid:
    A. Full Rule One (all 9 checks pass)            — column `decision_full_buy`
    B. Quant only   (no LLM 4Ms)                    — column `decision_quant_only_buy`
    C. Quant + Random Qualitative (negative control) — synthesized at backtest
       time using the variant-A LLM base rates.

For each variant we compute classification metrics (precision/recall/F1) vs.
the forward-CAGR label and a portfolio simulation (buy-and-hold equal weight).
SPY is added as a market baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from anthropic import Anthropic  # noqa: F401  (kept for symmetry with builder)

from quantitative_trading.agents.rule_one.agent import AgentResult
from quantitative_trading.backtest.metrics import (
    ClassificationMetrics,
    PortfolioMetrics,
    classification_metrics,
    portfolio_metrics,
)
from quantitative_trading.config import get_config
from quantitative_trading.data.prices import PriceClient


log = logging.getLogger(__name__)


# --------------------------------------------------------------- Random qual


def _hash_to_unit_interval(seed: str, key: str) -> float:
    """Deterministic float in [0, 1) for a given (seed, key)."""
    import hashlib
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def add_random_qual_decision(
    df: pd.DataFrame,
    *,
    seed: str = "rule-one-ablation-v1",
    decision_col: str = "decision_quant_random_qual_buy",
) -> pd.DataFrame:
    """Add a `decision_quant_random_qual_buy` column to df.

    The synthesized "qualitative pass" for each row is sampled deterministically
    from the variant-A base rates (overall fraction of times each M passed in
    the LLM-evaluated rows). Rows where quant fails are False; otherwise the
    three Ms are drawn independently and ANDed together.
    """
    out = df.copy()

    # Compute LLM base rates only over rows where the LLM was actually run.
    llm_rows = out[out["check_meaning"].notna()]
    if len(llm_rows) == 0:
        out[decision_col] = False
        return out

    base_meaning = float(llm_rows["check_meaning"].astype(bool).mean())
    base_moat = float(llm_rows["check_moat"].astype(bool).mean())
    base_management = float(llm_rows["check_management"].astype(bool).mean())

    log.info(
        "Random-qual base rates: meaning=%.3f moat=%.3f management=%.3f",
        base_meaning, base_moat, base_management,
    )

    decisions = []
    for _, row in out.iterrows():
        if not bool(row.get("decision_quant_pass", False)):
            decisions.append(False)
            continue
        key_base = f"{row['ticker']}|{row['trade_date'].isoformat() if hasattr(row['trade_date'], 'isoformat') else row['trade_date']}"
        passes_all = True
        for m, rate in (
            ("meaning", base_meaning),
            ("moat", base_moat),
            ("management", base_management),
        ):
            draw = _hash_to_unit_interval(seed, f"{key_base}|{m}")
            if draw >= rate:
                passes_all = False
                break
        decisions.append(passes_all)
    out[decision_col] = decisions
    return out


# ---------------------------------------------------------------- SPY column


def add_spy_forward_cagr(
    df: pd.DataFrame,
    *,
    spy_ticker: str = "SPY",
    label_horizon_years: int = 5,
) -> pd.DataFrame:
    """Add a `spy_forward_cagr` column (per-row SPY CAGR over the same horizon).

    Cached SPY history is fetched once via PriceClient.
    """
    out = df.copy()
    pc = PriceClient()
    pc.get_history(spy_ticker)  # warm cache

    cagrs: list[float | None] = []
    horizon_days = int(365.25 * label_horizon_years)
    for trade_date in out["trade_date"]:
        td = pd.Timestamp(trade_date).date()
        end = td + timedelta(days=horizon_days)
        cagr = pc.forward_total_return_cagr(spy_ticker, td, end)
        cagrs.append(cagr)
    out["spy_forward_cagr"] = cagrs
    return out


# --------------------------------------------------------------- Backtest run


VARIANTS: list[tuple[str, str]] = [
    ("Full Rule One (with LLM)", "decision_full_buy"),
    ("Quant only", "decision_quant_only_buy"),
    ("Quant + random qualitative", "decision_quant_random_qual_buy"),
]


@dataclass(frozen=True)
class BacktestReport:
    classification: pd.DataFrame  # one row per variant
    portfolio: pd.DataFrame  # one row per variant
    base_rate: float  # fraction of eligible rows whose label_passes
    n_eligible: int


def run_backtest(
    dataset_path: Path | None = None,
    *,
    add_spy: bool = True,
    label_horizon_years: int = 5,
    target_cagr: float = 0.15,
    output_dir: Path | None = None,
) -> BacktestReport:
    cfg = get_config()
    dataset_path = dataset_path or (cfg.dataset_dir / "dataset.parquet")
    output_dir = output_dir or cfg.dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    log.info("Loaded dataset: %d rows", len(df))

    df = add_random_qual_decision(df)
    if add_spy:
        df = add_spy_forward_cagr(df, label_horizon_years=label_horizon_years)
    enriched_path = output_dir / "dataset_enriched.parquet"
    df.to_parquet(enriched_path, index=False)
    log.info("Wrote enriched dataset: %s", enriched_path)

    # ---------- Classification metrics
    cls_rows: list[dict] = []
    for variant_name, decision_col in VARIANTS:
        m = classification_metrics(df, variant=variant_name, decision_col=decision_col)
        cls_rows.append(
            {
                "variant": m.variant,
                "n_eligible": m.n_eligible,
                "n_predicted_buy": m.n_predicted_buy,
                "buy_rate": m.buy_rate,
                "base_rate": m.base_rate,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "accuracy": m.accuracy,
                "lift": m.lift,
            }
        )
    cls_df = pd.DataFrame(cls_rows)
    base_rate = float(cls_df["base_rate"].iloc[0]) if len(cls_df) else 0.0
    n_eligible = int(cls_df["n_eligible"].iloc[0]) if len(cls_df) else 0

    # ---------- Portfolio metrics
    port_rows: list[dict] = []
    spy_col = "spy_forward_cagr" if add_spy else None
    for variant_name, decision_col in VARIANTS:
        p = portfolio_metrics(
            df, variant=variant_name, decision_col=decision_col,
            spy_cagr_col=spy_col, target_cagr=target_cagr,
        )
        port_rows.append(
            {
                "variant": p.variant,
                "n_trades": p.n_trades,
                "avg_forward_cagr": p.avg_forward_cagr,
                "median_forward_cagr": p.median_forward_cagr,
                "win_rate": p.win_rate,
                "pct_above_target": p.pct_above_target,
                "pct_above_market": p.pct_above_market,
                "avg_outperformance_vs_spy": p.avg_outperformance_vs_spy,
            }
        )
    port_df = pd.DataFrame(port_rows)

    # Save
    cls_df.to_csv(output_dir / "classification_metrics.csv", index=False)
    port_df.to_csv(output_dir / "portfolio_metrics.csv", index=False)
    log.info("Wrote metrics CSVs to %s", output_dir)

    return BacktestReport(
        classification=cls_df,
        portfolio=port_df,
        base_rate=base_rate,
        n_eligible=n_eligible,
    )
