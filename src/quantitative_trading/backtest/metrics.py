"""Classification + portfolio metrics for the backtest ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard binary classification metrics for one ablation variant."""

    variant: str
    n_eligible: int  # rows with valid label
    n_predicted_buy: int
    n_actual_pass: int
    n_true_positives: int
    n_false_positives: int
    n_true_negatives: int
    n_false_negatives: int
    base_rate: float  # P(label_passes) over eligible rows
    buy_rate: float  # P(predicted_buy)
    precision: float | None  # TP / (TP + FP) — given a buy, P(label_passes)
    recall: float | None  # TP / (TP + FN) — of all passers, fraction we caught
    f1: float | None
    accuracy: float
    lift: float | None  # precision / base_rate — value-add over random selection


def classification_metrics(
    df: pd.DataFrame,
    *,
    variant: str,
    decision_col: str,
    label_col: str = "label_passes",
) -> ClassificationMetrics:
    """Compute confusion-matrix-based metrics for one variant.

    Rows where the label is missing are excluded from the eligible set.
    """
    eligible = df[df[label_col].notna()].copy()
    eligible[label_col] = eligible[label_col].astype(bool)
    eligible[decision_col] = eligible[decision_col].fillna(False).astype(bool)

    n = len(eligible)
    if n == 0:
        return ClassificationMetrics(
            variant=variant, n_eligible=0, n_predicted_buy=0, n_actual_pass=0,
            n_true_positives=0, n_false_positives=0, n_true_negatives=0,
            n_false_negatives=0,
            base_rate=0.0, buy_rate=0.0, precision=None, recall=None, f1=None,
            accuracy=0.0, lift=None,
        )

    tp = int(((eligible[decision_col]) & (eligible[label_col])).sum())
    fp = int(((eligible[decision_col]) & (~eligible[label_col])).sum())
    tn = int(((~eligible[decision_col]) & (~eligible[label_col])).sum())
    fn = int(((~eligible[decision_col]) & (eligible[label_col])).sum())

    n_buy = tp + fp
    n_pass = tp + fn
    base_rate = n_pass / n
    buy_rate = n_buy / n

    precision = tp / n_buy if n_buy > 0 else None
    recall = tp / n_pass if n_pass > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    accuracy = (tp + tn) / n
    lift = precision / base_rate if (precision is not None and base_rate > 0) else None

    return ClassificationMetrics(
        variant=variant, n_eligible=n, n_predicted_buy=n_buy, n_actual_pass=n_pass,
        n_true_positives=tp, n_false_positives=fp, n_true_negatives=tn,
        n_false_negatives=fn,
        base_rate=base_rate, buy_rate=buy_rate, precision=precision,
        recall=recall, f1=f1, accuracy=accuracy, lift=lift,
    )


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioMetrics:
    """Simple equal-weight buy-and-hold portfolio metrics."""

    variant: str
    n_trades: int
    avg_forward_cagr: float | None
    median_forward_cagr: float | None
    win_rate: float | None  # fraction of trades with forward_cagr > 0
    pct_above_target: float | None  # fraction with forward_cagr >= target
    pct_above_market: float | None  # fraction beating SPY's same-period CAGR
    avg_outperformance_vs_spy: float | None  # avg (trade_cagr - spy_cagr)


def portfolio_metrics(
    df: pd.DataFrame,
    *,
    variant: str,
    decision_col: str,
    spy_cagr_col: str | None = None,
    target_cagr: float = 0.15,
) -> PortfolioMetrics:
    """Equal-weight buy-and-hold portfolio metrics for trades flagged by `decision_col`.

    Each `decision_col=True` row is a "trade"; the forward_cagr column is its
    return. Compare to spy_cagr_col (if supplied) for an outperformance read.
    """
    eligible = df[df["forward_cagr"].notna() & df[decision_col].fillna(False)]
    n = len(eligible)
    if n == 0:
        return PortfolioMetrics(
            variant=variant, n_trades=0, avg_forward_cagr=None,
            median_forward_cagr=None, win_rate=None, pct_above_target=None,
            pct_above_market=None, avg_outperformance_vs_spy=None,
        )

    cagrs = eligible["forward_cagr"].astype(float)
    avg = float(cagrs.mean())
    med = float(cagrs.median())
    win_rate = float((cagrs > 0).mean())
    pct_above_target = float((cagrs >= target_cagr).mean())

    pct_above_market: float | None = None
    avg_out: float | None = None
    if spy_cagr_col and spy_cagr_col in eligible.columns:
        spy = eligible[spy_cagr_col].astype(float)
        diff = cagrs - spy
        pct_above_market = float((diff > 0).mean())
        avg_out = float(diff.mean())

    return PortfolioMetrics(
        variant=variant, n_trades=n, avg_forward_cagr=avg, median_forward_cagr=med,
        win_rate=win_rate, pct_above_target=pct_above_target,
        pct_above_market=pct_above_market, avg_outperformance_vs_spy=avg_out,
    )
