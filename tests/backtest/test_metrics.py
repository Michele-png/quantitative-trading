"""Tests for the backtest classification + portfolio metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantitative_trading.backtest.metrics import (
    classification_metrics,
    portfolio_metrics,
)
from quantitative_trading.config import get_config


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_classification_metrics_perfect_classifier() -> None:
    """Decision == label everywhere → precision=recall=1, lift=1/base_rate."""
    df = _df([
        {"decision_x": True, "label_passes": True, "forward_cagr": 0.20},
        {"decision_x": True, "label_passes": True, "forward_cagr": 0.18},
        {"decision_x": False, "label_passes": False, "forward_cagr": 0.05},
        {"decision_x": False, "label_passes": False, "forward_cagr": -0.10},
    ])
    m = classification_metrics(df, variant="x", decision_col="decision_x")
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.base_rate == 0.5
    assert m.lift == 2.0  # precision / base_rate


def test_classification_metrics_random_classifier() -> None:
    """Random buys at the base rate → precision ≈ base_rate, lift ≈ 1."""
    df = _df([
        {"decision_x": True, "label_passes": True},
        {"decision_x": True, "label_passes": False},
        {"decision_x": False, "label_passes": True},
        {"decision_x": False, "label_passes": False},
    ])
    m = classification_metrics(df, variant="x", decision_col="decision_x")
    assert m.precision == 0.5
    assert m.base_rate == 0.5
    assert m.lift == 1.0


def test_classification_excludes_missing_label_rows() -> None:
    df = _df([
        {"decision_x": True, "label_passes": True},
        {"decision_x": True, "label_passes": None},  # excluded
        {"decision_x": False, "label_passes": False},
    ])
    m = classification_metrics(df, variant="x", decision_col="decision_x")
    assert m.n_eligible == 2


def test_classification_with_zero_buys_recall_zero() -> None:
    df = _df([
        {"decision_x": False, "label_passes": True},
        {"decision_x": False, "label_passes": True},
        {"decision_x": False, "label_passes": False},
    ])
    m = classification_metrics(df, variant="x", decision_col="decision_x")
    assert m.n_predicted_buy == 0
    assert m.precision is None  # division by zero
    assert m.recall == 0.0
    assert m.lift is None


def test_portfolio_metrics_basic() -> None:
    df = _df([
        {"decision_x": True, "forward_cagr": 0.20, "spy_forward_cagr": 0.10},
        {"decision_x": True, "forward_cagr": 0.18, "spy_forward_cagr": 0.10},
        {"decision_x": True, "forward_cagr": -0.05, "spy_forward_cagr": 0.10},
        {"decision_x": False, "forward_cagr": 0.30, "spy_forward_cagr": 0.10},  # ignored
    ])
    p = portfolio_metrics(
        df, variant="x", decision_col="decision_x",
        spy_cagr_col="spy_forward_cagr", target_cagr=0.15,
    )
    assert p.n_trades == 3
    assert p.avg_forward_cagr == pytest.approx((0.20 + 0.18 - 0.05) / 3, abs=1e-6)
    assert p.win_rate == pytest.approx(2 / 3, abs=1e-6)
    assert p.pct_above_target == pytest.approx(2 / 3, abs=1e-6)
    assert p.pct_above_market == pytest.approx(2 / 3, abs=1e-6)


def test_portfolio_metrics_no_trades() -> None:
    df = _df([
        {"decision_x": False, "forward_cagr": 0.20, "spy_forward_cagr": 0.10},
    ])
    p = portfolio_metrics(df, variant="x", decision_col="decision_x")
    assert p.n_trades == 0
    assert p.avg_forward_cagr is None
