"""Tests for the dataset builder (mocked agent + label)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantitative_trading.config import get_config
from quantitative_trading.dataset.builder import (
    _error_row,
    _row_from_result,
    consolidate,
    generate_trade_dates,
)


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def test_generate_trade_dates_quarterly() -> None:
    dates_ = generate_trade_dates(2018, 2019, months=(3, 6, 9, 12), day=15)
    assert len(dates_) == 8
    assert dates_[0] == date(2018, 3, 15)
    assert dates_[-1] == date(2019, 12, 15)


def test_generate_trade_dates_custom_months() -> None:
    dates_ = generate_trade_dates(2020, 2020, months=(1, 7), day=1)
    assert dates_ == [date(2020, 1, 1), date(2020, 7, 1)]


def test_error_row_has_required_decision_columns() -> None:
    row = _error_row("FAKE", date(2020, 1, 1), "boom")
    assert row["ticker"] == "FAKE"
    assert row["decision_full_buy"] is False
    assert row["decision_quant_only_buy"] is False
    assert row["agent_error"] == "boom"


def test_consolidate_merges_chunks(tmp_path: Path) -> None:
    output = tmp_path / "ds"
    chunks = output / "chunks"
    chunks.mkdir(parents=True)
    pd.DataFrame([{"ticker": "A", "decision_full_buy": True}]).to_parquet(
        chunks / "chunk_2020-01-01.parquet"
    )
    pd.DataFrame([{"ticker": "B", "decision_full_buy": False}]).to_parquet(
        chunks / "chunk_2020-04-01.parquet"
    )
    out = consolidate(output)
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["ticker"]) == {"A", "B"}


def test_consolidate_raises_when_no_chunks(tmp_path: Path) -> None:
    output = tmp_path / "ds"
    (output / "chunks").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate(output)
