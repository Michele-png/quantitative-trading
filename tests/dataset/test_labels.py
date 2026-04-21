"""Unit tests for forward-CAGR label computation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.config import get_config
from quantitative_trading.dataset.labels import compute_label


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def test_label_passes_when_cagr_above_target() -> None:
    pc = MagicMock()
    pc.forward_total_return_cagr.return_value = 0.20
    pc.was_delisted_before.return_value = False
    label = compute_label("FAKE", date(2015, 1, 1), pc, target_cagr=0.15)
    assert label.label_passes
    assert label.forward_cagr == 0.20
    assert not label.delisted_before_horizon
    assert label.error is None


def test_label_fails_when_cagr_below_target() -> None:
    pc = MagicMock()
    pc.forward_total_return_cagr.return_value = 0.10
    pc.was_delisted_before.return_value = False
    label = compute_label("FAKE", date(2015, 1, 1), pc, target_cagr=0.15)
    assert not label.label_passes
    assert label.forward_cagr == 0.10


def test_label_handles_no_data() -> None:
    pc = MagicMock()
    pc.forward_total_return_cagr.return_value = None
    label = compute_label("DELISTED", date(2015, 1, 1), pc)
    assert label.forward_cagr is None
    assert not label.label_passes
    assert label.error is not None


def test_label_records_delisting_flag() -> None:
    pc = MagicMock()
    pc.forward_total_return_cagr.return_value = -0.50
    pc.was_delisted_before.return_value = True
    label = compute_label("BANKRUPT", date(2015, 1, 1), pc)
    assert not label.label_passes
    assert label.delisted_before_horizon


def test_label_end_date_is_horizon_years_after_trade_date() -> None:
    pc = MagicMock()
    pc.forward_total_return_cagr.return_value = 0.15
    pc.was_delisted_before.return_value = False
    label = compute_label("FAKE", date(2015, 1, 1), pc, label_horizon_years=5)
    # 5 × 365.25 = 1826 days -> end ~2020-01-01
    assert label.label_end_date.year == 2020
    assert label.label_end_date.month == 1
