"""Unit tests for SP500Universe using a synthetic CSV fixture."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quantitative_trading.config import get_config
from quantitative_trading.data.universe import SP500Universe


_FIXTURE_CSV = """ticker,start_date,end_date
AAPL,1996-01-02,
AAL,1996-01-02,1997-01-15
AAL,2015-03-23,2024-09-23
ABMD,2018-05-31,2022-12-22
NEWCO,2020-01-01,
"""


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


@pytest.fixture
def universe(tmp_path: Path) -> SP500Universe:
    cache_dir = tmp_path / "universe"
    cache_dir.mkdir()
    (cache_dir / "sp500_ticker_start_end.csv").write_text(_FIXTURE_CSV)
    return SP500Universe(cache_dir=cache_dir)


def test_get_members_at_date_includes_active(universe: SP500Universe) -> None:
    members = universe.get_members(date(2020, 6, 1))
    assert "AAPL" in members
    assert "AAL" in members
    assert "ABMD" in members
    assert "NEWCO" in members


def test_get_members_excludes_pre_start(universe: SP500Universe) -> None:
    members = universe.get_members(date(2018, 5, 30))
    assert "ABMD" not in members  # joins next day
    assert "NEWCO" not in members  # joins 2020-01-01


def test_get_members_excludes_post_end(universe: SP500Universe) -> None:
    members = universe.get_members(date(2023, 1, 1))
    assert "ABMD" not in members  # left 2022-12-22
    assert "AAPL" in members
    assert "AAL" in members


def test_get_members_inclusive_at_boundaries(universe: SP500Universe) -> None:
    """A ticker should be a member on both its start_date and its end_date."""
    assert "ABMD" in universe.get_members(date(2018, 5, 31))
    assert "ABMD" in universe.get_members(date(2022, 12, 22))
    assert "ABMD" not in universe.get_members(date(2022, 12, 23))


def test_aal_two_separate_membership_periods(universe: SP500Universe) -> None:
    """AAL was member 1996-1997, then re-added 2015-2024. Gap should be excluded."""
    assert "AAL" in universe.get_members(date(1996, 6, 1))
    assert "AAL" not in universe.get_members(date(2000, 1, 1))  # in the gap
    assert "AAL" in universe.get_members(date(2020, 1, 1))


def test_get_membership_periods_returns_all_intervals(universe: SP500Universe) -> None:
    periods = universe.get_membership_periods("AAL")
    assert len(periods) == 2
    assert periods[0] == (date(1996, 1, 2), date(1997, 1, 15))
    assert periods[1] == (date(2015, 3, 23), date(2024, 9, 23))


def test_get_membership_periods_handles_active_with_none_end(
    universe: SP500Universe,
) -> None:
    periods = universe.get_membership_periods("AAPL")
    assert len(periods) == 1
    assert periods[0] == (date(1996, 1, 2), None)


def test_is_member_matches_get_members(universe: SP500Universe) -> None:
    target = date(2020, 6, 1)
    members = universe.get_members(target)
    for t in ["AAPL", "AAL", "ABMD", "NEWCO"]:
        assert universe.is_member(t, target) == (t in members)


def test_member_count_is_size_of_set(universe: SP500Universe) -> None:
    assert universe.member_count(date(2020, 6, 1)) == 4


def test_all_tickers_ever_dedupes_multiple_periods(universe: SP500Universe) -> None:
    tickers = universe.all_tickers_ever()
    assert tickers == {"AAPL", "AAL", "ABMD", "NEWCO"}


def test_ticker_case_normalized_in_lookup(universe: SP500Universe) -> None:
    assert universe.is_member("aapl", date(2020, 6, 1))
    assert universe.is_member("Aapl", date(2020, 6, 1))
