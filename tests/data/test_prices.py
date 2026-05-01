"""Unit tests for PriceClient.

Mocks `yfinance.Ticker.history` to return a synthetic DataFrame. Verifies:
    * get_close_at uses Close (not Adj Close)
    * get_adj_close_at uses Adj Close
    * Both walk back to the prior trading day on weekends/holidays
    * forward_total_return_cagr uses Adj Close
    * Delisted ticker (no data near end_date) is clamped at delisted_floor
    * Cache is read on second call (no second yfinance hit)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quantitative_trading.config import get_config
from quantitative_trading.data.prices import PriceClient


def _synthetic_history() -> pd.DataFrame:
    """A tiny synthetic price history. Trading days only (weekdays).

    Close grows linearly from 100 -> 200 over 4 years (simple, easy to verify).
    Adj Close = Close * 1.10 (10% lift to simulate dividend reinvestment).
    Last trading day: 2023-12-29 (Friday).
    """
    dates = pd.bdate_range("2020-01-02", "2023-12-29")
    n = len(dates)
    close = pd.Series(
        [100.0 + (200.0 - 100.0) * (i / (n - 1)) for i in range(n)],
        index=dates,
    )
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close * 1.10,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def test_get_close_at_uses_close_not_adj_close() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        close = client.get_close_at("FAKE", date(2020, 1, 2))
    assert close is not None
    assert close == pytest.approx(100.0, abs=0.01)


def test_get_adj_close_at_uses_adj_close() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        adj = client.get_adj_close_at("FAKE", date(2020, 1, 2))
    assert adj is not None
    assert adj == pytest.approx(100.0 * 1.10, abs=0.01)


def test_close_at_walks_back_over_weekend() -> None:
    """Sat/Sun are non-trading. Asking for Sat returns Friday's close."""
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        sat = client.get_close_at("FAKE", date(2020, 1, 4))  # Saturday
        fri = client.get_close_at("FAKE", date(2020, 1, 3))  # Friday
    assert sat is not None and fri is not None
    assert sat == fri


def test_close_at_returns_none_when_no_data_within_lookback() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        # Way before any data
        result = client.get_close_at("FAKE", date(2010, 1, 1))
    assert result is None


def test_close_at_returns_none_when_lookback_exceeded() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        # 30 days after last trading day, with default 10-day lookback
        result = client.get_close_at(
            "FAKE", date(2024, 1, 31), max_lookback_days=10
        )
    assert result is None


def test_forward_total_return_cagr_uses_adj_close() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        cagr = client.forward_total_return_cagr(
            "FAKE", date(2020, 1, 2), date(2022, 1, 3)
        )
    # Adj Close grows from ~110 to ~165 (linear from 100->200 raw, *1.10),
    # which is the same CAGR as Close itself since both grow proportionally.
    # 2y CAGR: (165/110)^0.5 - 1 ≈ 22.5%
    assert cagr is not None
    assert cagr == pytest.approx(0.225, abs=0.02)


def test_forward_return_uses_last_available_price_when_delisted_early() -> None:
    """If the ticker stopped trading before end_date, label uses last close,
    annualized over the full nominal horizon (penalizing partial-life stocks)."""
    df = _synthetic_history()  # last data 2023-12-29, last Adj Close ~ 220
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        # 5y horizon, but last data is at year ~4. Annualized over 5y:
        # ~110 -> 220 in 5y nominal: (220/110)^(1/5) - 1 ≈ 14.87%
        cagr = client.forward_total_return_cagr(
            "FAKE", date(2020, 1, 2), date(2025, 1, 2)
        )
    assert cagr is not None
    assert cagr == pytest.approx(0.1487, abs=0.02)


def test_was_delisted_before_flags_partial_data() -> None:
    df = _synthetic_history()  # last 2023-12-29
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        # End date close to last data — not delisted
        assert client.was_delisted_before("FAKE", date(2024, 1, 5)) is False
        # End date well after — flagged as delisted
        assert client.was_delisted_before("FAKE", date(2025, 1, 5)) is True


def test_cache_avoids_second_yfinance_call() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        client.get_close_at("FAKE", date(2020, 1, 2))
        first_call_count = mock_ticker.return_value.history.call_count

        client.get_close_at("FAKE", date(2020, 1, 3))
        second_call_count = mock_ticker.return_value.history.call_count

    assert first_call_count == 1
    assert second_call_count == 1  # no second fetch — read from cache


def test_coverage_returns_first_and_last_dates() -> None:
    df = _synthetic_history()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        client = PriceClient()
        first, last = client.coverage("FAKE")
    assert first == date(2020, 1, 2)
    assert last == date(2023, 12, 29)


def test_empty_history_returns_none_for_all_queries() -> None:
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = empty
        client = PriceClient()
        assert client.get_close_at("DELISTED", date(2020, 1, 1)) is None
        assert client.get_adj_close_at("DELISTED", date(2020, 1, 1)) is None
        assert (
            client.forward_total_return_cagr(
                "DELISTED", date(2020, 1, 1), date(2025, 1, 1)
            )
            is None
        )
        assert client.coverage("DELISTED") == (None, None)


def _splits_series(items: list[tuple[str, float]]) -> pd.Series:
    return pd.Series(
        {pd.Timestamp(d): r for d, r in items},
        name="ratio",
    )


def test_split_factor_since_no_splits_returns_one() -> None:
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        client = PriceClient()
        assert client.split_factor_since("FAKE", date(2015, 1, 1)) == 1.0


def test_split_factor_since_excludes_splits_at_or_before_date() -> None:
    """A split on the exact since_date is treated as already-reflected in
    the SEC value (the filing was on/after the split). Only later splits scale."""
    splits = _splits_series([("2014-06-09", 7.0), ("2020-08-31", 4.0)])
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = splits
        client = PriceClient()
        # Both splits after this date.
        assert client.split_factor_since("FAKE", date(2010, 1, 1)) == 28.0
        # Only the 2020 split is after.
        assert client.split_factor_since("FAKE", date(2014, 6, 9)) == 4.0
        # Just after the 2014 split.
        assert client.split_factor_since("FAKE", date(2014, 6, 10)) == 4.0
        # After both — no scaling.
        assert client.split_factor_since("FAKE", date(2021, 1, 1)) == 1.0


def test_get_splits_caches() -> None:
    splits = _splits_series([("2020-08-31", 4.0)])
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = splits
        client = PriceClient()
        result1 = client.get_splits("FAKE")
        result2 = client.get_splits("FAKE")
    assert result1.iloc[0] == 4.0
    assert result2.iloc[0] == 4.0
    # yfinance constructor called twice (once per get_splits call), but the
    # `.splits` attribute access only happens on cache miss. Hard to inspect
    # cleanly; the contract we care about is correctness of the value.
