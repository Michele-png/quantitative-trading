"""Tests for the backtest engine helpers.

Focus is on the vectorised ``add_spy_forward_cagr`` — we assert numerical
equality between the new searchsorted-based implementation and the original
``PriceClient.forward_total_return_cagr`` per-row formula for a controlled
synthetic SPY series.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantitative_trading.backtest.engine import add_spy_forward_cagr
from quantitative_trading.config import get_config
from quantitative_trading.data.prices import PriceClient


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _synthetic_spy_history(
    start: date = date(2010, 1, 4),
    n_days: int = 365 * 12,
    annual_growth: float = 0.10,
) -> pd.DataFrame:
    """Build a deterministic SPY-like daily price series.

    Skips weekends so the timestamps look like real trading days; the price
    follows ``p(t) = p0 * (1 + annual_growth) ** (days/365.25)`` so the
    forward CAGR has a clean closed form.
    """
    dates: list[pd.Timestamp] = []
    cur = pd.Timestamp(start)
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(dates).normalize()
    days_from_start = (idx - idx[0]).days.to_numpy(dtype=float)
    prices = 100.0 * (1.0 + annual_growth) ** (days_from_start / 365.25)
    return pd.DataFrame(
        {
            "Open": prices, "High": prices, "Low": prices,
            "Close": prices, "Adj Close": prices, "Volume": np.zeros_like(prices),
        },
        index=idx,
    )


def _seed_price_cache(tmp_path: Path, ticker: str, hist: pd.DataFrame) -> None:
    """Seed PriceClient's disk cache so neither the per-row nor vectorised
    implementation actually reaches the network during the test."""
    cache_dir = get_config().prices_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(cache_dir / f"{ticker.upper()}.parquet")
    # Empty splits cache so PriceClient.get_splits returns an empty Series.
    pd.DataFrame({"ratio": []}).to_parquet(
        cache_dir / f"{ticker.upper()}.splits.parquet"
    )


def _scalar_spy_cagr(
    pc: PriceClient,
    trade_date: date,
    *,
    spy_ticker: str = "SPY",
    label_horizon_years: int = 5,
) -> float | None:
    """Replicates the original per-row implementation for cross-checking."""
    horizon_days = int(365.25 * label_horizon_years)
    return pc.forward_total_return_cagr(
        spy_ticker, trade_date, trade_date + timedelta(days=horizon_days),
    )


def test_add_spy_forward_cagr_matches_per_row_scalar(tmp_path: Path) -> None:
    """Vectorised CAGR must equal the scalar PriceClient formula row-by-row."""
    hist = _synthetic_spy_history()
    _seed_price_cache(tmp_path, "SPY", hist)
    pc = PriceClient()
    pc.get_history("SPY")  # warm cache

    trade_dates = [
        date(2012, 3, 15), date(2013, 6, 15), date(2014, 9, 15),
        date(2015, 12, 15), date(2017, 1, 15),
    ]
    df = pd.DataFrame({
        "ticker": ["A"] * len(trade_dates),
        "trade_date": [pd.Timestamp(d) for d in trade_dates],
    })

    out = add_spy_forward_cagr(df, label_horizon_years=5)
    assert "spy_forward_cagr" in out.columns

    for td, vec in zip(trade_dates, out["spy_forward_cagr"], strict=True):
        scalar = _scalar_spy_cagr(pc, td, label_horizon_years=5)
        # Both implementations should agree (within float tolerance).
        if scalar is None:
            assert pd.isna(vec)
        else:
            assert vec == pytest.approx(scalar, abs=1e-12)


def test_add_spy_forward_cagr_recovers_known_growth_within_snap(
    tmp_path: Path,
) -> None:
    """A 10%/yr synthetic series should produce ≈ 0.10 CAGR.

    The tolerance accommodates the trading-day snap-back at both endpoints:
    if the trade date is on a weekend the start price is taken from the
    previous Friday, so the realised price ratio is over slightly more or
    fewer days than the constant ``horizon_days``. This drift is bounded
    by ~2 days / 5 years ≈ 0.1% in CAGR — which mirrors the scalar
    implementation exactly (verified in
    ``test_add_spy_forward_cagr_matches_per_row_scalar``).
    """
    hist = _synthetic_spy_history(annual_growth=0.10)
    _seed_price_cache(tmp_path, "SPY", hist)

    df = pd.DataFrame({
        "ticker": ["X", "Y", "Z"],
        "trade_date": [
            pd.Timestamp(2012, 3, 15),
            pd.Timestamp(2014, 6, 17),
            pd.Timestamp(2016, 9, 15),
        ],
    })
    out = add_spy_forward_cagr(df, label_horizon_years=5)
    for v in out["spy_forward_cagr"]:
        assert float(v) == pytest.approx(0.10, abs=2e-3)


def test_add_spy_forward_cagr_handles_empty_history(tmp_path: Path) -> None:
    """No SPY data → every output value is null, no exception."""
    empty = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    )
    _seed_price_cache(tmp_path, "SPY", empty)
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": [pd.Timestamp(2012, 3, 15), pd.Timestamp(2013, 6, 15)],
    })
    out = add_spy_forward_cagr(df, label_horizon_years=5)
    assert out["spy_forward_cagr"].isna().all()


def test_add_spy_forward_cagr_marks_pre_history_dates_null(tmp_path: Path) -> None:
    """Trade dates before SPY's first price must yield null, not a bogus CAGR."""
    hist = _synthetic_spy_history(start=date(2015, 1, 2))
    _seed_price_cache(tmp_path, "SPY", hist)
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": [pd.Timestamp(2010, 6, 15), pd.Timestamp(2016, 6, 15)],
    })
    out = add_spy_forward_cagr(df, label_horizon_years=5)
    assert pd.isna(out["spy_forward_cagr"].iloc[0])
    assert not pd.isna(out["spy_forward_cagr"].iloc[1])
