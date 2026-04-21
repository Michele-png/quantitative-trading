"""Unit tests for the Sticker Price + Margin of Safety + Payback Time math."""

from __future__ import annotations

from datetime import date

import pytest

from quantitative_trading.agents.rule_one.sticker_price import (
    DEFAULT_FGR_CAP,
    DEFAULT_FUTURE_PE_CAP,
    DEFAULT_REQUIRED_RETURN,
    compute_payback_years,
    compute_sticker_price,
)


# --------------------------------------------------------------------------
# compute_sticker_price — pure-math tests
# --------------------------------------------------------------------------


def test_sticker_price_with_fgr_equal_to_required_return_simplifies() -> None:
    """When FGR == required_return, (1+g)^10 / (1+r)^10 = 1 → Sticker = EPS × PE."""
    fgr, pe, future_eps, sticker = compute_sticker_price(
        eps_today_basis=2.0,
        historical_growth_rate=0.15,
    )
    assert fgr == 0.15
    assert pe == 30.0
    # Future EPS = 2 × 1.15^10 ≈ 8.09
    assert future_eps == pytest.approx(2.0 * 1.15 ** 10, abs=0.01)
    # Sticker = future_eps × pe / 1.15^10 = 2.0 × 30 = 60.0
    assert sticker == pytest.approx(60.0, abs=0.01)


def test_sticker_price_caps_fgr() -> None:
    """Historical 30% growth should be capped at 15% (Phil Town's default)."""
    fgr, _pe, _, _ = compute_sticker_price(
        eps_today_basis=2.0,
        historical_growth_rate=0.30,  # above cap
    )
    assert fgr == DEFAULT_FGR_CAP


def test_sticker_price_future_pe_caps() -> None:
    """Future PE = 2 × FGR%, capped at future_pe_cap."""
    # FGR = 15% → 2 × 15 = 30 (at the cap)
    _, pe, _, _ = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.15,
    )
    assert pe == 30.0

    # If we override the cap downward, it binds.
    _, pe2, _, _ = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.15,
        future_pe_cap=20.0,
    )
    assert pe2 == 20.0


def test_sticker_price_uses_historical_avg_pe_when_lower() -> None:
    _, pe, _, _ = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.15,
        historical_avg_pe=15.0,  # lower than 30 (PE from growth)
    )
    assert pe == 15.0


def test_sticker_price_lower_growth_lower_sticker() -> None:
    """Lower growth → lower future EPS → lower Sticker."""
    _, _, _, sticker_high = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.15,
    )
    _, _, _, sticker_low = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.05,
    )
    assert sticker_low < sticker_high


def test_sticker_price_zero_growth_produces_low_sticker() -> None:
    fgr, pe, future_eps, sticker = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.0,
    )
    assert fgr == 0.0
    assert pe == 0.0  # 2 × 0% = 0; capped above by future_pe_cap is no-op here
    assert future_eps == pytest.approx(2.0)
    assert sticker == pytest.approx(0.0)


def test_sticker_price_negative_historical_growth_treated_as_zero() -> None:
    fgr, _, _, _ = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=-0.05,
    )
    assert fgr == 0.0  # max(0, ...) clamps


def test_sticker_price_required_return_changes_result() -> None:
    """Higher required return → harder to justify → lower Sticker."""
    _, _, _, s_15 = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.10,
        required_return=0.15,
    )
    _, _, _, s_25 = compute_sticker_price(
        eps_today_basis=2.0, historical_growth_rate=0.10,
        required_return=0.25,
    )
    assert s_25 < s_15


# --------------------------------------------------------------------------
# compute_payback_years
# --------------------------------------------------------------------------


def test_payback_years_grows_to_meet_price() -> None:
    """EPS=$2 growing 10%, price=$25. Cumulative EPS over years 1..N should
    reach $25 around year 8 (rough hand-calc: 2.2+2.42+2.66+2.93+3.22+3.55+3.90+4.29 ≈ 25)."""
    n = compute_payback_years(eps_today_basis=2.0, growth_rate=0.10, current_price=25.0)
    assert n == 8


def test_payback_years_returns_one_for_huge_eps() -> None:
    n = compute_payback_years(eps_today_basis=100.0, growth_rate=0.0, current_price=50.0)
    assert n == 1


def test_payback_years_returns_none_for_zero_eps() -> None:
    assert compute_payback_years(eps_today_basis=0.0, growth_rate=0.10,
                                 current_price=10.0) is None


def test_payback_years_returns_none_for_negative_eps() -> None:
    assert compute_payback_years(eps_today_basis=-1.0, growth_rate=0.10,
                                 current_price=10.0) is None


def test_payback_years_returns_none_for_unattainable_target() -> None:
    """Tiny EPS, no growth, huge price → cumulative never reaches target."""
    assert compute_payback_years(
        eps_today_basis=0.01, growth_rate=0.0, current_price=1_000_000.0,
        max_years=10,
    ) is None
