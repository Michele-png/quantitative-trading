"""Unit tests for the Sticker Price + Margin of Safety + Payback Time math."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from value_investing_backend.agents.rule_one.sticker_price import (
    DEFAULT_FGR_CAP,
    DEFAULT_FUTURE_PE_CAP,
    DEFAULT_REQUIRED_RETURN,
    INPUT_DISABLED,
    INPUT_NOT_AVAILABLE,
    INPUT_USED,
    NullAnalystProvider,
    StickerPriceCalculator,
    compute_historical_avg_pe,
    compute_payback_years,
    compute_sticker_price,
    compute_sticker_sensitivity,
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


# --------------------------------------------------------------------------
# Historical average P/E
# --------------------------------------------------------------------------


def test_historical_avg_pe_uses_split_adjusted_close() -> None:
    """Average PE should reflect today's-basis price / today's-basis EPS."""
    # 5 fiscal years, EPS doubles overall, price triples → PE ratio rises.
    eps_history = {
        2020: 1.00, 2021: 1.20, 2022: 1.40, 2023: 1.60, 2024: 2.00,
    }
    fy_ends = {fy: date(fy, 12, 31) for fy in eps_history}
    prices = MagicMock()
    # Close at each year-end (already in today's basis).
    prices.get_close_at.side_effect = lambda t, d: {
        date(2020, 12, 31): 20.0, date(2021, 12, 31): 30.0,
        date(2022, 12, 31): 35.0, date(2023, 12, 31): 40.0,
        date(2024, 12, 31): 60.0,
    }[d]
    prices.split_factor_since.return_value = 1.0  # already in today's basis
    avg, records = compute_historical_avg_pe(
        eps_today_basis_series=eps_history,
        fiscal_year_ends=fy_ends,
        ticker="X",
        price_client=prices,
    )
    # PE per year: 20, 25, 25, 25, 30 → mean = 25
    assert avg == pytest.approx(25.0, abs=0.5)
    assert len(records) == 5
    assert records[0]["fiscal_year"] == 2020
    assert records[-1]["fiscal_year"] == 2024


def test_historical_avg_pe_returns_none_below_min_years() -> None:
    """If only 2 fiscal years can be priced we don't trust an average."""
    eps_history = {2023: 1.0, 2024: 1.0}
    fy_ends = {2023: date(2023, 12, 31), 2024: date(2024, 12, 31)}
    prices = MagicMock()
    prices.get_close_at.return_value = 10.0
    prices.split_factor_since.return_value = 1.0
    avg, records = compute_historical_avg_pe(
        eps_today_basis_series=eps_history,
        fiscal_year_ends=fy_ends,
        ticker="X",
        price_client=prices,
        min_years=3,
    )
    assert avg is None
    assert len(records) == 2  # still surfaced as evidence


def test_historical_avg_pe_handles_split_factor() -> None:
    """If yfinance returns the unadjusted Close, split factor pulls it
    down to today's basis. Ensures we don't double-count splits when
    EPS is already today's basis."""
    eps_history = {
        2018: 1.0, 2019: 1.0, 2020: 1.0, 2021: 1.0, 2022: 1.0,
    }
    fy_ends = {fy: date(fy, 12, 31) for fy in eps_history}
    prices = MagicMock()
    prices.get_close_at.return_value = 200.0
    # 4:1 split since 2018, no further splits.
    prices.split_factor_since.side_effect = lambda t, d: (
        4.0 if d.year == 2018 else 1.0
    )
    avg, records = compute_historical_avg_pe(
        eps_today_basis_series=eps_history,
        fiscal_year_ends=fy_ends,
        ticker="X",
        price_client=prices,
    )
    # 2018 PE: 200/4 / 1 = 50; later years: 200 / 1 = 200.
    # Average over 5 years: (50 + 200*4)/5 = 850/5 = 170.
    assert avg == pytest.approx(170.0, abs=0.5)


# --------------------------------------------------------------------------
# Sensitivity sweep
# --------------------------------------------------------------------------


def test_sensitivity_includes_base_and_neighbouring_growth_rates() -> None:
    """Sweep should always contain the base case (delta=0) and produce
    monotone stickers along the FGR axis (higher growth → higher sticker)."""
    out = compute_sticker_sensitivity(
        eps_today_basis=2.0,
        historical_growth_rate=0.10,
        historical_avg_pe=None,
        required_return=0.15,
        fgr_cap=0.15,
        future_pe_cap=30.0,
        horizon_years=10,
        current_price=50.0,
    )
    fgr_rows = out["future_growth_rate"]
    assert len(fgr_rows) == 5
    inputs = [r["input"] for r in fgr_rows]
    # Base (10%) is in the middle; capped extremes.
    assert pytest.approx(0.10, abs=1e-6) in inputs
    # Sticker is monotonically non-decreasing with FGR.
    stickers = [r["sticker"] for r in fgr_rows]
    assert stickers == sorted(stickers)
    # MoS price tracks sticker / 2.
    for r in fgr_rows:
        assert r["mos_price"] == pytest.approx(r["sticker"] / 2.0, abs=1e-9)


def test_sensitivity_handles_zero_current_price() -> None:
    """When current price is zero, ``implied_mos_pct`` is None — never crash."""
    out = compute_sticker_sensitivity(
        eps_today_basis=2.0,
        historical_growth_rate=0.10,
        historical_avg_pe=None,
        required_return=0.15,
        fgr_cap=0.15,
        future_pe_cap=30.0,
        horizon_years=10,
        current_price=0.0,
    )
    for r in out["future_growth_rate"]:
        assert r["implied_mos_pct"] is None


# --------------------------------------------------------------------------
# StickerPriceCalculator end-to-end (mocked clients)
# --------------------------------------------------------------------------


def _build_eps_facts(years: list[int], eps_vals: list[float]) -> dict:
    """SEC-shaped facts payload with EarningsPerShareDiluted only."""
    entries = []
    for fy, val in zip(years, eps_vals):
        entries.append({
            "end": f"{fy}-12-31", "start": f"{fy}-01-01",
            "val": val, "accn": f"A{fy}", "fy": fy, "fp": "FY",
            "form": "10-K", "filed": f"{fy + 1}-02-01",
        })
    return {
        "facts": {"us-gaap": {
            "EarningsPerShareDiluted": {
                "label": "EPS Diluted",
                "units": {"USD/shares": entries},
            },
            "Revenues": {
                "label": "Revenues",
                "units": {"USD": [
                    {"end": f"{fy}-12-31", "start": f"{fy}-01-01",
                     "val": 100.0, "accn": f"A{fy}", "fy": fy, "fp": "FY",
                     "form": "10-K", "filed": f"{fy + 1}-02-01"}
                    for fy in years
                ]},
            },
        }, "dei": {}},
        "cik": 999,
    }


def test_calculator_records_inputs_used_when_no_pe_history() -> None:
    """Without historical EPS / FY ends the historical-PE input is
    explicitly recorded as not_available and analyst as disabled (default
    null provider)."""
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = _build_eps_facts(
        list(range(2014, 2024)), [1.0 * 1.10 ** i for i in range(10)],
    )
    prices = MagicMock()
    prices.split_factor_since.return_value = 1.0
    prices.get_close_at.return_value = 50.0
    calc = StickerPriceCalculator(edgar, prices)
    sticker, _ = calc.evaluate(
        "FAKE", as_of=date(2025, 6, 1), historical_eps_growth=0.10,
    )
    assert sticker.inputs_used["historical_growth"] == INPUT_USED
    assert sticker.inputs_used["historical_avg_pe"] == INPUT_NOT_AVAILABLE
    assert sticker.inputs_used["analyst_growth"] == INPUT_DISABLED


def test_calculator_uses_historical_avg_pe_when_provided() -> None:
    """When EPS history + FY ends are passed in, the calculator computes
    a historical avg PE and records ``historical_avg_pe = used``."""
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = _build_eps_facts(
        list(range(2019, 2024)), [1.0, 1.10, 1.21, 1.33, 1.46],
    )
    prices = MagicMock()
    prices.split_factor_since.return_value = 1.0
    prices.get_close_at.return_value = 20.0
    calc = StickerPriceCalculator(edgar, prices)
    eps_hist = {fy: v for fy, v in zip(
        range(2019, 2024), [1.0, 1.10, 1.21, 1.33, 1.46],
    )}
    fy_ends = {fy: date(fy, 12, 31) for fy in eps_hist}
    sticker, _ = calc.evaluate(
        "FAKE", as_of=date(2025, 6, 1), historical_eps_growth=0.10,
        eps_history=eps_hist, fiscal_year_ends=fy_ends,
    )
    assert sticker.historical_avg_pe is not None
    assert sticker.inputs_used["historical_avg_pe"] == INPUT_USED
    assert len(sticker.pe_history) == 5
    # Sensitivity sweep populated.
    assert sticker.sensitivity["future_growth_rate"]
    assert sticker.sensitivity["required_return"]


def test_calculator_uses_analyst_provider_when_credentials_available() -> None:
    """A wired-up provider that returns a lower-than-historical estimate
    binds the FGR — and the result records ``analyst_growth = used``."""

    class StubProvider:
        def credentials_available(self) -> bool:
            return True

        def get_eps_growth_5y(self, ticker: str, as_of: date) -> float | None:  # noqa: ARG002
            return 0.06  # below historical 10%

    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = _build_eps_facts(
        list(range(2014, 2024)), [1.0 * 1.10 ** i for i in range(10)],
    )
    prices = MagicMock()
    prices.split_factor_since.return_value = 1.0
    prices.get_close_at.return_value = 50.0
    calc = StickerPriceCalculator(edgar, prices, analyst_provider=StubProvider())
    sticker, _ = calc.evaluate(
        "FAKE", as_of=date(2025, 6, 1), historical_eps_growth=0.10,
    )
    assert sticker.analyst_growth_estimate == pytest.approx(0.06, abs=1e-6)
    assert sticker.inputs_used["analyst_growth"] == INPUT_USED
    # Future growth was clamped to 6% (analyst), then to 0..15% by fgr_cap → 6%.
    assert sticker.future_growth_rate == pytest.approx(0.06, abs=1e-6)


def test_null_analyst_provider_returns_none_and_disables_input() -> None:
    p = NullAnalystProvider()
    assert p.credentials_available() is False
    assert p.get_eps_growth_5y("X", date(2024, 1, 1)) is None
