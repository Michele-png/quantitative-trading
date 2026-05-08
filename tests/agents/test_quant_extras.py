"""Tests for the Phil Town extra quantitative checks."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from quantitative_trading.agents.rule_one.quant_extras import (
    DEBT_PAYOFF_THRESHOLD_YEARS,
    DILUTION_YELLOW_MAX,
    QuantExtrasAnalyzer,
)
from quantitative_trading.config import get_config


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


# --------------------------------------------------------------------------
# Synthetic SEC facts builder
# --------------------------------------------------------------------------


def _build_facts(
    years: list[int],
    series_by_concept: dict[str, list[float]],
) -> dict:
    """Mirror the helper used in test_big_five.py — emit a SEC-shaped facts payload."""
    out: dict = {"facts": {"us-gaap": {}, "dei": {}}, "cik": 999, "entityName": "Test"}
    for concept, vals in series_by_concept.items():
        if concept.startswith("EarningsPerShare") or concept.startswith("CommonStockDividendsPer"):
            unit = "USD/shares"
            is_flow = True
        elif concept in (
            "StockholdersEquity", "LongTermDebtNoncurrent", "LongTermDebt",
            "AssetsCurrent", "LiabilitiesCurrent",
            "CommonStockSharesOutstanding", "CashAndCashEquivalentsAtCarryingValue",
        ):
            unit = "USD" if concept != "CommonStockSharesOutstanding" else "shares"
            is_flow = False
        elif concept.startswith("WeightedAverageNumberOf"):
            unit = "shares"
            is_flow = True
        else:
            unit = "USD"
            is_flow = True
        entries = []
        for fy, val in zip(years, vals):
            entry = {
                "end": f"{fy}-12-31",
                "val": val,
                "accn": f"A{fy}",
                "fy": fy,
                "fp": "FY",
                "form": "10-K",
                "filed": f"{fy + 1}-02-01",
            }
            if is_flow:
                entry["start"] = f"{fy}-01-01"
            entries.append(entry)
        out["facts"]["us-gaap"][concept] = {"label": concept, "units": {unit: entries}}
    return out


def _make_analyzer(facts: dict, current_price: float = 100.0,
                   split_factor: float = 1.0) -> QuantExtrasAnalyzer:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = facts
    prices = MagicMock()
    prices.get_close_at.return_value = current_price
    prices.split_factor_since.return_value = split_factor
    return QuantExtrasAnalyzer(edgar, prices)


def _passing_metric(name: str, value: float = 0.20,
                    series: dict[int, float | None] | None = None) -> MetricResult:
    return MetricResult(name=name, value=value, threshold=0.10, passes=True,
                        rationale="ok", series=series or {})


def _make_big_five(
    *, all_pass: bool = True, roic: float = 0.20,
    roic_series: dict[int, float | None] | None = None,
    sales_series: dict[int, float | None] | None = None,
) -> BigFiveResult:
    return BigFiveResult(
        ticker="FAKE", as_of=date(2024, 1, 1), latest_fiscal_year=2023,
        n_years_required=10,
        roic=MetricResult("ROIC", roic, 0.10, all_pass and roic >= 0.10, "ok",
                           series=roic_series or {}),
        sales_growth=MetricResult("Sales Growth", 0.15, 0.10, all_pass, "ok",
                                  series=sales_series or {}),
        eps_growth=_passing_metric("EPS Growth"),
        equity_growth=_passing_metric("Equity Growth"),
        ocf_growth=_passing_metric("OCF Growth"),
        current_ratio=MetricResult("Current Ratio", 2.5, 2.0, True, "ok"),
    )


# --------------------------------------------------------------------------
# Debt-Payoff
# --------------------------------------------------------------------------


def test_debt_payoff_passes_when_under_three_years() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        # FCF = 200 - 50 = 150; LT debt 300 → 2 years payoff
        "LongTermDebtNoncurrent": [300] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert res.debt_payoff.passes
    assert res.debt_payoff.value == pytest.approx(2.0, abs=1e-6)


def test_debt_payoff_fails_when_over_three_years() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [100] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        # FCF = 50; LT debt 300 → 6 years payoff → fail
        "LongTermDebtNoncurrent": [300] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert not res.debt_payoff.passes
    assert res.debt_payoff.value == pytest.approx(6.0, abs=1e-6)


def test_debt_payoff_passes_with_zero_debt() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [0] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert res.debt_payoff.passes
    assert res.debt_payoff.value == 0.0


def test_debt_payoff_fails_with_negative_fcf() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [50] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [80] * 10,  # FCF = -30
        "LongTermDebtNoncurrent": [100] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert not res.debt_payoff.passes
    assert "non-positive" in res.debt_payoff.rationale


# --------------------------------------------------------------------------
# Dilution
# --------------------------------------------------------------------------


def test_dilution_green_for_buybacks() -> None:
    years = list(range(2014, 2024))
    # Shares decreasing by 2% per year → CAGR ≈ -2%
    shares = [1_000_000 * (0.98 ** i) for i in range(10)]
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": shares,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert res.dilution.passes
    assert res.dilution.value < 0
    assert "GREEN" in res.dilution.rationale


def test_dilution_yellow_for_flat_count() -> None:
    years = list(range(2014, 2024))
    shares = [1_000_000] * 10  # flat
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": shares,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert res.dilution.passes
    assert res.dilution.value == pytest.approx(0.0)


def test_dilution_red_for_active_dilution() -> None:
    years = list(range(2014, 2024))
    # Shares growing 5% per year — over the 2%/yr threshold
    shares = [1_000_000 * (1.05 ** i) for i in range(10)]
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": shares,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five())
    assert not res.dilution.passes
    assert res.dilution.value > DILUTION_YELLOW_MAX
    assert "RED" in res.dilution.rationale


# --------------------------------------------------------------------------
# Dividend Quality
# --------------------------------------------------------------------------


def test_dividend_high_roic_no_dividend_passes() -> None:
    """Phil Town: high-ROIC compounder shouldn't be paying dividends."""
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [200] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [0] * 10,
    })
    res = _make_analyzer(facts).evaluate("FAKE", as_of=date(2025, 6, 1),
                                          big_five=_make_big_five(roic=0.25))
    assert res.dividend_quality.passes
    assert res.dividend_details.high_roic_compounder
    assert "compounder" in res.dividend_quality.rationale


def test_dividend_safe_payout_passes() -> None:
    years = list(range(2014, 2024))
    # FCF = 100, dividends = 30 → payout 30%
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [150] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [30] * 10,
        "CommonStockDividendsPerShareDeclared": [0.30] * 10,
    })
    res = _make_analyzer(facts).evaluate(
        "FAKE", as_of=date(2025, 6, 1), big_five=_make_big_five(roic=0.10)
    )
    assert res.dividend_quality.passes
    assert res.dividend_details.payout_band == "pass"


def test_dividend_excessive_payout_fails() -> None:
    years = list(range(2014, 2024))
    # FCF = 100, dividends = 90 → payout 90% → fail
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [150] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [90] * 10,
        "CommonStockDividendsPerShareDeclared": [0.90] * 10,
    })
    res = _make_analyzer(facts).evaluate(
        "FAKE", as_of=date(2025, 6, 1), big_five=_make_big_five(roic=0.10)
    )
    assert not res.dividend_quality.passes
    assert res.dividend_details.payout_band == "fail"


def test_dividend_debt_funded_red_flag_overrides_pass() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [150] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        # LT debt rising every year → debt-funded red flag
        "LongTermDebtNoncurrent": [100 + 10 * i for i in range(10)],
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [30] * 10,
        "CommonStockDividendsPerShareDeclared": [0.30] * 10,
    })
    res = _make_analyzer(facts).evaluate(
        "FAKE", as_of=date(2025, 6, 1), big_five=_make_big_five(roic=0.10)
    )
    assert not res.dividend_quality.passes
    assert res.dividend_details.debt_funded_dividend
    assert "debt-funded" in res.dividend_quality.rationale


def test_dividend_yield_trap_red_flag() -> None:
    years = list(range(2014, 2024))
    # Yield > 7% (DPS 8 / price 100) AND Big 5 trending down
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [150] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [30] * 10,
        "CommonStockDividendsPerShareDeclared": [8.0] * 10,  # 8 / 100 = 8% yield
    })
    big5 = _make_big_five(roic=0.10, sales_series={2022: 100, 2023: 90})  # falling
    res = _make_analyzer(facts).evaluate(
        "FAKE", as_of=date(2025, 6, 1), big_five=big5,
    )
    assert res.dividend_details.yield_trap
    assert not res.dividend_quality.passes
    assert "yield trap" in res.dividend_quality.rationale


def test_dividend_no_dividend_low_roic_still_passes() -> None:
    years = list(range(2014, 2024))
    facts = _build_facts(years, {
        "Revenues": [100] * 10,
        "NetCashProvidedByUsedInOperatingActivities": [150] * 10,
        "PaymentsToAcquirePropertyPlantAndEquipment": [50] * 10,
        "LongTermDebtNoncurrent": [10] * 10,
        "WeightedAverageNumberOfDilutedSharesOutstanding": [1_000_000] * 10,
        "PaymentsOfDividends": [0] * 10,
    })
    res = _make_analyzer(facts).evaluate(
        "FAKE", as_of=date(2025, 6, 1), big_five=_make_big_five(roic=0.08),
    )
    # A non-dividend payer with low ROIC is *not* ideal, but it doesn't fail
    # the dividend check itself — that's what dilution and other checks are for.
    assert res.dividend_quality.passes
    assert not res.dividend_details.pays_dividend


# --------------------------------------------------------------------------
# All-unable handling
# --------------------------------------------------------------------------


def test_all_unable_when_no_revenue_data() -> None:
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = {"facts": {"us-gaap": {}, "dei": {}}, "cik": 999}
    prices = MagicMock()
    analyzer = QuantExtrasAnalyzer(edgar, prices)
    res = analyzer.evaluate("EMPTY", as_of=date(2025, 1, 1), big_five=None)
    assert res.fiscal_year is None
    assert not res.debt_payoff.passes
    assert not res.dilution.passes
    assert not res.dividend_quality.passes


def test_default_thresholds_match_phil_town() -> None:
    assert DEBT_PAYOFF_THRESHOLD_YEARS == 3.0
    assert DILUTION_YELLOW_MAX == pytest.approx(0.02)
