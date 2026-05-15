"""Unit tests for BigFiveAnalyzer using mocked clients.

The data layer (EDGAR + prices) is exercised in its own test module. These
tests focus on the Big 5 analytical logic given known inputs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantitative_trading.agents.rule_one.big_five import (
    DEFAULT_THRESHOLD,
    BigFiveAnalyzer,
    _adjust_eps_for_splits,
    _cagr,
)
from quantitative_trading.config import get_config
from quantitative_trading.data.pit_facts import FactValue


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _fv(*, val: float, fy: int, filed: str = "2024-01-01") -> FactValue:
    """Build a FactValue with sensible defaults for testing."""
    return FactValue(
        concept="Revenues",
        end=date(fy, 12, 31),
        start=date(fy, 1, 1),
        val=val,
        accn=f"A{fy}",
        fy_filing=fy,
        fp="FY",
        form="10-K",
        filed=date.fromisoformat(filed),
        unit="USD",
    )


def test_cagr_basic_doubling_over_10_years() -> None:
    # Doubling in 10 years = 7.18% CAGR
    cagr = _cagr(100.0, 200.0, 10)
    assert cagr == pytest.approx(0.0718, abs=1e-3)


def test_cagr_15_pct_threshold_doubles_in_5_years() -> None:
    # Phil Town's "doubles in 5 years" payback rule corresponds to ~14.87% CAGR
    cagr = _cagr(100.0, 200.0, 5)
    assert cagr == pytest.approx(0.1487, abs=1e-3)


def test_cagr_undefined_for_negative_or_zero_first() -> None:
    assert _cagr(0.0, 100.0, 5) is None
    assert _cagr(-10.0, 100.0, 5) is None
    assert _cagr(100.0, -50.0, 5) is None


def test_adjust_eps_for_splits_normalizes_to_today_basis() -> None:
    """A pre-split EPS should be divided by the cumulative split factor since filing."""
    eps_series: dict[int, FactValue | None] = {
        2010: FactValue(
            concept="EarningsPerShareDiluted",
            end=date(2010, 12, 31), start=date(2010, 1, 1), val=15.41,
            accn="K2010", fy_filing=2010, fp="FY", form="10-K",
            filed=date(2010, 10, 27), unit="USD/shares",
        ),
        2014: FactValue(
            concept="EarningsPerShareDiluted",
            end=date(2014, 9, 27), start=date(2013, 9, 29), val=6.45,
            accn="K2014", fy_filing=2014, fp="FY", form="10-K",
            filed=date(2014, 10, 27), unit="USD/shares",
        ),
    }
    price_client = MagicMock()
    # Apple split 7:1 in 2014 and 4:1 in 2020. After 2010 filing: both → 28x.
    # After 2014 filing: only 4:1 → 4x.
    price_client.split_factor_since.side_effect = lambda t, d: (
        28.0 if d.year == 2010 else 4.0 if d.year == 2014 else 1.0
    )
    out = _adjust_eps_for_splits(eps_series, "AAPL", price_client)
    assert out[2010] == pytest.approx(15.41 / 28, abs=0.01)
    assert out[2014] == pytest.approx(6.45 / 4, abs=0.01)


def test_adjust_eps_handles_missing_year() -> None:
    eps_series: dict[int, FactValue | None] = {2010: None}
    out = _adjust_eps_for_splits(eps_series, "X", MagicMock())
    assert out[2010] is None


def _make_analyzer_with_facts(facts_payload: dict, monkeypatch: pytest.MonkeyPatch):
    """Build a BigFiveAnalyzer wired to mock clients returning a synthetic facts dict."""
    edgar = MagicMock()
    edgar.get_cik.return_value = 999
    edgar.get_company_facts.return_value = facts_payload
    prices = MagicMock()
    prices.split_factor_since.return_value = 1.0
    return BigFiveAnalyzer(edgar, prices)


def _build_facts(years: list[int], series_by_concept: dict[str, list[float]]) -> dict:
    """Build a SEC-shaped facts payload for `years` and the given concept values."""
    out: dict = {"facts": {"us-gaap": {}, "dei": {}}, "cik": 999, "entityName": "Test"}
    for concept, vals in series_by_concept.items():
        # Decide unit/period structure from concept name.
        if concept.startswith("EarningsPerShare"):
            unit = "USD/shares"
            is_flow = True
        elif concept in ("StockholdersEquity", "LongTermDebtNoncurrent",
                         "AssetsCurrent", "LiabilitiesCurrent"):
            unit = "USD"
            is_flow = False
        elif concept in (
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "CommonStockSharesOutstanding",
        ):
            # Share counts are flow-style in XBRL (have start/end) and
            # carry the ``shares`` unit, not ``USD``.
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


def test_company_passing_all_big_five(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic company growing every metric at 15%/yr passes all checks."""
    years = list(range(2014, 2024))
    rev = [100 * 1.15 ** (i) for i in range(10)]
    ni = [10 * 1.15 ** (i) for i in range(10)]
    eq = [50 * 1.15 ** (i) for i in range(10)]
    ocf = [12 * 1.15 ** (i) for i in range(10)]
    eps = [1.0 * 1.15 ** (i) for i in range(10)]
    debt = [20 * 1.0 for _ in range(10)]  # flat LT debt
    facts = _build_facts(
        years,
        {
            "Revenues": rev,
            "NetIncomeLoss": ni,
            "EarningsPerShareDiluted": eps,
            "StockholdersEquity": eq,
            "NetCashProvidedByUsedInOperatingActivities": ocf,
            "LongTermDebtNoncurrent": debt,
            "AssetsCurrent": [100] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1))

    assert result.latest_fiscal_year == 2023
    assert result.sales_growth.passes
    assert result.eps_growth.passes
    assert result.equity_growth.passes
    assert result.ocf_growth.passes
    # ROIC: 10*1.15^9 / (50*1.15^9 + 20) — ratio of NI to (eq + debt)
    # Most recent: 35.18 / (175.91 + 20) = ~17.96%, passes.
    assert result.roic.passes
    assert result.all_pass


def test_company_failing_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flat revenues -> 0% CAGR -> sales growth fails."""
    years = list(range(2014, 2024))
    flat = [100.0] * 10
    facts = _build_facts(
        years,
        {
            "Revenues": flat,
            "NetIncomeLoss": flat,
            "EarningsPerShareDiluted": [1.0] * 10,
            "StockholdersEquity": flat,
            "NetCashProvidedByUsedInOperatingActivities": flat,
            "LongTermDebtNoncurrent": [10.0] * 10,
            "AssetsCurrent": [100] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1))
    assert not result.sales_growth.passes
    assert not result.eps_growth.passes
    assert not result.all_pass


def test_company_with_no_data_returns_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = {"facts": {"us-gaap": {}, "dei": {}}, "cik": 999}
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("EMPTY", as_of=date(2025, 1, 1))
    assert result.latest_fiscal_year is None
    assert not result.all_pass
    assert "No fiscal year revenue data" in result.roic.rationale


def test_company_with_negative_starting_equity_eps_growth_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative starting EPS -> CAGR ill-defined -> growth check fails."""
    years = list(range(2014, 2024))
    facts = _build_facts(
        years,
        {
            "Revenues": [100 * 1.15 ** i for i in range(10)],
            "NetIncomeLoss": [10 * 1.15 ** i for i in range(10)],
            "EarningsPerShareDiluted": [-0.5] + [1.0 * 1.15 ** i for i in range(1, 10)],
            "StockholdersEquity": [50 * 1.15 ** i for i in range(10)],
            "NetCashProvidedByUsedInOperatingActivities": [12 * 1.15 ** i for i in range(10)],
            "LongTermDebtNoncurrent": [20] * 10,
            "AssetsCurrent": [100] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1))
    assert not result.eps_growth.passes
    assert "ill-defined" in result.eps_growth.rationale


def test_growth_with_partial_history_does_not_pass_decision_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phil Town's rule wants 10 years; 7 years of data must not be a
    decision-grade pass. The metric still surfaces the underlying CAGR
    (so the dashboard can show "would have been X%") but ``passes`` is
    False until the window is full.
    """
    years = list(range(2017, 2024))  # 7 years
    rev = [100 * 1.15 ** i for i in range(7)]
    facts = _build_facts(
        years,
        {
            "Revenues": rev,
            "NetIncomeLoss": [10 * 1.15 ** i for i in range(7)],
            "EarningsPerShareDiluted": [1.0 * 1.15 ** i for i in range(7)],
            "StockholdersEquity": [50 * 1.15 ** i for i in range(7)],
            "NetCashProvidedByUsedInOperatingActivities": [12 * 1.15 ** i for i in range(7)],
            "LongTermDebtNoncurrent": [20] * 7,
            "AssetsCurrent": [100] * 7,
            "LiabilitiesCurrent": [40] * 7,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1), n_years=10)
    # Underlying CAGR is well above 10% — but decision-grade pass is
    # blocked because we're missing 3 years of evidence.
    assert result.sales_growth.value is not None
    assert result.sales_growth.value >= 0.10
    assert not result.sales_growth.passes
    assert not result.sales_growth.decision_grade
    assert "decision-grade pass requires the full window" in result.sales_growth.rationale


def test_growth_value_is_none_below_min_years(monkeypatch: pytest.MonkeyPatch) -> None:
    """With <5 populated years we don't even compute a CAGR — too noisy."""
    years = list(range(2021, 2024))  # 3 years only
    rev = [100 * 1.15 ** i for i in range(3)]
    facts = _build_facts(
        years,
        {
            "Revenues": rev,
            "NetIncomeLoss": [10 * 1.15 ** i for i in range(3)],
            "EarningsPerShareDiluted": [1.0 * 1.15 ** i for i in range(3)],
            "StockholdersEquity": [50 * 1.15 ** i for i in range(3)],
            "NetCashProvidedByUsedInOperatingActivities": [12 * 1.15 ** i for i in range(3)],
            "LongTermDebtNoncurrent": [20] * 3,
            "AssetsCurrent": [100] * 3,
            "LiabilitiesCurrent": [40] * 3,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1), n_years=10)
    assert result.sales_growth.value is None
    assert not result.sales_growth.passes
    assert not result.sales_growth.decision_grade
    assert "need ≥ 5" in result.sales_growth.rationale


def test_full_10y_window_is_decision_grade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full 10-year window with passing CAGR sets ``decision_grade=True``."""
    years = list(range(2014, 2024))
    rev = [100 * 1.15 ** i for i in range(10)]
    facts = _build_facts(
        years,
        {
            "Revenues": rev,
            "NetIncomeLoss": [10 * 1.15 ** i for i in range(10)],
            "EarningsPerShareDiluted": [1.0 * 1.15 ** i for i in range(10)],
            "StockholdersEquity": [50 * 1.15 ** i for i in range(10)],
            "NetCashProvidedByUsedInOperatingActivities": [12 * 1.15 ** i for i in range(10)],
            "LongTermDebtNoncurrent": [20] * 10,
            "AssetsCurrent": [100] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1), n_years=10)
    assert result.sales_growth.passes
    assert result.sales_growth.decision_grade
    assert result.roic.passes
    assert result.roic.decision_grade


def test_eps_falls_back_to_ni_div_shares_when_concept_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issuer that omits ``EarningsPerShareDiluted`` but reports both
    ``NetIncomeLoss`` and ``WeightedAverageNumberOfDilutedSharesOutstanding``
    (e.g. companies that only tag the components, not the ratio) should
    still get a populated EPS series via the SEC NI/shares fallback.
    """
    years = list(range(2014, 2024))
    ni_vals = [10_000_000 * 1.15 ** i for i in range(10)]
    sh_vals = [1_000_000.0] * 10  # constant share count → EPS grows like NI
    facts = _build_facts(
        years,
        {
            "Revenues": [100 * 1.15 ** i for i in range(10)],
            "NetIncomeLoss": ni_vals,
            # NB: no EarningsPerShareDiluted at all — the primary concept
            # is missing, mirroring Visa's actual XBRL profile.
            "WeightedAverageNumberOfDilutedSharesOutstanding": sh_vals,
            "StockholdersEquity": [50_000_000 * 1.15 ** i for i in range(10)],
            "NetCashProvidedByUsedInOperatingActivities": [
                12_000_000 * 1.15 ** i for i in range(10)
            ],
            "LongTermDebtNoncurrent": [20_000_000] * 10,
            "AssetsCurrent": [100_000_000] * 10,
            "LiabilitiesCurrent": [40_000_000] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1), n_years=10)
    # EPS is recovered: NI / shares grows at 15%/yr like NI.
    assert result.eps_growth.value is not None
    assert result.eps_growth.value == pytest.approx(0.15, abs=0.01)
    assert result.eps_growth.passes
    assert result.eps_growth.data_source == "sec_ni_over_shares"


def test_eps_marked_unavailable_when_no_source_has_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither EPS nor NI/shares are tagged in XBRL and yfinance is
    stubbed empty, the EPS metric reports value=None and
    ``data_source='unavailable'`` so the dashboard can show NO DATA.
    """
    years = list(range(2014, 2024))
    facts = _build_facts(
        years,
        {
            "Revenues": [100 * 1.15 ** i for i in range(10)],
            # Intentional gap: no NI either (so the NI/shares fallback
            # also returns nothing).
            "StockholdersEquity": [50 * 1.15 ** i for i in range(10)],
            "NetCashProvidedByUsedInOperatingActivities": [
                12 * 1.15 ** i for i in range(10)
            ],
            "LongTermDebtNoncurrent": [20] * 10,
            "AssetsCurrent": [100] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    # Stub the yfinance fallback — tests must never hit the network.
    monkeypatch.setattr(
        "quantitative_trading.agents.rule_one.big_five._yfinance_eps_series",
        lambda *args, **kwargs: {},
    )
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1), n_years=10)
    assert result.eps_growth.value is None
    assert not result.eps_growth.passes
    assert result.eps_growth.data_source == "unavailable"


def test_current_ratio_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    years = list(range(2014, 2024))
    base = [100 * 1.15 ** i for i in range(10)]
    # current ratio = 200/40 = 5.0 → passes
    facts = _build_facts(
        years,
        {
            "Revenues": base,
            "NetIncomeLoss": [10 * 1.15 ** i for i in range(10)],
            "EarningsPerShareDiluted": [1.0 * 1.15 ** i for i in range(10)],
            "StockholdersEquity": [50 * 1.15 ** i for i in range(10)],
            "NetCashProvidedByUsedInOperatingActivities": [12 * 1.15 ** i for i in range(10)],
            "LongTermDebtNoncurrent": [20] * 10,
            "AssetsCurrent": [200] * 10,
            "LiabilitiesCurrent": [40] * 10,
        },
    )
    analyzer = _make_analyzer_with_facts(facts, monkeypatch)
    result = analyzer.evaluate("FAKE", as_of=date(2025, 6, 1))
    assert result.current_ratio.passes
    assert result.current_ratio.value == pytest.approx(5.0, abs=0.01)
