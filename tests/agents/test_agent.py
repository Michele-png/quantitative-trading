"""Unit tests for the RuleOneAgent orchestrator (mocked components)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantitative_trading.agents.rule_one.agent import AgentResult, RuleOneAgent
from quantitative_trading.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from quantitative_trading.agents.rule_one.four_ms_llm import (
    FourMsResult,
    MCheck,
)
from quantitative_trading.agents.rule_one.sticker_price import (
    PaybackTimeResult,
    StickerPriceResult,
)
from quantitative_trading.config import get_config


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _passing_metric(name: str) -> MetricResult:
    return MetricResult(name=name, value=0.20, threshold=0.10, passes=True,
                        rationale="passes")


def _failing_metric(name: str) -> MetricResult:
    return MetricResult(name=name, value=0.05, threshold=0.10, passes=False,
                        rationale="fails")


def _make_big_five(all_pass: bool) -> BigFiveResult:
    metric = _passing_metric if all_pass else _failing_metric
    return BigFiveResult(
        ticker="FAKE", as_of=date(2020, 1, 1), latest_fiscal_year=2019,
        n_years_required=10,
        roic=metric("ROIC"), sales_growth=metric("Sales Growth"),
        eps_growth=metric("EPS Growth"), equity_growth=metric("Equity Growth"),
        ocf_growth=metric("OCF Growth"),
        current_ratio=MetricResult("Current Ratio", 2.5, 2.0, True, "ok"),
    )


def _make_sticker(passes: bool) -> StickerPriceResult:
    return StickerPriceResult(
        ticker="FAKE", as_of=date(2020, 1, 1), fiscal_year_used=2019,
        eps_today_basis=2.0, historical_growth_rate=0.20,
        future_growth_rate=0.15, future_pe=30.0, future_eps=10.0,
        sticker_price=60.0, margin_of_safety_price=30.0,
        current_price=20.0 if passes else 50.0,
        margin_of_safety_passes=passes, rationale="ok",
    )


def _make_payback(passes: bool) -> PaybackTimeResult:
    return PaybackTimeResult(
        ticker="FAKE", as_of=date(2020, 1, 1),
        payback_years=5 if passes else 12,
        threshold_years=8.0, passes=passes, rationale="ok",
    )


def _make_4ms(meaning: bool, moat: bool, mgmt: bool) -> FourMsResult:
    return FourMsResult(
        ticker="FAKE", as_of=date(2020, 1, 1), fiscal_year=2019,
        accession="0001-1", model="claude-test",
        meaning=MCheck("Meaning", meaning, "ok"),
        moat=MCheck("Moat", moat, "ok"),
        management=MCheck("Management", mgmt, "ok"),
        cached=False, raw_response={},
    )


def _build_agent_result(*, big5_pass: bool, sticker_pass: bool,
                        payback_pass: bool, four_ms: FourMsResult | None) -> AgentResult:
    return AgentResult(
        ticker="FAKE", as_of=date(2020, 1, 1),
        big_five=_make_big_five(big5_pass),
        sticker=_make_sticker(sticker_pass),
        payback=_make_payback(payback_pass),
        four_ms=four_ms,
    )


# --------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------


def test_full_buy_requires_all_9_checks() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    assert r.is_buy_full
    assert r.is_buy_quant_only


def test_failing_one_quant_breaks_full_and_quant_only() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=False,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    assert not r.is_buy_full
    assert not r.is_buy_quant_only


def test_failing_one_llm_breaks_full_only() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, False, True))
    assert not r.is_buy_full
    assert r.is_buy_quant_only


def test_no_llm_run_means_full_is_false() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True, four_ms=None)
    assert not r.is_buy_full
    assert r.is_buy_quant_only


def test_quant_check_results_dict_has_seven_keys() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    assert set(r.quant_check_results.keys()) == {
        "roic", "sales_growth", "eps_growth", "equity_growth", "ocf_growth",
        "margin_of_safety", "payback_time",
    }


def test_random_qual_variant_deterministic_for_same_inputs() -> None:
    """Same (ticker, as_of, seed) → same draws → same answer."""
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    a1 = r.is_buy_quant_random_qual(seed="abc", base_rate_meaning=0.5,
                                     base_rate_moat=0.5, base_rate_management=0.5)
    a2 = r.is_buy_quant_random_qual(seed="abc", base_rate_meaning=0.5,
                                     base_rate_moat=0.5, base_rate_management=0.5)
    assert a1 == a2


def test_random_qual_variant_changes_with_seed() -> None:
    """Different seed → potentially different answer (over many draws)."""
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    answers = {
        r.is_buy_quant_random_qual(seed=str(s), base_rate_meaning=0.5,
                                    base_rate_moat=0.5, base_rate_management=0.5)
        for s in range(20)
    }
    # Over 20 different seeds we expect both True and False to appear.
    assert answers == {True, False}


def test_random_qual_zero_base_rates_always_false() -> None:
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    assert not r.is_buy_quant_random_qual(seed="x", base_rate_meaning=0.0,
                                           base_rate_moat=1.0, base_rate_management=1.0)


def test_random_qual_one_base_rates_equals_quant_pass() -> None:
    """Base rate 1.0 for all 3 Ms means qual draws always pass → equals quant_pass."""
    r = _build_agent_result(big5_pass=True, sticker_pass=True,
                             payback_pass=True,
                             four_ms=_make_4ms(True, True, True))
    assert r.is_buy_quant_random_qual(seed="x", base_rate_meaning=1.0,
                                       base_rate_moat=1.0, base_rate_management=1.0)
    r2 = _build_agent_result(big5_pass=False, sticker_pass=True,
                              payback_pass=True,
                              four_ms=_make_4ms(True, True, True))
    assert not r2.is_buy_quant_random_qual(seed="x", base_rate_meaning=1.0,
                                            base_rate_moat=1.0, base_rate_management=1.0)


# --------------------------------------------------------------------------
# RuleOneAgent.evaluate orchestration
# --------------------------------------------------------------------------


def test_evaluate_skips_llm_when_include_llm_false() -> None:
    edgar = MagicMock()
    prices = MagicMock()
    anthropic = MagicMock()

    with patch(
        "quantitative_trading.agents.rule_one.agent.BigFiveAnalyzer"
    ) as MockB5, patch(
        "quantitative_trading.agents.rule_one.agent.StickerPriceCalculator"
    ) as MockSP, patch(
        "quantitative_trading.agents.rule_one.agent.FourMsAnalyzer"
    ) as MockFM:
        MockB5.return_value.evaluate.return_value = _make_big_five(True)
        MockSP.return_value.evaluate.return_value = (
            _make_sticker(True), _make_payback(True)
        )
        MockFM.return_value.evaluate.return_value = _make_4ms(True, True, True)
        agent = RuleOneAgent(edgar, prices, anthropic_client=anthropic)
        result = agent.evaluate("FAKE", as_of=date(2020, 1, 1), include_llm=False)
    assert result.four_ms is None
    MockFM.return_value.evaluate.assert_not_called()


def test_evaluate_runs_llm_when_include_llm_true() -> None:
    edgar = MagicMock()
    prices = MagicMock()
    anthropic = MagicMock()

    with patch(
        "quantitative_trading.agents.rule_one.agent.BigFiveAnalyzer"
    ) as MockB5, patch(
        "quantitative_trading.agents.rule_one.agent.StickerPriceCalculator"
    ) as MockSP, patch(
        "quantitative_trading.agents.rule_one.agent.FourMsAnalyzer"
    ) as MockFM:
        MockB5.return_value.evaluate.return_value = _make_big_five(True)
        MockSP.return_value.evaluate.return_value = (
            _make_sticker(True), _make_payback(True)
        )
        MockFM.return_value.evaluate.return_value = _make_4ms(True, False, True)
        agent = RuleOneAgent(edgar, prices, anthropic_client=anthropic)
        result = agent.evaluate("FAKE", as_of=date(2020, 1, 1), include_llm=True)
    assert result.four_ms is not None
    assert result.is_buy_quant_only
    assert not result.is_buy_full  # moat failed
