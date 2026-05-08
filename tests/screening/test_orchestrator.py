"""Tests for the ScreeningOrchestrator hard-gating logic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.agents.rule_one.agent import AgentResult
from quantitative_trading.agents.rule_one.big_five import (
    BigFiveResult,
    MetricResult,
)
from quantitative_trading.agents.rule_one.four_ms_llm import (
    FourMsResult,
    MCheck,
)
from quantitative_trading.agents.rule_one.management_llm import (
    ManagementResult,
    SubCheck,
)
from quantitative_trading.agents.rule_one.quant_extras import (
    DividendQualityDetails,
    QuantExtrasResult,
)
from quantitative_trading.agents.rule_one.sticker_price import (
    PaybackTimeResult,
    StickerPriceResult,
)
from quantitative_trading.config import get_config
from quantitative_trading.screening.orchestrator import (
    ScreeningOrchestrator,
    refresh_records,
    screen_tickers,
)
from quantitative_trading.screening.records import HardGatePolicy


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _ok(name: str, value: float = 0.20) -> MetricResult:
    return MetricResult(name=name, value=value, threshold=0.10, passes=True,
                        rationale="ok")


def _fail(name: str) -> MetricResult:
    return MetricResult(name=name, value=0.05, threshold=0.10, passes=False,
                        rationale="fail")


def _make_big_five(*, all_pass: bool = True) -> BigFiveResult:
    metric = _ok if all_pass else _fail
    return BigFiveResult(
        ticker="FAKE", as_of=date(2024, 1, 1), latest_fiscal_year=2023,
        n_years_required=10,
        roic=metric("ROIC"), sales_growth=metric("Sales Growth"),
        eps_growth=metric("EPS Growth"), equity_growth=metric("Equity Growth"),
        ocf_growth=metric("OCF Growth"),
        current_ratio=MetricResult("Current Ratio", 2.5, 2.0, True, "ok"),
    )


def _make_quant_extras(
    *, debt: bool = True, dilution: bool = True, dividend: bool = True,
) -> QuantExtrasResult:
    return QuantExtrasResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year=2023,
        debt_payoff=MetricResult(
            "Debt Payoff (yrs)", 1.5, 3.0, debt, "ok",
        ),
        dilution=MetricResult(
            "Dilution CAGR", -0.01, 0.02, dilution, "buybacks",
        ),
        dividend_quality=MetricResult(
            "Dividend Quality", 0.30, 0.60, dividend, "ok",
        ),
        dividend_details=DividendQualityDetails(
            high_roic_compounder=False, pays_dividend=True,
            payout_ratio=0.30, payout_band="pass",
            debt_funded_dividend=False, yield_trap=False,
            dividend_yield=0.02, dividend_growth_pct=0.05,
        ),
    )


def _make_four_ms(
    *, meaning: bool = True, moat: bool = True, mgmt: bool = True,
) -> FourMsResult:
    return FourMsResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year=2023,
        accession="A1", model="claude-opus-4-7",
        meaning=MCheck("Meaning", meaning, "ok"),
        moat=MCheck("Moat", moat, "ok",
                    details={"moat_type": "brand"}),
        management=MCheck("Management", mgmt, "ok"),
        cached=False, raw_response={},
    )


def _make_management(
    *, blame: bool = True, long_short: bool = True, clarity: bool = True,
    compensation: bool = True, insider: bool = True,
) -> ManagementResult:
    return ManagementResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year=2023,
        bundle_hash="abc", model="claude-opus-4-7", cached=False,
        blame=SubCheck("Blame", blame, 1.0, "ok"),
        long_short=SubCheck("LongShort", long_short, 5.0, "ok",
                             details={"ratio": 5.0}),
        clarity=SubCheck("Clarity", clarity, 8.0, "ok"),
        compensation=SubCheck("Compensation", compensation, None, "ok"),
        insider=SubCheck("Insider", insider, 100_000.0, "ok",
                         details={"net_open_market_value_usd": 100_000.0}),
    )


def _make_sticker(*, mos_pass: bool = True) -> StickerPriceResult:
    return StickerPriceResult(
        ticker="FAKE", as_of=date(2024, 1, 1), fiscal_year_used=2023,
        eps_today_basis=2.0, historical_growth_rate=0.20,
        future_growth_rate=0.15, future_pe=30.0, future_eps=10.0,
        sticker_price=60.0, margin_of_safety_price=30.0,
        current_price=20.0 if mos_pass else 50.0,
        margin_of_safety_passes=mos_pass, rationale="ok",
    )


def _make_payback(passes: bool = True) -> PaybackTimeResult:
    return PaybackTimeResult(
        ticker="FAKE", as_of=date(2024, 1, 1),
        payback_years=5 if passes else 12,
        threshold_years=8.0, passes=passes, rationale="ok",
    )


def _agent_result(
    *, big5: bool = True, mos: bool = True, payback: bool = True,
    extras_pass: bool = True, fm_pass: bool = True, mgmt_pass: bool = True,
) -> AgentResult:
    return AgentResult(
        ticker="FAKE", as_of=date(2024, 1, 1),
        big_five=_make_big_five(all_pass=big5),
        sticker=_make_sticker(mos_pass=mos),
        payback=_make_payback(payback),
        four_ms=_make_four_ms(meaning=fm_pass, moat=fm_pass, mgmt=mgmt_pass),
        management=_make_management(
            blame=mgmt_pass, long_short=mgmt_pass, clarity=mgmt_pass,
            compensation=mgmt_pass, insider=mgmt_pass,
        ),
        quant_extras=_make_quant_extras(
            debt=extras_pass, dilution=extras_pass, dividend=extras_pass,
        ),
    )


# --------------------------------------------------------------------------
# Hard-gating logic
# --------------------------------------------------------------------------


def test_screen_passes_when_all_gates_satisfied() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result()
    orch = ScreeningOrchestrator(agent=agent)
    out = orch.screen(["FAKE"], date(2024, 1, 1))
    assert len(out) == 1
    rec = out[0]
    assert rec.screen_passes
    assert rec.failed_gates == []


def test_screen_fails_when_any_big_five_fails() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(big5=False)
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1)
    )[0]
    assert not rec.screen_passes
    assert "roic" in rec.failed_gates
    assert "sales_growth" in rec.failed_gates


def test_screen_fails_when_dilution_fails() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(extras_pass=False)
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1)
    )[0]
    assert not rec.screen_passes
    assert "dilution" in rec.failed_gates
    assert "debt_payoff" in rec.failed_gates
    assert "dividend_quality" in rec.failed_gates


def test_screen_fails_when_management_fails() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(mgmt_pass=False)
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1)
    )[0]
    assert not rec.screen_passes
    assert "management" in rec.failed_gates


def test_mos_is_NOT_a_gate_in_default_policy() -> None:
    agent = MagicMock()
    # MoS fails (price above safety price) but everything else passes.
    agent.evaluate.return_value = _agent_result(mos=False)
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1)
    )[0]
    assert rec.screen_passes
    assert rec.failed_gates == []


def test_refresh_includes_mos_percent() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(mos=True)
    orch = ScreeningOrchestrator(agent=agent)
    rec = orch.screen(["FAKE"], date(2024, 1, 1), include_mos=True)[0]
    assert rec.mos_percent is not None
    # MoS price 30, current 20 → (30-20)/20 = 0.50
    assert rec.mos_percent == pytest.approx(0.5)


def test_screen_does_not_populate_mos() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(mos=True)
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1), include_mos=False
    )[0]
    assert rec.mos_percent is None
    assert rec.sticker_price is None


def test_refresh_records_top_level_function_runs_full_pipeline() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result()
    orch = ScreeningOrchestrator(agent=agent)
    out = refresh_records(["FAKE"], date(2024, 1, 1), orchestrator=orch)
    assert out[0].mos_percent is not None
    assert out[0].screen_passes


def test_screen_tickers_top_level_function_skips_mos() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result()
    orch = ScreeningOrchestrator(agent=agent)
    out = screen_tickers(["FAKE"], date(2024, 1, 1), orchestrator=orch)
    assert out[0].mos_percent is None


def test_evaluation_error_yields_failed_record() -> None:
    agent = MagicMock()
    agent.evaluate.side_effect = RuntimeError("EDGAR exploded")
    orch = ScreeningOrchestrator(agent=agent)
    out = orch.screen(["FAKE"], date(2024, 1, 1))
    assert len(out) == 1
    assert not out[0].screen_passes
    assert out[0].error == "EDGAR exploded"
    assert out[0].failed_gates == ["evaluation_error"]


def test_blank_or_empty_tickers_skipped() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result()
    orch = ScreeningOrchestrator(agent=agent)
    out = orch.screen(["", "  ", "FAKE"], date(2024, 1, 1))
    assert len(out) == 1
    assert out[0].ticker == "FAKE"


def test_custom_policy_can_relax_management() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result(mgmt_pass=False)
    relaxed = HardGatePolicy.default()
    relaxed = HardGatePolicy(
        require_roic=True, require_sales_growth=True,
        require_eps_growth=True, require_equity_growth=True,
        require_ocf_growth=True, require_meaning=True, require_moat=True,
        require_management=False,  # relaxed
        require_debt_payoff=True, require_dilution=True,
        require_dividend_quality=True,
    )
    orch = ScreeningOrchestrator(agent=agent, policy=relaxed)
    rec = orch.screen(["FAKE"], date(2024, 1, 1))[0]
    assert rec.screen_passes
    assert "management" not in rec.failed_gates


def test_record_includes_management_sub_checks() -> None:
    agent = MagicMock()
    agent.evaluate.return_value = _agent_result()
    rec = ScreeningOrchestrator(agent=agent).screen(
        ["FAKE"], date(2024, 1, 1)
    )[0]
    assert rec.check_mgmt_blame
    assert rec.check_mgmt_clarity
    assert rec.value_mgmt_clarity_score == 8.0
    assert rec.value_mgmt_long_short_ratio == 5.0
    assert rec.value_mgmt_insider_net_usd == 100_000.0
