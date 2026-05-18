"""Screening orchestrator — turns a list of tickers into ScreenedRecords.

Two top-level functions:

    * ``screen_tickers`` — initial pass, no MoS evaluation. Used when a user
      adds firms to the dashboard for the first time. Hard-gates on every
      Phil Town criterion except Margin of Safety.
    * ``refresh_records`` — same pass plus Sticker Price + Margin of Safety,
      with ``mos_percent`` populated as a continuous percentage.

The orchestrator owns the hard-gating policy. The agent itself stays neutral
(soft flags) so the academic backtest's ``is_buy_full`` semantics are
unchanged.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

from anthropic import Anthropic

from value_investing_backend.agents.rule_one.agent import (
    AgentResult,
    RuleOneAgent,
)
from value_investing_backend.agents.rule_one.llm_client import LlmClient
from value_investing_backend.config import get_config
from value_investing_backend.data.edgar import EdgarClient
from value_investing_backend.data.prices import PriceClient
from value_investing_backend.data.transcripts import TranscriptProvider
from value_investing_backend.screening.evidence import (
    build_big_five_evidence,
    build_management_evidence,
)
from value_investing_backend.screening.records import (
    HardGatePolicy,
    ScreenedRecord,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure functions for testability
# --------------------------------------------------------------------------


def _failed_gates(result: AgentResult, policy: HardGatePolicy) -> list[str]:
    """List the gate names that block this firm given the policy."""
    failed: list[str] = []

    quant = result.quant_check_results
    if policy.require_roic and not quant.get("roic", False):
        failed.append("roic")
    if policy.require_sales_growth and not quant.get("sales_growth", False):
        failed.append("sales_growth")
    if policy.require_eps_growth and not quant.get("eps_growth", False):
        failed.append("eps_growth")
    if policy.require_equity_growth and not quant.get("equity_growth", False):
        failed.append("equity_growth")
    if policy.require_ocf_growth and not quant.get("ocf_growth", False):
        failed.append("ocf_growth")
    if policy.require_payback_time and not quant.get("payback_time", False):
        failed.append("payback_time")
    if policy.require_current_ratio and not result.big_five.current_ratio.passes:
        failed.append("current_ratio")

    extras = result.extra_check_results
    if policy.require_debt_payoff and not extras.get("debt_payoff", False):
        failed.append("debt_payoff")
    if policy.require_dilution and not extras.get("dilution", False):
        failed.append("dilution")
    if policy.require_dividend_quality and not extras.get("dividend_quality", False):
        failed.append("dividend_quality")

    llm = result.llm_check_results
    if policy.require_meaning and not llm.get("meaning", False):
        failed.append("meaning")
    if policy.require_moat and not llm.get("moat", False):
        failed.append("moat")
    # Management uses the dedicated ManagementResult aggregator when
    # available so partial 4Ms failures don't poison the gate.
    if policy.require_management:
        mgmt_passes = (
            result.management.passes
            if result.management is not None
            else llm.get("management", False)
        )
        if not mgmt_passes:
            failed.append("management")

    return failed


def _record_from_result(
    *,
    result: AgentResult,
    policy: HardGatePolicy,
    include_mos: bool,
) -> ScreenedRecord:
    b5 = result.big_five
    qx = result.quant_extras
    fm = result.four_ms
    mg = result.management
    sp = result.sticker

    failed = _failed_gates(result, policy)
    screen_passes = not failed

    mos_percent: float | None = None
    if (
        include_mos
        and sp.margin_of_safety_price is not None
        and sp.current_price is not None
        and sp.current_price > 0
    ):
        mos_percent = (
            sp.margin_of_safety_price - sp.current_price
        ) / sp.current_price

    return ScreenedRecord(
        ticker=result.ticker,
        as_of=result.as_of,
        fiscal_year=result.fiscal_year,
        model=fm.model if fm is not None else "",

        check_roic=b5.roic.passes,
        check_sales_growth=b5.sales_growth.passes,
        check_eps_growth=b5.eps_growth.passes,
        check_equity_growth=b5.equity_growth.passes,
        check_ocf_growth=b5.ocf_growth.passes,
        value_roic=b5.roic.value,
        value_sales_growth=b5.sales_growth.value,
        value_eps_growth=b5.eps_growth.value,
        value_equity_growth=b5.equity_growth.value,
        value_ocf_growth=b5.ocf_growth.value,
        value_current_ratio=b5.current_ratio.value,

        check_debt_payoff=qx.debt_payoff.passes if qx is not None else None,
        check_dilution=qx.dilution.passes if qx is not None else None,
        check_dividend_quality=(
            qx.dividend_quality.passes if qx is not None else None
        ),
        value_debt_payoff_years=qx.debt_payoff.value if qx is not None else None,
        value_dilution_cagr=qx.dilution.value if qx is not None else None,
        value_dividend_payout_ratio=(
            qx.dividend_details.payout_ratio if qx is not None else None
        ),
        value_dividend_yield=(
            qx.dividend_details.dividend_yield if qx is not None else None
        ),
        dividend_payout_band=(
            qx.dividend_details.payout_band if qx is not None else None
        ),
        dividend_debt_funded=(
            qx.dividend_details.debt_funded_dividend if qx is not None else None
        ),
        dividend_yield_trap=(
            qx.dividend_details.yield_trap if qx is not None else None
        ),

        check_meaning=fm.meaning.passes if fm is not None else None,
        check_moat=fm.moat.passes if fm is not None else None,
        moat_type=(
            fm.moat.details.get("moat_type") if fm is not None else None
        ),
        rationale_meaning=fm.meaning.rationale if fm is not None else None,
        rationale_moat=fm.moat.rationale if fm is not None else None,

        # Prefer the dedicated ManagementResult aggregator when available
        # (it survives partial 4Ms failures via _safe_eval). Fall back to
        # the four_ms.management slot only if the management pipeline didn't
        # run at all.
        check_management=(
            mg.passes if mg is not None
            else (fm.management.passes if fm is not None else None)
        ),
        check_mgmt_blame=mg.blame.passes if mg is not None else None,
        check_mgmt_long_short=mg.long_short.passes if mg is not None else None,
        check_mgmt_clarity=mg.clarity.passes if mg is not None else None,
        check_mgmt_compensation=mg.compensation.passes if mg is not None else None,
        check_mgmt_insider=mg.insider.passes if mg is not None else None,
        check_mgmt_capital_allocation=(
            mg.capital_allocation.passes
            if mg is not None and mg.capital_allocation is not None
            else None
        ),
        value_mgmt_clarity_score=mg.clarity.score if mg is not None else None,
        value_mgmt_long_short_ratio=(
            mg.long_short.details.get("ratio") if mg is not None else None
        ),
        value_mgmt_insider_net_usd=(
            mg.insider.details.get("net_open_market_value_usd")
            if mg is not None else None
        ),
        value_mgmt_capital_allocation_score=(
            mg.capital_allocation.score
            if mg is not None and mg.capital_allocation is not None
            else None
        ),
        rationale_management=mg.summary() if mg is not None else None,

        screen_passes=screen_passes,
        failed_gates=failed,

        sticker_price=sp.sticker_price if include_mos else None,
        margin_of_safety_price=sp.margin_of_safety_price if include_mos else None,
        current_price=sp.current_price if include_mos else None,
        mos_percent=mos_percent if include_mos else None,

        big_five_evidence=build_big_five_evidence(b5),
        management_evidence=build_management_evidence(
            management=mg, four_ms=fm,
        ),
    )


def _error_record(ticker: str, as_of: date, error: str) -> ScreenedRecord:
    return ScreenedRecord(
        ticker=ticker.upper(), as_of=as_of, fiscal_year=None, model="",
        check_roic=None, check_sales_growth=None, check_eps_growth=None,
        check_equity_growth=None, check_ocf_growth=None,
        value_roic=None, value_sales_growth=None, value_eps_growth=None,
        value_equity_growth=None, value_ocf_growth=None,
        value_current_ratio=None,
        check_debt_payoff=None, check_dilution=None, check_dividend_quality=None,
        value_debt_payoff_years=None, value_dilution_cagr=None,
        value_dividend_payout_ratio=None, value_dividend_yield=None,
        dividend_payout_band=None, dividend_debt_funded=None,
        dividend_yield_trap=None,
        check_meaning=None, check_moat=None, moat_type=None,
        rationale_meaning=None, rationale_moat=None,
        check_management=None, check_mgmt_blame=None,
        check_mgmt_long_short=None, check_mgmt_clarity=None,
        check_mgmt_compensation=None, check_mgmt_insider=None,
        check_mgmt_capital_allocation=None,
        value_mgmt_clarity_score=None, value_mgmt_long_short_ratio=None,
        value_mgmt_insider_net_usd=None,
        value_mgmt_capital_allocation_score=None,
        rationale_management=None,
        screen_passes=False, failed_gates=["evaluation_error"], error=error,
        big_five_evidence=build_big_five_evidence(None),
        management_evidence=build_management_evidence(
            management=None, four_ms=None,
        ),
    )


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


class ScreeningOrchestrator:
    """Reusable orchestrator that holds an agent and a hard-gating policy.

    Useful when a long-running service evaluates many tickers in a row and
    wants to share clients/caches across calls.
    """

    def __init__(
        self,
        *,
        agent: RuleOneAgent | None = None,
        edgar_client: EdgarClient | None = None,
        price_client: PriceClient | None = None,
        anthropic_client: Anthropic | None = None,
        llm_client: LlmClient | None = None,
        transcript_provider: TranscriptProvider | None = None,
        policy: HardGatePolicy | None = None,
    ) -> None:
        self.policy = policy or HardGatePolicy.default()
        if agent is not None:
            self._agent = agent
        else:
            cfg = get_config()
            self._edgar = edgar_client or EdgarClient()
            self._prices = price_client or PriceClient()
            self._llm = llm_client or LlmClient(
                anthropic_client=anthropic_client
                or Anthropic(api_key=cfg.anthropic_api_key)
            )
            self._agent = RuleOneAgent(
                edgar_client=self._edgar, price_client=self._prices,
                llm_client=self._llm, transcript_provider=transcript_provider,
            )

    @property
    def agent(self) -> RuleOneAgent:
        return self._agent

    def screen(
        self,
        tickers: Iterable[str],
        as_of: date,
        *,
        include_mos: bool = False,
    ) -> list[ScreenedRecord]:
        out: list[ScreenedRecord] = []
        for raw in tickers:
            ticker = raw.strip().upper()
            if not ticker:
                continue
            try:
                result = self._agent.evaluate(ticker, as_of)
                out.append(_record_from_result(
                    result=result, policy=self.policy, include_mos=include_mos,
                ))
            except Exception as exc:  # noqa: BLE001
                log.warning("Screening failed for %s @ %s: %s", ticker, as_of, exc)
                out.append(_error_record(ticker, as_of, str(exc)))
        return out


# --------------------------------------------------------------------------
# Top-level convenience entry points
# --------------------------------------------------------------------------


def screen_tickers(
    tickers: Iterable[str],
    as_of: date,
    *,
    orchestrator: ScreeningOrchestrator | None = None,
    policy: HardGatePolicy | None = None,
) -> list[ScreenedRecord]:
    """Initial screen — no Margin of Safety. Used when adding firms to the dashboard.

    A firm passes (``screen_passes=True``) iff every gate enabled in the
    supplied ``policy`` is met. The default policy hard-gates on the Big 5
    + meaning + moat + management + the three Phil Town extras and skips MoS.
    """
    orch = orchestrator or ScreeningOrchestrator(policy=policy)
    if policy is not None and orchestrator is not None:
        # Caller passed both → respect the policy override even on a shared orch.
        orch = ScreeningOrchestrator(agent=orch.agent, policy=policy)
    return orch.screen(tickers, as_of, include_mos=False)


def refresh_records(
    tickers: Iterable[str],
    as_of: date,
    *,
    orchestrator: ScreeningOrchestrator | None = None,
    policy: HardGatePolicy | None = None,
) -> list[ScreenedRecord]:
    """Weekly refresh — same as ``screen_tickers`` plus MoS as a continuous percentage."""
    orch = orchestrator or ScreeningOrchestrator(policy=policy)
    if policy is not None and orchestrator is not None:
        orch = ScreeningOrchestrator(agent=orch.agent, policy=policy)
    return orch.screen(tickers, as_of, include_mos=True)
