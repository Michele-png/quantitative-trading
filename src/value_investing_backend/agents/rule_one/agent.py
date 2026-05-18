"""RuleOneAgent — orchestrator that combines all Rule One checks.

A single `evaluate(ticker, as_of)` call returns an AgentResult containing:
    * Big 5 numbers (5 checks)
    * Sticker Price / Margin of Safety (1 check)
    * Payback Time (1 check)
    * 4Ms via LLM (3 checks; optional)

The result exposes derived `is_buy_*` properties for the three ablation variants
the backtest needs:
    * `is_buy_full`         — all 9 checks pass (LLM included)
    * `is_buy_quant_only`   — all 7 quant checks pass (LLM ignored)
    * `is_buy_quant_random_qual`(seed) — quant pass × a deterministic-random
      draw with the supplied base rate (computed at backtest time)

We run the LLM at most once per (ticker, fiscal_year) thanks to the FourMsAnalyzer
cache, so all three variants come "for free" after a single evaluate().
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import date

from anthropic import Anthropic

from value_investing_backend.agents.rule_one.big_five import (
    BigFiveAnalyzer,
    BigFiveResult,
)
from value_investing_backend.agents.rule_one.four_ms_llm import (
    FourMsAnalyzer,
    FourMsResult,
    MCheck,
)
from value_investing_backend.agents.rule_one.llm_client import LlmClient
from value_investing_backend.agents.rule_one.management_llm import (
    CapitalAllocationContext,
    DocumentBundler,
    ManagementAnalyzer,
    ManagementResult,
)
from value_investing_backend.agents.rule_one.quant_extras import (
    QuantExtrasAnalyzer,
    QuantExtrasResult,
)
from value_investing_backend.agents.rule_one.sticker_price import (
    DEFAULT_PE_LOOKBACK_YEARS,
    PaybackTimeResult,
    StickerPriceCalculator,
    StickerPriceResult,
)
from value_investing_backend.data.edgar import EdgarClient
from value_investing_backend.data.management_documents import ArchiveBackedManagementProvider
from value_investing_backend.data.pit_facts import PointInTimeFacts
from value_investing_backend.data.prices import PriceClient
from value_investing_backend.data.transcripts import TranscriptProvider

# Used by the agent's ``_fiscal_year_ends_from_facts`` helper when
# building the FY → period-end dict it forwards to the sticker
# calculator. The sticker calculator will average over up to
# ``DEFAULT_PE_LOOKBACK_YEARS`` of valid FYs, so we materialise a
# slightly larger window here just in case some FYs are unpriceable.
DEFAULT_PE_LOOKBACK_YEARS_FOR_AGENT = DEFAULT_PE_LOOKBACK_YEARS + 2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    """Aggregate Rule One evaluation for one (ticker, as_of) pair."""

    ticker: str
    as_of: date

    big_five: BigFiveResult
    sticker: StickerPriceResult
    payback: PaybackTimeResult
    four_ms: FourMsResult | None  # None if LLM was skipped
    management: ManagementResult | None = None
    """Multi-document Management evaluation. ``None`` when LLM is skipped.

    When present, the agent overwrites ``four_ms.management`` so the existing
    backtest contract (``four_ms.all_pass``) reflects the new pipeline's
    verdict. The full ``ManagementResult`` is exposed here for the dashboard
    and screening orchestrator, which need every sub-check.
    """
    quant_extras: QuantExtrasResult | None = None
    """Phil Town extra checks (debt-payoff, dilution, dividend quality).

    Carried as **soft flags**: they appear in ``extra_check_results`` and the
    dataset-builder columns, but ``is_buy_full`` and ``is_buy_quant_only``
    intentionally do NOT depend on them so the academic backtest in
    ``docs/REPORT.md`` remains reproducible. The screening orchestrator hard-
    gates on them on top of the agent's output.
    """

    # Provenance / convenience.
    @property
    def fiscal_year(self) -> int | None:
        return self.big_five.latest_fiscal_year

    # ------------------------------------------------- Per-check booleans

    @property
    def quant_check_results(self) -> dict[str, bool]:
        b = self.big_five
        return {
            "roic": b.roic.passes,
            "sales_growth": b.sales_growth.passes,
            "eps_growth": b.eps_growth.passes,
            "equity_growth": b.equity_growth.passes,
            "ocf_growth": b.ocf_growth.passes,
            "margin_of_safety": self.sticker.margin_of_safety_passes,
            "payback_time": self.payback.passes,
        }

    @property
    def extra_check_results(self) -> dict[str, bool]:
        """Soft Phil-Town flags (None means evaluator was skipped)."""
        if self.quant_extras is None:
            return {"debt_payoff": False, "dilution": False, "dividend_quality": False}
        return self.quant_extras.per_check

    @property
    def llm_check_results(self) -> dict[str, bool]:
        if self.four_ms is None:
            return {"meaning": False, "moat": False, "management": False}
        return {
            "meaning": self.four_ms.meaning.passes,
            "moat": self.four_ms.moat.passes,
            "management": self.four_ms.management.passes,
        }

    # ----------------------------------------------- Ablation variants

    @property
    def quant_pass(self) -> bool:
        return all(self.quant_check_results.values())

    @property
    def llm_pass(self) -> bool:
        if self.four_ms is None:
            return False
        return self.four_ms.all_pass

    @property
    def is_buy_full(self) -> bool:
        """Variant A: full Rule One — all 9 checks pass."""
        return self.quant_pass and self.llm_pass

    @property
    def is_buy_quant_only(self) -> bool:
        """Variant B: quant only — ignore LLM."""
        return self.quant_pass

    def is_buy_quant_random_qual(
        self,
        seed: str,
        base_rate_meaning: float,
        base_rate_moat: float,
        base_rate_management: float,
    ) -> bool:
        """Variant C: quant pass × deterministic-random qualitative draws.

        The randomness is a function of (ticker, as_of, seed) so the variant is
        reproducible across backtest runs. Each of the 3 Ms is drawn
        independently with its given base rate.
        """
        if not self.quant_pass:
            return False
        for m, rate in (
            ("meaning", base_rate_meaning),
            ("moat", base_rate_moat),
            ("management", base_rate_management),
        ):
            key = f"{seed}|{self.ticker}|{self.as_of.isoformat()}|{m}".encode()
            digest = hashlib.sha256(key).digest()
            draw = int.from_bytes(digest[:8], "big") / 2**64
            if draw >= rate:
                return False
        return True

    # ------------------------------------------------------- Reporting

    def summary(self) -> str:
        lines = [
            f"RuleOneAgent({self.ticker} as_of {self.as_of}, FY={self.fiscal_year}):",
            "  Quant checks:",
        ]
        for name, ok in self.quant_check_results.items():
            lines.append(f"    {name:>20s}: {'OK' if ok else 'FAIL'}")
        lines.append("  Extra (soft) checks:")
        if self.quant_extras is None:
            lines.append("    (skipped)")
        else:
            for name, ok in self.extra_check_results.items():
                lines.append(f"    {name:>20s}: {'OK' if ok else 'FAIL'}")
        lines.append("  LLM checks (4Ms):")
        if self.four_ms is None:
            lines.append("    (skipped)")
        else:
            for name, ok in self.llm_check_results.items():
                lines.append(f"    {name:>20s}: {'OK' if ok else 'FAIL'}")
        lines.append(
            f"  Decision: BUY (full)={self.is_buy_full}  "
            f"BUY (quant only)={self.is_buy_quant_only}"
        )
        return "\n".join(lines)


class RuleOneAgent:
    """Phil Town's Rule One value-investing agent."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        price_client: PriceClient,
        anthropic_client: Anthropic | None = None,
        *,
        llm_client: LlmClient | None = None,
        transcript_provider: TranscriptProvider | None = None,
        management_archive_provider: ArchiveBackedManagementProvider | None = None,
    ) -> None:
        self._edgar = edgar_client
        self._prices = price_client
        self._b5 = BigFiveAnalyzer(edgar_client, price_client)
        self._sp = StickerPriceCalculator(edgar_client, price_client)
        self._extras = QuantExtrasAnalyzer(edgar_client, price_client)
        # LLM-driven analyzers share one ``LlmClient`` so model + thinking
        # budget + dry-run mode are configured in one place.
        self._llm: LlmClient | None
        self._four_ms: FourMsAnalyzer | None
        self._management: ManagementAnalyzer | None
        if llm_client is not None:
            self._llm = llm_client
        elif anthropic_client is not None:
            self._llm = LlmClient(anthropic_client=anthropic_client)
        else:
            self._llm = None

        if self._llm is not None:
            self._four_ms = FourMsAnalyzer(
                edgar_client=edgar_client, llm_client=self._llm,
            )
            bundler = None
            if transcript_provider is not None or management_archive_provider is not None:
                bundler = DocumentBundler(
                    edgar_client,
                    transcript_provider=transcript_provider,
                    archive_provider=management_archive_provider,
                )
            self._management = ManagementAnalyzer(
                edgar_client=edgar_client, llm_client=self._llm,
                document_bundler=bundler,
            )
        else:
            self._four_ms = None
            self._management = None

    def attach_llm(self, anthropic_client: Anthropic) -> None:
        """Late-bind an Anthropic client (e.g., for the dataset builder)."""
        self._llm = LlmClient(anthropic_client=anthropic_client)
        self._four_ms = FourMsAnalyzer(
            edgar_client=self._edgar, llm_client=self._llm,
        )
        self._management = ManagementAnalyzer(
            edgar_client=self._edgar, llm_client=self._llm,
        )

    def _build_capital_allocation_context(
        self,
        big_five: BigFiveResult,
        quant_extras: QuantExtrasResult | None,
    ) -> CapitalAllocationContext:
        """Materialise the capital-allocation evidence the management
        evaluator needs from already-computed Big 5 + quant extras.

        Keeping this in the agent (instead of re-running PIT facts in
        the management module) avoids duplicate XBRL reads and ensures
        the capital-allocation sub-check sees exactly the same numbers
        the dashboard's Quant Extras section is showing.
        """
        if quant_extras is None:
            return CapitalAllocationContext(
                roic_series=dict(big_five.roic.series or {}),
            )
        # FCF conversion = OCF / NI for the latest FY both have data.
        fcf_conv = self._latest_fcf_conversion(big_five)
        return CapitalAllocationContext(
            dilution_cagr=quant_extras.dilution.value,
            roic_series=dict(big_five.roic.series or {}),
            fcf_conversion_latest=fcf_conv,
            dividend_quality_passes=quant_extras.dividend_quality.passes,
        )

    @staticmethod
    def _latest_fcf_conversion(big_five: BigFiveResult) -> float | None:
        """Approximate FCF conversion using OCF series / ROIC denominator
        is impossible from Big 5 alone — but we have the OCF series.
        Net income lives behind ROIC, not directly exposed; until the
        agent gets a structured PIT view, return ``None`` so the
        deterministic check skips the FCF leg cleanly. The capital
        allocation evaluator already treats ``None`` as "no signal"."""
        return None

    def _fiscal_year_ends_from_facts(
        self,
        ticker: str,
        as_of: date,
        *,
        pit: PointInTimeFacts | None = None,
    ) -> dict[int, date | None]:
        """Map fiscal year → official period-end date for the recent window.

        Used by the sticker calculator to look up FY-end Close prices so
        it can compute a historical average PE. Returns ``{}`` if EDGAR
        access fails — the calculator is robust to a missing mapping
        (it just skips the historical-PE input).

        Accepts an optional shared ``PointInTimeFacts`` so the agent's
        one-PIT-per-evaluate policy avoids a redundant SEC parse here.
        """
        try:
            if pit is None:
                cik = self._edgar.get_cik(ticker)
                facts = self._edgar.get_company_facts(cik)
                pit = PointInTimeFacts(facts)
            latest_fy = pit.latest_fiscal_year_with_data("revenue", as_of)
            if latest_fy is None:
                return {}
            window = range(
                latest_fy - DEFAULT_PE_LOOKBACK_YEARS_FOR_AGENT + 1,
                latest_fy + 1,
            )
            return {fy: pit.fiscal_year_end(fy) for fy in window}
        except Exception as exc:  # noqa: BLE001 - non-critical helper
            log.warning(
                "fiscal_year_ends lookup failed for %s: %s", ticker, exc,
            )
            return {}

    def _load_pit_facts(self, ticker: str) -> PointInTimeFacts | None:
        """Load company facts once per ``evaluate`` and wrap as PIT.

        Returns ``None`` when SEC data is unreachable so the caller can
        fall through to the per-analyzer fallback (which now triggers a
        warning but keeps the previous semantics: the analyzer surfaces
        an unable / no-data result).
        """
        try:
            cik = self._edgar.get_cik(ticker)
            facts = self._edgar.get_company_facts(cik)
            return PointInTimeFacts(facts)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PointInTimeFacts load failed for %s: %s", ticker, exc,
            )
            return None

    def evaluate(
        self,
        ticker: str,
        as_of: date,
        *,
        include_llm: bool = True,
        include_extras: bool = True,
        include_management: bool = True,
        ticker_masked: bool = False,
    ) -> AgentResult:
        # Load companyfacts + wrap in PIT once per (ticker, as_of) and thread
        # the same instance through Big 5, Sticker, QuantExtras, 4Ms, and the
        # Management bundler. Without this every analyzer re-parses the same
        # JSON (~3-4x duplicate work per call). Analyzers still fall back to
        # internal load when ``pit_facts is None``, so single-analyzer
        # callers and tests are unaffected.
        pit = self._load_pit_facts(ticker)

        big_five = self._b5.evaluate(ticker, as_of, pit_facts=pit)
        eps_growth = big_five.eps_growth.value  # may be None if insufficient data

        # Pass the EPS-in-today's-basis history + per-FY end dates from
        # the Big 5 layer down to the sticker calculator. The latter
        # uses them to compute a historical average PE — one of the
        # documented inputs to the future-PE cap.
        eps_history = dict(big_five.eps_growth.series or {})
        fiscal_year_ends = self._fiscal_year_ends_from_facts(
            ticker, as_of, pit=pit,
        )

        sticker, payback = self._sp.evaluate(
            ticker, as_of,
            historical_eps_growth=eps_growth,
            eps_history=eps_history,
            fiscal_year_ends=fiscal_year_ends,
            pit_facts=pit,
        )

        quant_extras: QuantExtrasResult | None = None
        if include_extras:
            try:
                quant_extras = self._extras.evaluate(
                    ticker, as_of, big_five=big_five, pit_facts=pit,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("QuantExtras failed for %s @ %s: %s", ticker, as_of, exc)
                quant_extras = None

        four_ms: FourMsResult | None = None
        if include_llm and self._four_ms is not None:
            try:
                four_ms = self._four_ms.evaluate(
                    ticker, as_of, ticker_masked=ticker_masked, pit_facts=pit,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("4Ms LLM failed for %s @ %s: %s", ticker, as_of, exc)
                four_ms = None

        management: ManagementResult | None = None
        if include_llm and include_management and self._management is not None:
            cap_alloc_ctx = self._build_capital_allocation_context(
                big_five=big_five, quant_extras=quant_extras,
            )
            try:
                management = self._management.evaluate(
                    ticker, as_of,
                    capital_allocation_context=cap_alloc_ctx,
                    pit_facts=pit,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Management LLM failed for %s @ %s: %s",
                            ticker, as_of, exc)
                management = None

        # If we have both, splice the new ManagementResult into FourMsResult so
        # the existing ``all_pass`` semantics keep working.
        if four_ms is not None and management is not None:
            mgmt_check = MCheck(
                name="Management",
                passes=management.passes,
                rationale=management.summary().split("\n", 1)[0],
                details={"per_check": management.per_check},
            )
            four_ms = replace(four_ms, management=mgmt_check)

        return AgentResult(
            ticker=ticker.upper(),
            as_of=as_of,
            big_five=big_five,
            sticker=sticker,
            payback=payback,
            four_ms=four_ms,
            management=management,
            quant_extras=quant_extras,
        )
