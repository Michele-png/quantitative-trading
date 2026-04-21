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
from dataclasses import dataclass
from datetime import date

from anthropic import Anthropic

from quantitative_trading.agents.rule_one.big_five import (
    BigFiveAnalyzer,
    BigFiveResult,
)
from quantitative_trading.agents.rule_one.four_ms_llm import (
    FourMsAnalyzer,
    FourMsResult,
)
from quantitative_trading.agents.rule_one.sticker_price import (
    PaybackTimeResult,
    StickerPriceCalculator,
    StickerPriceResult,
)
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.prices import PriceClient


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
    ) -> None:
        self._edgar = edgar_client
        self._prices = price_client
        self._b5 = BigFiveAnalyzer(edgar_client, price_client)
        self._sp = StickerPriceCalculator(edgar_client, price_client)
        # FourMsAnalyzer is created lazily so quant-only callers don't pay
        # the import cost / require a key.
        self._four_ms: FourMsAnalyzer | None = (
            FourMsAnalyzer(edgar_client=edgar_client, anthropic_client=anthropic_client)
            if anthropic_client is not None
            else None
        )

    def attach_llm(self, anthropic_client: Anthropic) -> None:
        """Late-bind an Anthropic client (e.g., for the dataset builder)."""
        self._four_ms = FourMsAnalyzer(
            edgar_client=self._edgar, anthropic_client=anthropic_client
        )

    def evaluate(
        self,
        ticker: str,
        as_of: date,
        *,
        include_llm: bool = True,
        ticker_masked: bool = False,
    ) -> AgentResult:
        big_five = self._b5.evaluate(ticker, as_of)
        eps_growth = big_five.eps_growth.value  # may be None if insufficient data
        sticker, payback = self._sp.evaluate(
            ticker, as_of, historical_eps_growth=eps_growth
        )

        four_ms: FourMsResult | None = None
        if include_llm and self._four_ms is not None:
            try:
                four_ms = self._four_ms.evaluate(
                    ticker, as_of, ticker_masked=ticker_masked
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("4Ms LLM failed for %s @ %s: %s", ticker, as_of, exc)
                four_ms = None

        return AgentResult(
            ticker=ticker.upper(),
            as_of=as_of,
            big_five=big_five,
            sticker=sticker,
            payback=payback,
            four_ms=four_ms,
        )
