"""Multi-document, multi-prompt evaluator for Phil Town's Management M.

Replaces the single-prompt Management check from ``four_ms_llm.py`` with a
pipeline that scores Phil Town's specific psychological / linguistic markers:

    1. **Blame Test** (transcripts) — Does the CEO take personal
       responsibility for misses, or scapegoat macro / weather / FX?
    2. **Long-Term vs Short-Term** (transcripts + shareholder letter) —
       Does language emphasize quarterly beats and stock price, or decade-
       long vision and intrinsic value?
    3. **Clarity vs Jargon** (shareholder letter) — Plain English, or
       corporate jargon designed to obscure?
    4. **Compensation Alignment** (DEF 14A CD&A) — Are CEO incentives tied
       to per-share value (ROIC, EPS, FCF/share), or to revenue / adjusted
       EBITDA, which encourage empire-building?
    5. **Insider Alignment** (Form 4) — Net open-market buying by named
       officers and directors over the last 24 months, no large coordinated
       sells in the last 90 days. Deterministic — no LLM call.

Aggregate ``ManagementResult.passes`` is the AND of all five. The dashboard
also surfaces each sub-check individually so partial failures are visible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from quantitative_trading.agents.rule_one.four_ms_llm import (
    extract_10k_sections,
    find_pit_10k,
    html_to_text,
)
from quantitative_trading.agents.rule_one.llm_client import LlmClient
from quantitative_trading.config import get_config
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.insider_trades import (
    InsiderAlignmentResult,
    fetch_insider_history,
    summarize_insider_alignment,
)
from quantitative_trading.data.pit_facts import PointInTimeFacts
from quantitative_trading.data.transcripts import (
    DefaultTranscriptProvider,
    EarningsTranscript,
    TranscriptProvider,
)


log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"
DEFAULT_TRANSCRIPT_QUARTERS = 8
DEF14A_CHARS_BUDGET = 200_000   # CD&A sections can be long; budget ~50k tokens.
TRANSCRIPT_CHARS_BUDGET = 60_000  # Per transcript; ~15k tokens each.
LETTER_CHARS_BUDGET = 80_000


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentBundle:
    """The corpus assembled for one (ticker, fiscal_year) management evaluation."""

    ticker: str
    as_of: date
    fiscal_year: int | None
    accession_10k: str | None
    accession_def14a: str | None
    mda_text: str
    def14a_compensation_text: str
    def14a_letter_text: str
    shareholder_letter_text: str  # explicit shareholder letter if found, else ""
    transcripts: list[EarningsTranscript] = field(default_factory=list)

    def hash(self) -> str:
        """Stable hash of the bundle for cache invalidation."""
        h = hashlib.sha256()
        h.update(self.mda_text.encode("utf-8", errors="ignore"))
        h.update(self.def14a_compensation_text.encode("utf-8", errors="ignore"))
        h.update(self.def14a_letter_text.encode("utf-8", errors="ignore"))
        h.update(self.shareholder_letter_text.encode("utf-8", errors="ignore"))
        for t in self.transcripts:
            h.update(f"{t.fiscal_year}Q{t.fiscal_quarter}|{t.source}|{len(t.text)}".encode())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class SubCheck:
    """One sub-check inside the Management aggregator."""

    name: str
    passes: bool
    score: float | None  # 1..10 for clarity, 0..1 for ratios; None when n/a
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagementResult:
    """Aggregate Management evaluation."""

    ticker: str
    as_of: date
    fiscal_year: int | None
    bundle_hash: str
    model: str
    cached: bool

    blame: SubCheck
    long_short: SubCheck
    clarity: SubCheck
    compensation: SubCheck
    insider: SubCheck

    @property
    def passes(self) -> bool:
        return all(
            (self.blame.passes, self.long_short.passes, self.clarity.passes,
             self.compensation.passes, self.insider.passes)
        )

    @property
    def per_check(self) -> dict[str, bool]:
        return {
            "blame": self.blame.passes,
            "long_short": self.long_short.passes,
            "clarity": self.clarity.passes,
            "compensation": self.compensation.passes,
            "insider": self.insider.passes,
        }

    def summary(self) -> str:
        lines = [f"Management({self.ticker} as_of {self.as_of}, FY={self.fiscal_year}):"]
        for sub in (self.blame, self.long_short, self.clarity,
                    self.compensation, self.insider):
            tag = "OK" if sub.passes else "FAIL"
            lines.append(f"  {sub.name:>14s}: {tag}  {sub.rationale[:120]}")
        lines.append(f"  ALL FIVE PASS: {self.passes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Document bundler
# --------------------------------------------------------------------------


_LETTER_HEADERS = (
    "letter to shareholders",
    "letter to our shareholders",
    "to our shareholders",
    "chairman's letter",
    "letter from the chairman",
    "letter from our ceo",
    "ceo letter",
)


_CDA_HEADERS = (
    "compensation discussion and analysis",
    "executive compensation",
    "cd&a",
)


def _slice_after_header(text: str, headers: tuple[str, ...], max_chars: int) -> str:
    """Return up to ``max_chars`` of text starting from the first matched header."""
    lower = text.lower()
    best: int | None = None
    for header in headers:
        idx = lower.find(header)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    if best is None:
        return ""
    return text[best : best + max_chars]


class DocumentBundler:
    """Assemble a ``DocumentBundle`` for one (ticker, as_of) evaluation."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        transcript_provider: TranscriptProvider | None = None,
        *,
        transcript_quarters: int = DEFAULT_TRANSCRIPT_QUARTERS,
    ) -> None:
        self._edgar = edgar_client
        self._transcripts = transcript_provider or DefaultTranscriptProvider(
            edgar_client=edgar_client
        )
        self._transcript_quarters = transcript_quarters

    def build(self, ticker: str, as_of: date) -> DocumentBundle:
        cik = self._edgar.get_cik(ticker)
        facts = self._edgar.get_company_facts(cik)
        pit = PointInTimeFacts(facts)

        mda_text = ""
        accession_10k: str | None = None
        fy: int | None = None
        filing = find_pit_10k(pit, self._edgar, cik, as_of)
        if filing is not None:
            accession_10k = filing.get("accessionNumber")
            fy = filing.get("fiscal_year")
            try:
                doc_html = self._edgar.fetch_filing_document(
                    cik=cik, accession=accession_10k,
                    primary_document=filing["primaryDocument"],
                )
                text = html_to_text(doc_html)
                sections = extract_10k_sections(text)
                mda_text = sections.get("item_7_mda", "") or text[:LETTER_CHARS_BUDGET]
            except Exception as exc:  # noqa: BLE001
                log.warning("DocumentBundler: 10-K fetch failed for %s: %s", ticker, exc)

        # DEF 14A — proxy statement: contains compensation discussion + sometimes
        # a "Letter to Shareholders" up front. Pick the most-recent one filed
        # before as_of.
        def14a_compensation_text = ""
        def14a_letter_text = ""
        accession_def14a: str | None = None
        try:
            def14a_filings = self._edgar.list_filings(
                cik, forms=("DEF 14A", "DEFA14A"),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("DocumentBundler: DEF 14A list failed for %s: %s", ticker, exc)
            def14a_filings = []

        def14a_pre_as_of: list[dict[str, Any]] = []
        for f in def14a_filings:
            try:
                fd = date.fromisoformat(str(f.get("filingDate", "")))
            except ValueError:
                continue
            if fd <= as_of:
                def14a_pre_as_of.append({**f, "_filing_date": fd})
        if def14a_pre_as_of:
            latest = max(def14a_pre_as_of, key=lambda f: f["_filing_date"])
            accession_def14a = latest.get("accessionNumber")
            try:
                doc = self._edgar.fetch_filing_document(
                    cik=cik, accession=accession_def14a,
                    primary_document=latest.get("primaryDocument", ""),
                )
                full_text = html_to_text(doc)
                def14a_compensation_text = _slice_after_header(
                    full_text, _CDA_HEADERS, DEF14A_CHARS_BUDGET,
                )
                def14a_letter_text = _slice_after_header(
                    full_text, _LETTER_HEADERS, LETTER_CHARS_BUDGET,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "DocumentBundler: DEF 14A fetch failed for %s/%s: %s",
                    ticker, accession_def14a, exc,
                )

        # Shareholder letter: if a dedicated one wasn't found in the proxy,
        # treat the 10-K Item 1 (Business) as a fallback — many companies
        # include their "vision" framing there.
        shareholder_letter_text = def14a_letter_text
        if not shareholder_letter_text and filing is not None:
            try:
                doc_html = self._edgar.fetch_filing_document(
                    cik=cik, accession=accession_10k,
                    primary_document=filing["primaryDocument"],
                )
                sections = extract_10k_sections(html_to_text(doc_html))
                shareholder_letter_text = sections.get(
                    "item_1_business", ""
                )[:LETTER_CHARS_BUDGET]
            except Exception:  # noqa: BLE001
                shareholder_letter_text = ""

        # Earnings transcripts: last N quarters before as_of.
        end = as_of
        start = as_of - timedelta(days=self._transcript_quarters * 95)
        try:
            transcripts = self._transcripts.get_transcripts(ticker, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            log.warning("DocumentBundler: transcripts failed for %s: %s", ticker, exc)
            transcripts = []
        # Cap each transcript to the per-doc budget.
        transcripts = [
            EarningsTranscript(
                ticker=t.ticker, fiscal_year=t.fiscal_year,
                fiscal_quarter=t.fiscal_quarter, call_date=t.call_date,
                source=t.source, text=t.text[:TRANSCRIPT_CHARS_BUDGET],
            )
            for t in transcripts[-self._transcript_quarters:]
        ]

        return DocumentBundle(
            ticker=ticker.upper(),
            as_of=as_of,
            fiscal_year=fy,
            accession_10k=accession_10k,
            accession_def14a=accession_def14a,
            mda_text=mda_text[:DEF14A_CHARS_BUDGET],
            def14a_compensation_text=def14a_compensation_text,
            def14a_letter_text=def14a_letter_text,
            shareholder_letter_text=shareholder_letter_text,
            transcripts=transcripts,
        )


# --------------------------------------------------------------------------
# Tool schemas — one per evaluator
# --------------------------------------------------------------------------


_BLAME_TOOL = {
    "name": "submit_blame_assessment",
    "description": (
        "Score the CEO's accountability across the supplied earnings call transcripts. "
        "When discussing missed targets or revenue drops, does the CEO take personal "
        "responsibility, or do they blame macro conditions, weather, FX, or supply chains?"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "takes_responsibility": {"type": "boolean"},
            "scapegoat_count": {
                "type": "integer",
                "description": "Distinct moments where blame was externalized rather than owned.",
            },
            "supporting_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific quotes from the transcripts that justify the verdict.",
            },
            "rationale": {"type": "string", "maxLength": 1000},
        },
        "required": ["takes_responsibility", "scapegoat_count", "rationale"],
        "additionalProperties": False,
    },
}


_LONG_SHORT_TOOL = {
    "name": "submit_horizon_assessment",
    "description": (
        "Count and analyze the relative frequency of short-term focus (e.g. 'beating "
        "quarterly estimates', 'stock price') vs long-term focus (e.g. 'decade-long "
        "vision', 'customer obsession', 'intrinsic value') in the supplied materials."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "short_term_mentions": {"type": "integer"},
            "long_term_mentions": {"type": "integer"},
            "ratio": {
                "type": "number",
                "description": "long_term_mentions / max(short_term_mentions, 1).",
            },
            "dominant_orientation": {
                "type": "string",
                "enum": ["long", "short", "mixed"],
            },
            "rationale": {"type": "string", "maxLength": 1000},
        },
        "required": [
            "short_term_mentions", "long_term_mentions",
            "dominant_orientation", "rationale",
        ],
        "additionalProperties": False,
    },
}


_CLARITY_TOOL = {
    "name": "submit_clarity_assessment",
    "description": (
        "Evaluate the CEO's annual letter / management discussion. Is it written in "
        "plain, understandable English, or filled with corporate jargon and complex "
        "accounting terms designed to obscure reality?"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "clarity_score": {
                "type": "integer", "minimum": 1, "maximum": 10,
                "description": "1 = pure jargon, 10 = exemplary plain English.",
            },
            "jargon_examples": {
                "type": "array", "items": {"type": "string"},
                "description": "Up to 5 jargon-laden phrases pulled from the source.",
            },
            "plain_english_examples": {
                "type": "array", "items": {"type": "string"},
                "description": "Up to 5 phrases that demonstrate clear writing.",
            },
            "rationale": {"type": "string", "maxLength": 1000},
        },
        "required": ["clarity_score", "rationale"],
        "additionalProperties": False,
    },
}


_COMP_TOOL = {
    "name": "submit_compensation_assessment",
    "description": (
        "Extract the metrics that drive CEO bonus and equity vesting from the DEF 14A "
        "Compensation Discussion and Analysis. Determine whether the metrics align "
        "the CEO with per-share shareholder value (ROIC, EPS, FCF/share) or instead "
        "reward empire-building (revenue growth, adjusted EBITDA, total assets)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array", "items": {"type": "string"},
                "description": "Each performance metric used in CEO comp.",
            },
            "shareholder_aligned_metrics": {
                "type": "array", "items": {"type": "string"},
                "description": "Subset that ties to per-share value.",
            },
            "empire_building_metrics": {
                "type": "array", "items": {"type": "string"},
                "description": "Subset that rewards size without per-share discipline.",
            },
            "aligned_with_shareholders": {"type": "boolean"},
            "rationale": {"type": "string", "maxLength": 1000},
        },
        "required": ["metrics", "aligned_with_shareholders", "rationale"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------


_BLAME_SYSTEM = """You are a senior equity analyst applying Phil Town's Rule One framework.
Your single task is the BLAME TEST: when the CEO discusses missed targets or
revenue drops in earnings calls, do they take PERSONAL responsibility, or do
they BLAME external factors (macro headwinds, weather, supply chain, FX)?

PASS examples: "We made a mistake in inventory forecasting." "I underestimated
how long the integration would take." "That was on me."
FAIL examples: "Macro headwinds and currency fluctuations impacted our bottom
line." "Weather affected store traffic." "Supply chain disruption hurt margins."

Be conservative. Use ONLY the supplied transcripts. Cite specific quotes.
"""


_LONG_SHORT_SYSTEM = """You are a senior equity analyst applying Phil Town's Rule One framework.
Your single task is the HORIZON TEST: count and analyze the relative frequency
of SHORT-TERM language (beating quarterly estimates, stock price, EPS guidance)
versus LONG-TERM language (decade-long vision, customer obsession, intrinsic
value, durable moat).

A long-term-oriented CEO talks about strategy spanning years; a short-term-
oriented CEO obsesses over the next print. Be conservative — when in doubt,
classify as "mixed" or "short". Use ONLY the supplied materials.
"""


_CLARITY_SYSTEM = """You are a senior equity analyst applying Phil Town's Rule One framework.
Your single task is the CLARITY TEST: rate from 1 to 10 how clearly the
CEO's annual letter / MD&A is written. Plain English with concrete numbers
and intuitive metaphors scores HIGH. Corporate jargon, defined-terms-laundering,
acronym soups, and "synergistic optimization" prose scores LOW.

Anchor scores:
  10 = Buffett / Bezos letter quality
  7  = Honest, readable, mostly plain
  5  = Mixed — readable in places, jargon in others
  3  = Jargon-heavy with little useful information
  1  = Essentially impossible to extract substance from

Be conservative. Use ONLY the supplied text.
"""


_COMP_SYSTEM = """You are a senior equity analyst applying Phil Town's Rule One framework.
Your single task: read the Compensation Discussion and Analysis and determine
which performance metrics drive the CEO's bonus and equity vesting.

CLASSIFY each metric into:
  * shareholder_aligned: ROIC, return on invested capital, EPS, free cash flow
    per share, total shareholder return, intrinsic value per share
  * empire_building: revenue growth, total revenue, adjusted EBITDA, gross
    profit dollars, total assets, sales bookings — anything that rewards
    growth in absolute size without per-share discipline

PASS criterion: at least one shareholder_aligned metric is present AND it is
NOT swamped by empire-building metrics. If the only metrics are revenue and
adjusted EBITDA, FAIL. Be conservative. Use ONLY the supplied CD&A text.
"""


def _format_transcripts(transcripts: list[EarningsTranscript]) -> str:
    parts: list[str] = []
    for t in transcripts:
        header = (
            f"=== {t.ticker} {t.fiscal_year}Q{t.fiscal_quarter} "
            f"({t.call_date.isoformat() if t.call_date else 'date unknown'}) "
            f"[source: {t.source}] ==="
        )
        parts.append(f"\n\n{header}\n{t.text}")
    return "".join(parts)


class BlameEvaluator:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def evaluate(self, bundle: DocumentBundle) -> SubCheck:
        if not bundle.transcripts:
            return SubCheck(
                name="Blame", passes=False, score=None,
                rationale="No earnings call transcripts available.",
            )
        body = _format_transcripts(bundle.transcripts)
        body = self._llm.truncate(body)
        result = self._llm.call(
            system_prompt=_BLAME_SYSTEM,
            user_prompt=(
                f"Evaluate {bundle.ticker} (as of {bundle.as_of}). "
                f"Transcripts:\n{body}"
            ),
            tool=_BLAME_TOOL,
            dry_run_payload={
                "takes_responsibility": False, "scapegoat_count": 0,
                "rationale": "[dry-run]",
            },
        )
        payload = result.payload
        takes_resp = bool(payload.get("takes_responsibility", False))
        scapegoats = int(payload.get("scapegoat_count", 0) or 0)
        passes = takes_resp and scapegoats <= 2
        return SubCheck(
            name="Blame",
            passes=passes,
            score=float(scapegoats),
            rationale=str(payload.get("rationale", ""))[:1000],
            details={
                "takes_responsibility": takes_resp,
                "scapegoat_count": scapegoats,
                "supporting_quotes": payload.get("supporting_quotes", []),
            },
        )


class LongShortEvaluator:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def evaluate(self, bundle: DocumentBundle) -> SubCheck:
        materials_parts: list[str] = []
        if bundle.shareholder_letter_text:
            materials_parts.append(
                f"\n\n=== SHAREHOLDER LETTER ===\n{bundle.shareholder_letter_text}"
            )
        if bundle.transcripts:
            materials_parts.append(_format_transcripts(bundle.transcripts))
        if not materials_parts:
            return SubCheck(
                name="LongShort", passes=False, score=None,
                rationale="No shareholder letter or transcripts available.",
            )
        body = self._llm.truncate("".join(materials_parts))
        result = self._llm.call(
            system_prompt=_LONG_SHORT_SYSTEM,
            user_prompt=(
                f"Evaluate {bundle.ticker} (as of {bundle.as_of}). "
                f"Materials:\n{body}"
            ),
            tool=_LONG_SHORT_TOOL,
            dry_run_payload={
                "short_term_mentions": 0, "long_term_mentions": 0,
                "dominant_orientation": "short", "rationale": "[dry-run]",
            },
        )
        payload = result.payload
        orientation = str(payload.get("dominant_orientation", "short")).lower()
        long_n = int(payload.get("long_term_mentions", 0) or 0)
        short_n = int(payload.get("short_term_mentions", 0) or 0)
        ratio = (
            payload.get("ratio")
            if payload.get("ratio") is not None
            else long_n / max(short_n, 1)
        )
        passes = orientation == "long"
        return SubCheck(
            name="LongShort",
            passes=passes,
            score=float(ratio) if ratio is not None else None,
            rationale=str(payload.get("rationale", ""))[:1000],
            details={
                "short_term_mentions": short_n,
                "long_term_mentions": long_n,
                "ratio": ratio,
                "dominant_orientation": orientation,
            },
        )


class ClarityEvaluator:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def evaluate(self, bundle: DocumentBundle) -> SubCheck:
        # Prefer the explicit shareholder letter; fall back to MD&A if absent.
        text = bundle.shareholder_letter_text or bundle.mda_text
        if not text:
            return SubCheck(
                name="Clarity", passes=False, score=None,
                rationale="No shareholder letter or MD&A text available.",
            )
        body = self._llm.truncate(text)
        result = self._llm.call(
            system_prompt=_CLARITY_SYSTEM,
            user_prompt=(
                f"Score clarity for {bundle.ticker} (as of {bundle.as_of}).\n\n{body}"
            ),
            tool=_CLARITY_TOOL,
            dry_run_payload={"clarity_score": 5, "rationale": "[dry-run]"},
        )
        payload = result.payload
        score = int(payload.get("clarity_score", 0) or 0)
        passes = score >= 7
        return SubCheck(
            name="Clarity",
            passes=passes,
            score=float(score),
            rationale=str(payload.get("rationale", ""))[:1000],
            details={
                "jargon_examples": payload.get("jargon_examples", []),
                "plain_english_examples": payload.get("plain_english_examples", []),
            },
        )


class CompensationEvaluator:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def evaluate(self, bundle: DocumentBundle) -> SubCheck:
        if not bundle.def14a_compensation_text:
            return SubCheck(
                name="Compensation", passes=False, score=None,
                rationale="No DEF 14A Compensation Discussion & Analysis available.",
            )
        body = self._llm.truncate(bundle.def14a_compensation_text)
        result = self._llm.call(
            system_prompt=_COMP_SYSTEM,
            user_prompt=(
                f"Compensation analysis for {bundle.ticker} (as of {bundle.as_of}).\n\n"
                f"{body}"
            ),
            tool=_COMP_TOOL,
            dry_run_payload={
                "metrics": [], "aligned_with_shareholders": False,
                "rationale": "[dry-run]",
            },
        )
        payload = result.payload
        aligned = bool(payload.get("aligned_with_shareholders", False))
        return SubCheck(
            name="Compensation",
            passes=aligned,
            score=None,
            rationale=str(payload.get("rationale", ""))[:1000],
            details={
                "metrics": payload.get("metrics", []),
                "shareholder_aligned_metrics": payload.get(
                    "shareholder_aligned_metrics", []
                ),
                "empire_building_metrics": payload.get(
                    "empire_building_metrics", []
                ),
            },
        )


class InsiderAlignmentEvaluator:
    """Deterministic: wraps ``summarize_insider_alignment`` (no LLM call)."""

    def __init__(self, edgar_client: EdgarClient) -> None:
        self._edgar = edgar_client

    def evaluate(self, ticker: str, as_of: date) -> SubCheck:
        try:
            cik = self._edgar.get_cik(ticker)
        except KeyError:
            return SubCheck(
                name="Insider", passes=False, score=None,
                rationale=f"No CIK for {ticker}.",
            )
        try:
            txns = fetch_insider_history(
                self._edgar, cik=cik,
                start=as_of - timedelta(days=24 * 30),
                end=as_of,
            )
        except Exception as exc:  # noqa: BLE001
            return SubCheck(
                name="Insider", passes=False, score=None,
                rationale=f"Form 4 fetch failed: {exc}",
            )
        result: InsiderAlignmentResult = summarize_insider_alignment(
            txns, as_of=as_of,
        )
        return SubCheck(
            name="Insider",
            passes=result.passes,
            score=result.net_open_market_value_usd,
            rationale=result.rationale,
            details={
                "open_market_buy_value_usd": result.open_market_buy_value_usd,
                "open_market_sell_value_usd": result.open_market_sell_value_usd,
                "net_open_market_value_usd": result.net_open_market_value_usd,
                "coordinated_buy": result.coordinated_buy,
                "coordinated_buy_count": result.coordinated_buy_count,
                "has_large_recent_sells": result.has_large_recent_sells,
                "by_role_net_usd": result.by_role_net_usd,
                "n_transactions": result.n_transactions,
            },
        )


# --------------------------------------------------------------------------
# Aggregator + cache
# --------------------------------------------------------------------------


def _cache_key(
    *,
    ticker: str,
    fiscal_year: int | None,
    bundle_hash: str,
    model: str,
    thinking_budget: int,
    prompt_version: str,
) -> str:
    raw = (
        f"{ticker.upper()}|FY={fiscal_year}|bundle={bundle_hash}|"
        f"model={model}|think={thinking_budget}|prompt={prompt_version}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24] + ".json"


def _decode_subcheck(payload: dict[str, Any], name: str) -> SubCheck:
    return SubCheck(
        name=name,
        passes=bool(payload.get("passes", False)),
        score=payload.get("score"),
        rationale=str(payload.get("rationale", "")),
        details=payload.get("details") or {},
    )


def _encode_subcheck(sub: SubCheck) -> dict[str, Any]:
    return {
        "name": sub.name,
        "passes": sub.passes,
        "score": sub.score,
        "rationale": sub.rationale,
        "details": sub.details,
    }


class ManagementAnalyzer:
    """End-to-end Management evaluator with disk caching."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        llm_client: LlmClient,
        document_bundler: DocumentBundler | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        cfg = get_config()
        self._edgar = edgar_client
        self._llm = llm_client
        self._bundler = document_bundler or DocumentBundler(edgar_client)
        self._blame = BlameEvaluator(llm_client)
        self._long_short = LongShortEvaluator(llm_client)
        self._clarity = ClarityEvaluator(llm_client)
        self._compensation = CompensationEvaluator(llm_client)
        self._insider = InsiderAlignmentEvaluator(edgar_client)
        self._cache_dir = cache_dir or (cfg.llm_cache_dir / "management")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, ticker: str, as_of: date) -> ManagementResult:
        bundle = self._bundler.build(ticker, as_of)
        bundle_hash = bundle.hash()
        cache_path = self._cache_dir / _cache_key(
            ticker=ticker, fiscal_year=bundle.fiscal_year,
            bundle_hash=bundle_hash,
            model=self._llm.model,
            thinking_budget=self._llm.thinking_budget_tokens,
            prompt_version=PROMPT_VERSION,
        )
        if cache_path.exists() and not self._llm.dry_run:
            try:
                cached = json.loads(cache_path.read_text())
                return ManagementResult(
                    ticker=ticker.upper(), as_of=as_of,
                    fiscal_year=bundle.fiscal_year,
                    bundle_hash=bundle_hash,
                    model=self._llm.model, cached=True,
                    blame=_decode_subcheck(cached["blame"], "Blame"),
                    long_short=_decode_subcheck(cached["long_short"], "LongShort"),
                    clarity=_decode_subcheck(cached["clarity"], "Clarity"),
                    compensation=_decode_subcheck(cached["compensation"], "Compensation"),
                    insider=_decode_subcheck(cached["insider"], "Insider"),
                )
            except Exception as exc:  # noqa: BLE001 - cache miss on parse failure
                log.warning("Management cache read failed for %s: %s", ticker, exc)

        # Insider check is cheap and runs first so we can short-circuit obvious
        # fails on bandwidth-constrained runs (we still run all so the dashboard
        # has full data, but logging the insider result first helps debugging).
        insider = self._insider.evaluate(ticker, as_of)
        blame = self._blame.evaluate(bundle)
        long_short = self._long_short.evaluate(bundle)
        clarity = self._clarity.evaluate(bundle)
        compensation = self._compensation.evaluate(bundle)

        result = ManagementResult(
            ticker=ticker.upper(), as_of=as_of,
            fiscal_year=bundle.fiscal_year,
            bundle_hash=bundle_hash,
            model=self._llm.model, cached=False,
            blame=blame, long_short=long_short, clarity=clarity,
            compensation=compensation, insider=insider,
        )
        if not self._llm.dry_run:
            cache_path.write_text(json.dumps({
                "blame": _encode_subcheck(blame),
                "long_short": _encode_subcheck(long_short),
                "clarity": _encode_subcheck(clarity),
                "compensation": _encode_subcheck(compensation),
                "insider": _encode_subcheck(insider),
            }, indent=2))
        return result
