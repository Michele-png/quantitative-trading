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
from collections.abc import Callable
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
from quantitative_trading.data.management_documents import (
    ArchiveBackedManagementProvider,
    ArchivedManagementDocument,
    default_management_archive_provider,
)
from quantitative_trading.data.pit_facts import PointInTimeFacts
from quantitative_trading.data.transcripts import (
    DefaultTranscriptProvider,
    EarningsTranscript,
    TranscriptProvider,
)

log = logging.getLogger(__name__)


# v1: original 5-subcheck pipeline.
# v2: adds CapitalAllocation, renames the 10-K Item 1 fallback source.
PROMPT_VERSION = "v2"
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
    # Tracks whether ``shareholder_letter_text`` is a real shareholder
    # letter (from DEF 14A) or the 10-K Item 1 Business Description
    # fallback. Evaluators that consume the slot (LongShort, Clarity)
    # use this to qualify their rationale rather than pretend the
    # business description is a CEO-to-shareholder letter.
    shareholder_letter_is_fallback: bool = False
    source_documents: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Stable hash of the bundle for cache invalidation."""
        h = hashlib.sha256()
        h.update(self.mda_text.encode("utf-8", errors="ignore"))
        h.update(self.def14a_compensation_text.encode("utf-8", errors="ignore"))
        h.update(self.def14a_letter_text.encode("utf-8", errors="ignore"))
        h.update(self.shareholder_letter_text.encode("utf-8", errors="ignore"))
        for t in self.transcripts:
            h.update(f"{t.fiscal_year}Q{t.fiscal_quarter}|{t.source}|{len(t.text)}".encode())
        h.update(json.dumps(self.source_documents, sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class SubCheck:
    """One sub-check inside the Management aggregator."""

    name: str
    passes: bool
    score: float | None  # 1..10 for clarity, 0..1 for ratios; None when n/a
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


def _safe_eval(name: str, fn: Callable[[], SubCheck]) -> SubCheck:
    """Run a sub-check evaluator, swallowing any exception into a fail SubCheck.

    A single LLM failure (e.g., the model writes prose instead of calling
    the tool, a transient Anthropic 429, a malformed response that fails
    JSON extraction) shouldn't zero out the whole ManagementResult. The
    aggregator records the failure on the affected sub-check only and
    keeps the successful sub-checks.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.warning("Management sub-check %s failed: %s", name, exc)
        return SubCheck(
            name=name,
            passes=False,
            score=None,
            rationale=f"Sub-check failed: {type(exc).__name__}: {exc}"[:1000],
            details={"error": True, "exception_type": type(exc).__name__},
        )


@dataclass(frozen=True)
class SourceCoverage:
    """Snapshot of which Management input documents were available for a run.

    Populated by ``ManagementAnalyzer.evaluate`` from the assembled
    ``DocumentBundle`` (and the insider sub-check's outcome). The dashboard
    uses this to distinguish a real "bad management signal" failure from
    "we couldn't evaluate this leg because the source was missing".

    ``shareholder_letter_source`` is one of:
        * ``"def14a"`` — the proxy statement contained an explicit letter.
        * ``"10-k_item1_fallback_business_description"`` — fell back to
          the 10-K Item 1 (Business Description). NOT the same as a
          shareholder letter; the dashboard should label this clearly
          and lower the Clarity / LongShort confidence accordingly.
        * ``None`` — no shareholder letter material was found.
    """

    transcripts_available: bool = False
    transcripts_count: int = 0
    transcripts_expected: int = 0
    def14a_compensation_available: bool = False
    def14a_letter_available: bool = False
    shareholder_letter_available: bool = False
    shareholder_letter_source: str | None = None
    mda_available: bool = False
    form4_available: bool = False
    form4_n_transactions: int | None = None
    source_documents: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapitalAllocationContext:
    """External signals required for the capital-allocation sub-check.

    The agent layer populates these from ``BigFiveResult`` and
    ``QuantExtrasResult`` so the management analyzer doesn't have to
    recompute them. All fields are optional — when missing, the
    deterministic side of the check degrades gracefully and the
    sub-check status reflects the loss of evidence.
    """

    dilution_cagr: float | None = None
    """10-year CAGR of weighted-average diluted shares outstanding.
    Negative = buybacks (good), >2% = active dilution (bad)."""

    roic_series: dict[int, float | None] = field(default_factory=dict)
    """Per-FY ROIC values from ``BigFiveResult.roic.series``. Used to
    classify reinvestment quality (rising / steady high / falling)."""

    fcf_conversion_latest: float | None = None
    """OCF / NI for the latest FY. Sustained <0.70 is a quality
    concern: management is reporting earnings the company can't fully
    convert to cash, which constrains real reinvestment options."""

    dividend_quality_passes: bool | None = None
    """``QuantExtrasResult.dividend_quality.passes`` — surfaces yield
    traps and debt-funded dividends, which are capital-allocation
    failures even when the headline payout looks safe."""


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
    capital_allocation: SubCheck | None = None
    """Capital-allocation sub-check (Phase 5).

    ``None`` when the cached entry pre-dates the introduction of the
    sub-check; new runs always populate it. The aggregate ``passes``
    treats a ``None`` capital allocation as a soft pass — older cache
    entries don't suddenly start failing the gate just because the
    schema gained a slot.
    """

    coverage: SourceCoverage | None = None
    """Which source documents were available for this run.

    ``None`` only when the result was hydrated from a pre-coverage cache
    file (older runs). Fresh runs always populate this so the dashboard
    can distinguish ``no_data`` from ``fail`` per sub-check.
    """

    @property
    def passes(self) -> bool:
        legs = [
            self.blame.passes, self.long_short.passes, self.clarity.passes,
            self.compensation.passes, self.insider.passes,
        ]
        if self.capital_allocation is not None:
            legs.append(self.capital_allocation.passes)
        return all(legs)

    @property
    def per_check(self) -> dict[str, bool]:
        out = {
            "blame": self.blame.passes,
            "long_short": self.long_short.passes,
            "clarity": self.clarity.passes,
            "compensation": self.compensation.passes,
            "insider": self.insider.passes,
        }
        if self.capital_allocation is not None:
            out["capital_allocation"] = self.capital_allocation.passes
        return out

    def summary(self) -> str:
        lines = [f"Management({self.ticker} as_of {self.as_of}, FY={self.fiscal_year}):"]
        subs = [
            self.blame, self.long_short, self.clarity,
            self.compensation, self.insider,
        ]
        if self.capital_allocation is not None:
            subs.append(self.capital_allocation)
        for sub in subs:
            tag = "OK" if sub.passes else "FAIL"
            lines.append(f"  {sub.name:>18s}: {tag}  {sub.rationale[:120]}")
        lines.append(f"  ALL PASS: {self.passes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Document bundler
# --------------------------------------------------------------------------


_LETTER_HEADERS = (
    "letter to shareholders",
    "letter to our shareholders",
    "to our shareholders",
    "to our stockholders",
    "letter to stockholders",
    "letter to our stockholders",
    "chairman's letter",
    "letter from the chairman",
    "letter from our ceo",
    "letter from the ceo",
    "ceo letter",
    "ceo's letter",
    "founder's letter",
    "founders' letter",
)


# CD&A header text varies a lot by issuer. The list below was assembled
# from a sample of large-cap proxies (META, MSFT, GOOGL, AAPL, V, COST,
# NVDA, JNJ, BRK.B, KO) and intentionally biases toward the precise CD&A
# heading rather than the broader "Executive Compensation" section,
# because the latter often pulls in tables and benefit detail that
# crowd out the qualitative discussion the LLM evaluates.
_CDA_HEADERS = (
    "compensation discussion and analysis",
    "compensation discussion & analysis",
    "executive compensation discussion and analysis",
    "executive compensation discussion & analysis",
    "cd&a",
    "compensation philosophy",
    "executive compensation",
    "named executive officer compensation",
    "named executive officers compensation",
)


# Source labels for the management ``shareholder_letter`` slot. We
# previously labelled the 10-K Item 1 fallback as "shareholder letter"
# even though Item 1 is a Business Description — semantically a very
# different document. The new label makes the distinction explicit so
# the dashboard can lower the LongShort / Clarity confidence when a
# fallback was used instead of a real shareholder-facing document.
SHAREHOLDER_LETTER_FROM_DEF14A = "def14a"
SHAREHOLDER_LETTER_FROM_BUSINESS_DESCRIPTION = "10-k_item1_fallback_business_description"
SHAREHOLDER_LETTER_FROM_ARCHIVE = "archive"


def _slice_after_header(text: str, headers: tuple[str, ...], max_chars: int) -> str:
    """Return up to ``max_chars`` of text starting from the first matched header.

    Uses a word-boundary regex so headers like ``cd&a`` aren't matched
    inside unrelated sentences (e.g. a paragraph about NCD-A products).
    Most proxies repeat each header twice — once in the table of
    contents, once at the actual section. We prefer the LAST occurrence
    to skip the TOC and land on the body section, mirroring the same
    trick ``extract_10k_sections`` uses for 10-K item bodies.
    """
    if not text:
        return ""
    best_pos: int | None = None
    for header in headers:
        # Word-boundary regex: ``\b`` for plain headers, lenient for
        # punctuation-bearing ones (``cd&a``).
        if re.search(r"[^a-z0-9&\s]", header, re.IGNORECASE):
            pattern = re.compile(re.escape(header), re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{re.escape(header)}\b", re.IGNORECASE)
        last_match: int | None = None
        for m in pattern.finditer(text):
            last_match = m.start()
        if last_match is None:
            continue
        if best_pos is None or last_match < best_pos:
            best_pos = last_match
    if best_pos is None:
        return ""
    return text[best_pos : best_pos + max_chars]


def _slice_compensation_fallback(text: str, max_chars: int) -> str:
    """Best-effort fallback for issuers whose CD&A header doesn't match.

    Scans for the densest ``compensation``-mention paragraph and slices
    from there. This deliberately runs *after* the named-header pass —
    if the issuer used a recognisable header we want that exact section,
    not whatever scored highest on the density heuristic. When even the
    fallback fails we return ``""`` and the management evaluator emits a
    NO_DATA result rather than confidently scoring noise.
    """
    if not text:
        return ""
    lower = text.lower()
    # Anchor on a full-sentence mention of executive compensation. The
    # ``executive`` qualifier matters: every 10-K mentions
    # "stock-based compensation expense" in the cash-flow statement and
    # we don't want that one.
    anchor_terms = (
        "executive compensation",
        "named executive officer",
        "ceo compensation",
        "compensation philosophy",
    )
    best: int | None = None
    for term in anchor_terms:
        idx = lower.find(term)
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
        archive_provider: ArchiveBackedManagementProvider | None = None,
        transcript_quarters: int = DEFAULT_TRANSCRIPT_QUARTERS,
    ) -> None:
        self._edgar = edgar_client
        self._archive = (
            archive_provider
            if archive_provider is not None
            else default_management_archive_provider()
        )
        self._transcripts = transcript_provider or DefaultTranscriptProvider(
            edgar_client=edgar_client
        )
        self._transcript_quarters = transcript_quarters

    def build(self, ticker: str, as_of: date) -> DocumentBundle:  # noqa: PLR0912, PLR0915
        ticker = ticker.upper()
        cik = self._edgar.get_cik(ticker)
        facts = self._edgar.get_company_facts(cik)
        pit = PointInTimeFacts(facts)
        source_documents: dict[str, Any] = {}

        mda_text = ""
        accession_10k: str | None = None
        fy: int | None = None
        archive_mda = self._latest_archive_doc(ticker, as_of, "ten_k_mda")
        if archive_mda is not None:
            mda_text = archive_mda.text
            accession_10k = f"archive:{archive_mda.content_hash[:12]}"
            fy = archive_mda.fiscal_year
            source_documents["mda"] = _archive_doc_evidence(archive_mda)
        filing = find_pit_10k(pit, self._edgar, cik, as_of)
        if not mda_text and filing is not None:
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
        archive_compensation = (
            self._archive.get_proxy_compensation(ticker, as_of)
            if self._archive is not None
            else None
        )
        if archive_compensation is not None:
            def14a_compensation_text = archive_compensation.text[:DEF14A_CHARS_BUDGET]
            accession_def14a = f"archive:{archive_compensation.content_hash[:12]}"
            source_documents["proxy_compensation"] = _archive_doc_evidence(
                archive_compensation
            )
        archive_proxy_letter = (
            self._archive.get_proxy_letter(ticker, as_of)
            if self._archive is not None
            else None
        )
        if archive_proxy_letter is not None:
            def14a_letter_text = archive_proxy_letter.text[:LETTER_CHARS_BUDGET]
            source_documents["proxy_letter"] = _archive_doc_evidence(archive_proxy_letter)
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
            accession_def14a = accession_def14a or latest.get("accessionNumber")
            try:
                if not def14a_compensation_text or not def14a_letter_text:
                    doc = self._edgar.fetch_filing_document(
                        cik=cik, accession=latest.get("accessionNumber"),
                        primary_document=latest.get("primaryDocument", ""),
                    )
                    full_text = html_to_text(doc)
                    if not def14a_compensation_text:
                        def14a_compensation_text = _slice_after_header(
                            full_text, _CDA_HEADERS, DEF14A_CHARS_BUDGET,
                        )
                        if not def14a_compensation_text:
                            # Header heuristics missed — try a broader scan
                            # anchored on "executive compensation"-style phrases.
                            def14a_compensation_text = _slice_compensation_fallback(
                                full_text, DEF14A_CHARS_BUDGET,
                            )
                            if def14a_compensation_text:
                                log.info(
                                    "DocumentBundler: CD&A header not found for %s "
                                    "DEF 14A %s; using compensation-fallback slice.",
                                    ticker, latest.get("accessionNumber"),
                                )
                    if not def14a_letter_text:
                        def14a_letter_text = _slice_after_header(
                            full_text, _LETTER_HEADERS, LETTER_CHARS_BUDGET,
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "DocumentBundler: DEF 14A fetch failed for %s/%s: %s",
                    ticker, accession_def14a, exc,
                )

        # Shareholder letter: if a dedicated one wasn't found in the
        # proxy, treat the 10-K Item 1 (Business Description) as a
        # *labelled fallback*. Item 1 is regulatory boilerplate about
        # products and competitive dynamics, not a CEO-to-shareholder
        # letter — we still surface it because it can carry useful
        # framing on long-term strategy, but downstream evaluators flag
        # the fallback so the dashboard can show "Clarity scored on
        # business description, not a real shareholder letter".
        archive_letter = (
            self._archive.get_shareholder_letter(ticker, as_of)
            if self._archive is not None
            else None
        )
        shareholder_letter_text = (
            archive_letter.text[:LETTER_CHARS_BUDGET]
            if archive_letter is not None
            else def14a_letter_text
        )
        shareholder_letter_is_fallback = False
        if archive_letter is not None:
            source_documents["shareholder_letter"] = _archive_doc_evidence(archive_letter)
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
                shareholder_letter_is_fallback = bool(shareholder_letter_text)
            except Exception:  # noqa: BLE001
                shareholder_letter_text = ""
                shareholder_letter_is_fallback = False

        # Earnings transcripts: last N quarters before as_of.
        end = as_of
        start = as_of - timedelta(days=self._transcript_quarters * 95)
        archive_transcripts = (
            self._archive.get_transcripts(ticker, start=start, end=end)
            if self._archive is not None
            else []
        )
        try:
            live_transcripts = self._transcripts.get_transcripts(ticker, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            log.warning("DocumentBundler: transcripts failed for %s: %s", ticker, exc)
            live_transcripts = []
        transcripts = _dedupe_transcripts([*archive_transcripts, *live_transcripts])
        if archive_transcripts:
            source_documents["transcripts"] = [
                {
                    "fiscal_year": t.fiscal_year,
                    "fiscal_quarter": t.fiscal_quarter,
                    "source": t.source,
                    "call_date": t.call_date.isoformat() if t.call_date else None,
                    "chars": len(t.text),
                }
                for t in archive_transcripts
            ]
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
            ticker=ticker,
            as_of=as_of,
            fiscal_year=fy,
            accession_10k=accession_10k,
            accession_def14a=accession_def14a,
            mda_text=mda_text[:DEF14A_CHARS_BUDGET],
            def14a_compensation_text=def14a_compensation_text,
            def14a_letter_text=def14a_letter_text,
            shareholder_letter_text=shareholder_letter_text,
            transcripts=transcripts,
            shareholder_letter_is_fallback=shareholder_letter_is_fallback,
            source_documents=source_documents,
        )

    def _latest_archive_doc(
        self,
        ticker: str,
        as_of: date,
        doc_type: str,
    ) -> ArchivedManagementDocument | None:
        if self._archive is None:
            return None
        docs = self._archive.get_documents(
            ticker=ticker,
            as_of=as_of,
            doc_type=doc_type,  # type: ignore[arg-type]
            limit=1,
        )
        return docs[0] if docs else None


def _dedupe_transcripts(transcripts: list[EarningsTranscript]) -> list[EarningsTranscript]:
    """Prefer earlier providers when multiple sources cover the same quarter."""

    seen: set[tuple[int, int]] = set()
    out: list[EarningsTranscript] = []
    for transcript in transcripts:
        key = (transcript.fiscal_year, transcript.fiscal_quarter)
        if key in seen:
            continue
        seen.add(key)
        out.append(transcript)
    out.sort(key=lambda item: (item.fiscal_year, item.fiscal_quarter))
    return out


def _archive_doc_evidence(doc: ArchivedManagementDocument) -> dict[str, Any]:
    """Compact provenance block safe to embed in management_evidence.

    Also surfaces ``manual_review_promoted`` so the dashboard can warn that
    the doc reached ``validated`` only because an operator overrode the
    automated policy via ``etl/promote_management_document.py`` -- which is
    weaker evidence than a doc that passed the policy on its own.
    """

    raw_evidence = doc.evidence if isinstance(doc.evidence, dict) else {}
    manual_review = raw_evidence.get("manual_review")
    promoted_by: str | None = None
    promoted_reason: str | None = None
    if isinstance(manual_review, dict):
        promoted_by = manual_review.get("promoted_by") or None
        promoted_reason = manual_review.get("reason") or None

    return {
        "doc_type": doc.doc_type,
        "provider": doc.provider,
        "source_url": doc.source_url,
        "storage_path": doc.storage_path,
        "content_hash": doc.content_hash,
        "published_date": doc.published_date.isoformat() if doc.published_date else None,
        "fiscal_year": doc.fiscal_year,
        "fiscal_quarter": doc.fiscal_quarter,
        "confidence": doc.confidence,
        "validation_notes": list(doc.validation_notes),
        "chars": len(doc.text),
        "manual_review_promoted": bool(promoted_by),
        "manual_review_promoted_by": promoted_by,
        "manual_review_reason": promoted_reason,
    }


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


_CAPITAL_ALLOC_TOOL = {
    "name": "submit_capital_allocation_assessment",
    "description": (
        "Read management's discussion of capital allocation in the 10-K MD&A and "
        "shareholder letter. Identify the stated priorities, evaluate whether "
        "they are aligned with high-ROIC compounding, and flag any explicit "
        "capital misallocation patterns (empire-building acquisitions, persistent "
        "buybacks at elevated multiples, dividend defended at the expense of "
        "reinvestment, etc.). Phil Town's framing: 'what does management do "
        "with the cash they earn, and is it making us richer per share?'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "stated_priorities": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Ranked list of capital-allocation priorities the filing "
                    "describes (e.g. 'reinvestment in core business', "
                    "'tuck-in acquisitions', 'opportunistic buybacks', "
                    "'progressive dividend')."
                ),
            },
            "discipline_score": {
                "type": "integer", "minimum": 1, "maximum": 10,
                "description": (
                    "1..10 score of capital-allocation discipline. 10 = "
                    "Buffett/Bezos: explicit reinvestment thresholds, "
                    "shareholder-aligned hurdle rates, willingness to return "
                    "cash when reinvestment is unavailable. 1 = empire-builder: "
                    "growth-for-its-own-sake acquisitions, vague language about "
                    "'creating long-term value' without numerical anchors."
                ),
            },
            "capital_misallocation_flags": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Specific concerns: 'M&A at peak multiples', 'buybacks "
                    "above intrinsic value', 'dividend funded by debt', "
                    "'reinvestment at sub-cost-of-capital'. Empty if none."
                ),
            },
            "rationale": {"type": "string", "maxLength": 1000},
        },
        "required": [
            "stated_priorities", "discipline_score", "rationale",
        ],
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


_CAPITAL_ALLOC_SYSTEM = """You are a senior equity analyst applying Phil Town's Rule One framework.
Your single task is the CAPITAL ALLOCATION TEST. Read the supplied 10-K MD&A
and (when present) the shareholder letter. Identify how management deploys
the cash the business produces — reinvestment, M&A, buybacks, dividends.

Score discipline 1..10:
  10 = Buffett / Constellation Software: explicit per-share value framing,
       hurdle-rate language, willingness to defer buybacks above intrinsic
       value, willingness to return cash when reinvestment is unavailable.
   7 = Sensible: stated priorities tied to per-share metrics, modest
       buybacks, ROIC-driven reinvestment thresholds.
   5 = Mixed: priorities listed but no quantitative discipline cues.
   3 = Empire-building: acquisition-driven growth narrative without
       per-share anchors, persistent buybacks at peak valuations.
   1 = Capital destruction: dividend funded by debt, M&A at any price,
       repurchases above intrinsic value through cycle.

Flag concrete capital-misallocation patterns. Be conservative: extract
language that justifies your verdict from the source. Do NOT speculate
about facts not in the supplied text.
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
            # When we're using the 10-K Item 1 fallback, label it
            # accurately so the LLM doesn't treat regulatory product
            # descriptions as if they were a CEO-to-shareholder letter.
            label = (
                "10-K BUSINESS DESCRIPTION (Item 1) — NOT a shareholder letter"
                if bundle.shareholder_letter_is_fallback
                else "SHAREHOLDER LETTER"
            )
            materials_parts.append(
                f"\n\n=== {label} ===\n{bundle.shareholder_letter_text}"
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
        # We pick the source explicitly so the rationale can label which
        # document drove the score (Item 1 fallback ≠ shareholder letter).
        if bundle.shareholder_letter_text and not bundle.shareholder_letter_is_fallback:
            text = bundle.shareholder_letter_text
            doc_label = "shareholder letter"
        elif bundle.mda_text:
            text = bundle.mda_text
            doc_label = "10-K Item 7 (MD&A)"
        elif bundle.shareholder_letter_text:
            # Last-resort: the 10-K Item 1 business-description fallback.
            text = bundle.shareholder_letter_text
            doc_label = "10-K Item 1 business description (fallback)"
        else:
            text = ""
            doc_label = ""
        if not text:
            return SubCheck(
                name="Clarity", passes=False, score=None,
                rationale="No shareholder letter or MD&A text available.",
            )
        body = self._llm.truncate(
            f"Source: {doc_label}\n\n{text}"
        )
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
        rationale = str(payload.get("rationale", ""))[:1000]
        # Surface the source label so the dashboard / cache reflects
        # which document the clarity score is actually about.
        rationale = f"[scored from {doc_label}] {rationale}".strip()
        return SubCheck(
            name="Clarity",
            passes=passes,
            score=float(score),
            rationale=rationale,
            details={
                "scored_from": doc_label,
                "jargon_examples": payload.get("jargon_examples", []),
                "plain_english_examples": payload.get("plain_english_examples", []),
            },
        )


class CapitalAllocationEvaluator:
    """Evaluate Phil Town's "what does management do with the cash?" criterion.

    Combines a deterministic XBRL signal (buyback discipline, reinvestment
    ROIC, FCF conversion, dividend appropriateness) with an LLM read of the
    10-K MD&A + shareholder letter. The deterministic side comes from the
    ``CapitalAllocationContext`` populated by the agent; the LLM side
    extracts stated priorities and discipline cues from the filing text.

    The pass criterion is "AND of both sides": LLM discipline_score >= 7
    with no misallocation flags, AND no deterministic concerns. Either
    side can fail the gate independently.
    """

    LLM_DISCIPLINE_PASS_FLOOR = 7
    DILUTION_HARD_FAIL = 0.02  # >2%/yr → dilution concern
    ROIC_REINVESTMENT_FLOOR = 0.10
    FCF_CONVERSION_CONCERN_FLOOR = 0.70

    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def evaluate(
        self,
        bundle: DocumentBundle,
        context: CapitalAllocationContext | None = None,
    ) -> SubCheck:
        ctx = context or CapitalAllocationContext()
        deterministic_concerns = self._deterministic_concerns(ctx)

        # --- LLM side ---------------------------------------------------
        text_parts: list[str] = []
        if bundle.shareholder_letter_text:
            label = (
                "10-K BUSINESS DESCRIPTION (Item 1) — fallback, not a real letter"
                if bundle.shareholder_letter_is_fallback
                else "SHAREHOLDER LETTER"
            )
            text_parts.append(
                f"\n\n=== {label} ===\n{bundle.shareholder_letter_text}"
            )
        if bundle.mda_text:
            text_parts.append(f"\n\n=== 10-K Item 7 (MD&A) ===\n{bundle.mda_text}")
        if not text_parts:
            return SubCheck(
                name="CapitalAllocation", passes=False, score=None,
                rationale=(
                    "No 10-K MD&A or shareholder-letter text available for "
                    "capital-allocation analysis."
                ),
                details=self._deterministic_details(
                    ctx, deterministic_concerns,
                ),
            )

        body = self._llm.truncate("".join(text_parts))
        result = self._llm.call(
            system_prompt=_CAPITAL_ALLOC_SYSTEM,
            user_prompt=(
                f"Capital allocation analysis for {bundle.ticker} "
                f"(as of {bundle.as_of}).{body}"
            ),
            tool=_CAPITAL_ALLOC_TOOL,
            dry_run_payload={
                "stated_priorities": [], "discipline_score": 5,
                "capital_misallocation_flags": [], "rationale": "[dry-run]",
            },
        )
        payload = result.payload
        discipline = int(payload.get("discipline_score", 0) or 0)
        flags = list(payload.get("capital_misallocation_flags") or [])
        priorities = list(payload.get("stated_priorities") or [])

        llm_pass = discipline >= self.LLM_DISCIPLINE_PASS_FLOOR and not flags
        passes = llm_pass and not deterministic_concerns

        details = self._deterministic_details(ctx, deterministic_concerns)
        details.update({
            "stated_priorities": priorities,
            "discipline_score": discipline,
            "capital_misallocation_flags": flags,
            "llm_pass": llm_pass,
        })

        rationale_parts: list[str] = []
        if ctx.dilution_cagr is not None:
            rationale_parts.append(f"dilution {ctx.dilution_cagr * 100:.1f}%/yr")
        if details.get("reinvestment_roic_avg") is not None:
            rationale_parts.append(
                f"avg ROIC {details['reinvestment_roic_avg'] * 100:.1f}%"
            )
        if ctx.fcf_conversion_latest is not None:
            rationale_parts.append(
                f"FCF conv {ctx.fcf_conversion_latest:.2f}"
            )
        rationale_parts.append(f"LLM discipline {discipline}/10")
        if flags:
            rationale_parts.append("LLM flags: " + ", ".join(flags))
        if deterministic_concerns:
            rationale_parts.append(
                "deterministic concerns: " + ", ".join(deterministic_concerns)
            )
        rationale = "; ".join(rationale_parts)
        # Append the LLM rationale (truncated) as additional context.
        llm_text = str(payload.get("rationale", "")).strip()
        if llm_text:
            rationale = f"{rationale} — {llm_text}"
        rationale = rationale[:1000]

        return SubCheck(
            name="CapitalAllocation",
            passes=passes,
            score=float(discipline),
            rationale=rationale,
            details=details,
        )

    # ----------------------------------------------------------- helpers

    def _deterministic_concerns(
        self, ctx: CapitalAllocationContext,
    ) -> list[str]:
        out: list[str] = []
        if ctx.dilution_cagr is not None and ctx.dilution_cagr > self.DILUTION_HARD_FAIL:
            out.append(f"dilution {ctx.dilution_cagr * 100:.1f}%/yr > 2% threshold")
        roic_avg, _ = self._roic_aggregates(ctx.roic_series)
        if roic_avg is not None and roic_avg < self.ROIC_REINVESTMENT_FLOOR:
            out.append(f"avg ROIC {roic_avg * 100:.1f}% below 10% reinvestment floor")
        if (
            ctx.fcf_conversion_latest is not None
            and ctx.fcf_conversion_latest < self.FCF_CONVERSION_CONCERN_FLOOR
        ):
            out.append(
                f"FCF conversion {ctx.fcf_conversion_latest:.2f} below "
                f"{self.FCF_CONVERSION_CONCERN_FLOOR:.2f} floor"
            )
        if ctx.dividend_quality_passes is False:
            out.append("dividend quality fails (see Quant Extras)")
        return out

    @staticmethod
    def _roic_aggregates(
        series: dict[int, float | None],
    ) -> tuple[float | None, float | None]:
        valid = [v for v in series.values() if v is not None]
        if not valid:
            return None, None
        avg = sum(valid) / len(valid)
        latest_fy = max(series)
        recent = series.get(latest_fy)
        return avg, recent

    def _deterministic_details(
        self,
        ctx: CapitalAllocationContext,
        concerns: list[str],
    ) -> dict[str, Any]:
        roic_avg, roic_recent = self._roic_aggregates(ctx.roic_series)
        if ctx.dilution_cagr is None:
            buyback_band = "unknown"
        elif ctx.dilution_cagr <= 0:
            buyback_band = "buybacks"
        elif ctx.dilution_cagr <= self.DILUTION_HARD_FAIL:
            buyback_band = "flat"
        else:
            buyback_band = "dilution"
        return {
            "buyback_discipline": buyback_band,
            "dilution_cagr": ctx.dilution_cagr,
            "reinvestment_roic_avg": roic_avg,
            "reinvestment_roic_recent": roic_recent,
            "fcf_conversion_latest": ctx.fcf_conversion_latest,
            "dividend_quality_passes": ctx.dividend_quality_passes,
            "deterministic_concerns": concerns,
        }


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


def _encode_coverage(coverage: SourceCoverage | None) -> dict[str, Any] | None:
    if coverage is None:
        return None
    return {
        "transcripts_available": coverage.transcripts_available,
        "transcripts_count": coverage.transcripts_count,
        "transcripts_expected": coverage.transcripts_expected,
        "def14a_compensation_available": coverage.def14a_compensation_available,
        "def14a_letter_available": coverage.def14a_letter_available,
        "shareholder_letter_available": coverage.shareholder_letter_available,
        "shareholder_letter_source": coverage.shareholder_letter_source,
        "mda_available": coverage.mda_available,
        "form4_available": coverage.form4_available,
        "form4_n_transactions": coverage.form4_n_transactions,
        "source_documents": coverage.source_documents,
    }


def _decode_coverage(payload: dict[str, Any] | None) -> SourceCoverage | None:
    """Hydrate ``SourceCoverage`` from a cache payload.

    Returns ``None`` for cache files written before coverage tracking
    existed so the dashboard can render an "unknown coverage" state until
    the next refresh repopulates the entry.
    """
    if not payload or not isinstance(payload, dict):
        return None
    return SourceCoverage(
        transcripts_available=bool(payload.get("transcripts_available", False)),
        transcripts_count=int(payload.get("transcripts_count", 0) or 0),
        transcripts_expected=int(payload.get("transcripts_expected", 0) or 0),
        def14a_compensation_available=bool(
            payload.get("def14a_compensation_available", False)
        ),
        def14a_letter_available=bool(payload.get("def14a_letter_available", False)),
        shareholder_letter_available=bool(
            payload.get("shareholder_letter_available", False)
        ),
        shareholder_letter_source=payload.get("shareholder_letter_source"),
        mda_available=bool(payload.get("mda_available", False)),
        form4_available=bool(payload.get("form4_available", False)),
        form4_n_transactions=payload.get("form4_n_transactions"),
        source_documents=payload.get("source_documents") or {},
    )


def _build_coverage(
    *, bundle: DocumentBundle, expected_transcripts: int, insider: SubCheck,
) -> SourceCoverage:
    """Capture which source documents were available for this evaluation.

    Insider/Form 4 availability is derived from the insider sub-check's
    ``details``: a successful Form 4 fetch leaves ``n_transactions`` in
    details (see ``InsiderAlignmentEvaluator``), while a fetch failure
    leaves only an error rationale.
    """
    if "shareholder_letter" in bundle.source_documents:
        source_payload = bundle.source_documents["shareholder_letter"]
        archived_type = (
            source_payload.get("doc_type")
            if isinstance(source_payload, dict)
            else "unknown"
        )
        letter_source = f"{SHAREHOLDER_LETTER_FROM_ARCHIVE}:{archived_type}"
    elif bundle.def14a_letter_text:
        letter_source = SHAREHOLDER_LETTER_FROM_DEF14A
    elif bundle.shareholder_letter_text:
        # NB: the fallback is the 10-K Item 1 *Business Description*.
        # That's a regulatory disclosure about products and competitive
        # dynamics, not a CEO-to-shareholder letter — see the source
        # constants up top for the rationale on the rename.
        letter_source = SHAREHOLDER_LETTER_FROM_BUSINESS_DESCRIPTION
    else:
        letter_source = None

    insider_details = insider.details or {}
    insider_errored = bool(insider_details.get("error"))
    n_txns = insider_details.get("n_transactions")
    form4_available = (not insider_errored) and isinstance(n_txns, int)

    return SourceCoverage(
        transcripts_available=len(bundle.transcripts) > 0,
        transcripts_count=len(bundle.transcripts),
        transcripts_expected=expected_transcripts,
        def14a_compensation_available=bool(bundle.def14a_compensation_text),
        def14a_letter_available=bool(bundle.def14a_letter_text),
        shareholder_letter_available=bool(bundle.shareholder_letter_text),
        shareholder_letter_source=letter_source,
        mda_available=bool(bundle.mda_text),
        form4_available=form4_available,
        form4_n_transactions=n_txns if isinstance(n_txns, int) else None,
        source_documents=bundle.source_documents,
    )


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
        self._capital_allocation = CapitalAllocationEvaluator(llm_client)
        self._insider = InsiderAlignmentEvaluator(edgar_client)
        self._cache_dir = cache_dir or (cfg.llm_cache_dir / "management")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        ticker: str,
        as_of: date,
        *,
        capital_allocation_context: CapitalAllocationContext | None = None,
    ) -> ManagementResult:
        bundle = self._bundler.build(ticker, as_of)
        bundle_hash = bundle.hash()
        # ``DocumentBundler`` stores its transcript-quarter target on
        # ``_transcript_quarters``; fall back to the module default so the
        # dashboard still gets a sensible "expected" count if the attribute
        # is ever renamed.
        expected_transcripts = int(
            getattr(self._bundler, "_transcript_quarters",
                    DEFAULT_TRANSCRIPT_QUARTERS)
        )
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
                insider = _decode_subcheck(cached["insider"], "Insider")
                # Coverage was not always cached. When loading a pre-coverage
                # entry, derive it from the freshly-built bundle. The bundle
                # content matches the cached entry by definition (same
                # ``bundle_hash``), so the SourceCoverage we compute now is
                # identical to what the original run would have produced.
                # Backfills the dashboard's ``source_coverage`` block + the
                # PARTIAL_DATA / NO_DATA per-subcheck status without paying
                # for any new LLM calls.
                coverage = _decode_coverage(cached.get("coverage"))
                if coverage is None:
                    coverage = _build_coverage(
                        bundle=bundle,
                        expected_transcripts=expected_transcripts,
                        insider=insider,
                    )
                cap_alloc_payload = cached.get("capital_allocation")
                cap_alloc = (
                    _decode_subcheck(cap_alloc_payload, "CapitalAllocation")
                    if cap_alloc_payload is not None
                    else None
                )
                return ManagementResult(
                    ticker=ticker.upper(), as_of=as_of,
                    fiscal_year=bundle.fiscal_year,
                    bundle_hash=bundle_hash,
                    model=self._llm.model, cached=True,
                    blame=_decode_subcheck(cached["blame"], "Blame"),
                    long_short=_decode_subcheck(cached["long_short"], "LongShort"),
                    clarity=_decode_subcheck(cached["clarity"], "Clarity"),
                    compensation=_decode_subcheck(cached["compensation"], "Compensation"),
                    insider=insider,
                    capital_allocation=cap_alloc,
                    coverage=coverage,
                )
            except Exception as exc:  # noqa: BLE001 - cache miss on parse failure
                log.warning("Management cache read failed for %s: %s", ticker, exc)

        # Each sub-check is wrapped in try/except so a single LLM failure
        # (e.g., the model writing prose instead of calling the tool, an
        # Anthropic 429, a malformed response) doesn't zero out the entire
        # ManagementResult. The failed sub-check becomes a deterministic
        # ``passes=False`` SubCheck with the exception in its rationale, and
        # the dashboard surfaces that as a partial-pass row rather than NULL.
        insider = _safe_eval("Insider", lambda: self._insider.evaluate(ticker, as_of))
        blame = _safe_eval("Blame", lambda: self._blame.evaluate(bundle))
        long_short = _safe_eval("LongShort", lambda: self._long_short.evaluate(bundle))
        clarity = _safe_eval("Clarity", lambda: self._clarity.evaluate(bundle))
        compensation = _safe_eval("Compensation", lambda: self._compensation.evaluate(bundle))
        capital_allocation = _safe_eval(
            "CapitalAllocation",
            lambda: self._capital_allocation.evaluate(
                bundle, capital_allocation_context,
            ),
        )

        coverage = _build_coverage(
            bundle=bundle,
            expected_transcripts=expected_transcripts,
            insider=insider,
        )

        result = ManagementResult(
            ticker=ticker.upper(), as_of=as_of,
            fiscal_year=bundle.fiscal_year,
            bundle_hash=bundle_hash,
            model=self._llm.model, cached=False,
            blame=blame, long_short=long_short, clarity=clarity,
            compensation=compensation, insider=insider,
            capital_allocation=capital_allocation,
            coverage=coverage,
        )
        if not self._llm.dry_run:
            cache_path.write_text(json.dumps({
                "blame": _encode_subcheck(blame),
                "long_short": _encode_subcheck(long_short),
                "clarity": _encode_subcheck(clarity),
                "compensation": _encode_subcheck(compensation),
                "insider": _encode_subcheck(insider),
                "capital_allocation": _encode_subcheck(capital_allocation),
                "coverage": _encode_coverage(coverage),
            }, indent=2))
        return result
