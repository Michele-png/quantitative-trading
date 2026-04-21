"""LLM-driven analysis of Phil Town's 4 Ms (Meaning, Moat, Management).

The 4th M (Margin of Safety) is computed deterministically in
`sticker_price.py`; this module handles the qualitative three.

For a (ticker, fiscal_year), we:
    1. Locate the most recent 10-K filed before `as_of` (PIT-correct).
    2. Fetch and clean the filing's primary document text.
    3. Extract Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A) when
       parseable; fall back to a truncated full document otherwise.
    4. Send to Claude with a structured tool schema asking for pass/fail +
       rationale on each M.
    5. Cache the structured response on disk keyed by
       (ticker, fiscal_year, prompt_version, model). Re-running with the same
       inputs costs zero additional API calls.

Cost considerations:
    * Default model: claude-sonnet-4-5 (~$3 in / $15 out per MTok).
    * 10-K text is truncated to `MAX_INPUT_CHARS` chars (default 80k ≈ 20k
      tokens). Typical cost per call: ~$0.06–0.12.
    * Caching makes the *full backtest* cheap — only fresh (ticker, FY) pairs
      hit the API.

Contamination caveat:
    The LLM has seen the future. We can't completely solve this, but we can
    estimate the contamination effect by re-running with `ticker_masked=True`,
    which scrubs the ticker symbol and company name from the prompt. This
    forces the model to reason from the filing text alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import warnings

from anthropic import Anthropic
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# SEC filings are sometimes XBRL/XML. We deliberately use the HTML parser to
# get plain text out — the XML warning is noise in this context.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from quantitative_trading.config import get_config
from quantitative_trading.data.edgar import EdgarClient
from quantitative_trading.data.pit_facts import PointInTimeFacts


log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"
MAX_INPUT_CHARS = 80_000  # ~20k tokens for the 10-K excerpt
MAX_OUTPUT_TOKENS = 1_500


# ---------------------------------------------------------------- Data types


@dataclass(frozen=True)
class MCheck:
    """One of the qualitative Ms: passes + rationale (+ optional details)."""

    name: str  # "Meaning" | "Moat" | "Management"
    passes: bool
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FourMsResult:
    """Aggregate result of the 3 LLM-driven Ms (4th = Margin of Safety, separate)."""

    ticker: str
    as_of: date
    fiscal_year: int | None
    accession: str | None
    model: str

    meaning: MCheck
    moat: MCheck
    management: MCheck

    cached: bool
    raw_response: dict[str, Any]

    @property
    def all_pass(self) -> bool:
        return self.meaning.passes and self.moat.passes and self.management.passes

    def summary(self) -> str:
        lines = [
            f"FourMs({self.ticker} as_of {self.as_of}, FY={self.fiscal_year}, "
            f"model={self.model}, cached={self.cached}):"
        ]
        for m in (self.meaning, self.moat, self.management):
            tag = "OK" if m.passes else "FAIL"
            lines.append(f"  {m.name:>12s}: {tag}  {m.rationale[:140]}")
        lines.append(f"  ALL THREE PASS: {self.all_pass}")
        return "\n".join(lines)


# ----------------------------------------------------------------- 10-K text


def html_to_text(html: str) -> str:
    """Strip HTML to plain text, collapsing whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t\n]+", "\n", text)
    return text.strip()


_ITEM_PATTERNS: dict[str, re.Pattern[str]] = {
    "item_1": re.compile(r"\bItem\s*1\.?\s+Business\b", re.IGNORECASE),
    "item_1a": re.compile(r"\bItem\s*1A\.?\s+Risk\s+Factors\b", re.IGNORECASE),
    "item_2": re.compile(r"\bItem\s*2\.?\s+Properties\b", re.IGNORECASE),
    "item_7": re.compile(
        r"\bItem\s*7\.?\s+Management.{0,5}s?\s+Discussion", re.IGNORECASE
    ),
    "item_7a": re.compile(
        r"\bItem\s*7A\.?\s+Quantitative\s+and\s+Qualitative", re.IGNORECASE
    ),
}


def extract_10k_sections(text: str) -> dict[str, str]:
    """Extract Item 1, 1A, and 7 sections from a 10-K plaintext.

    Most 10-Ks list these items twice — once in the table of contents, once in
    the actual body. We use the *last* occurrence of each item start, paired
    with the next item's start, so we always grab the body section. Sections
    that can't be located are returned as empty strings.
    """
    found: dict[str, list[int]] = {}
    for key, pat in _ITEM_PATTERNS.items():
        positions = [m.start() for m in pat.finditer(text)]
        if positions:
            found[key] = positions

    def slice_section(start_key: str, end_keys: list[str]) -> str:
        if start_key not in found:
            return ""
        start = found[start_key][-1]  # last occurrence (skip TOC)
        end_pos: int | None = None
        for ek in end_keys:
            if ek in found:
                ends_after_start = [p for p in found[ek] if p > start]
                if ends_after_start:
                    candidate = ends_after_start[0]
                    end_pos = candidate if end_pos is None else min(end_pos, candidate)
        section = text[start:end_pos] if end_pos else text[start:]
        return section.strip()

    return {
        "item_1_business": slice_section("item_1", ["item_1a", "item_2"]),
        "item_1a_risk_factors": slice_section("item_1a", ["item_2", "item_7"]),
        "item_7_mda": slice_section("item_7", ["item_7a"]),
    }


def find_pit_10k(
    pit: PointInTimeFacts,
    edgar: EdgarClient,
    cik: int,
    as_of: date,
) -> dict[str, Any] | None:
    """Find the most recent 10-K filed at or before `as_of` and return its filing meta.

    Looks at the submissions index (more reliable than facts for filing
    metadata). Returns dict with accessionNumber, filingDate, reportDate,
    primaryDocument, fy (fiscal_year), or None if no 10-K is visible.
    """
    filings = edgar.list_filings(cik, forms=("10-K", "10-K/A"))
    candidates: list[dict[str, Any]] = []
    for f in filings:
        try:
            fd = date.fromisoformat(f["filingDate"])
        except Exception:
            continue
        if fd > as_of:
            continue
        candidates.append({**f, "_filing_date": fd})
    if not candidates:
        return None
    latest = max(candidates, key=lambda f: f["_filing_date"])
    # Find fiscal year by matching reportDate against pit.fiscal_year_end mappings.
    report_date_str = latest.get("reportDate", "")
    fy = None
    if report_date_str:
        report_date = date.fromisoformat(report_date_str)
        for candidate_fy in range(report_date.year + 1, report_date.year - 2, -1):
            if pit.fiscal_year_end(candidate_fy) == report_date:
                fy = candidate_fy
                break
        if fy is None:
            fy = report_date.year
    latest["fiscal_year"] = fy
    return latest


# ------------------------------------------------------------------- Prompts


_SYSTEM_PROMPT = """You are a senior equity analyst applying Phil Town's Rule One value-investing framework.

You evaluate a company on three qualitative pillars (the "first three Ms"):

1. MEANING
   - The company's business is understandable to a competent investor.
   - The product/service has durable demand (people will still want it in 10+ years).
   - The business is in your "circle of competence" (mainstream, not exotic).
   - Pass criteria: clear understandable business + durable demand + not a fad.

2. MOAT (durable competitive advantage)
   - Identify which of Phil Town's 5 moat types applies, if any:
     * BRAND moat — customers pay more because of brand (Coke, Apple).
     * SECRETS moat — patents, proprietary tech, trade secrets.
     * TOLL moat — regulatory or natural monopoly (utilities, exchanges).
     * SWITCHING moat — high cost for customers to switch (enterprise software).
     * PRICE moat — durable cost advantage (Walmart, Costco).
   - Pass criteria: at least one identifiable moat type that is durable for 10+ years.

3. MANAGEMENT
   - CEO and leadership are trustworthy, candid, and focused on long-term shareholder value.
   - Capital allocation is rational (reinvestment, buybacks, dividends in sensible mix).
   - No major red flags: ongoing litigation, fraud allegations, executive churn,
     overcompensation relative to performance, related-party transactions, accounting concerns.
   - Pass criteria: visible signs of competent, owner-aligned management; no red flags.

You MUST be conservative. When in doubt, FAIL the check. Most companies do NOT pass all three.
Use ONLY information from the supplied 10-K text. Do not use information you have about the
company's later performance, news, or events that occurred after the filing date.

Provide a brief 1-3 sentence rationale per check, citing specific language from the filing
when possible."""


def _build_user_prompt(
    *,
    ticker: str,
    fiscal_year: int,
    filing_date: date,
    sections: dict[str, str],
    ticker_masked: bool,
    truncate_to: int,
) -> str:
    if ticker_masked:
        identifier_block = (
            f"FILING DATE: {filing_date}\n"
            f"FISCAL YEAR: FY{fiscal_year}\n"
            f"COMPANY: [redacted — please reason from the filing text alone]\n"
        )
    else:
        identifier_block = (
            f"TICKER: {ticker}\n"
            f"FILING DATE: {filing_date}\n"
            f"FISCAL YEAR: FY{fiscal_year}\n"
        )

    body_parts: list[str] = []
    for label, key in [
        ("ITEM 1 — BUSINESS", "item_1_business"),
        ("ITEM 1A — RISK FACTORS", "item_1a_risk_factors"),
        ("ITEM 7 — MD&A", "item_7_mda"),
    ]:
        section = sections.get(key, "").strip()
        if section:
            body_parts.append(f"\n\n=== {label} ===\n{section}")
    if not body_parts:
        body_parts.append("\n\n=== FILING TEXT (could not parse sections) ===\n")
    full = "".join(body_parts)
    if len(full) > truncate_to:
        full = full[:truncate_to] + "\n\n[... truncated for length ...]"

    return (
        f"Apply the Rule One Meaning / Moat / Management checks to the following 10-K filing.\n\n"
        f"{identifier_block}\n"
        f"--- 10-K excerpt begins ---{full}\n--- 10-K excerpt ends ---"
    )


# Tool schema for structured output (Anthropic's tool calling for guaranteed JSON).
_TOOL_DEF: dict[str, Any] = {
    "name": "submit_four_ms_assessment",
    "description": (
        "Submit your Rule One Meaning / Moat / Management assessment for the company "
        "based ONLY on the supplied 10-K text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "meaning": {
                "type": "object",
                "properties": {
                    "passes": {"type": "boolean"},
                    "rationale": {"type": "string", "maxLength": 600},
                },
                "required": ["passes", "rationale"],
                "additionalProperties": False,
            },
            "moat": {
                "type": "object",
                "properties": {
                    "passes": {"type": "boolean"},
                    "moat_type": {
                        "type": "string",
                        "enum": ["brand", "secrets", "toll", "switching", "price", "none"],
                    },
                    "rationale": {"type": "string", "maxLength": 600},
                },
                "required": ["passes", "moat_type", "rationale"],
                "additionalProperties": False,
            },
            "management": {
                "type": "object",
                "properties": {
                    "passes": {"type": "boolean"},
                    "rationale": {"type": "string", "maxLength": 600},
                    "red_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific red flags identified, if any",
                    },
                },
                "required": ["passes", "rationale"],
                "additionalProperties": False,
            },
        },
        "required": ["meaning", "moat", "management"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------- Cache I/O


def _cache_key(
    *,
    ticker: str,
    fiscal_year: int | None,
    accession: str | None,
    model: str,
    ticker_masked: bool,
    prompt_version: str,
) -> str:
    raw = (
        f"{ticker.upper()}|FY={fiscal_year}|accn={accession}|"
        f"model={model}|masked={int(ticker_masked)}|prompt={prompt_version}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24] + ".json"


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Best-effort coercion of an LLM-returned field to a dict.

    Claude's structured-output occasionally returns a nested object as a
    JSON-encoded string (sometimes with trailing-brace junk). We try, in order:
        1. If already a dict, return as-is.
        2. Strict JSON parse.
        3. Use raw_decode (tolerates trailing junk after a valid object).
        4. Regex-extract `passes` and `rationale` fields.
        5. Fallback: treat the whole string as a free-text rationale, fail-safe.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Tolerate trailing junk (e.g., extra "}" the LLM sometimes appends).
    try:
        parsed, _idx = json.JSONDecoder().raw_decode(value)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Last-resort regex extraction for the two essential fields.
    out: dict[str, Any] = {"passes": False}
    m_pass = re.search(r'"passes"\s*:\s*(true|false)', value, re.IGNORECASE)
    if m_pass:
        out["passes"] = m_pass.group(1).lower() == "true"
    m_rat = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', value)
    if m_rat:
        out["rationale"] = m_rat.group(1).encode().decode("unicode_escape")
    else:
        out["rationale"] = value
    return out


# ----------------------------------------------------------------- Analyzer


class FourMsAnalyzer:
    """LLM-driven Meaning / Moat / Management analyzer with disk caching."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        anthropic_client: Anthropic | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        cfg = get_config()
        self._edgar = edgar_client
        self._client = anthropic_client or Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.anthropic_model
        self._cache_dir = cache_dir or cfg.llm_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        ticker: str,
        as_of: date,
        *,
        ticker_masked: bool = False,
        max_input_chars: int = MAX_INPUT_CHARS,
    ) -> FourMsResult:
        cik = self._edgar.get_cik(ticker)
        facts = self._edgar.get_company_facts(cik)
        pit = PointInTimeFacts(facts)

        filing = find_pit_10k(pit, self._edgar, cik, as_of)
        if filing is None:
            return self._unable(
                ticker, as_of, fiscal_year=None, accession=None,
                reason="No 10-K filed before as_of.",
            )

        accession = filing["accessionNumber"]
        fy = filing["fiscal_year"]

        cache_path = self._cache_dir / _cache_key(
            ticker=ticker, fiscal_year=fy, accession=accession,
            model=self._model, ticker_masked=ticker_masked,
            prompt_version=PROMPT_VERSION,
        )
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return self._result_from_payload(
                ticker, as_of, fy, accession, cached, cached_flag=True,
            )

        try:
            doc_html = self._edgar.fetch_filing_document(
                cik=cik, accession=accession,
                primary_document=filing["primaryDocument"],
            )
        except Exception as exc:
            log.warning("Could not fetch filing document for %s: %s", ticker, exc)
            return self._unable(
                ticker, as_of, fiscal_year=fy, accession=accession,
                reason=f"Failed to fetch 10-K document: {exc}",
            )

        text = html_to_text(doc_html)
        sections = extract_10k_sections(text)
        if not any(sections.values()):
            sections = {"item_1_business": text}  # fallback: dump full text

        prompt = _build_user_prompt(
            ticker=ticker, fiscal_year=fy or 0,
            filing_date=date.fromisoformat(filing["filingDate"]),
            sections=sections,
            ticker_masked=ticker_masked,
            truncate_to=max_input_chars,
        )

        payload = self._call_llm(prompt)
        cache_path.write_text(json.dumps(payload, indent=2))
        return self._result_from_payload(
            ticker, as_of, fy, accession, payload, cached_flag=False,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, user_prompt: str) -> dict[str, Any]:
        log.info("Calling Anthropic %s for 4Ms analysis", self._model)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_DEF],
            tool_choice={"type": "tool", "name": _TOOL_DEF["name"]},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError(
            "Anthropic returned no tool_use block; response was: "
            f"{[b.type for b in response.content]}"
        )

    # ------------------------------------------------------------- Helpers

    def _result_from_payload(
        self,
        ticker: str,
        as_of: date,
        fy: int | None,
        accession: str | None,
        payload: dict[str, Any],
        *,
        cached_flag: bool,
    ) -> FourMsResult:
        meaning = _coerce_to_dict(payload.get("meaning"))
        moat = _coerce_to_dict(payload.get("moat"))
        mgmt = _coerce_to_dict(payload.get("management"))
        return FourMsResult(
            ticker=ticker.upper(),
            as_of=as_of,
            fiscal_year=fy,
            accession=accession,
            model=self._model,
            meaning=MCheck(
                name="Meaning",
                passes=bool(meaning.get("passes", False)),
                rationale=str(meaning.get("rationale", "")),
            ),
            moat=MCheck(
                name="Moat",
                passes=bool(moat.get("passes", False)),
                rationale=str(moat.get("rationale", "")),
                details={"moat_type": moat.get("moat_type", "none")},
            ),
            management=MCheck(
                name="Management",
                passes=bool(mgmt.get("passes", False)),
                rationale=str(mgmt.get("rationale", "")),
                details={"red_flags": mgmt.get("red_flags", [])},
            ),
            cached=cached_flag,
            raw_response=payload,
        )

    def _unable(
        self,
        ticker: str,
        as_of: date,
        *,
        fiscal_year: int | None,
        accession: str | None,
        reason: str,
    ) -> FourMsResult:
        empty = MCheck(name="", passes=False, rationale=reason)
        return FourMsResult(
            ticker=ticker.upper(), as_of=as_of, fiscal_year=fiscal_year,
            accession=accession, model=self._model,
            meaning=MCheck("Meaning", False, reason),
            moat=MCheck("Moat", False, reason),
            management=MCheck("Management", False, reason),
            cached=False, raw_response={},
        )
