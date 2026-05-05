"""Zero-shot LLM decision layer for the weekly paper trader."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from anthropic import Anthropic

from quantitative_trading.paper_trading.models import MarketContext, WeeklyDecision


class DecisionParseError(ValueError):
    """Raised when the LLM response is not valid decision JSON."""


SYSTEM_PROMPT = """You are a zero-shot weekly paper-trading decision agent.
You must choose exactly one stock from the provided candidate list.
You are not receiving examples and must not infer hidden examples.
Optimize expected account value over the next 8 weeks while respecting:
- S&P 500 long-only universe.
- One stock is bought this Monday and sold next Monday.
- The account is paper-only.
- The output must be valid JSON only.
"""


def build_user_prompt(context: MarketContext) -> str:
    """Build a zero-shot prompt with context and schema, but no examples."""
    payload = {
        "task": (
            "Choose exactly one ticker for a one-week long paper trade. "
            "Use only the supplied current-week market, news, and candidate data."
        ),
        "constraints": {
            "universe": "S&P 500 candidates supplied below",
            "holding_period": "buy Monday, sell next Monday",
            "positioning": "long-only, exactly one ticker",
            "prompting_mode": "zero-shot; no examples are provided or allowed",
        },
        "required_json_schema": {
            "ticker": "uppercase ticker from candidates",
            "confidence": "number between 0 and 1",
            "thesis": "concise investment thesis for the next week",
            "risks": "array of concise risk strings",
            "market_summary": "brief summary of market conditions used",
        },
        "context": context.to_prompt_dict(),
    }
    return json.dumps(payload, sort_keys=True)


class ZeroShotDecisionMaker:
    """Ask Anthropic for one structured weekly paper-trading decision."""

    def __init__(
        self,
        *,
        anthropic_client: Anthropic,
        model: str,
        max_output_tokens: int = 1_000,
    ) -> None:
        self._client = anthropic_client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def decide(self, context: MarketContext) -> WeeklyDecision:
        """Make one zero-shot decision for the supplied market context."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(context)}],
        )
        text = _extract_text(response)
        parsed = parse_decision_json(text)
        return decision_from_payload(
            payload=parsed,
            trade_week=context.trade_week,
            model=self.model,
            raw_response={"text": text},
        )


def parse_decision_json(text: str) -> dict[str, Any]:
    """Parse strict JSON from the LLM response, tolerating accidental code fences."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DecisionParseError(f"LLM response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecisionParseError("LLM response JSON must be an object.")
    return payload


def decision_from_payload(
    *,
    payload: dict[str, Any],
    trade_week: date,
    model: str,
    raw_response: dict[str, Any],
) -> WeeklyDecision:
    """Validate parsed JSON and convert it to ``WeeklyDecision``."""
    required = {"ticker", "confidence", "thesis", "risks", "market_summary"}
    missing = sorted(required - payload.keys())
    if missing:
        raise DecisionParseError(f"LLM decision missing required fields: {missing}.")

    ticker = str(payload["ticker"]).upper().strip()
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise DecisionParseError("confidence must be numeric.") from exc
    if not 0 <= confidence <= 1:
        raise DecisionParseError("confidence must be between 0 and 1.")

    risks_raw = payload["risks"]
    if not isinstance(risks_raw, list) or not risks_raw:
        raise DecisionParseError("risks must be a non-empty array.")

    thesis = str(payload["thesis"]).strip()
    market_summary = str(payload["market_summary"]).strip()
    if not ticker or not thesis or not market_summary:
        raise DecisionParseError("ticker, thesis, and market_summary must be non-empty.")

    return WeeklyDecision(
        trade_week=trade_week,
        ticker=ticker,
        confidence=confidence,
        thesis=thesis,
        risks=[str(risk).strip() for risk in risks_raw if str(risk).strip()],
        market_summary=market_summary,
        raw_response=raw_response,
        model=model,
    )


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        raise DecisionParseError("Anthropic response contained no content.")
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    if not parts:
        raise DecisionParseError("Anthropic response contained no text blocks.")
    return "\n".join(parts)
