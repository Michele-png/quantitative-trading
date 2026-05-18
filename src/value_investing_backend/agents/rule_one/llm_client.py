"""Shared Anthropic LLM wrapper used by Rule One qualitative analyzers.

Centralizes the bits we want to apply consistently across every qualitative
LLM call:

    * Model = ``cfg.anthropic_model`` (defaults to Opus 4.7).
    * Extended thinking enabled with ``cfg.anthropic_thinking_budget_tokens``.
    * Single tool-call output for guaranteed JSON.
    * Tenacity-backed retry with exponential backoff.
    * Per-call cost estimate logged before/after the request so callers can
      audit spend in the screening orchestrator's logs.
    * Optional dry-run mode that skips the network call entirely and emits a
      synthetic "dry run" payload — used by ``screen_tickers(dry_run=True)``.

Dry-run output deliberately fails every check so a dry-run screening result
is conservative (no firm should pass purely on dry-run signals).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from value_investing_backend.config import get_config

log = logging.getLogger(__name__)


# Match either a fenced ```json ... ``` block or the first balanced { ... }
# object in the text. Used as a fallback when the model writes prose instead
# of calling the tool we offered.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from free-form model text.

    Returns the parsed dict, or None if no parseable JSON is found.
    Tries fenced code blocks first, then falls back to the largest
    well-balanced ``{ ... }`` substring.
    """
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    # Largest balanced { ... } substring as a fallback.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# Anthropic Opus 4.7 list pricing as of release (USD per million tokens). Used
# only for cost-estimation logs — actual billing is per Anthropic's invoice.
OPUS_47_USD_PER_MTOK_INPUT = 5.0
OPUS_47_USD_PER_MTOK_OUTPUT = 25.0
# Extended-thinking tokens are billed at the output rate.
ANTHROPIC_TOKENS_PER_CHAR_ESTIMATE = 0.25  # ≈ 4 chars / token average


@dataclass(frozen=True)
class LlmCallResult:
    """Result of one Anthropic structured-output call."""

    payload: dict[str, Any]
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    dry_run: bool


def _estimate_input_tokens(system_prompt: str, user_prompt: str, tool: dict) -> int:
    raw = system_prompt + user_prompt + str(tool)
    return int(len(raw) * ANTHROPIC_TOKENS_PER_CHAR_ESTIMATE)


class LlmClient:
    """Thin wrapper around ``Anthropic.messages.create`` for structured output."""

    def __init__(
        self,
        anthropic_client: Anthropic | None = None,
        *,
        model: str | None = None,
        thinking_budget_tokens: int | None = None,
        max_input_chars: int | None = None,
        max_output_tokens: int = 8192,
        dry_run: bool = False,
    ) -> None:
        cfg = get_config()
        self._client = anthropic_client or Anthropic(api_key=cfg.anthropic_api_key)
        self._model = model or cfg.anthropic_model
        self._thinking_budget_tokens = (
            thinking_budget_tokens if thinking_budget_tokens is not None
            else cfg.anthropic_thinking_budget_tokens
        )
        self._max_input_chars = (
            max_input_chars if max_input_chars is not None
            else cfg.anthropic_max_input_chars
        )
        # Anthropic requires max_tokens >= thinking budget tokens. We keep a
        # comfortable margin so the model has room for the actual answer.
        self._max_output_tokens = max(max_output_tokens, self._thinking_budget_tokens + 2048)
        self._dry_run = dry_run

    @property
    def model(self) -> str:
        return self._model

    @property
    def thinking_budget_tokens(self) -> int:
        return self._thinking_budget_tokens

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def truncate(self, text: str) -> str:
        """Trim a single document so the combined prompt fits the input budget."""
        if len(text) <= self._max_input_chars:
            return text
        return text[: self._max_input_chars] + "\n\n[... truncated for length ...]"

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        dry_run_payload: dict[str, Any] | None = None,
    ) -> LlmCallResult:
        """Invoke Anthropic with a single forced tool call. Caches NOTHING (caller's job)."""
        est_in = _estimate_input_tokens(system_prompt, user_prompt, tool)
        if self._dry_run:
            log.info(
                "[dry-run] Skipping Anthropic call (~%d input tokens) for tool %s",
                est_in, tool.get("name"),
            )
            return LlmCallResult(
                payload=dry_run_payload if dry_run_payload is not None else {},
                estimated_input_tokens=est_in, estimated_output_tokens=0,
                estimated_cost_usd=0.0, dry_run=True,
            )
        # Estimate cost using max_output_tokens as an upper bound.
        est_cost = (
            est_in / 1_000_000 * OPUS_47_USD_PER_MTOK_INPUT
            + self._max_output_tokens / 1_000_000 * OPUS_47_USD_PER_MTOK_OUTPUT
        )
        log.info(
            "Anthropic call: model=%s tool=%s thinking=%d tokens "
            "est_input=%d tokens, est_max_cost=$%.3f",
            self._model, tool.get("name"), self._thinking_budget_tokens,
            est_in, est_cost,
        )
        payload = self._invoke(system_prompt, user_prompt, tool)
        return LlmCallResult(
            payload=payload,
            estimated_input_tokens=est_in,
            estimated_output_tokens=self._max_output_tokens,  # upper bound
            estimated_cost_usd=est_cost,
            dry_run=False,
        )

    @retry(
        # Anthropic 'overloaded' errors typically persist for 30-180s, so a
        # 2-30s backoff is too aggressive. 5 attempts at 4s, 8s, 16s, 32s, 64s
        # (capped at 120s) gives the best chance of riding out a transient
        # capacity blip without tying up the workflow forever.
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=4, min=4, max=120),
        reraise=True,
    )
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
    ) -> dict[str, Any]:
        # tool_choice nuance: Anthropic forbids combining extended thinking
        # with a forced specific tool ("Thinking may not be enabled when
        # tool_choice forces tool use"). When thinking is enabled, fall back
        # to ``tool_choice: auto`` — since we only ever provide ONE tool, the
        # model reliably selects it after thinking. When thinking is disabled
        # (e.g. unit tests), we keep the deterministic forced-tool path.
        if self._thinking_budget_tokens > 0:
            tool_choice: dict[str, Any] = {"type": "auto"}
        else:
            tool_choice = {"type": "tool", "name": tool["name"]}

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "system": system_prompt,
            "tools": [tool],
            "tool_choice": tool_choice,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._thinking_budget_tokens > 0:
            # Opus 4.7 dropped the legacy ``thinking: {type: enabled,
            # budget_tokens: N}`` shape and requires the new adaptive form
            # plus ``output_config.effort``. We map the legacy budget knob to
            # an effort level so existing env vars keep working:
            #   budget >= 32k tokens -> "high"   (deep, expensive)
            #   budget >=  8k tokens -> "medium"
            #   budget  >  0  tokens -> "low"
            if self._thinking_budget_tokens >= 32_000:
                effort = "high"
            elif self._thinking_budget_tokens >= 8_000:
                effort = "medium"
            else:
                effort = "low"
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        # Use streaming because Opus 4.7 with extended thinking and a 1M-token
        # context can exceed Anthropic's 10-minute non-streaming SLA. The
        # streaming path is functionally equivalent — `.get_final_message()`
        # returns the same Message structure as `.messages.create()` once the
        # stream completes — but is the only supported way to make long calls.
        # See https://github.com/anthropics/anthropic-sdk-python#long-requests.
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        # Preferred path: model called the tool we offered.
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        # Fallback path: with tool_choice=auto + extended thinking the model
        # occasionally writes prose instead of calling the tool. Try to
        # recover by extracting a JSON object from the first text block —
        # the prose usually IS the JSON the tool would have received.
        for block in response.content:
            if getattr(block, "type", None) == "text":
                payload = _extract_json_object(getattr(block, "text", "") or "")
                if payload is not None:
                    return payload
        raise RuntimeError(
            "Anthropic returned no tool_use block and no parseable JSON in "
            f"text; got blocks: {[getattr(b, 'type', None) for b in response.content]}"
        )
