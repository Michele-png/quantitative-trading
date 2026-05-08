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

import logging
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantitative_trading.config import get_config


log = logging.getLogger(__name__)


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
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "system": system_prompt,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._thinking_budget_tokens > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }
        response = self._client.messages.create(**kwargs)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError(
            "Anthropic returned no tool_use block; got blocks: "
            f"{[getattr(b, 'type', None) for b in response.content]}"
        )
