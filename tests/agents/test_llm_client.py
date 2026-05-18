"""Tests for the shared LlmClient wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from value_investing_backend.agents.rule_one.llm_client import LlmClient
from value_investing_backend.config import get_config


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _make_response(payload: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    response = MagicMock()
    response.content = [block]
    return response


def _wire_stream(anthropic: MagicMock, response: MagicMock) -> None:
    """Wire up an Anthropic mock so `messages.stream(...).get_final_message()` returns ``response``.

    The library uses ``with client.messages.stream(**kwargs) as stream:``,
    which requires the return value of ``messages.stream`` to be a context
    manager exposing ``get_final_message()``.
    """
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value = stream_cm
    stream_cm.__exit__.return_value = None
    stream_cm.get_final_message.return_value = response
    anthropic.messages.stream.return_value = stream_cm


_TOOL = {
    "name": "submit_test",
    "description": "test",
    "input_schema": {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    },
}


def test_call_invokes_anthropic_with_thinking_enabled() -> None:
    anthropic = MagicMock()
    _wire_stream(anthropic, _make_response({"x": 42}))
    llm = LlmClient(anthropic_client=anthropic, thinking_budget_tokens=10000)
    result = llm.call(
        system_prompt="sys", user_prompt="user", tool=_TOOL,
    )
    assert result.payload == {"x": 42}
    assert not result.dry_run
    kwargs = anthropic.messages.stream.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    # Opus 4.7 uses the adaptive thinking shape; budget_tokens is mapped to
    # output_config.effort. 10k tokens -> "medium".
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}
    # When thinking is enabled, Anthropic disallows forced specific-tool
    # selection, so we fall back to ``auto`` (the model picks the only tool
    # we provide).
    assert kwargs["tool_choice"] == {"type": "auto"}
    anthropic.messages.create.assert_not_called()


def test_dry_run_skips_anthropic_and_returns_payload() -> None:
    anthropic = MagicMock()
    llm = LlmClient(anthropic_client=anthropic, dry_run=True)
    result = llm.call(
        system_prompt="sys", user_prompt="user", tool=_TOOL,
        dry_run_payload={"x": 0},
    )
    assert result.dry_run
    assert result.payload == {"x": 0}
    anthropic.messages.create.assert_not_called()
    anthropic.messages.stream.assert_not_called()


def test_max_tokens_at_least_thinking_budget_plus_buffer() -> None:
    anthropic = MagicMock()
    _wire_stream(anthropic, _make_response({"x": 1}))
    llm = LlmClient(
        anthropic_client=anthropic, thinking_budget_tokens=20000,
        max_output_tokens=4096,
    )
    llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    kwargs = anthropic.messages.stream.call_args.kwargs
    # Anthropic requires max_tokens > thinking budget; we keep a 2k buffer.
    assert kwargs["max_tokens"] >= 20000 + 2000


def test_truncate_trims_long_text() -> None:
    llm = LlmClient(
        anthropic_client=MagicMock(), max_input_chars=100,
    )
    out = llm.truncate("a" * 200)
    assert len(out) < 200
    assert "truncated" in out


def test_zero_thinking_budget_omits_thinking_param() -> None:
    anthropic = MagicMock()
    _wire_stream(anthropic, _make_response({"x": 1}))
    llm = LlmClient(anthropic_client=anthropic, thinking_budget_tokens=0)
    llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    kwargs = anthropic.messages.stream.call_args.kwargs
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    # With thinking disabled, the deterministic forced-specific-tool path is safe.
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_test"}


def test_high_thinking_budget_maps_to_high_effort() -> None:
    anthropic = MagicMock()
    _wire_stream(anthropic, _make_response({"x": 1}))
    llm = LlmClient(anthropic_client=anthropic, thinking_budget_tokens=64_000)
    llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    kwargs = anthropic.messages.stream.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


def test_low_thinking_budget_maps_to_low_effort() -> None:
    anthropic = MagicMock()
    _wire_stream(anthropic, _make_response({"x": 1}))
    llm = LlmClient(anthropic_client=anthropic, thinking_budget_tokens=2_000)
    llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    kwargs = anthropic.messages.stream.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low"}


def test_call_raises_when_no_tool_use_or_parseable_text() -> None:
    block = MagicMock()
    block.type = "text"
    block.text = "I am not JSON, just an apology."
    response = MagicMock()
    response.content = [block]
    anthropic = MagicMock()
    _wire_stream(anthropic, response)
    llm = LlmClient(anthropic_client=anthropic)
    with pytest.raises(RuntimeError, match="no parseable JSON"):
        llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)


def test_call_falls_back_to_json_in_text_block() -> None:
    """When tool_choice=auto + thinking, the model sometimes writes prose
    containing the JSON instead of calling the tool. The fallback should
    extract it."""
    thinking = MagicMock()
    thinking.type = "thinking"
    text = MagicMock()
    text.type = "text"
    text.text = 'Here is the result:\n```json\n{"x": 99}\n```'
    response = MagicMock()
    response.content = [thinking, text]
    anthropic = MagicMock()
    _wire_stream(anthropic, response)
    llm = LlmClient(anthropic_client=anthropic)
    result = llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    assert result.payload == {"x": 99}


def test_call_falls_back_to_balanced_braces_when_no_fence() -> None:
    text = MagicMock()
    text.type = "text"
    text.text = 'My answer: {"x": 7, "rationale": "ok"} — done.'
    response = MagicMock()
    response.content = [text]
    anthropic = MagicMock()
    _wire_stream(anthropic, response)
    llm = LlmClient(anthropic_client=anthropic)
    result = llm.call(system_prompt="s", user_prompt="u", tool=_TOOL)
    assert result.payload == {"x": 7, "rationale": "ok"}
