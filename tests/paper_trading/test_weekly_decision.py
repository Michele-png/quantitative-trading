"""Tests for zero-shot weekly decision parsing."""

from __future__ import annotations

from datetime import date

import pytest

from quantitative_trading.paper_trading.weekly_decision import (
    DecisionParseError,
    decision_from_payload,
    parse_decision_json,
)


class TestParseDecisionJson:
    """Tests for LLM JSON parsing."""

    def test_parses_plain_json(self) -> None:
        payload = parse_decision_json('{"ticker": "AAPL"}')

        assert payload == {"ticker": "AAPL"}

    def test_parses_accidental_json_code_fence(self) -> None:
        payload = parse_decision_json('```json\n{"ticker": "AAPL"}\n```')

        assert payload == {"ticker": "AAPL"}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(DecisionParseError, match="valid JSON"):
            parse_decision_json("ticker: AAPL")


class TestDecisionFromPayload:
    """Tests for structured decision validation."""

    def test_valid_payload_builds_decision(self) -> None:
        decision = decision_from_payload(
            payload={
                "ticker": "aapl",
                "confidence": 0.73,
                "thesis": "Strong weekly setup.",
                "risks": ["Earnings volatility"],
                "market_summary": "Market is constructive.",
            },
            trade_week=date(2026, 5, 4),
            model="claude-test",
            raw_response={"text": "{}"},
        )

        assert decision.ticker == "AAPL"
        assert decision.confidence == 0.73

    def test_missing_field_raises(self) -> None:
        with pytest.raises(DecisionParseError, match="missing"):
            decision_from_payload(
                payload={"ticker": "AAPL"},
                trade_week=date(2026, 5, 4),
                model="claude-test",
                raw_response={},
            )

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(DecisionParseError, match="between 0 and 1"):
            decision_from_payload(
                payload={
                    "ticker": "AAPL",
                    "confidence": 2,
                    "thesis": "x",
                    "risks": ["risk"],
                    "market_summary": "x",
                },
                trade_week=date(2026, 5, 4),
                model="claude-test",
                raw_response={},
            )
