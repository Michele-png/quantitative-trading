"""Tests for the paper-trading JSONL ledger."""

from __future__ import annotations

from datetime import date

from quantitative_trading.paper_trading.ledger import PaperTradingLedger


class TestPaperTradingLedger:
    """Tests for append-only ledger behavior."""

    def test_append_and_read_events(self, tmp_path) -> None:
        ledger = PaperTradingLedger(tmp_path / "ledger.jsonl")

        ledger.append(
            event_type="dry_run_week",
            trade_week=date(2026, 5, 4),
            payload={"ticker": "AAPL"},
        )

        events = ledger.read_events()
        assert len(events) == 1
        assert events[0]["payload"]["ticker"] == "AAPL"

    def test_has_executed_week_only_matches_execute_events(self, tmp_path) -> None:
        ledger = PaperTradingLedger(tmp_path / "ledger.jsonl")
        week = date(2026, 5, 4)

        ledger.append(event_type="dry_run_week", trade_week=week, payload={})
        assert not ledger.has_executed_week(week)

        ledger.append(event_type="executed_week", trade_week=week, payload={})
        assert ledger.has_executed_week(week)
