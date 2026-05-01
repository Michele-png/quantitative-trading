"""Append-only JSONL ledger for paper-trading decisions and runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quantitative_trading.paper_trading.models import utc_now


class LedgerError(RuntimeError):
    """Raised when the paper-trading ledger cannot be read or written."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable.")


class PaperTradingLedger:
    """Append-only local record of decisions, orders, and accounting snapshots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, event_type: str, trade_week: date, payload: dict[str, Any]) -> None:
        """Append one event to the ledger."""
        record = {
            "event_type": event_type,
            "trade_week": trade_week.isoformat(),
            "created_at": utc_now().isoformat(),
            "payload": payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default, sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            raise LedgerError(f"Could not append to ledger {self.path}: {exc}") from exc

    def read_events(self) -> list[dict[str, Any]]:
        """Read all ledger events."""
        if not self.path.exists():
            return []
        events = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerError(f"Could not read ledger {self.path}: {exc}") from exc
        return events

    def has_executed_week(self, trade_week: date) -> bool:
        """Return whether an execute run already recorded this trade week."""
        week = trade_week.isoformat()
        return any(
            event.get("trade_week") == week and event.get("event_type") == "executed_week"
            for event in self.read_events()
        )

    def latest_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent events."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        return self.read_events()[-limit:]
