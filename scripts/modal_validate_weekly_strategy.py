"""Optional Modal parallel validation scaffold for weekly strategy replays.

Run only when local historical replay or candidate evaluation becomes slow:

    modal run scripts/modal_validate_weekly_strategy.py
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency scaffold
    modal = None


if modal is not None:
    image = modal.Image.debian_slim().pip_install(
        "anthropic>=0.40.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.0.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "tenacity>=8.2.0",
    )
    app = modal.App("weekly-paper-trader-validation", image=image)

    @app.function(cpu=2.0, memory=2048, timeout=3600)
    def validate_trade_week(trade_week: str) -> dict[str, str]:
        """Placeholder remote validation unit for a single trade week."""
        return {"trade_week": trade_week, "status": "not_implemented"}

    @app.local_entrypoint()
    def main() -> None:
        """Run placeholder validation over a small set of trade weeks."""
        trade_weeks = ["2026-05-04", "2026-05-11"]
        for result in validate_trade_week.map(trade_weeks):
            print(result)
else:

    def main() -> None:
        """Tell the user how to enable Modal validation."""
        raise SystemExit("Install and authenticate Modal before running this optional scaffold.")


if __name__ == "__main__":
    main()
