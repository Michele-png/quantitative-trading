"""Deterministic pre-screening before the zero-shot LLM decision."""

from __future__ import annotations

import math

from quantitative_trading.paper_trading.models import CandidateSnapshot


class CandidateScorer:
    """Rank liquid candidates using simple, auditable market features."""

    def score(self, candidates: list[CandidateSnapshot]) -> list[CandidateSnapshot]:
        """Return candidates sorted by deterministic score, descending."""
        if not candidates:
            return []

        dollar_volumes = [max(candidate.dollar_volume, 1.0) for candidate in candidates]
        max_log_volume = max(math.log10(value) for value in dollar_volumes)
        rescored = []
        for candidate in candidates:
            liquidity_score = math.log10(max(candidate.dollar_volume, 1.0)) / max_log_volume
            momentum_score = self._bounded_return_score(candidate.five_day_return)
            intraday_score = self._bounded_return_score(candidate.day_return)
            news_score = min(len(candidate.news_headlines), 5) / 5
            score = (0.45 * liquidity_score) + (0.30 * momentum_score) + (0.15 * intraday_score) + (
                0.10 * news_score
            )
            rescored.append(
                CandidateSnapshot(
                    symbol=candidate.symbol,
                    name=candidate.name,
                    price=candidate.price,
                    previous_close=candidate.previous_close,
                    day_return=candidate.day_return,
                    five_day_return=candidate.five_day_return,
                    dollar_volume=candidate.dollar_volume,
                    news_headlines=candidate.news_headlines,
                    score=score,
                )
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _bounded_return_score(value: float | None) -> float:
        if value is None:
            return 0.5
        return max(0.0, min(1.0, 0.5 + value * 5))
