"""Improvement levers over the base panel.

The assignment asks us to improve at least one dimension after establishing a
baseline. We provide two drop-in judges:

- CascadeJudge (COST lever): screen every item with the cheap model; only
  escalate borderline/uncertain items to the strong model. Most clear-cut items
  are decided cheaply, cutting cost while preserving agreement.
- JuryJudge (QUALITY lever): run the panel several times and take the median
  per-criterion score, reducing single-run variance and bias.
"""

from __future__ import annotations

import statistics

from .agents.aggregator import aggregate
from .agents.base import PASS_THRESHOLD
from .config import settings
from .graph import PanelJudge
from .schemas import AggregateResult, JudgeVerdict, TestItem


class CascadeJudge:
    """Cheap-first cascade: escalate only uncertain items to the strong model."""

    judge_mode = "cascade"

    def __init__(
        self,
        cheap_model: str | None = None,
        strong_model: str | None = None,
        band: float = 0.75,
    ) -> None:
        self._cheap = PanelJudge(model=cheap_model or settings.judge_model_cheap)
        self._strong = PanelJudge(model=strong_model or settings.judge_model)
        # Items whose score lands within +/- band of the pass threshold are
        # "uncertain" and get a second, stronger opinion.
        self._band = band

    def _is_borderline(self, result: AggregateResult) -> bool:
        return abs(result.aggregate_score - PASS_THRESHOLD) <= self._band

    def judge(self, item: TestItem) -> AggregateResult:
        cheap = self._cheap.judge(item)
        if not self._is_borderline(cheap):
            cheap.judge_mode = self.judge_mode
            cheap.rationale = "[cheap-only, confident] " + cheap.rationale
            return cheap

        strong = self._strong.judge(item)
        strong.judge_mode = self.judge_mode
        strong.total_cost_usd = round(cheap.total_cost_usd + strong.total_cost_usd, 6)
        strong.total_latency_s = round(cheap.total_latency_s + strong.total_latency_s, 4)
        strong.rationale = "[escalated to strong model] " + strong.rationale
        return strong


class JuryJudge:
    """Run the panel several times and take the median per-criterion score."""

    judge_mode = "jury"

    def __init__(self, model: str | None = None, rounds: int = 3) -> None:
        self._panel = PanelJudge(model=model or settings.judge_model)
        self._rounds = max(1, rounds)

    def judge(self, item: TestItem) -> AggregateResult:
        runs = [self._panel.judge(item) for _ in range(self._rounds)]

        # Group verdicts by criterion across runs.
        by_crit: dict = {}
        for run in runs:
            for v in run.verdicts:
                by_crit.setdefault(v.criterion, []).append(v)

        merged: list[JudgeVerdict] = []
        for crit, verdicts in by_crit.items():
            median_score = int(round(statistics.median(v.score for v in verdicts)))
            merged.append(
                JudgeVerdict(
                    criterion=crit,
                    score=median_score,
                    passed=median_score >= PASS_THRESHOLD,
                    rationale=f"median of {len(verdicts)} runs; " + verdicts[0].rationale,
                    evidence=verdicts[0].evidence,
                    model=verdicts[0].model,
                    latency_s=sum(v.latency_s or 0.0 for v in verdicts),
                    cost_usd=sum(v.cost_usd or 0.0 for v in verdicts),
                )
            )

        result = aggregate(
            item_id=item.id,
            task_type=item.task_type,
            verdicts=merged,
            agent_id=item.agent_id,
            judge_mode=self.judge_mode,
        )
        result.rationale = f"[jury of {self._rounds}] " + result.rationale
        return result
