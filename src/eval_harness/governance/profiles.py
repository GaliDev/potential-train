"""Per-agent performance profiles aggregated from the evaluation store.

This is the shared substrate the rest of the governance layer reads from:
routing, autonomy calibration, and performance reviews all derive their
decisions from these aggregated stats.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field

from ..schemas import Criterion, TaskType
from ..store import EvalRow, fetch_evals, list_agents


@dataclass
class AgentPerformance:
    agent_id: str
    task_type: str
    n_evals: int
    pass_rate: float
    avg_score: float
    score_std: float
    avg_latency_s: float
    avg_cost_usd: float
    per_criterion_avg: dict[str, float] = field(default_factory=dict)
    # Trend: recent-half avg score minus older-half avg score (>0 means improving).
    trend: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _parse_criteria(rows: list[EvalRow]) -> dict[str, float]:
    """Average each criterion's score across rows that recorded verdicts."""
    sums: dict[str, list[float]] = {}
    for row in rows:
        try:
            verdicts = json.loads(row.verdicts_json or "[]")
        except json.JSONDecodeError:
            continue
        for v in verdicts:
            crit = v.get("criterion")
            score = v.get("score")
            if crit is not None and score is not None:
                sums.setdefault(crit, []).append(float(score))
    return {c: round(statistics.fmean(vals), 3) for c, vals in sums.items() if vals}


def _trend(rows: list[EvalRow]) -> float:
    """Recent-half mean score minus older-half mean score (chronological)."""
    if len(rows) < 4:
        return 0.0
    ordered = sorted(rows, key=lambda r: r.created_at)
    mid = len(ordered) // 2
    older = [r.aggregate_score for r in ordered[:mid]]
    recent = [r.aggregate_score for r in ordered[mid:]]
    if not older or not recent:
        return 0.0
    return round(statistics.fmean(recent) - statistics.fmean(older), 3)


def compute_agent_performance(
    agent_id: str,
    task_type: TaskType | None = None,
    judge_mode: str | None = "panel",
) -> AgentPerformance | None:
    """Aggregate the store's eval rows for one agent into a profile."""
    rows = fetch_evals(agent_id=agent_id, task_type=task_type, judge_mode=judge_mode)
    if not rows:
        return None
    scores = [r.aggregate_score for r in rows]
    return AgentPerformance(
        agent_id=agent_id,
        task_type=task_type.value if task_type else rows[0].task_type,
        n_evals=len(rows),
        pass_rate=round(sum(1 for r in rows if r.overall_pass) / len(rows), 3),
        avg_score=round(statistics.fmean(scores), 3),
        score_std=round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
        avg_latency_s=round(statistics.fmean([r.total_latency_s for r in rows]), 4),
        avg_cost_usd=round(statistics.fmean([r.total_cost_usd for r in rows]), 6),
        per_criterion_avg=_parse_criteria(rows),
        trend=_trend(rows),
    )


def compute_all_performance(
    task_type: TaskType | None = None,
    judge_mode: str | None = "panel",
) -> list[AgentPerformance]:
    """Profiles for every registered agent that has evaluation history."""
    profiles: list[AgentPerformance] = []
    for agent in list_agents(task_type):
        perf = compute_agent_performance(agent.agent_id, agent.task_type, judge_mode=judge_mode)
        if perf is not None:
            profiles.append(perf)
    return profiles


def criterion_avg(perf: AgentPerformance, criterion: Criterion) -> float | None:
    return perf.per_criterion_avg.get(criterion.value)
