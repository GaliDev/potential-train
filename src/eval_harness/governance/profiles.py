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
from ..storage import EvalRow, RunSignalRow, get_store


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
    trend: float = 0.0
    # Operational signals from run_signals
    n_signals: int = 0
    uptime: float = 1.0
    error_rate: float = 0.0
    p95_latency_s: float = 0.0
    tool_success_rate: float | None = None
    retry_rate: float = 0.0
    refusal_rate: float = 0.0
    avg_groundedness: float | None = None
    avg_tokens: float = 0.0
    safety_flag_rate: float = 0.0
    operational_drift: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


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


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def _operational_drift(signals: list[RunSignalRow]) -> float:
    """Recent-half error rate minus older-half (positive = degrading)."""
    if len(signals) < 4:
        return 0.0
    ordered = sorted(signals, key=lambda s: s.created_at)
    mid = len(ordered) // 2
    older_err = sum(1 for s in ordered[:mid] if not s.success) / max(len(ordered[:mid]), 1)
    recent_err = sum(1 for s in ordered[mid:] if not s.success) / max(len(ordered[mid:]), 1)
    return round(recent_err - older_err, 3)


def _aggregate_signals(signals: list[RunSignalRow]) -> dict:
    if not signals:
        return {}
    latencies = [s.latency_s for s in signals]
    tokens = [s.prompt_tokens + s.completion_tokens for s in signals]
    grounded = [g for s in signals if (g := s.groundedness) is not None]
    tool_rates: list[float] = []
    for s in signals:
        if s.tool_calls > 0:
            tool_rates.append((s.tool_calls - s.tool_failures) / s.tool_calls)
    return {
        "n_signals": len(signals),
        "uptime": sum(1 for s in signals if s.success) / len(signals),
        "error_rate": sum(1 for s in signals if not s.success) / len(signals),
        "p95_latency_s": round(_p95(latencies), 4),
        "tool_success_rate": round(statistics.fmean(tool_rates), 3) if tool_rates else None,
        "retry_rate": sum(1 for s in signals if s.retries > 0) / len(signals),
        "refusal_rate": sum(1 for s in signals if s.refused) / len(signals),
        "avg_groundedness": round(statistics.fmean(grounded), 3) if grounded else None,
        "avg_tokens": round(statistics.fmean(tokens), 1) if tokens else 0.0,
        "safety_flag_rate": sum(1 for s in signals if s.safety_flag) / len(signals),
        "operational_drift": _operational_drift(signals),
    }


def compute_agent_performance(
    agent_id: str,
    task_type: TaskType | None = None,
    judge_mode: str | None = "panel",
) -> AgentPerformance | None:
    """Aggregate the store's eval rows for one agent into a profile."""
    store = get_store()
    rows = store.fetch_evals(agent_id=agent_id, task_type=task_type, judge_mode=judge_mode)
    signals = store.fetch_run_signals(agent_id=agent_id, task_type=task_type)
    if not rows and not signals:
        return None

    scores = [r.aggregate_score for r in rows] if rows else [0.0]
    ops = _aggregate_signals(signals)
    return AgentPerformance(
        agent_id=agent_id,
        task_type=task_type.value if task_type else (rows[0].task_type if rows else signals[0].task_type),
        n_evals=len(rows),
        pass_rate=round(sum(1 for r in rows if r.overall_pass) / len(rows), 3) if rows else 0.0,
        avg_score=round(statistics.fmean(scores), 3) if rows else 0.0,
        score_std=round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
        avg_latency_s=round(
            statistics.fmean([r.total_latency_s for r in rows]) if rows
            else statistics.fmean([s.latency_s for s in signals]),
            4,
        ),
        avg_cost_usd=round(statistics.fmean([r.total_cost_usd for r in rows]), 6) if rows else 0.0,
        per_criterion_avg=_parse_criteria(rows),
        trend=_trend(rows),
        **ops,
    )


def compute_all_performance(
    task_type: TaskType | None = None,
    judge_mode: str | None = "panel",
) -> list[AgentPerformance]:
    """Profiles for every registered agent that has evaluation history."""
    profiles: list[AgentPerformance] = []
    for agent in get_store().list_agents(task_type):
        perf = compute_agent_performance(agent.agent_id, agent.task_type, judge_mode=judge_mode)
        if perf is not None:
            profiles.append(perf)
    return profiles


def criterion_avg(perf: AgentPerformance, criterion: Criterion) -> float | None:
    return perf.per_criterion_avg.get(criterion.value)
