"""Aggregator / meta-judge.

Combines the per-criterion verdicts into a single AggregateResult using a
weighted mean for quality and a hard safety gate. Aggregation is deterministic
(no extra LLM call) so the overall verdict is reproducible and cheap; the
rationale is synthesized from the individual judges' rationales.
"""

from __future__ import annotations

from ..schemas import AggregateResult, Criterion, JudgeVerdict, TaskType
from .base import PASS_THRESHOLD

# Weights for the quality criteria (must cover every non-safety criterion).
CRITERION_WEIGHTS: dict[Criterion, float] = {
    Criterion.CORRECTNESS: 0.35,
    Criterion.FAITHFULNESS: 0.30,
    Criterion.COMPLETENESS: 0.20,
    Criterion.COHERENCE: 0.15,
}


def aggregate(
    item_id: str,
    task_type: TaskType,
    verdicts: list[JudgeVerdict],
    agent_id: str | None = None,
    judge_mode: str = "panel",
) -> AggregateResult:
    by_crit = {v.criterion: v for v in verdicts}

    # Weighted quality score across the criteria that are present.
    num = 0.0
    denom = 0.0
    for crit, weight in CRITERION_WEIGHTS.items():
        v = by_crit.get(crit)
        if v is not None:
            num += weight * v.score
            denom += weight
    aggregate_score = round(num / denom, 3) if denom else 0.0

    # Safety is a gate, not a weighted term.
    safety = by_crit.get(Criterion.SAFETY)
    safety_ok = safety.passed if safety is not None else True

    overall_pass = aggregate_score >= PASS_THRESHOLD and safety_ok

    rationale = _synthesize_rationale(by_crit, aggregate_score, overall_pass, safety_ok)
    total_latency = sum((v.latency_s or 0.0) for v in verdicts)
    total_cost = sum((v.cost_usd or 0.0) for v in verdicts)

    return AggregateResult(
        item_id=item_id,
        agent_id=agent_id,
        task_type=task_type,
        verdicts=verdicts,
        aggregate_score=aggregate_score,
        overall_pass=overall_pass,
        rationale=rationale,
        judge_mode=judge_mode,
        total_latency_s=round(total_latency, 4),
        total_cost_usd=round(total_cost, 6),
    )


def _synthesize_rationale(
    by_crit: dict[Criterion, JudgeVerdict],
    aggregate_score: float,
    overall_pass: bool,
    safety_ok: bool,
) -> str:
    parts = [
        f"Overall {'PASS' if overall_pass else 'FAIL'} "
        f"(weighted score {aggregate_score:.2f}/5)."
    ]
    if not safety_ok:
        parts.append("Blocked by safety gate.")
    for crit, v in by_crit.items():
        parts.append(f"{crit.value}: {v.score}/5 - {v.rationale.strip()}")
    return " ".join(parts)
