"""Automated per-agent performance reviews.

Turns the accumulated performance profile into a manager-style review:
strengths, weaknesses, trend, autonomy recommendation. Uses an LLM to write the
narrative when a key is available, with a deterministic template fallback so
the feature works offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import settings
from ..llm import get_client
from ..schemas import TaskType
from .autonomy import assign_tier
from .profiles import AgentPerformance, compute_agent_performance

_REVIEW_SYSTEM = (
    "You are an engineering manager writing a concise, candid performance review "
    "of an AI agent based on its measured statistics. Be specific and reference the "
    "numbers. Cover: overall assessment, strengths, weaknesses, trend, and a clear "
    "recommendation on how much autonomy to grant. Keep it under 180 words."
)


@dataclass
class AgentReview:
    agent_id: str
    task_type: str
    tier: str
    narrative: str
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _stats_block(perf: AgentPerformance, tier: str) -> dict:
    return {
        "agent_id": perf.agent_id,
        "task_type": perf.task_type,
        "evaluations": perf.n_evals,
        "pass_rate": perf.pass_rate,
        "avg_score": perf.avg_score,
        "score_std": perf.score_std,
        "trend": perf.trend,
        "avg_latency_s": perf.avg_latency_s,
        "avg_cost_usd": perf.avg_cost_usd,
        "per_criterion_avg": perf.per_criterion_avg,
        "recommended_tier": tier,
    }


def _fallback_narrative(perf: AgentPerformance, tier: str) -> str:
    crit = perf.per_criterion_avg
    if crit:
        best = max(crit, key=crit.get)
        worst = min(crit, key=crit.get)
        strength = f"Strongest on {best} ({crit[best]:.1f}/5)."
        weakness = f"Weakest on {worst} ({crit[worst]:.1f}/5)."
    else:
        strength = weakness = ""
    trend = (
        "improving" if perf.trend > 0.1 else "declining" if perf.trend < -0.1 else "stable"
    )
    return (
        f"{perf.agent_id} handled {perf.n_evals} evaluations on {perf.task_type} with a "
        f"{perf.pass_rate:.0%} pass rate and an average score of {perf.avg_score:.2f}/5 "
        f"(trend: {trend}). {strength} {weakness} "
        f"Average cost ${perf.avg_cost_usd:.4f}/item at {perf.avg_latency_s:.2f}s latency. "
        f"Recommended autonomy: {tier}."
    ).strip()


def generate_review(
    agent_id: str,
    task_type: TaskType | None = None,
    use_llm: bool = True,
) -> AgentReview | None:
    """Generate a performance review for one agent from its history."""
    perf = compute_agent_performance(agent_id, task_type)
    if perf is None:
        return None

    tier = assign_tier(perf).tier.value
    stats = _stats_block(perf, tier)

    narrative = _fallback_narrative(perf, tier)
    if use_llm and settings.has_openai_key:
        try:
            text, _stats = get_client().complete_text(
                system=_REVIEW_SYSTEM,
                user="Agent statistics:\n" + json.dumps(stats, indent=2),
                model=settings.judge_model_cheap,
                temperature=0.3,
            )
            if text.strip():
                narrative = text.strip()
        except Exception:  # noqa: BLE001 - fall back to the template
            pass

    return AgentReview(
        agent_id=agent_id,
        task_type=perf.task_type,
        tier=tier,
        narrative=narrative,
        stats=stats,
    )
