"""Performance-aware task router: which agent should get the next task?

Scores each eligible agent from its history (quality, reliability, cost,
latency) and picks the best. Agents with too little history get an exploration
bonus so the fleet keeps gathering evidence on newcomers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..storage import get_store
from .profiles import AgentPerformance, compute_all_performance

# Weights for the routing score. Quality and reliability reward; cost and
# latency penalize. Tune per use case.
DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "reliability": 0.25,
    "cost": 0.08,
    "latency": 0.07,
    "error_rate": 0.15,
    "p95_latency": 0.10,
}
MIN_PASS_RATE = 0.5
MAX_ERROR_RATE = 0.25
MIN_TOOL_SUCCESS = 0.7
# Exploration bonus for under-sampled agents (fewer than this many evals).
EXPLORE_MIN_SAMPLES = 5
EXPLORE_BONUS = 0.05


@dataclass
class AgentScore:
    agent_id: str
    score: float
    pass_rate: float
    avg_score: float
    avg_cost_usd: float
    avg_latency_s: float
    eligible: bool
    note: str = ""


@dataclass
class RoutingDecision:
    task_type: str
    chosen_agent: str | None
    ranking: list[AgentScore] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "chosen_agent": self.chosen_agent,
            "rationale": self.rationale,
            "ranking": [s.__dict__ for s in self.ranking],
        }


def _normalize(values: list[float]) -> dict[int, float]:
    """Min-max normalize to 0-1 by index; all-equal -> 1.0 for everyone."""
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if hi == lo:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - lo) / (hi - lo) for i, v in enumerate(values)}


def score_agents(
    profiles: list[AgentPerformance],
    weights: dict[str, float] | None = None,
) -> list[AgentScore]:
    weights = weights or DEFAULT_WEIGHTS
    if not profiles:
        return []

    norm_quality = _normalize([p.avg_score for p in profiles])
    norm_cost = _normalize([p.avg_cost_usd for p in profiles])
    norm_latency = _normalize([p.avg_latency_s for p in profiles])
    norm_error = _normalize([p.error_rate for p in profiles])
    norm_p95 = _normalize([p.p95_latency_s for p in profiles])

    scored: list[AgentScore] = []
    for i, p in enumerate(profiles):
        eligible = (
            p.pass_rate >= MIN_PASS_RATE
            and p.error_rate <= MAX_ERROR_RATE
            and (p.tool_success_rate is None or p.tool_success_rate >= MIN_TOOL_SUCCESS)
        )
        score = (
            weights["quality"] * norm_quality.get(i, 0.0)
            + weights["reliability"] * p.pass_rate
            - weights["cost"] * norm_cost.get(i, 0.0)
            - weights["latency"] * norm_latency.get(i, 0.0)
            - weights["error_rate"] * norm_error.get(i, 0.0)
            - weights["p95_latency"] * norm_p95.get(i, 0.0)
        )
        note = ""
        if p.n_evals < EXPLORE_MIN_SAMPLES:
            score += EXPLORE_BONUS
            note = f"exploration bonus (only {p.n_evals} evals)"
        if not eligible:
            if p.pass_rate < MIN_PASS_RATE:
                note = f"ineligible: pass rate {p.pass_rate:.0%} < {MIN_PASS_RATE:.0%}"
            elif p.error_rate > MAX_ERROR_RATE:
                note = f"ineligible: error rate {p.error_rate:.0%} > {MAX_ERROR_RATE:.0%}"
            elif p.tool_success_rate is not None and p.tool_success_rate < MIN_TOOL_SUCCESS:
                note = f"ineligible: tool success {p.tool_success_rate:.0%} < {MIN_TOOL_SUCCESS:.0%}"
        scored.append(
            AgentScore(
                agent_id=p.agent_id,
                score=round(score, 4),
                pass_rate=p.pass_rate,
                avg_score=p.avg_score,
                avg_cost_usd=p.avg_cost_usd,
                avg_latency_s=p.avg_latency_s,
                eligible=eligible,
                note=note,
            )
        )
    scored.sort(key=lambda s: (s.eligible, s.score), reverse=True)
    return scored


def route_next_task(
    task_type,
    weights: dict[str, float] | None = None,
    audit: bool = True,
) -> RoutingDecision:
    """Pick the best agent for an incoming task of `task_type`."""
    profiles = compute_all_performance(task_type)
    ranking = score_agents(profiles, weights)
    eligible = [s for s in ranking if s.eligible]
    chosen = eligible[0].agent_id if eligible else None

    tt = task_type.value if hasattr(task_type, "value") else str(task_type)
    if chosen:
        top = eligible[0]
        rationale = (
            f"Selected {chosen}: score {top.score} "
            f"(pass {top.pass_rate:.0%}, avg {top.avg_score:.2f}, "
            f"${top.avg_cost_usd:.4f}/item, {top.avg_latency_s:.2f}s)."
        )
    else:
        rationale = "No eligible agent met the minimum pass rate; escalate to a human."

    decision = RoutingDecision(task_type=tt, chosen_agent=chosen, ranking=ranking, rationale=rationale)
    if audit:
        get_store().log_audit("task_routed", chosen or "none", decision.to_dict())
    return decision
