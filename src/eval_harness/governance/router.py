"""Performance-aware task router: which agent should get the next task?

Scores each eligible agent from its history (quality, reliability, cost,
latency) and picks the best. Agents with too little history get an exploration
bonus so the fleet keeps gathering evidence on newcomers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..storage import get_store
from .policy_config import DEFAULT_POLICY, PolicyConfig
from .profiles import AgentPerformance, compute_all_performance

# Backward-compatible aliases (single source of truth: PolicyConfig).
DEFAULT_WEIGHTS = DEFAULT_POLICY.weights
MIN_PASS_RATE = DEFAULT_POLICY.route_min_pass_rate
MAX_ERROR_RATE = DEFAULT_POLICY.route_max_error_rate
MIN_TOOL_SUCCESS = DEFAULT_POLICY.route_min_tool_success
EXPLORE_MIN_SAMPLES = DEFAULT_POLICY.explore_min_samples
EXPLORE_BONUS = DEFAULT_POLICY.explore_bonus


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


def _task_quality(p: AgentPerformance, task_type, config: PolicyConfig) -> float:
    """Quality on the criteria that matter for this task family.

    Falls back to the blended avg_score when no task hint is given or the
    relevant per-criterion scores are unavailable.
    """
    if task_type is None:
        return p.avg_score
    tt = task_type.value if hasattr(task_type, "value") else str(task_type)
    crit_weights = config.task_criterion_weights.get(tt)
    if not crit_weights:
        return p.avg_score
    num = den = 0.0
    for crit, w in crit_weights.items():
        val = p.per_criterion_avg.get(crit)
        if val is not None:
            num += w * val
            den += w
    return round(num / den, 3) if den else p.avg_score


def score_agents(
    profiles: list[AgentPerformance],
    weights: dict[str, float] | None = None,
    config: PolicyConfig = DEFAULT_POLICY,
    task_type=None,
) -> list[AgentScore]:
    weights = weights or config.weights
    if not profiles:
        return []

    norm_quality = _normalize([_task_quality(p, task_type, config) for p in profiles])
    norm_cost = _normalize([p.avg_cost_usd for p in profiles])
    norm_latency = _normalize([p.avg_latency_s for p in profiles])
    norm_error = _normalize([p.error_rate for p in profiles])
    norm_p95 = _normalize([p.p95_latency_s for p in profiles])

    scored: list[AgentScore] = []
    for i, p in enumerate(profiles):
        eligible = (
            p.pass_rate >= config.route_min_pass_rate
            and p.error_rate <= config.route_max_error_rate
            and (p.tool_success_rate is None or p.tool_success_rate >= config.route_min_tool_success)
            and p.uptime >= config.route_min_uptime
            and p.retry_rate <= config.route_max_retry_rate
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
        if p.n_evals < config.explore_min_samples:
            score += config.explore_bonus
            note = f"exploration bonus (only {p.n_evals} evals)"
        if not eligible:
            if p.pass_rate < config.route_min_pass_rate:
                note = f"ineligible: pass rate {p.pass_rate:.0%} < {config.route_min_pass_rate:.0%}"
            elif p.error_rate > config.route_max_error_rate:
                note = f"ineligible: error rate {p.error_rate:.0%} > {config.route_max_error_rate:.0%}"
            elif p.tool_success_rate is not None and p.tool_success_rate < config.route_min_tool_success:
                note = f"ineligible: tool success {p.tool_success_rate:.0%} < {config.route_min_tool_success:.0%}"
            elif p.uptime < config.route_min_uptime:
                note = f"ineligible: uptime {p.uptime:.0%} < {config.route_min_uptime:.0%}"
            elif p.retry_rate > config.route_max_retry_rate:
                note = f"ineligible: retry rate {p.retry_rate:.0%} > {config.route_max_retry_rate:.0%}"
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
    config: PolicyConfig = DEFAULT_POLICY,
) -> RoutingDecision:
    """Pick the best agent for an incoming task of `task_type`."""
    profiles = compute_all_performance(task_type)
    ranking = score_agents(profiles, weights, config, task_type=task_type)
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
