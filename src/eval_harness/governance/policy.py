"""Policy and governance engine.

Given an agent's earned autonomy tier and the risk of an incoming task, decide
whether the agent may act automatically, needs human approval, or is blocked.
Every decision is written to the audit log for accountability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..schemas import AutonomyTier, TaskType
from ..store import log_audit
from .autonomy import assign_tier
from .profiles import compute_agent_performance


class TaskRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"  # safety/compliance critical


@dataclass
class PolicyDecision:
    agent_id: str
    task_type: str
    task_risk: str
    tier: str
    allow: bool
    requires_approval: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def decide(
    agent_id: str,
    task_type: TaskType,
    tier: AutonomyTier,
    task_risk: TaskRisk = TaskRisk.LOW,
    audit: bool = True,
) -> PolicyDecision:
    """Apply governance rules to an (agent, task, tier, risk) tuple."""
    reasons: list[str] = []
    allow = True
    requires_approval = False

    if tier == AutonomyTier.BLOCKED:
        allow = False
        reasons.append("Agent is blocked by autonomy calibration.")
    elif tier == AutonomyTier.HUMAN_IN_LOOP:
        requires_approval = True
        reasons.append("Agent is human-in-the-loop; output requires approval before use.")
    elif tier == AutonomyTier.AUTO_SPOT_CHECK:
        reasons.append("Agent may act, but outputs are subject to random spot checks.")
    else:  # FULL_AUTO
        reasons.append("Agent is trusted for full autonomy.")

    # High-risk tasks always require a human, regardless of tier (unless blocked).
    if task_risk == TaskRisk.HIGH and allow:
        requires_approval = True
        reasons.append("High-risk task: human approval required by policy.")

    decision = PolicyDecision(
        agent_id=agent_id,
        task_type=task_type.value,
        task_risk=task_risk.value,
        tier=tier.value,
        allow=allow,
        requires_approval=requires_approval,
        reasons=reasons,
    )
    if audit:
        log_audit("policy_decision", agent_id, decision.to_dict())
    return decision


def decide_for_agent(
    agent_id: str,
    task_type: TaskType,
    task_risk: TaskRisk = TaskRisk.LOW,
    audit: bool = True,
) -> PolicyDecision | None:
    """Convenience: compute the agent's tier from history, then apply policy."""
    perf = compute_agent_performance(agent_id, task_type)
    if perf is None:
        return None
    tier = assign_tier(perf).tier
    return decide(agent_id, task_type, tier, task_risk, audit=audit)
