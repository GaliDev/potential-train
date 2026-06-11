"""The decision engine: compose all signals into one resolved, auditable decision.

This is the deterministic core of the governance layer. It does **not** call an
LLM - it reads aggregated signals (through the storage seam, so it is backend
agnostic) and resolves them into a single decision using an explicit precedence:

  1. Safety / policy hard block         (overrides everything)
  2. Active drift / incident            (force demotion + flag)
  3. Earned autonomy tier               (capped by evidence/confidence)
  4. Routing preference                 (quality / cost / reliability)
  5. Policy gate                        (risk -> allow / approval / block)

Two stability properties make it production-credible:
- **Evidence gating**: every decision carries a confidence from sample size and
  signal coverage; low-coverage decisions default to the safe side.
- **Asymmetric hysteresis**: demotions apply immediately (safety first), but a
  promotion only sticks if confidence clears `promote_confidence` and no drift
  is active - preventing tier flapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import AutonomyTier, TaskType
from ..storage import get_store
from .autonomy import assign_tier
from .drift import detect_drift_for_agent
from .policy import TaskRisk, decide
from .policy_config import DEFAULT_POLICY, PolicyConfig
from .profiles import AgentPerformance, compute_agent_performance
from .router import route_next_task

# Worst -> best, used for hysteresis comparisons.
_TIER_ORDER = [
    AutonomyTier.BLOCKED,
    AutonomyTier.HUMAN_IN_LOOP,
    AutonomyTier.AUTO_SPOT_CHECK,
    AutonomyTier.FULL_AUTO,
]


def _rank(tier: AutonomyTier) -> int:
    return _TIER_ORDER.index(tier)


@dataclass
class SignalCoverage:
    """Which required signals were available to base a decision on."""

    present: list[str]
    missing: list[str]
    ratio: float


@dataclass
class Decision:
    """One resolved governance decision with full provenance."""

    subject: str                       # agent_id or task type
    kind: str                          # "agent" | "task"
    action: str                        # auto | approval_required | block | route:<id>|... | escalate_to_human
    tier: str | None
    allow: bool
    requires_approval: bool
    confidence: float
    signals_used: list[str] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)
    drift_alerts: list[dict] = field(default_factory=list)
    precedence: list[str] = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    rationale: str = ""
    policy_version: str = "v1"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def signal_coverage(perf: AgentPerformance, config: PolicyConfig = DEFAULT_POLICY) -> SignalCoverage:
    """Report which required signals are present for this agent's profile."""
    present: list[str] = []
    missing: list[str] = []

    has_quality = perf.n_evals > 0
    for sig in config.required_scalar_quality:
        (present if has_quality else missing).append(sig)

    # Each judge criterion is a required signal in its own right.
    for crit in config.required_criteria:
        (present if perf.per_criterion_avg.get(crit) is not None else missing).append(crit)

    has_ops = perf.n_signals > 0
    for sig in config.required_operational:
        (present if has_ops else missing).append(sig)

    total = len(present) + len(missing)
    ratio = round(len(present) / total, 3) if total else 0.0
    return SignalCoverage(present=present, missing=missing, ratio=ratio)


def confidence(perf: AgentPerformance, config: PolicyConfig = DEFAULT_POLICY) -> float:
    """Confidence in a decision: evidence volume x signal coverage, in [0, 1]."""
    evidence = perf.n_evals + perf.n_signals
    base = min(1.0, evidence / max(config.full_confidence_samples, 1))
    cov = signal_coverage(perf, config)
    return round(base * cov.ratio, 3)


def _apply_hysteresis(
    raw: AutonomyTier,
    conf: float,
    previous_tier: AutonomyTier | None,
    config: PolicyConfig,
    drift_alerts: list | None,
) -> AutonomyTier:
    if previous_tier is None:
        return raw
    if _rank(raw) < _rank(previous_tier):
        # Demotion: apply immediately (safety first).
        return raw
    if _rank(raw) > _rank(previous_tier):
        # Promotion: only stick with enough confidence and no active drift.
        if conf >= config.promote_confidence and not drift_alerts:
            return raw
        return previous_tier
    return raw


def resolve_tier(
    perf: AgentPerformance,
    previous_tier: AutonomyTier | None = None,
    config: PolicyConfig = DEFAULT_POLICY,
    drift_alerts: list | None = None,
) -> tuple[AutonomyTier, float]:
    """Apply asymmetric hysteresis to the raw tier; return (tier, confidence)."""
    raw = assign_tier(perf, config).tier
    conf = confidence(perf, config)
    return _apply_hysteresis(raw, conf, previous_tier, config, drift_alerts), conf


def build_agent_decision(
    perf: AgentPerformance,
    task_type: TaskType,
    task_risk: TaskRisk = TaskRisk.LOW,
    previous_tier: AutonomyTier | None = None,
    config: PolicyConfig = DEFAULT_POLICY,
) -> Decision:
    """Pure composition (no store I/O beyond drift's signal lookup)."""
    precedence: list[str] = []
    cov = signal_coverage(perf, config)
    drift_alerts = detect_drift_for_agent(perf, config)
    if drift_alerts:
        precedence.append("drift_detected")

    auto = assign_tier(perf, config)
    conf = confidence(perf, config)
    tier = _apply_hysteresis(auto.tier, conf, previous_tier, config, drift_alerts)

    if cov.missing:
        precedence.append("partial_signal_coverage")
        # Low confidence + a high tier -> cap to human-in-the-loop (safe default).
        if conf < config.promote_confidence and _rank(tier) > _rank(AutonomyTier.HUMAN_IN_LOOP):
            tier = AutonomyTier.HUMAN_IN_LOOP
            precedence.append("low_confidence_cap")

    pol = decide(perf.agent_id, task_type, tier, task_risk, audit=False, config=config)
    precedence.append("autonomy_tier")
    precedence.append("policy_gate")

    if not pol.allow:
        action = "block"
    elif pol.requires_approval:
        action = "approval_required"
    else:
        action = "auto"

    rationale_bits = []
    if drift_alerts:
        rationale_bits.append(f"{len(drift_alerts)} drift/incident alert(s)")
    rationale_bits.append(f"tier={tier.value} (confidence {conf:.0%})")
    if auto.rationale:
        rationale_bits.append(auto.rationale)
    rationale_bits.extend(pol.reasons)
    if cov.missing:
        rationale_bits.append(f"missing signals: {', '.join(cov.missing)}")

    return Decision(
        subject=perf.agent_id,
        kind="agent",
        action=action,
        tier=tier.value,
        allow=pol.allow,
        requires_approval=pol.requires_approval,
        confidence=conf,
        signals_used=cov.present,
        missing_signals=cov.missing,
        drift_alerts=[a.to_dict() for a in drift_alerts],
        precedence=precedence,
        criteria=dict(perf.per_criterion_avg),
        rationale="; ".join(rationale_bits),
        policy_version=config.version,
    )


def decide_for_agent(
    agent_id: str,
    task_type: TaskType,
    task_risk: TaskRisk = TaskRisk.LOW,
    previous_tier: AutonomyTier | None = None,
    config: PolicyConfig = DEFAULT_POLICY,
    audit: bool = True,
) -> Decision:
    """Resolve the full decision for one agent on a task type."""
    perf = compute_agent_performance(agent_id, task_type)
    if perf is None:
        d = Decision(
            subject=agent_id, kind="agent", action="block", tier=AutonomyTier.BLOCKED.value,
            allow=False, requires_approval=True, confidence=0.0,
            signals_used=[], missing_signals=["all"], drift_alerts=[],
            precedence=["insufficient_data"],
            rationale="No evaluation history; defaulting to safe (blocked) until evidence exists.",
            policy_version=config.version,
        )
    else:
        d = build_agent_decision(perf, task_type, task_risk, previous_tier, config)
    if audit:
        get_store().log_audit("engine_decision", agent_id, d.to_dict())
    return d


def decide_for_task(
    task_type: TaskType,
    task_risk: TaskRisk = TaskRisk.LOW,
    config: PolicyConfig = DEFAULT_POLICY,
    audit: bool = True,
) -> Decision:
    """Resolve the full decision for an incoming task: route, then gate the pick."""
    tt = task_type.value if hasattr(task_type, "value") else str(task_type)
    routing = route_next_task(task_type, audit=False, config=config)
    if routing.chosen_agent is None:
        d = Decision(
            subject=tt, kind="task", action="escalate_to_human", tier=None,
            allow=False, requires_approval=True, confidence=0.0,
            precedence=["no_eligible_agent"], rationale=routing.rationale,
            policy_version=config.version,
        )
    else:
        agent_dec = decide_for_agent(
            routing.chosen_agent, task_type, task_risk, config=config, audit=False
        )
        d = Decision(
            subject=tt, kind="task",
            action=f"route:{routing.chosen_agent}|{agent_dec.action}",
            tier=agent_dec.tier, allow=agent_dec.allow,
            requires_approval=agent_dec.requires_approval, confidence=agent_dec.confidence,
            signals_used=agent_dec.signals_used, missing_signals=agent_dec.missing_signals,
            drift_alerts=agent_dec.drift_alerts, criteria=agent_dec.criteria,
            precedence=["routing", *agent_dec.precedence],
            rationale=f"Routed to {routing.chosen_agent}: {agent_dec.rationale}",
            policy_version=config.version,
        )
    if audit:
        get_store().log_audit("engine_task_decision", tt, d.to_dict())
    return d
