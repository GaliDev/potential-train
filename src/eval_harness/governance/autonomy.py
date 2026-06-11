"""Autonomy calibration: how much independence has each agent earned?

Maps measured reliability (pass rate, score, safety, sample size) to an
explicit autonomy tier. Trust is granted from evidence, not assumed: agents
with too little history are capped at human-in-the-loop regardless of score.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import AutonomyTier, Criterion
from ..storage import get_store
from .policy_config import DEFAULT_POLICY, PolicyConfig
from .profiles import AgentPerformance, compute_all_performance, criterion_avg

# Backward-compatible aliases (single source of truth: PolicyConfig).
MIN_SAMPLES_FOR_TRUST = DEFAULT_POLICY.min_samples_for_trust
SAFETY_FLOOR = DEFAULT_POLICY.safety_score_floor


@dataclass
class AutonomyDecision:
    agent_id: str
    tier: AutonomyTier
    pass_rate: float
    avg_score: float
    n_evals: int
    rationale: str

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["tier"] = self.tier.value
        return d


MAX_OPERATIONAL_ERROR = DEFAULT_POLICY.operational_error_max
MAX_REFUSAL_RATE = DEFAULT_POLICY.refusal_rate_max
DRIFT_DEMOTE_THRESHOLD = DEFAULT_POLICY.drift_demote_threshold


def assign_tier(perf: AgentPerformance, config: PolicyConfig = DEFAULT_POLICY) -> AutonomyDecision:
    safety = criterion_avg(perf, Criterion.SAFETY)
    reasons: list[str] = []

    # --- Hard gates: block regardless of quality ---
    if perf.safety_flag_rate > config.safety_flag_rate_max:
        reasons.append(f"safety flags in {perf.safety_flag_rate:.0%} of runs")
        return _decision(perf, AutonomyTier.BLOCKED, reasons)

    if safety is not None and safety < config.safety_score_floor:
        reasons.append(f"safety {safety:.1f} below floor {config.safety_score_floor}")
        return _decision(perf, AutonomyTier.BLOCKED, reasons)

    if perf.error_rate > config.operational_error_max:
        reasons.append(f"operational error rate {perf.error_rate:.0%} > {config.operational_error_max:.0%}")
        return _decision(perf, AutonomyTier.BLOCKED, reasons)

    # Per-criterion hard floors (e.g. faithfulness = hallucination control).
    for crit, floor in config.criterion_block_floors.items():
        val = perf.per_criterion_avg.get(crit)
        if val is not None and val < floor:
            reasons.append(f"{crit} {val:.1f} below block floor {floor}")
            return _decision(perf, AutonomyTier.BLOCKED, reasons)

    # --- Quality band from pass rate ---
    if perf.pass_rate < config.pass_rate_block:
        tier = AutonomyTier.BLOCKED
        reasons.append(f"pass rate {perf.pass_rate:.0%} < {config.pass_rate_block:.0%}")
    elif perf.pass_rate < config.pass_rate_hil:
        tier = AutonomyTier.HUMAN_IN_LOOP
        reasons.append(f"pass rate {perf.pass_rate:.0%} in {config.pass_rate_block:.0%}-{config.pass_rate_hil:.0%}")
    elif perf.pass_rate < config.pass_rate_spot:
        tier = AutonomyTier.AUTO_SPOT_CHECK
        reasons.append(f"pass rate {perf.pass_rate:.0%} in {config.pass_rate_hil:.0%}-{config.pass_rate_spot:.0%}")
    else:
        tier = AutonomyTier.FULL_AUTO
        reasons.append(f"pass rate {perf.pass_rate:.0%} >= {config.pass_rate_spot:.0%}")

    # Cap unproven agents: not enough evidence to grant high autonomy.
    if perf.n_evals < config.min_samples_for_trust and tier in (
        AutonomyTier.FULL_AUTO,
        AutonomyTier.AUTO_SPOT_CHECK,
    ):
        tier = AutonomyTier.HUMAN_IN_LOOP
        reasons.append(f"only {perf.n_evals} evals (< {config.min_samples_for_trust}); capped")

    if (
        tier == AutonomyTier.AUTO_SPOT_CHECK
        and perf.avg_score >= config.spot_to_full_score
        and perf.n_evals >= config.min_samples_for_trust
    ):
        tier = AutonomyTier.FULL_AUTO
        reasons.append(f"avg score {perf.avg_score:.1f} >= {config.spot_to_full_score} lifts to full auto")

    if perf.refusal_rate > config.refusal_rate_max and tier != AutonomyTier.BLOCKED:
        tier = AutonomyTier.HUMAN_IN_LOOP
        reasons.append(f"refusal rate {perf.refusal_rate:.0%} > {config.refusal_rate_max:.0%}")

    # --- Drift demotions (one step down from full auto), using all drift signals ---
    if perf.operational_drift >= config.drift_demote_threshold and tier == AutonomyTier.FULL_AUTO:
        tier = AutonomyTier.AUTO_SPOT_CHECK
        reasons.append(f"operational drift {perf.operational_drift:.0%} demotes to spot-check")

    if perf.trend <= config.quality_trend_demote and tier == AutonomyTier.FULL_AUTO:
        tier = AutonomyTier.AUTO_SPOT_CHECK
        reasons.append(f"quality trend {perf.trend:+.2f} demotes to spot-check")

    if perf.groundedness_drift <= config.groundedness_drift_demote and tier == AutonomyTier.FULL_AUTO:
        tier = AutonomyTier.AUTO_SPOT_CHECK
        reasons.append(f"groundedness drift {perf.groundedness_drift:+.2f} demotes to spot-check")

    # Per-criterion demotion floors: a weak individual criterion caps full auto.
    for crit, floor in config.criterion_demote_floors.items():
        val = perf.per_criterion_avg.get(crit)
        if val is not None and val < floor and tier == AutonomyTier.FULL_AUTO:
            tier = AutonomyTier.AUTO_SPOT_CHECK
            reasons.append(f"{crit} {val:.1f} below floor {floor} demotes to spot-check")

    return _decision(perf, tier, reasons)


def _decision(perf: AgentPerformance, tier: AutonomyTier, reasons: list[str]) -> AutonomyDecision:
    return AutonomyDecision(
        agent_id=perf.agent_id,
        tier=tier,
        pass_rate=perf.pass_rate,
        avg_score=perf.avg_score,
        n_evals=perf.n_evals,
        rationale="; ".join(reasons),
    )


def calibrate_fleet(
    task_type=None, audit: bool = True, config: PolicyConfig = DEFAULT_POLICY
) -> list[AutonomyDecision]:
    """Assign an autonomy tier to every agent with evaluation history."""
    decisions = [assign_tier(p, config) for p in compute_all_performance(task_type)]
    if audit:
        for d in decisions:
            get_store().log_audit("autonomy_assigned", d.agent_id, d.to_dict())
    return decisions
