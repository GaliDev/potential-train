"""Calibrated runtime simulator: timestamped eval + operational signals at scale.

Calibrates per-agent distributions from real ``RunSignalRow`` history when
available, otherwise falls back to ``demo_seed`` profiles. Supports injectable
drift/incident scenarios for governance demos.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .agents.aggregator import aggregate
from .demo_seed import _PROFILE as _DEMO_PROFILE
from .fleet.configs import build_fleet
from .schemas import Criterion, ExecutionTrace, JudgeVerdict, TaskType
from .store import fetch_run_signals, init_db, record_eval, record_run_signal, upsert_agent


@dataclass
class AgentCalibration:
    avg_latency_s: float
    avg_tokens: int
    tool_failure_rate: float
    error_rate: float
    groundedness: float
    avg_score: float
    tool_calls: int
    retries: float


@dataclass
class DriftScenario:
    """Inject degradation for one agent after a fraction of the time window."""

    agent_id: str
    start_fraction: float = 0.6
    error_rate_boost: float = 0.25
    groundedness_drop: float = 0.3
    latency_multiplier: float = 2.0
    safety_incident: bool = False


def _clamp_score(x: float) -> int:
    return max(1, min(5, int(round(x))))


def calibrate_agent(agent_id: str) -> AgentCalibration:
    """Build per-agent distributions from real signals or demo_seed fallback."""
    signals = fetch_run_signals(agent_id=agent_id)
    base_score, base_cost, base_lat = _DEMO_PROFILE.get(agent_id, (4.0, 0.001, 1.0))

    if signals:
        latencies = [s.latency_s for s in signals]
        tokens = [s.prompt_tokens + s.completion_tokens for s in signals]
        tool_calls = statistics.fmean([s.tool_calls for s in signals])
        retries = statistics.fmean([s.retries for s in signals])
        grounded = [g for s in signals if (g := s.groundedness) is not None]
        errors = sum(1 for s in signals if not s.success) / len(signals)
        tool_fails = []
        for s in signals:
            if s.tool_calls > 0:
                tool_fails.append(s.tool_failures / s.tool_calls)
        return AgentCalibration(
            avg_latency_s=statistics.fmean(latencies) if latencies else base_lat,
            avg_tokens=int(statistics.fmean(tokens)) if tokens else 200,
            tool_failure_rate=statistics.fmean(tool_fails) if tool_fails else 0.05,
            error_rate=errors,
            groundedness=statistics.fmean(grounded) if grounded else 0.7,
            avg_score=base_score,
            tool_calls=max(0, int(round(tool_calls))),
            retries=retries,
        )

    weak = agent_id.endswith("_weak")
    return AgentCalibration(
        avg_latency_s=base_lat,
        avg_tokens=150 if weak else 350,
        tool_failure_rate=0.15 if weak else 0.03,
        error_rate=0.12 if weak else 0.02,
        groundedness=0.35 if weak else 0.85,
        avg_score=base_score,
        tool_calls=2 if agent_id.startswith(("rag_", "trans_")) else 1,
        retries=0.4 if weak else 0.1,
    )


def _apply_scenario(
    cal: AgentCalibration,
    scenario: DriftScenario | None,
    *,
    t_fraction: float,
    rng: random.Random,
) -> AgentCalibration:
    if scenario is None or t_fraction < scenario.start_fraction:
        return cal
    boosted = AgentCalibration(
        avg_latency_s=cal.avg_latency_s * scenario.latency_multiplier,
        avg_tokens=cal.avg_tokens,
        tool_failure_rate=min(1.0, cal.tool_failure_rate + 0.2),
        error_rate=min(1.0, cal.error_rate + scenario.error_rate_boost),
        groundedness=max(0.0, cal.groundedness - scenario.groundedness_drop),
        avg_score=max(1.0, cal.avg_score - 1.2),
        tool_calls=cal.tool_calls,
        retries=cal.retries + 0.5,
    )
    if scenario.safety_incident and rng.random() < 0.1:
        boosted = AgentCalibration(**{**boosted.__dict__, "avg_score": 2.0})
    return boosted


def _synthetic_verdicts(
    agent_id: str,
    model: str,
    base_score: float,
    latency: float,
    cost: float,
    safety_flag: bool,
    rng: random.Random,
) -> list[JudgeVerdict]:
    criteria = [
        Criterion.CORRECTNESS, Criterion.FAITHFULNESS,
        Criterion.COMPLETENESS, Criterion.COHERENCE,
    ]
    verdicts: list[JudgeVerdict] = []
    for crit in criteria:
        score = _clamp_score(base_score + rng.uniform(-0.6, 0.6))
        verdicts.append(
            JudgeVerdict(
                criterion=crit, score=score, passed=score >= 4,
                rationale="simulated", model=model,
                latency_s=latency / 5, cost_usd=cost / 5,
            )
        )
    safety_score = 2 if safety_flag else 5
    verdicts.append(
        JudgeVerdict(
            criterion=Criterion.SAFETY, score=safety_score, passed=safety_score >= 4,
            rationale="simulated safety", model=model,
            latency_s=latency / 5, cost_usd=cost / 5,
        )
    )
    return verdicts


def simulate_runtime(
    *,
    n_per_agent: int = 30,
    days: int = 30,
    seed: int = 7,
    scenarios: list[DriftScenario] | None = None,
) -> int:
    """Generate timestamped EvalRow + RunSignalRow pairs. Returns row count."""
    rng = random.Random(seed)
    init_db()
    scenario_map = {s.agent_id: s for s in (scenarios or [])}
    start = datetime.now(timezone.utc) - timedelta(days=days)
    span = timedelta(days=days)
    count = 0

    for agent in build_fleet():
        upsert_agent(agent.profile)
        base_cal = calibrate_agent(agent.agent_id)
        scenario = scenario_map.get(agent.agent_id)

        for k in range(n_per_agent):
            t_fraction = k / max(n_per_agent - 1, 1)
            ts = start + span * t_fraction
            cal = _apply_scenario(base_cal, scenario, t_fraction=t_fraction, rng=rng)

            success = rng.random() >= cal.error_rate
            tool_failures = 0
            if cal.tool_calls > 0 and rng.random() < cal.tool_failure_rate:
                tool_failures = 1
            refused = cal.groundedness < 0.25 and rng.random() < 0.5
            safety_flag = scenario is not None and scenario.safety_incident and t_fraction >= scenario.start_fraction and rng.random() < 0.08
            retries = int(rng.random() < cal.retries)
            latency = max(0.1, cal.avg_latency_s * rng.uniform(0.7, 1.4))
            tokens = max(10, int(cal.avg_tokens * rng.uniform(0.8, 1.2)))
            cost = tokens * 0.000002

            trace = ExecutionTrace(
                steps=max(1, cal.tool_calls + 1),
                tool_calls=cal.tool_calls,
                tool_failures=tool_failures,
                retries=retries,
                error=None if success else "simulated failure",
                refused=refused,
                groundedness=cal.groundedness * rng.uniform(0.9, 1.05),
                latency_s=latency,
                prompt_tokens=int(tokens * 0.7),
                completion_tokens=int(tokens * 0.3),
                cost_usd=cost,
                safety_flag=safety_flag,
                success=success,
                model=agent.model,
                created_at=ts,
            )

            item_id = f"{agent.agent_id}_sim_{k}"
            record_run_signal(
                item_id=item_id,
                agent_id=agent.agent_id,
                task_type=agent.task_type,
                trace=trace,
                created_at=ts,
            )

            if success:
                verdicts = _synthetic_verdicts(
                    agent.agent_id, agent.model, cal.avg_score, latency, cost,
                    safety_flag, rng,
                )
                result = aggregate(
                    item_id=item_id,
                    task_type=agent.task_type,
                    verdicts=verdicts,
                    agent_id=agent.agent_id,
                    judge_mode="panel",
                )
                result.created_at = ts
                record_eval(result)
                count += 1

    return count


if __name__ == "__main__":
    n = simulate_runtime(
        scenarios=[
            DriftScenario(agent_id="rag_weak", safety_incident=True),
            DriftScenario(agent_id="sum_weak", latency_multiplier=3.0),
        ],
    )
    print(f"Simulated {n} evaluation + signal pairs across the fleet.")
