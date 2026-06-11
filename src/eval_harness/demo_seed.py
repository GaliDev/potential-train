"""Seed the store with synthetic evaluation history for demos.

Lets the dashboard and governance layer be explored without spending API
calls. Engineers clear quality gaps between agents so routing, autonomy tiers,
and reviews show meaningful differences.
"""

from __future__ import annotations

import random

from .agents.aggregator import aggregate
from .fleet.configs import build_fleet
from .schemas import Criterion, JudgeVerdict, TaskType
from .store import init_db, record_eval, upsert_agent

# Target average quality (1-5) per agent, and rough cost/latency per item.
_PROFILE = {
    "rag_strong": (4.8, 0.0042, 1.7),
    "rag_cheap": (4.2, 0.0008, 0.9),
    "rag_weak": (2.4, 0.0006, 0.8),
    "sum_strong": (4.7, 0.0045, 1.8),
    "sum_cheap": (4.1, 0.0008, 0.9),
    "sum_weak": (2.6, 0.0006, 0.8),
    "trans_strong": (4.7, 0.0044, 1.7),
    "trans_cheap": (4.0, 0.0008, 0.9),
    "trans_weak": (2.5, 0.0006, 0.8),
}

# The LangGraph agents under agents/. They share the same store and dashboard,
# so we seed them synthetically too — generation is free (use_llm=False), but
# the 5-judge panel that would score a real run needs an OPENAI_API_KEY.
_LANGGRAPH_PROFILE = {
    "rag_react": (4.5, 0.0, 1.1),
    "sum_refine": (4.3, 0.0, 1.3),
    "trans_backcheck": (4.4, 0.0, 1.2),
}

_QUALITY_CRITERIA = [
    Criterion.CORRECTNESS,
    Criterion.FAITHFULNESS,
    Criterion.COMPLETENESS,
    Criterion.COHERENCE,
]


def _clamp(x: float) -> int:
    return max(1, min(5, int(round(x))))


def _seed_one_agent(profile, base, cost, latency, n_per_agent, rng) -> int:
    """Seed synthetic panel evaluations for a single agent. Returns row count."""
    upsert_agent(profile)
    n = 0
    for k in range(n_per_agent):
        verdicts: list[JudgeVerdict] = []
        for crit in _QUALITY_CRITERIA:
            score = _clamp(base + rng.uniform(-0.6, 0.6))
            verdicts.append(
                JudgeVerdict(
                    criterion=crit, score=score, passed=score >= 4,
                    rationale="synthetic", model=profile.model,
                    latency_s=latency / 5, cost_usd=cost / 5,
                )
            )
        # Safety stays high for all agents; weak agents fail on quality, not safety.
        verdicts.append(
            JudgeVerdict(
                criterion=Criterion.SAFETY, score=5, passed=True,
                rationale="no safety issues", model=profile.model,
                latency_s=latency / 5, cost_usd=cost / 5,
            )
        )
        result = aggregate(
            item_id=f"{profile.agent_id}_demo_{k}",
            task_type=profile.task_type,
            verdicts=verdicts,
            agent_id=profile.agent_id,
            judge_mode="panel",
        )
        record_eval(result)
        n += 1
    return n


def seed_demo_data(n_per_agent: int = 12, seed: int = 7) -> int:
    """Populate the store with synthetic panel evaluations. Returns row count.

    Seeds both the built-in prompt/model fleet (build_fleet) and the LangGraph
    agents discovered under agents/, so all of them appear in the dashboard
    without spending API calls.
    """
    rng = random.Random(seed)
    init_db()
    n = 0
    for agent in build_fleet():
        base, cost, latency = _PROFILE.get(agent.agent_id, (4.0, 0.001, 1.0))
        n += _seed_one_agent(agent.profile, base, cost, latency, n_per_agent, rng)

    # Also seed the registered LangGraph agents (rag_react, sum_refine, ...).
    from .fleet.registry import list_registered_agents

    for agent in list_registered_agents():
        if agent.agent_id not in _LANGGRAPH_PROFILE:
            continue
        base, cost, latency = _LANGGRAPH_PROFILE[agent.agent_id]
        n += _seed_one_agent(agent.profile, base, cost, latency, n_per_agent, rng)
    return n


if __name__ == "__main__":
    count = seed_demo_data()
    print(f"Seeded {count} synthetic evaluations across the fleet.")
