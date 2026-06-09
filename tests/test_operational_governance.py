"""Tests for operational-signal-aware governance."""

from __future__ import annotations

from eval_harness.governance.autonomy import assign_tier
from eval_harness.governance.drift import detect_drift_for_agent
from eval_harness.governance.profiles import AgentPerformance
from eval_harness.governance.router import score_agents
from eval_harness.schemas import AutonomyTier


def _perf(**kwargs) -> AgentPerformance:
    defaults = dict(
        agent_id="a", task_type="rag_qa", n_evals=10, pass_rate=0.9,
        avg_score=4.5, score_std=0.2, avg_latency_s=1.0, avg_cost_usd=0.001,
        per_criterion_avg={"safety": 5.0}, trend=0.0,
        n_signals=20, uptime=0.95, error_rate=0.05, p95_latency_s=2.0,
        tool_success_rate=0.95, retry_rate=0.1, refusal_rate=0.05,
        avg_groundedness=0.8, avg_tokens=200.0, safety_flag_rate=0.0,
        operational_drift=0.0,
    )
    defaults.update(kwargs)
    return AgentPerformance(**defaults)


def test_router_penalizes_high_error_rate():
    good = _perf(agent_id="good", error_rate=0.02)
    bad = _perf(agent_id="bad", error_rate=0.4, pass_rate=0.9)
    scores = {s.agent_id: s for s in score_agents([good, bad])}
    assert scores["good"].eligible is True
    assert scores["bad"].eligible is False


def test_autonomy_blocks_high_operational_error():
    tier = assign_tier(_perf(error_rate=0.35)).tier
    assert tier == AutonomyTier.BLOCKED


def test_autonomy_demotes_on_drift():
    tier = assign_tier(_perf(operational_drift=0.2)).tier
    assert tier == AutonomyTier.AUTO_SPOT_CHECK


def test_drift_detects_error_spike():
    alerts = detect_drift_for_agent(_perf(operational_drift=0.2, error_rate=0.25))
    types = {a.alert_type for a in alerts}
    assert "error_rate_drift" in types
    assert "high_error_rate" in types
