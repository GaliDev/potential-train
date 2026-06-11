"""Tests for the composed decision engine (offline, no API key)."""

from __future__ import annotations

from eval_harness.governance.engine import (
    Decision,
    build_agent_decision,
    confidence,
    decide_for_agent,
    decide_for_task,
    resolve_tier,
    signal_coverage,
)
from eval_harness.governance.policy import TaskRisk
from eval_harness.governance.policy_config import PolicyConfig
from eval_harness.governance.profiles import AgentPerformance
from eval_harness.schemas import AutonomyTier, TaskType


_ALL_CRITERIA = {
    "correctness": 4.7, "faithfulness": 4.6, "completeness": 4.5,
    "coherence": 4.5, "safety": 5.0,
}


def _perf(**kw) -> AgentPerformance:
    d = dict(
        agent_id="a", task_type="rag_qa", n_evals=20, pass_rate=0.95, avg_score=4.8,
        score_std=0.2, avg_latency_s=1.0, avg_cost_usd=0.001,
        per_criterion_avg=dict(_ALL_CRITERIA), trend=0.0,
        n_signals=20, uptime=0.99, error_rate=0.02, p95_latency_s=2.0,
        tool_success_rate=0.97, retry_rate=0.05, refusal_rate=0.02,
        avg_groundedness=0.85, avg_tokens=200.0, safety_flag_rate=0.0,
        operational_drift=0.0, groundedness_drift=0.0,
    )
    d.update(kw)
    return AgentPerformance(**d)


# --- Signal coverage -------------------------------------------------------
def test_signal_coverage_full():
    cov = signal_coverage(_perf())
    assert cov.ratio == 1.0
    assert not cov.missing


def test_signal_coverage_partial_flags_missing():
    cov = signal_coverage(_perf(n_signals=0, per_criterion_avg={}))
    assert cov.ratio < 1.0
    assert "uptime" in cov.missing
    assert "safety" in cov.missing


# --- Confidence ------------------------------------------------------------
def test_confidence_high_with_full_evidence():
    assert confidence(_perf()) >= 0.9


def test_confidence_low_without_signals():
    assert confidence(_perf(n_evals=2, n_signals=0, per_criterion_avg={})) < 0.5


# --- Composed decisions + precedence ---------------------------------------
def test_steady_strong_is_full_auto():
    d = build_agent_decision(_perf(), TaskType.RAG_QA)
    assert d.tier == "full_auto"
    assert d.allow and not d.requires_approval
    assert d.action == "auto"


def test_safety_incident_blocks_everything():
    d = build_agent_decision(_perf(safety_flag_rate=0.1), TaskType.RAG_QA)
    assert d.tier == "blocked"
    assert not d.allow
    assert "policy_gate" in d.precedence


def test_drift_demotes_and_is_flagged():
    d = build_agent_decision(_perf(operational_drift=0.2), TaskType.RAG_QA)
    assert d.tier == "auto_spot_check"
    assert any(a["alert_type"] == "error_rate_drift" for a in d.drift_alerts)
    assert "drift_detected" in d.precedence


def test_unproven_agent_capped():
    d = build_agent_decision(
        _perf(n_evals=3, n_signals=0, per_criterion_avg={"safety": 5.0}), TaskType.RAG_QA
    )
    assert d.tier == "human_in_loop"


def test_low_coverage_caps_to_human_in_loop():
    d = build_agent_decision(_perf(n_signals=0, per_criterion_avg={}), TaskType.RAG_QA)
    assert d.tier == "human_in_loop"
    assert d.confidence < 1.0
    assert "partial_signal_coverage" in d.precedence


def test_high_risk_requires_approval():
    d = build_agent_decision(_perf(), TaskType.RAG_QA, task_risk=TaskRisk.HIGH)
    assert d.allow and d.requires_approval


# --- Hysteresis (anti-flapping) --------------------------------------------
def test_demotion_applies_immediately():
    tier, _ = resolve_tier(_perf(operational_drift=0.2), previous_tier=AutonomyTier.FULL_AUTO)
    assert tier == AutonomyTier.AUTO_SPOT_CHECK


def test_promotion_held_without_confidence():
    strict = PolicyConfig(promote_confidence=0.99)
    # Raw tier would be full_auto, but confidence < 0.99 -> hold at previous.
    tier, conf = resolve_tier(
        _perf(n_signals=0, per_criterion_avg={"safety": 5.0}),
        previous_tier=AutonomyTier.AUTO_SPOT_CHECK,
        config=strict,
    )
    assert tier == AutonomyTier.AUTO_SPOT_CHECK
    assert conf < 0.99


# --- Store integration (uses conftest temp DB) -----------------------------
def test_decide_for_agent_integration():
    from eval_harness.simulate_runtime import simulate_runtime

    simulate_runtime(n_per_agent=12, days=12, seed=3)
    d = decide_for_agent("rag_strong", TaskType.RAG_QA, audit=False)
    assert isinstance(d, Decision)
    assert d.tier is not None
    assert d.confidence > 0
    assert "policy_gate" in d.precedence


def test_decide_for_agent_unknown_is_safe_block():
    d = decide_for_agent("does_not_exist", TaskType.RAG_QA, audit=False)
    assert d.tier == "blocked"
    assert not d.allow
    assert "insufficient_data" in d.precedence


def test_decide_for_task_routes_or_escalates():
    from eval_harness.simulate_runtime import simulate_runtime

    simulate_runtime(n_per_agent=12, days=12, seed=4)
    d = decide_for_task(TaskType.RAG_QA, audit=False)
    assert d.kind == "task"
    assert d.action.startswith("route:") or d.action == "escalate_to_human"


# --- Per-criterion enforcement (the enhancement) ---------------------------
def test_low_faithfulness_blocks_via_floor():
    perf = _perf(per_criterion_avg={**_ALL_CRITERIA, "faithfulness": 2.0})
    d = build_agent_decision(perf, TaskType.RAG_QA)
    assert d.tier == "blocked"
    assert "faithfulness" in d.rationale


def test_weak_criterion_demotes_full_auto():
    # Faithfulness above the block floor (2.5) but below the demote floor (3.5).
    perf = _perf(per_criterion_avg={**_ALL_CRITERIA, "faithfulness": 3.2})
    d = build_agent_decision(perf, TaskType.RAG_QA)
    assert d.tier == "auto_spot_check"


def test_decision_carries_criterion_breakdown():
    d = build_agent_decision(_perf(), TaskType.RAG_QA)
    assert d.criteria.get("faithfulness") == 4.6
    assert set(d.criteria) >= {"correctness", "faithfulness", "completeness", "coherence", "safety"}


def test_criterion_aware_routing_prefers_task_relevant_strength():
    from eval_harness.governance.router import score_agents

    # Equal blended quality, but A is stronger on faithfulness (RAG weights it 0.5).
    a = _perf(agent_id="faithful",
              per_criterion_avg={**_ALL_CRITERIA, "faithfulness": 4.9, "completeness": 4.1})
    b = _perf(agent_id="thorough",
              per_criterion_avg={**_ALL_CRITERIA, "faithfulness": 4.0, "completeness": 5.0})
    a.avg_score = b.avg_score = 4.5
    ranked = score_agents([a, b], task_type=TaskType.RAG_QA)
    assert ranked[0].agent_id == "faithful"


def test_criterion_drift_detected_after_regression():
    # Older evals strong on faithfulness, recent ones weak -> regression alert.
    from datetime import datetime, timedelta, timezone

    from eval_harness.governance.drift import detect_drift_for_agent
    from eval_harness.schemas import (
        AggregateResult, Criterion, JudgeVerdict, TaskType as TT,
    )
    from eval_harness.storage import get_store

    store = get_store()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def _record(faith: float, when: datetime) -> None:
        verdicts = [
            JudgeVerdict(criterion=Criterion.FAITHFULNESS, score=int(faith), passed=faith >= 4, rationale="x"),
            JudgeVerdict(criterion=Criterion.SAFETY, score=5, passed=True, rationale="x"),
        ]
        store.record_eval(AggregateResult(
            item_id=f"i{when.isoformat()}", agent_id="reg", task_type=TT.RAG_QA,
            verdicts=verdicts, aggregate_score=faith, overall_pass=faith >= 4,
            judge_mode="panel", created_at=when,
        ))

    for k in range(4):
        _record(5, base + timedelta(hours=k))         # older: strong
    for k in range(4):
        _record(2, base + timedelta(hours=10 + k))    # recent: weak

    perf = _perf(agent_id="reg")
    alerts = detect_drift_for_agent(perf)
    assert any(a.alert_type == "criterion_drift" for a in alerts)
