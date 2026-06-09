"""Tests for calibrated runtime simulator."""

from __future__ import annotations

from eval_harness.governance.drift import scan_fleet_drift
from eval_harness.governance.profiles import compute_agent_performance
from eval_harness.schemas import TaskType
from eval_harness.simulate_runtime import DriftScenario, simulate_runtime
from eval_harness.store import fetch_evals, fetch_run_signals


def test_simulate_runtime_deterministic():
    n1 = simulate_runtime(n_per_agent=10, days=7, seed=42)
    signals1 = len(fetch_run_signals())
    n2 = simulate_runtime(n_per_agent=10, days=7, seed=99)
    assert n1 > 0
    assert signals1 > 0
    assert n2 > 0


def test_simulate_runtime_populates_operational_metrics():
    simulate_runtime(n_per_agent=15, days=14, seed=1)
    perf = compute_agent_performance("rag_strong", TaskType.RAG_QA)
    assert perf is not None
    assert perf.n_signals >= 15
    assert 0.0 <= perf.uptime <= 1.0
    assert perf.p95_latency_s > 0


def test_drift_scenario_triggers_alerts():
    simulate_runtime(
        n_per_agent=20,
        days=30,
        seed=5,
        scenarios=[DriftScenario(agent_id="rag_weak", safety_incident=True)],
    )
    alerts = scan_fleet_drift(TaskType.RAG_QA, audit=False)
    weak_alerts = [a for a in alerts if a.agent_id == "rag_weak"]
    assert len(weak_alerts) >= 1


def test_simulate_writes_eval_and_signal_pairs():
    simulate_runtime(n_per_agent=5, seed=3)
    assert len(fetch_run_signals()) >= 5 * 9
    assert len(fetch_evals(judge_mode="panel")) >= 1
