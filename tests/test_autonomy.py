from eval_harness.governance.autonomy import assign_tier
from eval_harness.governance.profiles import AgentPerformance
from eval_harness.schemas import AutonomyTier


def _perf(pass_rate, avg_score, n, safety=5.0):
    return AgentPerformance(
        agent_id="a", task_type="rag_qa", n_evals=n, pass_rate=pass_rate,
        avg_score=avg_score, score_std=0.2, avg_latency_s=1.0, avg_cost_usd=0.001,
        per_criterion_avg={"safety": safety}, trend=0.0,
    )


def test_full_auto_for_reliable_agent():
    assert assign_tier(_perf(0.95, 4.8, 10)).tier == AutonomyTier.FULL_AUTO


def test_spot_check_band():
    assert assign_tier(_perf(0.8, 4.2, 10)).tier == AutonomyTier.AUTO_SPOT_CHECK


def test_human_in_loop_band():
    assert assign_tier(_perf(0.6, 3.5, 10)).tier == AutonomyTier.HUMAN_IN_LOOP


def test_blocked_low_pass_rate():
    assert assign_tier(_perf(0.3, 2.0, 10)).tier == AutonomyTier.BLOCKED


def test_safety_floor_blocks_regardless():
    assert assign_tier(_perf(0.99, 5.0, 50, safety=2.0)).tier == AutonomyTier.BLOCKED


def test_unproven_agent_is_capped():
    # High pass rate but too few evals -> capped to human in loop.
    assert assign_tier(_perf(0.95, 4.8, 3)).tier == AutonomyTier.HUMAN_IN_LOOP
