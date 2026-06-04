import pytest

from eval_harness.demo_seed import seed_demo_data
from eval_harness.governance import autonomy, policy, reviews, router
from eval_harness.governance.profiles import compute_agent_performance, compute_all_performance
from eval_harness.schemas import AutonomyTier, TaskType


@pytest.fixture
def seeded():
    return seed_demo_data(n_per_agent=12, seed=1)


def test_profiles_built_for_fleet(seeded):
    profiles = compute_all_performance(TaskType.RAG_QA)
    assert {p.agent_id for p in profiles} == {"rag_strong", "rag_cheap", "rag_weak"}
    weak = compute_agent_performance("rag_weak", TaskType.RAG_QA)
    strong = compute_agent_performance("rag_strong", TaskType.RAG_QA)
    assert strong.avg_score > weak.avg_score


def test_autonomy_blocks_weak_agent(seeded):
    tiers = {d.agent_id: d.tier for d in autonomy.calibrate_fleet(TaskType.RAG_QA, audit=False)}
    assert tiers["rag_weak"] == AutonomyTier.BLOCKED
    assert tiers["rag_strong"] == AutonomyTier.FULL_AUTO


def test_router_avoids_ineligible_agent(seeded):
    decision = router.route_next_task(TaskType.RAG_QA, audit=False)
    assert decision.chosen_agent in {"rag_strong", "rag_cheap"}
    assert decision.chosen_agent != "rag_weak"


def test_policy_denies_blocked_agent(seeded):
    decision = policy.decide_for_agent("rag_weak", TaskType.RAG_QA, policy.TaskRisk.HIGH)
    assert decision is not None
    assert decision.allow is False


def test_policy_high_risk_requires_approval_for_trusted_agent(seeded):
    decision = policy.decide_for_agent("rag_strong", TaskType.RAG_QA, policy.TaskRisk.HIGH)
    assert decision.allow is True
    assert decision.requires_approval is True


def test_review_fallback_without_llm(seeded):
    review = reviews.generate_review("rag_strong", TaskType.RAG_QA, use_llm=False)
    assert review is not None
    assert review.agent_id == "rag_strong"
    assert len(review.narrative) > 0


def test_review_none_for_unknown_agent(seeded):
    assert reviews.generate_review("does_not_exist", use_llm=False) is None
