from eval_harness.agents.aggregator import aggregate
from eval_harness.schemas import Criterion, JudgeVerdict, TaskType


def _verdict(crit, score):
    return JudgeVerdict(criterion=crit, score=score, passed=score >= 4, rationale="r",
                        latency_s=0.1, cost_usd=0.001)


def test_weighted_score_and_pass():
    verdicts = [
        _verdict(Criterion.CORRECTNESS, 5),
        _verdict(Criterion.FAITHFULNESS, 5),
        _verdict(Criterion.COMPLETENESS, 5),
        _verdict(Criterion.COHERENCE, 5),
        _verdict(Criterion.SAFETY, 5),
    ]
    res = aggregate("i1", TaskType.RAG_QA, verdicts)
    assert res.aggregate_score == 5.0
    assert res.overall_pass is True
    assert res.total_cost_usd == round(5 * 0.001, 6)
    assert res.total_latency_s == round(5 * 0.1, 4)


def test_safety_gate_forces_fail():
    verdicts = [
        _verdict(Criterion.CORRECTNESS, 5),
        _verdict(Criterion.FAITHFULNESS, 5),
        _verdict(Criterion.COMPLETENESS, 5),
        _verdict(Criterion.COHERENCE, 5),
        JudgeVerdict(criterion=Criterion.SAFETY, score=1, passed=False, rationale="unsafe"),
    ]
    res = aggregate("i1", TaskType.RAG_QA, verdicts)
    assert res.aggregate_score == 5.0  # quality is high
    assert res.overall_pass is False   # but safety gate blocks


def test_low_quality_fails():
    verdicts = [
        _verdict(Criterion.CORRECTNESS, 2),
        _verdict(Criterion.FAITHFULNESS, 2),
        _verdict(Criterion.COMPLETENESS, 3),
        _verdict(Criterion.COHERENCE, 3),
        _verdict(Criterion.SAFETY, 5),
    ]
    res = aggregate("i1", TaskType.RAG_QA, verdicts)
    assert res.aggregate_score < 4
    assert res.overall_pass is False
