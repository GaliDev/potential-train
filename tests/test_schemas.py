from eval_harness.schemas import (
    AggregateResult,
    Criterion,
    JudgeVerdict,
    TaskType,
    TestItem,
)


def test_score_for_returns_matching_criterion():
    result = AggregateResult(
        item_id="i1",
        task_type=TaskType.RAG_QA,
        verdicts=[
            JudgeVerdict(criterion=Criterion.CORRECTNESS, score=4, passed=True, rationale="r"),
            JudgeVerdict(criterion=Criterion.SAFETY, score=5, passed=True, rationale="r"),
        ],
        aggregate_score=4.0,
        overall_pass=True,
    )
    assert result.score_for(Criterion.CORRECTNESS) == 4
    assert result.score_for(Criterion.SAFETY) == 5
    assert result.score_for(Criterion.COHERENCE) is None


def test_test_item_defaults():
    item = TestItem(id="x", task_type=TaskType.SUMMARIZATION, task_prompt="sum")
    assert item.candidate_output == ""
    assert item.gold_pass is None
