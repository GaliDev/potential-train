from eval_harness.runner import run_evaluation
from eval_harness.schemas import AggregateResult, TaskType, TestItem
from eval_harness.store import fetch_evals


class _StubJudge:
    judge_mode = "panel"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()

    def judge(self, item):
        if item.id in self.fail_on:
            raise ValueError("boom")
        return AggregateResult(item_id=item.id, agent_id=item.agent_id, task_type=item.task_type,
                               aggregate_score=4.0, overall_pass=True, judge_mode="panel",
                               total_latency_s=0.5, total_cost_usd=0.001)


def _items(n):
    return [TestItem(id=f"i{i}", task_type=TaskType.RAG_QA, task_prompt="q",
                     candidate_output="a", agent_id="a1") for i in range(n)]


def test_runner_persists_and_aggregates():
    report = run_evaluation(_items(3), _StubJudge(), persist=True, save_run=False)
    assert report.count == 3
    assert report.pass_rate == 1.0
    assert report.total_cost_usd == 0.003
    assert len(fetch_evals(agent_id="a1")) == 3


def test_runner_captures_errors_without_aborting():
    report = run_evaluation(_items(3), _StubJudge(fail_on={"i1"}), persist=False, save_run=False)
    assert report.count == 2
    assert len(report.errors) == 1
    assert report.errors[0]["item_id"] == "i1"
