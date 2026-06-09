from eval_harness.schemas import (
    AgentProfile,
    AggregateResult,
    Criterion,
    ExecutionTrace,
    JudgeVerdict,
    TaskType,
)
from eval_harness.store import (
    fetch_audit,
    fetch_evals,
    fetch_run_signals,
    list_agents,
    log_audit,
    record_eval,
    record_run_signal,
    upsert_agent,
)


def _agent(agent_id="a1", task=TaskType.RAG_QA):
    return AgentProfile(agent_id=agent_id, name="A", task_type=task, model="gpt-4o")


def test_upsert_and_list_agents():
    upsert_agent(_agent("a1"))
    upsert_agent(_agent("a2", TaskType.SUMMARIZATION))
    assert {a.agent_id for a in list_agents()} == {"a1", "a2"}
    assert [a.agent_id for a in list_agents(TaskType.RAG_QA)] == ["a1"]


def test_upsert_is_idempotent_update():
    upsert_agent(_agent("a1"))
    upsert_agent(AgentProfile(agent_id="a1", name="renamed", task_type=TaskType.RAG_QA, model="gpt-4o-mini"))
    agents = list_agents()
    assert len(agents) == 1
    assert agents[0].name == "renamed"
    assert agents[0].model == "gpt-4o-mini"


def test_record_and_fetch_evals():
    result = AggregateResult(
        item_id="i1", agent_id="a1", task_type=TaskType.RAG_QA,
        verdicts=[JudgeVerdict(criterion=Criterion.CORRECTNESS, score=4, passed=True, rationale="r")],
        aggregate_score=4.0, overall_pass=True, judge_mode="panel",
        total_latency_s=1.0, total_cost_usd=0.002,
    )
    row_id = record_eval(result)
    assert row_id > 0
    rows = fetch_evals(agent_id="a1")
    assert len(rows) == 1
    assert rows[0].overall_pass is True
    assert fetch_evals(judge_mode="baseline") == []


def test_audit_log():
    log_audit("test_action", "subject1", {"k": "v"})
    entries = fetch_audit()
    assert len(entries) == 1
    assert entries[0].action == "test_action"
