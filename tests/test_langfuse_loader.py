"""Tests for the Langfuse trace -> TestItem mapping and score push-back.

Everything here is offline: the HTTP client is never constructed; score
push-back is exercised against a fake client that records calls.
"""

from __future__ import annotations

from eval_harness.datasets.langfuse_loader import (
    _as_text,
    infer_task_type,
    push_result_scores,
    trace_id_for_item_id,
    trace_to_test_item,
)
from eval_harness.schemas import (
    AggregateResult,
    Criterion,
    JudgeVerdict,
    TaskType,
)


def make_trace(**overrides) -> dict:
    trace = {
        "id": "tr-123",
        "name": "rag-pipeline",
        "input": "What is the capital of France?",
        "output": "Paris is the capital of France.",
        "tags": [],
        "metadata": {},
    }
    trace.update(overrides)
    return trace


# --- _as_text -----------------------------------------------------------


def test_as_text_passes_strings_through():
    assert _as_text("  hello ") == "hello"


def test_as_text_takes_last_chat_message_content():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Translate this."},
    ]
    assert _as_text(messages) == "Translate this."


def test_as_text_unwraps_common_dict_keys():
    assert _as_text({"content": "the answer"}) == "the answer"
    assert _as_text({"messages": [{"role": "user", "content": "hi"}]}) == "hi"


def test_as_text_falls_back_to_json_for_unknown_dicts():
    assert '"foo"' in _as_text({"foo": "bar"})


# --- task type inference --------------------------------------------------


def test_infer_task_type_from_metadata_wins():
    trace = make_trace(metadata={"task_type": "translation"}, tags=["summarization"])
    assert infer_task_type(trace) == TaskType.TRANSLATION


def test_infer_task_type_from_tags():
    assert infer_task_type(make_trace(tags=["prod", "summarize"])) == TaskType.SUMMARIZATION


def test_infer_task_type_from_name_substring():
    assert infer_task_type(make_trace(name="nightly-translation-run")) == TaskType.TRANSLATION


def test_infer_task_type_default():
    trace = make_trace(name="misc", tags=[])
    assert infer_task_type(trace, default=TaskType.SUMMARIZATION) == TaskType.SUMMARIZATION


# --- trace_to_test_item -----------------------------------------------------


def test_trace_to_test_item_maps_core_fields():
    item = trace_to_test_item(make_trace())
    assert item is not None
    assert item.id == "lf_tr-123"
    assert item.task_prompt.startswith("What is the capital")
    assert item.candidate_output.startswith("Paris")
    assert item.agent_id == "rag-pipeline"
    assert trace_id_for_item_id(item.id) == "tr-123"


def test_trace_to_test_item_requires_input_and_output():
    assert trace_to_test_item(make_trace(output=None)) is None
    assert trace_to_test_item(make_trace(input="")) is None


def test_trace_to_test_item_context_from_metadata():
    trace = make_trace(metadata={"context": "France is a country in Europe."})
    item = trace_to_test_item(trace)
    assert item.context == "France is a country in Europe."


def test_trace_to_test_item_context_from_retrieval_observation():
    trace = make_trace(
        observations=[
            {"name": "vector-retriever", "output": "doc: Paris facts"},
            {"name": "llm-call", "output": "ignored"},
        ]
    )
    item = trace_to_test_item(trace)
    assert item.context == "doc: Paris facts"


def test_trace_to_test_item_agent_id_from_metadata():
    item = trace_to_test_item(make_trace(metadata={"agent_id": "agent-7"}))
    assert item.agent_id == "agent-7"


# --- push_result_scores ------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_score(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"id": f"score-{len(self.calls)}"}


def make_result(item_id: str = "lf_tr-123") -> AggregateResult:
    verdicts = [
        JudgeVerdict(criterion=c, score=4, passed=True, rationale=f"{c.value} ok")
        for c in Criterion
    ]
    return AggregateResult(
        item_id=item_id,
        task_type=TaskType.RAG_QA,
        verdicts=verdicts,
        aggregate_score=4.0,
        overall_pass=True,
        rationale="solid",
    )


def test_push_result_scores_pushes_criteria_aggregate_and_pass():
    client = FakeClient()
    created = push_result_scores(client, make_result())
    # 5 criteria + aggregate + boolean pass
    assert created == 7
    names = [c["name"] for c in client.calls]
    assert "judge_correctness" in names
    assert "judge_safety" in names
    assert "judge_aggregate" in names
    assert "judge_overall_pass" in names
    assert all(c["trace_id"] == "tr-123" for c in client.calls)
    pass_call = next(c for c in client.calls if c["name"] == "judge_overall_pass")
    assert pass_call["data_type"] == "BOOLEAN"
    assert pass_call["value"] == 1.0


def test_push_result_scores_skips_non_langfuse_items():
    client = FakeClient()
    created = push_result_scores(client, make_result(item_id="gold_1"))
    assert created == 0
    assert client.calls == []
