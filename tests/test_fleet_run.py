from evaluation import fleet_run
from eval_harness.schemas import TaskType, TestItem


def _item(item_id: str, task_type: TaskType) -> TestItem:
    return TestItem(
        id=item_id,
        task_type=task_type,
        task_prompt="Do the task.",
        context="Source material.",
    )


def test_build_real_fleet_items_limits_each_task_type(monkeypatch):
    tasks = [
        _item("rag-1", TaskType.RAG_QA),
        _item("rag-2", TaskType.RAG_QA),
        _item("sum-1", TaskType.SUMMARIZATION),
        _item("sum-2", TaskType.SUMMARIZATION),
    ]

    def fake_generate_for_fleet(selected, task_type):
        return [
            task.model_copy(
                update={
                    "id": f"{task.id}::{task_type.value}",
                    "agent_id": f"{task_type.value}_agent",
                    "candidate_output": "generated",
                }
            )
            for task in selected
        ]

    monkeypatch.setattr(fleet_run, "load_gold", lambda task_type=None: tasks)
    monkeypatch.setattr(fleet_run, "generate_for_fleet", fake_generate_for_fleet)

    produced = fleet_run.build_real_fleet_items(limit_per_type=1)

    assert [item.id for item in produced] == [
        "rag-1::rag_qa",
        "sum-1::summarization",
    ]
    assert all(item.agent_id is not None for item in produced)
    assert all(item.candidate_output == "generated" for item in produced)


def test_build_real_fleet_items_rejects_zero_limit():
    try:
        fleet_run.build_real_fleet_items(limit_per_type=0)
    except ValueError as exc:
        assert "limit_per_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")
