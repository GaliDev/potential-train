from eval_harness.fleet.configs import build_fleet, get_fleet
from eval_harness.fleet.generator import _build_user_prompt, make_task
from eval_harness.schemas import TaskType


def test_fleet_has_six_agents_two_task_types():
    fleet = build_fleet()
    assert len(fleet) == 6
    assert len(get_fleet(TaskType.RAG_QA)) == 3
    assert len(get_fleet(TaskType.SUMMARIZATION)) == 3


def test_agent_ids_unique():
    ids = [a.agent_id for a in build_fleet()]
    assert len(ids) == len(set(ids))


def test_rag_prompt_includes_context_and_question():
    item = make_task(TaskType.RAG_QA, "What is X?", context="X is 5.")
    prompt = _build_user_prompt(item)
    assert "X is 5." in prompt
    assert "What is X?" in prompt


def test_summarization_prompt_uses_source():
    item = make_task(TaskType.SUMMARIZATION, "Summarize this.", context="Long source text.")
    prompt = _build_user_prompt(item)
    assert "Long source text." in prompt
