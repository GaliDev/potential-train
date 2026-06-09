"""Offline tests for LangGraph fleet agents (no API key)."""

from __future__ import annotations

from eval_harness.fleet.registry import reset_registry
from eval_harness.schemas import TaskType, TestItem


def _rag_task() -> TestItem:
    return TestItem(
        id="t1",
        task_type=TaskType.RAG_QA,
        task_prompt="What is the capital?",
        context="France is a country. Paris is the capital of France.",
    )


def _sum_task() -> TestItem:
    return TestItem(
        id="t2",
        task_type=TaskType.SUMMARIZATION,
        task_prompt="Summarize the source.",
        context="Alpha project launched in 2020. Beta phase started in 2021. Revenue grew 40 percent.",
    )


def _trans_task() -> TestItem:
    return TestItem(
        id="t3",
        task_type=TaskType.TRANSLATION,
        task_prompt="Translate to French.",
        context="Hello world",
    )


def test_rag_react_agent_offline():
    reset_registry()
    from agents.rag_react_agent import RagReactAgent

    agent = RagReactAgent(use_llm=False)
    output, trace = agent.run(_rag_task())
    assert len(output) > 0
    assert "don't know" in output.lower() or "paris" in output.lower() or "france" in output.lower()
    assert trace.steps >= 2
    assert trace.tool_calls >= 1


def test_summarizer_refine_agent_offline():
    reset_registry()
    from agents.summarizer_refine_agent import SummarizerRefineAgent

    agent = SummarizerRefineAgent(use_llm=False)
    output, trace = agent.run(_sum_task())
    assert len(output) > 0
    assert trace.steps >= 1


def test_translator_backcheck_agent_offline():
    reset_registry()
    from agents.translator_backcheck_agent import TranslatorBackcheckAgent

    agent = TranslatorBackcheckAgent(use_llm=False)
    output, trace = agent.run(_trans_task())
    assert len(output) > 0
    assert trace.steps >= 2
