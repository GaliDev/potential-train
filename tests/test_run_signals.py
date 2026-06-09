"""Tests for operational run signal persistence."""

from __future__ import annotations

from eval_harness.schemas import ExecutionTrace, TaskType
from eval_harness.store import fetch_run_signals, record_run_signal


def test_record_and_fetch_run_signal():
    trace = ExecutionTrace(
        steps=3,
        tool_calls=2,
        tool_failures=0,
        retries=1,
        groundedness=0.85,
        latency_s=1.5,
        prompt_tokens=100,
        completion_tokens=50,
        success=True,
        model="gpt-4o-mini",
    )
    row_id = record_run_signal(
        item_id="task1::rag_strong",
        agent_id="rag_strong",
        task_type=TaskType.RAG_QA,
        trace=trace,
    )
    assert row_id > 0
    rows = fetch_run_signals(agent_id="rag_strong")
    assert len(rows) == 1
    assert rows[0].tool_calls == 2
    assert rows[0].success is True


def test_failed_execution_signal():
    trace = ExecutionTrace(model="gpt-4o", success=False, error="timeout")
    record_run_signal(
        item_id="task2::rag_weak",
        agent_id="rag_weak",
        task_type=TaskType.RAG_QA,
        trace=trace,
    )
    rows = fetch_run_signals(agent_id="rag_weak")
    assert rows[0].success is False
    assert rows[0].error == "timeout"
