"""Run fleet agents over tasks to produce the candidate outputs we evaluate.

Given a set of tasks (a question + optional context for RAG, a source document
for summarization, or source text for translation), each relevant fleet agent
generates an output.
Those outputs become `TestItem`s that flow into the judge panel.
"""

from __future__ import annotations

import uuid

from ..llm import CallStats, LLMClient, get_client
from ..schemas import TaskType, TestItem
from ..store import upsert_agent
from .configs import FleetAgent, get_fleet


def _build_user_prompt(item: TestItem) -> str:
    """Render the task for an agent, including grounding context when present."""
    if item.task_type == TaskType.RAG_QA:
        ctx = item.context or "(no context provided)"
        return f"Context:\n{ctx}\n\nQuestion: {item.task_prompt}"
    if item.task_type == TaskType.TRANSLATION:
        # task_prompt is the translation instruction; context is the source text.
        source = item.context or item.task_prompt
        instruction = item.task_prompt if item.context else "Translate the following text."
        return f"{instruction}\n\nText to translate:\n{source}"
    # Summarization: task_prompt is the instruction, context is the source doc.
    source = item.context or item.task_prompt
    instruction = item.task_prompt if item.context else "Summarize the following source."
    return f"{instruction}\n\nSource:\n{source}"


def generate_output(
    agent: FleetAgent,
    task: TestItem,
    client: LLMClient | None = None,
) -> tuple[TestItem, CallStats]:
    """Produce one agent's output for a task, returned as a new TestItem."""
    client = client or get_client()
    text, stats = client.complete_text(
        system=agent.system_prompt,
        user=_build_user_prompt(task),
        model=agent.model,
        temperature=agent.temperature,
    )
    produced = task.model_copy(
        update={
            "id": f"{task.id}::{agent.agent_id}",
            "agent_id": agent.agent_id,
            "candidate_output": text.strip(),
        }
    )
    return produced, stats


def generate_for_fleet(
    tasks: list[TestItem],
    task_type: TaskType,
    register: bool = True,
    client: LLMClient | None = None,
) -> list[TestItem]:
    """Generate outputs from every agent of `task_type` for each task.

    Returns a flat list of produced TestItems (one per agent per task). When
    `register` is True, the fleet agents are also upserted into the store so
    the governance layer can reference them.
    """
    client = client or get_client()
    agents = get_fleet(task_type)
    if register:
        for a in agents:
            upsert_agent(a.profile)

    produced: list[TestItem] = []
    for task in tasks:
        for agent in agents:
            item, _stats = generate_output(agent, task, client=client)
            produced.append(item)
    return produced


def make_task(
    task_type: TaskType,
    prompt: str,
    context: str | None = None,
    reference: str | None = None,
    task_id: str | None = None,
) -> TestItem:
    """Convenience constructor for an input task (no candidate output yet)."""
    return TestItem(
        id=task_id or uuid.uuid4().hex[:12],
        task_type=task_type,
        task_prompt=prompt,
        context=context,
        reference=reference,
    )
