"""Shared helpers for judges (baseline and panel).

Keeps prompt rendering and the pass threshold in one place so the baseline
and the multiagent panel judge identical inputs on a consistent scale.
"""

from __future__ import annotations

from ..schemas import MAX_SCORE, MIN_SCORE, TaskType, TestItem

# A score at or above this (on the 1-5 scale) counts as a "pass".
PASS_THRESHOLD = 4


def render_item_for_judge(item: TestItem) -> str:
    """Render a test item into the text a judge sees.

    Includes the task, any grounding context/reference, and the candidate
    output being judged.
    """
    parts: list[str] = []
    label = "Question" if item.task_type == TaskType.RAG_QA else "Task"
    parts.append(f"{label}:\n{item.task_prompt}")

    if item.context:
        ctx_label = "Context (source of truth)" if item.task_type == TaskType.RAG_QA else "Source document"
        parts.append(f"{ctx_label}:\n{item.context}")

    if item.reference:
        parts.append(f"Reference answer:\n{item.reference}")

    parts.append(f"Candidate response to evaluate:\n{item.candidate_output}")
    return "\n\n".join(parts)


def clamp_score(score: int) -> int:
    return max(MIN_SCORE, min(MAX_SCORE, score))
