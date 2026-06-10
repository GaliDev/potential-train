"""Judge prompts for open instruct models.

Mirrors the OpenAI panel's per-criterion system prompt (same criterion guides,
same 1-5 scale, same pass threshold) but asks for a strict JSON object instead
of relying on OpenAI's structured-output parsing, which open models lack.
The item rendering is reused verbatim so every judge sees identical inputs.
"""

from __future__ import annotations

from eval_harness.agents.base import PASS_THRESHOLD, render_item_for_judge
from eval_harness.agents.criteria import CRITERION_GUIDES
from eval_harness.schemas import MAX_SCORE, MIN_SCORE, Criterion, TestItem

JSON_FORMAT_INSTRUCTIONS = (
    'Respond with ONLY a single JSON object, no other text, in this exact shape:\n'
    '{"score": <integer 1-5>, "passed": <true|false>, '
    '"rationale": "<one or two sentences>", "evidence": ["<short quote>", ...]}'
)


def criterion_brief(criterion: Criterion) -> str:
    """The judge instruction without output-format directions.

    Used directly for providers with native structured output (Claude);
    `system_prompt` adds the strict-JSON instructions for everyone else.
    """
    return (
        f"You are an expert evaluator assessing ONE quality dimension of an AI "
        f"assistant's response.\n\n{CRITERION_GUIDES[criterion]}\n\n"
        f"Score only this dimension on an integer scale from {MIN_SCORE} (poor) to "
        f"{MAX_SCORE} (excellent). Set passed=true only if the dimension is "
        f"acceptable (score >= {PASS_THRESHOLD}). Give a brief rationale and up to "
        f"three short supporting quotes in evidence."
    )


def system_prompt(criterion: Criterion, suffix: str = "") -> str:
    return f"{criterion_brief(criterion)}\n\n{JSON_FORMAT_INSTRUCTIONS}{suffix}"


def user_prompt(item: TestItem) -> str:
    return render_item_for_judge(item)
