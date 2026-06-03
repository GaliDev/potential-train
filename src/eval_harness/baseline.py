"""Baseline judge: a single GPT-4o call with a simple prompt.

This is the simplest possible evaluator and the bar the multiagent panel must
beat. Per the project methodology we establish this baseline first, measure its
agreement with human gold labels, and only then add complexity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .agents.base import PASS_THRESHOLD, clamp_score, render_item_for_judge
from .config import settings
from .llm import LLMClient, get_client
from .schemas import MAX_SCORE, MIN_SCORE, AggregateResult, TestItem

_SYSTEM = (
    "You are a strict evaluator of AI assistant outputs. Given a task, optional "
    "context or reference, and the assistant's response, judge the OVERALL quality "
    f"on an integer scale from {MIN_SCORE} (poor) to {MAX_SCORE} (excellent). "
    "Consider correctness, grounding in any provided context, completeness, and "
    "clarity. Decide whether the response is acceptable to ship, and give a brief "
    "rationale."
)


class BaselineJudgement(BaseModel):
    """Structured output for the single-call baseline."""

    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE, description="Overall quality, 1-5")
    passed: bool = Field(description="Whether the response is acceptable to ship")
    rationale: str = Field(description="One or two sentences explaining the score")


class BaselineJudge:
    """Single-model, single-prompt evaluator."""

    judge_mode = "baseline"

    def __init__(self, client: LLMClient | None = None, model: str | None = None) -> None:
        self._client = client or get_client()
        self._model = model or settings.judge_model

    def judge(self, item: TestItem) -> AggregateResult:
        judgement, stats = self._client.complete_structured(
            system=_SYSTEM,
            user=render_item_for_judge(item),
            response_model=BaselineJudgement,
            model=self._model,
        )
        score = clamp_score(judgement.score)
        return AggregateResult(
            item_id=item.id,
            agent_id=item.agent_id,
            task_type=item.task_type,
            verdicts=[],
            aggregate_score=float(score),
            overall_pass=judgement.passed if judgement.passed is not None else score >= PASS_THRESHOLD,
            rationale=judgement.rationale,
            judge_mode=self.judge_mode,
            total_latency_s=stats.latency_s,
            total_cost_usd=stats.cost_usd,
        )
