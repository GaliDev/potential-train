"""Core domain models shared across the platform.

These are plain Pydantic models (DTOs). Persistence-layer tables live in
`store.py`; governance decision models live alongside their logic in the
`governance/` package where helpful, but the shared primitives are here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskType(str, Enum):
    """Task categories the managed fleet performs."""

    RAG_QA = "rag_qa"
    SUMMARIZATION = "summarization"


class Criterion(str, Enum):
    """Quality dimensions scored by the judge panel."""

    CORRECTNESS = "correctness"
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    SAFETY = "safety"


class AutonomyTier(str, Enum):
    """How much independence an agent has earned, worst -> best."""

    BLOCKED = "blocked"
    HUMAN_IN_LOOP = "human_in_loop"
    AUTO_SPOT_CHECK = "auto_spot_check"
    FULL_AUTO = "full_auto"


# Score range used everywhere for per-criterion scoring.
MIN_SCORE = 1
MAX_SCORE = 5


class TestItem(BaseModel):
    """A single unit of work that was (or will be) evaluated.

    `candidate_output` is what an agent produced for `task_prompt`. `reference`
    and `context` are optional grounding material (e.g. retrieved docs or a
    reference summary) used by the faithfulness/correctness judges.
    """

    id: str
    task_type: TaskType
    task_prompt: str
    candidate_output: str = ""
    agent_id: str | None = None
    context: str | None = None
    reference: str | None = None
    # Optional human gold labels for benchmarking the judge itself.
    gold_score: float | None = None
    gold_pass: bool | None = None


class JudgeVerdict(BaseModel):
    """One judge's assessment of one criterion."""

    criterion: Criterion
    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE)
    passed: bool
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    # Telemetry (filled in by the runner, not the LLM).
    model: str | None = None
    latency_s: float | None = None
    cost_usd: float | None = None


class AggregateResult(BaseModel):
    """The meta-judge's combined verdict for a single test item."""

    item_id: str
    agent_id: str | None = None
    task_type: TaskType
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    aggregate_score: float
    overall_pass: bool
    rationale: str = ""
    # Which evaluation path produced this ("baseline" or "panel").
    judge_mode: str = "panel"
    total_latency_s: float = 0.0
    total_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)

    def score_for(self, criterion: Criterion) -> int | None:
        for v in self.verdicts:
            if v.criterion == criterion:
                return v.score
        return None


class AgentProfile(BaseModel):
    """A member of the managed fleet."""

    agent_id: str
    name: str
    task_type: TaskType
    model: str
    prompt_variant: str = "default"
    description: str = ""
