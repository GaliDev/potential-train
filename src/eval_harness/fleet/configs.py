"""The managed fleet under evaluation.

Two task types (RAG Q&A and summarization), each served by three competing
agent configs: a strong model, a cheaper model, and a deliberately weaker
prompt variant. The performance gaps are intentional - they make routing,
autonomy tiers, and performance reviews show clear, demo-able differences.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..schemas import AgentProfile, TaskType

# --- System prompts -------------------------------------------------------

_RAG_STRONG = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context. Paraphrase or quote the relevant facts. "
    "If the answer is not contained in the context, say you do not know. "
    "Be concise, accurate, and grounded."
)

_RAG_WEAK = (
    "Answer the question quickly from your general knowledge. You do not need to "
    "rely on the provided context - just give a short, confident answer."
)

_SUM_STRONG = (
    "You are an expert summarizer. Produce a faithful, concise summary that "
    "preserves the key points of the source. Do not add information or claims "
    "that are not present in the source."
)

_SUM_WEAK = (
    "Write a very short, catchy one-line summary of the source. Prioritize being "
    "punchy over being complete."
)


@dataclass(frozen=True)
class FleetAgent:
    """An agent config: its profile plus how it generates output."""

    profile: AgentProfile
    system_prompt: str
    temperature: float = 0.2

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    @property
    def task_type(self) -> TaskType:
        return self.profile.task_type

    @property
    def model(self) -> str:
        return self.profile.model


def build_fleet() -> list[FleetAgent]:
    """Construct the six fleet agents from current model settings."""
    strong = settings.judge_model          # gpt-4o
    cheap = settings.judge_model_cheap      # gpt-4o-mini
    return [
        # --- RAG / Q&A ---
        FleetAgent(
            profile=AgentProfile(
                agent_id="rag_strong", name="RAG Pro (gpt-4o)", task_type=TaskType.RAG_QA,
                model=strong, prompt_variant="grounded",
                description="Strong model, grounded prompt. Expected best quality.",
            ),
            system_prompt=_RAG_STRONG, temperature=0.1,
        ),
        FleetAgent(
            profile=AgentProfile(
                agent_id="rag_cheap", name="RAG Lite (gpt-4o-mini)", task_type=TaskType.RAG_QA,
                model=cheap, prompt_variant="grounded",
                description="Cheaper model, same grounded prompt. Cost/quality trade-off.",
            ),
            system_prompt=_RAG_STRONG, temperature=0.1,
        ),
        FleetAgent(
            profile=AgentProfile(
                agent_id="rag_weak", name="RAG Loose (gpt-4o-mini)", task_type=TaskType.RAG_QA,
                model=cheap, prompt_variant="ungrounded",
                description="Cheap model with an ungrounded prompt. Expected to hallucinate.",
            ),
            system_prompt=_RAG_WEAK, temperature=0.9,
        ),
        # --- Summarization ---
        FleetAgent(
            profile=AgentProfile(
                agent_id="sum_strong", name="Summarizer Pro (gpt-4o)", task_type=TaskType.SUMMARIZATION,
                model=strong, prompt_variant="faithful",
                description="Strong model, faithful prompt. Expected best quality.",
            ),
            system_prompt=_SUM_STRONG, temperature=0.2,
        ),
        FleetAgent(
            profile=AgentProfile(
                agent_id="sum_cheap", name="Summarizer Lite (gpt-4o-mini)", task_type=TaskType.SUMMARIZATION,
                model=cheap, prompt_variant="faithful",
                description="Cheaper model, same faithful prompt. Cost/quality trade-off.",
            ),
            system_prompt=_SUM_STRONG, temperature=0.2,
        ),
        FleetAgent(
            profile=AgentProfile(
                agent_id="sum_weak", name="Summarizer Snappy (gpt-4o-mini)", task_type=TaskType.SUMMARIZATION,
                model=cheap, prompt_variant="terse",
                description="Cheap model with a one-liner prompt. Expected to drop key points.",
            ),
            system_prompt=_SUM_WEAK, temperature=0.9,
        ),
    ]


def get_fleet(task_type: TaskType | None = None) -> list[FleetAgent]:
    fleet = build_fleet()
    if task_type is None:
        return fleet
    return [a for a in fleet if a.task_type == task_type]
