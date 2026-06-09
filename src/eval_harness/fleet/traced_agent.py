"""Shared helpers for LangGraph agents that emit ExecutionTrace telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import CallStats, LLMClient, get_client
from ..schemas import ExecutionTrace, TestItem


@dataclass
class TraceAccumulator:
    """Mutable trace builder used inside LangGraph agent nodes."""

    steps: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    error: str | None = None
    refused: bool = False
    groundedness: float | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    safety_flag: bool = False
    success: bool = True
    model: str = ""

    def step(self) -> None:
        self.steps += 1

    def record_tool(self, *, success: bool = True) -> None:
        self.tool_calls += 1
        if not success:
            self.tool_failures += 1

    def add_stats(self, stats: CallStats) -> None:
        self.latency_s += stats.latency_s
        self.prompt_tokens += stats.prompt_tokens
        self.completion_tokens += stats.completion_tokens
        self.cost_usd += stats.cost_usd
        if not self.model:
            self.model = stats.model

    def to_trace(self) -> ExecutionTrace:
        return ExecutionTrace(
            steps=max(self.steps, 1),
            tool_calls=self.tool_calls,
            tool_failures=self.tool_failures,
            retries=self.retries,
            error=self.error,
            refused=self.refused,
            groundedness=self.groundedness,
            latency_s=self.latency_s,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
            safety_flag=self.safety_flag,
            success=self.success,
            model=self.model,
        )


class TracedLangGraphAgent:
    """Base for contributor LangGraph agents with optional LLM client injection."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def _llm_text(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        acc: TraceAccumulator,
    ) -> str:
        acc.step()
        text, stats = self.client.complete_text(
            system=system, user=user, model=model, temperature=temperature,
        )
        acc.add_stats(stats)
        return text.strip()
