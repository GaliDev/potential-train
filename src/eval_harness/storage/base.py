"""Storage interface contracts for the governance and evaluation layers.

Protocols define what the decision core needs (reads + audit) versus what
writers need (persisting agents, evals, run signals). A future Langfuse-backed
store can implement the same interface without touching governance logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..schemas import AgentProfile, AggregateResult, ExecutionTrace, TaskType
from .sqlite_store import AgentRow, AuditRow, EvalRow, RunSignalRow

__all__ = [
    "AgentRow",
    "AuditRow",
    "EvalRow",
    "RunSignalRow",
    "PerformanceSource",
    "AuditLog",
    "ResultSink",
    "GovernanceStore",
]


class PerformanceSource(Protocol):
    """Read path used by profiles, routing, and autonomy."""

    def list_agents(self, task_type: TaskType | None = None) -> list[AgentProfile]: ...

    def fetch_evals(
        self,
        agent_id: str | None = None,
        task_type: TaskType | None = None,
        judge_mode: str | None = None,
    ) -> list[EvalRow]: ...

    def fetch_run_signals(
        self,
        agent_id: str | None = None,
        task_type: TaskType | None = None,
    ) -> list[RunSignalRow]: ...


class AuditLog(Protocol):
    def log_audit(
        self,
        action: str,
        subject: str,
        detail: dict | None = None,
        actor: str = "system",
    ) -> None: ...

    def fetch_audit(self, limit: int = 200) -> list[AuditRow]: ...


class ResultSink(Protocol):
    def upsert_agent(self, profile: AgentProfile) -> None: ...

    def record_eval(self, result: AggregateResult) -> int: ...

    def record_run_signal(
        self,
        *,
        item_id: str,
        agent_id: str,
        task_type: TaskType,
        trace: ExecutionTrace,
        created_at: datetime | None = None,
    ) -> int: ...


class GovernanceStore(PerformanceSource, AuditLog, ResultSink, Protocol):
    """Full storage backend: reads, writes, audit, and lifecycle."""

    def setup(self) -> None: ...

    def reset(self) -> None: ...
