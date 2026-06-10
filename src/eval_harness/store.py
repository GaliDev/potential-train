"""Compatibility facade over the pluggable storage backend.

Legacy callers import from here; the active backend is resolved via
``eval_harness.storage.get_store()`` (default: ``SqliteStore``).
"""

from __future__ import annotations

from datetime import datetime

from .schemas import AgentProfile, AggregateResult, ExecutionTrace, TaskType
from .storage import get_store
from .storage.sqlite_store import AgentRow, AuditRow, EvalRow, RunSignalRow

__all__ = [
    "AgentRow",
    "AuditRow",
    "EvalRow",
    "RunSignalRow",
    "init_db",
    "reset_db",
    "upsert_agent",
    "list_agents",
    "record_eval",
    "fetch_evals",
    "record_run_signal",
    "fetch_run_signals",
    "log_audit",
    "fetch_audit",
]


def init_db() -> None:
    """Create tables if they do not exist."""
    get_store().setup()


def reset_db() -> None:
    """Drop and recreate all tables. Intended for tests."""
    get_store().reset()


def upsert_agent(profile: AgentProfile) -> None:
    get_store().upsert_agent(profile)


def list_agents(task_type: TaskType | None = None) -> list[AgentProfile]:
    return get_store().list_agents(task_type)


def record_eval(result: AggregateResult) -> int:
    return get_store().record_eval(result)


def fetch_evals(
    agent_id: str | None = None,
    task_type: TaskType | None = None,
    judge_mode: str | None = None,
) -> list[EvalRow]:
    return get_store().fetch_evals(agent_id, task_type, judge_mode)


def record_run_signal(
    *,
    item_id: str,
    agent_id: str,
    task_type: TaskType,
    trace: ExecutionTrace,
    created_at: datetime | None = None,
) -> int:
    return get_store().record_run_signal(
        item_id=item_id,
        agent_id=agent_id,
        task_type=task_type,
        trace=trace,
        created_at=created_at,
    )


def fetch_run_signals(
    agent_id: str | None = None,
    task_type: TaskType | None = None,
) -> list[RunSignalRow]:
    return get_store().fetch_run_signals(agent_id, task_type)


def log_audit(
    action: str,
    subject: str,
    detail: dict | None = None,
    actor: str = "system",
) -> None:
    get_store().log_audit(action, subject, detail, actor)


def fetch_audit(limit: int = 200) -> list[AuditRow]:
    return get_store().fetch_audit(limit)
