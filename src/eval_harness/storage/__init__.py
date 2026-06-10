"""Pluggable storage backends for evaluation and governance data."""

from __future__ import annotations

from .base import (
    AgentRow,
    AuditLog,
    AuditRow,
    EvalRow,
    GovernanceStore,
    PerformanceSource,
    ResultSink,
    RunSignalRow,
)
from .sqlite_store import SqliteStore

_active_store: GovernanceStore | None = None


def get_store() -> GovernanceStore:
    """Return the active storage backend (default: lazily-built SqliteStore)."""
    global _active_store
    if _active_store is None:
        _active_store = SqliteStore()
    return _active_store


def set_store(store: GovernanceStore) -> None:
    """Replace the active storage backend (e.g. for tests or Langfuse adapter)."""
    global _active_store
    _active_store = store


def reset_store() -> None:
    """Clear the active backend so the next get_store() builds a fresh default."""
    global _active_store
    _active_store = None


__all__ = [
    "AgentRow",
    "AuditLog",
    "AuditRow",
    "EvalRow",
    "GovernanceStore",
    "PerformanceSource",
    "ResultSink",
    "RunSignalRow",
    "SqliteStore",
    "get_store",
    "reset_store",
    "set_store",
]
