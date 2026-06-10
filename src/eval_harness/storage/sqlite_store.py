"""SQLite-backed implementation of the governance storage interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlmodel import Field, Session, SQLModel, select

from ..config import PROJECT_ROOT, settings
from ..schemas import AgentProfile, AggregateResult, ExecutionTrace, TaskType, utcnow


class AgentRow(SQLModel, table=True):
    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    agent_id: str = Field(primary_key=True)
    name: str
    task_type: str
    model: str
    prompt_variant: str = "default"
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class EvalRow(SQLModel, table=True):
    __tablename__ = "evals"
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    item_id: str = Field(index=True)
    agent_id: str | None = Field(default=None, index=True)
    task_type: str = Field(index=True)
    judge_mode: str = "panel"
    aggregate_score: float = 0.0
    overall_pass: bool = False
    gold_score: float | None = None
    gold_pass: bool | None = None
    verdicts_json: str = "[]"
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    created_at: datetime = Field(default_factory=utcnow, index=True)


class RunSignalRow(SQLModel, table=True):
    """Per-execution operational telemetry (uptime, latency, tools, drift inputs)."""

    __tablename__ = "run_signals"
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    item_id: str = Field(index=True)
    agent_id: str = Field(index=True)
    task_type: str = Field(index=True)
    steps: int = 1
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
    created_at: datetime = Field(default_factory=utcnow, index=True)


class AuditRow(SQLModel, table=True):
    __tablename__ = "audit_log"
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    actor: str = "system"
    action: str = ""
    subject: str = ""
    detail_json: str = "{}"


def _resolve_sqlite_path(database_url: str) -> str:
    """Make relative sqlite paths absolute against the project root and mkdir."""
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        raw = database_url[len(prefix):]
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{p}"
    return database_url


class SqliteStore:
    """SQLite performance store via SQLModel."""

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or settings.database_url
        self._engine = create_engine(_resolve_sqlite_path(url), echo=False)

    def _session(self) -> Session:
        return Session(self._engine)

    def setup(self) -> None:
        """Create tables if they do not exist."""
        SQLModel.metadata.create_all(self._engine)

    def reset(self) -> None:
        """Drop and recreate all tables. Intended for tests."""
        SQLModel.metadata.drop_all(self._engine)
        SQLModel.metadata.create_all(self._engine)

    def upsert_agent(self, profile: AgentProfile) -> None:
        self.setup()
        with self._session() as session:
            row = session.get(AgentRow, profile.agent_id)
            if row is None:
                row = AgentRow(
                    agent_id=profile.agent_id,
                    name=profile.name,
                    task_type=profile.task_type.value,
                    model=profile.model,
                    prompt_variant=profile.prompt_variant,
                    description=profile.description,
                )
            else:
                row.name = profile.name
                row.task_type = profile.task_type.value
                row.model = profile.model
                row.prompt_variant = profile.prompt_variant
                row.description = profile.description
            session.add(row)
            session.commit()

    def list_agents(self, task_type: TaskType | None = None) -> list[AgentProfile]:
        self.setup()
        with self._session() as session:
            stmt = select(AgentRow)
            if task_type is not None:
                stmt = stmt.where(AgentRow.task_type == task_type.value)
            rows = session.exec(stmt).all()
        return [
            AgentProfile(
                agent_id=r.agent_id,
                name=r.name,
                task_type=TaskType(r.task_type),
                model=r.model,
                prompt_variant=r.prompt_variant,
                description=r.description,
            )
            for r in rows
        ]

    def record_eval(self, result: AggregateResult) -> int:
        """Persist one aggregate evaluation result; returns the new row id."""
        self.setup()
        row = EvalRow(
            item_id=result.item_id,
            agent_id=result.agent_id,
            task_type=result.task_type.value,
            judge_mode=result.judge_mode,
            aggregate_score=result.aggregate_score,
            overall_pass=result.overall_pass,
            verdicts_json=json.dumps([v.model_dump(mode="json") for v in result.verdicts]),
            total_cost_usd=result.total_cost_usd,
            total_latency_s=result.total_latency_s,
            created_at=result.created_at,
        )
        with self._session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id or -1

    def fetch_evals(
        self,
        agent_id: str | None = None,
        task_type: TaskType | None = None,
        judge_mode: str | None = None,
    ) -> list[EvalRow]:
        self.setup()
        with self._session() as session:
            stmt = select(EvalRow)
            if agent_id is not None:
                stmt = stmt.where(EvalRow.agent_id == agent_id)
            if task_type is not None:
                stmt = stmt.where(EvalRow.task_type == task_type.value)
            if judge_mode is not None:
                stmt = stmt.where(EvalRow.judge_mode == judge_mode)
            return list(session.exec(stmt).all())

    def record_run_signal(
        self,
        *,
        item_id: str,
        agent_id: str,
        task_type: TaskType,
        trace: ExecutionTrace,
        created_at: datetime | None = None,
    ) -> int:
        """Persist one operational execution trace; returns the new row id."""
        self.setup()
        row = RunSignalRow(
            item_id=item_id,
            agent_id=agent_id,
            task_type=task_type.value,
            steps=trace.steps,
            tool_calls=trace.tool_calls,
            tool_failures=trace.tool_failures,
            retries=trace.retries,
            error=trace.error,
            refused=trace.refused,
            groundedness=trace.groundedness,
            latency_s=trace.latency_s,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            cost_usd=trace.cost_usd,
            safety_flag=trace.safety_flag,
            success=trace.success,
            model=trace.model,
            created_at=created_at or trace.created_at,
        )
        with self._session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id or -1

    def fetch_run_signals(
        self,
        agent_id: str | None = None,
        task_type: TaskType | None = None,
    ) -> list[RunSignalRow]:
        self.setup()
        with self._session() as session:
            stmt = select(RunSignalRow)
            if agent_id is not None:
                stmt = stmt.where(RunSignalRow.agent_id == agent_id)
            if task_type is not None:
                stmt = stmt.where(RunSignalRow.task_type == task_type.value)
            stmt = stmt.order_by(RunSignalRow.created_at)
            return list(session.exec(stmt).all())

    def log_audit(
        self,
        action: str,
        subject: str,
        detail: dict | None = None,
        actor: str = "system",
    ) -> None:
        self.setup()
        row = AuditRow(
            actor=actor,
            action=action,
            subject=subject,
            detail_json=json.dumps(detail or {}, default=str),
        )
        with self._session() as session:
            session.add(row)
            session.commit()

    def fetch_audit(self, limit: int = 200) -> list[AuditRow]:
        self.setup()
        with self._session() as session:
            stmt = select(AuditRow).order_by(AuditRow.ts.desc()).limit(limit)
            return list(session.exec(stmt).all())
