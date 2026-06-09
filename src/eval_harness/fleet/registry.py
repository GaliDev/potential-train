"""Pluggable agent registry for local and future hosted agents.

Contributors register agents via ``@register_agent`` or by dropping a module
under ``agents/``. Built-in prompt/model fleet agents are wrapped as adapters
so ``fleet_run`` can evaluate every agent through one interface.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from ..config import PROJECT_ROOT
from ..llm import CallStats
from ..schemas import AgentProfile, ExecutionTrace, TaskType, TestItem
from .configs import FleetAgent, build_fleet
from .generator import generate_output

T = TypeVar("T")


class ManagedAgent(Protocol):
    """Contract for any agent the evaluation platform can run."""

    @property
    def agent_id(self) -> str: ...

    @property
    def profile(self) -> AgentProfile: ...

    def answer(self, task: TestItem) -> str:
        """Return the candidate output for one benchmark task."""

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        """Return output plus operational telemetry for one task."""


def trace_from_call_stats(stats: CallStats, *, steps: int = 1, tool_calls: int = 0) -> ExecutionTrace:
    """Build a minimal trace from a single LLM call (prompt-only fleet agents)."""
    return ExecutionTrace(
        steps=steps,
        tool_calls=tool_calls,
        latency_s=stats.latency_s,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        cost_usd=stats.cost_usd,
        model=stats.model,
        success=True,
    )


class FleetAgentAdapter:
    """Wraps a built-in ``FleetAgent`` (prompt + model) as a ``ManagedAgent``."""

    def __init__(self, fleet_agent: FleetAgent) -> None:
        self._fleet = fleet_agent

    @property
    def agent_id(self) -> str:
        return self._fleet.agent_id

    @property
    def profile(self) -> AgentProfile:
        return self._fleet.profile

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        produced, stats = generate_output(self._fleet, task)
        return produced.candidate_output, trace_from_call_stats(stats)

    def answer(self, task: TestItem) -> str:
        output, _trace = self.run(task)
        return output


_REGISTRY: dict[str, ManagedAgent] = {}
_INITIALIZED = False


def register_agent(obj: T) -> T:
    """Decorator or direct call to add an agent to the registry."""
    if isinstance(obj, type):
        instance = obj()
        _register(instance)
        return obj
    _register(obj)
    return obj


def _register(agent: ManagedAgent) -> None:
    agent_id = agent.agent_id
    if agent_id in _REGISTRY:
        raise ValueError(f"Agent id already registered: {agent_id}")
    _REGISTRY[agent_id] = agent


def profile_from_class(agent: object) -> AgentProfile:
    """Build an ``AgentProfile`` from class/instance attributes."""
    stored = getattr(agent, "__dict__", {}).get("profile")
    if isinstance(stored, AgentProfile):
        return stored
    agent_id = getattr(agent, "agent_id", None)
    if not agent_id:
        raise ValueError(f"Agent {agent!r} must define agent_id or profile")
    task_type = getattr(agent, "task_type", None)
    if task_type is None:
        raise ValueError(f"Agent {agent_id} must define task_type or profile")
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    return AgentProfile(
        agent_id=agent_id,
        name=getattr(agent, "name", agent_id),
        task_type=task_type,
        model=getattr(agent, "model", "custom:unknown"),
        prompt_variant=getattr(agent, "prompt_variant", "default"),
        description=getattr(agent, "description", ""),
    )


class ClassBasedAgent:
    """Mixin-style base for contributor agents defined as plain classes."""

    agent_id: str
    name: str
    task_type: TaskType
    model: str = "custom:unknown"
    prompt_variant: str = "default"
    description: str = ""

    @property
    def profile(self) -> AgentProfile:
        return AgentProfile(
            agent_id=self.agent_id,
            name=self.name,
            task_type=self.task_type,
            model=self.model,
            prompt_variant=self.prompt_variant,
            description=self.description,
        )

    def answer(self, task: TestItem) -> str:
        raise NotImplementedError

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        return self.answer(task), ExecutionTrace(model=self.model, success=True)


def initialize_registry(*, rediscover: bool = False) -> None:
    """Load built-in fleet agents and discover community modules under agents/."""
    global _INITIALIZED
    if _INITIALIZED and not rediscover:
        return
    if rediscover:
        _REGISTRY.clear()

    for fleet_agent in build_fleet():
        adapter = FleetAgentAdapter(fleet_agent)
        if adapter.agent_id not in _REGISTRY:
            _REGISTRY[adapter.agent_id] = adapter

    _discover_community_agents()
    _INITIALIZED = True


def _discover_community_agents(agents_dir: Path | None = None) -> None:
    root = agents_dir or (PROJECT_ROOT / "agents")
    if not root.is_dir():
        return
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"community_agents.{path.stem}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def list_registered_agents(task_type: TaskType | None = None) -> list[ManagedAgent]:
    initialize_registry()
    agents = list(_REGISTRY.values())
    if task_type is not None:
        agents = [a for a in agents if a.profile.task_type == task_type]
    return sorted(agents, key=lambda a: a.agent_id)


def get_registered_agent(agent_id: str) -> ManagedAgent:
    initialize_registry()
    agent = _REGISTRY.get(agent_id)
    if agent is None:
        raise KeyError(f"Unknown agent: {agent_id}")
    return agent


def produce_item(agent: ManagedAgent, task: TestItem) -> tuple[TestItem, ExecutionTrace]:
    """Wrap an agent answer as an eval-ready ``TestItem`` plus its trace."""
    output, trace = agent.run(task)
    item = task.model_copy(
        update={
            "id": f"{task.id}::{agent.agent_id}",
            "agent_id": agent.agent_id,
            "candidate_output": output.strip(),
        }
    )
    return item, trace


def reset_registry() -> None:
    """Clear the registry. Intended for tests."""
    global _INITIALIZED
    _REGISTRY.clear()
    _INITIALIZED = False
    for name in list(sys.modules):
        if name.startswith("community_agents."):
            del sys.modules[name]
