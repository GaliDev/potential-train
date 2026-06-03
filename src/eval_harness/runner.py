"""Run a judge over a set of test items and persist the results.

The runner is judge-agnostic: it accepts anything exposing `.judge(item)` and
`.judge_mode` (the baseline judge today, the LangGraph panel later), so both
share the same persistence, telemetry, and run-report path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import RUNS_DIR
from .schemas import AggregateResult, TestItem
from .store import record_eval


class Judge(Protocol):
    judge_mode: str

    def judge(self, item: TestItem) -> AggregateResult: ...


@dataclass
class RunReport:
    judge_mode: str
    results: list[AggregateResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.overall_pass) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "judge_mode": self.judge_mode,
            "count": self.count,
            "pass_rate": self.pass_rate,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_s": self.total_latency_s,
            "avg_latency_s": (self.total_latency_s / self.count) if self.count else 0.0,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.model_dump(mode="json") for r in self.results],
            "errors": self.errors,
        }


def run_evaluation(
    items: list[TestItem],
    judge: Judge,
    persist: bool = True,
    save_run: bool = True,
) -> RunReport:
    """Evaluate every item with `judge`, persisting per-item results.

    A failure on one item is captured in `report.errors` and does not abort the
    batch. Returns a RunReport with aggregate cost/latency/pass-rate.
    """
    report = RunReport(judge_mode=judge.judge_mode, started_at=_now())

    for item in items:
        try:
            result = judge.judge(item)
        except Exception as exc:  # noqa: BLE001 - capture and continue
            report.errors.append({"item_id": item.id, "error": str(exc)})
            continue
        report.results.append(result)
        report.total_cost_usd += result.total_cost_usd
        report.total_latency_s += result.total_latency_s
        if persist:
            record_eval(result)

    report.finished_at = _now()
    if save_run:
        _save_run(report)
    return report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_run(report: RunReport) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run_{report.judge_mode}_{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
