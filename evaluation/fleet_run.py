"""Generate real fleet outputs, panel-judge them, and persist dashboard data.

This is the production-shaped counterpart to `demo_seed.py`: it spends real
LLM calls to have each managed fleet agent answer benchmark tasks, then runs
the panel judge over those outputs with `persist=True` so Streamlit can read
the resulting SQLite rows.

Run with:

    PYTHONPATH=src python -m evaluation.fleet_run --limit-per-type 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval_harness.config import RUNS_DIR
from eval_harness.datasets.loaders import load_gold
from eval_harness.fleet.configs import get_fleet
from eval_harness.fleet.generator import generate_for_fleet, generate_output
from eval_harness.fleet.registry import list_registered_agents, produce_item
from eval_harness.graph import PanelJudge
from eval_harness.runner import RunReport, run_evaluation
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem
from eval_harness.store import fetch_evals, record_run_signal, upsert_agent


def _group_by_task_type(items: list[TestItem]) -> dict[TaskType, list[TestItem]]:
    grouped: dict[TaskType, list[TestItem]] = defaultdict(list)
    for item in items:
        grouped[item.task_type].append(item)
    return dict(grouped)


def build_real_fleet_items(
    *,
    task_type: TaskType | None = None,
    limit_per_type: int = 1,
) -> list[TestItem]:
    """Generate agent-tagged candidate outputs from local gold tasks."""
    if limit_per_type < 1:
        raise ValueError("limit_per_type must be at least 1")

    tasks = load_gold(task_type)
    produced: list[TestItem] = []
    for tt, tt_tasks in _group_by_task_type(tasks).items():
        produced.extend(generate_for_fleet(tt_tasks[:limit_per_type], tt))
    return produced


def run_real_fleet_evaluation(
    *,
    task_type: TaskType | None = None,
    limit_per_type: int = 1,
    save_run: bool = True,
) -> RunReport:
    """Generate fleet outputs and persist panel evaluations for the dashboard.

    This streams one generated item at a time so long runs can be resumed safely
    after interruption. Existing panel evals are skipped before generation,
    avoiding duplicate rows and repeated generation spend.
    """
    if limit_per_type < 1:
        raise ValueError("limit_per_type must be at least 1")

    tasks = _group_by_task_type(load_gold(task_type))
    judge = PanelJudge()
    report = RunReport(judge_mode=judge.judge_mode, started_at=_now())
    existing = _existing_panel_item_ids()

    total = sum(
        min(limit_per_type, len(tt_tasks)) * len(list_registered_agents(tt))
        for tt, tt_tasks in tasks.items()
    )
    done = 0
    skipped = 0

    for tt, tt_tasks in tasks.items():
        agents = list_registered_agents(tt)
        for agent in agents:
            upsert_agent(agent.profile)

        for task in tt_tasks[:limit_per_type]:
            for agent in agents:
                done += 1
                item_id = f"{task.id}::{agent.agent_id}"
                prefix = f"[{done}/{total}] {item_id}"

                if item_id in existing:
                    skipped += 1
                    print(f"{prefix} skipped existing", flush=True)
                    continue

                try:
                    produced, trace = produce_item(agent, task)
                    record_run_signal(
                        item_id=item_id,
                        agent_id=agent.agent_id,
                        task_type=tt,
                        trace=trace,
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue
                    err = {"item_id": item_id, "error": f"generation failed: {exc}"}
                    report.errors.append(err)
                    record_run_signal(
                        item_id=item_id,
                        agent_id=agent.agent_id,
                        task_type=tt,
                        trace=ExecutionTrace(
                            model=getattr(agent.profile, "model", ""),
                            success=False,
                            error=str(exc),
                        ),
                    )
                    print(f"{prefix} ERROR {err['error']}", flush=True)
                    continue

                item_report = run_evaluation([produced], judge, persist=True, save_run=False)
                report.results.extend(item_report.results)
                report.errors.extend(item_report.errors)
                report.total_cost_usd += item_report.total_cost_usd
                report.total_latency_s += item_report.total_latency_s

                if item_report.results:
                    existing.add(item_id)
                    result = item_report.results[0]
                    print(
                        f"{prefix} saved score={result.aggregate_score} "
                        f"pass={result.overall_pass} cost=${result.total_cost_usd:.6f}",
                        flush=True,
                    )
                else:
                    error = item_report.errors[-1]["error"] if item_report.errors else "unknown"
                    print(f"{prefix} ERROR {error}", flush=True)

    report.finished_at = _now()
    print(f"Skipped existing: {skipped}", flush=True)
    if save_run:
        _save_report(report)
    return report


def _existing_panel_item_ids() -> set[str]:
    return {row.item_id for row in fetch_evals(judge_mode="panel")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_report(report: RunReport) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run_{report.judge_mode}_{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def _parse_task_type(value: str) -> TaskType | None:
    return None if value == "all" else TaskType(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate the Streamlit dashboard with real fleet evaluation data."
    )
    parser.add_argument(
        "--task-type",
        choices=["all", *[tt.value for tt in TaskType]],
        default="all",
        help="Task family to run. Defaults to one batch for every task type.",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=1,
        help="Number of source tasks per selected task type. Each task is run by 3 agents.",
    )
    parser.add_argument(
        "--no-save-run",
        action="store_true",
        help="Persist eval rows but skip writing data/runs/run_panel_*.json.",
    )
    args = parser.parse_args()

    task_type = _parse_task_type(args.task_type)
    report = run_real_fleet_evaluation(
        task_type=task_type,
        limit_per_type=args.limit_per_type,
        save_run=not args.no_save_run,
    )

    print(f"Panel evaluations persisted this run: {report.count}")
    print(f"Errors: {len(report.errors)}")
    print(f"Total judge cost: ${report.total_cost_usd:.6f}")
    print(f"Average judge latency: {report.to_dict()['avg_latency_s']:.4f}s")
    if report.errors:
        print("\nFirst errors:")
        for err in report.errors[:5]:
            print(f"  - {err['item_id']}: {err['error']}")


if __name__ == "__main__":
    main()
