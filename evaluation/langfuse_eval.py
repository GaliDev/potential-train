"""Judge live Langfuse traces with the panel and push scores back.

The continuous-evaluation loop on top of Langfuse:

1. Pull recent traces from Langfuse (time window + optional tag/name filters).
2. Map each trace onto a TestItem and run the judge panel over it
   (correctness, faithfulness, completeness, coherence, safety).
3. Persist results to the local performance store (same path as fleet runs),
   and optionally push the five criterion scores + aggregate + pass/fail back
   to Langfuse so they appear on each trace in the UI.

Run with:

    PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --limit 20 --push-scores

Requires LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env
(and OPENAI_API_KEY for the judges).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from eval_harness.baseline import BaselineJudge
from eval_harness.datasets.langfuse_loader import (
    LangfuseClient,
    load_langfuse_items,
    push_result_scores,
)
from eval_harness.graph import PanelJudge
from eval_harness.llm import LLMClient
from eval_harness.runner import run_evaluation
from eval_harness.schemas import AgentProfile, Criterion, TaskType
from eval_harness.store import upsert_agent


def register_discovered_agents(items: list) -> None:
    """Register agents seen in Langfuse traces so dashboards include them.

    The leaderboard and governance views iterate over the agent registry, so
    evals for an unregistered agent_id would be invisible.
    """
    seen: dict[str, AgentProfile] = {}
    for item in items:
        if item.agent_id and item.agent_id not in seen:
            seen[item.agent_id] = AgentProfile(
                agent_id=item.agent_id,
                name=item.agent_id,
                task_type=item.task_type,
                model="external",
                description="Discovered from Langfuse traces",
            )
    for profile in seen.values():
        upsert_agent(profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="Look-back window for traces")
    parser.add_argument("--limit", type=int, default=20, help="Max traces to evaluate")
    parser.add_argument("--name", default=None, help="Filter traces by name")
    parser.add_argument("--user-id", default=None, help="Filter traces by user id")
    parser.add_argument("--session-id", default=None, help="Filter traces by session id")
    parser.add_argument("--tags", nargs="*", default=None, help="Filter traces by tag(s)")
    parser.add_argument(
        "--default-task-type",
        choices=[t.value for t in TaskType],
        default=TaskType.RAG_QA.value,
        help="Task family assumed when a trace carries no hint",
    )
    parser.add_argument(
        "--with-observations",
        action="store_true",
        help="Fetch full traces to extract retrieval context (one extra call per trace)",
    )
    parser.add_argument(
        "--judge",
        choices=["panel", "baseline"],
        default="panel",
        help="panel = five criterion judges + aggregator; baseline = single call",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["default", "local"],
        default="default",
        help="default = per-criterion models (OpenAI/HF router); "
        "local = force all judges onto a local Ollama-style server",
    )
    parser.add_argument(
        "--judge-model",
        default="llama3.1",
        help="Model id for --judge-backend local (default: llama3.1)",
    )
    parser.add_argument(
        "--judge-base-url",
        default="http://localhost:11434/v1",
        help="Base URL for --judge-backend local (default: Ollama at :11434)",
    )
    parser.add_argument("--push-scores", action="store_true", help="Write scores back to Langfuse")
    parser.add_argument("--no-persist", action="store_true", help="Skip the local SQLite store")
    parser.add_argument("--dry-run", action="store_true", help="Only show mapped items; no judging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = LangfuseClient()

    items = load_langfuse_items(
        client=client,
        hours=args.hours,
        name=args.name,
        user_id=args.user_id,
        session_id=args.session_id,
        tags=args.tags,
        limit=args.limit,
        default_task_type=TaskType(args.default_task_type),
        fetch_observations=args.with_observations,
    )
    print(f"Fetched {len(items)} judge-ready traces from Langfuse ({args.hours}h window)")
    if not items:
        return

    if args.dry_run:
        for item in items:
            prompt_preview = item.task_prompt.replace("\n", " ")[:80]
            print(f"  {item.id}  [{item.task_type.value}]  agent={item.agent_id}  {prompt_preview}")
        return

    if not args.no_persist:
        register_discovered_agents(items)

    judge_client = None
    judge_model = None
    if args.judge_backend == "local":
        judge_client = LLMClient(api_key="local", base_url=args.judge_base_url)
        judge_model = args.judge_model
        print(f"Judging with local backend {judge_model} @ {args.judge_base_url}")
    if args.judge == "panel":
        judge = PanelJudge(model=judge_model, client=judge_client)
    else:
        judge = BaselineJudge(client=judge_client, model=judge_model)
    report = run_evaluation(items, judge, persist=not args.no_persist)

    print(
        f"\nJudged {report.count} traces  pass_rate={report.pass_rate:.0%}  "
        f"cost=${report.total_cost_usd:.4f}  errors={len(report.errors)}"
    )
    by_criterion: dict[Criterion, list[int]] = defaultdict(list)
    for result in report.results:
        for verdict in result.verdicts:
            by_criterion[verdict.criterion].append(verdict.score)
    for criterion, scores in by_criterion.items():
        print(f"  {criterion.value:<13} avg={sum(scores) / len(scores):.2f}  n={len(scores)}")

    if args.push_scores:
        pushed = 0
        for result in report.results:
            try:
                pushed += push_result_scores(client, result)
            except RuntimeError as exc:
                print(f"  [push-scores] {result.item_id}: {exc}")
        print(f"Pushed {pushed} scores back to Langfuse")

    client.close()


if __name__ == "__main__":
    main()
