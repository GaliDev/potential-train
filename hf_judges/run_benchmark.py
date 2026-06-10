"""Benchmark open HF models as judges against the human gold set.

For each candidate model: run the five-criterion panel over a stratified
slice of the gold set, compute agreement with the human labels (the same
metrics the GPT-4o panel is held to), and render a comparison as a
self-contained HTML page.

Run with:

    PYTHONPATH=src python -m hf_judges.run_benchmark --limit-per-type 10

Requires HF_TOKEN in .env. Add --with-openai-baseline to also run the
existing GPT-4o panel on the same items (needs OPENAI_API_KEY, costs more).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval_harness.datasets.loaders import load_gold
from eval_harness.metrics import compute_agreement
from eval_harness.runner import run_evaluation
from eval_harness.schemas import Criterion, TaskType, TestItem

from .config import MODELS, RESULTS_DIR
from .judges import HFJudgeClient, HFPanelJudge
from .report_html import render_report

CRITERIA = [c.value for c in Criterion]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        choices=list(MODELS),
        help="Which candidate judges to benchmark",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=10,
        help="Gold items sampled per task family (rag_qa / summarization / translation)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed")
    parser.add_argument(
        "--with-openai-baseline",
        action="store_true",
        help="Also run the existing GPT-4o panel on the same items",
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR), help="Output directory for reports"
    )
    return parser.parse_args()


def stratified_gold_sample(limit_per_type: int, seed: int) -> list[TestItem]:
    """Deterministic per-task-family sample of the gold set."""
    rng = random.Random(seed)
    by_type: dict[TaskType, list[TestItem]] = defaultdict(list)
    for item in load_gold():
        by_type[item.task_type].append(item)
    sample: list[TestItem] = []
    for task_type in sorted(by_type, key=lambda t: t.value):
        pool = sorted(by_type[task_type], key=lambda it: it.id)
        rng.shuffle(pool)
        sample.extend(pool[:limit_per_type])
    return sample


def per_criterion_mean(results) -> dict[str, float | None]:
    sums: dict[str, list[int]] = defaultdict(list)
    for result in results:
        for verdict in result.verdicts:
            sums[verdict.criterion.value].append(verdict.score)
    return {
        c: (round(sum(v) / len(v), 3) if (v := sums.get(c)) else None) for c in CRITERIA
    }


def benchmark_model(judge, display_name: str, model_id: str, items: list[TestItem]) -> dict:
    print(f"\n--- {display_name} ({model_id}) over {len(items)} items ---")
    report = run_evaluation(items, judge, persist=False, save_run=False)
    metrics = compute_agreement(report.results, items)
    row = {
        "key": getattr(getattr(judge, "spec", None), "key", judge.judge_mode),
        "display_name": display_name,
        "model_id": model_id,
        "judge_mode": judge.judge_mode,
        "metrics": metrics.to_dict(),
        "per_criterion_mean": per_criterion_mean(report.results),
        "parse_counts": dict(getattr(judge, "parse_counts", {})),
        "errors": len(report.errors),
        "error_detail": report.errors[:5],
    }
    m = row["metrics"]
    print(
        f"    pass_acc={m['pass_accuracy']}  kappa={m['cohen_kappa']}  "
        f"spearman={m['spearman']}  mae={m['score_mae']}  "
        f"latency/item={m['avg_latency_s']}s  errors={row['errors']}"
    )
    return row


def build_payload(items: list[TestItem], model_rows: list[dict]) -> dict:
    task_mix: dict[str, int] = defaultdict(int)
    for item in items:
        task_mix[item.task_type.value] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(items),
        "task_mix": dict(task_mix),
        "criteria": CRITERIA,
        "models": model_rows,
    }


def main() -> None:
    args = parse_args()
    items = stratified_gold_sample(args.limit_per_type, args.seed)
    print(f"Benchmarking on {len(items)} gold items (seed={args.seed})")

    client = HFJudgeClient()
    rows: list[dict] = []
    for key in args.models:
        spec = MODELS[key]
        judge = HFPanelJudge(spec, client=client)
        rows.append(benchmark_model(judge, spec.display_name, spec.model_id, items))

    if args.with_openai_baseline:
        from eval_harness.graph import PanelJudge

        rows.append(
            benchmark_model(PanelJudge(), "GPT-4o panel (baseline)", "gpt-4o", items)
        )

    payload = build_payload(items, rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"benchmark_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    html_path = out_dir / f"benchmark_{stamp}.html"
    html_path.write_text(render_report(payload))
    # Stable alias for "the latest report".
    latest = out_dir / "latest.html"
    latest.write_text(render_report(payload))

    print(f"\nJSON:  {json_path}")
    print(f"HTML:  {html_path}")
    print(f"Open:  file://{latest.resolve()}")


if __name__ == "__main__":
    main()
