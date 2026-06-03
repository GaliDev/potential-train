"""Benchmark the judge against human gold labels and compare configurations.

Runs a judge (baseline or panel) over a labeled set, computes agreement
metrics, and can compare configurations side by side (e.g. baseline vs panel).
Designed to be run as a script:

    PYTHONPATH=src python -m evaluation.benchmark --mode compare
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval_harness.baseline import BaselineJudge
from eval_harness.config import RUNS_DIR
from eval_harness.datasets.loaders import load_gold, load_hybrid
from eval_harness.graph import PanelJudge
from eval_harness.improvement import CascadeJudge, JuryJudge
from eval_harness.metrics import AgreementMetrics, compute_agreement
from eval_harness.runner import RunReport, run_evaluation
from eval_harness.schemas import TestItem


def evaluate_judge(
    judge,
    items: list[TestItem],
    persist: bool = False,
    save_run: bool = False,
) -> tuple[RunReport, AgreementMetrics]:
    report = run_evaluation(items, judge, persist=persist, save_run=save_run)
    metrics = compute_agreement(report.results, items)
    return report, metrics


def build_configs(with_improve: bool = False) -> dict:
    configs = {
        "baseline": BaselineJudge(),
        "panel": PanelJudge(),
    }
    if with_improve:
        configs["cascade"] = CascadeJudge()
        configs["jury"] = JuryJudge()
    return configs


def compare(
    items: list[TestItem],
    persist: bool = False,
    with_improve: bool = False,
) -> dict:
    """Evaluate each judge configuration on the same items and compare."""
    out: dict = {"n_items": len(items), "configs": {}}
    for name, judge in build_configs(with_improve).items():
        _report, metrics = evaluate_judge(judge, items, persist=persist)
        out["configs"][name] = metrics.to_dict()
    return out


def _save(report: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"benchmark_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def _print_metrics(name: str, m: dict) -> None:
    print(f"\n[{name}]  n={m['n']}")
    print(f"  pass accuracy : {m['pass_accuracy']}")
    print(f"  cohen kappa   : {m['cohen_kappa']}")
    print(f"  precision/rec : {m['precision']} / {m['recall']}  (F1 {m['f1']})")
    print(f"  spearman      : {m['spearman']} (p={m['spearman_p']})")
    print(f"  score MAE     : {m['score_mae']}")
    print(f"  avg latency s : {m['avg_latency_s']}")
    print(f"  cost / item $ : {m['cost_per_item_usd']}  (total ${m['total_cost_usd']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the judge vs human gold labels.")
    parser.add_argument(
        "--mode", choices=["baseline", "panel", "cascade", "jury", "compare"], default="compare"
    )
    parser.add_argument("--public", action="store_true", help="Include the public SummEval slice")
    parser.add_argument("--public-limit", type=int, default=40)
    parser.add_argument("--persist", action="store_true", help="Write results to the store")
    parser.add_argument(
        "--with-improve", action="store_true", help="Also benchmark cascade + jury in compare mode"
    )
    args = parser.parse_args()

    items = load_hybrid(public_limit=args.public_limit) if args.public else load_gold()
    print(f"Loaded {len(items)} labeled items.")

    single = {
        "baseline": BaselineJudge,
        "panel": PanelJudge,
        "cascade": CascadeJudge,
        "jury": JuryJudge,
    }
    if args.mode == "compare":
        report = compare(items, persist=args.persist, with_improve=args.with_improve)
        for name, m in report["configs"].items():
            _print_metrics(name, m)
    else:
        judge = single[args.mode]()
        _report, metrics = evaluate_judge(judge, items, persist=args.persist)
        report = {"n_items": len(items), "configs": {args.mode: metrics.to_dict()}}
        _print_metrics(args.mode, metrics.to_dict())

    path = _save(report)
    print(f"\nSaved benchmark report to {path}")


if __name__ == "__main__":
    main()
