"""Benchmark candidate LLM judges against the human gold set.

Candidates: open HF models (via the Inference Providers router), Claude
(Anthropic SDK), and Grok (xAI). For each: run the five-criterion panel over
the gold set (data/gold/rag_qa_gold.jsonl, summarization_gold.jsonl,
translation_gold.jsonl), compute agreement with the human labels - overall
AND per criterion - and declare the best judge overall plus the best judge
for each criterion (correctness, faithfulness, completeness, coherence,
safety). Results render as a self-contained HTML page.

Run with:

    PYTHONPATH=src python -m hf_judges.run_benchmark

Keys in .env per provider: HF_TOKEN (open models), ANTHROPIC_API_KEY
(claude), XAI_API_KEY (grok). Models whose key is missing are skipped with a
warning. Add --with-openai-baseline to include the GPT-4o panel
(OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import spearmanr

from eval_harness.datasets.loaders import load_gold
from eval_harness.metrics import compute_agreement
from eval_harness.runner import run_evaluation
from eval_harness.schemas import Criterion, TaskType, TestItem

from .config import MODELS, RESULTS_DIR
from .judges import build_panel_judge
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
        default=0,
        help="Gold items per task family; 0 = the full gold files "
        "(data/gold/rag_qa_gold.jsonl, summarization_gold.jsonl, translation_gold.jsonl)",
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


def gold_items(limit_per_type: int, seed: int) -> list[TestItem]:
    """The gold set: all of data/gold/*.jsonl, optionally sampled per family."""
    by_type: dict[TaskType, list[TestItem]] = defaultdict(list)
    for item in load_gold():
        by_type[item.task_type].append(item)
    rng = random.Random(seed)
    sample: list[TestItem] = []
    for task_type in sorted(by_type, key=lambda t: t.value):
        pool = sorted(by_type[task_type], key=lambda it: it.id)
        if limit_per_type > 0:
            rng.shuffle(pool)
            pool = pool[:limit_per_type]
        sample.extend(pool)
    return sample


def _safe_spearman(pred: list[float], gold: list[float]) -> float | None:
    if len(pred) < 2 or len(set(pred)) < 2 or len(set(gold)) < 2:
        return None
    rho, _ = spearmanr(pred, gold)
    if rho != rho:  # NaN
        return None
    return round(float(rho), 4)


def per_criterion_stats(results, items: list[TestItem]) -> dict[str, dict]:
    """Per-criterion agreement with the (item-level) human gold labels.

    The gold set carries one human score per item, so a criterion's quality is
    measured by how well that single dimension tracks the human's overall
    judgment: Spearman + MAE of the criterion score vs gold_score, and
    accuracy of the criterion's pass flag vs gold_pass.
    """
    gold = {it.id: it for it in items}
    out: dict[str, dict] = {}
    for criterion in CRITERIA:
        pred_scores: list[float] = []
        gold_scores: list[float] = []
        pass_hits: list[int] = []
        for result in results:
            g = gold.get(result.item_id)
            verdict = next(
                (v for v in result.verdicts if v.criterion.value == criterion), None
            )
            if g is None or verdict is None:
                continue
            if g.gold_score is not None:
                pred_scores.append(float(verdict.score))
                gold_scores.append(float(g.gold_score))
            if g.gold_pass is not None:
                pass_hits.append(int(verdict.passed == g.gold_pass))
        mae = (
            round(
                sum(abs(p - g) for p, g in zip(pred_scores, gold_scores))
                / len(pred_scores),
                4,
            )
            if pred_scores
            else None
        )
        out[criterion] = {
            "n": len(pred_scores),
            "spearman": _safe_spearman(pred_scores, gold_scores),
            "mae": mae,
            "pass_accuracy": round(sum(pass_hits) / len(pass_hits), 4) if pass_hits else None,
            "mean_score": round(sum(pred_scores) / len(pred_scores), 3) if pred_scores else None,
        }
    return out


def _num(value, default: float = -9.0) -> float:
    return default if value is None else float(value)


def pick_overall_winner(rows: list[dict]) -> dict | None:
    """Best judge overall: kappa, then Spearman, then MAE, then pass accuracy."""
    scored = [r for r in rows if r["metrics"].get("n")]
    if not scored:
        return None
    best = max(
        scored,
        key=lambda r: (
            _num(r["metrics"].get("cohen_kappa")),
            _num(r["metrics"].get("spearman")),
            -_num(r["metrics"].get("score_mae"), default=9.0),
            _num(r["metrics"].get("pass_accuracy")),
        ),
    )
    return {"key": best["key"], "display_name": best["display_name"]}


def pick_criterion_winners(rows: list[dict]) -> dict[str, dict]:
    """Best judge per criterion: Spearman vs gold, then MAE, then pass accuracy."""
    winners: dict[str, dict] = {}
    for criterion in CRITERIA:
        candidates = [r for r in rows if r.get("per_criterion", {}).get(criterion)]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                _num(r["per_criterion"][criterion].get("spearman")),
                -_num(r["per_criterion"][criterion].get("mae"), default=9.0),
                _num(r["per_criterion"][criterion].get("pass_accuracy")),
            ),
        )
        winners[criterion] = {
            "key": best["key"],
            "display_name": best["display_name"],
            **best["per_criterion"][criterion],
        }
    return winners


def benchmark_model(judge, display_name: str, model_id: str, items: list[TestItem]) -> dict:
    print(f"\n--- {display_name} ({model_id}) over {len(items)} items ---")
    report = run_evaluation(items, judge, persist=False, save_run=False)
    metrics = compute_agreement(report.results, items)
    criterion_stats = per_criterion_stats(report.results, items)
    row = {
        "key": getattr(getattr(judge, "spec", None), "key", judge.judge_mode),
        "display_name": display_name,
        "model_id": model_id,
        "judge_mode": judge.judge_mode,
        "metrics": metrics.to_dict(),
        "per_criterion": criterion_stats,
        "per_criterion_mean": {
            c: stats.get("mean_score") for c, stats in criterion_stats.items()
        },
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
        "winners": {
            "overall": pick_overall_winner(model_rows),
            "per_criterion": pick_criterion_winners(model_rows),
        },
    }


def main() -> None:
    args = parse_args()
    items = gold_items(args.limit_per_type, args.seed)
    print(f"Benchmarking on {len(items)} gold items")

    rows: list[dict] = []
    for key in args.models:
        spec = MODELS[key]
        try:
            judge = build_panel_judge(spec)
        except RuntimeError as exc:
            print(f"\n[skip] {spec.display_name}: {exc}")
            continue
        rows.append(benchmark_model(judge, spec.display_name, spec.model_id, items))

    if args.with_openai_baseline:
        from eval_harness.graph import PanelJudge

        rows.append(
            benchmark_model(PanelJudge(), "GPT-4o panel (baseline)", "gpt-4o", items)
        )

    if not rows:
        print("No models ran - check the API keys in .env.")
        return

    payload = build_payload(items, rows)

    winners = payload["winners"]
    if winners["overall"]:
        print(f"\n=== Best judge overall: {winners['overall']['display_name']} ===")
    for criterion, w in winners["per_criterion"].items():
        print(
            f"  best {criterion:<13} {w['display_name']:<24} "
            f"(rho={w['spearman']}  mae={w['mae']}  pass_acc={w['pass_accuracy']})"
        )

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
