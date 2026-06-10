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


def _safe_kappa(pred: list[int], gold: list[int]) -> float | None:
    if not pred or len(set(pred)) < 2 or len(set(gold)) < 2:
        return None
    from sklearn.metrics import cohen_kappa_score

    return round(float(cohen_kappa_score(gold, pred)), 4)


def per_criterion_stats(results, items: list[TestItem]) -> dict[str, dict]:
    """Per-criterion agreement with the (item-level) human gold labels.

    The gold set carries one human score per item, so a criterion's quality is
    measured by how well that single dimension tracks the human's overall
    judgment: pass accuracy / Cohen's kappa of the criterion's pass flag vs
    gold_pass, Spearman + MAE of the criterion score vs gold_score, plus the
    criterion's own latency and cost telemetry.
    """
    gold = {it.id: it for it in items}
    out: dict[str, dict] = {}
    for criterion in CRITERIA:
        pred_scores: list[float] = []
        gold_scores: list[float] = []
        pred_pass: list[int] = []
        gold_pass: list[int] = []
        latencies: list[float] = []
        total_cost = 0.0
        for result in results:
            g = gold.get(result.item_id)
            verdict = next(
                (v for v in result.verdicts if v.criterion.value == criterion), None
            )
            if g is None or verdict is None:
                continue
            if verdict.latency_s is not None:
                latencies.append(verdict.latency_s)
            total_cost += verdict.cost_usd or 0.0
            if g.gold_score is not None:
                pred_scores.append(float(verdict.score))
                gold_scores.append(float(g.gold_score))
            if g.gold_pass is not None:
                pred_pass.append(int(verdict.passed))
                gold_pass.append(int(g.gold_pass))
        mae = (
            round(
                sum(abs(p - g) for p, g in zip(pred_scores, gold_scores))
                / len(pred_scores),
                4,
            )
            if pred_scores
            else None
        )
        pass_acc = (
            round(sum(int(p == g) for p, g in zip(pred_pass, gold_pass)) / len(pred_pass), 4)
            if pred_pass
            else None
        )
        out[criterion] = {
            "n": len(pred_scores),
            "pass_accuracy": pass_acc,
            "cohen_kappa": _safe_kappa(pred_pass, gold_pass),
            "spearman": _safe_spearman(pred_scores, gold_scores),
            "mae": mae,
            "mean_score": round(sum(pred_scores) / len(pred_scores), 3) if pred_scores else None,
            "avg_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "total_cost_usd": round(total_cost, 6),
        }
    return out


# Weighting for the recommendation: quality dominates; latency and cost act
# as real-world tie-breakers. Each part is normalized to 0..1 across models.
# At 80/10/10 the latency+cost advantage can swing at most 0.20, so a judge
# must be within ~0.25 quality of the leader before speed/price can flip the
# recommendation - cheapness decides near-ties, never crowns a bad judge.
COMPOSITE_WEIGHTS = {"quality": 0.80, "latency": 0.10, "cost": 0.10}


def _quality_score(stats: dict, mae_key: str = "mae") -> float | None:
    """Average of the 0..1 quality signals: pass acc, kappa, spearman, 1 - MAE/4."""
    parts: list[float] = []
    for key in ("pass_accuracy", "cohen_kappa", "spearman"):
        if stats.get(key) is not None:
            parts.append(max(0.0, float(stats[key])))
    if stats.get(mae_key) is not None:
        parts.append(max(0.0, 1.0 - float(stats[mae_key]) / 4.0))
    return round(sum(parts) / len(parts), 4) if parts else None


def _relative(values: list[float | None]) -> list[float | None]:
    """Min-max normalize where LOWER is better -> best gets 1.0, worst 0.0."""
    nums = [v for v in values if v is not None]
    if not nums:
        return [None] * len(values)
    lo, hi = min(nums), max(nums)
    out: list[float | None] = []
    for v in values:
        if v is None:
            out.append(None)
        elif hi == lo:
            out.append(1.0)
        else:
            out.append(round(1.0 - (v - lo) / (hi - lo), 4))
    return out


def _combine(quality: float | None, latency_score: float | None, cost_score: float | None) -> float | None:
    if quality is None:
        return None
    w = COMPOSITE_WEIGHTS
    total = w["quality"] * quality
    denom = w["quality"]
    if latency_score is not None:
        total += w["latency"] * latency_score
        denom += w["latency"]
    if cost_score is not None:
        total += w["cost"] * cost_score
        denom += w["cost"]
    return round(total / denom, 4)


def apply_composites(rows: list[dict]) -> None:
    """Attach composite (quality + latency + cost) scores, overall and per criterion."""
    lat_scores = _relative([r["metrics"].get("avg_latency_s") for r in rows])
    cost_scores = _relative([r["metrics"].get("total_cost_usd") for r in rows])
    for row, ls, cs in zip(rows, lat_scores, cost_scores):
        row["composite"] = _combine(_quality_score(row["metrics"], mae_key="score_mae"), ls, cs)

    for criterion in CRITERIA:
        # Per-criterion latency/cost when recorded; the model-level value is
        # the fallback (a model's speed/price doesn't change per criterion).
        def _lat(r: dict) -> float | None:
            stats = (r.get("per_criterion") or {}).get(criterion) or {}
            return stats.get("avg_latency_s") or r["metrics"].get("avg_latency_s")

        def _cost(r: dict) -> float | None:
            stats = (r.get("per_criterion") or {}).get(criterion) or {}
            return stats.get("total_cost_usd") or r["metrics"].get("total_cost_usd")

        lat_c = _relative([_lat(r) for r in rows])
        cost_c = _relative([_cost(r) for r in rows])
        for row, ls, cs in zip(rows, lat_c, cost_c):
            stats = (row.get("per_criterion") or {}).get(criterion)
            if stats is None:
                continue
            stats["composite"] = _combine(_quality_score(stats), ls, cs)


def _num(value, default: float = -9.0) -> float:
    return default if value is None else float(value)


def pick_overall_winner(rows: list[dict]) -> dict | None:
    """Best judge overall by composite (quality 80% + latency 10% + cost 10%)."""
    scored = [r for r in rows if r["metrics"].get("n")]
    if not scored:
        return None
    best = max(
        scored,
        key=lambda r: (
            _num(r.get("composite")),
            _num(r["metrics"].get("cohen_kappa")),
            _num(r["metrics"].get("spearman")),
        ),
    )
    return {
        "key": best["key"],
        "display_name": best["display_name"],
        "composite": best.get("composite"),
    }


def pick_criterion_winners(rows: list[dict]) -> dict[str, dict]:
    """Best judge per criterion by that criterion's composite score."""
    winners: dict[str, dict] = {}
    for criterion in CRITERIA:
        candidates = [r for r in rows if r.get("per_criterion", {}).get(criterion)]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                _num(r["per_criterion"][criterion].get("composite")),
                _num(r["per_criterion"][criterion].get("spearman")),
                -_num(r["per_criterion"][criterion].get("mae"), default=9.0),
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
    apply_composites(model_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(items),
        "task_mix": dict(task_mix),
        "criteria": CRITERIA,
        "models": model_rows,
        "weights": COMPOSITE_WEIGHTS,
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
        print(
            f"\n=== Best judge overall: {winners['overall']['display_name']} "
            f"(composite={winners['overall'].get('composite')}; "
            f"quality 80% + latency 10% + cost 10%) ==="
        )
    for criterion, w in winners["per_criterion"].items():
        print(
            f"  best {criterion:<13} {w['display_name']:<24} "
            f"(composite={w.get('composite')}  rho={w['spearman']}  mae={w['mae']}  "
            f"pass_acc={w['pass_accuracy']})"
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
