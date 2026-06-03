"""Agreement metrics: how well the judge matches human gold labels.

The headline success metric for the platform is agreement with humans:
pass/fail accuracy and Cohen's kappa on the binary verdict, plus Spearman
correlation and MAE on the 1-5 score. Latency and cost round out the picture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from .schemas import AggregateResult, TestItem


@dataclass
class AgreementMetrics:
    n: int
    pass_accuracy: float | None
    cohen_kappa: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    spearman: float | None
    spearman_p: float | None
    score_mae: float | None
    avg_latency_s: float
    total_cost_usd: float
    cost_per_item_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, 4)


def compute_agreement(
    results: list[AggregateResult],
    gold_items: list[TestItem],
) -> AgreementMetrics:
    """Align results to gold items by id and compute agreement metrics."""
    gold = {it.id: it for it in gold_items}

    pred_pass: list[int] = []
    true_pass: list[int] = []
    pred_score: list[float] = []
    true_score: list[float] = []
    latencies: list[float] = []
    total_cost = 0.0

    for r in results:
        latencies.append(r.total_latency_s)
        total_cost += r.total_cost_usd
        g = gold.get(r.item_id)
        if g is None:
            continue
        if g.gold_pass is not None:
            pred_pass.append(int(r.overall_pass))
            true_pass.append(int(g.gold_pass))
        if g.gold_score is not None:
            pred_score.append(float(r.aggregate_score))
            true_score.append(float(g.gold_score))

    n = len(results)
    pass_acc = _safe(accuracy_score(true_pass, pred_pass)) if pred_pass else None

    kappa = None
    precision = recall = f1 = None
    if pred_pass and len(set(true_pass)) > 1 and len(set(pred_pass)) > 1:
        kappa = _safe(cohen_kappa_score(true_pass, pred_pass))
    if pred_pass:
        p, r_, f, _ = precision_recall_fscore_support(
            true_pass, pred_pass, average="binary", zero_division=0
        )
        precision, recall, f1 = _safe(p), _safe(r_), _safe(f)

    spearman = spearman_p = None
    if len(pred_score) >= 2 and len(set(true_score)) > 1 and len(set(pred_score)) > 1:
        rho, pval = spearmanr(pred_score, true_score)
        spearman, spearman_p = _safe(rho), _safe(pval)

    mae = None
    if pred_score:
        mae = _safe(sum(abs(a - b) for a, b in zip(pred_score, true_score)) / len(pred_score))

    avg_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

    return AgreementMetrics(
        n=n,
        pass_accuracy=pass_acc,
        cohen_kappa=kappa,
        precision=precision,
        recall=recall,
        f1=f1,
        spearman=spearman,
        spearman_p=spearman_p,
        score_mae=mae,
        avg_latency_s=avg_latency,
        total_cost_usd=round(total_cost, 6),
        cost_per_item_usd=round(total_cost / n, 6) if n else 0.0,
    )
