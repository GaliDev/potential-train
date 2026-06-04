from eval_harness.metrics import compute_agreement
from eval_harness.schemas import AggregateResult, TaskType, TestItem


def _gold(i, gp, gs):
    return TestItem(id=f"x{i}", task_type=TaskType.RAG_QA, task_prompt="q", gold_pass=gp, gold_score=gs)


def _pred(i, op, score):
    return AggregateResult(item_id=f"x{i}", task_type=TaskType.RAG_QA, aggregate_score=score,
                           overall_pass=op, total_latency_s=1.0, total_cost_usd=0.002)


def test_perfect_agreement():
    gold, pred = [], []
    for i in range(6):
        gp = i % 2 == 0
        gs = 5.0 if gp else 2.0
        gold.append(_gold(i, gp, gs))
        pred.append(_pred(i, gp, gs))
    m = compute_agreement(pred, gold)
    assert m.pass_accuracy == 1.0
    assert m.cohen_kappa == 1.0
    assert m.score_mae == 0.0
    assert m.cost_per_item_usd == 0.002


def test_partial_agreement():
    gold, pred = [], []
    for i in range(10):
        gp = i % 2 == 0
        gold.append(_gold(i, gp, 5.0 if gp else 2.0))
        # disagree on the last two
        op = gp if i < 8 else (not gp)
        pred.append(_pred(i, op, 5.0 if op else 2.0))
    m = compute_agreement(pred, gold)
    assert m.pass_accuracy == 0.8
    assert m.n == 10


def test_handles_single_class_without_crashing():
    gold = [_gold(i, True, 5.0) for i in range(3)]
    pred = [_pred(i, True, 5.0) for i in range(3)]
    m = compute_agreement(pred, gold)
    # kappa undefined with a single class -> None, but accuracy still computed
    assert m.pass_accuracy == 1.0
    assert m.cohen_kappa is None
