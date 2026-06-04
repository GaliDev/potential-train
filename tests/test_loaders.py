from eval_harness.datasets.loaders import load_gold, load_jsonl, save_jsonl
from eval_harness.schemas import TaskType, TestItem


def test_gold_set_loads_and_is_labeled():
    items = load_gold()
    assert len(items) >= 16
    assert all(it.gold_score is not None for it in items)
    assert all(it.gold_pass is not None for it in items)


def test_gold_filter_by_task_type():
    rag = load_gold(TaskType.RAG_QA)
    assert all(it.task_type == TaskType.RAG_QA for it in rag)


def test_coerce_gold_pass_from_score(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id":"a","task_type":"rag_qa","task_prompt":"q","candidate_output":"o","gold_score":5}\n'
        '{"id":"b","task_type":"rag_qa","task_prompt":"q","candidate_output":"o","gold_score":2}\n'
    )
    items = load_jsonl(p)
    assert items[0].gold_pass is True   # 5 >= threshold
    assert items[1].gold_pass is False  # 2 < threshold


def test_save_and_reload_roundtrip(tmp_path):
    items = [TestItem(id="a", task_type=TaskType.RAG_QA, task_prompt="q", gold_score=5, gold_pass=True)]
    out = save_jsonl(items, tmp_path / "out.jsonl")
    reloaded = load_jsonl(out)
    assert reloaded[0].id == "a"
    assert reloaded[0].gold_score == 5
