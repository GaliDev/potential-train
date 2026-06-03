"""Loaders for the hybrid benchmark set.

- Custom gold set: hand-labeled JSONL under data/gold/ (tracked in git).
- Public slice: SummEval summaries with human quality ratings, fetched via the
  HuggingFace `datasets` library and mapped onto our TestItem schema.

Both yield `TestItem`s carrying `gold_score` (1-5) and `gold_pass`, which the
benchmark layer compares against the judge's verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agents.base import PASS_THRESHOLD
from ..config import GOLD_DIR, PUBLIC_DIR
from ..schemas import TaskType, TestItem


def _coerce_gold(record: dict) -> dict:
    """Fill gold_pass from gold_score when only the score is provided."""
    if record.get("gold_pass") is None and record.get("gold_score") is not None:
        record["gold_pass"] = float(record["gold_score"]) >= PASS_THRESHOLD
    return record


def load_jsonl(path: str | Path) -> list[TestItem]:
    """Load TestItems from a JSONL file (one JSON object per line)."""
    path = Path(path)
    items: list[TestItem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(TestItem.model_validate(_coerce_gold(json.loads(line))))
    return items


def load_gold(task_type: TaskType | None = None) -> list[TestItem]:
    """Load every *.jsonl gold file under data/gold/, optionally filtered."""
    items: list[TestItem] = []
    if GOLD_DIR.exists():
        for path in sorted(GOLD_DIR.glob("*.jsonl")):
            items.extend(load_jsonl(path))
    if task_type is not None:
        items = [it for it in items if it.task_type == task_type]
    return items


def _normalize_summeval_score(expert_scores: list[dict] | dict | None) -> float | None:
    """Average SummEval expert ratings (1-5) across dimensions and annotators."""
    if not expert_scores:
        return None
    dims = ("coherence", "consistency", "fluency", "relevance")
    vals: list[float] = []
    annotators = expert_scores if isinstance(expert_scores, list) else [expert_scores]
    for ann in annotators:
        for d in dims:
            v = ann.get(d)
            if v is not None:
                vals.append(float(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def load_summeval(limit: int = 80, cache: bool = True) -> list[TestItem]:
    """Load a slice of SummEval as summarization TestItems with gold labels.

    Requires the `datasets` package and network access on first download. The
    raw slice is cached to data/public/summeval.jsonl so later runs are offline.
    """
    cache_path = PUBLIC_DIR / "summeval.jsonl"
    if cache and cache_path.exists():
        return load_jsonl(cache_path)[:limit]

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required to fetch SummEval. Install requirements.txt."
        ) from exc

    ds = load_dataset("mteb/summeval", split="test")
    items: list[TestItem] = []
    for i, row in enumerate(ds):
        if len(items) >= limit:
            break
        text = row.get("text") or row.get("article") or ""
        machine = row.get("machine_summaries") or []
        human_scores = row.get("relevance") or row.get("expert_annotations")
        # mteb/summeval stores parallel lists of machine summaries and scores.
        rel = row.get("relevance") or []
        con = row.get("consistency") or []
        coh = row.get("coherence") or []
        flu = row.get("fluency") or []
        for j, summary in enumerate(machine):
            dim_vals = [
                lst[j] for lst in (rel, con, coh, flu)
                if isinstance(lst, list) and j < len(lst)
            ]
            score = round(sum(dim_vals) / len(dim_vals), 3) if dim_vals else None
            if score is None:
                continue
            items.append(
                TestItem(
                    id=f"summeval_{i}_{j}",
                    task_type=TaskType.SUMMARIZATION,
                    task_prompt="Summarize the following source.",
                    context=text,
                    candidate_output=str(summary).strip(),
                    gold_score=score,
                    gold_pass=score >= PASS_THRESHOLD,
                )
            )
            if len(items) >= limit:
                break

    if cache:
        save_jsonl(items, cache_path)
    return items


def save_jsonl(items: list[TestItem], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def load_hybrid(
    public_limit: int = 60,
    include_public: bool = True,
) -> list[TestItem]:
    """The hybrid benchmark: custom gold set + a public SummEval slice."""
    items = load_gold()
    if include_public:
        try:
            items.extend(load_summeval(limit=public_limit))
        except Exception as exc:  # noqa: BLE001 - public set is optional
            print(f"[loaders] Skipping public SummEval slice: {exc}")
    return items
