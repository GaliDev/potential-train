"""Offline tests: verdict parsing, prompt building, report rendering.

No network, no HF token required.
"""

from __future__ import annotations

from eval_harness.schemas import Criterion, TaskType, TestItem

from hf_judges.parsers import parse_verdict
from hf_judges.prompts import system_prompt, user_prompt
from hf_judges.report_html import render_report


# --- parse_verdict --------------------------------------------------------


def test_parses_clean_json():
    v = parse_verdict('{"score": 4, "passed": true, "rationale": "good", "evidence": ["x"]}')
    assert v.score == 4 and v.passed is True
    assert v.rationale == "good"
    assert v.evidence == ["x"]
    assert v.parse_confidence == "json"


def test_parses_json_in_markdown_fence():
    raw = 'Here is my verdict:\n```json\n{"score": 2, "passed": false, "rationale": "bad"}\n```'
    v = parse_verdict(raw)
    assert v.score == 2 and v.passed is False
    assert v.parse_confidence == "json"


def test_strips_think_tags():
    raw = "<think>The response misses half the task...</think>\n" '{"score": 3, "passed": false, "rationale": "partial"}'
    v = parse_verdict(raw)
    assert v.score == 3
    assert v.parse_confidence == "json"


def test_json_with_surrounding_prose():
    raw = 'Sure! {"score": 5, "passed": true, "rationale": "excellent"} Hope that helps.'
    v = parse_verdict(raw)
    assert v.score == 5 and v.parse_confidence == "json"


def test_regex_rescue_when_json_broken():
    v = parse_verdict('The "score": 4 because it is mostly right, though...')
    assert v.score == 4
    assert v.passed is True  # >= PASS_THRESHOLD
    assert v.parse_confidence == "regex"


def test_fallback_on_garbage_defaults_mid_scale():
    v = parse_verdict("I cannot evaluate this.")
    assert 1 <= v.score <= 5
    assert v.parse_confidence == "fallback"


def test_score_clamped_to_range():
    v = parse_verdict('{"score": 9, "passed": true, "rationale": "x"}')
    # 9 is not a valid 1-5 JSON score for our scale; clamp keeps range
    assert 1 <= v.score <= 5


def test_passed_derived_from_score_when_missing():
    v = parse_verdict('{"score": 5, "rationale": "great"}')
    assert v.passed is True
    v = parse_verdict('{"score": 2, "rationale": "weak"}')
    assert v.passed is False


# --- prompts ---------------------------------------------------------------


def make_item() -> TestItem:
    return TestItem(
        id="t1",
        task_type=TaskType.RAG_QA,
        task_prompt="What is 2+2?",
        candidate_output="4",
        context="Basic arithmetic.",
    )


def test_system_prompt_mentions_criterion_and_json():
    s = system_prompt(Criterion.FAITHFULNESS)
    assert "Faithfulness" in s
    assert '"score"' in s


def test_system_prompt_suffix_appended():
    s = system_prompt(Criterion.SAFETY, suffix="\n/no_think")
    assert s.endswith("/no_think")


def test_user_prompt_contains_item_fields():
    u = user_prompt(make_item())
    assert "What is 2+2?" in u
    assert "Basic arithmetic." in u


# --- per-criterion stats and winners ----------------------------------------


def make_row(key: str, kappa, spearman, mae, per_criterion: dict) -> dict:
    return {
        "key": key,
        "display_name": key,
        "metrics": {
            "n": 10,
            "cohen_kappa": kappa,
            "spearman": spearman,
            "score_mae": mae,
            "pass_accuracy": 0.9,
        },
        "per_criterion": per_criterion,
    }


def crit_stats(spearman, mae=0.5, pass_accuracy=0.9) -> dict:
    return {"spearman": spearman, "mae": mae, "pass_accuracy": pass_accuracy, "n": 10}


def test_pick_overall_winner_prefers_kappa_then_spearman():
    from hf_judges.run_benchmark import pick_overall_winner

    rows = [
        make_row("a", kappa=0.8, spearman=0.99, mae=0.2, per_criterion={}),
        make_row("b", kappa=0.9, spearman=0.50, mae=0.4, per_criterion={}),
    ]
    assert pick_overall_winner(rows)["key"] == "b"  # kappa wins first
    rows[0]["metrics"]["cohen_kappa"] = 0.9
    assert pick_overall_winner(rows)["key"] == "a"  # tie -> spearman decides


def test_pick_overall_winner_handles_none_metrics():
    from hf_judges.run_benchmark import pick_overall_winner

    rows = [
        make_row("broken", kappa=None, spearman=None, mae=None, per_criterion={}),
        make_row("ok", kappa=0.5, spearman=0.5, mae=0.5, per_criterion={}),
    ]
    assert pick_overall_winner(rows)["key"] == "ok"


def test_pick_criterion_winners_selects_best_per_criterion():
    from hf_judges.run_benchmark import pick_criterion_winners

    rows = [
        make_row(
            "a", 0.5, 0.5, 0.5,
            per_criterion={
                "correctness": crit_stats(spearman=0.9),
                "safety": crit_stats(spearman=0.2),
            },
        ),
        make_row(
            "b", 0.5, 0.5, 0.5,
            per_criterion={
                "correctness": crit_stats(spearman=0.7),
                "safety": crit_stats(spearman=0.8),
            },
        ),
    ]
    winners = pick_criterion_winners(rows)
    assert winners["correctness"]["key"] == "a"
    assert winners["safety"]["key"] == "b"


def test_per_criterion_stats_computes_against_gold():
    from hf_judges.run_benchmark import per_criterion_stats
    from eval_harness.schemas import AggregateResult, JudgeVerdict

    items = [
        TestItem(id=f"g{i}", task_type=TaskType.RAG_QA, task_prompt="q",
                 candidate_output="a", gold_score=float(s), gold_pass=s >= 4)
        for i, s in enumerate([1, 3, 5])
    ]
    results = []
    for item in items:
        score = int(item.gold_score)  # judge agrees perfectly
        verdicts = [
            JudgeVerdict(criterion=c, score=score, passed=score >= 4, rationale="r")
            for c in Criterion
        ]
        results.append(AggregateResult(
            item_id=item.id, task_type=item.task_type, verdicts=verdicts,
            aggregate_score=float(score), overall_pass=score >= 4,
        ))
    stats = per_criterion_stats(results, items)
    assert stats["correctness"]["n"] == 3
    assert stats["correctness"]["spearman"] == 1.0
    assert stats["correctness"]["mae"] == 0.0
    assert stats["correctness"]["pass_accuracy"] == 1.0


# --- report ----------------------------------------------------------------


def test_render_report_smoke():
    payload = {
        "generated_at": "2026-06-10T00:00:00Z",
        "n_items": 6,
        "task_mix": {"rag_qa": 2, "summarization": 2, "translation": 2},
        "criteria": [c.value for c in Criterion],
        "models": [
            {
                "key": "qwen3-8b",
                "display_name": "Qwen3 8B",
                "model_id": "Qwen/Qwen3-8B",
                "judge_mode": "hf_qwen3-8b",
                "metrics": {
                    "pass_accuracy": 0.83,
                    "cohen_kappa": 0.65,
                    "spearman": 0.7,
                    "spearman_p": 0.01,
                    "score_mae": 0.6,
                    "avg_latency_s": 4.2,
                    "total_cost_usd": 0.02,
                    "cost_per_item_usd": 0.003,
                    "n": 6,
                    "precision": 0.8,
                    "recall": 0.9,
                    "f1": 0.85,
                },
                "per_criterion_mean": {c.value: 3.5 for c in Criterion},
                "parse_counts": {"json": 28, "regex": 2, "fallback": 0},
                "errors": 0,
                "error_detail": [],
            }
        ],
        "winners": {
            "overall": {"key": "qwen3-8b", "display_name": "Qwen3 8B"},
            "per_criterion": {
                "correctness": {
                    "key": "qwen3-8b", "display_name": "Qwen3 8B",
                    "spearman": 0.7, "mae": 0.6, "pass_accuracy": 0.83, "n": 6,
                },
            },
        },
    }
    html = render_report(payload)
    assert "Qwen3 8B" in html
    assert "meets" in html  # target bar evaluation
    assert "Best judge overall" in html
    assert "Best judge per criterion" in html
    assert "<table>" in html


def test_render_report_without_winners_section():
    payload = {
        "generated_at": "2026-06-10T00:00:00Z",
        "n_items": 0,
        "task_mix": {},
        "criteria": [c.value for c in Criterion],
        "models": [],
    }
    html = render_report(payload)
    assert "Best judge overall" not in html
