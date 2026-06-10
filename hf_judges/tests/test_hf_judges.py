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
    }
    html = render_report(payload)
    assert "Qwen3 8B" in html
    assert "meets" in html  # target bar evaluation
    assert "<table>" in html
