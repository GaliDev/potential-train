"""Tests for the per-criterion judge model mapping (config/judge_panel.json)."""

from __future__ import annotations

import json

import pytest

import eval_harness.agents.criteria as criteria_mod
from eval_harness.judge_panel_config import (
    DEFAULT_CONFIG_PATH,
    JudgePanelConfigError,
    get_panel_config,
    load_panel_config,
)
from eval_harness.llm import CallStats
from eval_harness.schemas import Criterion, TaskType, TestItem


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    get_panel_config.cache_clear()
    yield
    get_panel_config.cache_clear()


def _valid_config_dict() -> dict:
    return {
        "models": {
            "m-small": {"model_id": "org/Small", "provider": "hf-router"},
            "m-mid": {"model_id": "org/Mid", "provider": "hf-router"},
        },
        "criteria": {
            "correctness": "m-small",
            "faithfulness": "m-small",
            "completeness": "m-mid",
            "coherence": "m-mid",
            "safety": "m-mid",
        },
    }


def _item():
    return TestItem(id="t1", task_type=TaskType.RAG_QA, task_prompt="q",
                    context="c", candidate_output="a", agent_id="rag_strong")


class FakeJSONClient:
    """Open-model fake: records calls, answers with a strict-JSON verdict."""

    def __init__(self, score: int = 5):
        self.score = score
        self.calls: list[tuple[str, str]] = []  # (model, system)

    def complete_text(self, *, system, user, model=None, temperature=None, max_tokens=None):
        self.calls.append((model, system))
        text = json.dumps(
            {"score": self.score, "passed": self.score >= 4,
             "rationale": "ok", "evidence": ["quote"]}
        )
        return text, CallStats(model=model, prompt_tokens=80, completion_tokens=20,
                               latency_s=0.1, cost_usd=0.0001)


# --- config file loading ----------------------------------------------------

def test_repo_config_maps_benchmark_winners():
    config = load_panel_config(DEFAULT_CONFIG_PATH)
    assert config.spec_for(Criterion.CORRECTNESS).key == "qwen3-8b"
    assert config.spec_for(Criterion.FAITHFULNESS).key == "llama31-8b"
    assert config.spec_for(Criterion.COMPLETENESS).key == "qwen3-14b"
    assert config.spec_for(Criterion.COHERENCE).key == "qwen3-14b"
    assert config.spec_for(Criterion.SAFETY).key == "qwen3-14b"
    assert all(spec.provider == "hf-router" for spec in config.models.values())


def test_missing_criterion_raises(tmp_path):
    raw = _valid_config_dict()
    del raw["criteria"]["safety"]
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(JudgePanelConfigError, match="safety"):
        load_panel_config(path)


def test_unknown_model_key_raises(tmp_path):
    raw = _valid_config_dict()
    raw["criteria"]["coherence"] = "no-such-model"
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(JudgePanelConfigError, match="no-such-model"):
        load_panel_config(path)


def test_unknown_provider_raises(tmp_path):
    raw = _valid_config_dict()
    raw["models"]["m-small"]["provider"] = "carrier-pigeon"
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(JudgePanelConfigError, match="carrier-pigeon"):
        load_panel_config(path)


def test_env_override_and_missing_file(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_valid_config_dict()))
    monkeypatch.setenv("JUDGE_PANEL_CONFIG", str(path))
    config = get_panel_config()
    assert config is not None and config.path == path

    get_panel_config.cache_clear()
    monkeypatch.setenv("JUDGE_PANEL_CONFIG", str(tmp_path / "nope.json"))
    assert get_panel_config() is None  # missing file -> fallback, not an error


# --- panel routing ----------------------------------------------------------

def test_panel_routes_each_criterion_to_mapped_model(monkeypatch):
    fake = FakeJSONClient(score=5)
    monkeypatch.setattr(criteria_mod, "get_hf_router_client", lambda: fake)
    from eval_harness.graph import PanelJudge

    result = PanelJudge().judge(_item())

    config = get_panel_config()
    expected = {c: config.spec_for(c).model_id for c in Criterion}
    actual = {v.criterion: v.model for v in result.verdicts}
    assert actual == expected
    assert result.aggregate_score == 5.0
    assert len(fake.calls) == 5

    # Qwen3 hybrid judges get their /no_think suffix appended to the prompt.
    qwen_systems = [system for model, system in fake.calls if model.startswith("Qwen/")]
    assert qwen_systems and all("/no_think" in s for s in qwen_systems)


def test_forced_model_bypasses_routing(monkeypatch, fake_client_factory):
    structured = fake_client_factory(score_fn=lambda m: 5)
    hf = FakeJSONClient()
    monkeypatch.setattr(criteria_mod, "get_client", lambda: structured)
    monkeypatch.setattr(criteria_mod, "get_hf_router_client", lambda: hf)
    from eval_harness.graph import PanelJudge

    result = PanelJudge(model="gpt-4o-mini").judge(_item())
    assert {v.model for v in result.verdicts} == {"gpt-4o-mini"}
    assert hf.calls == []  # routing never consulted
