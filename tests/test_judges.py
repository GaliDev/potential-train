"""Tests for the baseline, panel, and improvement judges using a fake client."""

import eval_harness.agents.criteria as criteria_mod
import eval_harness.baseline as baseline_mod
from eval_harness.schemas import TaskType, TestItem


def _item():
    return TestItem(id="t1", task_type=TaskType.RAG_QA, task_prompt="q",
                    context="c", candidate_output="a", agent_id="rag_strong")


def test_baseline_judge(monkeypatch, fake_client_factory):
    fake = fake_client_factory(score_fn=lambda m: 5)
    monkeypatch.setattr(baseline_mod, "get_client", lambda: fake)
    from eval_harness.baseline import BaselineJudge

    result = BaselineJudge().judge(_item())
    assert result.judge_mode == "baseline"
    assert result.aggregate_score == 5.0
    assert result.overall_pass is True


def test_panel_judge_collects_all_criteria(monkeypatch, fake_client_factory):
    fake = fake_client_factory(score_fn=lambda m: 5)
    monkeypatch.setattr(criteria_mod, "get_client", lambda: fake)
    from eval_harness.graph import PanelJudge

    result = PanelJudge().judge(_item())
    assert result.judge_mode == "panel"
    assert len(result.verdicts) == 5
    assert result.aggregate_score == 5.0


def test_cascade_escalates_borderline(monkeypatch, fake_client_factory):
    # cheap (mini) -> 4 (borderline), strong -> 5
    fake = fake_client_factory(
        score_fn=lambda m: 4 if "mini" in (m or "") else 5,
        cost_fn=lambda m: 0.0005 if "mini" in (m or "") else 0.004,
    )
    monkeypatch.setattr(criteria_mod, "get_client", lambda: fake)
    from eval_harness.improvement import CascadeJudge

    result = CascadeJudge(band=0.75).judge(_item())
    assert result.judge_mode == "cascade"
    # 5 cheap + 5 strong structured calls
    assert len(fake.calls) == 10
    assert "escalated" in result.rationale


def test_cascade_skips_escalation_when_confident(monkeypatch, fake_client_factory):
    fake = fake_client_factory(score_fn=lambda m: 2)  # far from threshold -> confident
    monkeypatch.setattr(criteria_mod, "get_client", lambda: fake)
    from eval_harness.improvement import CascadeJudge

    result = CascadeJudge(band=0.75).judge(_item())
    assert len(fake.calls) == 5  # cheap only
    assert "cheap-only" in result.rationale


def test_jury_runs_multiple_rounds(monkeypatch, fake_client_factory):
    fake = fake_client_factory(score_fn=lambda m: 5)
    monkeypatch.setattr(criteria_mod, "get_client", lambda: fake)
    from eval_harness.improvement import JuryJudge

    result = JuryJudge(rounds=3).judge(_item())
    assert result.judge_mode == "jury"
    assert len(fake.calls) == 15  # 3 rounds x 5 criteria
    assert len(result.verdicts) == 5
