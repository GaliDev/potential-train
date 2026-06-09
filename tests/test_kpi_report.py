"""Offline tests for the KPI report (no API key, synthetic data + seeded store)."""

from __future__ import annotations

from eval_harness.demo_seed import seed_demo_data
from evaluation.kpi_report import (
    business_kpis,
    generate_kpi_report,
    technical_kpis,
)


def _bench(panel_cpi: float = 0.004, cascade_cpi: float | None = 0.0015) -> dict:
    """A benchmark-report-shaped dict with panel (+ optional cascade) metrics."""
    configs = {
        "panel": {
            "n": 16,
            "pass_accuracy": 0.875,
            "cohen_kappa": 0.71,
            "spearman": 0.74,
            "avg_latency_s": 1.4,
            "cost_per_item_usd": panel_cpi,
        }
    }
    if cascade_cpi is not None:
        configs["cascade"] = {"n": 16, "cost_per_item_usd": cascade_cpi}
    return {"n_items": 16, "configs": configs}


def _run(count: int = 18, errors: int = 2) -> dict:
    return {"count": count, "errors": [{"item_id": f"e{i}", "error": "boom"} for i in range(errors)]}


# --------------------------------------------------------------------------- #
# Technical KPIs
# --------------------------------------------------------------------------- #
def test_technical_kpis_pull_metrics_and_targets():
    rows = {r.metric: r for r in technical_kpis(_bench(), _run())}
    assert rows["Accuracy (judge vs human pass/fail)"].achieved == "88%"
    assert rows["Cohen's kappa (judge vs human)"].achieved == "0.710"
    assert rows["Spearman (score vs human)"].achieved == "0.740"
    assert rows["Latency / item"].achieved == "1.40 s"


def test_uptime_and_error_rate_from_run():
    rows = {r.metric: r for r in technical_kpis(_bench(), _run(count=18, errors=2))}
    # 18 completed / 20 attempted = 90% uptime, 10% error rate.
    assert rows["Uptime (fleet operations)"].achieved == "90%"
    assert rows["Error rate (fleet operations)"].achieved == "10%"


def test_technical_kpis_degrade_without_data():
    rows = {r.metric: r for r in technical_kpis(None, None)}
    assert rows["Accuracy (judge vs human pass/fail)"].achieved.startswith("n/a")
    assert rows["Uptime (fleet operations)"].achieved.startswith("n/a")


# --------------------------------------------------------------------------- #
# Business KPIs
# --------------------------------------------------------------------------- #
def test_cost_lever_reduction_computed():
    biz = {r.metric: r for r in business_kpis(_bench(panel_cpi=0.004, cascade_cpi=0.001),
                                              human_minutes=4, hourly_cost=40)}
    # (1 - 0.001/0.004) = 75% reduction.
    assert biz["Judge cost / item (cascade lever)"].improvement == "-75%"


def test_cost_lever_na_without_cascade():
    biz = {r.metric: r for r in business_kpis(_bench(cascade_cpi=None),
                                              human_minutes=4, hourly_cost=40)}
    assert biz["Judge cost / item (cascade lever)"].improvement.startswith("n/a")


def test_business_kpis_with_seeded_store():
    seed_demo_data(n_per_agent=12, seed=1)
    biz = {r.metric: r for r in business_kpis(_bench(), human_minutes=4, hourly_cost=40)}

    prod = biz["Evaluation throughput (productivity)"]
    assert "throughput" in prod.improvement
    assert "eval-hrs" in prod.improvement
    assert "|" not in prod.improvement  # must not break the markdown table

    # Blocking the weak agents should not lower served quality.
    quality = biz["Served answer quality (satisfaction proxy)"]
    assert quality.improvement.endswith("pts")
    assert not quality.baseline.startswith("n/a")

    # Routing produces an estimate (>=0% savings) once history exists.
    routing = biz["Fleet inference cost / task (routing)"]
    assert routing.improvement != "n/a (no run yet)"


def test_routing_savings_na_without_history():
    # No seed -> empty store -> routing has nothing to compare.
    biz = {r.metric: r for r in business_kpis(None, human_minutes=4, hourly_cost=40)}
    assert biz["Fleet inference cost / task (routing)"].improvement.startswith("n/a")


# --------------------------------------------------------------------------- #
# End-to-end render
# --------------------------------------------------------------------------- #
def test_generate_report_renders_both_tables():
    seed_demo_data(n_per_agent=8, seed=2)
    md = generate_kpi_report(_bench(), _run(), human_minutes=4, hourly_cost=40)
    assert "## Technical KPIs" in md
    assert "## Business KPIs" in md
    assert "## Assumptions" in md
    assert "Cohen's kappa" in md


def test_generate_report_no_data_is_graceful():
    md = generate_kpi_report(None, None, human_minutes=4, hourly_cost=40)
    assert "## Technical KPIs" in md
    assert "n/a (no run yet)" in md
