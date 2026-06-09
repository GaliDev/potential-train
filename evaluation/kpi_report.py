"""Render the assignment's Technical and Business KPI tables from run data.

The rubric's Implementation section asks for two KPI tables -- Technical
(accuracy / latency / uptime / error rate) and Business (cost reduction /
productivity / customer satisfaction / revenue). `metrics.py` already computes
the raw agreement numbers; this module turns them, plus the governance store,
into the exact Target-vs-Achieved tables and writes a markdown artifact.

Sources it reads (all optional; missing data renders as "n/a (no run yet)"):
- a benchmark JSON (output of `evaluation.benchmark`) for judge agreement
  metrics and the cascade cost lever,
- a run JSON (output of `runner` / `evaluation.fleet_run`) for uptime + error
  rate (completed vs failed items),
- the SQLite performance store for governance-derived business value
  (routing savings and governed answer quality).

Every number is either MEASURED (from a real run or the store) or ESTIMATED
from a clearly-stated assumption (human eval time, hourly cost, model list
prices). Nothing is invented.

Run:
    PYTHONPATH=src python -m evaluation.kpi_report
    PYTHONPATH=src python -m evaluation.kpi_report --human-minutes 4 --hourly-cost 40
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from eval_harness.config import RUNS_DIR
from eval_harness.governance.autonomy import calibrate_fleet
from eval_harness.governance.router import route_next_task
from eval_harness.llm import _PER_1M as _MODEL_PRICES
from eval_harness.schemas import AutonomyTier, TaskType
from eval_harness.store import fetch_evals, fetch_run_signals, list_agents

# Targets are decisions made before running, with a defensible basis:
# - 0.80 pass accuracy / 0.60 kappa / 0.70 spearman are the standard bars for a
#   trustworthy judge (kappa >= 0.6 == "substantial agreement", Landis & Koch);
# - latency / uptime / error-rate targets follow the assignment's reference KPIs.
TARGET_ACCURACY = 0.80
TARGET_KAPPA = 0.60
TARGET_SPEARMAN = 0.70
TARGET_LATENCY_S = 2.0
TARGET_UPTIME = 0.99
TARGET_ERROR_RATE = 0.05

# Which judge configuration represents the production system, in preference order.
_PROD_CONFIG_ORDER = ("panel", "jury", "baseline", "cascade")


# --------------------------------------------------------------------------- #
# Row models
# --------------------------------------------------------------------------- #
@dataclass
class TechKpi:
    metric: str
    target: str
    achieved: str
    basis: str = "measured"
    note: str = ""


@dataclass
class BizKpi:
    metric: str
    baseline: str
    after: str
    improvement: str
    basis: str = "measured"
    note: str = ""


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
_NA = "n/a (no run yet)"


def _pct(x: float | None, nd: int = 0) -> str:
    return _NA if x is None else f"{x * 100:.{nd}f}%"


def _num(x: float | None, nd: int = 3) -> str:
    return _NA if x is None else f"{x:.{nd}f}"


def _usd(x: float | None, nd: int = 6) -> str:
    return _NA if x is None else f"${x:.{nd}f}"


def _sec(x: float | None) -> str:
    return _NA if x is None else f"{x:.2f} s"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _price_index(model: str) -> float:
    """A single blended price number per model (avg of input/output $/1M).

    Used only for the routing-savings *estimate*; it assumes comparable token
    usage across models, which is stated as an assumption in the report.
    """
    rates = _MODEL_PRICES.get(model, _MODEL_PRICES["gpt-4o"])
    return (rates["input"] + rates["output"]) / 2.0


# --------------------------------------------------------------------------- #
# Source loading
# --------------------------------------------------------------------------- #
def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _latest(pattern: str) -> Path | None:
    """Newest file matching pattern in RUNS_DIR (timestamped names sort lexically)."""
    files = sorted(RUNS_DIR.glob(pattern))
    return files[-1] if files else None


def _pick_config(bench: dict | None) -> dict | None:
    """Return the production judge's metrics dict from a benchmark report."""
    if not bench:
        return None
    configs = bench.get("configs", {})
    for name in _PROD_CONFIG_ORDER:
        if name in configs:
            return configs[name]
    # Fall back to any available config.
    return next(iter(configs.values()), None)


# --------------------------------------------------------------------------- #
# Technical KPIs
# --------------------------------------------------------------------------- #
def _fleet_operational_kpis() -> dict[str, float | None]:
    """Aggregate operational signals across all run_signals rows."""
    signals = fetch_run_signals()
    if not signals:
        return {}
    latencies = sorted(s.latency_s for s in signals)
    p95_idx = int(0.95 * (len(latencies) - 1))
    tool_rates: list[float] = []
    for s in signals:
        if s.tool_calls > 0:
            tool_rates.append((s.tool_calls - s.tool_failures) / s.tool_calls)
    return {
        "uptime": sum(1 for s in signals if s.success) / len(signals),
        "error_rate": sum(1 for s in signals if not s.success) / len(signals),
        "p95_latency_s": latencies[p95_idx] if latencies else None,
        "tool_success_rate": statistics.fmean(tool_rates) if tool_rates else None,
    }


def technical_kpis(bench: dict | None, run: dict | None) -> list[TechKpi]:
    m = _pick_config(bench)

    acc = m.get("pass_accuracy") if m else None
    kappa = m.get("cohen_kappa") if m else None
    spearman = m.get("spearman") if m else None
    latency = m.get("avg_latency_s") if m else None
    cost_item = m.get("cost_per_item_usd") if m else None

    ops = _fleet_operational_kpis()
    uptime = ops.get("uptime")
    error_rate = ops.get("error_rate")
    p95_latency = ops.get("p95_latency_s")
    tool_success = ops.get("tool_success_rate")

    if uptime is None and run is not None:
        completed = int(run.get("count", 0))
        failed = len(run.get("errors", []))
        attempted = completed + failed
        if attempted > 0:
            uptime = completed / attempted
            error_rate = failed / attempted

    return [
        TechKpi("Accuracy (judge vs human pass/fail)", f">= {TARGET_ACCURACY:.0%}", _pct(acc),
                note="agreement of judge verdicts with human gold labels"),
        TechKpi("Cohen's kappa (judge vs human)", f">= {TARGET_KAPPA:.2f}", _num(kappa, 3),
                note="chance-corrected agreement; >=0.6 is 'substantial'"),
        TechKpi("Spearman (score vs human)", f">= {TARGET_SPEARMAN:.2f}", _num(spearman, 3),
                note="rank correlation on the 1-5 score"),
        TechKpi("Latency / item", f"< {TARGET_LATENCY_S:.0f} s", _sec(latency),
                note="avg wall-clock per evaluation"),
        TechKpi("Cost / item", "report", _usd(cost_item),
                note="avg judge cost per evaluation"),
        TechKpi("Uptime (fleet operations)", f">= {TARGET_UPTIME:.0%}", _pct(uptime),
                note="successful agent executions / total run_signals"),
        TechKpi("Error rate (fleet operations)", f"< {TARGET_ERROR_RATE:.0%}", _pct(error_rate),
                note="failed agent executions / total run_signals"),
        TechKpi("P95 latency (agent execution)", f"< {TARGET_LATENCY_S:.0f} s", _sec(p95_latency),
                note="95th percentile agent wall-clock from run_signals"),
        TechKpi("Tool success rate", ">= 90%", _pct(tool_success),
                note="successful tool calls / total tool calls across fleet"),
    ]


# --------------------------------------------------------------------------- #
# Business KPIs
# --------------------------------------------------------------------------- #
def _cost_lever_kpi(bench: dict | None) -> BizKpi:
    configs = (bench or {}).get("configs", {})
    panel = configs.get("panel")
    cascade = configs.get("cascade")
    base = panel.get("cost_per_item_usd") if panel else None
    after = cascade.get("cost_per_item_usd") if cascade else None
    if base and after and base > 0:
        red = (1 - after / base) * 100
        impr = f"-{red:.0f}%"
    else:
        impr = _NA
    return BizKpi(
        "Judge cost / item (cascade lever)", _usd(base), _usd(after), impr,
        note="cheap-first cascade vs full panel; run benchmark --with-improve",
    )


def _routing_savings_kpi() -> BizKpi:
    agents = list_agents()
    if not agents:
        return BizKpi("Fleet inference cost / task (routing)", _NA, _NA, _NA,
                      basis="estimated", note="no agent history; seed or run the fleet first")

    id_to_model = {a.agent_id: a.model for a in agents}
    base_indices: list[float] = []
    routed_indices: list[float] = []
    for tt in TaskType:
        decision = route_next_task(tt, audit=False)
        if decision.chosen_agent is None:
            continue
        strong = [a for a in agents if a.task_type == tt and a.agent_id.endswith("_strong")]
        strong_model = strong[0].model if strong else "gpt-4o"
        chosen_model = id_to_model.get(decision.chosen_agent, strong_model)
        base_indices.append(_price_index(strong_model))
        routed_indices.append(_price_index(chosen_model))

    base = _mean(base_indices)
    routed = _mean(routed_indices)
    if base and routed is not None and base > 0:
        red = (1 - routed / base) * 100
        impr = f"-{red:.0f}% (est.)"
    else:
        impr = _NA
    return BizKpi(
        "Fleet inference cost / task (routing)",
        _num(base, 2) + " idx" if base is not None else _NA,
        _num(routed, 2) + " idx" if routed is not None else _NA,
        impr,
        basis="estimated (model list price)",
        note="route-to-cheapest-good-enough vs always-strong; blended $/1M index",
    )


def _productivity_kpi(human_minutes: float, hourly_cost: float) -> BizKpi:
    rows = fetch_evals(judge_mode="panel")
    if not rows:
        return BizKpi("Evaluation throughput (productivity)", _NA, _NA, _NA,
                      basis="measured + assumption", note="no panel evals in store yet")

    n = len(rows)
    avg_lat = _mean([r.total_latency_s for r in rows]) or 0.0
    human_rate = 60.0 / human_minutes if human_minutes > 0 else float("inf")
    system_rate = 3600.0 / avg_lat if avg_lat > 0 else None

    human_hours = n * human_minutes / 60.0
    machine_hours = n * avg_lat / 3600.0
    saved_hours = human_hours - machine_hours
    dollars = saved_hours * hourly_cost

    if system_rate is not None and human_rate > 0:
        mult = system_rate / human_rate
        impr = f"{mult:.0f}x throughput - {saved_hours:.1f} eval-hrs - ${dollars:,.0f} saved"
    else:
        impr = _NA
    return BizKpi(
        "Evaluation throughput (productivity)",
        f"{human_rate:.0f}/hr (human est.)",
        f"{system_rate:.0f}/hr (system)" if system_rate is not None else _NA,
        impr,
        basis="measured latency x assumption",
        note=f"assumes {human_minutes:g} min/item human review @ ${hourly_cost:g}/hr; n={n}",
    )


def _served_quality_kpi() -> BizKpi:
    rows = fetch_evals(judge_mode="panel")
    if not rows:
        return BizKpi("Served answer quality (satisfaction proxy)", _NA, _NA, _NA,
                      note="no panel evals in store yet")

    baseline = _mean([1.0 if r.overall_pass else 0.0 for r in rows])
    blocked = {d.agent_id for d in calibrate_fleet(audit=False) if d.tier == AutonomyTier.BLOCKED}
    served = [r for r in rows if r.agent_id not in blocked]
    after = _mean([1.0 if r.overall_pass else 0.0 for r in served]) if served else baseline

    if baseline is not None and after is not None:
        impr = f"+{(after - baseline) * 100:.0f} pts"
    else:
        impr = _NA
    return BizKpi(
        "Served answer quality (satisfaction proxy)", _pct(baseline), _pct(after), impr,
        note=f"fleet pass rate before vs after blocking low-autonomy agents ({len(blocked)} blocked)",
    )


def _revenue_kpi() -> BizKpi:
    return BizKpi(
        "Revenue impact", "-", "-", "projected (out of POC scope)",
        basis="projection",
        note="not measured in a student POC; would track incidents avoided / churn reduced",
    )


def business_kpis(bench: dict | None, *, human_minutes: float, hourly_cost: float) -> list[BizKpi]:
    return [
        _cost_lever_kpi(bench),
        _routing_savings_kpi(),
        _productivity_kpi(human_minutes, hourly_cost),
        _served_quality_kpi(),
        _revenue_kpi(),
    ]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_markdown(
    tech: list[TechKpi],
    biz: list[BizKpi],
    *,
    human_minutes: float,
    hourly_cost: float,
    sources: dict[str, str],
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# KPI Report - Agent Workforce Governance")
    lines.append("")
    lines.append(f"_Generated {stamp}. Every value is **measured** from a run/the store "
                 "or **estimated** from a stated assumption (see Basis); `n/a` means no run "
                 "has produced that number yet._")
    lines.append("")

    lines.append("## Technical KPIs")
    lines.append("")
    lines.append("| Metric | Target | Achieved | Basis | Notes |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in tech:
        lines.append(f"| {r.metric} | {r.target} | {r.achieved} | {r.basis} | {r.note} |")
    lines.append("")

    lines.append("## Business KPIs")
    lines.append("")
    lines.append("| Metric | Baseline | After | Improvement | Basis | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in biz:
        lines.append(
            f"| {r.metric} | {r.baseline} | {r.after} | {r.improvement} | {r.basis} | {r.note} |"
        )
    lines.append("")

    lines.append("## Assumptions")
    lines.append("")
    lines.append(f"- Human evaluation time: **{human_minutes:g} min/item**.")
    lines.append(f"- Loaded labeling cost: **${hourly_cost:g}/hr**.")
    lines.append("- Routing savings use blended model list prices (avg of input/output $/1M), "
                 "assuming comparable token usage across models.")
    lines.append("")

    lines.append("## Data sources")
    lines.append("")
    for label, value in sources.items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    return "\n".join(lines)


def load_latest_sources() -> tuple[dict | None, dict | None, dict[str, str]]:
    """Locate and load the newest benchmark + run JSON in RUNS_DIR.

    Returns ``(bench, run, sources)`` where ``sources`` documents what was
    found. Shared by the CLI and the dashboard/API so they all read the same
    artifacts and report provenance the same way.
    """
    bench_path = _latest("benchmark_*.json")
    run_path = _latest("run_*.json")
    bench = _load_json(bench_path)
    run = _load_json(run_path)
    sources = {
        "benchmark": str(bench_path) if bench else "none found (run evaluation.benchmark)",
        "run": str(run_path) if run else "none found (run evaluation.fleet_run / runner)",
        "store": "data/governance.db (governance + productivity + quality)",
    }
    return bench, run, sources


def generate_kpi_report(
    bench: dict | None,
    run: dict | None,
    *,
    human_minutes: float = 4.0,
    hourly_cost: float = 40.0,
    sources: dict[str, str] | None = None,
) -> str:
    """Build the full markdown KPI report from already-loaded sources."""
    tech = technical_kpis(bench, run)
    biz = business_kpis(bench, human_minutes=human_minutes, hourly_cost=hourly_cost)
    return _render_markdown(
        tech, biz,
        human_minutes=human_minutes, hourly_cost=hourly_cost,
        sources=sources or {"benchmark": "none", "run": "none", "store": "in-process"},
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Technical + Business KPI tables.")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Path to a benchmark_*.json (default: latest in data/runs/).")
    parser.add_argument("--run", type=Path, default=None,
                        help="Path to a run_*.json for uptime/error rate (default: latest).")
    parser.add_argument("--human-minutes", type=float, default=4.0,
                        help="Assumed human minutes per evaluation (productivity baseline).")
    parser.add_argument("--hourly-cost", type=float, default=40.0,
                        help="Assumed loaded labeling cost per hour (USD).")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "kpi_report.md",
                        help="Where to write the markdown report.")
    args = parser.parse_args()

    if args.benchmark is None and args.run is None:
        bench, run, sources = load_latest_sources()
    else:
        bench_path = args.benchmark or _latest("benchmark_*.json")
        run_path = args.run or _latest("run_*.json")
        bench = _load_json(bench_path)
        run = _load_json(run_path)
        sources = {
            "benchmark": str(bench_path) if bench else "none found (run evaluation.benchmark)",
            "run": str(run_path) if run else "none found (run evaluation.fleet_run / runner)",
            "store": "data/governance.db (governance + productivity + quality)",
        }

    report = generate_kpi_report(
        bench, run,
        human_minutes=args.human_minutes, hourly_cost=args.hourly_cost,
        sources=sources,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nSaved KPI report to {args.out}")


if __name__ == "__main__":
    main()
