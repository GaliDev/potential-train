"""Drift and incident detection over operational run signals."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from ..storage import get_store
from .policy_config import DEFAULT_POLICY, PolicyConfig
from .profiles import AgentPerformance, compute_all_performance

# Backward-compatible aliases (single source of truth: PolicyConfig).
ERROR_DRIFT_THRESHOLD = DEFAULT_POLICY.error_drift_threshold
GROUNDEDNESS_DRIFT_THRESHOLD = DEFAULT_POLICY.groundedness_drift_threshold
HIGH_ERROR_RATE = DEFAULT_POLICY.high_error_rate


@dataclass
class DriftAlert:
    agent_id: str
    alert_type: str
    severity: str
    message: str
    metric_value: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def detect_drift_for_agent(
    perf: AgentPerformance, config: PolicyConfig = DEFAULT_POLICY
) -> list[DriftAlert]:
    alerts: list[DriftAlert] = []
    if perf.operational_drift >= config.error_drift_threshold:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="error_rate_drift",
            severity="high",
            message=f"Error rate rising: recent vs older delta {perf.operational_drift:.0%}",
            metric_value=perf.operational_drift,
        ))
    if perf.error_rate >= config.high_error_rate:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="high_error_rate",
            severity="medium",
            message=f"Sustained error rate {perf.error_rate:.0%}",
            metric_value=perf.error_rate,
        ))
    if perf.safety_flag_rate > config.safety_flag_rate_max:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="safety_incident",
            severity="critical",
            message=f"Safety flags in {perf.safety_flag_rate:.0%} of runs",
            metric_value=perf.safety_flag_rate,
        ))
    signals = get_store().fetch_run_signals(agent_id=perf.agent_id)
    if len(signals) >= 4:
        ordered = sorted(signals, key=lambda s: s.created_at)
        mid = len(ordered) // 2
        older_g = [s.groundedness for s in ordered[:mid] if s.groundedness is not None]
        recent_g = [s.groundedness for s in ordered[mid:] if s.groundedness is not None]
        if older_g and recent_g:
            import statistics
            delta = statistics.fmean(recent_g) - statistics.fmean(older_g)
            if delta <= config.groundedness_drift_threshold:
                alerts.append(DriftAlert(
                    agent_id=perf.agent_id,
                    alert_type="groundedness_drift",
                    severity="medium",
                    message=f"Groundedness dropped {delta:.2f} pts (recent vs older)",
                    metric_value=delta,
                ))

    # Per-criterion regression: catch a single criterion (e.g. faithfulness)
    # dropping after a model swap, even when the aggregate looks steady.
    alerts.extend(_criterion_drift(perf.agent_id, config))
    return alerts


def _criterion_mean_scores(rows) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for row in rows:
        try:
            verdicts = json.loads(row.verdicts_json or "[]")
        except json.JSONDecodeError:
            continue
        for v in verdicts:
            crit, score = v.get("criterion"), v.get("score")
            if crit is not None and score is not None:
                sums.setdefault(crit, []).append(float(score))
    return {c: statistics.fmean(vs) for c, vs in sums.items() if vs}


def _criterion_drift(agent_id: str, config: PolicyConfig) -> list[DriftAlert]:
    rows = get_store().fetch_evals(agent_id=agent_id)
    if len(rows) < 4:
        return []
    ordered = sorted(rows, key=lambda r: r.created_at)
    mid = len(ordered) // 2
    older = _criterion_mean_scores(ordered[:mid])
    recent = _criterion_mean_scores(ordered[mid:])
    alerts: list[DriftAlert] = []
    for crit, recent_mean in recent.items():
        if crit in older:
            delta = round(recent_mean - older[crit], 3)
            if delta <= config.criterion_drift_threshold:
                alerts.append(DriftAlert(
                    agent_id=agent_id,
                    alert_type="criterion_drift",
                    severity="high",
                    message=f"{crit} dropped {delta:+.2f} pts (recent vs older)",
                    metric_value=delta,
                ))
    return alerts


def scan_fleet_drift(
    task_type=None, *, audit: bool = True, config: PolicyConfig = DEFAULT_POLICY
) -> list[DriftAlert]:
    """Scan all agents for operational drift and optionally log alerts."""
    all_alerts: list[DriftAlert] = []
    for perf in compute_all_performance(task_type):
        alerts = detect_drift_for_agent(perf, config)
        all_alerts.extend(alerts)
        if audit:
            for alert in alerts:
                get_store().log_audit("drift_alert", alert.agent_id, alert.to_dict())
    return all_alerts
