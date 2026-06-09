"""Drift and incident detection over operational run signals."""

from __future__ import annotations

from dataclasses import dataclass

from ..store import fetch_run_signals, log_audit
from .profiles import AgentPerformance, compute_all_performance

ERROR_DRIFT_THRESHOLD = 0.15
GROUNDEDNESS_DRIFT_THRESHOLD = -0.2
HIGH_ERROR_RATE = 0.2


@dataclass
class DriftAlert:
    agent_id: str
    alert_type: str
    severity: str
    message: str
    metric_value: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def detect_drift_for_agent(perf: AgentPerformance) -> list[DriftAlert]:
    alerts: list[DriftAlert] = []
    if perf.operational_drift >= ERROR_DRIFT_THRESHOLD:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="error_rate_drift",
            severity="high",
            message=f"Error rate rising: recent vs older delta {perf.operational_drift:.0%}",
            metric_value=perf.operational_drift,
        ))
    if perf.error_rate >= HIGH_ERROR_RATE:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="high_error_rate",
            severity="medium",
            message=f"Sustained error rate {perf.error_rate:.0%}",
            metric_value=perf.error_rate,
        ))
    if perf.safety_flag_rate > 0.05:
        alerts.append(DriftAlert(
            agent_id=perf.agent_id,
            alert_type="safety_incident",
            severity="critical",
            message=f"Safety flags in {perf.safety_flag_rate:.0%} of runs",
            metric_value=perf.safety_flag_rate,
        ))
    signals = fetch_run_signals(agent_id=perf.agent_id)
    if len(signals) >= 4:
        ordered = sorted(signals, key=lambda s: s.created_at)
        mid = len(ordered) // 2
        older_g = [s.groundedness for s in ordered[:mid] if s.groundedness is not None]
        recent_g = [s.groundedness for s in ordered[mid:] if s.groundedness is not None]
        if older_g and recent_g:
            import statistics
            delta = statistics.fmean(recent_g) - statistics.fmean(older_g)
            if delta <= GROUNDEDNESS_DRIFT_THRESHOLD:
                alerts.append(DriftAlert(
                    agent_id=perf.agent_id,
                    alert_type="groundedness_drift",
                    severity="medium",
                    message=f"Groundedness dropped {delta:.2f} pts (recent vs older)",
                    metric_value=delta,
                ))
    return alerts


def scan_fleet_drift(task_type=None, *, audit: bool = True) -> list[DriftAlert]:
    """Scan all agents for operational drift and optionally log alerts."""
    all_alerts: list[DriftAlert] = []
    for perf in compute_all_performance(task_type):
        alerts = detect_drift_for_agent(perf)
        all_alerts.extend(alerts)
        if audit:
            for alert in alerts:
                log_audit("drift_alert", alert.agent_id, alert.to_dict())
    return all_alerts
