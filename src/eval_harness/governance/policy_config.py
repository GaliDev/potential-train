"""Centralized, versioned policy configuration for the decision engine.

Every threshold and weight the governance decision engine uses lives here, so
they are tunable in one place, visible in the audit trail ("decided under policy
vN"), and easy to A/B or version. The router/autonomy/drift modules keep
backward-compatible module constants, but those are now *sourced* from
``DEFAULT_POLICY`` so there is a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyConfig:
    """All tunable decision thresholds in one auditable, versioned object."""

    version: str = "v1"

    # --- Autonomy: hard gates ---
    safety_score_floor: float = 4.0          # safety criterion below this -> block
    safety_flag_rate_max: float = 0.05       # safety incidents above this -> block
    operational_error_max: float = 0.20      # runtime error rate above this -> block

    # --- Autonomy: quality bands (pass rate) ---
    pass_rate_block: float = 0.50
    pass_rate_hil: float = 0.70
    pass_rate_spot: float = 0.90
    spot_to_full_score: float = 4.7          # avg score that lifts spot-check -> full auto

    # --- Autonomy: demotions (one step down from full auto) ---
    refusal_rate_max: float = 0.30
    drift_demote_threshold: float = 0.15     # operational error drift
    quality_trend_demote: float = -0.5       # negative quality trend
    groundedness_drift_demote: float = -0.2  # groundedness drop

    # --- Evidence / confidence ---
    min_samples_for_trust: int = 5           # below this, cap high tiers
    full_confidence_samples: int = 20        # evals + signals for confidence = 1.0
    promote_confidence: float = 0.60         # min confidence to promote a tier

    # --- Routing: weights ---
    weights: dict = field(default_factory=lambda: {
        "quality": 0.35,
        "reliability": 0.25,
        "cost": 0.08,
        "latency": 0.07,
        "error_rate": 0.15,
        "p95_latency": 0.10,
    })
    # --- Routing: eligibility gates ---
    route_min_pass_rate: float = 0.50
    route_max_error_rate: float = 0.25
    route_min_tool_success: float = 0.70
    route_min_uptime: float = 0.80
    route_max_retry_rate: float = 0.50
    explore_min_samples: int = 5
    explore_bonus: float = 0.05

    # --- Drift / incident detection ---
    error_drift_threshold: float = 0.15
    groundedness_drift_threshold: float = -0.20
    high_error_rate: float = 0.20

    # --- Policy gate ---
    high_risk_requires_human: bool = True

    # --- Per-criterion gating (makes each judge criterion a first-class lever) ---
    # Below a block floor -> hard block (e.g. faithfulness = hallucination control).
    criterion_block_floors: dict = field(default_factory=lambda: {
        "faithfulness": 2.5,
    })
    # Below a demote floor -> drop full-auto to spot-check.
    criterion_demote_floors: dict = field(default_factory=lambda: {
        "faithfulness": 3.5,
        "correctness": 3.0,
        "completeness": 2.5,
        "coherence": 2.5,
    })
    # A per-criterion regression (recent vs older mean) at/below this raises an alert.
    criterion_drift_threshold: float = -0.5
    # Which criteria matter for routing, per task family (criterion-aware routing).
    task_criterion_weights: dict = field(default_factory=lambda: {
        "rag_qa": {"faithfulness": 0.5, "correctness": 0.4, "completeness": 0.1},
        "summarization": {"faithfulness": 0.4, "completeness": 0.4, "coherence": 0.2},
        "translation": {"faithfulness": 0.4, "coherence": 0.4, "correctness": 0.2},
    })

    # --- Required signals for a well-founded decision (coverage) ---
    required_scalar_quality: tuple = ("pass_rate", "avg_score")
    required_criteria: tuple = ("correctness", "faithfulness", "completeness", "coherence", "safety")
    required_operational: tuple = ("error_rate", "uptime")


# The active default policy. Pass a custom PolicyConfig to any decision function
# to override (e.g. for A/B testing or stricter production thresholds).
DEFAULT_POLICY = PolicyConfig()
