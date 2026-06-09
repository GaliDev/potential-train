"""Streamlit dashboard for the Agent Workforce Governance platform.

Run with:

    PYTHONPATH=src streamlit run app/ui.py

Surfaces the four governance decisions on top of the evaluation store:
fleet leaderboard, autonomy tiers, task routing, performance reviews, and
policy + audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the src/ package importable when launched via `streamlit run`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval_harness.demo_seed import seed_demo_data  # noqa: E402
from eval_harness.governance import autonomy, policy, reviews, router  # noqa: E402
from eval_harness.governance.drift import scan_fleet_drift  # noqa: E402
from eval_harness.governance.profiles import compute_all_performance  # noqa: E402
from eval_harness.schemas import TaskType  # noqa: E402
from eval_harness.simulate_runtime import DriftScenario, simulate_runtime  # noqa: E402
from eval_harness.store import fetch_audit, fetch_run_signals, list_agents  # noqa: E402

st.set_page_config(page_title="Agent Workforce Governance", layout="wide")

TASK_OPTIONS = [tt.value for tt in TaskType]

TIER_LABELS = {
    "full_auto": "Full auto",
    "auto_spot_check": "Auto + spot check",
    "human_in_loop": "Human in the loop",
    "blocked": "Blocked",
}


def _task_type_from_label(label: str) -> TaskType | None:
    return None if label == "All" else TaskType(label)


# --- Sidebar -------------------------------------------------------------
st.sidebar.title("Agent Workforce Governance")
task_label = st.sidebar.selectbox("Task type", ["All", *TASK_OPTIONS])
task_type = _task_type_from_label(task_label)

st.sidebar.markdown("---")
if st.sidebar.button("Seed demo data"):
    count = seed_demo_data()
    st.sidebar.success(f"Seeded {count} synthetic evaluations.")
if st.sidebar.button("Simulate runtime (30d)"):
    count = simulate_runtime(
        n_per_agent=30,
        scenarios=[
            DriftScenario(agent_id="rag_weak", safety_incident=True),
            DriftScenario(agent_id="sum_weak", latency_multiplier=3.0),
        ],
    )
    st.sidebar.success(f"Simulated {count} eval + signal pairs.")

has_data = bool(compute_all_performance(task_type))
if not has_data:
    st.info("No evaluation history yet. Click **Seed demo data** in the sidebar, "
            "or run the benchmark with your OpenAI key to populate real results.")

st.title("Agent Workforce Governance")
st.caption(
    "Manage a fleet of RAG, summarization, and translation agents like a team - "
    "the four questions eval platforms skip."
)

tabs = st.tabs(
    [
        "Fleet leaderboard",
        "Operations",
        "Autonomy tiers",
        "Task routing",
        "Performance reviews",
        "Governance & audit",
    ]
)

# --- Fleet leaderboard ---------------------------------------------------
with tabs[0]:
    st.subheader("Which agents perform best?")
    profiles = compute_all_performance(task_type)
    if profiles:
        df = pd.DataFrame([p.to_dict() for p in profiles])
        cols = ["agent_id", "task_type", "n_evals", "pass_rate", "avg_score",
                "score_std", "trend", "avg_latency_s", "avg_cost_usd"]
        df = df[cols].sort_values("avg_score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("agent_id")[["avg_score"]])
        st.caption("Per-criterion averages")
        crit_df = pd.DataFrame(
            {p.agent_id: p.per_criterion_avg for p in profiles}
        ).T
        st.dataframe(crit_df, use_container_width=True)
    else:
        st.write("No data.")

# --- Operations ----------------------------------------------------------
with tabs[1]:
    st.subheader("Operational runtime signals")
    profiles = compute_all_performance(task_type)
    if profiles:
        ops_cols = [
            "agent_id", "n_signals", "uptime", "error_rate", "p95_latency_s",
            "tool_success_rate", "retry_rate", "refusal_rate",
            "avg_groundedness", "avg_tokens", "operational_drift",
        ]
        ops_df = pd.DataFrame([p.to_dict() for p in profiles])[ops_cols]
        st.dataframe(ops_df.sort_values("error_rate"), use_container_width=True, hide_index=True)
        st.line_chart(
            ops_df.set_index("agent_id")[["uptime", "error_rate"]],
        )
    else:
        st.write("No operational data.")

    signals = fetch_run_signals()
    if signals:
        sig_df = pd.DataFrame([
            {
                "ts": s.created_at.isoformat(),
                "agent_id": s.agent_id,
                "success": s.success,
                "latency_s": s.latency_s,
                "tool_calls": s.tool_calls,
                "retries": s.retries,
            }
            for s in signals
        ])
        st.caption("Recent execution signals")
        st.dataframe(sig_df.tail(50), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Drift & alerts")
    alerts = scan_fleet_drift(task_type, audit=False)
    if alerts:
        st.dataframe(
            pd.DataFrame([a.to_dict() for a in alerts]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No drift detected.")

# --- Autonomy tiers ------------------------------------------------------
with tabs[2]:
    st.subheader("How much autonomy has each agent earned?")
    decisions = autonomy.calibrate_fleet(task_type, audit=False)
    if decisions:
        rows = []
        for d in decisions:
            rows.append({
                "agent_id": d.agent_id,
                "tier": TIER_LABELS.get(d.tier.value, d.tier.value),
                "pass_rate": d.pass_rate,
                "avg_score": d.avg_score,
                "n_evals": d.n_evals,
                "rationale": d.rationale,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No data.")

# --- Task routing --------------------------------------------------------
with tabs[3]:
    st.subheader("Which agent should get the next task?")
    route_tt = task_type or TaskType.RAG_QA
    st.write(f"Routing for task type: **{route_tt.value}**")
    if st.button("Route next task"):
        decision = router.route_next_task(route_tt, audit=True)
        if decision.chosen_agent:
            st.success(f"Chosen agent: {decision.chosen_agent}")
        else:
            st.warning("No eligible agent - escalate to a human.")
        st.write(decision.rationale)
        st.dataframe(
            pd.DataFrame([s.__dict__ for s in decision.ranking]),
            use_container_width=True, hide_index=True,
        )

# --- Performance reviews -------------------------------------------------
with tabs[4]:
    st.subheader("Agent performance reviews")
    agents = list_agents(task_type)
    if agents:
        agent_id = st.selectbox("Agent", [a.agent_id for a in agents])
        use_llm = st.checkbox("Use LLM to write the review (needs API key)", value=False)
        if st.button("Generate review"):
            review = reviews.generate_review(agent_id, use_llm=use_llm)
            if review is None:
                st.warning("No history for this agent.")
            else:
                st.markdown(f"**Recommended autonomy:** {TIER_LABELS.get(review.tier, review.tier)}")
                st.write(review.narrative)
                st.json(review.stats)
    else:
        st.write("No agents registered.")

# --- Governance & audit --------------------------------------------------
with tabs[5]:
    st.subheader("Policy check")
    agents = list_agents(task_type)
    if agents:
        c1, c2, c3 = st.columns(3)
        agent_id = c1.selectbox("Agent", [a.agent_id for a in agents], key="policy_agent")
        pol_tt = c2.selectbox("Task type", TASK_OPTIONS, key="policy_tt")
        risk = c3.selectbox("Task risk", [r.value for r in policy.TaskRisk], key="policy_risk")
        if st.button("Evaluate policy"):
            decision = policy.decide_for_agent(agent_id, TaskType(pol_tt), policy.TaskRisk(risk))
            if decision is None:
                st.warning("No history for this agent.")
            else:
                verdict = "ALLOWED" if decision.allow else "BLOCKED"
                approval = " (requires human approval)" if decision.requires_approval else ""
                (st.success if decision.allow else st.error)(f"{verdict}{approval}")
                for reason in decision.reasons:
                    st.write(f"- {reason}")

    st.markdown("---")
    st.subheader("Audit log")
    audit = fetch_audit(limit=100)
    if audit:
        st.dataframe(
            pd.DataFrame([
                {"ts": r.ts.isoformat(), "actor": r.actor, "action": r.action, "subject": r.subject}
                for r in audit
            ]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No audit entries yet.")
