"""FastAPI service exposing the governance platform.

Read-only views over the performance store plus decision endpoints (routing,
policy). Run with:

    PYTHONPATH=src uvicorn app.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

from evaluation import kpi_report

from eval_harness.governance import autonomy, engine, policy, reviews, router
from eval_harness.governance.policy import TaskRisk
from eval_harness.governance.profiles import compute_all_performance
from eval_harness.schemas import TaskType
from eval_harness.store import fetch_audit, list_agents

app = FastAPI(title="Agent Workforce Governance", version="0.1.0")

# Allow the static console (opened from file:// or a dev server) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_task_type(value: str | None) -> TaskType | None:
    if value is None:
        return None
    try:
        return TaskType(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown task_type: {value}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/agents")
def get_agents(task_type: str | None = None) -> list[dict]:
    return [a.model_dump(mode="json") for a in list_agents(_parse_task_type(task_type))]


@app.get("/performance")
def get_performance(task_type: str | None = None) -> list[dict]:
    return [p.to_dict() for p in compute_all_performance(_parse_task_type(task_type))]


@app.get("/autonomy")
def get_autonomy(task_type: str | None = None) -> list[dict]:
    return [d.to_dict() for d in autonomy.calibrate_fleet(_parse_task_type(task_type), audit=False)]


def _parse_task_risk(value: str) -> TaskRisk:
    try:
        return TaskRisk(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown task_risk: {value}") from exc


def _agent_task_type(agent_id: str) -> TaskType | None:
    for a in list_agents():
        if a.agent_id == agent_id:
            return a.task_type
    return None


@app.get("/decision/{agent_id}")
def get_decision(
    agent_id: str, task_type: str | None = None, task_risk: str = "low"
) -> dict:
    """The fully resolved governance decision for one agent.

    This is what the console's decision spotlight renders: tier, verdict,
    confidence, precedence chain, drift alerts, and rationale. If task_type is
    omitted it is taken from the agent's registered profile.
    """
    tt = _parse_task_type(task_type) or _agent_task_type(agent_id)
    if tt is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_id}")
    return engine.decide_for_agent(
        agent_id, tt, _parse_task_risk(task_risk), audit=False
    ).to_dict()


@app.get("/decisions")
def get_decisions(task_type: str | None = None, task_risk: str = "low") -> list[dict]:
    """Resolved decisions for the whole fleet - drives the fleet table."""
    risk = _parse_task_risk(task_risk)
    profiles = compute_all_performance(_parse_task_type(task_type))
    out = []
    for p in profiles:
        tt = TaskType(p.task_type)
        out.append(engine.decide_for_agent(p.agent_id, tt, risk, audit=False).to_dict())
    return out


@app.post("/route")
def post_route(task_type: str) -> dict:
    tt = _parse_task_type(task_type)
    if tt is None:
        raise HTTPException(status_code=400, detail="task_type is required")
    return router.route_next_task(tt).to_dict()


@app.get("/review/{agent_id}")
def get_review(agent_id: str, use_llm: bool = False) -> dict:
    review = reviews.generate_review(agent_id, use_llm=use_llm)
    if review is None:
        raise HTTPException(status_code=404, detail=f"No history for agent {agent_id}")
    return review.to_dict()


class PolicyRequest(BaseModel):
    agent_id: str
    task_type: str
    task_risk: str = "low"


@app.post("/policy")
def post_policy(req: PolicyRequest) -> dict:
    tt = _parse_task_type(req.task_type)
    if tt is None:
        raise HTTPException(status_code=400, detail="task_type is required")
    try:
        risk = policy.TaskRisk(req.task_risk)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown task_risk: {req.task_risk}") from exc
    decision = policy.decide_for_agent(req.agent_id, tt, risk)
    if decision is None:
        raise HTTPException(status_code=404, detail=f"No history for agent {req.agent_id}")
    return decision.to_dict()


@app.get("/kpi")
def get_kpi(human_minutes: float = 4.0, hourly_cost: float = 40.0) -> dict:
    """KPIs split into Platform (the judge/governance system) and Agent (the fleet)."""
    bench, run, sources = kpi_report.load_latest_sources()
    return {
        "platform": {
            "performance": [r.__dict__ for r in kpi_report.judge_kpis(bench)],
            "value": [r.__dict__ for r in kpi_report.platform_value_kpis(
                bench, human_minutes=human_minutes, hourly_cost=hourly_cost)],
        },
        "agents": {
            "quality": [r.__dict__ for r in kpi_report.agent_quality_kpis()],
            "reliability": [r.__dict__ for r in kpi_report.agent_reliability_kpis(run)],
        },
        "assumptions": {"human_minutes": human_minutes, "hourly_cost": hourly_cost},
        "sources": sources,
        "markdown": kpi_report.generate_kpi_report(
            bench, run, human_minutes=human_minutes, hourly_cost=hourly_cost, sources=sources,
        ),
    }


@app.get("/audit")
def get_audit(limit: int = 100) -> list[dict]:
    return [
        {
            "ts": r.ts.isoformat(),
            "actor": r.actor,
            "action": r.action,
            "subject": r.subject,
            "detail": r.detail_json,
        }
        for r in fetch_audit(limit=limit)
    ]


@app.post("/admin/seed")
def admin_seed() -> dict:
    """Seed demo data."""
    try:
        from eval_harness.demo_seed import seed_demo_data
        seed_demo_data()
        return {"status": "success", "message": "Demo data seeded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/simulate")
def admin_simulate() -> dict:
    """Simulate runtime with drift scenarios."""
    try:
        from eval_harness.simulate_runtime import simulate_runtime, DriftScenario
        simulate_runtime(
            n_per_agent=30,
            scenarios=[
                DriftScenario(agent_id="rag_weak", safety_incident=True),
                DriftScenario(agent_id="sum_weak", latency_multiplier=3.0),
            ],
        )
        return {"status": "success", "message": "Runtime simulation completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Console buttons pass the module name; map it to the registered agent_id.
_AGENT_NAME_TO_ID = {
    "rag_react_agent": "rag_react",
    "summarizer_refine_agent": "sum_refine",
    "translator_backcheck_agent": "trans_backcheck",
}


@app.post("/admin/agent/{agent_name}")
def admin_run_agent(agent_name: str, limit: int = 1) -> dict:
    """Run a LangGraph agent for real and persist its judged scores.

    Invokes the agent's graph over `limit` gold tasks, records the run signal
    (operational telemetry), then scores each output with the 5-judge panel and
    persists — so the agent shows up in the dashboard with true scores.

    The panel judges call the OpenAI API, so this needs OPENAI_API_KEY. Without
    a key, use POST /admin/seed for offline synthetic scores instead.
    """
    from eval_harness.config import settings
    from eval_harness.datasets.loaders import load_gold
    from eval_harness.fleet.registry import get_registered_agent, produce_item
    from eval_harness.graph import PanelJudge
    from eval_harness.runner import run_evaluation
    from eval_harness.store import record_run_signal, upsert_agent

    agent_id = _AGENT_NAME_TO_ID.get(agent_name, agent_name)
    try:
        agent = get_registered_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_id}") from exc

    if not settings.has_openai_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "No OPENAI_API_KEY set - the 5-judge panel needs it to score a real run. "
                "Add a key to .env, or use 'Seed data' for offline synthetic scores."
            ),
        )

    tt = agent.profile.task_type
    tasks = load_gold(tt)[: max(1, limit)]
    if not tasks:
        raise HTTPException(status_code=404, detail=f"No gold tasks for {tt.value}")

    upsert_agent(agent.profile)
    judge = PanelJudge()
    scores = []
    try:
        for task in tasks:
            produced, trace = produce_item(agent, task)
            record_run_signal(
                item_id=produced.id, agent_id=agent.agent_id, task_type=tt, trace=trace
            )
            rep = run_evaluation([produced], judge, persist=True, save_run=False)
            for r in rep.results:
                scores.append(
                    {"item": r.item_id, "score": r.aggregate_score, "pass": r.overall_pass}
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "success",
        "agent": agent_id,
        "evaluated": len(scores),
        "scores": scores,
    }


# Mount the static console at the root.
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="console")
