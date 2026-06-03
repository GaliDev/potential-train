"""FastAPI service exposing the governance platform.

Read-only views over the performance store plus decision endpoints (routing,
policy). Run with:

    PYTHONPATH=src uvicorn app.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from eval_harness.governance import autonomy, policy, reviews, router
from eval_harness.governance.profiles import compute_all_performance
from eval_harness.schemas import TaskType
from eval_harness.store import fetch_audit, list_agents

app = FastAPI(title="Agent Workforce Governance", version="0.1.0")


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
