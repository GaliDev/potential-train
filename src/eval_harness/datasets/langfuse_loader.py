"""Pull traces from Langfuse and map them onto TestItems; push scores back.

This is the bridge between Langfuse (observability: what the agents actually
did in production) and the judge panel (quality: how well they did it):

1. `fetch_traces` pages through GET /api/public/traces for a time window.
2. `trace_to_test_item` maps a Langfuse trace onto our `TestItem` schema so
   the existing panel (correctness/faithfulness/completeness/coherence/safety)
   can score it unchanged.
3. `push_result_scores` writes the verdicts back via POST /api/public/scores,
   so each trace shows the five criterion scores in the Langfuse UI.

Talks to the Langfuse Public API directly over HTTP (basic auth with the
project's public/secret key pair) - no SDK dependency, works the same against
self-hosted and cloud.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import settings
from ..schemas import AggregateResult, TaskType, TestItem

# Trace id prefix so judge results remain traceable back to Langfuse.
ITEM_ID_PREFIX = "lf_"

# How trace tags / metadata / names map onto our task families.
_TASK_TYPE_HINTS: dict[str, TaskType] = {
    "rag": TaskType.RAG_QA,
    "rag_qa": TaskType.RAG_QA,
    "qa": TaskType.RAG_QA,
    "question": TaskType.RAG_QA,
    "summarization": TaskType.SUMMARIZATION,
    "summarize": TaskType.SUMMARIZATION,
    "summary": TaskType.SUMMARIZATION,
    "translation": TaskType.TRANSLATION,
    "translate": TaskType.TRANSLATION,
}

# Observation names that look like retrieval steps (context for faithfulness).
_CONTEXT_NAME_HINTS = ("retriev", "context", "search", "lookup", "vector")


class LangfuseClient:
    """Minimal client for the Langfuse Public API (read traces, write scores)."""

    def __init__(
        self,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.host = (host or settings.langfuse_host).rstrip("/")
        public_key = public_key or settings.langfuse_public_key
        secret_key = secret_key or settings.langfuse_secret_key
        if not (public_key and secret_key):
            raise RuntimeError(
                "Langfuse credentials missing. Set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY (and LANGFUSE_HOST) in your .env."
            )
        self._http = httpx.Client(
            base_url=self.host,
            auth=httpx.BasicAuth(public_key, secret_key),
            timeout=timeout_s,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        resp = self._http.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Langfuse API {method} {path} failed ({resp.status_code}): {resp.text[:500]}"
            )
        return resp.json()

    def list_traces(
        self,
        from_timestamp: datetime,
        to_timestamp: datetime | None = None,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        limit: int = 50,
    ) -> dict:
        """One page of GET /api/public/traces (returns {data, meta})."""
        params: dict[str, Any] = {
            "page": page,
            "limit": limit,
            "fromTimestamp": from_timestamp.astimezone(timezone.utc).isoformat(),
            "fields": "core,io",
            "orderBy": "timestamp.desc",
        }
        if to_timestamp:
            params["toTimestamp"] = to_timestamp.astimezone(timezone.utc).isoformat()
        if name:
            params["name"] = name
        if user_id:
            params["userId"] = user_id
        if session_id:
            params["sessionId"] = session_id
        if tags:
            params["tags"] = tags
        return self._request("GET", "/api/public/traces", params=params)

    def get_trace(self, trace_id: str) -> dict:
        """Full trace including observations (GET /api/public/traces/{id})."""
        return self._request("GET", f"/api/public/traces/{trace_id}")

    def create_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
        observation_id: str | None = None,
        data_type: str = "NUMERIC",
    ) -> dict:
        """Attach a score to a trace (POST /api/public/scores)."""
        body: dict[str, Any] = {
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": data_type,
        }
        if comment:
            body["comment"] = comment[:1000]
        if observation_id:
            body["observationId"] = observation_id
        return self._request("POST", "/api/public/scores", json=body)

    def create_trace(
        self,
        *,
        name: str,
        input: Any,
        output: Any,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create a trace via the ingestion endpoint; returns the trace id.

        Uses POST /api/public/ingestion with a single `trace-create` event, the
        same Public-API-over-HTTP path as the rest of this client (no SDK). The
        worker ingests asynchronously, so a freshly created trace may take a few
        seconds to become queryable via list_traces.
        """
        trace_id = trace_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        body: dict[str, Any] = {
            "id": trace_id,
            "name": name,
            "input": input,
            "output": output,
            "metadata": metadata or {},
            "tags": tags or [],
            "timestamp": now,
        }
        if user_id:
            body["userId"] = user_id
        if session_id:
            body["sessionId"] = session_id
        event = {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now,
            "body": body,
        }
        self._request("POST", "/api/public/ingestion", json={"batch": [event]})
        return trace_id

    def close(self) -> None:
        self._http.close()


def _as_text(value: Any) -> str:
    """Flatten Langfuse JSON input/output (str, chat messages, dict) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        # Chat-message style: [{"role": ..., "content": ...}, ...] -> last
        # message with content; otherwise join the flattened elements.
        contents = [
            _as_text(m.get("content"))
            for m in value
            if isinstance(m, dict) and m.get("content")
        ]
        if contents:
            return contents[-1]
        return "\n".join(_as_text(v) for v in value if v)
    if isinstance(value, dict):
        for key in ("content", "output", "answer", "text", "input", "question", "query"):
            if value.get(key):
                return _as_text(value[key])
        if value.get("messages"):
            return _as_text(value["messages"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def infer_task_type(trace: dict, default: TaskType = TaskType.RAG_QA) -> TaskType:
    """Guess the task family from tags, metadata, or the trace name."""
    metadata = trace.get("metadata") or {}
    if isinstance(metadata, dict):
        declared = str(metadata.get("task_type", "")).lower()
        if declared in {t.value for t in TaskType}:
            return TaskType(declared)
    candidates = [str(t).lower() for t in (trace.get("tags") or [])]
    candidates.append(str(trace.get("name") or "").lower())
    for text in candidates:
        for hint, task_type in _TASK_TYPE_HINTS.items():
            if hint in text:
                return task_type
    return default


def _extract_context(trace: dict) -> str | None:
    """Pull grounding context from metadata or retrieval-shaped observations."""
    metadata = trace.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("context", "retrieved_context", "documents", "sources"):
            if metadata.get(key):
                return _as_text(metadata[key])
    chunks: list[str] = []
    for obs in trace.get("observations") or []:
        if not isinstance(obs, dict):
            continue  # list endpoint returns observation ids, not objects
        obs_name = str(obs.get("name") or "").lower()
        if any(hint in obs_name for hint in _CONTEXT_NAME_HINTS) and obs.get("output"):
            chunks.append(_as_text(obs["output"]))
    return "\n\n".join(chunks) if chunks else None


def trace_to_test_item(trace: dict, default_task_type: TaskType = TaskType.RAG_QA) -> TestItem | None:
    """Map one Langfuse trace onto a TestItem; None if it has no usable IO."""
    task_prompt = _as_text(trace.get("input"))
    candidate_output = _as_text(trace.get("output"))
    if not task_prompt or not candidate_output:
        return None
    metadata = trace.get("metadata") or {}
    agent_id = None
    if isinstance(metadata, dict):
        agent_id = metadata.get("agent_id") or metadata.get("agent")
    return TestItem(
        id=f"{ITEM_ID_PREFIX}{trace['id']}",
        task_type=infer_task_type(trace, default_task_type),
        task_prompt=task_prompt,
        candidate_output=candidate_output,
        agent_id=agent_id or trace.get("name"),
        context=_extract_context(trace),
        reference=_as_text(metadata.get("reference")) if isinstance(metadata, dict) and metadata.get("reference") else None,
    )


def trace_id_for_item_id(item_id: str) -> str | None:
    """Recover the Langfuse trace id from an item id produced by this loader."""
    if item_id.startswith(ITEM_ID_PREFIX):
        return item_id[len(ITEM_ID_PREFIX):]
    return None


def fetch_traces(
    client: LangfuseClient,
    hours: float = 24.0,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Page through the trace list until `limit` traces or no more pages."""
    if from_timestamp is None:
        from_timestamp = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces: list[dict] = []
    page = 1
    while len(traces) < limit:
        batch = client.list_traces(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            page=page,
            limit=min(100, limit - len(traces)),
        )
        data = batch.get("data") or []
        traces.extend(data)
        meta = batch.get("meta") or {}
        if page >= int(meta.get("totalPages") or 1) or not data:
            break
        page += 1
    return traces[:limit]


def load_langfuse_items(
    client: LangfuseClient | None = None,
    hours: float = 24.0,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 50,
    default_task_type: TaskType = TaskType.RAG_QA,
    fetch_observations: bool = False,
) -> list[TestItem]:
    """Fetch recent Langfuse traces and map them to judge-ready TestItems.

    `fetch_observations=True` makes one extra API call per trace to pull its
    observations, which lets `_extract_context` find retrieval output to feed
    the faithfulness judge. Leave it off for prompt/output-only evaluation.
    """
    client = client or LangfuseClient()
    traces = fetch_traces(
        client,
        hours=hours,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        name=name,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        limit=limit,
    )
    items: list[TestItem] = []
    for trace in traces:
        if fetch_observations:
            try:
                trace = client.get_trace(trace["id"])
            except RuntimeError as exc:
                print(f"[langfuse_loader] Skipping observations for {trace['id']}: {exc}")
        item = trace_to_test_item(trace, default_task_type)
        if item is not None:
            items.append(item)
    return items


def push_result_scores(client: LangfuseClient, result: AggregateResult) -> int:
    """Write a judge result back to its Langfuse trace as scores.

    Pushes one numeric score per criterion (1-5, rationale as comment), the
    aggregate score, and a boolean overall pass/fail. Returns how many scores
    were created. The item must originate from this loader (id = lf_<traceId>).
    """
    trace_id = trace_id_for_item_id(result.item_id)
    if trace_id is None:
        return 0
    created = 0
    for verdict in result.verdicts:
        client.create_score(
            trace_id=trace_id,
            name=f"judge_{verdict.criterion.value}",
            value=float(verdict.score),
            comment=verdict.rationale,
        )
        created += 1
    client.create_score(
        trace_id=trace_id,
        name="judge_aggregate",
        value=round(result.aggregate_score, 3),
        comment=result.rationale or None,
    )
    client.create_score(
        trace_id=trace_id,
        name="judge_overall_pass",
        value=1.0 if result.overall_pass else 0.0,
        data_type="BOOLEAN",
    )
    return created + 2
