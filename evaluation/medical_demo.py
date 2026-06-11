"""Run a medical RAG agent over the gold set and push each run to Langfuse.

End-to-end demo of the observability half of the platform:

1. Load the clinical Q + medical-document items from data/gold/medical_rag_gold.jsonl.
2. Run the chosen LangGraph agent (--agent med_rag_weak | med_rag_strong) over each item.
3. Record operational telemetry to the local store (so governance sees run signals).
4. Create one Langfuse trace per item, tagged `medical-demo`, with the question as
   input, the agent answer as output, and agent_id / task_type / context in metadata.

The two agents are governed independently (separate agent_ids), so running each
populates its own fleet row, scorecard, and autonomy tier. The traces this writes
are exactly what `evaluation.langfuse_eval --tags medical-demo` then fetches and
judges, persisting the five criterion scores to the platform store.

Run with:

    PYTHONPATH=src python -m evaluation.medical_demo --agent med_rag_strong
    PYTHONPATH=src python -m evaluation.medical_demo --agent med_rag_weak

Requires LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and OPENAI_API_KEY in
.env (or use --backend local to run against a local Ollama server with no cloud keys).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_harness.config import PROJECT_ROOT, settings
from eval_harness.datasets.langfuse_loader import ITEM_ID_PREFIX, LangfuseClient
from eval_harness.fleet.registry import get_registered_agent, initialize_registry
from eval_harness.llm import LLMClient, get_client, get_hf_router_client
from eval_harness.schemas import TaskType, TestItem
from eval_harness.store import record_run_signal, upsert_agent

GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "medical_rag_gold.jsonl"
DEMO_TAG = "medical-demo"
AGENT_CHOICES = ["med_rag_weak", "med_rag_strong"]

# Fallback model per non-OpenAI backend (OpenAI model ids don't exist on HF/local,
# so they are swapped for the backend's model; the openai backend keeps the
# agent's own model — gpt-4o-mini for weak, gpt-4o for strong).
_BACKEND_DEFAULT_MODEL = {
    "openai": None,
    "hf": "meta-llama/Llama-3.1-8B-Instruct",
    "local": "llama3.1",
}
_LOCAL_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint


def _make_backend_client(backend: str, base_url: str) -> LLMClient | None:
    """Build the LLM client that powers the agent for the chosen backend.

    - openai: the default OpenAI client (keeps the agent's own model).
    - hf:     the HuggingFace Inference Providers router (a hosted Llama).
    - local:  any OpenAI-compatible local server, e.g. Ollama (a local Llama).
    """
    if backend == "openai":
        return get_client()
    if backend == "hf":
        return get_hf_router_client()
    if backend == "local":
        return LLMClient(api_key="local", base_url=base_url)
    raise ValueError(f"Unknown backend: {backend}")


def load_gold(path: Path, agent_id: str) -> list[TestItem]:
    """Read the medical gold JSONL into TestItems (question + clinical context)."""
    items: list[TestItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(
            TestItem(
                id=row["id"],
                task_type=TaskType(row.get("task_type", "rag_qa")),
                task_prompt=row["task_prompt"],
                context=row.get("context"),
                reference=row.get("candidate_output"),
                agent_id=agent_id,
            )
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="med_rag_strong",
        help="Which medical RAG agent to run (each is governed separately)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max gold items to run")
    parser.add_argument(
        "--backend",
        choices=["openai", "hf", "local"],
        default="openai",
        help="LLM backend: openai (the agent's own model — gpt-4o-mini/gpt-4o), "
        "hf (hosted Llama via HF router), or local (Ollama-style OpenAI-compatible server)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the agent model id (otherwise the agent's own model on openai, "
        "or the backend default on hf/local)",
    )
    parser.add_argument(
        "--base-url",
        default=_LOCAL_BASE_URL,
        help="Base URL for the local backend (default: Ollama at :11434)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip live LLM calls (extractive fallback) for a dry, no-cost run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not settings.has_langfuse:
        raise SystemExit(
            "Langfuse credentials missing. Set LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY in your .env."
        )

    agent_id = args.agent
    initialize_registry()
    agent = get_registered_agent(agent_id)
    agent._use_llm = not args.no_llm  # noqa: SLF001 - demo toggles live LLM on the instance
    if agent._use_llm:
        agent._client = _make_backend_client(args.backend, args.base_url)  # noqa: SLF001
        backend_model = args.model or _BACKEND_DEFAULT_MODEL[args.backend]
        if backend_model:  # None on openai -> keep the agent's own model
            agent.model_name = backend_model
    upsert_agent(agent.profile)

    items = load_gold(GOLD_PATH, agent_id)
    if args.limit:
        items = items[: args.limit]
    backend_desc = "extractive (no LLM)" if args.no_llm else f"{args.backend}:{agent.model_name}"
    print(f"Loaded {len(items)} medical gold items; running agent '{agent_id}' [{backend_desc}]")

    client = LangfuseClient()
    pushed = 0
    for i, item in enumerate(items, 1):
        output, trace = agent.run(item)

        trace_id = client.create_trace(
            name=agent_id,
            input=item.task_prompt,
            output=output,
            metadata={
                "agent_id": agent_id,
                "task_type": item.task_type.value,
                "context": item.context,
                "gold_id": item.id,
            },
            tags=[DEMO_TAG],
        )
        # Share the lf_<traceId> item id so the run signal and the judge eval row
        # (written later by langfuse_eval) line up for the same trace.
        record_run_signal(
            item_id=f"{ITEM_ID_PREFIX}{trace_id}",
            agent_id=agent_id,
            task_type=item.task_type,
            trace=trace,
        )
        pushed += 1
        preview = output.replace("\n", " ")[:70]
        print(f"  [{i}/{len(items)}] {item.id} -> trace {trace_id[:8]}  {preview}")

    client.close()
    print(
        f"\nPushed {pushed} traces to Langfuse tagged '{DEMO_TAG}' ({settings.langfuse_host}).\n"
        f"Next: PYTHONPATH=src python -m evaluation.langfuse_eval --tags {DEMO_TAG} "
        f"--with-observations"
    )


if __name__ == "__main__":
    main()
