"""Run the med_rag agent over the medical gold set and push each run to Langfuse.

End-to-end demo of the observability half of the platform:

1. Load the clinical Q + medical-document items from data/gold/medical_rag_gold.jsonl.
2. Run the registered `med_rag` LangGraph agent (gpt-4o-mini) over each item.
3. Record operational telemetry to the local store (so governance sees run signals).
4. Create one Langfuse trace per item, tagged `medical-demo`, with the question as
   input, the agent answer as output, and agent_id / task_type / context in metadata.

The traces this writes are exactly what `evaluation.langfuse_eval --tags medical-demo
--push-scores` then fetches, judges, and writes the five criterion scores back onto.

Run with:

    PYTHONPATH=src python -m evaluation.medical_demo

Requires LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and OPENAI_API_KEY in .env.
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
AGENT_ID = "med_rag"

# Default model per execution backend for the med_rag agent.
_BACKEND_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "hf": "meta-llama/Llama-3.1-8B-Instruct",
    "local": "llama3.2",
}
_LOCAL_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint


def _make_backend_client(backend: str, base_url: str) -> LLMClient | None:
    """Build the LLM client that powers med_rag for the chosen backend.

    - openai: the default OpenAI client (gpt-4o-mini).
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


def load_gold(path: Path) -> list[TestItem]:
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
                agent_id=AGENT_ID,
            )
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max gold items to run")
    parser.add_argument(
        "--backend",
        choices=["openai", "hf", "local"],
        default="openai",
        help="LLM backend for med_rag: openai (gpt-4o-mini), hf (hosted Llama via HF router), "
        "or local (Ollama-style OpenAI-compatible server)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the agent model id (defaults per backend)",
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

    initialize_registry()
    agent = get_registered_agent(AGENT_ID)
    agent._use_llm = not args.no_llm  # noqa: SLF001 - demo toggles live LLM on the instance
    if agent._use_llm:
        agent._client = _make_backend_client(args.backend, args.base_url)  # noqa: SLF001
        agent.model_name = args.model or _BACKEND_DEFAULT_MODEL[args.backend]
    upsert_agent(agent.profile)

    items = load_gold(GOLD_PATH)
    if args.limit:
        items = items[: args.limit]
    backend_desc = "extractive (no LLM)" if args.no_llm else f"{args.backend}:{agent.model_name}"
    print(f"Loaded {len(items)} medical gold items; running agent '{AGENT_ID}' [{backend_desc}]")

    client = LangfuseClient()
    pushed = 0
    for i, item in enumerate(items, 1):
        output, trace = agent.run(item)

        trace_id = client.create_trace(
            name=AGENT_ID,
            input=item.task_prompt,
            output=output,
            metadata={
                "agent_id": AGENT_ID,
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
            agent_id=AGENT_ID,
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
        f"--with-observations --push-scores"
    )


if __name__ == "__main__":
    main()
