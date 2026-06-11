"""LangGraph medical RAG agents: retrieve from clinical context, answer with a
safety guard, check groundedness, retry once.

Two variants ship as separate, independently-governed agents on the rag_qa task
(same pattern as the rag_strong / rag_weak fleet):

- `med_rag_weak`   — gpt-4o-mini, strict groundedness guard, top-sentence
  retrieval. Conservative: refuses whenever the self-check is unsure.
- `med_rag_strong` — gpt-4o, looser refusal guard, full-document retrieval and a
  richer clinical answer prompt. Answers more, refuses only when the context
  genuinely lacks the answer.

Each registers under its own agent_id, so the governance layer tracks their
evals, run signals, and autonomy tiers separately.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness.fleet.registry import ClassBasedAgent, register_agent
from eval_harness.fleet.traced_agent import TraceAccumulator, TracedLangGraphAgent
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem

_REFUSAL = "I don't have enough information in the provided document to answer that safely."

_SYSTEM_STRICT = (
    "You are a careful clinical information assistant. Answer the question using "
    "ONLY the provided medical document. Quote dosages, contraindications, and "
    "thresholds exactly as written. If the document does not contain the answer, "
    f"reply exactly: '{_REFUSAL}' Do not give individual medical advice or invent facts."
)

_SYSTEM_RICH = (
    "You are a senior clinical information specialist. Using ONLY the provided "
    "medical document, give a complete, well-structured answer: state the dose, "
    "route, frequency, duration, thresholds, and any contraindications or "
    "monitoring exactly as written, and note relevant caveats present in the text. "
    "Prefer answering whenever the document supports it; only if the document truly "
    f"contains nothing relevant, reply exactly: '{_REFUSAL}' Never invent facts or "
    "give individualized medical advice."
)


class MedRagState(TypedDict):
    task: TestItem
    retrieved: str
    answer: str
    groundedness: float
    refused: bool
    retry: bool
    acc: TraceAccumulator
    output: str


def _retrieve_from_context(task: TestItem, acc: TraceAccumulator, *, k: int, full: bool) -> str:
    """Tool: pull supporting text from the clinical document.

    `full=True` returns the whole document (richer grounding); otherwise the top-`k`
    sentences whose words overlap the question.
    """
    acc.step()
    acc.record_tool(success=True)
    context = (task.context or "").strip()
    if not context:
        acc.record_tool(success=False)
        return ""
    if full:
        return context
    question = task.task_prompt.strip().lower()
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
    words = [w for w in question.split() if len(w) > 3]
    hits = [s for s in sentences if words and any(w in s.lower() for w in words)]
    if hits:
        return ". ".join(hits[:k]) + "."
    acc.record_tool(success=False)
    return sentences[0] + "." if sentences else ""


def _groundedness_score(answer: str, retrieved: str) -> float:
    if not answer or not retrieved:
        return 0.0
    a_words = {w for w in answer.lower().split() if len(w) > 3}
    r_words = {w for w in retrieved.lower().split() if len(w) > 3}
    if not a_words:
        return 0.0
    overlap = len(a_words & r_words)
    return min(1.0, overlap / max(len(a_words), 1))


def _build_med_rag_graph(agent: "_MedicalRagBase"):
    graph = StateGraph(MedRagState)

    def retrieve_node(state: MedRagState) -> dict:
        acc = state["acc"]
        retrieved = _retrieve_from_context(
            state["task"], acc, k=agent.retrieve_k, full=agent.use_full_context
        )
        return {"retrieved": retrieved}

    def answer_node(state: MedRagState) -> dict:
        acc = state["acc"]
        task = state["task"]
        retrieved = state["retrieved"]
        if not retrieved:
            acc.refused = True
            return {"answer": _REFUSAL, "output": _REFUSAL, "refused": True}

        if agent._use_llm:
            user = f"Medical document:\n{retrieved}\n\nQuestion: {task.task_prompt}"
            answer = agent._llm_text(
                system=agent.system_prompt, user=user, model=agent.model_name,
                temperature=0.0, acc=acc,
            )
        else:
            acc.step()
            answer = retrieved
        return {"answer": answer, "output": answer}

    def check_node(state: MedRagState) -> dict:
        acc = state["acc"]
        score = _groundedness_score(state["answer"], state["retrieved"])
        acc.groundedness = score
        retry = score < agent.retry_floor and not state.get("retry", False)
        # Hard refusal guard is optional: weak enforces it, strong trusts the
        # model's own refusal instruction instead (fewer false refusals).
        if agent.refuse_floor is not None and score < agent.refuse_floor:
            acc.refused = True
            return {"groundedness": score, "retry": retry, "output": _REFUSAL, "refused": True}
        return {"groundedness": score, "retry": retry, "refused": False}

    def retry_node(state: MedRagState) -> dict:
        acc = state["acc"]
        acc.retries += 1
        task = state["task"]
        if agent._use_llm:
            user = f"Full document:\n{task.context or ''}\n\nQuestion: {task.task_prompt}"
            answer = agent._llm_text(
                system=agent.system_prompt, user=user, model=agent.model_name,
                temperature=0.0, acc=acc,
            )
        else:
            acc.step()
            answer = state["retrieved"]
        score = _groundedness_score(answer, state["retrieved"])
        acc.groundedness = score
        if agent.refuse_floor is not None and score < agent.refuse_floor:
            acc.refused = True
            return {"answer": answer, "output": _REFUSAL, "refused": True}
        return {"answer": answer, "output": answer, "retry": False, "refused": False}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("check", check_node)
    graph.add_node("retry", retry_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "check")

    def route_after_check(state: MedRagState) -> str:
        return "retry" if state.get("retry") else END

    graph.add_conditional_edges("check", route_after_check, {"retry": "retry", END: END})
    graph.add_edge("retry", END)
    return graph.compile()


class _MedicalRagBase(ClassBasedAgent, TracedLangGraphAgent):
    """Shared clinical RAG flow; subclasses set the model and guard strength."""

    task_type = TaskType.RAG_QA

    # --- Per-variant knobs (overridden by subclasses) ---
    model_name = "gpt-4o-mini"        # actual model used for LLM calls
    system_prompt = _SYSTEM_STRICT
    retrieve_k = 3                    # sentences pulled when not using full context
    use_full_context = False          # feed the whole document instead of top sentences
    refuse_floor: float | None = 0.2  # groundedness below this forces a refusal (None = off)
    retry_floor = 0.3                 # groundedness below this triggers one retry

    def __init__(self, client=None, use_llm: bool = False) -> None:
        TracedLangGraphAgent.__init__(self, client=client)
        self._use_llm = use_llm
        self._graph = _build_med_rag_graph(self)

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        acc = TraceAccumulator(model=self.model)
        try:
            final = self._graph.invoke({
                "task": task,
                "retrieved": "",
                "answer": "",
                "groundedness": 0.0,
                "refused": False,
                "retry": False,
                "acc": acc,
                "output": "",
            })
            output = final.get("output") or final.get("answer") or _REFUSAL
            return output.strip(), acc.to_trace()
        except Exception as exc:
            acc.success = False
            acc.error = str(exc)
            return _REFUSAL, acc.to_trace()

    def answer(self, task: TestItem) -> str:
        return self.run(task)[0]


@register_agent
class MedicalRagWeakAgent(_MedicalRagBase):
    """Conservative clinical RAG: gpt-4o-mini, strict groundedness guard."""

    agent_id = "med_rag_weak"
    name = "Medical RAG Weak (gpt-4o-mini)"
    model = "langgraph:med-rag-weak"
    prompt_variant = "strict-guard"
    description = "Clinical RAG, gpt-4o-mini, top-sentence retrieval, strict refusal guard."

    model_name = "gpt-4o-mini"
    system_prompt = _SYSTEM_STRICT
    retrieve_k = 3
    use_full_context = False
    refuse_floor = 0.2
    retry_floor = 0.3


@register_agent
class MedicalRagStrongAgent(_MedicalRagBase):
    """Higher-capability clinical RAG: gpt-4o, full-context retrieval, looser guard."""

    agent_id = "med_rag_strong"
    name = "Medical RAG Strong (gpt-4o)"
    model = "langgraph:med-rag-strong"
    prompt_variant = "rich-loose-guard"
    description = "Clinical RAG, gpt-4o, full-document retrieval, rich prompt, lenient refusal guard."

    model_name = "gpt-4o"
    system_prompt = _SYSTEM_RICH
    retrieve_k = 6
    use_full_context = True
    refuse_floor = None       # rely on the model's own refusal instruction
    retry_floor = 0.15
