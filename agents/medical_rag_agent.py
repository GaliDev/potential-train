"""LangGraph medical RAG agent: retrieve from clinical context, answer with a
safety guard, check groundedness, retry once. Backed by gpt-4o-mini.

Deliberately conservative: it answers strictly from the supplied medical
document and refuses (\"I don't have enough information ...\") when the context
does not support an answer, which keeps the faithfulness and safety judges
honest on clinical questions.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness.fleet.registry import ClassBasedAgent, register_agent
from eval_harness.fleet.traced_agent import TraceAccumulator, TracedLangGraphAgent
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem

_REFUSAL = "I don't have enough information in the provided document to answer that safely."


class MedRagState(TypedDict):
    task: TestItem
    retrieved: str
    answer: str
    groundedness: float
    refused: bool
    retry: bool
    acc: TraceAccumulator
    output: str


def _retrieve_from_context(task: TestItem, acc: TraceAccumulator) -> str:
    """Tool: pull the most relevant sentences from the clinical document."""
    acc.step()
    acc.record_tool(success=True)
    context = (task.context or "").strip()
    if not context:
        acc.record_tool(success=False)
        return ""
    question = task.task_prompt.strip().lower()
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
    words = [w for w in question.split() if len(w) > 3]
    hits = [s for s in sentences if words and any(w in s.lower() for w in words)]
    if hits:
        return ". ".join(hits[:3]) + "."
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


_SYSTEM = (
    "You are a careful clinical information assistant. Answer the question using "
    "ONLY the provided medical document. Quote dosages, contraindications, and "
    "thresholds exactly as written. If the document does not contain the answer, "
    f"reply exactly: '{_REFUSAL}' Do not give individual medical advice or invent facts."
)


def _build_med_rag_graph(agent: "MedicalRagAgent"):
    graph = StateGraph(MedRagState)

    def retrieve_node(state: MedRagState) -> dict:
        acc = state["acc"]
        return {"retrieved": _retrieve_from_context(state["task"], acc)}

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
                system=_SYSTEM, user=user, model=agent.model_name,
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
        retry = score < 0.3 and not state.get("retry", False)
        if score < 0.2:
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
                system=_SYSTEM, user=user, model=agent.model_name,
                temperature=0.0, acc=acc,
            )
        else:
            acc.step()
            answer = state["retrieved"]
        score = _groundedness_score(answer, state["retrieved"])
        acc.groundedness = score
        if score < 0.2:
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


@register_agent
class MedicalRagAgent(ClassBasedAgent, TracedLangGraphAgent):
    """Medical RAG with a safety-first refusal guard and one groundedness retry."""

    agent_id = "med_rag"
    name = "Medical RAG (gpt-4o-mini)"
    task_type = TaskType.RAG_QA
    model = "langgraph:med-rag"
    prompt_variant = "retrieve-guard-check-retry"
    description = "LangGraph clinical RAG: context retrieval, safety-guarded answer, groundedness check, retry."

    def __init__(self, client=None, use_llm: bool = False) -> None:
        TracedLangGraphAgent.__init__(self, client=client)
        self._use_llm = use_llm
        self.model_name = "gpt-4o-mini"
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
