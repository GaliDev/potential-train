"""LangGraph RAG agent: retrieve from context, answer, groundedness check + retry."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness.fleet.registry import ClassBasedAgent, register_agent
from eval_harness.fleet.traced_agent import TraceAccumulator, TracedLangGraphAgent
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem


class RagState(TypedDict):
    task: TestItem
    retrieved: str
    answer: str
    groundedness: float
    refused: bool
    retry: bool
    acc: TraceAccumulator
    output: str


def _retrieve_from_context(task: TestItem, acc: TraceAccumulator) -> str:
    """Tool: pick the best-matching sentence from provided context."""
    acc.step()
    acc.record_tool(success=True)
    context = (task.context or "").strip()
    if not context:
        acc.record_tool(success=False)
        return ""
    question = task.task_prompt.strip().lower()
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
    words = [w for w in question.split() if len(w) > 3]
    for sentence in sentences:
        if words and any(w in sentence.lower() for w in words):
            return sentence if sentence.endswith(".") else f"{sentence}."
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


def _build_rag_graph(agent: "RagReactAgent"):
    graph = StateGraph(RagState)

    def retrieve_node(state: RagState) -> dict:
        acc = state["acc"]
        retrieved = _retrieve_from_context(state["task"], acc)
        return {"retrieved": retrieved}

    def answer_node(state: RagState) -> dict:
        acc = state["acc"]
        task = state["task"]
        retrieved = state["retrieved"]
        if not retrieved:
            acc.refused = True
            return {"answer": "I don't know.", "output": "I don't know.", "refused": True}

        if agent._use_llm:
            system = (
                "Answer the question using ONLY the retrieved context. "
                "If the context does not contain the answer, say 'I don't know.'"
            )
            user = f"Context:\n{retrieved}\n\nQuestion: {task.task_prompt}"
            answer = agent._llm_text(
                system=system, user=user, model=agent.model_name,
                temperature=0.1, acc=acc,
            )
        else:
            acc.step()
            answer = retrieved

        return {"answer": answer, "output": answer}

    def check_node(state: RagState) -> dict:
        acc = state["acc"]
        score = _groundedness_score(state["answer"], state["retrieved"])
        acc.groundedness = score
        retry = score < 0.3 and not state.get("retry", False)
        if score < 0.2:
            acc.refused = True
            return {
                "groundedness": score,
                "retry": retry,
                "output": "I don't know.",
                "refused": True,
            }
        return {"groundedness": score, "retry": retry, "refused": False}

    def retry_node(state: RagState) -> dict:
        acc = state["acc"]
        acc.retries += 1
        task = state["task"]
        if agent._use_llm:
            system = "Re-read the full context and answer precisely from it only."
            user = f"Full context:\n{task.context or ''}\n\nQuestion: {task.task_prompt}"
            answer = agent._llm_text(
                system=system, user=user, model=agent.model_name,
                temperature=0.0, acc=acc,
            )
        else:
            acc.step()
            answer = state["retrieved"]
        score = _groundedness_score(answer, state["retrieved"])
        acc.groundedness = score
        if score < 0.2:
            acc.refused = True
            return {"answer": answer, "output": "I don't know.", "refused": True}
        return {"answer": answer, "output": answer, "retry": False, "refused": False}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("check", check_node)
    graph.add_node("retry", retry_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "check")

    def route_after_check(state: RagState) -> str:
        if state.get("retry"):
            return "retry"
        return END

    graph.add_conditional_edges("check", route_after_check, {"retry": "retry", END: END})
    graph.add_edge("retry", END)
    return graph.compile()


@register_agent
class RagReactAgent(ClassBasedAgent, TracedLangGraphAgent):
    """Retrieve-from-context RAG with groundedness self-check and one retry."""

    agent_id = "rag_react"
    name = "RAG ReAct (LangGraph)"
    task_type = TaskType.RAG_QA
    model = "langgraph:rag-react"
    prompt_variant = "retrieve-check-retry"
    description = "LangGraph agent: context retrieval tool, answer, groundedness check, retry."

    def __init__(self, client=None, use_llm: bool = False) -> None:
        TracedLangGraphAgent.__init__(self, client=client)
        self._use_llm = use_llm
        self.model_name = "gpt-4o-mini"
        self._graph = _build_rag_graph(self)

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
            output = final.get("output") or final.get("answer") or "I don't know."
            return output.strip(), acc.to_trace()
        except Exception as exc:
            acc.success = False
            acc.error = str(exc)
            return "I don't know.", acc.to_trace()

    def answer(self, task: TestItem) -> str:
        return self.run(task)[0]
