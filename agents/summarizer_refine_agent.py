"""LangGraph summarizer: draft -> verify coverage -> optional refine."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness.fleet.registry import ClassBasedAgent, register_agent
from eval_harness.fleet.traced_agent import TraceAccumulator, TracedLangGraphAgent
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem


class SumState(TypedDict):
    task: TestItem
    source: str
    draft: str
    coverage: float
    refined: bool
    acc: TraceAccumulator
    output: str


def _coverage_score(summary: str, source: str) -> float:
    if not summary or not source:
        return 0.0
    s_words = {w.lower() for w in source.split() if len(w) > 4}
    sum_words = {w.lower() for w in summary.split() if len(w) > 4}
    if not s_words:
        return 0.0
    return min(1.0, len(sum_words & s_words) / max(len(s_words) * 0.15, 1))


def _offline_draft(source: str) -> str:
    sentences = [s.strip() for s in source.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return ""
    return ". ".join(sentences[:2]) + ("." if sentences else "")


def _build_sum_graph(agent: "SummarizerRefineAgent"):
    graph = StateGraph(SumState)

    def draft_node(state: SumState) -> dict:
        acc = state["acc"]
        task = state["task"]
        source = state["source"]
        if agent._use_llm:
            draft = agent._llm_text(
                system="Write a faithful concise summary of the source.",
                user=f"Source:\n{source}",
                model=agent.model_name,
                temperature=0.2,
                acc=acc,
            )
        else:
            acc.step()
            draft = _offline_draft(source)
        return {"draft": draft, "output": draft}

    def verify_node(state: SumState) -> dict:
        acc = state["acc"]
        acc.step()
        coverage = _coverage_score(state["draft"], state["source"])
        acc.groundedness = coverage
        needs_refine = coverage < 0.4
        return {"coverage": coverage, "refined": needs_refine}

    def refine_node(state: SumState) -> dict:
        acc = state["acc"]
        acc.retries += 1
        if agent._use_llm:
            refined = agent._llm_text(
                system="Improve the summary to cover more key points from the source.",
                user=f"Source:\n{state['source']}\n\nDraft:\n{state['draft']}",
                model=agent.model_name,
                temperature=0.1,
                acc=acc,
            )
        else:
            acc.step()
            sentences = [s.strip() for s in state["source"].replace("\n", " ").split(".") if s.strip()]
            refined = ". ".join(sentences[:4]) + ("." if sentences else "")
        coverage = _coverage_score(refined, state["source"])
        acc.groundedness = coverage
        return {"draft": refined, "output": refined, "coverage": coverage, "refined": False}

    graph.add_node("draft", draft_node)
    graph.add_node("verify", verify_node)
    graph.add_node("refine", refine_node)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "verify")

    def route_after_verify(state: SumState) -> str:
        if state.get("refined") and state.get("coverage", 1.0) < 0.4:
            return "refine"
        return END

    graph.add_conditional_edges("verify", route_after_verify, {"refine": "refine", END: END})
    graph.add_edge("refine", END)
    return graph.compile()


@register_agent
class SummarizerRefineAgent(ClassBasedAgent, TracedLangGraphAgent):
    """Draft summarizer with coverage verification and one refine pass."""

    agent_id = "sum_refine"
    name = "Summarizer Refine (LangGraph)"
    task_type = TaskType.SUMMARIZATION
    model = "langgraph:sum-refine"
    prompt_variant = "draft-verify-refine"
    description = "LangGraph agent: draft summary, verify coverage, refine if thin."

    def __init__(self, client=None, use_llm: bool = False) -> None:
        TracedLangGraphAgent.__init__(self, client=client)
        self._use_llm = use_llm
        self.model_name = "gpt-4o-mini"
        self._graph = _build_sum_graph(self)

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        acc = TraceAccumulator(model=self.model)
        source = (task.context or task.task_prompt).strip()
        try:
            final = self._graph.invoke({
                "task": task,
                "source": source,
                "draft": "",
                "coverage": 0.0,
                "refined": False,
                "acc": acc,
                "output": "",
            })
            output = final.get("output") or final.get("draft") or ""
            return output.strip(), acc.to_trace()
        except Exception as exc:
            acc.success = False
            acc.error = str(exc)
            return "", acc.to_trace()

    def answer(self, task: TestItem) -> str:
        return self.run(task)[0]
