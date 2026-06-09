"""LangGraph translator: translate -> back-translate check -> optional retry."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from eval_harness.fleet.registry import ClassBasedAgent, register_agent
from eval_harness.fleet.traced_agent import TraceAccumulator, TracedLangGraphAgent
from eval_harness.schemas import ExecutionTrace, TaskType, TestItem


class TransState(TypedDict):
    task: TestItem
    source: str
    translation: str
    back_translation: str
    fidelity: float
    retry: bool
    acc: TraceAccumulator
    output: str


def _fidelity_score(source: str, back: str) -> float:
    if not source or not back:
        return 0.0
    s = {w.lower() for w in source.split() if len(w) > 3}
    b = {w.lower() for w in back.split() if len(w) > 3}
    if not s:
        return 0.0
    return min(1.0, len(s & b) / max(len(s), 1))


def _offline_translate(source: str) -> str:
    """Deterministic pseudo-translation for offline tests."""
    return " ".join(reversed(source.split()))


def _build_trans_graph(agent: "TranslatorBackcheckAgent"):
    graph = StateGraph(TransState)

    def translate_node(state: TransState) -> dict:
        acc = state["acc"]
        source = state["source"]
        instruction = state["task"].task_prompt
        if agent._use_llm:
            translation = agent._llm_text(
                system="Translate the text faithfully. Output only the translation.",
                user=f"{instruction}\n\nText:\n{source}",
                model=agent.model_name,
                temperature=0.1,
                acc=acc,
            )
        else:
            acc.step()
            translation = _offline_translate(source)
        return {"translation": translation, "output": translation}

    def backcheck_node(state: TransState) -> dict:
        acc = state["acc"]
        acc.step()
        acc.record_tool(success=True)
        if agent._use_llm:
            back = agent._llm_text(
                system="Translate the following text back to English. Output only the text.",
                user=state["translation"],
                model=agent.model_name,
                temperature=0.0,
                acc=acc,
            )
        else:
            acc.step()
            back = state["source"]
        fidelity = _fidelity_score(state["source"], back)
        acc.groundedness = fidelity
        retry = fidelity < 0.5 and not state.get("retry", False)
        return {"back_translation": back, "fidelity": fidelity, "retry": retry}

    def retry_node(state: TransState) -> dict:
        acc = state["acc"]
        acc.retries += 1
        if agent._use_llm:
            translation = agent._llm_text(
                system="Translate carefully, preserving all names and numbers.",
                user=f"{state['task'].task_prompt}\n\nText:\n{state['source']}",
                model=agent.model_name,
                temperature=0.0,
                acc=acc,
            )
        else:
            acc.step()
            translation = state["source"]
        fidelity = _fidelity_score(state["source"], translation)
        acc.groundedness = fidelity
        return {"translation": translation, "output": translation, "retry": False}

    graph.add_node("translate", translate_node)
    graph.add_node("backcheck", backcheck_node)
    graph.add_node("retry", retry_node)
    graph.add_edge(START, "translate")
    graph.add_edge("translate", "backcheck")

    def route_after_backcheck(state: TransState) -> str:
        if state.get("retry"):
            return "retry"
        return END

    graph.add_conditional_edges("backcheck", route_after_backcheck, {"retry": "retry", END: END})
    graph.add_edge("retry", END)
    return graph.compile()


@register_agent
class TranslatorBackcheckAgent(ClassBasedAgent, TracedLangGraphAgent):
    """Professional translator with back-translation fidelity check and retry."""

    agent_id = "trans_backcheck"
    name = "Translator Backcheck (LangGraph)"
    task_type = TaskType.TRANSLATION
    model = "langgraph:trans-backcheck"
    prompt_variant = "translate-backcheck-retry"
    description = "LangGraph agent: translate, back-translate check, retry if low fidelity."

    def __init__(self, client=None, use_llm: bool = False) -> None:
        TracedLangGraphAgent.__init__(self, client=client)
        self._use_llm = use_llm
        self.model_name = "gpt-4o-mini"
        self._graph = _build_trans_graph(self)

    def run(self, task: TestItem) -> tuple[str, ExecutionTrace]:
        acc = TraceAccumulator(model=self.model)
        source = (task.context or task.task_prompt).strip()
        try:
            final = self._graph.invoke({
                "task": task,
                "source": source,
                "translation": "",
                "back_translation": "",
                "fidelity": 0.0,
                "retry": False,
                "acc": acc,
                "output": "",
            })
            output = final.get("output") or final.get("translation") or ""
            return output.strip(), acc.to_trace()
        except Exception as exc:
            acc.success = False
            acc.error = str(exc)
            return "", acc.to_trace()

    def answer(self, task: TestItem) -> str:
        return self.run(task)[0]
