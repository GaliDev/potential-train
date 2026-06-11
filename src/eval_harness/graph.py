"""LangGraph orchestration of the multiagent judge panel.

The graph fans out from START to all per-criterion judges (which run in the
same superstep), then joins at the aggregator/meta-judge node:

    START -> [correctness, faithfulness, completeness, coherence, safety] -> aggregate -> END

`PanelJudge` wraps the compiled graph behind the same `.judge(item)` interface
the runner expects, so it is a drop-in replacement for the baseline.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.aggregator import aggregate
from .agents.criteria import PANEL_CRITERIA, CriterionJudge
from .schemas import AggregateResult, Criterion, JudgeVerdict, TestItem

AGGREGATE_NODE = "aggregate"


class PanelState(TypedDict):
    item: TestItem
    # None -> each criterion judge uses the model mapped to it in
    # config/judge_panel.json; a string forces that model for all criteria.
    model: Optional[str]
    verdicts: Annotated[list[JudgeVerdict], add]
    result: Optional[AggregateResult]


def _make_judge_node(criterion: Criterion):
    def node(state: PanelState) -> dict:
        judge = CriterionJudge(criterion, model=state.get("model"))
        verdict = judge.judge(state["item"])
        return {"verdicts": [verdict]}

    return node


def _aggregate_node(state: PanelState) -> dict:
    item = state["item"]
    result = aggregate(
        item_id=item.id,
        task_type=item.task_type,
        verdicts=state["verdicts"],
        agent_id=item.agent_id,
        judge_mode="panel",
    )
    return {"result": result}


def build_graph():
    """Compile the judge-panel graph."""
    graph = StateGraph(PanelState)
    graph.add_node(AGGREGATE_NODE, _aggregate_node)
    for crit in PANEL_CRITERIA:
        graph.add_node(crit.value, _make_judge_node(crit))
        graph.add_edge(START, crit.value)
        graph.add_edge(crit.value, AGGREGATE_NODE)
    graph.add_edge(AGGREGATE_NODE, END)
    return graph.compile()


class PanelJudge:
    """Multiagent panel judge, drop-in compatible with the runner's Judge protocol."""

    judge_mode = "panel"

    def __init__(self, model: str | None = None) -> None:
        self._graph = build_graph()
        # None = per-criterion models from config/judge_panel.json.
        self._model = model

    def judge(self, item: TestItem) -> AggregateResult:
        final = self._graph.invoke(
            {"item": item, "model": self._model, "verdicts": [], "result": None}
        )
        result = final.get("result")
        if result is None:
            raise RuntimeError("Panel graph did not produce a result.")
        return result
