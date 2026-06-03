---
name: Agent Workforce Governance
overview: Build an Agent Workforce Governance platform (Python, LangGraph, OpenAI GPT-4o/4o-mini). A multiagent LLM-as-judge eval engine scores a managed fleet of agents over time, and a governance layer turns that performance data into four decisions competitors can't make - which agent gets the next task, how much autonomy each agent gets, per-agent performance reviews, and policy/governance enforcement. Validated by agreement with human gold labels on a hybrid (public + custom) test set.
todos:
  - id: scaffold
    content: "Scaffold Python project: requirements.txt (langgraph, langchain-openai, openai, pydantic, pandas, scipy, scikit-learn, fastapi, uvicorn, streamlit, python-dotenv, sqlmodel), .env.example, .gitignore, README, and src/app/data/evaluation folder structure."
    status: pending
  - id: schemas-store
    content: "Implement schemas.py (TestItem/JudgeVerdict/AggregateResult/AgentProfile/AutonomyTier/Policy) plus llm.py (OpenAI wrapper, model selection, token/cost tracking) and a SQLite performance store (registry + per-run results)."
    status: pending
  - id: fleet
    content: "Define the managed fleet: 2 task types (RAG/Q&A + summarization) x 3 agent configs each (gpt-4o, gpt-4o-mini, weaker-prompt variant); add a generator that produces their outputs for evaluation."
    status: pending
  - id: baseline
    content: "Build the single GPT-4o baseline judge with a simple prompt and a runner that scores the fleet outputs (establish baseline before complexity)."
    status: pending
  - id: datasets
    content: "Implement dataset loaders: ingest a public human-labeled eval slice + create the custom ~30-50 item hand-labeled gold set in data/gold/."
    status: pending
  - id: graph
    content: "Implement the multiagent judge panel (correctness/faithfulness/completeness/coherence/safety) + aggregator/meta-judge, orchestrated in graph.py via LangGraph; persist verdicts to the store."
    status: pending
  - id: metrics
    content: "Implement metrics.py + evaluation/benchmark.py to compute agreement (accuracy, Cohen's kappa, Spearman), latency, and cost vs the gold labels."
    status: pending
  - id: governance
    content: "Build the governance layer on top of the store: router (next-task assignment), autonomy calibrator (tiers from reliability), review generator (LLM-written per-agent scorecards), policy/governance engine + audit log."
    status: pending
  - id: improve
    content: "Add an improvement lever: quality (jury/debate + bias mitigation + calibration) and/or cost (gpt-4o-mini -> gpt-4o cascade); compare against baseline."
    status: pending
  - id: app
    content: "Build FastAPI endpoints + Streamlit dashboard: eval drill-down, fleet leaderboard, routing simulator, autonomy tiers, performance reviews, governance/audit views."
    status: pending
  - id: writeup
    content: "Draft assignment write-ups in docs/ (problem, market gap vs eval platforms, architecture diagram, POC scope, KPIs) and prepare the demo/pitch."
    status: pending
isProject: true
---

# Agent Workforce Governance Platform

## Concept
Most tools stop at *scoring* model outputs. This platform manages a **fleet of AI agents like a team**: a multiagent LLM-as-judge **eval engine** continuously scores agents, and a **governance layer** turns that history into four decisions:

1. Which agent should get the next task? (performance-aware routing)
2. How much autonomy should each agent have? (reliability -> autonomy tier)
3. Agent performance reviews (longitudinal per-agent scorecards)
4. Organizational governance (policy enforcement, escalation, audit trail)

The eval engine's own quality is validated by **agreement with human gold labels** (accuracy, Cohen's kappa, Spearman) - the headline success metric that keeps the project rigorous.

- Stack: Python, LangGraph, OpenAI (`gpt-4o` + `gpt-4o-mini`), Pydantic, SQLite (SQLModel) for the performance store, pandas/scipy/scikit-learn (metrics), FastAPI (API), Streamlit (dashboard).
- Data: Hybrid - a public human-labeled eval slice (e.g. MT-Bench judgments or SummEval) + a custom ~30-50 item hand-labeled gold set in the demo domain.

## Managed Fleet (the thing being governed)
- 2 task types: **RAG/Q&A** and **summarization** (both have public gold data and are easy to grade).
- 3 competing agent configs per task type = **6 logical agents**: `gpt-4o`, `gpt-4o-mini`, and a deliberately weaker prompt/temperature variant.
- Deliberate performance gaps make routing, reviews, and autonomy tiers show clear, demo-able differences.

## POC Scope (assignment "Scoping Your POC" section)
- Input: `(task, agent_id, candidate_output, optional reference/context, rubric)` records.
- Output (eval): per-criterion scores (1-5) + pass/fail + aggregate + rationale with quoted evidence.
- Output (governance): a routing decision, an autonomy tier, and a per-agent review derived from accumulated scores.
- Success metric: agreement of harness verdicts with human gold labels.
- Target: >= 80% pass/fail agreement and Cohen's kappa >= 0.6 (Spearman >= 0.7 on scores) before adding complexity.
- Minimum viable test set: custom gold set (~30-50 items) + public slice (~50-100 items).
- Baseline first: single GPT-4o judge + simple prompt, then add the multiagent panel and governance.

## Architecture

```mermaid
flowchart TD
    Fleet["Managed fleet: 6 agent configs x 2 task types"] --> Gen[Output generator]
    Gen --> Pre[Preprocessor: normalize, attach rubric, randomize position]

    subgraph eval [Eval Engine - LangGraph judge panel]
      Pre --> Correct[Correctness]
      Pre --> Faith[Faithfulness]
      Pre --> Complete[Completeness]
      Pre --> Coh[Coherence]
      Pre --> Safe[Safety]
      Correct --> Agg[Aggregator / Meta-Judge]
      Faith --> Agg
      Complete --> Agg
      Coh --> Agg
      Safe --> Agg
      Agg --> Cal[Calibrator: bias mitigation + thresholds]
    end

    Cal --> Store[(Performance store: per-agent, per-criterion, over time)]
    Gold["Human gold labels"] --> Bench["Benchmark: kappa / accuracy / Spearman / latency / cost"]
    Store --> Bench

    subgraph gov [Governance Layer]
      Store --> Router[Task Router: which agent next]
      Store --> Autonomy[Autonomy Calibrator: tier per agent]
      Store --> Review[Performance Review generator]
      Router --> Policy[Policy + Governance engine + audit log]
      Autonomy --> Policy
    end

    Bench --> UI["FastAPI + Streamlit dashboard"]
    Policy --> UI
    Review --> UI
```

## Governance Layer detail
- Router: score each agent for an incoming task (task-type competence, reliability, cost, latency); pick the best, with a tie-break/explore option.
- Autonomy calibrator: map reliability + error/variance + safety flags to tiers - full auto / auto + spot-check / human-in-the-loop / blocked - with explicit thresholds.
- Performance reviews: aggregate the store into per-agent scorecards (strengths/weaknesses by criterion & task-type, trend, regressions), rendered as an LLM-written review.
- Policy/governance engine: rules that gate actions (low-autonomy agents need approval, safety-critical tasks require human review) + an audit log of every decision.

## Improvement Dimension (assignment requires improving at least one)
- Quality lever: jury/debate pass + rubric decomposition + bias mitigation (position & verbosity) -> higher agreement vs single-judge baseline.
- Cost lever: model cascade - `gpt-4o-mini` screens every item, escalate only borderline items to `gpt-4o`; report cost/latency vs all-`gpt-4o`.

## Proposed Repo Structure
- `src/eval_harness/schemas.py` - Pydantic models (`TestItem`, `JudgeVerdict`, `AggregateResult`, `AgentProfile`, `AutonomyTier`, `Policy`).
- `src/eval_harness/llm.py` - OpenAI wrapper, model selection, token/cost tracking.
- `src/eval_harness/store.py` - SQLite (SQLModel) registry + per-run results.
- `src/eval_harness/fleet/` - agent configs + output generator for the managed fleet.
- `src/eval_harness/graph.py` - `build_graph()` LangGraph judge panel.
- `src/eval_harness/agents/` - `correctness.py`, `faithfulness.py`, `completeness.py`, `coherence.py`, `safety.py`, `aggregator.py`, `jury.py`.
- `src/eval_harness/calibration.py` - bias mitigation + thresholds.
- `src/eval_harness/metrics.py` - kappa / Spearman / accuracy vs gold.
- `src/eval_harness/datasets/loaders.py` - public + custom gold-set loaders.
- `src/eval_harness/governance/` - `router.py`, `autonomy.py`, `reviews.py`, `policy.py`.
- `src/eval_harness/runner.py` - run engine over a set, persist to store + `data/runs/`.
- `evaluation/benchmark.py` - agreement metrics + report.
- `app/api.py` (FastAPI) and `app/ui.py` (Streamlit dashboard).
- `data/public/`, `data/gold/`, `data/runs/`; plus `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.

## 10-Day Plan for 2 Developers
- Dev A (eval engine): scaffold, schemas+store, fleet generator, baseline judge, multiagent panel, metrics/benchmark, improvement lever.
- Dev B (governance + product): dataset/gold set, router, autonomy calibrator, review generator, policy/audit engine, FastAPI + Streamlit dashboard.
- Shared: write-ups, demo, pitch.
- Days 1-2: scaffold, schemas, store, fleet + gold set. Days 3-4: baseline + multiagent panel + metrics (hit target). Days 5-6: governance layer (router/autonomy/reviews/policy). Days 7-8: dashboard + improvement lever + scale test. Days 9-10: write-ups, KPIs, demo video, pitch rehearsal.

## Deliverables mapped to the rubric
- Problem + market gap (vs pure eval platforms) + architecture write-ups (sections 1-3) in `docs/`.
- Working platform + baseline-vs-improved comparison (section 4 / 50%).
- Technical KPIs (agreement, latency, cost) + business framing (governance/decision automation, eval hours saved) for the pitch (sections 4-5).
