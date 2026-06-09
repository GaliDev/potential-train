# Agent Workforce Governance

Manage a fleet of AI agents like a team - not just a leaderboard.

Most evaluation tools hand you a score and stop. This platform adds the
**decision layer** on top: a multiagent LLM-as-judge engine continuously
scores a fleet of agents, and a governance layer turns that history into the
four questions other platforms skip:

1. **Which agent should get the next task?** (performance-aware routing)
2. **How much autonomy should each agent have?** (reliability -> autonomy tier)
3. **How is each agent performing over time?** (automated performance reviews)
4. **How do we govern the fleet?** (policy enforcement + audit log)

The judge engine is held to a hard standard: its verdicts are validated
against human gold labels (target: >= 80% agreement, Cohen's kappa >= 0.6).

The current managed fleet covers three demo task families: **RAG Q&A**,
**summarization**, and **translation**. Each task family has a strong agent, a
cheap agent, and a deliberately weaker prompt variant so routing, autonomy, and
performance reviews have visible quality differences.

## Tech stack

| Component        | Choice                          |
| ---------------- | ------------------------------- |
| Orchestration    | LangGraph                       |
| LLM              | OpenAI `gpt-4o` + `gpt-4o-mini` |
| Data models      | Pydantic                        |
| Performance store| SQLite via SQLModel             |
| Metrics          | pandas, scipy, scikit-learn     |
| API              | FastAPI                         |
| Dashboard        | Streamlit                       |

## Project layout

```
src/eval_harness/
  config.py            # env-driven settings
  schemas.py           # Pydantic models (TestItem, JudgeVerdict, AgentProfile, ...)
  llm.py               # OpenAI wrapper + token/cost tracking
  store.py             # SQLite performance store (registry + results)
  graph.py             # LangGraph judge panel
  runner.py            # run the engine over a test set
  agents/              # per-criterion judges + aggregator + jury
  datasets/            # public slice + custom gold-set loaders
  fleet/               # managed agent configs + output generator
  governance/          # router, autonomy, reviews, policy
  calibration.py       # bias mitigation + thresholds
  metrics.py           # kappa / Spearman / accuracy vs gold
evaluation/benchmark.py # agreement metrics + report
app/api.py             # FastAPI endpoints
app/ui.py              # Streamlit dashboard
data/gold/             # curated hand-labeled gold set (tracked)
data/public/           # cached public eval slices (gitignored)
data/runs/             # generated run results (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run

```bash
# Benchmark the judge against human gold labels (needs OPENAI_API_KEY)
PYTHONPATH=src python -m evaluation.benchmark --mode compare --with-improve

# Include public translation data (WMT) in addition to local gold labels
PYTHONPATH=src python -m evaluation.benchmark --mode compare --wmt --wmt-limit 40

# Populate the dashboard with real fleet outputs + persisted panel judgments
PYTHONPATH=src python -m evaluation.fleet_run --limit-per-type 1

# Judge live Langfuse traces and push the 5 criterion scores back to Langfuse
# (needs LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env)
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --limit 20 --push-scores

# Dashboard (works offline via the "Seed demo data" button)
PYTHONPATH=src streamlit run app/ui.py

# API
PYTHONPATH=src uvicorn app.api:app --reload

# Tests (offline, no API key needed)
pytest
```

## Langfuse integration

The platform can act as an evaluation layer on top of [Langfuse](https://langfuse.com):
it pulls the traces Langfuse collected from your LLM apps, runs the five-criterion
judge panel (correctness, faithfulness, completeness, coherence, safety) over them
on your side, stores the results in the local performance store, and (optionally)
pushes the scores back onto each trace in the Langfuse UI.

### 0. Prerequisite: run Langfuse with Docker

Langfuse must be running **before** any of the commands below. Self-host it with
Docker (requires [Docker Desktop](https://docs.docker.com/get-docker/)):

```bash
# Download (clone) Langfuse and start the full stack
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d

# Wait until the web container is healthy, then verify:
curl http://localhost:3000/api/public/health
```

By default the UI is at http://localhost:3000. If that port is taken (as on this
machine, where the UI runs on **3001** via `docker-compose.override.yml`), adjust
`LANGFUSE_HOST` accordingly. Create a project in the UI (Settings -> API Keys)
and put the keys in `.env`:

```bash
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 1. Retrieve data from Langfuse and judge it

All retrieval goes through `evaluation.langfuse_eval`, which fetches traces via
the Langfuse Public API, maps them onto TestItems, and runs the judge panel:

```bash
# Preview which traces would be fetched - no LLM calls, no cost
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --dry-run

# Pull the last 24h of traces (max 20), judge them, store results locally
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --limit 20

# Also push the per-criterion scores back onto the traces in Langfuse
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --limit 20 --push-scores

# Filter which traces to pull
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 48 --tags my-tag        # by tag(s)
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 48 --name rag-pipeline  # by trace name
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 48 --user-id user-42    # by user
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 48 --session-id sess-7  # by session

# Fetch full traces (extra API call each) to extract retrieval context
# from observations - feeds the faithfulness judge
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --with-observations

# Cheaper single-call judge instead of the 5-judge panel
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --judge baseline

# Task family used when a trace has no tag/metadata hint (default: rag_qa)
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --default-task-type summarization

# Judge without writing to the local SQLite store
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --no-persist
```

Results land in three places: `data/runs/run_panel_<timestamp>.json` (full
verdicts + rationales), the `evals` table in `data/governance.db` (feeds the
Streamlit leaderboard and governance views), and - with `--push-scores` - as
`judge_*` scores on each trace in the Langfuse UI.

### 2. Add an agent and example data to Langfuse

Traces are normally produced by your Langfuse-instrumented apps, but you can
insert one manually via the API. The `agent_id` in metadata becomes the agent
on the fleet leaderboard (registered automatically on the next eval run);
`tags` / `metadata.task_type` route the trace to the right task family
(`rag_qa`, `summarization`, `translation`); `metadata.context` grounds the
faithfulness judge and `metadata.reference` the correctness judge.

```bash
curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  -X POST "$LANGFUSE_HOST/api/public/traces" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-agent-run",
    "input": "What year was the Eiffel Tower completed?",
    "output": "The Eiffel Tower was completed in 1889.",
    "tags": ["rag", "manual-demo"],
    "metadata": {
      "agent_id": "my-new-agent",
      "context": "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889.",
      "reference": "It was completed in 1889."
    }
  }'
```

Ingestion is asynchronous - wait a few seconds, then judge just that trace by
its tag and watch `my-new-agent` appear on the dashboard leaderboard:

```bash
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 1 --tags manual-demo --push-scores
```

## Docs

- [docs/report.md](docs/report.md) - full assignment write-up (problem, market, architecture, implementation, KPIs).
- [docs/pitch.md](docs/pitch.md) - the pitch.
- [docs/plan.md](docs/plan.md) - the build plan.
