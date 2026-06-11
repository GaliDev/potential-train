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
| Judge LLMs       | Per-criterion open models via the HF Inference Providers router (Qwen3 8B / Llama 3.1 8B / Qwen3 14B, see `config/judge_panel.json`); OpenAI `gpt-4o` + `gpt-4o-mini` for forced single-model modes |
| Data models      | Pydantic                        |
| Performance store| SQLite via SQLModel             |
| Metrics          | pandas, scipy, scikit-learn     |
| API              | FastAPI                         |
| Console (GUI)    | Static SPA served by FastAPI    |

## Project layout

```
config/
  judge_panel.json     # criterion -> judge model mapping (see "Configuring the judges")
src/eval_harness/
  config.py            # env-driven settings
  judge_panel_config.py # loader/validator for config/judge_panel.json
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
app/api.py             # FastAPI endpoints + serves the web console
app/web/index.html     # web console (single-page GUI)
data/gold/             # curated hand-labeled gold set (tracked)
data/public/           # cached public eval slices (gitignored)
data/runs/             # generated run results (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your HF_TOKEN (default judge panel)
                       # and OPENAI_API_KEY (forced single-model modes)
```

## Run

```bash
# Benchmark the judge against human gold labels (needs HF_TOKEN for the
# per-criterion panel; OPENAI_API_KEY for baseline/cascade/jury modes)
PYTHONPATH=src python -m evaluation.benchmark --mode compare --with-improve

# Include public translation data (WMT) in addition to local gold labels
PYTHONPATH=src python -m evaluation.benchmark --mode compare --wmt --wmt-limit 40

# Populate the dashboard with real fleet outputs + persisted panel judgments
PYTHONPATH=src python -m evaluation.fleet_run --limit-per-type 1

# Judge live Langfuse traces and push the 5 criterion scores back to Langfuse
# (needs LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env)
PYTHONPATH=src python -m evaluation.langfuse_eval --hours 24 --limit 20 --push-scores

# Benchmark candidate judge LLMs (HF open models + Claude + Grok) against the
# gold set and pick the best judge overall and per criterion (see hf_judges/README.md)
PYTHONPATH=src python -m hf_judges.run_benchmark

# GUI + API: FastAPI serves the web console at http://localhost:8000
# (works offline via the "⋯" menu → "Seed data")
PYTHONPATH=src uvicorn app.api:app --reload

# Tests (offline, no API key needed)
pytest
```

## Configuring the judges

The five-criterion panel no longer runs on a single model: each criterion is
judged by the model that won it in the `hf_judges` benchmark. The mapping
lives in a dedicated config file, **`config/judge_panel.json`**:

```json
{
  "models": {
    "qwen3-8b":   { "model_id": "Qwen/Qwen3-8B", "provider": "hf-router", "prompt_suffix": "\n/no_think", "est_price_per_1m": [0.05, 0.10] },
    "llama31-8b": { "model_id": "meta-llama/Llama-3.1-8B-Instruct", "provider": "hf-router", "est_price_per_1m": [0.05, 0.10] },
    "qwen3-14b":  { "model_id": "Qwen/Qwen3-14B", "provider": "hf-router", "prompt_suffix": "\n/no_think", "est_price_per_1m": [0.08, 0.16] }
  },
  "criteria": {
    "correctness":  "qwen3-8b",
    "faithfulness": "llama31-8b",
    "completeness": "qwen3-14b",
    "coherence":    "qwen3-14b",
    "safety":       "qwen3-14b"
  }
}
```

- **`models`** is a registry of judge models. `provider` is `hf-router`
  (HuggingFace Inference Providers, OpenAI chat protocol, strict-JSON
  prompting + lenient parsing) or `openai` (native structured outputs).
  `prompt_suffix` is appended to the system prompt (e.g. `/no_think` disables
  Qwen3 thinking mode); `est_price_per_1m` is `[input, output]` USD per 1M
  tokens, used for cost tracking in reports.
- **`criteria`** maps each of the five criteria to a model key. All five must
  be mapped, and every key must exist in `models` - the loader fails fast on
  typos.

To change which model judges a criterion, edit the file (add the model under
`models` if new) - no code changes needed. Point the platform at a different
mapping file with the `JUDGE_PANEL_CONFIG` env var.

Requirements and fallbacks:

- `hf-router` models need `HF_TOKEN` in `.env`
  ([create one here](https://huggingface.co/settings/tokens)); `openai`
  models need `OPENAI_API_KEY`. `HF_TIMEOUT_S` (default 180) bounds each
  HF-router call.
- If the config file is missing, the panel logs a warning and falls back to
  the old behavior: every criterion judged by `JUDGE_MODEL` (default
  `gpt-4o`).
- Forcing a single model bypasses the mapping entirely: `PanelJudge(model=...)`,
  and the cascade/jury improvement judges (which use `JUDGE_MODEL` /
  `JUDGE_MODEL_CHEAP`), keep their previous behavior.
- Each verdict records the model that produced it, so per-criterion models
  show up as-is in run files, the dashboard, and Langfuse scores.

## Governance Console (Web UI)

The governance console is a single-page dashboard served directly by the FastAPI
server. No separate build step — just start the API and open a browser.

### Quick start

```bash
# 1. Activate your virtualenv (if not already active)
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Start the API server
PYTHONPATH=src uvicorn app.api:app --reload

# 3. Open the console
#    http://localhost:8000
```

The console polls `GET /decisions` on load. If the store is empty you'll see a
"No agent data yet" prompt — use the **⋯ → Seed data** button (no API key
needed) to populate 144 synthetic evaluations across all 12 agents and reload.

### Admin actions (⋯ menu, top-right)

| Action | What it does | Key needed? |
|---|---|---|
| **Seed data** | Inserts synthetic eval history for all 12 agents | No |
| **Simulate runtime** | Adds 30 eval rows per agent with drift scenarios (rag_weak safety incident, sum_weak latency spike) | No |
| **Run rag_react_agent** | Runs the real LangGraph RAG agent on 1 gold task, scores it with the 5-judge panel, persists | Yes (`OPENAI_API_KEY`) |
| **Run summarizer_refine_agent** | Same, for the refine-based summarizer | Yes |
| **Run translator_backcheck_agent** | Same, for the back-translation checker | Yes |

### Features

- **Fleet table** — tier badge, verdict, confidence, score bar, drift alerts for every agent
- **Decision spotlight** — click any row for the full governance decision: precedence chain, rationale, autonomy tier
- **Policy simulator** — drag the risk slider; debounced live `POST /policy` call shows how the verdict changes
- **Dark mode** — follows `prefers-color-scheme` automatically; no toggle needed

### Windows

```powershell
$env:PYTHONPATH = "src"
uvicorn app.api:app --reload
```

Then open http://localhost:8000 in your browser.

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
console leaderboard and governance views), and - with `--push-scores` - as
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
