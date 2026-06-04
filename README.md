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

# Dashboard (works offline via the "Seed demo data" button)
PYTHONPATH=src streamlit run app/ui.py

# API
PYTHONPATH=src uvicorn app.api:app --reload

# Tests (offline, no API key needed)
pytest
```

## Docs

- [docs/report.md](docs/report.md) - full assignment write-up (problem, market, architecture, implementation, KPIs).
- [docs/pitch.md](docs/pitch.md) - the pitch.
- [docs/plan.md](docs/plan.md) - the build plan.
