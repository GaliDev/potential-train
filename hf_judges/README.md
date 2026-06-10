# LLM-as-Judge Benchmark

Benchmarks candidate LLM judges - open HuggingFace models, **Claude**, and
**Grok** - against the project's hand-labeled gold set, and declares the best
judge **overall** and the best judge **per criterion** (correctness,
faithfulness, completeness, coherence, safety), so the platform can run a
mixed panel built from the best LLM for each criterion. Standalone layer:
imports `eval_harness` read-only and changes nothing in `src/`.

## Candidate judges

| Key          | Model                              | Provider / API                | License         | Key needed          |
| ------------ | ---------------------------------- | ----------------------------- | --------------- | ------------------- |
| `qwen3-8b`   | `Qwen/Qwen3-8B`                    | HF Inference Providers router | Apache 2.0      | `HF_TOKEN`          |
| `llama31-8b` | `meta-llama/Llama-3.1-8B-Instruct` | HF Inference Providers router | Llama Community | `HF_TOKEN`          |
| `qwen3-14b`  | `Qwen/Qwen3-14B`                   | HF Inference Providers router | Apache 2.0      | `HF_TOKEN`          |
| `claude`     | `claude-opus-4-8`                  | Anthropic SDK (native structured output + adaptive thinking) | proprietary API | `ANTHROPIC_API_KEY` |
| `grok`       | `grok-4`                           | xAI (OpenAI protocol)         | proprietary API | `XAI_API_KEY`       |

The open models are dense, fine-tune-friendly bases (the long-term plan:
benchmark -> pick a winner -> train it in-house); Claude and Grok set the
frontier reference points. Models whose key is missing from `.env` are
skipped with a warning, not a crash. Specialized judge models (Prometheus 2,
Selene 1 Mini, Glider) were evaluated first but are not routable via HF
serverless (curated catalog only).

## Gold data

The benchmark runs over the project's full hand-labeled gold set by default:

- `data/gold/rag_qa_gold.jsonl` - RAG Q&A
- `data/gold/summarization_gold.jsonl` - summarization
- `data/gold/translation_gold.jsonl` - translation

Each item carries a human `gold_score` (1-5) and `gold_pass` label that the
judges' verdicts are compared against.

## Run

```bash
# Full benchmark: every candidate with a configured key, full gold set
PYTHONPATH=src python -m hf_judges.run_benchmark

# Smaller slice (N items per task family) while iterating
PYTHONPATH=src python -m hf_judges.run_benchmark --limit-per-type 5

# Subset of models
PYTHONPATH=src python -m hf_judges.run_benchmark --models claude grok qwen3-8b

# Include the current GPT-4o panel on the same items (needs OPENAI_API_KEY)
PYTHONPATH=src python -m hf_judges.run_benchmark --with-openai-baseline
```

Output lands in `hf_judges/results/`:

- `benchmark_<timestamp>.html` - the comparison report (open in a browser)
- `latest.html` - stable alias to the newest report
- `benchmark_<timestamp>.json` - raw numbers

Cost note: the open models cost cents per run; Claude Opus and Grok 4 are
frontier-priced - a full 76-item run (380 judge calls each) is roughly $5-12
for Claude and $2-5 for Grok, depending on thinking/reasoning token usage.

## What is measured

Each model judges every item on the same five criteria as the production
panel (correctness, faithfulness, completeness, coherence, safety), with the
same prompts, deterministic aggregator, and metrics module.

**Winner selection** (shown at the top of the HTML report):

- **Best overall** - ranked by Cohen's κ, then Spearman ρ, then score MAE,
  then pass accuracy against the human gold labels.
- **Best per criterion** - for each criterion, how well that dimension's 1-5
  score tracks the human gold score (Spearman ρ, then MAE, then pass
  accuracy). This table is the recommended mixed panel: the best LLM for
  each criterion.

**Also reported per model:**

- Agreement with human gold labels: pass/fail accuracy, Cohen's κ, Spearman
  ρ, score MAE - the platform bar is >= 80% accuracy and κ >= 0.6.
- Per-criterion mean scores - exposes lenient/harsh judging styles.
- Output reliability - clean JSON vs regex-rescued vs fallback verdicts
  (Claude uses native structured output, so it is always clean by
  construction).
- Latency and estimated cost per item.

## Tests (offline)

```bash
pytest hf_judges/tests/
```
