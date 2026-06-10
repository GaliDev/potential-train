# HF LLM-as-Judge Benchmark

Benchmarks open HuggingFace models as replacements for the GPT-4o judge
panel, against the project's hand-labeled gold set. Standalone layer: imports
`eval_harness` read-only and changes nothing in `src/`.

## Why these models

Long-term plan: pick the winning judge and fine-tune it in-house. The
candidates are therefore dense, small-to-mid models with training-friendly
licenses (every successful specialized judge - Prometheus 2, Selene, Glider -
was fine-tuned from exactly this class of base model):

| Key          | Model                               | Tier  | License          |
| ------------ | ----------------------------------- | ----- | ---------------- |
| `qwen3-8b`   | `Qwen/Qwen3-8B`                     | small | Apache 2.0       |
| `llama31-8b` | `meta-llama/Llama-3.1-8B-Instruct`  | small | Llama Community  |
| `qwen3-14b`  | `Qwen/Qwen3-14B`                    | mid   | Apache 2.0       |

Specialized judge models (Prometheus 2, Selene 1 Mini, Glider) were evaluated
first but are not routable via HF serverless Inference Providers (curated
catalog only); they remain options via local Ollama or a Featherless
subscription.

Models are called through the HF Inference Providers router
(`https://router.huggingface.co/v1`, OpenAI protocol). Requires `HF_TOKEN`
in `.env`. A full default run (30 items x 5 criteria x 3 models = 450 calls)
costs cents.

## Run

```bash
# Default: 3 models, 10 gold items per task family, HTML + JSON report
PYTHONPATH=src python -m hf_judges.run_benchmark

# Smaller / bigger slice
PYTHONPATH=src python -m hf_judges.run_benchmark --limit-per-type 4

# Subset of models
PYTHONPATH=src python -m hf_judges.run_benchmark --models qwen3-8b llama31-8b

# Include the current GPT-4o panel on the same items (needs OPENAI_API_KEY)
PYTHONPATH=src python -m hf_judges.run_benchmark --with-openai-baseline
```

Output lands in `hf_judges/results/`:

- `benchmark_<timestamp>.html` - the comparison report (open in a browser)
- `latest.html` - stable alias to the newest report
- `benchmark_<timestamp>.json` - raw numbers

## What is measured

Each model judges every item on the same five criteria as the production
panel (correctness, faithfulness, completeness, coherence, safety), with the
same prompts, deterministic aggregator, and metrics module:

- **Agreement with human gold labels**: pass/fail accuracy, Cohen's κ,
  Spearman ρ, score MAE - the platform bar is >= 80% accuracy and κ >= 0.6.
- **Per-criterion mean scores** - exposes lenient/harsh judging styles.
- **Output reliability** - how often the model emits clean JSON vs needing
  regex rescue (sloppy output = noisy verdicts).
- **Latency and estimated cost** per item.

## Tests (offline)

```bash
pytest hf_judges/tests/
```
