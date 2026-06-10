"""Model registry and HF router configuration for the judge benchmark.

The three default candidates are dense, permissively licensed models that are
practical to fine-tune later (the long-term plan: benchmark -> pick a winner ->
train it as the in-house judge). All are served via the HuggingFace Inference
Providers router, which speaks the OpenAI chat-completions protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

load_dotenv(PROJECT_ROOT / ".env")

ROUTER_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class JudgeModelSpec:
    key: str  # short name used on the CLI and in reports
    model_id: str  # HF hub id, as accepted by the router
    display_name: str
    tier: str  # small / mid / large (within this benchmark)
    license: str
    # Rough USD per 1M tokens (input, output) across serverless providers.
    # Indicative only - used to estimate run cost in the report.
    est_price_per_1m: tuple[float, float] = (0.05, 0.10)
    # Appended to the system prompt; Qwen3 hybrid models use "/no_think" to
    # disable thinking mode so we get the JSON verdict directly.
    prompt_suffix: str = ""


MODELS: dict[str, JudgeModelSpec] = {
    "qwen3-8b": JudgeModelSpec(
        key="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        display_name="Qwen3 8B",
        tier="small",
        license="Apache 2.0",
        est_price_per_1m=(0.05, 0.10),
        prompt_suffix="\n/no_think",
    ),
    "llama31-8b": JudgeModelSpec(
        key="llama31-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        tier="small",
        license="Llama Community",
        est_price_per_1m=(0.05, 0.10),
    ),
    "qwen3-14b": JudgeModelSpec(
        key="qwen3-14b",
        model_id="Qwen/Qwen3-14B",
        display_name="Qwen3 14B",
        tier="mid",
        license="Apache 2.0",
        est_price_per_1m=(0.08, 0.16),
        prompt_suffix="\n/no_think",
    ),
}


def hf_token() -> str:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add your HuggingFace access token to .env "
            "(get one at https://huggingface.co/settings/tokens)."
        )
    return token
