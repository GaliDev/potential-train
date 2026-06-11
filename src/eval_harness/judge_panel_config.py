"""Per-criterion judge model mapping, loaded from a dedicated config file.

The benchmark in `hf_judges/` compares candidate judge LLMs per criterion; the
winners are recorded in `config/judge_panel.json` (criterion -> model key plus
a model registry). The panel reads that mapping here so each criterion judge
runs on its own model. Override the file location with the JUDGE_PANEL_CONFIG
environment variable.

A missing config file degrades gracefully (the panel falls back to the single
`settings.judge_model`); a present-but-invalid file raises, so a typo never
silently reroutes the panel.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import PROJECT_ROOT
from .llm import register_model_pricing
from .schemas import Criterion

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "judge_panel.json"

_VALID_PROVIDERS = {"hf-router", "openai"}


class JudgePanelConfigError(ValueError):
    """Raised when the judge panel config file exists but is invalid."""


@dataclass(frozen=True)
class PanelModelSpec:
    """One judge model as described in the config file's `models` registry."""

    key: str
    model_id: str
    display_name: str
    provider: str  # "hf-router" (JSON prompting) or "openai" (structured outputs)
    prompt_suffix: str = ""  # e.g. "/no_think" for Qwen3 hybrid models
    est_price_per_1m: tuple[float, float] = (0.05, 0.10)  # USD (input, output)


@dataclass(frozen=True)
class JudgePanelConfig:
    models: dict[str, PanelModelSpec]
    criteria: dict[Criterion, str]  # criterion -> model key
    path: Path

    def spec_for(self, criterion: Criterion) -> PanelModelSpec:
        return self.models[self.criteria[criterion]]


def _config_path() -> Path:
    override = os.getenv("JUDGE_PANEL_CONFIG", "")
    return Path(override) if override else DEFAULT_CONFIG_PATH


def _parse_model(key: str, raw: dict) -> PanelModelSpec:
    if not isinstance(raw, dict) or not raw.get("model_id"):
        raise JudgePanelConfigError(f"models[{key!r}] must be an object with a 'model_id'.")
    provider = raw.get("provider", "hf-router")
    if provider not in _VALID_PROVIDERS:
        raise JudgePanelConfigError(
            f"models[{key!r}] has unknown provider {provider!r} "
            f"(expected one of {sorted(_VALID_PROVIDERS)})."
        )
    price = raw.get("est_price_per_1m", [0.05, 0.10])
    if not (isinstance(price, (list, tuple)) and len(price) == 2):
        raise JudgePanelConfigError(
            f"models[{key!r}].est_price_per_1m must be [input, output] USD per 1M tokens."
        )
    return PanelModelSpec(
        key=key,
        model_id=str(raw["model_id"]),
        display_name=str(raw.get("display_name", key)),
        provider=provider,
        prompt_suffix=str(raw.get("prompt_suffix", "")),
        est_price_per_1m=(float(price[0]), float(price[1])),
    )


def load_panel_config(path: Path) -> JudgePanelConfig:
    """Load and validate a judge panel config file. Raises on invalid content."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise JudgePanelConfigError(f"{path} is not valid JSON: {exc}") from exc

    models = {
        key: _parse_model(key, spec) for key, spec in (raw.get("models") or {}).items()
    }
    if not models:
        raise JudgePanelConfigError(f"{path} defines no models under 'models'.")

    raw_criteria = raw.get("criteria") or {}
    criteria: dict[Criterion, str] = {}
    for name, model_key in raw_criteria.items():
        try:
            criterion = Criterion(name)
        except ValueError as exc:
            raise JudgePanelConfigError(
                f"{path}: unknown criterion {name!r} "
                f"(expected one of {[c.value for c in Criterion]})."
            ) from exc
        if model_key not in models:
            raise JudgePanelConfigError(
                f"{path}: criteria[{name!r}] references undefined model {model_key!r}."
            )
        criteria[criterion] = model_key

    missing = [c.value for c in Criterion if c not in criteria]
    if missing:
        raise JudgePanelConfigError(f"{path}: criteria mapping is missing {missing}.")

    # Make cost tracking in llm.CallStats use the configured rates.
    for spec in models.values():
        register_model_pricing(spec.model_id, *spec.est_price_per_1m)

    return JudgePanelConfig(models=models, criteria=criteria, path=path)


@lru_cache(maxsize=1)
def get_panel_config() -> JudgePanelConfig | None:
    """Cached panel config; None when the config file does not exist."""
    path = _config_path()
    if not path.exists():
        logger.warning(
            "Judge panel config %s not found; the panel falls back to the single "
            "JUDGE_MODEL (%s).", path, os.getenv("JUDGE_MODEL", "gpt-4o"),
        )
        return None
    return load_panel_config(path)
