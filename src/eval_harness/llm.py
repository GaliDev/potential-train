"""Thin OpenAI client wrapper with structured output and cost/latency tracking.

Every call returns the parsed result plus a `CallStats` record so the runner
can attribute latency and dollar cost down to the individual judge level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel

from .config import settings

T = TypeVar("T", bound=BaseModel)

# USD per token (derived from per-1M-token list prices). Update as needed.
# HF-router judge models register their rates from config/judge_panel.json
# via `register_model_pricing`.
_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# OpenAI-compatible endpoint of the HuggingFace Inference Providers router.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


def register_model_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register USD-per-1M-token rates so CallStats costs unknown models correctly."""
    _PER_1M[model] = {"input": input_per_1m, "output": output_per_1m}

_MAX_RETRIES = 6
_RETRY_BASE_DELAY_S = 1.0
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Fall back to the gpt-4o rate for unknown models so cost is never zero.
    rates = _PER_1M.get(model, _PER_1M["gpt-4o"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


def _with_retries(fn):
    """Retry transient OpenAI failures, especially TPM rate-limit bursts."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                time.sleep(min(delay, 30.0))
        raise last_exc  # pragma: no cover - loop always returns or raises earlier

    return wrapper


@dataclass
class CallStats:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """Accumulates cost/latency across many calls within a run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, stats: CallStats) -> None:
        self.calls += 1
        self.prompt_tokens += stats.prompt_tokens
        self.completion_tokens += stats.completion_tokens
        self.latency_s += stats.latency_s
        self.cost_usd += stats.cost_usd
        self.by_model[stats.model] = self.by_model.get(stats.model, 0.0) + stats.cost_usd


class LLMClient:
    """Wraps the OpenAI SDK for both free-text and structured (Pydantic) output."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_s if timeout_s is not None else settings.openai_timeout_s,
        )

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, CallStats]:
        model = model or settings.judge_model
        temperature = settings.judge_temperature if temperature is None else temperature
        extra = {} if max_tokens is None else {"max_tokens": max_tokens}
        start = time.perf_counter()
        resp = _with_retries(self._client.chat.completions.create)(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        stats = self._stats(model, resp, start)
        return resp.choices[0].message.content or "", stats

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[T, CallStats]:
        """Return a parsed Pydantic object using OpenAI structured outputs."""
        model = model or settings.judge_model
        temperature = settings.judge_temperature if temperature is None else temperature
        start = time.perf_counter()
        resp = _with_retries(self._client.beta.chat.completions.parse)(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
        )
        stats = self._stats(model, resp, start)
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Model {model} returned no parseable structured output.")
        return parsed, stats

    @staticmethod
    def _stats(model: str, resp, start: float) -> CallStats:
        latency = time.perf_counter() - start
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return CallStats(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cost_usd=_price(model, prompt_tokens, completion_tokens),
        )


_default_client: LLMClient | None = None
_hf_router_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Lazily build a shared client so importing this module never needs a key."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def get_hf_router_client() -> LLMClient:
    """Shared client for the HuggingFace Inference Providers router.

    The router speaks the OpenAI chat protocol, so it reuses LLMClient with a
    different base URL and the HF token as the API key.
    """
    global _hf_router_client
    if _hf_router_client is None:
        if not settings.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. The judge panel config maps criteria to "
                "HuggingFace-router models; add your HF access token to .env "
                "(get one at https://huggingface.co/settings/tokens)."
            )
        _hf_router_client = LLMClient(
            api_key=settings.hf_token,
            base_url=HF_ROUTER_BASE_URL,
            timeout_s=settings.hf_timeout_s,
        )
    return _hf_router_client
