"""Candidate judge models wrapped as panel judges.

Three provider paths, one Judge protocol (`.judge(item) -> AggregateResult`,
`.judge_mode`) so the existing runner, metrics, and benchmark machinery work
unchanged for every candidate:

- HF Inference Providers router and xAI both speak the OpenAI chat protocol ->
  `OpenAIProtocolJudgeClient` + JSON prompting + lenient parsing.
- Claude is called through the official Anthropic SDK with native structured
  outputs (`messages.parse` into the same `CriterionResponse` model the
  production OpenAI panel uses).

The five criterion calls run concurrently per item; aggregation reuses the
deterministic meta-judge.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from eval_harness.agents.aggregator import aggregate
from eval_harness.agents.base import PASS_THRESHOLD, clamp_score
from eval_harness.agents.criteria import PANEL_CRITERIA, CriterionResponse
from eval_harness.schemas import AggregateResult, Criterion, JudgeVerdict, TestItem

from .config import (
    ROUTER_BASE_URL,
    XAI_BASE_URL,
    JudgeModelSpec,
    anthropic_api_key,
    hf_token,
    xai_api_key,
)
from .parsers import parse_verdict
from .prompts import criterion_brief, system_prompt, user_prompt

_MAX_RETRIES = 5
_RETRY_BASE_DELAY_S = 2.0
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class OpenAIProtocolJudgeClient:
    """Chat-completions client for OpenAI-protocol providers (HF router, xAI)."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 180.0) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 600,
    ) -> tuple[str, float, int, int]:
        """Returns (text, latency_s, prompt_tokens, completion_tokens)."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    raise
                time.sleep(min(_RETRY_BASE_DELAY_S * (2**attempt), 30.0))
                continue
            latency = time.perf_counter() - start
            usage = getattr(resp, "usage", None)
            return (
                resp.choices[0].message.content or "",
                latency,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
        raise last_exc  # pragma: no cover


class HFPanelJudge:
    """Five-criterion panel judge backed by one OpenAI-protocol model."""

    def __init__(
        self,
        spec: JudgeModelSpec,
        client: OpenAIProtocolJudgeClient | None = None,
        max_workers: int = 5,
    ) -> None:
        self.spec = spec
        self.judge_mode = f"hf_{spec.key}"
        self._client = client or _openai_protocol_client(spec)
        self._max_workers = max_workers
        # Parse-quality telemetry across the whole run.
        self.parse_counts: dict[str, int] = {"json": 0, "regex": 0, "fallback": 0}

    def judge(self, item: TestItem) -> AggregateResult:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            verdicts = list(
                pool.map(lambda c: self._judge_criterion(c, item), PANEL_CRITERIA)
            )
        return aggregate(
            item_id=item.id,
            task_type=item.task_type,
            verdicts=verdicts,
            agent_id=item.agent_id,
            judge_mode=self.judge_mode,
        )

    def _judge_criterion(self, criterion: Criterion, item: TestItem) -> JudgeVerdict:
        text, latency, p_tokens, c_tokens = self._client.complete(
            system=system_prompt(criterion, self.spec.prompt_suffix),
            user=user_prompt(item),
            model=self.spec.model_id,
        )
        parsed = parse_verdict(text)
        self.parse_counts[parsed.parse_confidence] = (
            self.parse_counts.get(parsed.parse_confidence, 0) + 1
        )
        in_rate, out_rate = self.spec.est_price_per_1m
        return JudgeVerdict(
            criterion=criterion,
            score=parsed.score,
            passed=parsed.passed,
            rationale=parsed.rationale,
            evidence=parsed.evidence,
            model=self.spec.model_id,
            latency_s=round(latency, 4),
            cost_usd=round((p_tokens * in_rate + c_tokens * out_rate) / 1_000_000, 8),
        )


class ClaudePanelJudge:
    """Five-criterion panel judge backed by Claude via the Anthropic SDK.

    Uses native structured outputs (`messages.parse`) into the same
    `CriterionResponse` model the production OpenAI panel parses into, plus
    adaptive thinking - so every verdict is schema-valid by construction.
    """

    def __init__(self, spec: JudgeModelSpec, max_workers: int = 5) -> None:
        import anthropic

        anthropic_api_key()  # fail fast with a clear message if unset
        self.spec = spec
        self.judge_mode = f"hf_{spec.key}"
        self._client = anthropic.Anthropic(max_retries=4)
        self._max_workers = max_workers
        self.parse_counts: dict[str, int] = {"json": 0, "regex": 0, "fallback": 0}

    def judge(self, item: TestItem) -> AggregateResult:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            verdicts = list(
                pool.map(lambda c: self._judge_criterion(c, item), PANEL_CRITERIA)
            )
        return aggregate(
            item_id=item.id,
            task_type=item.task_type,
            verdicts=verdicts,
            agent_id=item.agent_id,
            judge_mode=self.judge_mode,
        )

    def _judge_criterion(self, criterion: Criterion, item: TestItem) -> JudgeVerdict:
        start = time.perf_counter()
        resp = self._client.messages.parse(
            model=self.spec.model_id,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=criterion_brief(criterion),
            messages=[{"role": "user", "content": user_prompt(item)}],
            output_format=CriterionResponse,
        )
        latency = time.perf_counter() - start
        parsed = resp.parsed_output
        if parsed is None:
            raise ValueError(
                f"Claude returned no parseable structured output "
                f"(stop_reason={resp.stop_reason})."
            )
        self.parse_counts["json"] += 1
        score = clamp_score(parsed.score)
        in_rate, out_rate = self.spec.est_price_per_1m
        usage = resp.usage
        return JudgeVerdict(
            criterion=criterion,
            score=score,
            passed=parsed.passed if parsed.passed is not None else score >= PASS_THRESHOLD,
            rationale=parsed.rationale,
            evidence=parsed.evidence[:3],
            model=self.spec.model_id,
            latency_s=round(latency, 4),
            cost_usd=round(
                (usage.input_tokens * in_rate + usage.output_tokens * out_rate)
                / 1_000_000,
                8,
            ),
        )


def _openai_protocol_client(spec: JudgeModelSpec) -> OpenAIProtocolJudgeClient:
    if spec.provider == "xai":
        return OpenAIProtocolJudgeClient(base_url=XAI_BASE_URL, api_key=xai_api_key())
    return OpenAIProtocolJudgeClient(base_url=ROUTER_BASE_URL, api_key=hf_token())


def build_panel_judge(spec: JudgeModelSpec):
    """Construct the right panel judge for a spec's provider.

    Raises RuntimeError with a clear message when the provider's API key is
    missing, so the benchmark can skip that model instead of crashing.
    """
    if spec.provider == "anthropic":
        return ClaudePanelJudge(spec)
    return HFPanelJudge(spec)
