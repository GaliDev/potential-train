"""Lenient extraction of a judge verdict from open-model output.

Open models do not guarantee structured output, so parsing degrades
gracefully: strip reasoning tags, find the first JSON object, fall back to
regex on "score", and finally to any standalone 1-5 digit. A verdict is
always produced; `parse_confidence` records which path was taken.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from eval_harness.agents.base import PASS_THRESHOLD, clamp_score

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*"?([1-5])(?:\.\d+)?"?', re.IGNORECASE)
_DIGIT_RE = re.compile(r"\b([1-5])\s*(?:/\s*5)?\b")


@dataclass
class ParsedVerdict:
    score: int
    passed: bool
    rationale: str
    evidence: list[str] = field(default_factory=list)
    parse_confidence: str = "json"  # json | regex | fallback


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced {...} block that parses as JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_verdict(raw: str) -> ParsedVerdict:
    text = _THINK_RE.sub("", raw or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip() or text

    obj = _first_json_object(text)
    if obj is not None and obj.get("score") is not None:
        try:
            score = clamp_score(int(float(obj["score"])))
        except (TypeError, ValueError):
            score = None
        if score is not None:
            passed = obj.get("passed")
            evidence = obj.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            return ParsedVerdict(
                score=score,
                passed=bool(passed) if passed is not None else score >= PASS_THRESHOLD,
                rationale=str(obj.get("rationale") or "").strip(),
                evidence=[str(e) for e in evidence[:3]],
                parse_confidence="json",
            )

    m = _SCORE_RE.search(text)
    if m:
        score = clamp_score(int(m.group(1)))
        return ParsedVerdict(
            score=score,
            passed=score >= PASS_THRESHOLD,
            rationale=text[:300],
            parse_confidence="regex",
        )

    m = _DIGIT_RE.search(text)
    score = clamp_score(int(m.group(1))) if m else 3
    return ParsedVerdict(
        score=score,
        passed=score >= PASS_THRESHOLD,
        rationale=text[:300],
        parse_confidence="fallback",
    )
