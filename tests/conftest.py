"""Shared test fixtures.

Points the store at a throwaway SQLite database (so tests never touch the real
data/governance.db) and resets tables between tests.
"""

from __future__ import annotations

import os
import tempfile

# Must be set BEFORE eval_harness.config is first imported so the store engine
# binds to the temp database.
_TMP_DB = os.path.join(tempfile.gettempdir(), "awg_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ.setdefault("OPENAI_API_KEY", "")

import pytest  # noqa: E402

from eval_harness.llm import CallStats  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_db():
    """Reset all tables before each test for isolation."""
    from eval_harness.store import reset_db

    reset_db()
    yield
    reset_db()


class FakeClient:
    """Stand-in LLM client that returns scripted structured/text responses.

    `score_fn(model)` decides the score so tests can simulate model-dependent
    behavior (e.g. cheap vs strong in the cascade).
    """

    def __init__(self, score_fn=None, cost_fn=None):
        self.score_fn = score_fn or (lambda model: 5)
        self.cost_fn = cost_fn or (lambda model: 0.001)
        self.calls = []

    def complete_structured(self, *, system, user, response_model, model=None, temperature=None):
        self.calls.append(("structured", model))
        score = self.score_fn(model)
        resp = response_model(
            score=score,
            passed=score >= 4,
            rationale="fake rationale",
            evidence=["quote"],
        )
        return resp, CallStats(model=model or "gpt-4o", prompt_tokens=50,
                               completion_tokens=10, latency_s=0.2, cost_usd=self.cost_fn(model))

    def complete_text(self, *, system, user, model=None, temperature=None, max_tokens=None):
        self.calls.append(("text", model))
        return "fake text", CallStats(model=model or "gpt-4o", prompt_tokens=50,
                                      completion_tokens=10, latency_s=0.2, cost_usd=self.cost_fn(model))


@pytest.fixture
def fake_client_factory():
    return FakeClient
