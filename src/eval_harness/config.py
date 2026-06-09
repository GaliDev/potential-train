"""Central configuration loaded from environment variables.

Reads a local .env file (see .env.example) if present. All other modules
import `settings` from here rather than reading os.environ directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Project root = two levels up from this file (src/eval_harness/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
PUBLIC_DIR = DATA_DIR / "public"
RUNS_DIR = DATA_DIR / "runs"

load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """Typed view over the environment configuration."""

    openai_api_key: str = ""
    judge_model: str = "gpt-4o"
    judge_model_cheap: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    openai_timeout_s: float = 60.0
    max_concurrency: int = 5
    database_url: str = "sqlite:///data/governance.db"
    # Langfuse integration (self-hosted or cloud). Keys come from the
    # project settings page in the Langfuse UI.
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the settings object from the current environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4o"),
        judge_model_cheap=os.getenv("JUDGE_MODEL_CHEAP", "gpt-4o-mini"),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
        openai_timeout_s=float(os.getenv("OPENAI_TIMEOUT_S", "60.0")),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "5")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/governance.db"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    )


settings = get_settings()
