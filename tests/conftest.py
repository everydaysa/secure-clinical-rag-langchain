"""Shared fixtures.

Every test runs against the mock provider: no Azure, no credentials, no spend,
no network. That is what makes the suite usable in CI and repeatable enough to
assert on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.config import Settings  # noqa: E402
from clinical_rag.corpus import load_corpus  # noqa: E402
from clinical_rag.vectorstore import build_index  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Offline settings pointing at a throwaway index and audit log."""
    tmp = tmp_path_factory.mktemp("clinical_rag")
    return Settings(
        _env_file=None,
        llm_provider="mock",
        index_dir=str(tmp / "index"),
        audit_log_path=str(tmp / "audit.jsonl"),
    )


@pytest.fixture(scope="session")
def indexed(settings: Settings) -> Settings:
    """Build the test index once for the whole session."""
    build_index(load_corpus(ROOT / "data" / "corpus"), settings=settings)
    return settings
