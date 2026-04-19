"""
Shared pytest fixtures for the news engine tests.

These tests are fully offline. The `fixtures/` directory holds synthetic but
realistic payloads shaped like real API responses. During first real-network
development, replace these with recorded responses via `make news-smoke`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `trading-bot/` importable so tests can do `from strategies.news import ...`.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_fixture():
    """Load a JSON fixture by filename relative to fixtures/."""
    def _load(name: str):
        path = FIXTURES_DIR / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return _load


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch):
    """Point NEWS_CACHE_DIR at a tmp dir so tests never touch the real cache."""
    cache_dir = tmp_path / "news_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("NEWS_CACHE_DIR", str(cache_dir))
    return cache_dir
