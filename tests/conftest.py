"""Shared pytest fixtures for Chains."""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic_traces import build_rag_traces


@pytest.fixture
def rag_traces():
    return build_rag_traces(n_traces=100, failure_rate=0.15, seed=42)


@pytest.fixture
def small_traces():
    return build_rag_traces(n_traces=20, failure_rate=0.3, seed=99)
