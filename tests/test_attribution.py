"""Tests for failure attribution engine."""

from __future__ import annotations

import pytest

from chains.discovery.causal import discover_step_quality_causation
from chains.attribution.engine import attribute_failure, attribute_failures


class TestAttribution:

    def test_attributes_to_retrieval(self, rag_traces):
        """Failures should be attributed to retrieval, not summarization."""
        graph = discover_step_quality_causation(rag_traces, significance_level=0.05)
        failed = [t for t in rag_traces if t.is_failure]
        assert len(failed) > 0

        rc = attribute_failure(failed[0], graph, rag_traces)
        # retrieval is the root cause (long queries → bad retrieval → bad quality)
        assert rc.root_step == "retrieval"
        assert rc.confidence > 0

    def test_attribute_all_failures(self, rag_traces):
        graph = discover_step_quality_causation(rag_traces, significance_level=0.05)
        results = attribute_failures(rag_traces, graph)
        assert len(results) > 0
        # Majority should trace to retrieval
        retrieval_count = sum(1 for r in results if r.root_step == "retrieval")
        assert retrieval_count >= len(results) * 0.5

    def test_root_cause_has_evidence(self, rag_traces):
        graph = discover_step_quality_causation(rag_traces, significance_level=0.05)
        failed = [t for t in rag_traces if t.is_failure]
        rc = attribute_failure(failed[0], graph, rag_traces)
        assert "z_score" in rc.evidence
        assert "causal_weight" in rc.evidence
