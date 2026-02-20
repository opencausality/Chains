"""Tests for causal discovery module."""

from __future__ import annotations

import pytest

from chains.discovery.causal import discover_step_quality_causation


class TestCausalDiscovery:

    def test_discovers_retrieval_impact(self, rag_traces):
        graph = discover_step_quality_causation(rag_traces, significance_level=0.05)
        assert graph.n_traces == len(rag_traces)
        retrieval = graph.get_impact("retrieval")
        assert retrieval is not None
        assert retrieval.effect_size > 0

    def test_step_order(self, rag_traces):
        graph = discover_step_quality_causation(rag_traces)
        assert "retrieval" in graph.step_order
        assert "summarization" in graph.step_order
        assert "generation" in graph.step_order

    def test_significant_steps(self, rag_traces):
        graph = discover_step_quality_causation(rag_traces, significance_level=0.05)
        sig = graph.significant_steps
        sig_names = [s.step_name for s in sig]
        # retrieval should be significant because it drives quality
        assert "retrieval" in sig_names

    def test_minimum_traces(self):
        """Too few traces should not crash."""
        from chains.instrumentation.logger import PipelineTrace, PipelineStep
        traces = [PipelineTrace(trace_id="a", steps=[], quality_score=0.5)]
        graph = discover_step_quality_causation(traces)
        assert graph.n_traces == 1
        assert len(graph.step_impacts) == 0
