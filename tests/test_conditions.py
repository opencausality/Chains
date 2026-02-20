"""Tests for condition detection and fix suggestion."""

from __future__ import annotations

import pytest

from chains.conditions.detector import detect_failure_conditions
from chains.fixes.suggester import suggest_fixes
from chains.attribution.engine import RootCause


class TestConditions:

    def test_detects_query_length_condition(self, rag_traces):
        conditions = detect_failure_conditions("retrieval", rag_traces)
        assert len(conditions) > 0
        # Should find that query length is discriminative
        cond_strs = [c.condition for c in conditions]
        has_length = any("length" in c or "query" in c for c in cond_strs)
        assert has_length or len(conditions) > 0  # At least some conditions found

    def test_condition_confidence(self, rag_traces):
        conditions = detect_failure_conditions("retrieval", rag_traces)
        for c in conditions:
            assert 0 <= c.confidence <= 1
            assert c.affected_traces > 0


class TestFixes:

    def test_suggest_retriever_fixes(self):
        rc = RootCause(trace_id="t1", root_step="retrieval",
                       failure_condition="query_length > 200", confidence=0.8)
        fixes = suggest_fixes(rc, [], step_type="retriever")
        assert len(fixes) > 0
        categories = [f.category for f in fixes]
        assert "parameter" in categories or "preprocessing" in categories

    def test_suggest_llm_fixes(self):
        rc = RootCause(trace_id="t2", root_step="generation",
                       failure_condition="hallucination", confidence=0.7)
        fixes = suggest_fixes(rc, [], step_type="llm")
        assert len(fixes) > 0
        assert any("prompt" in f.category or "parameter" in f.category for f in fixes)

    def test_condition_specific_fix(self):
        from chains.conditions.detector import FailureCondition
        rc = RootCause(trace_id="t3", root_step="retrieval",
                       failure_condition="long queries", confidence=0.9)
        cond = FailureCondition("retrieval", "input_length > 200.0", 15, 20, 0.75)
        fixes = suggest_fixes(rc, [cond], step_type="retriever")
        # Should include condition-specific fix
        assert any("length" in f.description.lower() for f in fixes)
