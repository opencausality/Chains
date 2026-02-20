"""Tests for instrumentation: TraceLogger and data model."""

from __future__ import annotations

import json
import pytest

from chains.instrumentation.logger import TraceLogger, PipelineTrace, PipelineStep, load_traces


class TestTraceLogger:

    def test_full_lifecycle(self):
        logger = TraceLogger()
        tid = logger.start_trace(context={"query": "test"})
        logger.log_step("retrieval", "retriever", inputs={"query": "test"},
                        outputs={"text": "doc content"}, latency_ms=100)
        logger.log_step("generation", "llm", inputs={"text": "doc content"},
                        outputs={"text": "answer"}, latency_ms=300)
        trace = logger.end_trace(final_output="answer", quality_score=0.9)

        assert trace.trace_id == tid
        assert len(trace.steps) == 2
        assert trace.quality_score == 0.9
        assert not trace.is_failure

    def test_failure_detection(self):
        trace = PipelineTrace(trace_id="t1", steps=[], quality_score=0.3)
        assert trace.is_failure == True

    def test_step_names(self):
        trace = PipelineTrace(
            trace_id="t1",
            steps=[
                PipelineStep("retrieval", "retriever"),
                PipelineStep("generation", "llm"),
            ],
        )
        assert trace.step_names == ["retrieval", "generation"]

    def test_export_and_load(self, tmp_path):
        logger = TraceLogger()
        logger.start_trace()
        logger.log_step("step1", "llm", outputs={"text": "hello"})
        logger.end_trace(quality_score=0.8)
        logger.start_trace()
        logger.log_step("step1", "llm", outputs={"text": "world"})
        logger.end_trace(quality_score=0.3)

        out = tmp_path / "traces.json"
        logger.export(str(out))
        assert out.exists()

        loaded = load_traces(str(out))
        assert len(loaded) == 2
        assert loaded[0].quality_score == 0.8
        assert loaded[1].quality_score == 0.3
