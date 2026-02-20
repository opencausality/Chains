"""API routes for Chains."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chains import __version__
from chains.config import LLMProvider, get_settings
from chains.instrumentation.logger import PipelineStep, PipelineTrace
from chains.discovery.causal import discover_step_quality_causation
from chains.attribution.engine import attribute_failures

router = APIRouter()


class StepPayload(BaseModel):
    name: str
    step_type: str = "llm"
    inputs: dict = {}
    outputs: dict = {}
    metadata: dict = {}
    latency_ms: float = 0


class TracePayload(BaseModel):
    trace_id: str = ""
    steps: list[StepPayload]
    quality_score: float | None = None
    context: dict = {}


class AnalyzeRequest(BaseModel):
    traces: list[TracePayload]


class RootCauseResult(BaseModel):
    trace_id: str
    root_step: str
    condition: str
    confidence: float


class AnalyzeResponse(BaseModel):
    status: str
    n_traces: int
    n_failures: int
    root_causes: list[RootCauseResult]


@router.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@router.get("/providers")
def list_providers():
    return {"providers": [p.value for p in LLMProvider]}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(request: AnalyzeRequest):
    traces = []
    for tp in request.traces:
        steps = [PipelineStep(name=s.name, step_type=s.step_type, inputs=s.inputs,
                              outputs=s.outputs, metadata=s.metadata, latency_ms=s.latency_ms)
                 for s in tp.steps]
        traces.append(PipelineTrace(trace_id=tp.trace_id, steps=steps,
                                     quality_score=tp.quality_score, context=tp.context))

    settings = get_settings()
    graph = discover_step_quality_causation(traces, settings.significance_level)
    root_causes = attribute_failures(traces, graph)

    return AnalyzeResponse(
        status="success",
        n_traces=len(traces),
        n_failures=sum(1 for t in traces if t.is_failure),
        root_causes=[
            RootCauseResult(
                trace_id=rc.trace_id, root_step=rc.root_step,
                condition=rc.failure_condition, confidence=rc.confidence,
            )
            for rc in root_causes[:10]
        ],
    )
