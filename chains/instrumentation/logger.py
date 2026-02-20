"""Pipeline execution data model and trace logger."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step execution record."""

    name: str
    step_type: str  # "retriever", "llm", "transformer", "router"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineTrace:
    """One full execution of the pipeline."""

    trace_id: str
    steps: list[PipelineStep]
    final_output: Any = None
    quality_score: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    @property
    def is_failure(self) -> bool:
        if self.quality_score is not None:
            return self.quality_score < 0.5
        return any(s.error is not None for s in self.steps)


class TraceLogger:
    """Collects pipeline execution traces."""

    def __init__(self) -> None:
        self._traces: list[PipelineTrace] = []
        self._current_steps: list[PipelineStep] = []

    @property
    def traces(self) -> list[PipelineTrace]:
        return self._traces

    def start_trace(self, context: dict | None = None) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self._current_steps = []
        self._current_context = context or {}
        self._current_trace_id = trace_id
        return trace_id

    def log_step(
        self,
        name: str,
        step_type: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
        metadata: dict | None = None,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        step = PipelineStep(
            name=name,
            step_type=step_type,
            inputs=inputs or {},
            outputs=outputs or {},
            metadata=metadata or {},
            latency_ms=latency_ms,
            error=error,
        )
        self._current_steps.append(step)

    def end_trace(self, final_output: Any = None, quality_score: float | None = None) -> PipelineTrace:
        trace = PipelineTrace(
            trace_id=getattr(self, "_current_trace_id", str(uuid.uuid4())[:8]),
            steps=list(self._current_steps),
            final_output=final_output,
            quality_score=quality_score,
            context=getattr(self, "_current_context", {}),
        )
        self._traces.append(trace)
        self._current_steps = []
        return trace

    def export(self, path: str) -> None:
        data = []
        for t in self._traces:
            data.append({
                "trace_id": t.trace_id,
                "quality_score": t.quality_score,
                "timestamp": t.timestamp.isoformat(),
                "context": t.context,
                "steps": [
                    {
                        "name": s.name,
                        "step_type": s.step_type,
                        "inputs": _safe_serialize(s.inputs),
                        "outputs": _safe_serialize(s.outputs),
                        "metadata": s.metadata,
                        "latency_ms": s.latency_ms,
                        "error": s.error,
                    }
                    for s in t.steps
                ],
            })
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info("Exported %d traces to %s", len(data), path)


def load_traces(path: str) -> list[PipelineTrace]:
    """Load traces from a JSONL or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    text = p.read_text()
    if text.strip().startswith("["):
        data = json.loads(text)
    else:  # JSONL
        data = [json.loads(line) for line in text.strip().split("\n") if line.strip()]

    traces = []
    for item in data:
        steps = [
            PipelineStep(
                name=s["name"],
                step_type=s.get("step_type", "unknown"),
                inputs=s.get("inputs", {}),
                outputs=s.get("outputs", {}),
                metadata=s.get("metadata", {}),
                latency_ms=s.get("latency_ms", 0),
                error=s.get("error"),
            )
            for s in item.get("steps", [])
        ]
        ts = datetime.fromisoformat(item["timestamp"]) if "timestamp" in item else datetime.now()
        traces.append(PipelineTrace(
            trace_id=item.get("trace_id", ""),
            steps=steps,
            quality_score=item.get("quality_score"),
            timestamp=ts,
            context=item.get("context", {}),
        ))
    return traces


def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)
