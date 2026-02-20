"""Attribute pipeline failures to root cause steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chains.instrumentation.logger import PipelineTrace, PipelineStep
from chains.discovery.causal import CausalGraph

logger = logging.getLogger(__name__)


@dataclass
class RootCause:
    """Attribution of a failure to a specific pipeline step."""

    trace_id: str
    root_step: str
    failure_condition: str
    confidence: float
    evidence: dict = field(default_factory=dict)
    suggested_fix: str = ""


def attribute_failure(
    failed_trace: PipelineTrace,
    causal_graph: CausalGraph,
    all_traces: list[PipelineTrace],
) -> RootCause:
    """
    For a trace with low quality, identify which step(s) caused the failure.

    Algorithm:
    1. Walk backwards through the pipeline steps
    2. Compare each step's output to successful trace outputs
    3. Use causal graph to weight which deviations caused the quality drop
    4. Return root cause with evidence
    """
    successful = [t for t in all_traces if t.quality_score is not None and t.quality_score >= 0.7]

    if not successful:
        return RootCause(
            trace_id=failed_trace.trace_id,
            root_step=failed_trace.steps[0].name if failed_trace.steps else "unknown",
            failure_condition="No successful traces for comparison",
            confidence=0.0,
        )

    # Score each step by how much it deviated from successful traces
    step_scores: list[tuple[str, float, dict]] = []

    for step in reversed(failed_trace.steps):
        # Find matching steps in successful traces
        successful_outputs = []
        for st in successful:
            matching = next((s for s in st.steps if s.name == step.name), None)
            if matching:
                out_text = str(matching.outputs.get("text", matching.outputs.get("output", "")))
                successful_outputs.append(len(out_text))

        if not successful_outputs:
            continue

        # Compare failed step output to successful distribution
        failed_output = str(step.outputs.get("text", step.outputs.get("output", "")))
        failed_len = len(failed_output)
        mean_success = np.mean(successful_outputs)
        std_success = max(np.std(successful_outputs), 1)

        # Z-score: how far is this step's output from the successful mean?
        z_score = abs(failed_len - mean_success) / std_success

        # Check for errors
        if step.error:
            z_score += 5.0  # Strong signal

        # Weight by causal impact
        impact = causal_graph.get_impact(step.name)
        causal_weight = impact.effect_size if impact else 0.1

        deviation_score = z_score * causal_weight

        evidence = {
            "failed_output_length": failed_len,
            "mean_successful_length": round(float(mean_success), 1),
            "z_score": round(z_score, 2),
            "causal_weight": round(causal_weight, 3),
            "has_error": step.error is not None,
        }
        step_scores.append((step.name, deviation_score, evidence))

    if not step_scores:
        return RootCause(
            trace_id=failed_trace.trace_id,
            root_step="unknown",
            failure_condition="Could not determine failure source",
            confidence=0.0,
        )

    # The step with highest deviation × causal weight is the root cause
    step_scores.sort(key=lambda x: x[1], reverse=True)
    root_name, root_score, root_evidence = step_scores[0]

    # Confidence based on deviation magnitude
    confidence = min(root_score / 5.0, 1.0)

    # Build failure condition description
    if root_evidence.get("has_error"):
        condition = f"Step '{root_name}' threw an error"
    elif root_evidence.get("z_score", 0) > 2:
        condition = f"Step '{root_name}' output deviated significantly from successful traces (z={root_evidence['z_score']:.1f})"
    else:
        condition = f"Step '{root_name}' contributed to quality degradation"

    return RootCause(
        trace_id=failed_trace.trace_id,
        root_step=root_name,
        failure_condition=condition,
        confidence=round(confidence, 3),
        evidence=root_evidence,
    )


def attribute_failures(
    traces: list[PipelineTrace],
    causal_graph: CausalGraph,
) -> list[RootCause]:
    """Attribute all failures in the trace set."""
    failed = [t for t in traces if t.is_failure]
    results = []
    for trace in failed:
        result = attribute_failure(trace, causal_graph, traces)
        results.append(result)
    logger.info("Attributed %d failures", len(results))
    return results
