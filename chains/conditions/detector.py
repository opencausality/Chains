"""Detect conditions under which pipeline steps fail."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from chains.instrumentation.logger import PipelineTrace, PipelineStep

logger = logging.getLogger(__name__)


@dataclass
class FailureCondition:
    """A detected condition under which a step tends to fail."""

    step_name: str
    condition: str
    affected_traces: int
    total_failed: int
    confidence: float


def detect_failure_conditions(
    root_step: str,
    traces: list[PipelineTrace],
) -> list[FailureCondition]:
    """
    Identify conditions under which the root step fails.

    Compares features of the root step in failed vs. successful traces
    to find discriminative patterns.
    """
    failed_features: list[dict] = []
    success_features: list[dict] = []

    for trace in traces:
        step = next((s for s in trace.steps if s.name == root_step), None)
        if step is None:
            continue

        features = _extract_features(step, trace)
        if trace.is_failure:
            failed_features.append(features)
        else:
            success_features.append(features)

    if not failed_features or not success_features:
        return []

    conditions: list[FailureCondition] = []

    # Check each numeric feature for threshold-based conditions
    all_feature_keys = set()
    for f in failed_features + success_features:
        all_feature_keys.update(f.keys())

    for key in all_feature_keys:
        failed_vals = [f[key] for f in failed_features if key in f and isinstance(f[key], (int, float))]
        success_vals = [f[key] for f in success_features if key in f and isinstance(f[key], (int, float))]

        if len(failed_vals) < 3 or len(success_vals) < 3:
            continue

        failed_mean = np.mean(failed_vals)
        success_mean = np.mean(success_vals)

        if abs(failed_mean - success_mean) < 0.01:
            continue

        # Find a discriminative threshold
        if failed_mean > success_mean:
            threshold = (failed_mean + success_mean) / 2
            affected = sum(1 for v in failed_vals if v > threshold)
            condition = f"{key} > {threshold:.1f}"
        else:
            threshold = (failed_mean + success_mean) / 2
            affected = sum(1 for v in failed_vals if v < threshold)
            condition = f"{key} < {threshold:.1f}"

        confidence = affected / max(len(failed_vals), 1)

        if confidence >= 0.5:
            conditions.append(FailureCondition(
                step_name=root_step,
                condition=condition,
                affected_traces=affected,
                total_failed=len(failed_vals),
                confidence=round(confidence, 3),
            ))

    conditions.sort(key=lambda c: c.confidence, reverse=True)
    logger.info("Found %d failure conditions for step '%s'", len(conditions), root_step)
    return conditions


def _extract_features(step: PipelineStep, trace: PipelineTrace) -> dict:
    """Extract numeric features from a step execution."""
    features: dict = {}

    # Input features
    input_text = str(step.inputs.get("query", step.inputs.get("text", step.inputs.get("input", ""))))
    features["input_length"] = len(input_text)

    # Output features
    output_text = str(step.outputs.get("text", step.outputs.get("output", "")))
    features["output_length"] = len(output_text)

    # Metadata features
    features["latency_ms"] = step.latency_ms

    # Context features
    query = str(trace.context.get("query", trace.context.get("input", "")))
    features["query_length"] = len(query)

    return features
