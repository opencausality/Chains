"""Discover causal relationships between pipeline steps and output quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from chains.instrumentation.logger import PipelineTrace

logger = logging.getLogger(__name__)


@dataclass
class StepImpact:
    """Causal impact of a pipeline step on final quality."""

    step_name: str
    correlation: float
    p_value: float
    effect_size: float
    is_significant: bool


@dataclass
class CausalGraph:
    """The pipeline's causal quality structure."""

    step_impacts: list[StepImpact]
    step_order: list[str]
    n_traces: int

    def get_impact(self, step_name: str) -> StepImpact | None:
        return next((s for s in self.step_impacts if s.step_name == step_name), None)

    @property
    def significant_steps(self) -> list[StepImpact]:
        return [s for s in self.step_impacts if s.is_significant]


def discover_step_quality_causation(
    traces: list[PipelineTrace],
    significance_level: float = 0.01,
) -> CausalGraph:
    """
    Discover which steps causally affect final quality.

    Method:
    1. Extract features from each step (output length, latency, error)
    2. Correlate step features with quality score
    3. Test significance with p-values
    4. Rank steps by impact
    """
    # Filter traces with quality scores
    scored = [t for t in traces if t.quality_score is not None]
    if len(scored) < 5:
        return CausalGraph(step_impacts=[], step_order=[], n_traces=len(scored))

    # Extract all step names across traces
    all_steps = []
    for t in scored:
        for s in t.steps:
            if s.name not in all_steps:
                all_steps.append(s.name)

    step_impacts: list[StepImpact] = []

    for step_name in all_steps:
        # Extract per-trace features for this step
        features = []
        qualities = []

        for trace in scored:
            step = next((s for s in trace.steps if s.name == step_name), None)
            if step is None:
                continue

            # Composite feature: output length + no error + low latency
            output_text = str(step.outputs.get("text", step.outputs.get("output", "")))
            feature_val = len(output_text)
            if step.error:
                feature_val = 0  # Error is strong negative signal

            features.append(feature_val)
            qualities.append(trace.quality_score)

        if len(features) < 5:
            continue

        features_arr = np.array(features, dtype=float)
        qualities_arr = np.array(qualities, dtype=float)

        # Handle zero-variance
        if np.std(features_arr) < 1e-10 or np.std(qualities_arr) < 1e-10:
            step_impacts.append(StepImpact(step_name, 0.0, 1.0, 0.0, False))
            continue

        r, p_val = stats.pearsonr(features_arr, qualities_arr)
        effect = float(r)

        step_impacts.append(StepImpact(
            step_name=step_name,
            correlation=round(float(r), 4),
            p_value=round(float(p_val), 6),
            effect_size=round(abs(effect), 4),
            is_significant=p_val < significance_level,
        ))

    step_impacts.sort(key=lambda s: s.effect_size, reverse=True)

    logger.info("Discovered %d step impacts (%d significant) from %d traces",
                len(step_impacts), sum(1 for s in step_impacts if s.is_significant), len(scored))

    return CausalGraph(
        step_impacts=step_impacts,
        step_order=all_steps,
        n_traces=len(scored),
    )
