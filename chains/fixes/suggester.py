"""Suggest targeted fixes for root cause pipeline steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chains.attribution.engine import RootCause
from chains.conditions.detector import FailureCondition

logger = logging.getLogger(__name__)


@dataclass
class Fix:
    """A recommended fix for a pipeline failure."""

    step_name: str
    category: str  # "prompt", "parameter", "model", "preprocessing", "architecture"
    description: str
    expected_impact: str
    code_hint: str


# Heuristic fix templates based on step type + condition patterns
_FIX_TEMPLATES = {
    "retriever": [
        Fix("", "parameter", "Increase retrieval k for failing queries",
            "30-50% reduction in retrieval failures",
            "retriever.k = 12 if condition else 5"),
        Fix("", "preprocessing", "Chunk/decompose long queries before retrieval",
            "40-60% reduction for long-query failures",
            "Add QueryDecomposer step before retrieval"),
        Fix("", "architecture", "Use hybrid search (semantic + keyword)",
            "20-30% improvement in edge cases",
            'retriever.search_type = "hybrid"'),
    ],
    "llm": [
        Fix("", "prompt", "Add grounding constraints to prevent hallucination",
            "25-40% reduction in hallucination",
            'prompt += "Only use facts from the provided context."'),
        Fix("", "parameter", "Lower temperature for failing step",
            "15-30% reduction in inconsistency",
            "temperature = 0.3"),
        Fix("", "model", "Use a stronger model for this step",
            "20-40% quality improvement",
            'model = "gpt-4o" for this step'),
    ],
    "transformer": [
        Fix("", "preprocessing", "Add input validation before transformation",
            "10-20% reduction in format errors",
            "validate_schema(input) before transform"),
        Fix("", "postprocessing", "Add output validation after transformation",
            "15-25% bad output filtering",
            "validate_output(result) with fallback"),
    ],
}


def suggest_fixes(
    root_cause: RootCause,
    conditions: list[FailureCondition],
    step_type: str = "llm",
) -> list[Fix]:
    """Generate targeted fix suggestions based on root cause and conditions."""
    templates = _FIX_TEMPLATES.get(step_type, _FIX_TEMPLATES["llm"])

    fixes = []
    for template in templates:
        fix = Fix(
            step_name=root_cause.root_step,
            category=template.category,
            description=template.description,
            expected_impact=template.expected_impact,
            code_hint=template.code_hint,
        )
        fixes.append(fix)

    # Add condition-specific fixes
    for cond in conditions[:2]:
        if "length" in cond.condition.lower() and ">" in cond.condition:
            fixes.insert(0, Fix(
                step_name=root_cause.root_step,
                category="preprocessing",
                description=f"Add input length guard: when {cond.condition}, use alternative strategy",
                expected_impact=f"{int(cond.confidence * 100)}% of failures match this condition",
                code_hint=f"if {cond.condition}: use_fallback_strategy()",
            ))

    logger.info("Generated %d fix suggestions for step '%s'", len(fixes), root_cause.root_step)
    return fixes
