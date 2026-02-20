"""Synthetic pipeline traces for testing."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from chains.instrumentation.logger import PipelineStep, PipelineTrace


def build_rag_traces(
    n_traces: int = 100,
    failure_rate: float = 0.15,
    seed: int = 42,
) -> list[PipelineTrace]:
    """
    Generate synthetic RAG pipeline traces with a known failure mode:
    - Long queries (> 200 chars) cause bad retrieval
    - Bad retrieval causes low final quality
    - Summarization and generation work fine regardless

    This is the canonical test scenario from the Chains spec.
    """
    rng = random.Random(seed)
    traces = []

    for i in range(n_traces):
        # Determine if this will be a long query (correlates with failure)
        is_long = rng.random() < failure_rate + 0.05
        query_len = rng.randint(250, 500) if is_long else rng.randint(20, 180)
        query = "x" * query_len

        # Retrieval step
        if is_long:
            retrieval_score = round(rng.uniform(0.2, 0.5), 2)
            retrieval_output = "irrelevant doc " * rng.randint(1, 3)
        else:
            retrieval_score = round(rng.uniform(0.7, 0.95), 2)
            retrieval_output = "relevant document content " * rng.randint(3, 8)

        retrieval_step = PipelineStep(
            name="retrieval",
            step_type="retriever",
            inputs={"query": query},
            outputs={"text": retrieval_output, "score": retrieval_score},
            metadata={"k": 5, "index": "main"},
            latency_ms=rng.uniform(50, 200),
        )

        # Summarization step (always works reasonably)
        summary_output = f"Summary of {len(retrieval_output)} chars"
        summary_step = PipelineStep(
            name="summarization",
            step_type="llm",
            inputs={"text": retrieval_output},
            outputs={"text": summary_output},
            metadata={"model": "llama3.1", "temperature": 0.3},
            latency_ms=rng.uniform(200, 500),
        )

        # Generation step (quality depends on retrieval quality)
        gen_output = f"Answer based on: {summary_output[:50]}"
        gen_step = PipelineStep(
            name="generation",
            step_type="llm",
            inputs={"text": summary_output, "query": query},
            outputs={"text": gen_output},
            metadata={"model": "llama3.1", "temperature": 0.7},
            latency_ms=rng.uniform(300, 800),
        )

        # Quality is primarily driven by retrieval quality
        if is_long:
            quality = round(rng.uniform(0.2, 0.5), 2)
        else:
            quality = round(rng.uniform(0.75, 0.98), 2)

        trace = PipelineTrace(
            trace_id=str(uuid.uuid4())[:8],
            steps=[retrieval_step, summary_step, gen_step],
            final_output=gen_output,
            quality_score=quality,
            timestamp=datetime.now() - timedelta(hours=rng.randint(0, 48)),
            context={"query": query, "session": f"s_{i}"},
        )
        traces.append(trace)

    return traces
