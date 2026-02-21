<div align="center">

# 🔗 Chains

**Debugging is causal reasoning. Make it systematic.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/your-username/chains/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/chains/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Chains debugs multi-step LLM pipelines causally. Instead of *"the RAG pipeline gives wrong answers sometimes,"* get *"retrieval step causes bad outputs when query length > 200 tokens, not the generation step."*

[The Problem](#-the-core-problem) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Example Output](#-example-output) · [API](#-api)

</div>

---

## 🧠 Philosophy

- 🏠 **Local-first** — Ollama is the default. Your traces never leave your machine.
- 🔍 **Step-level attribution** — Pinpoints which pipeline steps cause failures.
- 📊 **Data-driven** — Uses execution logs and statistics, not speculation.
- 🎯 **Actionable** — Suggests specific fixes, not vague advice.
- 🚫 **No telemetry** — All analysis happens locally.

---

## 📖 The Core Problem

Multi-step LLM pipelines (RAG, agent workflows, prompt chains) fail in opaque ways.

```
Symptom:    "My RAG system gives wrong answers 15% of the time"
Question:   Is it the retrieval, the summarization, or the generation?
Today:      Manual inspection, A/B testing each component, guesswork.
Chains:     Statistical causal attribution → "retrieval fails on long queries"
```

| Without Chains | With Chains |
|---|---|
| "Something's wrong with the pipeline" | "Retrieval step causes 87% of failures" |
| Manually inspect 100 failed traces | Auto-detect failure conditions |
| Guess which step to fix | Get ranked fix suggestions with impact estimates |
| Can't tell root cause from symptom | Causal attribution separates cause from effect |

---

## 🚀 Quick Start

```bash
git clone https://github.com/opencausality/chains.git
cd chains
pip install -e ".[dev]"
cp .env.example .env
```

Analyze pipeline traces:

```bash
chains analyze --traces pipeline_logs.json --top 3
```

---

## 🏗️ How It Works

```
Pipeline Execution Logs   →   Causal Discovery
(step inputs, outputs,        (which steps affect quality?)
 quality scores)                     │
                                     ▼
                            Failure Attribution
                            (for bad outputs, which step caused it?)
                                     │
                                     ▼
                            Condition Detection
                            (WHEN does this step fail?)
                                     │
                                     ▼
                            Fix Suggestions
                            (specific, ranked recommendations)
```

1. **Instrumentation** — `TraceLogger` captures step-level execution: inputs, outputs, latency, errors. JSON export for analysis.

2. **Causal Discovery** — Correlates step output features with final quality scores. Tests significance with p-values to identify which steps actually affect quality.

3. **Failure Attribution** — For each failed trace, walks backwards through the pipeline comparing step outputs to successful baselines. Weights deviations by causal impact (z-score × effect size).

4. **Condition Detection** — Compares features of the root step in failed vs. successful traces. Finds discriminative thresholds like "query_length > 200" or "retrieval_score < 0.7".

5. **Fix Suggestions** — Template-based + condition-specific recommendations ranked by expected impact. Provides code-level hints.

---

## 📊 Example Output

```console
$ chains analyze --traces rag_traces.json

📂 Loading traces from rag_traces.json...
   100 traces (100 scored, 15 failures)

🔬 Discovering step → quality causation...
   2 significant step impacts found

🎯 Attributing failures to root cause steps...

========================================================
=== Chains Debug Report ===
========================================================

Traces analyzed: 100
Failure rate: 15/100 (15%)

Root Cause Breakdown:

  retrieval: 12 failures (80% of all failures)
    Condition: query_length > 200.0 (confidence: 75%)
    Fix: Add input length guard [75% of failures match this condition]
    Fix: Increase retrieval k for failing queries [30-50% reduction]

  generation: 3 failures (20% of all failures)
    Fix: Add grounding constraints to prevent hallucination [25-40% reduction]
    Fix: Lower temperature for failing step [15-30% reduction]
```

---

## 🔬 LLM vs Causal Debugging

| | Manual/LLM Debugging | Chains (Causal) |
|---|---|---|
| **Method** | Read logs, guess root cause | Statistical attribution with evidence |
| **Evidence** | "I think retrieval is the problem" | "z-score=3.2, p<0.001, r=0.72" |
| **Conditions** | "Sometimes it fails" | "Fails when query_length > 200" |
| **Fixes** | "Make retrieval better" | "Increase k to 12 for long queries" |
| **Scalable** | Breaks at 1000+ traces | Designed for large trace sets |

---

## 🌐 API

### CLI Commands

```bash
chains analyze --traces logs.json --top 3
chains providers
chains --version
```

### REST API

```bash
uvicorn chains.api.server:app --reload
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check + version |
| `GET` | `/providers` | Available LLM providers |
| `POST` | `/analyze` | Analyze pipeline traces |

### Python SDK

```python
from chains.instrumentation.logger import TraceLogger
from chains.discovery.causal import discover_step_quality_causation
from chains.attribution.engine import attribute_failures

# 1. Capture traces
logger = TraceLogger()
logger.start_trace(context={"query": user_query})
logger.log_step("retrieval", "retriever", outputs={"text": docs})
logger.log_step("generation", "llm", outputs={"text": answer})
logger.end_trace(quality_score=score)

# 2. Analyze
traces = logger.traces
graph = discover_step_quality_causation(traces)
root_causes = attribute_failures(traces, graph)

for rc in root_causes:
    print(f"{rc.root_step}: {rc.failure_condition} (confidence: {rc.confidence:.0%})")
```

---

## ⚙️ Supported Providers

| Provider | Config | Notes |
|---|---|---|
| `ollama` | Default | Local, private, free |
| `openai` | `OPENAI_API_KEY` | GPT-4o recommended |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet |
| `groq` | `GROQ_API_KEY` | Fast inference |
| `mistral` | `MISTRAL_API_KEY` | Open-weight models |
| `together` | `TOGETHER_API_KEY` | Llama, Mixtral |

---

## 🧪 Testing

23 tests across 6 modules:

| Module | Coverage |
|---|---|
| `test_discovery.py` | Retrieval impact detection, step ordering, significance |
| `test_attribution.py` | Root cause is retrieval (not generation), evidence |
| `test_conditions.py` | Query-length discrimination, fix templates |
| `test_instrumentation.py` | Logger lifecycle, export/load round-trip |
| `test_cli.py` | Version, providers, analyze, error handling |
| `test_api.py` | Health, providers, /analyze endpoint |

```bash
pytest tests/ -v
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*"Debugging is causal reasoning. Make it systematic."*

</div>
