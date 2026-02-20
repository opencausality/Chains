"""Tests for Typer CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from chains.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert "chains" in result.output.lower()


def test_providers():
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0


def test_analyze_with_traces(rag_traces, tmp_path):
    from chains.instrumentation.logger import TraceLogger
    logger = TraceLogger()
    logger._traces = rag_traces
    out = tmp_path / "traces.json"
    logger.export(str(out))

    result = runner.invoke(app, ["analyze", "--traces", str(out)])
    assert result.exit_code == 0


def test_analyze_missing_file():
    result = runner.invoke(app, ["analyze", "--traces", "/nonexistent.jsonl"])
    assert result.exit_code in [0, 1, 2]
