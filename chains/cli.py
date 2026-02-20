"""Chains CLI — Debug multi-step LLM pipelines causally."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from chains import __version__
from chains.config import LLMProvider, configure_logging, get_settings

app = typer.Typer(
    name="chains",
    help="🔗 Chains — Debug multi-step LLM pipelines causally.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"chains {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", callback=version_callback, is_eager=True),
) -> None:
    """Chains — Debugging is causal reasoning. Make it systematic."""


@app.command()
def analyze(
    traces_path: str = typer.Option(..., "--traces", "-t", help="Path to trace file (JSON/JSONL)"),
    top_n: int = typer.Option(3, "--top", "-n", help="Top N root causes to show"),
) -> None:
    """Analyze pipeline traces and find root causes."""
    settings = get_settings()
    configure_logging(settings.log_level)

    console.print(f"\n📂 [bold]Loading traces from[/bold] {traces_path}...")
    try:
        from chains.instrumentation.logger import load_traces
        traces = load_traces(traces_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    n_failed = sum(1 for t in traces if t.is_failure)
    n_scored = sum(1 for t in traces if t.quality_score is not None)
    console.print(f"   {len(traces)} traces ({n_scored} scored, {n_failed} failures)\n")

    # Discovery
    console.print("🔬 [bold]Discovering step → quality causation...[/bold]")
    from chains.discovery.causal import discover_step_quality_causation
    graph = discover_step_quality_causation(traces, settings.significance_level)

    sig = graph.significant_steps
    console.print(f"   {len(sig)} significant step impacts found\n")

    # Attribution
    console.print("🎯 [bold]Attributing failures to root cause steps...[/bold]")
    from chains.attribution.engine import attribute_failures
    root_causes = attribute_failures(traces, graph)

    # Aggregate by step
    step_counts: dict[str, int] = {}
    for rc in root_causes:
        step_counts[rc.root_step] = step_counts.get(rc.root_step, 0) + 1

    # Print report
    console.print("\n" + "=" * 56)
    console.print("[bold]=== Chains Debug Report ===[/bold]")
    console.print("=" * 56 + "\n")

    console.print(f"Traces analyzed: {len(traces)}")
    console.print(f"Failure rate: {n_failed}/{len(traces)} ({n_failed/max(len(traces),1)*100:.0f}%)\n")

    if not root_causes:
        console.print("[green]No failures detected.[/green]")
        return

    console.print("[bold]Root Cause Breakdown:[/bold]\n")
    for step, count in sorted(step_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        pct = count / max(n_failed, 1) * 100
        console.print(f"  [bold red]{step}[/bold red]: {count} failures ({pct:.0f}% of all failures)")

        # Conditions
        from chains.conditions.detector import detect_failure_conditions
        conditions = detect_failure_conditions(step, traces)
        for cond in conditions[:2]:
            console.print(f"    Condition: {cond.condition} (confidence: {cond.confidence:.0%})")

        # Fixes
        from chains.fixes.suggester import suggest_fixes
        rc = next((r for r in root_causes if r.root_step == step), None)
        if rc:
            fixes = suggest_fixes(rc, conditions)
            for fix in fixes[:2]:
                console.print(f"    Fix: {fix.description} [{fix.expected_impact}]")
        console.print()


@app.command()
def providers() -> None:
    """Show supported LLM providers."""
    table = Table(title="Supported LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Config", style="green")
    for p in LLMProvider:
        env = f"{p.value.upper()}_API_KEY" if p != LLMProvider.OLLAMA else "(local, no key)"
        table.add_row(p.value, env)
    console.print(table)
