"""PyVis interactive visualization of pipeline causal flow."""

from __future__ import annotations

import logging
from pathlib import Path

from pyvis.network import Network

from chains.discovery.causal import CausalGraph

logger = logging.getLogger(__name__)


def render_pipeline_flow(
    graph: CausalGraph,
    output_path: str = "chains_flow.html",
) -> Path:
    """Render the pipeline's causal flow as an interactive graph."""
    net = Network(height="700px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Add quality target node
    net.add_node("quality", label="Quality Score", size=35, color="#e94560", title="⭐ Final quality score")

    for impact in graph.step_impacts:
        if impact.is_significant:
            color = "#ff6b6b" if impact.correlation < 0 else "#51cf66"
            size = 25
        else:
            color = "#0f3460"
            size = 18

        title = (f"Step: {impact.step_name}\n"
                 f"Correlation: {impact.correlation:.3f}\n"
                 f"p-value: {impact.p_value:.4f}\n"
                 f"Significant: {'✓' if impact.is_significant else '✗'}")
        net.add_node(impact.step_name, label=impact.step_name, size=size, color=color, title=title)

        width = max(1, abs(impact.effect_size) * 8)
        edge_color = "#ff6b6b" if impact.correlation < 0 else "#51cf66"
        net.add_edge(impact.step_name, "quality", width=width, color=edge_color,
                     title=f"effect: {impact.effect_size:.3f}")

    # Add step order edges
    for i in range(len(graph.step_order) - 1):
        s1, s2 = graph.step_order[i], graph.step_order[i + 1]
        if s1 in [n["id"] for n in net.nodes] and s2 in [n["id"] for n in net.nodes]:
            net.add_edge(s1, s2, width=1, color="#16213e", dashes=True, title="execution order")

    out = Path(output_path)
    net.save_graph(str(out))
    logger.info("Pipeline flow saved to %s", out)
    return out
