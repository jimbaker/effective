"""``CardSpec`` → Markdown. The always-works Tier-1 fallback (no host, no JS).

A chart can't render inline in Markdown, so it degrades to a labelled
placeholder; everything else is faithful. This is the surface an MCP host falls
back to when it can't render an HTML resource or an App iframe.
"""

from effective.cards.spec import CardSpec


def render_markdown(spec: CardSpec) -> str:
    lines: list[str] = []

    title = f"### {spec.title}"
    if spec.badge is not None:
        title += f"  ·  _{spec.badge.text}_"
    lines.append(title)

    if spec.question:
        lines.append(f"\n> {spec.question}")

    if spec.metrics:
        lines.append("\n| Metric | Value |")
        lines.append("|---|---|")
        for cell in spec.metrics:
            lines.append(f"| {cell.label} | {cell.render_value()} |")

    if spec.chart is not None:
        label = spec.chart.title or "chart"
        lines.append(f"\n_[{label} — Vega-Lite chart]_")

    if spec.summary:
        lines.append(f"\n{spec.summary}")

    if spec.actions:
        rendered = ", ".join(f"**{a.label}** (`{a.cmd}` → {a.target})" for a in spec.actions)
        lines.append(f"\n**Actions:** {rendered}")

    for slot in spec.extras:
        lines.append(f"\n_{slot.fallback or 'interactive region — open in app'}_")

    return "\n".join(lines) + "\n"
