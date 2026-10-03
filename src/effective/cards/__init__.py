"""The view axis: a card is a typed ``CardSpec`` rendered to many targets.

`effective.channels` reifies the **data axis** (a prompt is a `t"…"` whose holes
are typed I/O channels). `effective.cards` is the **view axis**: a card is a
typed `CardSpec` whose pieces are a small wrapper vocabulary (`Metric`,
`Interval`, `Badge`, `Action`, `VegaChart`). Pure renderers turn one spec into
many surfaces: a self-contained HTML fragment (Shiny embeds it via `ui.HTML`,
an MCP-App resource serves it directly), a Shiny tag tree, and a Markdown fallback.
`manifest` exposes the inspectable surface for agent-customization safety.

A card is authored two ways that produce the *same* `CardSpec`: declaratively, by
building the spec, or as a PEP 750 template, `card(t"…")`.

Domain-neutral and dep-light by design: charts ride in as Vega-Lite `dict`s
(the domain builds them with altair and passes `.to_dict()`), so this package
imports neither altair nor shiny. A domain imports this package, never the
reverse.
"""

from effective.cards.manifest import manifest
from effective.cards.render_html import VEGA_BOOTSTRAP, render_html
from effective.cards.render_markdown import render_markdown
from effective.cards.render_shiny import action_id, render_shiny
from effective.cards.spec import (
    Action,
    Badge,
    CardSpec,
    Cell,
    Interval,
    Metric,
    Slot,
    VegaChart,
    format_value,
)
from effective.cards.tstring import CardId, Question, Summary, Title, card

__all__ = [
    "VEGA_BOOTSTRAP",
    "Action",
    "Badge",
    "CardId",
    "CardSpec",
    "Cell",
    "Interval",
    "Metric",
    "Question",
    "Slot",
    "Summary",
    "Title",
    "VegaChart",
    "action_id",
    "card",
    "format_value",
    "manifest",
    "render_html",
    "render_markdown",
    "render_shiny",
]
