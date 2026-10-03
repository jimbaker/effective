"""``CardSpec`` → a self-contained HTML fragment.

The chart is emitted as a ``data-vega-spec`` div that a *one-time* page bootstrap
(`VEGA_BOOTSTRAP`) hydrates with vegaEmbed. It needs no Shiny id-wiring, so the
*same* fragment serves a Shiny app (`ui.HTML(...)`)
and an MCP-App resource. The fragment carries semantic classes (``ev-…``); the
host page owns the CSS.
"""

import json
from string.templatelib import Template

from tdom import html

from effective.cards.spec import Action, Badge, CardSpec, Cell, VegaChart

# Drop this once into the host page (after vega / vega-lite / vega-embed). It
# hydrates every chart div the renderer emits — the card fragment stays inert.
VEGA_BOOTSTRAP = """
<script>
(function () {
  function hydrate(el) {
    if (el.dataset.vegaHydrated) return;
    el.dataset.vegaHydrated = "1";
    try { window.vegaEmbed(el, JSON.parse(el.dataset.vegaSpec), {actions: false}); }
    catch (e) { el.textContent = "chart unavailable"; }
  }
  function run() { document.querySelectorAll(".ev-chart[data-vega-spec]").forEach(hydrate); }
  if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);
})();
</script>
""".strip()


def _badge(badge: Badge) -> Template:
    return t'<span class="ev-badge ev-badge--{badge.tone}">{badge.text}</span>'


def _cell(cell: Cell) -> Template:
    return t"""<div class="ev-metric ev-metric--{cell.kind}">\
<dt>{cell.label}</dt><dd>{cell.render_value()}</dd></div>"""


def _chart(chart: VegaChart) -> Template:
    """The spec rides an ATTRIBUTE, so the quoting is the template's job and not a call's.

    `json.dumps` produces the bytes; where they land decides how they are escaped, and only the
    processor knows where they land. The previous version passed `quote=True` by hand — correct,
    and correct by the author remembering a keyword argument."""
    spec = json.dumps(chart.spec, separators=(",", ":"))
    title = t'<div class="ev-chart__title">{chart.title}</div>' if chart.title else ""
    return t'{title}<div class="ev-chart" data-vega-spec="{spec}"></div>'


def _action(action: Action) -> Template:
    return t"""<button type="button" class="ev-action ev-action--{action.risk}" \
data-cmd="{action.cmd}" data-target="{action.target}">{action.label}</button>"""


def render_html(spec: CardSpec) -> str:
    """Render ``spec`` to a self-contained HTML fragment string.

    Composed as nested `Template`s over tdom rather than concatenated strings, so escaping is a
    property of where a value LANDS instead of a call the author has to remember. Every helper
    above returns a `Template`; `tdom.html` splices one, a list of them, or a list of lists, and
    escapes every non-`Template` value it reaches. `str` out because the contract is a fragment
    — `ui.HTML(...)` on the Shiny side, an MCP-App resource on the other.
    """
    header: list[Template] = [t'<h3 class="ev-card__title">{spec.title}</h3>']
    if spec.badge is not None:
        header.append(_badge(spec.badge))

    body: list[Template] = [t'<header class="ev-card__header">{header}</header>']
    if spec.question:
        body.append(t'<p class="ev-card__question">{spec.question}</p>')
    if spec.metrics:
        cells = [_cell(cell) for cell in spec.metrics]
        body.append(t'<dl class="ev-metrics">{cells}</dl>')
    if spec.chart is not None:
        body.append(_chart(spec.chart))
    if spec.summary:
        body.append(t'<p class="ev-card__summary">{spec.summary}</p>')
    if spec.actions:
        buttons = [_action(action) for action in spec.actions]
        body.append(t'<div class="ev-actions">{buttons}</div>')
    for slot in spec.extras:  # the escape hatch degrades to its fallback off-Shiny
        text = slot.fallback or "interactive region — open in app"
        body.append(t'<div class="ev-slot ev-slot--fallback">{text}</div>')

    return str(html(t'<article class="ev-card" data-card-id="{spec.card_id}">{body}</article>'))
