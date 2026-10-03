"""`Geometry` + `LayoutGraph` → SVG, composed as PEP 750 `Template`s over tdom.

Two jobs, and it is worth being clear that they are two:

- It is the **static export**: a run graph in a report, a snapshot test, a FastAPI page with no
  JavaScript at all. Such a page shows where a run is without a bundler, a Web Worker, or an npm
  asset in the request path.
- It is the **shape the browser renderer must produce too**, so the classes here are the contract.
  A node is a `<g>` carrying `data-node-id`, the *semantic* id, which is what a click reports and
  what survives a live update. Nothing downstream should ever key on position.

**Markup is the fourth t-string grammar**, after SQL (`effective.sql`), prompts
(`effective.channels`) and keys (`compose_key`). The same rule holds in all four: the static spans
are the *delimiters* and the interpolations are *data*, and a processor that can still see which is
which does not need escaping as a patch. An `escape(...)` spelled at each call site is correct by
convention, and one omission is an injection. `svg(t"…")` is correct by construction, and it knows
three things a hand-rolled `escape` cannot, because an f-string has already destroyed the structure
by the time the escape runs:

1. **Position.** A value in `class="{…}"` is attribute data; the same value in `<text>{…}</text>`
   is text data. They need different escapes. The processor knows which from the static span it
   sits in; `escape(value)` cannot know, so it has to be told — by a human, every time.
2. **Content model.** `<style>` is a raw-text element, so CSS must NOT be escaped or a `>` in a
   selector breaks. The old version happened to be right here only because it forgot the escape
   at that one site. Right by luck is the failure mode this removes.
3. **Composition.** A list of *`Template`s* splices; a list of *strings* is escaped as text. So
   building markup by concatenating strings — the Bobby Tables shape in a third grammar — is not
   something you can do by accident. Sub-fragments below are returned as `Template`, never as
   `str`, which is what makes that guarantee reach all the way up.

Measured cost of the seam: ~25 µs per node versus ~0.8 µs for the f-string it replaces, so a
40-node folded run renders in ~1 ms — three orders below the layout round trip that precedes it.
The unrolled 10,000-node stress case is the one where it would show, and that is the case the
folded view exists for.

Following `effective.cards.render_html`: the fragment carries semantic classes and the host page
owns the CSS (`GRAPH_CSS` is a usable default, not a requirement).
"""

from itertools import pairwise
from math import hypot
from string.templatelib import Template

from tdom import Markup
from tdom import svg as render_svg

from effective.graphlayout.model import EdgeSection, Geometry, LayoutGraph, LayoutNode, Rect
from effective.graphview import format_cost

GRAPH_CSS = """
.ev-g {
  --ev-ink: #1e2a2e;
  --ev-ink-muted: #4b5a5f;
  --ev-ink-badge: #3d4c51;
  --ev-line: #7e9095;
  --ev-fill: #f7f9f9;
  --ev-fill-sunken: #e9eeee;
  --ev-green: #5f8a44;
  --ev-parked: #c07408;
  --ev-failed: #96261c;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  .ev-g {
    --ev-ink: #b9c6c2;
    --ev-ink-muted: #9fb0b2;
    --ev-ink-badge: #aebdbd;
    --ev-line: #5d7075;
    --ev-fill: #1e2a2e;
    --ev-fill-sunken: #182226;
    --ev-green: #94b56e;
    --ev-parked: #bf8a22;
    --ev-failed: #d95a6e;
  }
}
.ev-g-node rect, .ev-g-node polygon, .ev-g-node path, .ev-g-node ellipse {
  fill: var(--ev-fill); stroke: var(--ev-line); stroke-width: 1.5;
}
.ev-g-node text { fill: var(--ev-ink); font-size: 13px; }
.ev-g-node .ev-g-detail { fill: var(--ev-ink-muted); font-size: 11px; }
.ev-g-node .ev-g-badge { fill: var(--ev-ink-badge); font-size: 11px; font-weight: 600; }
.ev-g-node--ledger rect { fill: var(--ev-fill-sunken); stroke: var(--ev-ink-muted); }
.ev-g-node--sleep rect { stroke-dasharray: 3 3; }
.ev-g-node--parked :is(rect, polygon, path) { stroke: var(--ev-parked); stroke-width: 3; }
.ev-g-node--parked text { fill: var(--ev-ink); }
.ev-g-node--failed :is(rect, polygon, path) { stroke: var(--ev-failed); stroke-width: 3; }
.ev-g-node--refused :is(rect, polygon, path) {
  stroke: var(--ev-failed); stroke-width: 3; stroke-dasharray: 4 3;
}
.ev-g-node--hypothetical :is(rect, polygon, path) { stroke-dasharray: 5 4; opacity: 0.75; }
.ev-g-edge path { fill: none; stroke: var(--ev-line); stroke-width: 1.5; }
.ev-g-edge--feedback path { stroke: var(--ev-ink-muted); stroke-dasharray: 6 4; }
.ev-g-edge--commit-order path { stroke-dasharray: 2 4; }
.ev-g-edge text { fill: var(--ev-ink-muted); font-size: 10px; }
.ev-g-arrowhead { fill: var(--ev-line); stroke: none; }
.ev-g-arrowhead--feedback { fill: var(--ev-ink-muted); }
""".strip()
"""The default palette — **graphite carries structure, hue is reserved for state.**

Tokens come from the Effective mark itself: the cool charcoal `#24333a`/`#1e2a2e` and the muted
green `#5f8a44` (`#94b56e` on dark). The palette lives here and imports no other brand's values.

**Green is chrome, never a node state**, and that is a measured constraint rather than a taste:
the brand green against a parked amber scores OKLab ΔE 4.3 under deuteranopia and 12.7 under
*normal* vision (dataviz `validate_palette.js`), i.e. two marks a full-color reader already
struggles to separate. A green "committed" node would also paint ~90% of any Effective graph,
since retry, budget, permission and refusal are handler interpretations rather than nodes, and an
accent covering the whole drawing has stopped accenting. Green belongs to
the page frame and the run-level status chip, which is why no rule below uses it.

The state pair that DOES co-occur was chosen against the validator: `#c07408` parked vs `#96261c`
failed clears ΔE 18.4 deutan / 20.1 normal on light, and `#bf8a22` vs `#d95a6e` clears 8.5 / 16.2
on dark — dark being *selected* against the dark surface, not a flipped light ramp. Every ink
step also clears WCAG for its size (the 11px detail line is small text, so it is held to 4.5:1,
not 3:1).

**State is never colour-alone.** A non-committed node carries three redundant signals: its stroke
hue, a 3px weight (dashed for refused), and the state word itself rendered on the detail line by
`_detail`. That triple is the contract — red/amber is precisely the pair that collapses for the
commonest colour-vision deficiency, so the text line is load-bearing, not decoration.

Dark mode overrides **values, not rules**: the custom properties are redeclared on `.ev-g` and
every selector below is written once. A host page can rebrand the same rendered SVG by setting
those properties, which matters because a standalone SVG artifact cannot see a host's theme
toggle. Still a default, not a requirement — every decision stays reachable through `ev-g-*`."""

HEXAGONAL = frozenset({"await"})
"""Kinds drawn as a hexagon rather than a rectangle — the shapes `graphview.to_mermaid` already
uses, kept identical so the two renderers do not teach a reader two vocabularies. A hexagon means
*the run can stop here*, which is the one distinction the interaction surface depends on."""

ARROWS: tuple[tuple[str, str], ...] = (
    ("ev-g-arrow", "ev-g-arrowhead"),
    ("ev-g-arrow-fb", "ev-g-arrowhead ev-g-arrowhead--feedback"),
)
"""`(marker id, class)`. Two arrowheads, because a marker's fill cannot be inherited from the
path's stroke: SVG markers are their own painting context, so an edge color has to be restated
for its head. They carry a CLASS rather than a literal fill so the palette owns both: a custom
property cannot be reached from a presentation attribute (`fill="var(--ev-line)"` is not valid
SVG), so a hard-coded fill here would be the one color a host page could not restyle."""


def _detail(node: LayoutNode) -> str:
    """The node's second line: state when it is not the ordinary one, then cost and duration."""
    parts = [node.state] if node.state != "committed" else []
    # `is not None`, not truthiness — a measured-and-free `0.0` must not render as an unmeasured
    # node. `to_mermaid` carries the argument; this is the second renderer that has to agree.
    if node.cost is not None:
        parts.append(format_cost(node.cost))
    if node.duration_ns is not None:
        parts.append(f"{node.duration_ns / 1e6:.0f}ms")
    return " · ".join(parts)


def _hexagon(box: Rect) -> str:
    """A polygon's `points` attribute — a number list, so an f-string is the right backend here.

    The rule this file follows is a LAYERING rule, not "f-strings bad": the decision about what is
    delimiter and what is data happens in the templates below, and once it is made, formatting
    numbers is exactly what an f-string is for. This value reaches the document through an
    interpolation, so it is escaped as attribute data like any other."""
    notch = min(12.0, box.height / 4)
    points = (
        (box.x + notch, box.y),
        (box.x + box.width - notch, box.y),
        (box.x + box.width, box.y + box.height / 2),
        (box.x + box.width - notch, box.y + box.height),
        (box.x + notch, box.y + box.height),
        (box.x, box.y + box.height / 2),
    )
    return " ".join(f"{x:g},{y:g}" for x, y in points)


def _path(section: EdgeSection) -> str:
    """An SVG path `d` — likewise a number list, formatted by its rendering backend."""
    return " ".join(
        f"{'M' if index == 0 else 'L'} {point.x:g} {point.y:g}"
        for index, point in enumerate(section.points)
    )


def _midpoint(section: EdgeSection) -> tuple[float, float]:
    """Halfway ALONG the polyline, not the middle point of the list.

    An orthogonal route's vertices are not evenly spaced, so `points[len // 2]` can land on the
    arrowhead — which is exactly where the first version put the traversal count."""
    segments = list(pairwise(section.points))
    spans = [hypot(end.x - start.x, end.y - start.y) for start, end in segments]
    target = sum(spans) / 2
    for (start, end), span in zip(segments, spans, strict=True):
        if span >= target:
            ratio = target / span if span else 0.0
            return start.x + (end.x - start.x) * ratio, start.y + (end.y - start.y) * ratio
        target -= span
    return section.points[-1].x, section.points[-1].y


def _parallelogram(box: Rect) -> str:
    """An artifact's `points` — `to_mermaid`'s `[/ /]`, drawn. A number list, so an f-string is
    the right backend (see `_hexagon`)."""
    slant = min(14.0, box.width / 6)
    points = (
        (box.x + slant, box.y),
        (box.x + box.width, box.y),
        (box.x + box.width - slant, box.y + box.height),
        (box.x, box.y + box.height),
    )
    return " ".join(f"{x:g},{y:g}" for x, y in points)


def _shape(node: LayoutNode, box: Rect) -> Template:
    """The node's outline, keyed by KIND — because kind must not be carried by colour.

    Shape is the kind axis and hue is the state axis, kept disjoint so the whole colour budget
    stays available for the thing an operator acts on (`GRAPH_CSS`). The vocabulary is
    `graphview.to_mermaid`'s `_SHAPES`, so the two renderers do not teach a reader two languages:
    hexagon = the run can stop here, parallelogram = an artifact, stadium = a sleep, rect = a step.

    `ledger` is the one kind still drawn as a rect — `to_mermaid` gives it a cylinder (`[( )]`) and
    the honest counterpart here is an arc path this does not yet build; it is distinguished by a
    graphite *tint* (`--ev-fill-sunken`) rather than a hue in the meantime, so the disjointness
    rule holds even though the vocabularies are not yet at parity. That gap is the follow-up."""
    match node.kind:
        case kind if kind in HEXAGONAL:
            return t'<polygon points="{_hexagon(box)}"></polygon>'
        case "artifact":
            return t'<polygon points="{_parallelogram(box)}"></polygon>'
        case "sleep":
            radius = box.height / 2
            return (
                t'<rect x="{box.x:g}" y="{box.y:g}" width="{box.width:g}" '
                t'height="{box.height:g}" rx="{radius:g}"></rect>'
            )
        case _:
            return (
                t'<rect x="{box.x:g}" y="{box.y:g}" width="{box.width:g}" '
                t'height="{box.height:g}" rx="6"></rect>'
            )


def _detail_line(node: LayoutNode, box: Rect) -> Template | str:
    """The state / cost / duration line, or nothing.

    An empty `str` is the honest "no element here": it interpolates as empty text, whereas `None`
    would render as the word None. The optional-fragment idiom throughout this file."""
    if not (detail := _detail(node)):
        return ""
    baseline = box.y + box.height / 2 + 14
    return t'<text class="ev-g-detail" x="{box.x + 14:g}" y="{baseline:g}">{detail}</text>'


def _badge(node: LayoutNode, box: Rect) -> Template | str:
    if node.count <= 1:
        return ""
    baseline = box.y + box.height / 2 + 4
    return (
        t'<text class="ev-g-badge" text-anchor="end" '
        t'x="{box.x + box.width - 12:g}" y="{baseline:g}">x{node.count}</text>'
    )


def _node(node: LayoutNode, box: Rect) -> Template:
    """One box, as a `Template` — never as a string.

    Returning `str` here would let a caller build the document by concatenation, and the splice
    rule (`Template` splices, `str` escapes) would then be a trap rather than a guarantee. A
    fragment that stays a `Template` composes; one that has already been flattened cannot."""
    classes = f"ev-g-node ev-g-node--{node.kind} ev-g-node--{node.state}"
    label_y = box.y + box.height / 2 + (-2 if _detail(node) else 5)
    return t"""<g class="{classes}" data-node-id="{node.id}">
      {t"<title>{node.id}</title>"}
      {_shape(node, box)}
      <text x="{box.x + 14:g}" y="{label_y:g}">{node.label}</text>
      {_detail_line(node, box)}
      {_badge(node, box)}
    </g>"""


def _count_label(sections: tuple[EdgeSection, ...], count: int) -> Template | str:
    if count <= 1 or not sections:
        return ""
    # Offset in BOTH axes rather than centred: a centred label sits on top of a vertical edge's
    # own line, which is most of them once the direction is DOWN.
    x, y = _midpoint(sections[0])
    return t'<text x="{x + 6:g}" y="{y - 4:g}">x{count}</text>'


def _edge(kind: str, edge_id: str, sections: tuple[EdgeSection, ...], count: int) -> Template:
    marker = "ev-g-arrow-fb" if kind == "feedback" else "ev-g-arrow"
    paths = [
        t'<path d="{_path(section)}" marker-end="url(#{marker})"></path>' for section in sections
    ]
    return t"""<g class="ev-g-edge ev-g-edge--{kind}" data-edge-id="{edge_id}">
      {paths}{_count_label(sections, count)}
    </g>"""


def _defs() -> Template:
    markers = [
        t'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        t'markerHeight="6" orient="auto-start-reverse">'
        t'<path class="{css_class}" d="M 0 0 L 10 5 L 0 10 z"></path></marker>'
        for name, css_class in ARROWS
    ]
    return t"<defs>{markers}</defs>"


def to_svg(
    graph: LayoutGraph,
    geometry: Geometry,
    *,
    inline_css: bool = False,
    title: str | None = None,
) -> str:
    """Render a laid-out graph.

    `inline_css=True` produces a standalone file (a report, a snapshot); the default emits a bare
    fragment for a page that already carries `GRAPH_CSS`.

    The CSS rides through `Markup`, which is the one place this file asserts "these bytes are
    already markup". That assertion is safe for a module constant and would not be for anything a
    caller supplied — which is exactly why it has to be spelled, once, in the open."""
    placed = {node.id: node.bounds for node in geometry.nodes}
    routes = {edge.id: edge.sections for edge in geometry.edges}
    frame = geometry.bounds
    width = max(frame.width, 1)
    height = max(frame.height, 1)

    edges = [
        _edge(edge.kind, edge.id, routes[edge.id], edge.count)
        for edge in graph.edges
        if edge.id in routes
    ]
    nodes = [_node(node, placed[node.id]) for node in graph.nodes if node.id in placed]

    return render_svg(t"""<svg class="ev-g" xmlns="http://www.w3.org/2000/svg"
      viewBox="{frame.x:g} {frame.y:g} {width:g} {height:g}"
      width="{width:g}" height="{height:g}" data-graph-id="{graph.id}">
      {t"<title>{title}</title>" if title else ""}
      {t"<style>{Markup(GRAPH_CSS)}</style>" if inline_css else ""}
      {_defs()}
      <g class="ev-g-edges">{edges}</g>
      <g class="ev-g-nodes">{nodes}</g>
    </svg>""")


__all__ = ["ARROWS", "GRAPH_CSS", "HEXAGONAL", "to_svg"]
