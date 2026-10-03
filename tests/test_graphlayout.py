"""The layout seam (`effective.graphlayout`).

Two halves, deliberately separated by what they need:

| half   | where                              | runs                                          |
|--------|------------------------------------|-----------------------------------------------|
| pure   | here: op keys to ELK JSON, and SVG | everywhere; `prepare` is a pure function of a |
|        | given a geometry                   | `RunGraph`, itself pure over recorded keys    |
| engine | `tests/test_graphlayout_elkjs.py`  | in the pinned container; skips without it     |

`test_dropping_feedback_edges_leaves_a_dag` is the layout thesis's falsifiable claim. If Python
has resolved the cycle question before an engine sees the graph, deleting the edges Python marked
`feedback` must leave a DAG. The checker re-derives that with Kahn's algorithm, sharing none of
the classifier's reasoning.
"""

import json
from xml.etree import ElementTree

import pytest

from effective.graphlayout import (
    LayoutError,
    canonical_json,
    check_geometry,
    check_graph,
    elide,
    envelope,
    forward_is_acyclic,
    from_elk,
    overlaps,
    prepare,
    to_elk,
    to_svg,
)
from effective.graphlayout.model import (
    EdgeGeometry,
    EdgeSection,
    Geometry,
    LayoutEdge,
    LayoutGraph,
    LayoutNode,
    NodeGeometry,
    Point,
    Rect,
    Size,
)
from effective.graphlayout.svg import GRAPH_CSS
from effective.graphview import fold_cycles, from_keys
from effective.keys import Index, Key

LINEAR = ["extract", "ask", "ledger;r1:request_processed"]
LOOP = ["plan", "ask", "act", "ask#2", "act#2", "ask#3", "ledger;r1:done"]
GATHER = [
    "gather:0,0;fetch",
    "gather:0,1;fetch",
    "gather:0,0;ledger;r1:a",
    "gather:0,1;ledger;r1:b",
    "summarize",
]


# --- the semantics Python resolves before the engine sees anything --------------------------


def test_a_linear_run_is_all_forward_edges():
    graph = prepare(from_keys("r1", LINEAR))
    assert [node.id for node in graph.nodes] == LINEAR
    assert [node.order for node in graph.nodes] == [0, 1, 2]
    assert {edge.kind for edge in graph.edges} == {"forward"}
    assert forward_is_acyclic(graph)


def test_the_folded_loop_marks_exactly_the_back_edge_as_feedback():
    """The one thing ELK could not have told us. `act -> ask` closes the loop the fold recovered;
    every other edge runs forward in tape order."""
    graph = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    kinds = {(edge.source, edge.target): edge.kind for edge in graph.edges}
    assert kinds == {
        ("plan", "ask"): "forward",
        ("ask", "act"): "forward",
        ("act", "ask"): "feedback",
        ("ask", "ledger;r1:done"): "forward",
    }
    assert graph.cyclic is True


def test_dropping_feedback_edges_leaves_a_dag():
    """THE claim of the semantic layer, stated so it can fail.

    Checked on a genuinely cyclic graph (the folded loop) and on the unrolled one, which has no
    feedback edges at all: an unrolled trace is a chain of occurrences, so it cannot cycle."""
    folded = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    assert any(edge.kind == "feedback" for edge in folded.edges)
    assert forward_is_acyclic(folded)

    unrolled = prepare(from_keys("r1", LOOP))
    assert not any(edge.kind == "feedback" for edge in unrolled.edges)
    assert forward_is_acyclic(unrolled)


def test_a_cross_branch_edge_is_commit_order_not_a_causal_edge():
    """Adjacency across two gather branches is commit order; neither branch caused the other. A
    renderer must be able to tell commit order from causation, so the distinction lives in the
    classification, where every renderer reads it."""
    graph = prepare(from_keys("r1", GATHER))
    kinds = {(edge.source, edge.target): edge.kind for edge in graph.edges}
    assert kinds[("gather:0,0;fetch", "gather:0,1;fetch")] == "commit-order"
    assert kinds[("gather:0,1;fetch", "gather:0,0;ledger;r1:a")] == "commit-order"
    # ... and leaving the region entirely is a join.
    assert kinds[("gather:0,1;ledger;r1:b", "summarize")] == "join"


def test_entering_a_branch_is_a_gather_edge():
    graph = prepare(from_keys("r1", ["start", "gather:0,0;fetch", "done"]))
    kinds = {(edge.source, edge.target): edge.kind for edge in graph.edges}
    assert kinds[("start", "gather:0,0;fetch")] == "gather"
    assert kinds[("gather:0,0;fetch", "done")] == "join"


def test_the_fold_drops_the_branch_coordinate_so_commit_order_edges_do_not_survive_it():
    """A limitation worth pinning rather than discovering. `fold_cycles` drops the branch axis, so
    a folded view cannot distinguish commit order from program order, so the classification
    degrades to forward/feedback. The unrolled view is where concurrency structure is legible."""
    graph = prepare(fold_cycles(from_keys("r1", GATHER), drop=(Index,)))
    assert {edge.kind for edge in graph.edges} <= {"forward", "feedback"}


def test_node_kind_and_state_come_through_unchanged():
    """This layer is not a second authority on either vocabulary."""
    graph = prepare(from_keys("r1", LINEAR, states={"ask": "parked"}))
    assert [node.kind for node in graph.nodes] == ["step", "step", "ledger"]
    assert graph.node("ask").state == "parked"


# --- sizing: the cheap half of layout stability ---------------------------------------------


def test_a_count_badge_does_not_resize_a_node():
    """x99 -> x100 must not reflow the drawing. The badge is a fixed width, so
    node size depends on the count only through `count > 1`."""
    small = prepare(fold_cycles(from_keys("r1", ["ask", "act"] * 50), drop=(Index,)))
    large = prepare(fold_cycles(from_keys("r1", ["ask", "act"] * 500), drop=(Index,)))
    assert {node.id: node.count for node in small.nodes} == {"ask": 50, "act": 50}
    assert {node.id: node.count for node in large.nodes} == {"ask": 500, "act": 500}
    assert [node.size for node in small.nodes] == [node.size for node in large.nodes]


def test_a_long_key_is_elided_in_the_middle_and_kept_whole_as_the_id():
    key = "gather:0,1;ledger;r1:request_processed:2026-07-25T12:00:00Z:msg-abcdef"
    label = elide(key)
    assert len(label) <= 42
    assert label.startswith("gather:0,1;")  # the namespace half
    assert label.endswith("msg-abcdef")  # the author's own half
    assert "…" in label

    graph = prepare(from_keys("r1", [key]))
    assert graph.nodes[0].id == key  # identity is never elided
    assert graph.nodes[0].label == label


def test_every_node_gets_a_positive_finite_box():
    for trace in (LINEAR, LOOP, GATHER):
        check_graph(prepare(from_keys("r1", trace)))
        check_graph(prepare(fold_cycles(from_keys("r1", trace), drop=(Index,))))


# --- the wire format --------------------------------------------------------------------------


def test_elk_json_carries_geometry_input_only():
    """No `effective.*` keys ride inside the ELK graph: an engine is not asked to preserve fields
    it does not understand."""
    elk = to_elk(prepare(from_keys("r1", LINEAR)))
    assert elk["id"] == "r1"
    assert elk["layoutOptions"]["elk.algorithm"] == "layered"
    assert elk["layoutOptions"]["elk.direction"] == "RIGHT"
    assert [child["id"] for child in elk["children"]] == LINEAR
    assert all(set(child) == {"id", "width", "height"} for child in elk["children"])
    assert all(set(edge) == {"id", "sources", "targets"} for edge in elk["edges"])


def test_children_are_serialized_in_tape_order():
    """Model order is how the drawing ends up reading in the order the run happened — and it is
    carried by child order, so the serialization order is load-bearing, not cosmetic."""
    elk = to_elk(prepare(from_keys("r1", LOOP)))
    assert [child["id"] for child in elk["children"]] == LOOP


def test_the_envelope_keeps_semantics_beside_the_graph_joined_by_id():
    payload = envelope(prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,))))
    assert payload["schema_version"] == 1
    assert set(payload["metadata"]["nodes"]) == {
        child["id"] for child in payload["graph"]["children"]
    }
    assert set(payload["metadata"]["edges"]) == {edge["id"] for edge in payload["graph"]["edges"]}
    assert payload["metadata"]["nodes"]["ask"]["count"] == 3
    assert payload["metadata"]["cyclic"] is True


def test_the_envelope_ships_a_member_count_not_the_members():
    """A folded node standing for a hundred executions is one box and one number; the hundred
    keys stay server-side."""
    payload = envelope(prepare(fold_cycles(from_keys("r1", ["ask", "act"] * 50), drop=(Index,))))
    assert payload["metadata"]["nodes"]["ask"]["member_count"] == 50
    assert "members" not in payload["metadata"]["nodes"]["ask"]
    assert "ask#37" not in json.dumps(payload)


def test_the_encoding_is_byte_stable():
    first = canonical_json(envelope(prepare(from_keys("r1", LOOP))))
    second = canonical_json(envelope(prepare(from_keys("r1", LOOP))))
    assert first == second


def test_edge_ids_are_unique_and_composed_from_ordinals():
    """Not from the two keys: joining author-controlled strings with a separator is the
    adjacent-interpolation injectivity hole. Integers cannot carry a delimiter."""
    graph = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    ids = [edge.id for edge in graph.edges]
    assert len(set(ids)) == len(ids)
    assert all(
        edge.id == f"e{graph.node(edge.source).order}-{graph.node(edge.target).order}"
        for edge in graph.edges
    )


def test_a_key_containing_the_edge_separator_cannot_forge_another_edges_id():
    forged = prepare(from_keys("r1", ["e0-1", "x", "e0-1->x"]))
    check_graph(forged)  # no duplicate edge ids, however the keys are spelled


# --- validation -------------------------------------------------------------------------------


def test_a_dangling_edge_is_refused_here_rather_than_by_the_engine():
    graph = LayoutGraph(
        id="r1",
        nodes=(LayoutNode(id="a", label="a", size=Size(80, 40), kind="step"),),
        edges=(),
    )
    check_graph(graph)
    broken = LayoutGraph(
        id="r1", nodes=graph.nodes, edges=(LayoutEdge(id="e", source="a", target="ghost"),)
    )
    with pytest.raises(LayoutError, match="unknown node 'ghost'"):
        check_graph(broken)


def test_a_zero_width_node_is_refused():
    graph = LayoutGraph(
        id="r1", nodes=(LayoutNode(id="a", label="a", size=Size(0, 40), kind="step"),)
    )
    with pytest.raises(LayoutError, match="non-positive size"):
        check_graph(graph)


def test_a_node_escaping_the_reported_bounds_is_refused():
    graph = prepare(from_keys("r1", ["a"]))
    geometry = Geometry(
        graph_id="r1",
        bounds=Rect(0, 0, 100, 100),
        nodes=(NodeGeometry(id="a", bounds=Rect(90, 0, 200, 40)),),
    )
    with pytest.raises(LayoutError, match="escapes the graph bounds"):
        check_geometry(graph, geometry)


def test_an_unplaced_node_is_refused():
    graph = prepare(from_keys("r1", ["a", "b"]))
    geometry = Geometry(
        graph_id="r1",
        bounds=Rect(0, 0, 500, 100),
        nodes=(NodeGeometry(id="a", bounds=Rect(0, 0, 100, 40)),),
    )
    with pytest.raises(LayoutError, match="node 'b' has no geometry"):
        check_geometry(graph, geometry)


def test_overlap_detection_finds_stacked_boxes():
    geometry = Geometry(
        graph_id="r1",
        bounds=Rect(0, 0, 200, 200),
        nodes=(
            NodeGeometry(id="a", bounds=Rect(0, 0, 100, 40)),
            NodeGeometry(id="b", bounds=Rect(50, 10, 100, 40)),
            NodeGeometry(id="c", bounds=Rect(0, 100, 100, 40)),
        ),
    )
    assert overlaps(geometry) == (("a", "b"),)


# --- normalizing an engine response ------------------------------------------------------------


ELK_RESPONSE = {
    "id": "r1",
    "x": 0,
    "y": 0,
    "width": 608,
    "height": 99,
    "$H": 13,  # GWT bookkeeping — not part of the contract, and it must not leak through
    "children": [
        {"id": "extract", "width": 160, "height": 56, "x": 16, "y": 25.333333333333336, "$H": 16},
        {"id": "ask", "width": 160, "height": 56, "x": 224, "y": 16, "$H": 18},
    ],
    "edges": [
        {
            "id": "e0-1",
            "sources": ["extract"],
            "targets": ["ask"],
            "sections": [
                {
                    "id": "e0-1_s0",
                    "startPoint": {"x": 176, "y": 44},
                    "bendPoints": [{"x": 200, "y": 44}],
                    "endPoint": {"x": 224, "y": 44},
                    "incomingShape": "extract",
                }
            ],
        }
    ],
}


def test_normalizing_an_elk_response_keeps_only_the_contract():
    geometry = from_elk(ELK_RESPONSE, engine_version="0.12.0")
    assert geometry.graph_id == "r1"
    assert geometry.bounds == Rect(0.0, 0.0, 608.0, 99.0)
    assert geometry.node("ask").bounds == Rect(224.0, 16.0, 160.0, 56.0)
    assert geometry.engine_version == "0.12.0"
    section = geometry.edges[0].sections[0]
    assert section.points == (Point(176, 44), Point(200, 44), Point(224, 44))


def test_an_edge_with_no_route_normalizes_to_no_sections():
    geometry = from_elk({"id": "r1", "edges": [{"id": "e", "sources": ["a"], "targets": ["b"]}]})
    assert geometry.edges[0].sections == ()


# --- SVG ----------------------------------------------------------------------------------------


def _fake_geometry(graph: LayoutGraph) -> Geometry:
    """A grid placement — enough to render without an engine, so SVG is testable everywhere."""
    return Geometry(
        graph_id=graph.id,
        bounds=Rect(0, 0, 400 * max(len(graph.nodes), 1), 200),
        nodes=tuple(
            NodeGeometry(
                id=node.id, bounds=Rect(index * 400, 20, node.size.width, node.size.height)
            )
            for index, node in enumerate(graph.nodes)
        ),
        edges=tuple(
            EdgeGeometry(
                id=edge.id,
                sections=(EdgeSection(start=Point(0, 40), bends=(), end=Point(100, 40)),),
            )
            for edge in graph.edges
        ),
    )


def test_the_projection_sigil_survives_the_svg_render():
    """`*` marks a coordinate a view dropped, so it reaches a label wherever a fold does. It lands
    in element text, in a `<title>`, and in a `data-node-id` attribute, and XML takes it verbatim
    in all three. Pinned because a renderer gaining an escape table is where a quotient would
    quietly become a literal asterisk in somebody's key."""
    folded = fold_cycles(
        from_keys("r1", ["gather:0,0;step;tool:work", "gather:0,1;step;tool:work"]), drop=(Index,)
    )
    graph = prepare(folded)
    svg = to_svg(graph, _fake_geometry(graph))
    ElementTree.fromstring(svg)  # the property: still well-formed XML with the sigil in it
    assert 'data-node-id="gather:*,*;step;tool:work"' in svg
    assert "<title>gather:*,*;step;tool:work</title>" in svg


def test_svg_carries_the_semantic_node_id_and_the_kind_class():
    graph = prepare(from_keys("r1", LINEAR, states={"ask": "parked"}))
    svg = to_svg(graph, _fake_geometry(graph))
    assert 'data-node-id="ledger;r1:request_processed"' in svg
    assert "ev-g-node--ledger" in svg
    assert "ev-g-node--parked" in svg
    assert svg.count("data-node-id=") == 3


def test_a_parked_await_is_drawn_as_a_hexagon():
    """The one visual distinction the interaction surface depends on: the run can stop here."""
    graph = prepare(from_keys("r1", [], pending=Key.parse("event;review:m1")))
    svg = to_svg(graph, _fake_geometry(graph))
    assert "<polygon" in svg
    assert "<rect" not in svg


def test_a_label_carrying_markup_is_escaped_not_injected():
    """A key's terminal position carries author text, so a request id with a tag in it
    reaches the renderer. It must arrive as characters."""
    key = 'ledger;r1:<script>alert("x")</script>'
    graph = prepare(from_keys("r1", [key]))
    svg = to_svg(graph, _fake_geometry(graph))
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_feedback_and_commit_order_edges_get_their_own_classes():
    folded = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    assert "ev-g-edge--feedback" in to_svg(folded, _fake_geometry(folded))
    unrolled = prepare(from_keys("r1", GATHER))
    assert "ev-g-edge--commit-order" in to_svg(unrolled, _fake_geometry(unrolled))


# --- the markup t-string seam ------------------------------------------------------------------


def test_a_quote_in_a_key_cannot_break_out_of_an_attribute():
    """Position-aware escaping — the thing an f-string plus `escape()` structurally cannot do,
    because by the time the escape runs the value's position is gone. Here the SAME key lands in
    an attribute and in text, and each gets the escape its position needs."""
    key = 'ledger;r1:" onload="alert(1)'
    graph = prepare(from_keys("r1", [key]))
    svg = to_svg(graph, _fake_geometry(graph))
    assert "onload=" not in svg.replace("&#34; onload=", "")  # no attribute was forged
    assert "&#34;" in svg


def test_the_stylesheet_is_raw_text_not_escaped_text():
    """`<style>` is a raw-text element, so a `>` in a CSS selector must survive. The string
    version of this renderer got it right only by forgetting an escape at that one site; the
    processor gets it right from the element's content model."""
    graph = prepare(from_keys("r1", ["a"]))
    svg = to_svg(graph, _fake_geometry(graph), inline_css=True)
    assert "@media (prefers-color-scheme: dark)" in svg
    assert "&gt;" not in svg


def test_fragments_stay_templates_so_they_can_only_compose_structurally():
    """The composition guarantee: a list of `Template`s splices, a list of `str` is escaped as
    text. That is what makes markup-by-concatenation — the Bobby Tables shape in a third grammar —
    unavailable by accident, and it only reaches the top if every fragment stays a `Template`."""
    from string.templatelib import Template

    from tdom import svg as render_svg

    from effective.graphlayout.svg import _node

    graph = prepare(from_keys("r1", ["a"]))
    fragment = _node(graph.nodes[0], Rect(0, 0, 100, 40))
    assert isinstance(fragment, Template)

    # ... and had it been flattened to a str first, splicing it would ESCAPE it rather than
    # silently accept someone else's markup:
    assert "&lt;rect&gt;" in render_svg(t"<g>{['<rect></rect>']}</g>")


def test_the_brand_green_never_paints_a_node():
    """Green is CHROME (the page frame and the run-level status chip), never a node state.

    Two independent reasons, both recorded in `GRAPH_CSS`. Measured: the brand green `#5f8a44`
    against the parked amber scores OKLab ΔE 4.3 under deuteranopia and 12.7 under *normal*
    vision, so as two marks they are not reliably separable by anyone. Structural: retry, budget,
    permission and refusal are handler interpretations rather than nodes, so a green
    `committed` would paint almost the whole drawing, and an accent covering everything has
    stopped accenting. The logo agrees: its green pawl is a point at the origin, ~1% of the mark.
    """
    node_rules = [
        line
        for line in GRAPH_CSS.splitlines()
        if line.startswith((".ev-g-node", ".ev-g-edge", ".ev-g-arrowhead"))
    ]
    assert node_rules  # the sweep is only meaningful if it actually matched rules
    assert not [line for line in node_rules if "--ev-green" in line]
    assert "--ev-green:" in GRAPH_CSS  # ...but the token IS defined, for the host page to use


def test_a_state_is_never_carried_by_colour_alone():
    """Red/amber is exactly the pair that collapses for the commonest colour-vision deficiency,
    so the state word on the detail line is load-bearing rather than decorative. A non-committed
    node carries three redundant signals: hue, a heavier (or dashed) stroke, and the word."""
    graph = prepare(from_keys("r1", [], pending=Key.parse("event;review:m1")))
    svg = to_svg(graph, _fake_geometry(graph), inline_css=True)
    assert "parked" in svg  # the WORD, not just the colour
    assert "--ev-parked" in svg  # the hue
    assert "stroke-width: 3" in svg  # the weight


def test_dark_mode_overrides_values_not_rules():
    """The custom properties are redeclared on `.ev-g`; every selector is written once. A host
    page can therefore rebrand a rendered SVG by setting properties — which matters because a
    standalone SVG artifact cannot see a host's theme toggle."""
    dark = GRAPH_CSS.split("@media (prefers-color-scheme: dark)")[1]
    block = dark[: dark.index("}\n}") + 3]
    assert "--ev-ink:" in block
    assert "--ev-parked:" in block
    # nothing but property redeclarations inside — no second copy of a rule to drift
    assert ".ev-g-node" not in block
    assert ".ev-g-edge" not in block


def test_kind_is_carried_by_shape_so_hue_stays_free_for_state():
    """The disjointness rule: shape is the KIND axis, hue is the STATE axis. Before this, kind was
    spent on fill hue (a yellow artifact, a blue await), which is what left no budget for state."""
    shapes = {}
    for kind, keys in (
        ("step", ["extract"]),
        ("artifact", ["artifact:application/json:sha256-abc"]),
        ("await", ["event;review:m1"]),
    ):
        graph = prepare(from_keys("r1", keys))
        svg = to_svg(graph, _fake_geometry(graph))
        shapes[kind] = "polygon" if "<polygon" in svg else "rect"
    assert shapes == {"step": "rect", "artifact": "polygon", "await": "polygon"}
