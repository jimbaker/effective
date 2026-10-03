"""The layout seam: `RunGraph` in, geometry out, with every Effective question answered in Python.

    RunGraph ──prepare──▶ LayoutGraph ──to_elk──▶ ELK JSON ──engine──▶ Geometry ──to_svg──▶ SVG
              (semantics)               (wire)              (elkjs)              (render)

**Python decides what the graph means; the engine decides where to draw it.**
The split earns its keep on facts a generic layout engine cannot recover from a node-and-edge list:
which edge is the loop a fold recovered, which two nodes are adjacent only by commit order inside
a gather, which node the run is parked on. `prepare.classify` resolves each of those into a
`LayoutEdge.kind` before serialization, and `validate.forward_is_acyclic` is the falsifiable claim
that it did.

It is the same layering the substrate already uses twice: ops-as-data with a swappable handler on
the control axis, typed channels with a processor on the data axis. Here the description is a
`LayoutGraph` and the interpreter is a layout engine; `elkjs` is one, and the seam is small enough
that graphviz or a restricted native layout is a swap rather than a rewrite.

Nothing in this package reads a database, an engine, or a clock. `prepare` is a pure function of a
`RunGraph`, which is a pure function of recorded op keys, so the whole read path down to SVG is
testable from a list of strings, and geometry stays disposable (never durable, never hand-placed).
"""

from effective.graphlayout.elk import DEFAULT_OPTIONS, canonical_json, envelope, from_elk, to_elk
from effective.graphlayout.model import (
    EdgeGeometry,
    EdgeKind,
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
from effective.graphlayout.prepare import classify, elide, node_size, prepare
from effective.graphlayout.svg import GRAPH_CSS, to_svg
from effective.graphlayout.validate import (
    LayoutError,
    check_geometry,
    check_graph,
    forward_is_acyclic,
    overlaps,
)

__all__ = [
    "DEFAULT_OPTIONS",
    "GRAPH_CSS",
    "EdgeGeometry",
    "EdgeKind",
    "EdgeSection",
    "Geometry",
    "LayoutEdge",
    "LayoutError",
    "LayoutGraph",
    "LayoutNode",
    "NodeGeometry",
    "Point",
    "Rect",
    "Size",
    "canonical_json",
    "check_geometry",
    "check_graph",
    "classify",
    "elide",
    "envelope",
    "forward_is_acyclic",
    "from_elk",
    "node_size",
    "overlaps",
    "prepare",
    "to_elk",
    "to_svg",
]
