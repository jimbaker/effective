"""The layout IR: what a layout engine needs, and nothing about Effective it does not.

Two frozen value types, in the shape of `effective.cards.spec`: a `LayoutGraph` going *in* and a
`Geometry` coming *out*. They define the seam:

    RunGraph  ──prepare──▶  LayoutGraph  ──engine──▶  Geometry  ──render──▶  SVG

**Python decides what the graph means; the engine decides where to draw it.** Everything that
requires knowing Effective (which edge is a loop back to the top of the ReAct cycle, which two
nodes only *look* sequential because they raced in a gather, which node the run is parked on) is
resolved into `LayoutGraph` fields *before* any engine sees the graph. ELK cannot infer those from
a cycle check, and it should not have to try.

**Ownership of the vocabularies is the reason `kind` and `state` are bare `str` here while
`EdgeKind` is a `Literal`.** A node's `kind` comes from the key grammar and belongs to
`effective.graphview.KINDS`; its `state` belongs to whatever reader produced it (the park reader
adds `parked`). Re-declaring either as a closed `Literal` in this module would make the layout
layer a second, drifting authority over a vocabulary it does not own, and would break the moment
a reader names a new state. Edge kinds are different: no upstream module has a
notion of edge kind at all, so this layer *is* their authority, and they get the closed type.

Nothing here is durable. Geometry is a disposable function of a `LayoutGraph`; the ledger is the
record (two bookkeepers), and a coordinate is never written back.
"""

from dataclasses import dataclass
from typing import Literal

type EdgeKind = Literal["forward", "feedback", "gather", "join", "commit-order"]
"""Why an edge exists: the semantics a generic layout engine cannot recover.

| kind           | the edge                                                                    |
|----------------|-----------------------------------------------------------------------------|
| `forward`      | program order within one scope; the ordinary case                           |
| `feedback`     | a back edge, whose target appeared earlier in tape order. Only a *folded*   |
|                | view makes one (a cycle in the program is a chain in the trace, so the      |
|                | unrolled DAG has none), and it is the loop the fold recovered               |
| `gather`       | into a deeper branch coordinate: the tape entering a gather region          |
| `join`         | out of one: the tape leaving it                                             |
| `commit-order` | between two *different* branches at the same depth. Concurrent branches     |
|                | commit in a race, so the edge means "these committed in this order in this  |
|                | run", and a renderer must not draw it as a cause                            |
"""


@dataclass(frozen=True)
class Size:
    width: float
    height: float


@dataclass(frozen=True)
class LayoutNode:
    """One box. `id` is the semantic key (an Effective op key, or a folded class of them), and
    it is the only identity that crosses to the browser and back."""

    id: str
    label: str
    size: Size
    kind: str
    """`effective.graphview.kind_of`'s vocabulary: step / ledger / artifact / sleep / event /
    await. Not re-declared as a `Literal` here — see the module docstring."""

    state: str = "committed"
    """Whatever the projector recorded. `parked` is the one the interaction hinges on."""

    order: int = 0
    """Position in tape order — first occurrence for a folded node. Handed to ELK as model
    order, which is how the drawing stays in the order the run actually happened."""

    count: int = 1
    """Executions this node stands for (>1 only in a folded view)."""

    members: tuple[str, ...] = ()
    """The unrolled keys behind a folded node — the drill-down. Carried in the *metadata* half
    of the ELK envelope, never as layout input: a hundred members are one box."""

    cost: float | None = None
    duration_ns: int | None = None
    """`None` is UNMEASURED, not free — carried through from `graphview.Node`, where the
    distinction is explained. `svg.py` and `elk.py` both read these falsily/pass-through, so
    an unmeasured node simply has no badge rather than a `$0.0000` one."""


@dataclass(frozen=True)
class LayoutEdge:
    id: str
    source: str
    target: str
    kind: EdgeKind = "forward"
    count: int = 1
    order: int = 0


@dataclass(frozen=True)
class LayoutGraph:
    id: str
    nodes: tuple[LayoutNode, ...] = ()
    edges: tuple[LayoutEdge, ...] = ()
    direction: Literal["RIGHT", "DOWN", "LEFT", "UP"] = "RIGHT"
    cyclic: bool = False

    def node(self, node_id: str) -> LayoutNode:
        """The node with this id. Raises rather than returning `None`: an id that is not in the
        graph is a bug in whatever produced it, and a silent `None` would surface as a missing
        box three layers later."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)


# --- geometry: what comes back ------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class EdgeSection:
    """One routed run of an edge: start, orthogonal bends, end."""

    start: Point
    bends: tuple[Point, ...]
    end: Point

    @property
    def points(self) -> tuple[Point, ...]:
        return (self.start, *self.bends, self.end)


@dataclass(frozen=True)
class NodeGeometry:
    id: str
    bounds: Rect


@dataclass(frozen=True)
class EdgeGeometry:
    id: str
    sections: tuple[EdgeSection, ...]


@dataclass(frozen=True)
class Geometry:
    """A layout result, normalized away from any one engine's response shape.

    `engine`/`engine_version` are diagnostics, not identity: two engines may legitimately place
    the same graph differently, and the acceptance test is the invariants in
    `graphlayout.validate`, never pixel equality."""

    graph_id: str
    bounds: Rect
    nodes: tuple[NodeGeometry, ...] = ()
    edges: tuple[EdgeGeometry, ...] = ()
    engine: str = "elkjs"
    engine_version: str | None = None

    def node(self, node_id: str) -> NodeGeometry:
        """This node's placement, or `KeyError` — see `LayoutGraph.node`."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)


__all__ = [
    "EdgeGeometry",
    "EdgeKind",
    "EdgeSection",
    "Geometry",
    "LayoutEdge",
    "LayoutGraph",
    "LayoutNode",
    "NodeGeometry",
    "Point",
    "Rect",
    "Size",
]
