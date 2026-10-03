"""`RunGraph` → `LayoutGraph`: resolve every semantic question before an engine sees the graph.

Four things happen here, and each is something ELK would otherwise have to guess at:

1. **Order.** `graphview` emits nodes in tape order (first occurrence, for a folded node), so
   the index *is* the order. It rides along as `LayoutNode.order` and becomes ELK's model order.
2. **Edge kind.** Which edge is a loop, which crosses a gather boundary, which is commit order
   only.
   See `classify`: the whole classification falls out of two facts the projection already has:
   position in tape order, and the branch coordinate.
3. **Size.** Canonically, in Python, so the Node facade and a browser lay out the same graph
   identically. Measuring text in the browser would make server geometry a second
   implementation.
4. **Label.** The key, middle-elided to fit the box. Never the count: a badge that grows from
   `x99` to `x100` must not resize a node and reflow the whole drawing.

Nothing here reads a DB or an engine. It is a pure function of a `RunGraph`, which is itself a
pure function of recorded keys, so the entire read path down to geometry is testable with a
list of strings.
"""

from effective.graphlayout.model import EdgeKind, LayoutEdge, LayoutGraph, LayoutNode, Size
from effective.graphview import Node, RunGraph

CHAR_WIDTH = 7.0
"""Conservative advance width for the label font (13px system sans). Over- rather than
under-estimating: a box slightly too wide is invisible, a box too narrow clips its text."""

MAX_TEXT_WIDTH = 300.0
MIN_TEXT_WIDTH = 96.0
ICON_WIDTH = 24.0  # the kind glyph
BADGE_WIDTH = 44.0  # the execution-count badge — a CONSTANT, whatever the count
PADDING = 32.0
NODE_HEIGHT = 56.0  # two lines: the key, then state / cost / duration

MAX_LABEL_CHARS = int(MAX_TEXT_WIDTH / CHAR_WIDTH)  # 42


def elide(key: str, limit: int = MAX_LABEL_CHARS) -> str:
    """Shorten a key to fit its box by dropping the MIDDLE.

    Both ends of an op key carry meaning and the middle rarely does: the head is the namespace
    that says what kind of thing this is (`gather:0,1;ledger;`) and the tail is the author's own
    text, which the key grammar keeps readable on purpose (`…:ticket_processed:m1`). Truncating
    from the right would throw away exactly the half a human is looking for. The full key stays on
    `LayoutNode.id`; the label is display only."""
    if len(key) <= limit:
        return key
    head = (limit - 1) * 2 // 5
    tail = limit - 1 - head
    return f"{key[:head]}…{key[-tail:]}"


def node_size(node: Node, label: str) -> Size:
    """A node's box, from its label and kind alone.

    Deliberately independent of `count` except through `count > 1`: the badge is a fixed width,
    so a run whose loop ticks from 99 to 100 iterations re-renders with *identical* geometry.
    That is the cheap half of layout stability: no diffing and no position hints, just a size
    function that does not depend on the thing that changes."""
    text = min(MAX_TEXT_WIDTH, max(MIN_TEXT_WIDTH, len(label) * CHAR_WIDTH))
    icon = 0.0 if node.kind == "step" else ICON_WIDTH
    badge = BADGE_WIDTH if node.count > 1 else 0.0
    return Size(width=round(text + icon + badge + PADDING, 1), height=NODE_HEIGHT)


def classify(source: Node, target: Node, *, order: dict[str, int]) -> EdgeKind:
    """Why this edge exists, from tape position and the branch coordinate alone.

    The back-edge test is a comparison in ONE total order (first occurrence in the tape) rather
    than a DFS, which buys two things a depth-first classification does not: it is deterministic
    without a start-node convention, and removing every `feedback` edge provably leaves an
    acyclic graph, because each remaining edge strictly increases the index. That property is
    what lets a layered engine draw the loop as a loop instead of breaking the cycle wherever it
    happened to enter.

    `commit-order` is the one that matters most and is easiest to get wrong. Two nodes in
    *different* gather branches are adjacent in the tape only because they committed in that
    order, and concurrent branches commit in a race. Drawing it as a plain arrow would assert
    causation the record does not contain."""
    if order[target.key] <= order[source.key]:
        return "feedback"
    if len(target.path) > len(source.path):
        return "gather"
    if len(target.path) < len(source.path):
        return "join"
    if target.path != source.path:
        return "commit-order"
    return "forward"


def prepare(
    graph: RunGraph,
    *,
    direction: str = "RIGHT",
) -> LayoutGraph:
    """Project a run graph onto its layout IR.

    `direction` is the reading axis: `RIGHT` for a run (time runs left to right, the shape a
    linear trace wants), `DOWN` for a folded program graph. Edges whose endpoints are not both
    present are dropped — `fold_cycles` can emit them, and an engine treats a dangling endpoint
    as a hard error rather than a missing box."""
    order = {node.key: index for index, node in enumerate(graph.nodes)}
    nodes = tuple(
        LayoutNode(
            id=node.key,
            label=elide(node.key),
            size=node_size(node, elide(node.key)),
            kind=node.kind,
            state=node.state,
            order=order[node.key],
            count=node.count,
            members=node.members,
            cost=node.cost,
            duration_ns=node.duration_ns,
        )
        for node in graph.nodes
    )
    by_key = {node.key: node for node in graph.nodes}
    edges = tuple(
        # The id is composed from the two ORDINALS, not from the keys. Joining two keys with a
        # separator is the adjacent-interpolation injectivity hole: keys carry author text
        # and can contain any delimiter you pick. Integers cannot, so
        # this composition is injective by construction, and it is prefix-stable for an
        # append-only run because a node's ordinal does not change when the tape grows.
        LayoutEdge(
            id=f"e{order[edge.src]}-{order[edge.dst]}",
            source=edge.src,
            target=edge.dst,
            kind=classify(by_key[edge.src], by_key[edge.dst], order=order),
            count=edge.count,
            order=index,
        )
        for index, edge in enumerate(graph.edges)
        if edge.src in order and edge.dst in order
    )
    return LayoutGraph(
        id=graph.run_id,
        nodes=nodes,
        edges=edges,
        direction=direction,  # ty: ignore[invalid-argument-type]
        cyclic=graph.cyclic,
    )


__all__ = ["CHAR_WIDTH", "NODE_HEIGHT", "classify", "elide", "node_size", "prepare"]
