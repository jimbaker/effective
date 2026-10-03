"""The checks that say whether the semantic layer is doing its job.

Three families, and the middle one is the interesting one:

- **Input** (`check_graph`) — the malformed-graph cases an engine reports as an opaque failure:
  duplicate ids, a dangling endpoint, a zero-width box. Catching them here turns a JS stack trace
  into a Python error naming the node.
- **Semantic** (`forward_is_acyclic`) — *the* falsifiable claim of the layout thesis. If Python
  has really resolved the cycle question before ELK sees the graph, then deleting the edges Python
  marked `feedback` must leave a DAG. Nothing in the type system enforces that; it is a property
  of `prepare.classify`, and it is pinned as a test rather than asserted in prose.
- **Output** (`check_geometry`, `overlaps`) — every node placed, every coordinate finite, nothing
  outside the reported bounds. These hold whichever engine produced the geometry, which is what
  makes them an acceptance contract for a *second* engine rather than a description of ELK.
"""

from math import isfinite

from effective.graphlayout.model import Geometry, LayoutGraph


class LayoutError(ValueError):
    """A graph or a geometry violated the layout contract."""


def check_graph(graph: LayoutGraph) -> None:
    """Refuse a graph no engine could lay out sensibly."""
    seen: set[str] = set()
    for node in graph.nodes:
        if node.id in seen:
            raise LayoutError(f"duplicate node id {node.id!r}")
        seen.add(node.id)
        if not (isfinite(node.size.width) and isfinite(node.size.height)):
            raise LayoutError(f"node {node.id!r} has a non-finite size")
        if node.size.width <= 0 or node.size.height <= 0:
            raise LayoutError(f"node {node.id!r} has a non-positive size {node.size}")

    edge_ids: set[str] = set()
    for edge in graph.edges:
        if edge.id in edge_ids:
            raise LayoutError(f"duplicate edge id {edge.id!r}")
        edge_ids.add(edge.id)
        for endpoint in (edge.source, edge.target):
            if endpoint not in seen:
                raise LayoutError(f"edge {edge.id!r} names unknown node {endpoint!r}")


def forward_is_acyclic(graph: LayoutGraph) -> bool:
    """Does dropping the `feedback` edges leave a DAG?

    The claim `prepare.classify` makes, stated so it can fail. It holds for a reason worth
    knowing: a `feedback` edge is exactly one whose target is not *later* in tape order, so every
    surviving edge strictly increases the order index — and a strictly increasing relation on a
    finite total order cannot close a cycle. Kahn's algorithm re-derives that the long way, which
    is the point: the test does not share the classification's reasoning."""
    order = {node.id: node.order for node in graph.nodes}
    indegree = {node.id: 0 for node in graph.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.kind == "feedback":
            continue
        if edge.source not in order or edge.target not in order:
            continue
        outgoing[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    peeled = 0
    while ready:
        node = ready.pop()
        peeled += 1
        for successor in outgoing[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return peeled == len(indegree)


def _check_placement(graph: LayoutGraph, geometry: Geometry) -> None:
    placed = {node.id: node.bounds for node in geometry.nodes}
    for node in graph.nodes:
        bounds = placed.get(node.id)
        if bounds is None:
            raise LayoutError(f"node {node.id!r} has no geometry")
        if not all(isfinite(value) for value in (bounds.x, bounds.y, bounds.width, bounds.height)):
            raise LayoutError(f"node {node.id!r} has a non-finite rect {bounds}")
        if bounds.width < 0 or bounds.height < 0:
            raise LayoutError(f"node {node.id!r} has a negative rect {bounds}")
    if unknown := set(placed) - {node.id for node in graph.nodes}:
        raise LayoutError(f"geometry names unknown nodes {sorted(unknown)}")


def _check_routes(geometry: Geometry) -> None:
    for edge in geometry.edges:
        for section in edge.sections:
            for point in section.points:
                if not (isfinite(point.x) and isfinite(point.y)):
                    raise LayoutError(f"edge {edge.id!r} has a non-finite point {point}")


def _check_bounds(geometry: Geometry, *, tolerance: float = 0.5) -> None:
    frame = geometry.bounds
    for node in geometry.nodes:
        box = node.bounds
        if (
            box.x < frame.x - tolerance
            or box.y < frame.y - tolerance
            or box.x + box.width > frame.x + frame.width + tolerance
            or box.y + box.height > frame.y + frame.height + tolerance
        ):
            raise LayoutError(f"node {node.id!r} at {box} escapes the graph bounds {frame}")


def check_geometry(graph: LayoutGraph, geometry: Geometry) -> None:
    """Every node placed, every number finite, everything inside the reported bounds."""
    _check_placement(graph, geometry)
    _check_routes(geometry)
    _check_bounds(geometry)


def overlaps(geometry: Geometry, *, tolerance: float = 0.5) -> tuple[tuple[str, str], ...]:
    """Pairs of node rectangles that intersect — a readability gate, not a correctness one.

    Quadratic, and that is fine: the interactive corpus is tens of nodes, and the folded view is
    the answer for anything larger."""
    found: list[tuple[str, str]] = []
    boxes = geometry.nodes
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            a, b = first.bounds, second.bounds
            if (
                a.x + a.width - tolerance > b.x
                and b.x + b.width - tolerance > a.x
                and a.y + a.height - tolerance > b.y
                and b.y + b.height - tolerance > a.y
            ):
                found.append((first.id, second.id))
    return tuple(found)


__all__ = ["LayoutError", "check_geometry", "check_graph", "forward_is_acyclic", "overlaps"]
