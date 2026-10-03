"""The ELK wire format — one canonical encoding in, one normalized `Geometry` out.

The encoder is deliberately the *only* place ELK's option names appear. Everything upstream
(`RunGraph`, `LayoutGraph`) is engine-neutral, which is what makes a second engine — graphviz, a
restricted native layout, a hand-rolled linear placer — a swap rather than a rewrite. Same shape as
the substrate's own control axis: ops are data, and the handler is the thing you replace.

**Semantics travel beside the graph, not inside it** (`envelope`). ELK is not asked to carry
`effective.*` keys through its layout pipeline and hand them back; a renderer that needed a node's
kind would then be depending on an engine's field-preservation behavior. Instead the wire format is
two halves: `graph` is what ELK consumes, `metadata` is what the renderer consumes, joined by node
and edge id. Neither half can silently lose the other's fields.

**Model order is the load-bearing option.** Children are serialized in tape order and
`considerModelOrder` tells ELK to honor it, so the drawing reads in the order the run happened —
and `cycleBreaking.strategy=MODEL_ORDER` then reverses exactly the edges that run backwards in that
order, which is the same set `prepare.classify` marked `feedback`. The semantic classification and
the engine's cycle break agree by construction rather than by luck.
"""

import json
from collections.abc import Mapping
from typing import Any

from effective.graphlayout.model import (
    EdgeGeometry,
    EdgeSection,
    Geometry,
    LayoutGraph,
    NodeGeometry,
    Point,
    Rect,
)

SCHEMA_VERSION = 1

DEFAULT_OPTIONS: Mapping[str, str] = {
    "elk.algorithm": "layered",
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.randomSeed": "1",
    "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    "elk.layered.cycleBreaking.strategy": "MODEL_ORDER",
    "elk.spacing.nodeNode": "24",
    "elk.layered.spacing.nodeNodeBetweenLayers": "48",
    "elk.padding": "[top=16,left=16,bottom=16,right=16]",
}
"""A narrow, versioned option set. Changing it changes every drawing, so treat an edit as a visual
regression requiring snapshot review."""


def to_elk(graph: LayoutGraph, *, options: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`LayoutGraph` → the ELK JSON an engine consumes. Geometry input only; no semantics."""
    return {
        "id": graph.id,
        "layoutOptions": {
            **DEFAULT_OPTIONS,
            "elk.direction": graph.direction,
            **(options or {}),
        },
        "children": [
            {"id": node.id, "width": node.size.width, "height": node.size.height}
            for node in graph.nodes
        ],
        "edges": [
            {"id": edge.id, "sources": [edge.source], "targets": [edge.target]}
            for edge in graph.edges
        ],
    }


def envelope(graph: LayoutGraph, *, options: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The full wire payload: the ELK graph, plus the semantics keyed by the same ids.

    This is what crosses to a browser. `members` is deliberately absent from the node metadata —
    a folded node standing for a hundred executions ships its *count*, and the hundred keys stay
    server-side behind a drill-down query."""
    return {
        "schema_version": SCHEMA_VERSION,
        "graph": to_elk(graph, options=options),
        "metadata": {
            "direction": graph.direction,
            "cyclic": graph.cyclic,
            "nodes": {
                node.id: {
                    "label": node.label,
                    "kind": node.kind,
                    "state": node.state,
                    "order": node.order,
                    "count": node.count,
                    "member_count": len(node.members),
                    "cost": node.cost,
                    "duration_ns": node.duration_ns,
                }
                for node in graph.nodes
            },
            "edges": {
                edge.id: {
                    "kind": edge.kind,
                    "count": edge.count,
                    "source": edge.source,
                    "target": edge.target,
                }
                for edge in graph.edges
            },
        },
    }


def canonical_json(value: Any) -> bytes:
    """Byte-stable serialization — for cache keys, fixture identity, and golden tests."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _point(raw: Mapping[str, Any]) -> Point:
    return Point(x=float(raw["x"]), y=float(raw["y"]))


def _section(raw: Mapping[str, Any]) -> EdgeSection:
    return EdgeSection(
        start=_point(raw["startPoint"]),
        bends=tuple(_point(bend) for bend in raw.get("bendPoints", ())),
        end=_point(raw["endPoint"]),
    )


def from_elk(
    result: Mapping[str, Any],
    *,
    engine: str = "elkjs",
    engine_version: str | None = None,
) -> Geometry:
    """An ELK response → `Geometry`, normalized once so nothing downstream reads ELK's shape.

    ELK returns the input graph *decorated* — the same children and edges with coordinates added,
    plus GWT bookkeeping (`$H`) that is not part of the format. Reading only the fields the
    contract names is what keeps that engine detail from leaking into the renderer."""
    children = result.get("children", ())
    return Geometry(
        graph_id=str(result["id"]),
        bounds=Rect(
            x=float(result.get("x", 0.0)),
            y=float(result.get("y", 0.0)),
            width=float(result.get("width", 0.0)),
            height=float(result.get("height", 0.0)),
        ),
        nodes=tuple(
            NodeGeometry(
                id=str(child["id"]),
                bounds=Rect(
                    x=float(child.get("x", 0.0)),
                    y=float(child.get("y", 0.0)),
                    width=float(child.get("width", 0.0)),
                    height=float(child.get("height", 0.0)),
                ),
            )
            for child in children
        ),
        edges=tuple(
            EdgeGeometry(
                id=str(edge["id"]),
                sections=tuple(_section(section) for section in edge.get("sections", ())),
            )
            for edge in result.get("edges", ())
        ),
        engine=engine,
        engine_version=engine_version,
    )


__all__ = [
    "DEFAULT_OPTIONS",
    "SCHEMA_VERSION",
    "canonical_json",
    "envelope",
    "from_elk",
    "to_elk",
]
