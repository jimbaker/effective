"""Projection helpers the fixtures share: a map that knows a fixture's own tags, and the fold
that answers "how often did this OP run" whatever frames the quotient kept.

`fold_cycles(drop=…)` reads roles out of `build/key-registry.json`, which is built from `src/` and
`examples/` and deliberately not from `tests/`: the source map answers "which line produced this
key?" about production, and a fixture's `talk:` has no business in that answer
(`justfile`, `--key-registry`).

So a fixture that mints its own scope declares the role at its own mint and hands the fold a map
that read its own file. That is the replacement for `drop_scopes=…`, which named a stranger's tag
at the FOLD and could say nothing about what the coordinate meant.
"""

from pathlib import Path

from effective.graphview import RunGraph, bare_name, regroup
from effective.keys.registry import KeyMap
from effective.lint import registry_of


def including(*paths: str | Path) -> KeyMap:
    """The shipped map plus whatever `paths` mint. The substrate's declaration wins a tie.

    A tag already in the shipped map keeps it: two variants of one tag have to be provably
    disjoint before a registry admits them (`separated`), and a fixture re-declaring a substrate
    tag is a collision rather than an extension. A tag the shipped map does not hold arrives with
    EVERY variant the fixtures mint, grouped by `from_shapes` the way `load` groups them, so a
    fixture spelling `talk:{a}` and `talk:{a},{b}` gets both arms."""
    shapes, _ = registry_of({str(p): Path(p).read_text() for p in paths})
    shipped = KeyMap.load().variants
    minted = KeyMap.from_shapes([s for s in shapes if s.tag]).variants
    return KeyMap(variants=shipped | {t: v for t, v in minted.items() if t not in shipped})


def by_op(graph: RunGraph) -> dict[str, int]:
    """`{op: executions}` over a folded graph, summing across the frames it kept.

    A folded label carries its surviving frames (`gather:*,*;talk:*;step:audit`), and a sweep over
    generated widths cannot spell one: whether a `gather:` wraps the drill is a property of the
    case. `bare_name` is the op under whatever frames, so this is `regroup` with that as the
    quotient, which is the same arithmetic `fold_cycles` uses one level up."""
    folded = regroup(graph, bare_name, dropped=("frames",))
    return {node.key: node.count for node in folded.nodes}
