"""The real engine: `effective.graphlayout.elkjs` against the pinned, sandboxed elkjs.

Skips without the image — `just elk-image` builds it (rootless Podman, digest-pinned node base,
`npm ci --ignore-scripts`, bundle hashes verified at build). Same posture as the Podman-Postgres
replay tests: the instrument is a container, and its absence is a skip, never a silent fallback to
an unsandboxed run.

**Reproduce, don't reason** is the point of this file. Everything in `test_graphlayout.py` is a
claim about what Python decides; these are the claims that need an engine to be true — that the
ELK JSON this repo emits is accepted, that a folded loop's feedback edge actually routes, that a
run's boxes do not overlap. A design-only pass would have missed the `$H` fields and the
container's `--read-only` behavior alike.
"""

import pytest

from effective.graphlayout import (
    check_geometry,
    check_graph,
    overlaps,
    prepare,
    to_svg,
)
from effective.graphlayout import elkjs as engine_module
from effective.graphview import fold_cycles, from_keys
from effective.keys import Index, Key

pytestmark = pytest.mark.skipif(
    not engine_module.available(),
    reason="no pinned elkjs worker — run `just elk-image` (or set EFFECTIVE_ELK_HOST=1)",
)

LINEAR = ["extract", "ask", "ledger;r1:request_processed"]
LOOP = ["plan", "ask", "act", "ask#2", "act#2", "ask#3", "ledger;r1:done"]
PARKED = ["extract", "ledger;r1:submitted", "$awaitEvent:approve;r1"]


@pytest.fixture(scope="module")
def elk():
    with engine_module.ElkJs() as worker:
        yield worker


def test_a_linear_run_lays_out_left_to_right_in_tape_order(elk):
    """Model order is not a hint here — it is the contract that makes a run read as a run."""
    graph = prepare(from_keys("r1", LINEAR))
    check_graph(graph)
    geometry = elk.layout(graph)
    check_geometry(graph, geometry)

    xs = [geometry.node(node.id).bounds.x for node in graph.nodes]
    assert xs == sorted(xs)
    assert overlaps(geometry) == ()
    assert geometry.engine_version == "0.12.0"


def test_the_folded_loop_routes_its_feedback_edge_with_bends(elk):
    """A back edge cannot be drawn as a straight arrow between adjacent layers; if ELK routed it
    at all, it went around. That is the observable difference between a graph whose cycle was
    declared and one whose cycle was guessed at."""
    graph = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    geometry = elk.layout(graph)
    check_geometry(graph, geometry)

    feedback = next(edge for edge in graph.edges if edge.kind == "feedback")
    routed = next(edge for edge in geometry.edges if edge.id == feedback.id)
    assert routed.sections
    assert any(section.bends for section in routed.sections)
    assert overlaps(geometry) == ()


def test_a_parked_await_is_placed_like_any_other_node(elk):
    """The node the dashboard's interaction hinges on is not special to the layout engine — its
    state rides in the metadata half and its shape is the renderer's business."""
    graph = prepare(from_keys("r1", PARKED, states={"$awaitEvent:approve;r1": "parked"}))
    geometry = elk.layout(graph)
    check_geometry(graph, geometry)
    assert geometry.node("$awaitEvent:approve;r1").bounds.width > 0

    svg = to_svg(graph, geometry, inline_css=True)
    assert svg.startswith("<svg")
    assert 'data-node-id="$awaitEvent:approve;r1"' in svg
    assert "ev-g-node--parked" in svg


def test_the_same_graph_lays_out_identically_twice(elk):
    """Determinism is what makes golden geometry and SVG snapshots possible at all — and it is a
    property of the pinned option set (`elk.randomSeed`), not of the algorithm."""
    graph = prepare(fold_cycles(from_keys("r1", LOOP), drop=(Index,)))
    assert elk.layout(graph) == elk.layout(graph)


def test_a_worker_that_dies_between_requests_is_replaced_transparently():
    """The failure mode the facade exists to absorb. A dead worker costs a container start, not a
    failed request — the process is disposable, which is the whole reason layout may live outside
    the Python process at all."""
    with engine_module.ElkJs() as worker:
        graph = prepare(from_keys("r1", LINEAR))
        assert worker.layout(graph).nodes

        process = worker._process
        assert process is not None
        process.kill()  # simulating the crash IS the test
        process.wait(timeout=5)

        assert worker.layout(graph).nodes


def test_a_wedged_worker_times_out_and_is_restarted():
    """A worker that accepts a request and never answers must not hang the caller. Driven against
    a deliberately mute process, because the real engine has no way to wedge on demand — and the
    timeout path is the one that would otherwise only be exercised in production."""

    class MuteWorker(engine_module.ElkJs):
        def _command(self) -> list[str]:
            return ["sh", "-c", "cat > /dev/null"]

    worker = MuteWorker(timeout=0.3)
    with pytest.raises(engine_module.ElkLayoutError, match="did not answer in time"):
        worker.layout(prepare(from_keys("r1", LINEAR)))
    assert worker._process is None  # killed, so the next call starts clean
    worker.close()


def test_a_malformed_graph_fails_as_a_layout_error_not_a_hang(elk):
    """An engine-side refusal arrives as a typed Python error carrying ELK's own message."""
    with pytest.raises(engine_module.ElkLayoutError, match="elkjs layout failed"):
        elk.request({"id": "r1", "children": [{"id": "a"}], "edges": [{"id": "e"}]})
    # ... and the worker is still usable afterwards: a refusal is not a protocol break.
    assert elk.layout(prepare(from_keys("r1", LINEAR))).nodes


def test_a_hundred_iteration_loop_folds_to_a_drawing_a_human_can_read(elk):
    """The legibility experiment, carried one stage further than `graphview` could take it: a
    drawing whose area shrinks with its node count."""
    # `Key.occurrence`, not `f"{op}#{i + 1}"`: a FIRST occurrence has no wire form — `#1` is a
    # parse error, not a synonym for the bare name — so the hand-spelled form emitted `ask#1`,
    # which `fold_cycles` could not parse and therefore could not fold (4 nodes, not 2). Deriving
    # through the substrate's own producer makes that off-by-one unwritable.
    trace = [
        Key.parse(op).occurrence(index + 1).stored()
        for index in range(100)
        for op in ("ask", "act")
    ]
    unrolled = prepare(from_keys("r1", trace))
    folded = prepare(fold_cycles(from_keys("r1", trace), drop=(Index,)))

    big = elk.layout(unrolled)
    small = elk.layout(folded)

    assert len(unrolled.nodes) == 200
    assert len(folded.nodes) == 2
    assert small.bounds.width * small.bounds.height < big.bounds.width * big.bounds.height / 50
    assert overlaps(small) == ()
