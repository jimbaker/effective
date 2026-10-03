"""π over the tape's own coordinate space — the projection whose axes are discovered.

`fold_cycles` takes its axes from the ROLE each coordinate declares at its mint, read back out
of `build/key-registry.json`: a coordinate is an axis because the site that composed it said so.
`project` asks the TAPE instead, and this file is the evidence that the two agree where the
declaration is right and that the discovered one reaches where no declaration exists.

The cases are the repo's own examples, driven rather than constructed wherever one exists:
the **funnel** (a lane ident in bijection with its gather branch), the **cart** (a ledger row per
line at one program point), and an **agent loop** (the hundred-node graph nobody wants to look at).
"""

import _cart as cart
import _coding as coding
import _funnel as funnel
import _mcts as mcts
import pytest
from _keymap import including

from effective.graphview import (
    Edge,
    Node,
    RunGraph,
    axes,
    fold_cycles,
    from_keys,
    kind_of,
    project,
    restrict,
)
from effective.handlers.recording import RecordingHandler, Suspended

pytestmark = pytest.mark.spine


def _tape(program, answers, reply: str) -> list[str]:
    handler = RecordingHandler(responses=answers)
    outcome = handler.run(program)
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(reply)
    return [entry.key.stored() for entry in handler.trace]


def funnel_tape() -> list[str]:
    return _tape(lambda: funnel.triage("cfp-2026"), funnel.Answers(funnel.SUBMISSIONS), "uphold")


def cart_tape() -> list[str]:
    return _tape(lambda: cart.checkout(cart.ORDER_ID), cart.Answers(cart.CART), cart.APPROVED)


def agent_loop(iterations: int = 100) -> list[str]:
    """The shape the legibility argument is about: one program point per op, walked N times,
    with a DISTINCT authored ledger id each round. A real agent loop must write that, and
    `fold_cycles` cannot collapse it."""
    tape: list[str] = []
    for n in range(iterations):
        tape += [f"rec:{n};step:think", f"rec:{n};step:search", f"rec:{n};ledger;iteration:m{n}"]
    return tape


def test_the_lane_axis_is_discovered_without_being_declared():
    """The funnel's `talk:{ident}` is an axis, and nothing says so: it is in BIJECTION with the
    gather branch, so collapsing the branch while keeping the ident would reduce nothing. That
    bijection is exactly why `drop_scopes=("talk",)` had to exist.

    Reddens when: the closure step is dropped, leaving only integer-valued seeds."""
    graph = from_keys("cfp-2026", funnel_tape())
    lane_columns = {
        tag for tags, columns in axes(graph).items() for _, tag, _ in columns if "talk" in tags
    }
    assert "talk" in lane_columns
    assert "gather" in lane_columns  # the branch index it rides with


def test_a_dependency_that_is_not_a_bijection_survives():
    """The near-miss beside it, and the reason the closure tests bijection rather than dependency:
    the branch DETERMINES which rubric a lane used, but a rubric does not determine the branch —
    two talks share `story`. So `score:systems` and `score:story` must stay two nodes.

    Reddens when: the closure keeps a column that is merely determined by a dropped one."""
    projected = project(from_keys("cfp-2026", funnel_tape()))
    scored = {node.key: node.count for node in projected.nodes if "score" in node.key}
    assert len(scored) == 2, scored
    assert sum(scored.values()) == len(funnel.SUBMISSIONS)


def test_the_discovered_projection_agrees_with_the_declared_one_on_the_funnel():
    """The calibration: where `fold_cycles` is told the right answer, `project` finds it. Node and
    edge counts both, since a projection that merged correctly and lost the edges would pass a
    node count alone.

    Reddens when: the two quotients disagree on a case the declared one already handles."""
    graph = from_keys("cfp-2026", funnel_tape())
    declared = fold_cycles(graph, keymap=including("tests/_funnel.py"))
    discovered = project(graph)
    assert len(discovered.nodes) == len(declared.nodes)
    assert len(discovered.edges) == len(declared.edges)
    assert discovered.cyclic == declared.cyclic


def test_the_cart_collapses_its_ledger_axis_and_keeps_the_order_row():
    """The case the declared fold cannot reach even when told about `item:`: the ledger rows carry
    an authored id per line, so they stay one node each. `placed:` is a DIFFERENT position and
    must survive — a projection that collapsed everything would be as wrong as one that collapsed
    nothing.

    Reddens when: the ledger id stops being read as a column, or `placed:` folds in with it."""
    graph = from_keys(cart.ORDER_ID, cart_tape())
    declared = fold_cycles(graph, keymap=including("tests/_cart.py"))
    discovered = project(graph)

    reserved = [n for n in discovered.nodes if n.key.endswith("reserved")]
    placed = [n for n in discovered.nodes if "placed" in n.key]
    assert len(reserved) == 1, reserved
    assert reserved[0].count == len(cart.CART)
    assert len(placed) == 1, placed
    assert placed[0].count == 1
    assert len([n for n in declared.nodes if "reserved" in n.key]) == len(cart.CART)
    assert len(discovered.nodes) < len(declared.nodes)


def test_the_hundred_node_graph_becomes_three_nodes_and_keeps_its_backedge():
    """The whole point, on the shape that motivates it. The backedge is what makes the projection
    a program rather than a summary, and its count is one short of the node counts because the
    last iteration does not loop back.

    Reddens when: edges are collapsed by node identity but the cycle is broken — a fold that
    reduces nodes and loses the loop has projected onto the wrong thing."""
    tape = agent_loop()
    graph = from_keys("r", tape)
    discovered = project(graph)

    assert len(fold_cycles(graph).nodes) == 102  # the declared fold, for contrast
    assert len(discovered.nodes) == 3
    assert {node.count for node in discovered.nodes} == {100}
    assert discovered.cyclic
    (backedge,) = [e for e in discovered.edges if "ledger" in e.src]
    assert backedge.count == 99


def test_a_constant_column_is_not_an_axis():
    """The guard that a first version lacked: two constant columns are trivially mutually
    determining, so without it a one-row group collapses every column it has and the run's own
    names vanish. A single-line cart must project onto itself.

    Reddens when: the varies-check is removed from either the seed or the closure.

    Asserted on the KEYS, not the node count: a count stays green under both mutations, because
    in a tape whose every group is a singleton, eliding a column changes each node's name and
    merges nothing. A projection that quietly renamed every node while keeping the arithmetic
    right is green-and-wrong under a count."""
    one_line = cart.CART[:1]
    tape = _tape(lambda: cart.checkout(cart.ORDER_ID), cart.Answers(one_line), cart.APPROVED)
    graph = from_keys(cart.ORDER_ID, tape)
    assert {node.key for node in project(graph).nodes} == {node.key for node in graph.nodes}


@pytest.mark.parametrize(
    "tape",
    [funnel_tape, cart_tape, agent_loop],
    ids=["funnel", "cart", "agent-loop"],
)
def test_a_projection_shows_fewer_nodes_never_fewer_facts(tape):
    """`executions` is invariant across projections — the promise that makes a compact view
    honest, and `members` is the drill-down that makes it a lens rather than a summary.

    Reddens when: `regroup` stops summing counts, or drops a member on the way through."""
    graph = from_keys("r", tape())
    discovered = project(graph)
    assert discovered.executions == graph.executions
    assert sorted(m for node in discovered.nodes for m in node.members) == sorted(
        node.key for node in graph.nodes
    )


# --- SELECTION: choose the rows, where π chose the columns ----------------------------------


def test_restrict_contracts_edges_rather_than_deleting_them():
    """Hiding the tool calls between two ledger rows must still draw the edge between them. A
    filtered graph that fell into disconnected islands would be worse than the unfiltered one,
    which is why selection contracts through what it hides.

    Asserted on the UNPROJECTED graph, and that is the point: after `project` the cart's last
    reservation already sits next to the order row, so the pair a naive filter would break is not
    even adjacent there. A first version asserted that pair and stayed green under its own named
    mutation — the test picked the one edge contraction was not needed for.

    Reddens when: edges are kept only where BOTH endpoints survive."""
    graph = from_keys(cart.ORDER_ID, cart_tape())
    world = restrict(graph, lambda node: kind_of(node.key) == "ledger")

    assert {kind_of(node.key) for node in world.nodes} == {"ledger"}
    reserved = sorted(n.key for n in world.nodes if "reserved" in n.key)
    # four ops separate one line's reservation from the next; the edge survives all four
    assert any(e.src == reserved[0] and e.dst == reserved[1] for e in world.edges), world.edges
    assert len(world.edges) >= len(reserved), "every reservation reaches the next"


def test_restrict_says_what_it_stopped_drawing():
    """*Filtering is a projection, never a redaction.* A view that silently omitted nodes would
    be a second, lossy bookkeeper, so selection records them, and every one is still a key the
    checkpoint store answers to.

    Reddens when: `hidden` stops being populated, or a hidden node is also reported as kept."""
    graph = project(from_keys(cart.ORDER_ID, cart_tape()))
    world = restrict(graph, lambda node: kind_of(node.key) == "ledger")
    assert set(world.hidden) | {n.key for n in world.nodes} == {n.key for n in graph.nodes}
    assert not set(world.hidden) & {n.key for n in world.nodes}


def test_a_contracted_edge_carries_the_flow_along_its_path():
    """The count on a contracted edge is the minimum along the way — exact on a chain, which is
    the shape a sequence of steps produces.

    Reddens when: the contraction carries the FIRST hop's count instead of the minimum, which
    over-reports whenever the hidden region narrows."""
    graph = from_keys("r", ["step:a", "step:b", "step:c"])
    graph = RunGraph(
        run_id="r",
        nodes=graph.nodes,
        edges=(Edge("step:a", "step:b", 9), Edge("step:b", "step:c", 4)),
    )
    (edge,) = restrict(graph, lambda node: node.key != "step:b").edges
    assert (edge.src, edge.dst, edge.count) == ("step:a", "step:c", 4)


def test_selection_and_projection_compose_in_either_order():
    """π chooses columns and selection chooses rows, so they are orthogonal and both orders are
    meaningful. Salience is a selection, which is how it composes with the fold.

    Reddens when: either operation depends on the other having run first."""
    graph = from_keys(cart.ORDER_ID, cart_tape())
    ledger_only = lambda node: kind_of(node.key) == "ledger"  # noqa: E731
    fold_then_filter = restrict(project(graph), ledger_only)
    filter_then_fold = project(restrict(graph, ledger_only))

    assert {n.key for n in fold_then_filter.nodes} == {n.key for n in filter_then_fold.nodes}


def test_a_filtered_view_shows_fewer_FACTS_and_that_is_the_difference_from_a_fold():
    """The one promise selection does NOT make. π is `executions`-invariant — fewer nodes, never
    fewer facts, because it re-groups every node. Selection removes rows, so its `executions` is
    smaller, and `hidden` is where the rest went.

    Stated as a test because the two operations are one import apart and the invariant is the
    thing a reader would carry across by mistake."""
    graph = project(from_keys(cart.ORDER_ID, cart_tape()))
    world = restrict(graph, lambda node: kind_of(node.key) == "ledger")
    assert world.executions < graph.executions
    assert restrict(graph, lambda _: True) is graph  # nothing to hide, nothing to rebuild


def test_restrict_never_invents_an_edge_the_run_did_not_take():
    """The walk stops AT a kept node rather than passing through it. Continuing would add a
    transitive edge for every path, so `a -> b -> c` with all three kept would gain a spurious
    `a -> c` — a view claiming control flowed somewhere it never did, which is worse than one
    that hides too much.

    Reddens when: the walk recurses past a kept destination instead of ending there."""
    graph = RunGraph(
        run_id="r",
        nodes=tuple(Node(key=k, kind="step") for k in ("step:a", "step:b", "step:c", "step:x")),
        edges=(Edge("step:a", "step:b"), Edge("step:b", "step:c"), Edge("step:c", "step:x")),
    )
    kept = restrict(graph, lambda node: node.key != "step:x")
    assert {(e.src, e.dst) for e in kept.edges} == {
        ("step:a", "step:b"),
        ("step:b", "step:c"),
    }


# --- the search tree ---------------------------------------------------------------------------


def mcts_tape() -> list[str]:
    _, _, tape = mcts.scripted_run()
    return tape


def test_the_projection_does_NOT_flatten_a_search_tree():
    """Folding collapses the siblings of a fan-out and NOT those of a search, which a tree search
    could not tolerate, and the reason is exact.

    Siblings merge only when they are in bijection with an index. A real search REVISITS a
    promising child, so the candidate a rollout was about is not a function of the round, the
    bijection fails, and every candidate survives by name carrying its visits. The round that
    expands a move and the rounds that select within it are different program points, so a
    candidate appears once under each.

    Reddens when: the closure weakens to plain dependency, which WOULD collapse the siblings."""
    tape = mcts_tape()
    projected = project(from_keys(mcts.ROUTE_ID, tape))
    rollouts = [node for node in projected.nodes if mcts.nodes_in(node.key)]
    assert {name for node in rollouts for name in mcts.nodes_in(node.key)} == {
        stop.name for stop in mcts.ROUTE
    }
    assert sum(node.count for node in rollouts) == len([k for k in tape if mcts.nodes_in(k)])
    assert max(node.count for node in rollouts) > 1, (
        "a search that visited nothing twice is a fan-out"
    )
    # the revisits are self-edges: the search returning to a child it liked
    assert any(edge.src == edge.dst for edge in projected.edges)


def test_a_ledger_append_in_a_loop_must_sit_INSIDE_the_loops_scope():
    """The authoring rule this fixture measured, and the cheapest answer yet to the positional-
    ledger-key question. An authored `event_id` varying per iteration folds when the substrate has
    placed an INDEX somewhere in the key, and not otherwise — so where the append sits decides
    whether any projection can see the axis at all.

    Inside `scoped(t"move:{n}")` the three dispatches are one node with `count=3`. Lift them out
    and they are three nodes that no projection can merge, because nothing in
    `ledger;dispatched:{stop}` is an index. One character of placement, and it needs no change to
    the key grammar, no migration, and it keeps a fork cut nameable.

    Reddens when: the seed admits a NAME-kinded column, which would merge the three by their
    authored ids and make the placement irrelevant — for the wrong reason."""
    tape = mcts_tape()
    inside = len([n for n in project(from_keys("r", tape)).nodes if "dispatched" in n.key])

    hoisted = [
        key.partition(";")[2] if key.startswith("move:") and ";ledger;" in key else key
        for key in tape
    ]
    outside = len([n for n in project(from_keys("r", hoisted)).nodes if "dispatched" in n.key])

    assert inside == 1
    assert outside == len(mcts.ROUTE)


# --- the state machine: a projection that IS a state diagram ------------------------------------


def coding_tape(task=None) -> list[str]:
    if task is None:
        _, _, tape = coding.scripted_run()
        return tape
    handler = RecordingHandler(responses=coding.Answers(task))
    outcome = handler.run(lambda: coding.work(coding.RUN_ID, task))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(coding.APPROVED)
    return [entry.key.stored() for entry in handler.trace]


def test_the_projection_recovers_the_state_diagram_backedges_and_all():
    """The sharpest form of *the fold IS the program*. Nobody declared a state machine anywhere —
    the tape is a flat sequence of placed keys — and projecting it gives the transition graph
    with counts on the edges and both backedges intact.

    `phase:` survives because a phase is not in bijection with the turn: `code` runs at turns 3
    and 5, so the turn determines the phase and the phase does not determine the turn. That is
    the same discrimination that keeps two rubrics apart in the funnel, doing the work a state
    machine needs.

    Reddens when: the closure weakens to dependency, which collapses every phase into one box and
    leaves a state diagram with a single state."""
    projected = project(from_keys(coding.RUN_ID, coding_tape()))
    phases = {p for node in projected.nodes for p in coding.phases_in(node.key)}
    assert phases == set(coding.PHASES), phases
    assert projected.cyclic

    def edge_between(src: str, dst: str) -> bool:
        return any(
            src in coding.phases_in(e.src) and dst in coding.phases_in(e.dst)
            for e in projected.edges
        )

    assert edge_between("test", "code"), "the red-suite backedge"
    assert edge_between("refactor", "test"), "the re-test backedge"
    assert edge_between("test", "finalize"), "and the way out"


def test_selection_reduces_a_long_session_to_its_commitments():
    """What a reviewer of a long debugging session actually wants: the commits and the human's
    word, still connected, with the sandbox work hidden rather than deleted.

    The task is deliberately longer than the scripted one — more red rounds and more refactors —
    because the claim is about SCALE, and a run with two commits would demonstrate nothing."""
    long_task = coding.Task(goal="a long grind", passes_at=24, refactors=3)
    projected = project(from_keys(coding.RUN_ID, coding_tape(long_task)))
    world = restrict(projected, lambda node: kind_of(node.key) in ("ledger", "await"))

    assert len(world.nodes) < len(projected.nodes)
    assert {kind_of(node.key) for node in world.nodes} == {"ledger", "await"}
    assert world.hidden, "the sandbox work is hidden, not deleted"
    # every surviving node is reachable — a filtered view that disconnected would be useless
    assert len(world.edges) >= len(world.nodes) - 1
