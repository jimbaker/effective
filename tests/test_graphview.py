"""The run-graph projection and its cycle fold (`effective.graphview`).

Pure examples throughout: synthetic workflows on the embedded engine. The projection is a fold
over recorded op KEYS, so a synthetic workflow exercises it exactly as any other would.

The two experiments are the last two tests: **legibility at scale** (does a
hundreds-of-nodes run render readably, and does folding fix it?) and **fold cost** (is the
projection a live fold or does it need an indexed read-model?). Both assert, so they cannot rot
into anecdotes.
"""

import time
from pathlib import Path
from uuid import UUID

import pytest
from _keymap import including

from effective.api import append_ledger, ask_llm, await_event, call_tool, gather
from effective.checkpoints import keys, read_sqlite_conn
from effective.combinators import Answered, Deeper, descend, recurse
from effective.cost import MeteredInterpreter, Usage
from effective.domain import CallTool
from effective.graphview import (
    PARKED,
    STATE_PRIORITY,
    Edge,
    Node,
    RunGraph,
    bare_name,
    branch_path,
    fold_cycles,
    format_cost,
    from_keys,
    identity,
    kind_of,
    project,
    strip_branches,
    summarize,
    to_mermaid,
    to_text,
)
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import op_key
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Index, Key, Segment, compose_key
from effective.ops import AwaitEvent, LedgerRow, Step
from effective.parked import ParkedTask, pending_key, read_sqlite_parked_conn
from effective.sqlite import SqliteApp, SqliteLedger

# A hand-built `ParkedTask` for the bridge tests — no engine involved, so the id is a
# placeholder. It read `"t1"` until the task id became a `UUID` on both engines.
_SYNTHETIC_TASK = UUID("019fa000-0000-7000-8000-000000000001")

# --- the projection itself ------------------------------------------------------------------


def test_the_unrolled_graph_is_one_node_per_execution_in_commit_order():
    graph = from_keys("r1", ["a", "b", "a#2"])
    assert [n.key for n in graph.nodes] == ["a", "b", "a#2"]
    assert [(e.src, e.dst) for e in graph.edges] == [("a", "b"), ("b", "a#2")]
    assert graph.cyclic is False  # acyclic BY CONSTRUCTION — the occurrence index is the reason
    assert graph.executions == 3


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("extract", "step"),
        ("ledger;r1:e1", "ledger"),
        ("artifact:app/json:abc", "artifact"),
        ("sleep:2026-01-01", "sleep"),
        ("gather:0,1;ledger;r1:e1", "ledger"),  # a branch's ledger op is a LEDGER node
        ("gather:0,1;tool:a", "step"),
    ],
)
def test_kind_comes_from_the_key_grammar(key, kind):
    """Kind is derived from identity, not annotated beside it — so it cannot drift."""
    assert kind_of(key) == kind


def test_folding_recovers_the_programs_loop_from_the_traces_chain():
    """A trace unrolls a loop into a chain of occurrences; folding the occurrence coordinate
    folds it back into a loop, with the counts kept."""
    trace = ["plan", "ask", "act", "ask#2", "act#2", "ask#3", "ledger;r1:done"]
    folded = fold_cycles(from_keys("r1", trace), drop=(Index,))

    assert [n.key for n in folded.nodes] == ["plan", "ask", "act", "ledger;r1:done"]
    assert folded.cyclic is True  # a genuine cycle, which the unrolled graph cannot contain
    assert {(e.src, e.dst): e.count for e in folded.edges} == {
        ("plan", "ask"): 1,
        ("ask", "act"): 2,
        ("act", "ask"): 2,
        ("ask", "ledger;r1:done"): 1,
    }
    assert folded.executions == len(trace)  # fewer NODES, never fewer FACTS


def test_folding_also_drops_the_branch_coordinate():
    """N identical gather branches are the same repeated shape as N loop iterations — one axis
    over — so the same projection collapses them."""
    trace = ["gather:0,0;fetch", "gather:0,1;fetch", "gather:0,2;fetch", "join"]
    folded = fold_cycles(from_keys("r1", trace), drop=(Index,))

    assert [(n.key, n.count) for n in folded.nodes] == [("gather:*,*;fetch", 3), ("join", 1)]
    assert folded.executions == 4


# --- the third coordinate: a combinator's scope frame ----------------------------------------
#
# A scope frame is dropped by the fold exactly when its coordinate is an UNROLLING INDEX — the
# same kind of coordinate occurrence and branch already are. `recurse` mints `rec:{i}` per leaf
# and `fold:{lv},{k}` per tree-merge group, `descend` mints `d:{depth}` per level, `improve` mints
# `gen:{r}` / `cand:{r},{i}` per round: all counters the program walks, so two of them are two
# executions of one position. `sub:{name}` and `task:{id}` name DISTINCT positions and must
# survive, or two different subagents become one node — the fold reporting fewer facts.
#
# The traces below are recorded from the real combinators rather than typed by hand: the keys are
# what the composers produce, so a change to the frame grammar reaches these tests.


def _recorded(responses, workflow) -> list[str]:
    """The keys a workflow records under the in-memory recorder — the composers' own output."""
    handler = RecordingHandler(responses=responses)
    handler.run(workflow)
    return [entry.key.stored() for entry in handler.trace]


def _recurse_keys() -> list[str]:
    """A two-chunk `recurse`: two leaves under `gather:0,{i};rec:{i};`, one merge under
    `gather:1,0;fold:0,0;`. Both frames on every leaf, neither of them authored."""

    def decompose(_ctx):
        return (yield from call_tool("split", {}, list))

    def leaf(_chunk):
        return (yield from call_tool("work", {}, int))

    def combine(_group):
        return (yield from call_tool("merge", {}, int))

    return _recorded(
        {"tool:split": ["a", "b"], "tool:work": 1, "tool:merge": 2},
        lambda: recurse("ctx", decompose=decompose, leaf=leaf, combine=combine, fanin=2),
    )


def _descend_keys() -> list[str]:
    """A `descend` drilling until its budget is exhausted — one judge per level, each under its
    own `d:{depth}`."""

    def judge(ctx, level):
        value = yield from call_tool("judge", {}, int)
        return Answered(value) if level.final else Deeper(ctx)

    return _recorded({"tool:judge": 7}, lambda: descend("ctx", judge, budget=3))


def test_a_recurse_trace_folds_its_scope_frames_as_well_as_its_branches():
    """The `recurse` case: two leaves are two executions of ONE position, and the gather
    coordinate alone does not say so; `rec:{i}` distinguishes them just as hard."""
    trace = _recurse_keys()
    assert trace == [
        "step;tool:split",
        "gather:0,0;rec:0;step;tool:work",
        "gather:0,1;rec:1;step;tool:work",
        "gather:1,0;fold:0,0;step;tool:merge",
    ]

    folded = fold_cycles(from_keys("r1", trace), drop=(Index,))
    assert [(n.key, n.count) for n in folded.nodes] == [
        ("step;tool:split", 1),
        ("gather:*,*;rec:*;step;tool:work", 2),
        ("gather:*,*;fold:*,*;step;tool:merge", 1),
    ]
    assert folded.executions == len(trace)  # fewer NODES, never fewer FACTS


def test_a_descend_drill_folds_back_into_the_loop_it_unrolled():
    """The `descend` half: a drill is a loop the trace unrolled along depth, so it folds into a
    self-edge — the same sentence as `fold_cycles`' headline, read one coordinate over."""
    trace = _descend_keys()
    assert trace == [f"d:{depth};step;tool:judge" for depth in range(4)]

    folded = fold_cycles(from_keys("r1", trace), drop=(Index,))
    assert [(n.key, n.count) for n in folded.nodes] == [("d:*;step;tool:judge", 4)]
    assert folded.cyclic is True  # `judge -> judge`, the drill's own shape
    assert folded.executions == len(trace)


def test_a_folded_scope_still_unrolls_to_the_keys_it_came_from():
    """The fold is a lens, not a summary — dropping a third coordinate must not cost the
    drill-down that makes that true."""
    trace = _descend_keys()
    (node,) = fold_cycles(from_keys("r1", trace), drop=(Index,)).nodes
    assert node.members == tuple(trace)


FRAME_STRUCTURE = (
    (
        "gather nested in a gather",
        ["gather:0,0;gather:0,1;tool:a", "gather:0,1;tool:a"],
        ["gather:*,*;gather:*,*;tool:a", "gather:*,*;tool:a"],
    ),
    (
        "gather outside a scope, and inside one",
        ["gather:0,1;rec:1;step;tool:a", "rec:0;gather:0,0;step;tool:a"],
        ["gather:*,*;rec:*;step;tool:a", "rec:*;gather:*,*;step;tool:a"],
    ),
    (
        "a branch's park, and the same await outside one",
        ["gather:0,0;$awaitEvent:ask:q", "$awaitEvent:ask:q"],
        ["gather:*,*;$awaitEvent:ask:*", "$awaitEvent:ask:*"],
    ),
)
"""Tapes whose two keys carry the same coordinates in different FRAME STRUCTURE."""


@pytest.mark.parametrize(
    ("tape", "classes"),
    [(tape, classes) for _id, tape, classes in FRAME_STRUCTURE],
    ids=[i for i, _t, _c in FRAME_STRUCTURE],
)
def test_keys_whose_frames_differ_in_structure_land_in_different_classes(tape, classes):
    """A drop set names coordinates, so the frames carrying them survive the quotient.

    `recurse` puts its gather outside the scope and a `scoped(...)` around a `gather` puts it
    inside; one walk over the leading terms reaches both, where two stacked passes over anchored
    patterns would be a union to keep in sync.

    `separated` is not this word: that one asks whether two VARIANTS' languages are disjoint, so
    no key decodes as both. This asks which keys a quotient sends to one class."""
    folded = fold_cycles(from_keys("r1", tape), drop=(Index,))
    assert [node.key for node in folded.nodes] == classes
    assert folded.executions == len(tape)


def test_a_scope_the_fold_must_KEEP_does_not_hide_the_ones_it_must_drop():
    """A kept frame on the outside does not stop the walk.

    Both rows of the interleave test above open with a droppable tag, so they exercise only the
    prefix-droppable path. The shape every bench trial runs puts a frame that must be KEPT on the
    outside: `agent.contrastbench` wraps a whole trial in `task:{id}`, and a `descend`/`recurse`
    mints its frames INSIDE that. A walk that stops at the first frame it cannot drop lets that
    one frame hide every frame behind it, including the gather branch, while `dropped` goes on
    reporting `"branch"`."""
    drill = [f"task:t1;d:{depth};step;tool:judge" for depth in range(4)]
    assert [
        (n.key, n.count) for n in fold_cycles(from_keys("r1", drill), drop=(Index,)).nodes
    ] == [
        ("task:t1;d:*;step;tool:judge", 4)  # the task scope survives; the depth does not
    ]

    branches = [
        "task:t1;gather:0,0;rec:0;step;tool:work",
        "task:t1;gather:0,1;rec:1;step;tool:work",
    ]
    assert [
        (n.key, n.count) for n in fold_cycles(from_keys("r1", branches), drop=(Index,)).nodes
    ] == [("task:t1;gather:*,*;rec:*;step;tool:work", 2)]


def test_the_published_example_folds(capsys):
    """A real-corpus pin over the published showcase — the file where the truncating walk was most
    visible: 19 nodes folded to 19, so a reader of the exemplar saw a fan of identical boxes where
    the program has a loop.

    **The tape is RUN here, not read from `EXPECTED_TAPE`.** The literal is a fine pin for the
    example's own test, which asserts the two are equal and therefore breaks loudly when a
    convention moves — but this test is about the FOLD, and reading a literal would make it a
    second snapshot to maintain. Driving the workflow means the keys are whatever the composers
    mint today: the conventions are still moving (pre-alpha, deliberately), and a live tape tracks
    them for free where a copied one would need regenerating.

    Loaded by PATH rather than imported: `examples/` is not on the package path (nothing else in
    `tests/` imports an example — they read the files as text), and putting it there to satisfy one
    test would make `ty` resolve a module the package layout does not contain."""
    import _funnel as funnel

    handler = RecordingHandler(responses=funnel.Answers(funnel.SUBMISSIONS))
    outcome = handler.run(lambda: funnel.triage("cfp-2026"))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume("uphold")  # each in-lane park; the tape completes after them
    tape = [entry.key.stored() for entry in handler.trace]

    unrolled = from_keys("cfp", tape)
    folded = fold_cycles(unrolled, drop=(Index,), keymap=including("tests/_funnel.py"))
    assert len(folded.nodes) < len(unrolled.nodes), [n.key for n in folded.nodes]
    assert folded.executions == len(tape)  # fewer NODES, never fewer FACTS
    # The `recurse` under `rank` collapses on the substrate's own coordinates alone.
    assert any(node.count > 1 for node in folded.nodes), [(n.key, n.count) for n in folded.nodes]

    # `talk:` is a consumer's tag, so the substrate's registry does not carry it and the fixture
    # declares `Index` at its own mint, handing the fold a map built from its own file. The three
    # lanes are one equivalence class under that map and three under the shipped one.
    assert any("talk:*" in node.key for node in folded.nodes)
    without_the_map = fold_cycles(unrolled, drop=(Index,))
    assert len(folded.nodes) < len(without_the_map.nodes), "the map is what folds the lanes"
    assert without_the_map.executions == folded.executions == len(tape)
    with capsys.disabled():
        print(
            f"\nfunnel_triage (live tape): {len(unrolled.nodes)} unrolled -> "
            f"{len(without_the_map.nodes)} on the substrate's roles alone -> "
            f"{len(folded.nodes)} with the fixture's own"
        )


def test_the_kind_of_a_scoped_ledger_node_survives_the_frames_recurse_actually_mints():
    """`bare_name` alternates `unframed` and `strip_branches` to a FIXED POINT, and one pass is
    not enough for the shape a `recurse` branch produces.

    `unframed` returns from the earliest boundary starting a known tag, and `gather:` is in its
    vocabulary — so `gather:0,0;rec:0;ledger;x:1` came back whole and `strip_branches` peeled only
    the gather, leaving a key that starts with no kind tag. Consequences, both real: the node drew
    as a plain rectangle instead of a ledger cylinder, and `ledger_collisions` — the detector for
    *two writers, one `event_id`, one row silently lost* — reported nothing for it.

    The pin that existed used `rec:0;gather:0,0;…`. `_recurse_keys()` in this file shows the real
    order is the other one, which is the wrong-target shape: the instrument measured a key the
    combinator does not mint."""
    from effective.graphview import ledger_collisions

    minted = ["gather:0,0;rec:0;ledger;x:1", "gather:0,1;rec:1;ledger;x:1"]
    assert bare_name(minted[0]) == "ledger;x:1"
    assert kind_of(minted[0]) == "ledger"  # was: "step"

    (collision,) = ledger_collisions(from_keys("r", minted))  # was: ()
    assert collision.event_id == "x:1"
    assert collision.writers == tuple(minted)

    # A `recurse` INSIDE a `recurse` needs THREE passes, and two is what a plausible fix writes.
    # Pinned because the mutation to exactly-two-passes left all 2620 tests green: every other key
    # here reaches its fixed point in two, so nothing distinguished the loop from a pair.
    nested = "gather:0,0;rec:0;gather:0,1;rec:1;ledger;x:1"
    assert bare_name(nested) == "ledger;x:1"
    assert kind_of(nested) == "ledger"


def test_the_normalizer_stops_at_the_STEP_arm_so_an_authored_name_is_not_read_as_substrate():
    """`grammar.STEP_ARM` — *the one arm no structural rule may read past* — applied here, because
    `bare_name` was reading past it and `ledger_collisions` now trusts this normalizer.

    A step may legally be named with structure (`step_key` splices a structured name as terms), so
    a workflow may have a step called `ledger;dup:m1`. Read past the `step` arm and that becomes
    `ledger;dup:m1`: kind `ledger`, and two ordinary runs of one step reported as a LOST LEDGER
    ROW. A false positive on the two-bookkeepers detector is worse than a miss — it sends a reader
    hunting a data-loss bug that did not happen."""
    from effective.graphview import ledger_collisions

    authored = "step;ledger;dup:m1"
    assert bare_name(authored) == authored  # not `ledger;dup:m1`
    assert kind_of(authored) == "step"
    assert ledger_collisions(from_keys("r", [authored, f"{authored}#2"])) == ()

    # ...and the stop is at the arm, not at the string: frames OUTSIDE it still peel.
    assert bare_name("gather:0,0;rec:0;step;ledger;dup:m1") == authored


def test_a_frame_must_FRAME_something_to_be_dropped():
    """A bare `gather` qualifier carries no coordinate, so it is a term in its own right, not a
    frame, and `past_frames` keeps it.

    The substrate cannot reach this (`op_key(Gather())` raises; `gather_frame` always composes
    `{g},{i}`), and it is pinned anyway because nothing else in the suite sees it: without the
    coordinate test, BOTH halves of a refusal move with no test going red. The composer accepts
    `event;gather;foo` at a `domain=address` splice and `unmintable` stops reporting it."""
    from effective.keys.frame import past_frames, unmintable

    assert past_frames("gather;foo") == "gather;foo"  # kept: no coordinate, so it frames nothing
    assert past_frames("gather:0,0;foo") == "foo"  # dropped: it frames `foo`
    assert unmintable("event;gather;budget-grant:pass-n=m1") is not None


def test_the_fold_names_the_third_axis_it_drops_AND_actually_drops_it():
    """`dropped` is the projection's own statement of what it threw away, and a reader who trusts
    it while a coordinate silently survives is the failure this whole arc is about.

    So both halves, in one test on purpose. A mutation round caught the first draft asserting the
    label alone — which a fold that dropped NOTHING would also satisfy, since `dropped` is a
    literal on the returned graph."""
    folded = fold_cycles(from_keys("r1", ["d:0;step;tool:a", "d:1;step;tool:a"]), drop=(Index,))
    # The axes a ROLE quotient drops are the roles themselves. `branch` and `unrolling-scope` were
    # the table's names for two things one role now covers: a gather's coordinate and a scope's
    # are both indices, and saying so twice was the table having no word for what they share.
    assert folded.dropped == ("occurrence", "index")
    assert [(n.key, n.count) for n in folded.nodes] == [("d:*;step;tool:a", 2)]


def test_a_scope_that_NAMES_rather_than_counts_is_not_folded():
    """The load-bearing negative, and the reason the fold does not simply reuse `bare_name`.

    `sub:{name}` is a subagent's name and `task:{id}` a bench task's — distinct program positions,
    not two turns of one. Folding them would merge two different subagents into one node, which is
    the fold reporting fewer FACTS rather than fewer nodes. Green before and after the scope fold:
    it pins the boundary, not the change."""
    subagents = ["sub:planner;step;tool:ask", "sub:critic;step;tool:ask"]
    assert len(fold_cycles(from_keys("r1", subagents), drop=(Index,)).nodes) == 2

    tasks = ["task:t1;step;tool:ask", "task:t2;step;tool:ask"]
    assert len(fold_cycles(from_keys("r1", tasks), drop=(Index,)).nodes) == 2


def test_a_scope_with_no_coordinate_is_not_folded():
    """`improve`'s `seed` frame carries nothing to index, so there is no second execution it could
    be confused with — and the seed IS scored at a different program position from a round's
    candidate. It needs no ruling; a coordinate-free term simply is not an unrolling one."""
    trace = ["seed;step:score", "cand:0,0;step:score"]
    assert "seed;step:score" in [
        n.key for n in fold_cycles(from_keys("r1", trace), drop=(Index,)).nodes
    ]


def test_a_scope_the_map_cannot_resolve_is_KEPT_and_hides_nothing_behind_it():
    """A stranger's `scoped(compose_key(t"vendor:{id}"))` is minted outside this repo, so the map
    holds no variant for it and no role to read. The fold keeps a coordinate it cannot resolve
    rather than merging positions that differ, which is what lets `examples/` and any downstream
    consumer mint a scope the substrate never heard of.

    It also does not block what sits behind it: the walk continues past a term it could not
    project, so an `Index` nested inside a stranger's scope still folds. The walk once truncated
    there, letting one unresolved frame keep every frame behind it."""
    unruled = ["vendor:7;step;tool:a", "vendor:8;step;tool:a"]
    assert len(fold_cycles(from_keys("r1", unruled), drop=(Index,)).nodes) == 2

    nested = ["vendor:7;d:0;step;tool:a", "vendor:7;d:1;step;tool:a"]
    assert [
        (n.key, n.count) for n in fold_cycles(from_keys("r1", nested), drop=(Index,)).nodes
    ] == [("vendor:7;d:*;step;tool:a", 2)]


def test_the_fold_is_TOTAL_over_what_callers_hand_it_not_only_over_the_language():
    """The regression that decided how the drop is implemented, and it is not hypothetical: it
    reddened `test_experiment_legibility_at_three_scales` before it was a pin of its own.

    `effective.lineage.canonical` rewrites a scope token to the hole `{run}`, deliberately not in
    the key language, because it marks the place a token was scrubbed. A drop that requires a valid
    `parse` returns such a key unchanged, silently keeping the gather coordinate the fold has just
    reported as `dropped`. Nothing about a frame boundary needs the atoms to validate."""
    from effective.lineage import canonical

    scrubbed = canonical(
        ["gather:0,1;ledger;r-fan:m", "gather:0,2;ledger;r-fan:m"], scrub=["r-fan"]
    )
    assert scrubbed == ("gather:0,1;ledger;{run}:m", "gather:0,2;ledger;{run}:m")
    assert [
        (n.key, n.count) for n in fold_cycles(from_keys("r", scrubbed), drop=(Index,)).nodes
    ] == [("gather:*,*;ledger;{run}:m", 2)]


def test_dropping_the_frames_never_rewrites_the_key_that_survives():
    """Dropping a frame in front of a park must leave the park's own bytes alone.

    The walk parses and re-renders, so `render` has to be the identity on a foreign term that is
    not first — otherwise this would fold `gather:0,0;$awaitEvent:ask:q` onto bytes Absurd never
    wrote, and the two would still collapse to one node while the key was wrong.
    """
    folded = fold_cycles(
        from_keys("r", ["gather:0,0;$awaitEvent:ask:q", "$awaitEvent:ask:q"]), drop=(Index,)
    )
    assert [(n.key, n.count) for n in folded.nodes] == [
        ("gather:*,*;$awaitEvent:ask:*", 1),
        ("$awaitEvent:ask:*", 1),
    ]


@pytest.mark.parametrize(
    "key",
    [
        "ledger;done:m1",
        "step;tool:fetch",
        "artifact:message/rfc822,sha256-9f2c",  # a `/` PATH inside a coordinate
        "$awaitEvent:ask:q",  # Absurd's foreign tag
        "event;gather:0,0;ev:r1",  # a branch coordinate INSIDE the arm, not leading
        "depth-grant:r,0,0",
        "plan",
    ],
)
@pytest.mark.parametrize("frames", ["", "gather:0,0;", "gather:0,0;gather:1,1;", "rec:0;"])
def test_the_frame_walk_agrees_with_the_anchored_pattern_it_replaced(key, frames):
    """A differential over the shapes a real graph carries, so the new walk cannot quietly change
    an answer `strip_branches` was already giving. `rec:0;` is a frame neither strips: `rec` is
    no frame arm, so both leave it in place, and a walk that began dropping it would differ."""
    from effective.keys.frame import FRAME_ARMS, past_frames

    framed = frames + key
    assert past_frames(framed, drop=FRAME_ARMS) == strip_branches(framed)


def test_a_parked_state_survives_into_the_view():
    graph = from_keys("r1", ["a", "b"], states={"b": "parked"})
    assert to_mermaid(graph).count("(parked)") == 1


# --- the pending node -----------------------------------------------------------------------
#
# Each of these fails against a projection with no pending-node bridge.


def _park(wake_event: str) -> ParkedTask:
    """A park record shaped like a reader's, with only the field the bridge reads filled in."""
    return ParkedTask(task_id=_SYNTHETIC_TASK, task_name="wf", wake_event=wake_event)


def test_a_bare_wake_registration_would_render_as_a_rectangle_which_is_why_the_key_is_tagged():
    """Why the pending key carries the `event;` tag.

    `kind_of` derives kind from the key alone (that is what stops kind drifting from identity), so
    a pending node keyed on the RAW registration is a `step`: it renders as a rectangle, and the
    hexagon, the shape the click-to-answer interaction targets, never appears. Silent: the
    diagram renders fine, wrongly."""
    assert kind_of("review:m1") == "step"  # the raw registration, as the reader reports it
    assert to_mermaid(from_keys("r1", ["a"], pending=Key.parse("review:m1"))).count("{{") == 0

    assert pending_key(_park("review:m1")).stored() == "event;review:m1"
    assert kind_of(pending_key(_park("review:m1")).display()) == "await"
    rendered = to_mermaid(from_keys("r1", ["a"], pending=pending_key(_park("review:m1"))))
    assert rendered.count("{{") == 1  # a hexagon: the run can stop here
    assert "(parked)" in rendered
    assert "event;review:m1" in rendered


def test_the_pending_key_is_exactly_the_op_key_of_the_await():
    """The spelling is the identity the op already has, so the pending node and the op it stands
    for are the same address, and the source map decodes it."""
    assert pending_key(_park("review:m1")) == op_key(
        AwaitEvent(name=Key.parse("review:m1"), schema=dict)
    )


def test_a_pending_node_cannot_collide_with_a_recorded_key():
    """Disjoint BY CONSTRUCTION, with no guard.

    `to_mermaid` ids nodes by key, so a synthesized key equal to a recorded one emits two node
    lines sharing an id and a self-edge, silently wrong. It cannot arise: a Step's key opens
    with its OWN arm, so an author naming a step `event;review:m1` gets `step;event;review:m1`,
    which is not an await's key and cannot be, whatever the author writes. **Disjoint by
    construction beats a list somebody has to keep complete**, which is why this asserts "the key
    it composes is not the one it imitates" rather than "the name is refused"."""
    for imitation in ("event;review:m1", "event;x"):
        composed = op_key(Step(name=imitation, op=CallTool(name="x", result_schema=int)))
        assert composed.stored() != imitation
        assert composed.stored() == f"step;{imitation}"


def test_states_alone_cannot_introduce_the_pending_node():
    """Why `pending` is its own parameter. `states` maps a key that is already in `keys` to a
    state; an entry for a key that is not there is silently ignored — so the existing seam can
    decorate the pending node but cannot produce it, and a caller who assumed otherwise would get
    a graph with no await and no error."""
    ignored = from_keys("r1", ["a", "b"], states={"event;review:r1": "parked"})
    assert [n.key for n in ignored.nodes] == ["a", "b"]
    assert "(parked)" not in to_mermaid(ignored)

    graph = from_keys("r1", ["a", "b"], pending=Key.parse("event;review:r1"))
    assert [n.key for n in graph.nodes] == ["a", "b", "event;review:r1"]
    assert [(e.src, e.dst) for e in graph.edges] == [("a", "b"), ("b", "event;review:r1")]
    assert graph.nodes[-1].state == PARKED  # the default, and the whole point of the parameter
    # the state still rides the existing seam when a caller wants to say something else
    assert (
        from_keys("r1", ["a"], pending=Key.parse("event;e"), states={"event;e": "refused"})
        .nodes[-1]
        .state
        == "refused"
    )


def test_a_pending_node_is_the_only_difference_from_a_finished_runs_graph():
    """Everything else about the projection is unchanged — a caller adds one argument."""
    finished = from_keys("r1", ["a", "b"])
    parked = from_keys("r1", ["a", "b"], pending=Key.parse("event;review:r1"))
    assert parked.nodes[: len(finished.nodes)] == finished.nodes
    assert parked.edges[: len(finished.edges)] == finished.edges
    assert parked.executions == finished.executions + 1


def test_a_pending_node_on_an_empty_run_has_no_dangling_edge():
    graph = from_keys("r1", [], pending=Key.parse("event;review:r1"))
    assert [n.key for n in graph.nodes] == ["event;review:r1"]
    assert graph.edges == ()


def test_parked_wins_the_fold_over_committed():
    """A folded node is parked when any member is, in the cycle view that is the default.

    An agent loop parked at its third `review` unrolls to `review, review#2, review#3(parked)`.
    First-node-wins would fold that to a committed node, so the 84x-legible cycle view would be
    precisely the view in which a parked loop looked finished."""
    trace = ["plan", "review", "review#2"]
    graph = from_keys(
        "r1", trace, pending=Key.parse("review#3")
    )  # keyed bare here to isolate the FOLD
    folded = fold_cycles(graph, drop=(Index,))

    (node,) = [n for n in folded.nodes if n.key == "review"]
    assert (node.count, node.state) == (3, PARKED)
    assert "(parked)" in to_mermaid(folded)
    assert folded.executions == 4  # fewer nodes, never fewer facts
    assert node.members == ("review", "review#2", "review#3")  # the drill-down still lands


def test_an_unranked_state_outranks_every_ranked_one_in_the_fold():
    """The priority is extensible in the safe direction: a state the fold has no vocabulary for
    surfaces. Two such states, `refused` and `not-reached`, have no producer yet; when they grow
    one, they show up without a code change here, and their relative order can be decided then."""
    assert STATE_PRIORITY == ("committed", "parked")
    folded = fold_cycles(from_keys("r1", ["a", "a#2"], states={"a#2": "refused"}), drop=(Index,))
    assert [(n.key, n.state) for n in folded.nodes] == [("a", "refused")]


def test_a_parked_gather_branch_keeps_its_branch_coordinate_inside_the_tag():
    """The branch coordinate sits under the `event;` tag, where `strip_branches` cannot reach it.

    The engines carry a gather branch's coordinate INSIDE the awaited event name
    (`_PrefixedCtx.await_event` prepends `gather:{g},{i};` to the name it registers), not around
    the op key, so a park reader reports `gather:0,0;ev:r1` and the composed pending key is
    `event;gather:0,0;ev:r1`, never `gather:0,0;event;ev:r1`.

    The node is still a hexagon, `kind_of` reading the leading tag. The quotient reaches the
    coordinate all the same: `event;{}` splices its payload, so the projection recurses into the
    name and finds a `gather:` whose coordinates declare `Index`. `strip_branches` is anchored
    `^gather:`, which the third assertion below pins."""
    key = pending_key(_park("gather:0,0;ev:r1"))
    assert key.stored() == "event;gather:0,0;ev:r1"
    assert kind_of(key.display()) == "await"  # still a hexagon
    assert strip_branches(key.display()) == key.display()  # ...the coordinate stays put

    folded = fold_cycles(
        from_keys("r1", ["gather:0,0;tool:a", "gather:0,1;tool:a"], pending=key), drop=(Index,)
    )
    assert [(n.key, n.count, n.state) for n in folded.nodes] == [
        ("gather:*,*;tool:a", 2, "committed"),  # the recorded branch keys fold as always...
        ("event;gather:*,*;ev:r1", 1, PARKED),  # ...and so does the park, inside its tag
    ]


def test_mermaid_shapes_and_counts_render():
    folded = fold_cycles(from_keys("r1", ["a", "a#2", "ledger;r1:x"]), drop=(Index,))
    out = to_mermaid(folded, direction="LR", title="run r1")
    assert "graph LR" in out
    assert "title: run r1" in out
    assert '"a x2"' in out  # the count rides the label
    assert "[(" in out  # a ledger node is a cylinder
    assert '-->|"x1"|' not in out  # a once-taken edge carries no noise


def test_the_projection_is_pure_and_engine_free():
    """No DB, no engine, no `agent` import — `effective.graphview` is a fold over strings, which
    is what lets the same code serve both engines and a recorded in-process trace."""
    import effective.graphview as gv

    source = (gv.__file__ or "").replace("graphview.py", "")
    assert source  # sanity
    module_text = open(gv.__file__).read()  # noqa: SIM115 - a one-shot read in a test
    assert "import agent" not in module_text
    assert "sqlite3" not in module_text
    assert "psycopg" not in module_text


# --- the workflows the experiments run (pure, synthetic) ------------------------------------


def _domain():
    return MeteredInterpreter(llm=lambda _op: ({"answer": "ans"}, Usage()), tools=lambda _op: 7)


def small_wf(message_id: str):
    """~10 nodes: the decision shape, which is a step, a ledger row, a gate and a branch.

    Every name it authors is scoped on the run's SUBJECT, never on its run id. That makes it
    forkable, and is why it is spelled exactly like `test_fork_sweep.decision_wf`: *the run you
    look at and the run you fork should be the same run*.
    `test_the_run_you_look_at_is_a_run_you_can_fork` holds the alignment as a test. The run id
    scopes the ledger ROW, as a column."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    decision = yield from await_event(f"review:{message_id}", dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"reviewed:{Segment(message_id)}"), kind="reviewed")
    )
    if decision["decision"] == "approve":
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"committed:{Segment(message_id)}"), kind="committed")
        )
    return decision["decision"]


def agent_loop_wf(message_id: str, iterations: int = 100):
    """Hundreds of nodes: the agent-loop shape — the same three ops, over and over, which is
    exactly the trace whose legibility the experiment is about.

    Subject-scoped like its siblings, though this one has **no await and therefore no fork
    point** — a shape a fork cannot cut at all, whatever it names its rows.

    The event id is `iteration:`, not `step:`: `step` is an ARM the substrate owns, and an
    `event_id` (declared `domain=address`) that opens with an op arm is refused, because nothing
    would ever mint the key it composes."""
    for _ in range(iterations):
        yield from ask_llm("think", [], dict)
        yield from call_tool("search", {}, int)
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"iteration:{Segment(message_id)}"), kind="step")
        )
    return "done"


def fan_wf(message_id: str, width: int = 4):
    """A gather-bearing shape, so the branch coordinate is exercised on a real run. No await
    here either, so like `agent_loop_wf` it is un-forkable structurally, not by its naming."""

    def branch(i: int):
        def thunk():
            value = yield from call_tool("fetch", {}, int)
            yield from append_ledger(
                LedgerRow(event_id=Key.parse(f"b{i}:{message_id}"), kind="branch")
            )
            return value

        return thunk

    results = yield from gather([branch(i) for i in range(width)])
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"joined:{Segment(message_id)}"), kind="joined")
    )
    return results


def _run(
    app: SqliteApp,
    name: str,
    workflow,
    run_id: str,
    message_id: str,
    *,
    answer=None,
    commit: bool = True,
) -> UUID:
    """`commit=False` runs with NO ledger writer: the durable no-commit mode `_record_ledger`
    has for counterfactuals.

    It exists for `agent_loop_wf`, which appends `iteration:{message_id}` on all 100 iterations:
    one authored `event_id`, a hundred placed writers, so the canonical record would keep ONE row
    and lose 99, silently, because every test here reads the graph and none reads the ledger. It
    is a GRAPH fixture; the keys are the point, and a fixture should make no canonical-record
    claim it does not mean.

    A distinct id per iteration, which is what a real loop must write, BREAKS the experiment this
    file exists to run. Measured: the fold goes from 300 nodes -> 3 to 300 -> 102, because
    `fold_cycles` recovers a program's shape from OCCURRENCE and a distinct authored name is not
    one. That the ledger requires what the projection cannot fold is an open design question.
    Checkpoint keys are identical either way, since `_record_ledger` checkpoints when
    `ledger is None`, so the measurement is untouched.

    **The scope fold leaves this open.** The two quotients that disagree here are the fold's and
    the ledger's. The fold drops a combinator's scope frame, which covers a coordinate that is the
    SUBSTRATE's: the `Index` role the combinator declares at its mint. The coordinate above is the
    DOMAIN's, an authored `event_id` varying per iteration, and no projection may drop that
    because the record cannot. Positional ledger keys would."""

    @app.register_task(name)
    def task(params, ctx):
        ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock) if commit else None
        return DurableHandler(ctx, _domain(), ledger=ledger).run(
            lambda: workflow(params["message_id"])
        )

    task_id = app.spawn(name, {"run_id": run_id, "message_id": message_id})
    _no_failure(app.run_until_result(task_id, max_batches=8))
    if answer is not None:
        app.emit_event(f"review:{message_id}", answer)
        _no_failure(app.run_until_result(task_id, max_batches=8))
    return task_id


def _no_failure(snapshot):
    """A FAILED task is never what any of these tests set up, so say so here.

    Without this the helper returned a task id whatever happened, and a workflow that raised
    read downstream as a SHORT RUN: `agent_loop_wf` died on its first ledger append and the
    experiment reported `2 == 300` instead of the ValueError that caused it. A parked task is
    still fine — several callers depend on it — so this checks only for failure."""
    assert snapshot is None or snapshot.state != "failed", snapshot
    return snapshot


def keys_of(graph) -> list[str]:
    return [node.key for node in graph.nodes]


# --- experiment 1: legibility at scale -------------------------------------------------------


def test_experiment_legibility_at_three_scales(tmp_path, capsys, sqlite_app):
    """Does a real trace render readably as Mermaid, and does folding rescue the big one?

    Measured 2026-07-25:

    | run | unrolled nodes | folded nodes | ratio |
    |---|---|---|---|
    | decision (small) | 4 | 4 | 1.0x |
    | agent loop x100 | 300 | 3 | 100x |
    | gather fan (4) | 9 | 6 (3 after a scrub) | 1.5x / 3x |

    The conclusion the experiment was run to reach: **folding is the collapse rule.** A 300-node
    unrolled trace is unreadable as a diagram and a 3-node cycle graph with x100 on the edges is
    readable. Nothing bespoke was needed: a long run is a repeated shape, and the fold is the
    projection that recognizes repetition."""
    app = sqlite_app(str(tmp_path / "scale.db"))
    graphs = []
    for name, workflow, run_id, answer, commit in (
        ("small", small_wf, "r-small", {"decision": "approve"}, True),
        # `commit=False` for the loop ONLY — see `_run`. It appends one authored `event_id` on
        # every iteration, which keeps 1 canonical row of 100; the keys it draws are unchanged.
        ("loop", agent_loop_wf, "r-loop", None, False),
        ("fan", fan_wf, "r-fan", None, True),
    ):
        task_id = _run(app, name, workflow, run_id, f"m-{name}", answer=answer, commit=commit)
        graph = from_keys(run_id, list(keys(read_sqlite_conn(app.conn, task_id))))
        graphs.append((name, graph, fold_cycles(graph, drop=(Index,))))

    report = ["", "| run | unrolled nodes | folded nodes | mermaid chars (unrolled -> folded) |"]
    report.append("|---|---|---|---|")
    for name, unrolled, folded in graphs:
        report.append(
            f"| {name} | {len(unrolled.nodes)} | {len(folded.nodes)} | "
            f"{len(to_mermaid(unrolled))} -> {len(to_mermaid(folded))} |"
        )
    with capsys.disabled():
        print("\n".join(report))
        print(summarize([g for _, g, _ in graphs] + [f for _, _, f in graphs]))
        print("\n--- the loop run, folded ---")
        print(to_mermaid(next(f for n, _, f in graphs if n == "loop"), direction="LR"))

    by_name = {name: (unrolled, folded) for name, unrolled, folded in graphs}
    loop_unrolled, loop_folded = by_name["loop"]
    assert len(loop_unrolled.nodes) == 300  # three ops x 100 iterations
    assert len(loop_folded.nodes) == 3  # ...and three boxes with counts
    assert loop_folded.cyclic is True
    assert loop_folded.executions == 300  # the facts survive the fold
    assert len(to_mermaid(loop_folded)) < len(to_mermaid(loop_unrolled)) / 20

    fan_unrolled, fan_folded = by_name["fan"]
    assert len(fan_unrolled.nodes) == 9  # 4 branches x (tool + ledger) + the join row
    # SIX, not three — and this is the experiment's second finding. The branch COORDINATE folds
    # (four `gather:0,i;tool:fetch` become one `tool:fetch` x4), but each branch's ledger row was
    # given a branch-varying event id by its author (`b0:{subject}`…`b3:{subject}`), and the fold
    # works on
    # the KEY. Data-dependent names are different nodes, because as ledger rows they really are.
    assert len(fan_folded.nodes) == 6
    assert next(n.count for n in fan_folded.nodes if n.key == "gather:*,*;step;tool:fetch") == 4

    # The remedy already exists and is the same one cross-run alignment uses: canonicalize the
    # varying tokens first (`effective.lineage.canonical`), then fold. That is a CALLER's choice:
    # only the caller knows which tokens are scope and which are content.
    from effective.lineage import canonical

    scrubbed = fold_cycles(
        from_keys(
            "r-fan", list(canonical(keys_of(fan_unrolled), scrub=[f"b{i}" for i in range(4)]))
        )
    )
    assert len(scrubbed.nodes) == 3  # fetch, the branch row, the join row

    small_unrolled, small_folded = by_name["small"]
    assert len(small_folded.nodes) == len(small_unrolled.nodes)  # nothing repeats: fold is a no-op


# --- experiment 2: fold cost -----------------------------------------------------------------


def test_experiment_fold_cost_is_a_live_fold_not_a_read_model(capsys):
    """Is the projection cheap enough to compute on demand, or does it need an indexed table?

    Measured 2026-07-25 over synthetic key sequences (the ADR quotes these): a 10,000-op run
    projects and folds in single-digit milliseconds, which settles the question — **a live fold,
    no read-model, no schema**. The read from the engine dominates, and that read is one indexed
    SELECT the fork path already does.

    The assertion is a 2-SECOND ceiling, not a tight one — and that number is the second lesson.
    A 100 ms bound looked "generously loose" and flaked on the third consecutive suite run under
    load. What the test is for is catching an accidental QUADRATIC, and at 10k ops a quadratic is
    ~100M operations: seconds, not milliseconds. So the ceiling only has to separate those two
    regimes, and anything tighter is measuring the machine."""
    sizes = [100, 1_000, 10_000]
    rows = ["", "| ops | from_keys (ms) | fold_cycles (ms) | folded nodes |", "|---|---|---|---|"]
    for size in sizes:
        trace = [
            f"{op}{'' if i == 0 else f'#{i + 1}'}"
            for i in range(size // 3)
            for op in ("think", "search", "write")
        ]
        start = time.perf_counter()
        graph = from_keys("r", trace)
        built = time.perf_counter()
        folded = fold_cycles(graph, drop=(Index,))
        done = time.perf_counter()
        rows.append(
            f"| {len(trace)} | {(built - start) * 1000:.1f} | {(done - built) * 1000:.1f} | "
            f"{len(folded.nodes)} |"
        )
        if len(trace) >= 9_000:
            assert (done - start) < 2.0, f"{len(trace)} ops took {(done - start) * 1000:.0f} ms"
            assert len(folded.nodes) == 3
    with capsys.disabled():
        print("\n".join(rows))


def test_the_projection_types_are_hashable_and_frozen():
    """Small but load-bearing: a view is data, so it can be cached, compared and sent to a
    renderer without anyone worrying about who owns it."""
    assert hash(Node(key="a", kind="step"))
    assert hash(Edge("a", "b"))
    assert RunGraph("r") == RunGraph("r")


def test_a_folded_node_can_always_be_unrolled_again():
    """The fold is a LENS, not a summary: every folded node keeps the unrolled keys it stands for,
    so a view can drill from `ask x3` straight back to the three executions — and each member is a
    key the checkpoint store answers to.

    This is the tape model paying out. A declared-graph tool cannot promise it: there the cycle is
    the primitive, and the iterations may never have been recorded as distinct things at all."""
    trace = ["plan", "ask", "act", "ask#2", "act#2", "ask#3"]
    folded = fold_cycles(from_keys("r1", trace), drop=(Index,))

    by_key = {node.key: node for node in folded.nodes}
    assert by_key["ask"].members == ("ask", "ask#2", "ask#3")
    assert by_key["act"].members == ("act", "act#2")
    assert by_key["plan"].members == ("plan",)
    # every member is a real recorded key — the drill-down lands on something you can look up
    assert set(m for node in folded.nodes for m in node.members) == set(trace)
    assert sum(len(node.members) for node in folded.nodes) == len(trace)


def test_folding_twice_is_stable_and_keeps_every_member():
    """Idempotence, and the reason it matters: a view that re-folds an already-folded graph (a
    caller re-projecting after a scrub) must not lose the drill-down."""
    once = fold_cycles(
        from_keys("r1", ["a", "a#2", "gather:0,0;b", "gather:0,1;b"]), drop=(Index,)
    )
    twice = fold_cycles(once, drop=(Index,))
    assert [(n.key, n.count) for n in twice.nodes] == [(n.key, n.count) for n in once.nodes]
    assert {n.key: n.members for n in twice.nodes} == {n.key: n.members for n in once.nodes}


def test_the_folded_graph_is_the_program_and_each_node_points_at_its_source():
    """The folded graph IS the program, so it inherits the key SOURCE MAP, and
    the view becomes a profiler: program structure, execution counts, each box pointing at the
    line that produced it.

    `compose_key` is handed a `Template`, so each hole's source expression survives into the
    registry; `keymap.explain` reads it back. That is the ingredient a declared graph does not
    have and an f-string-built key would have thrown away."""

    from effective.keys.registry import KeyMap
    from effective.lint import build_key_registry

    root = Path(__file__).parent.parent / "src"
    files = [f for tree in ("effective", "agent") for f in (root / tree).rglob("*.py")]
    shapes, _ = build_key_registry(files)
    keymap = KeyMap.from_shapes(shapes)

    folded = fold_cycles(from_keys("r1", ["ledger;r1:done", "ledger;r1:done#2"]), drop=(Index,))
    (node,) = folded.nodes
    assert node.count == 2  # ran twice...

    explained = keymap.explain(node.key)
    assert explained.bindings  # ...the named fields the key carries...
    assert ":" in explained.site  # ...and the file:line that mints that identity
    assert explained.site.split(":")[-1].isdigit()


def test_telemetry_attaches_to_nodes_and_sums_through_the_fold():
    """Spans on the graph: cost and duration attach per op key, and the fold SUMS them. That
    turns the cycle view into a profiler ("this loop cost $2.10 over 100 iterations"), attributed
    to a node that already knows its producing line.

    The spans arrive as a plain mapping, so the projection stays a pure fold and the caller
    decides where spans live (a table, a JSONL sidecar, a collector)."""
    trace = ["ask", "tool:x", "ask#2", "tool:x#2"]
    telemetry = {"ask": (0.01, 500_000_000), "ask#2": (0.02, 250_000_000)}
    folded = fold_cycles(from_keys("r1", trace, telemetry=telemetry), drop=(Index,))

    ask = next(n for n in folded.nodes if n.key == "ask")
    assert ask.cost == pytest.approx(0.03)  # summed across both executions
    assert ask.duration_ns == 750_000_000
    # `None`, NOT `0.0` — "nobody measured this" and "this was free" are different facts, and
    # this assertion read `== 0.0` (commented "no span, no claim") while making exactly the claim
    # it disclaimed. Nothing in `src/` produces a telemetry mapping today, so every node on every
    # real graph is this case.
    unmeasured = next(n for n in folded.nodes if n.key == "tool:x")
    assert unmeasured.cost is None
    assert unmeasured.duration_ns is None
    assert "$0.0300" in to_mermaid(folded)
    assert "750ms" in to_mermaid(folded)


def test_a_label_carrying_author_text_does_not_break_the_render():
    """Keys carry author text in their terminal position by design (`review:m1`, unencoded), so
    a quote or a newline in an event id reaches the renderer. Unescaped, either one breaks the
    diagram silently with a parse error."""
    out = to_mermaid(from_keys("r1", ['ledger;r1:he said "hi"', "ledger;r1:a\nb"]))
    assert '"hi"' not in out
    assert "#quot;" in out
    assert "<br/>" in out
    # the real property the tautology missed: a label's newline became a `<br/>`, so the diagram
    # is one line per node/edge and the renderer never sees a raw newline mid-label
    assert len(out.splitlines()) == 4  # `graph TD` + 2 nodes + 1 edge
    assert "#quot;" in out.splitlines()[1]  # the quote escaped IN PLACE, not stripped

    # The structural property, which is the half we can settle WITHOUT a mermaid renderer: labels
    # are emitted wrapped in quotes (`n0["…"]` — Mermaid's own documented handling for troublesome
    # characters), so a bare `"` surviving inside the wrapper is what turns a diagram into a parse
    # error. Assert the wrapper is intact on every emitted line rather than trusting one example:
    # quotes appear ONLY as the delimiting pair: 2 on a labelled line, 0 on a bare edge. Note
    # `% 2 == 0` would be too weak here and is the trap this exact fixture springs — `he said
    # "hi"` unescaped contributes TWO bare quotes, so the count stays even while the label is
    # broken.
    for line in out.splitlines():
        assert line.count('"') in (0, 2), line


def test_the_projection_sigil_reaches_every_renderer_unescaped():
    """`*` marks a coordinate a view dropped, so it lands in a label wherever a fold does. It needs
    no escape anywhere the graph renders: XML text and attribute data both take it verbatim, and
    Rich's markup is bracketed. Pinned because a renderer gaining an escape table is where a
    quotient would quietly become a literal asterisk in somebody's key."""
    keys = ["gather:0,0;step;tool:work", "gather:0,1;step;tool:work"]
    folded = fold_cycles(from_keys("r1", keys), drop=(Index,))
    assert [n.key for n in folded.nodes] == ["gather:*,*;step;tool:work"]
    assert to_text(folded).splitlines()[1:] == ["└─ gather:*,*", "   └─ step;tool:work  (x2)"]

    mermaid = to_mermaid(folded)
    assert 'n0["gather:*,*;step;tool:work x2"]' in mermaid
    for line in mermaid.splitlines():
        assert line.count('"') in (0, 2), line

    # The SVG half lives with the renderer that has a geometry stub:
    # `tests/test_graphlayout.py::test_the_projection_sigil_survives_the_svg_render`.


def test_the_two_projections_are_selectable_but_this_is_NOT_the_pending_node_fix(
    tmp_path, sqlite_app
):
    """`exclude` separates the SEED's projection from the VIEW's, and that is ALL it does. This
    test pins the limitation as loudly as the capability.

    What `exclude` buys: on Absurd, `$awaitEvent:` rows exist, so a view can ask for them instead
    of silently inheriting the fork's filter.

    What it leaves alone: on **SQLite an await writes no checkpoint row at all**; the park lives
    in `tasks.waiting_event` (`sqlite.py`'s `_Suspend` arm). Measured below on a really-parked run:
    filtered and unfiltered reads are IDENTICAL, and neither contains the await. A
    checkpoint-reader parameter cannot conjure a row the engine never wrote, so the pending node
    comes from a SECOND reader, the task/park reader, through a `pending=` argument. The next test
    is the same run projected with that argument supplied."""
    from effective.checkpoints import is_engine_internal

    assert is_engine_internal("$awaitEvent:review:m1") is True  # the seed's projection drops it
    assert is_engine_internal("$awaitEvent:review:m1", ()) is False  # the view's would keep it

    app = sqlite_app(str(tmp_path / "parked.db"))
    task_id = _run(app, "p", small_wf, "r-p", "m-p")  # parks at review:m-p; no answer emitted
    waiting = app.conn.execute(
        "SELECT state, waiting_event FROM tasks WHERE task_id=?", (task_id,)
    ).fetchone()
    assert waiting == ("waiting", "review:m-p")  # the ENGINE knows exactly where it is...

    filtered = list(keys(read_sqlite_conn(app.conn, task_id)))
    unfiltered = list(keys(read_sqlite_conn(app.conn, task_id, exclude=())))
    assert filtered == unfiltered  # ...and the checkpoint reader cannot see it, either way
    assert not any("review" in k for k in unfiltered)
    assert "{{" not in to_mermaid(from_keys("r-p", unfiltered))  # no await node to render
    app.close()


def test_a_parked_run_projects_a_pending_node_end_to_end_on_the_embedded_engine(
    tmp_path, sqlite_app
):
    """The whole bridge on a REAL parked run, 0↔1 half: engine → park reader → `pending_key` →
    `from_keys(pending=…)` → Mermaid. Nothing hand-built; every input has a producer.

    Then the resume transition, where the pending node changes identity across its own
    lifecycle. On this engine the pending node **simply disappears, and no committed counterpart
    replaces it**. SQLite writes no checkpoint for an await at any point in its life, neither
    while parked nor on delivery, so the answered await leaves no trace in the key sequence. The
    evidence that the answer landed is the ops that ran *after* it (`ledger;reviewed:m-p`,
    `ledger;committed:m-p`), not a node where the await was. The Absurd half of this transition is
    `test_parked_reader.py::test_the_pending_node_on_a_real_absurd_park_and_what_resume_does`,
    where the answer is different, which is why this is pinned per engine."""
    app = sqlite_app(str(tmp_path / "e2e.db"))
    task_id = _run(app, "p", small_wf, "r-p", "m-p")  # parks at review:m-p; no answer emitted

    (park,) = read_sqlite_parked_conn(app.conn)
    assert park.wake_event == "review:m-p"  # RAW, as the reader reports it
    recorded = list(keys(read_sqlite_conn(app.conn, task_id)))
    assert recorded == ["step:extract", "ledger;extracted:m-p"]

    graph = from_keys("r-p", recorded, pending=pending_key(park))
    assert [n.key for n in graph.nodes] == [*recorded, "event;review:m-p"]
    assert graph.nodes[-1].kind == "await"
    assert graph.nodes[-1].state == PARKED
    assert graph.edges[-1] == Edge("ledger;extracted:m-p", "event;review:m-p")

    rendered = to_mermaid(graph)
    assert '{{"event;review:m-p<br/>(parked)"}}' in rendered  # the hexagon, labelled
    assert (
        fold_cycles(graph, drop=(Index,)).nodes[-1].state == PARKED
    )  # ...and it survives the fold

    # --- the resume transition ---------------------------------------------------------------
    app.emit_event("review:m-p", {"decision": "approve"})
    snap = app.run_until_result(task_id, max_batches=8)
    assert snap is not None
    assert snap.state == "completed", snap

    assert read_sqlite_parked_conn(app.conn) == ()  # no park, so no pending node to synthesize
    after = list(keys(read_sqlite_conn(app.conn, task_id, exclude=())))
    assert after == [
        "step:extract",
        "ledger;extracted:m-p",
        "ledger;reviewed:m-p",
        "ledger;committed:m-p",
    ]
    assert not any(k == "review:m-p" or k.startswith(("event;", "$awaitEvent:")) for k in after)
    assert "{{" not in to_mermaid(from_keys("r-p", after))  # the node is gone, not replaced
    app.close()


def test_the_run_you_look_at_is_a_run_you_can_fork(tmp_path, sqlite_app):
    """The projection's example and the fork's example agree, and a test holds them to it.

    `small_wf` scopes its ledger event ids and its await on the SUBJECT, not the run id, and
    that is what keeps the run this file projects forkable: a child re-runs the same generator
    under its own run id, so a run-scoped id would mint `ledger;r-fork:extracted` where the seed
    holds `ledger;r-base:extracted` and `SeedingCtx` would refuse the whole fork
    (`SeedBoundaryError`). The dashboard's premise is that you look at a run and then fork it, so
    an exemplar that cannot be forked is a defect in the example vocabulary.

    Scoping on the SUBJECT fixes both name axes at once, and the proof is a real fork of the very
    run the projection tests read: same seed, same fork point, a counterfactual that reaches
    `approve` where the base reached `reject`, on its own sealed hypothetical lineage.

    Its siblings `agent_loop_wf` and `fan_wf` carry the same id shape but are **structurally**
    un-forkable: neither contains an `await`, so neither has a fork point to cut at. That is a
    property of the shapes, and no naming change reaches it."""
    from effective.fork import fork_seed, run_fork
    from effective.handlers.absurd import fork_event_name

    app = sqlite_app(str(tmp_path / "forkable.db"))
    message_id = "m-fork"
    base_id = _run(app, "base", small_wf, "r-base", message_id, answer={"decision": "reject"})

    base_keys = list(keys(read_sqlite_conn(app.conn, base_id)))
    seed = fork_seed(read_sqlite_conn(app.conn, base_id), through=f"ledger;extracted:{message_id}")
    assert [k.stored() for k in seed] == [
        "step:extract",
        f"ledger;extracted:{message_id}",
    ]  # the prefix, nothing more

    @app.register_task("child")
    def child_task(params, ctx):
        hypothetical = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: small_wf(message_id),  # the child inherits the BASE's subject
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hypothetical,
            domain=_domain(),
            forked_from="r-base",
            forked_at_event=f"extracted:{message_id}",
            fork_point=compose_key(t"review:{Segment(message_id)}"),
            delta={"decision": "approve"},
        )

    child_id = app.spawn("child", {"run_id": "r-fork"})
    app.run_until_result(child_id, max_batches=8)
    app.emit_event(
        fork_event_name("r-fork", compose_key(t"review:{Segment(message_id)}")).stored(),
        {"decision": "approve"},
    )
    snap = app.run_until_result(child_id, max_batches=8)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == "approve"  # the counterfactual, from the run the view projected

    kinds = [
        kind
        for (kind,) in app.conn.execute(
            "SELECT kind FROM ledger WHERE workflow_run_id='r-fork' ORDER BY seq"
        )
    ]
    assert kinds == ["forked", "reviewed", "committed", "fork_sealed"]  # sealed => valid marginal

    # ...and the child's own trace projects with the same projector, sharing the base's prefix —
    # which is what "the same run" means to the view: one graph vocabulary across the fork edge.
    child_keys = list(keys(read_sqlite_conn(app.conn, child_id)))
    assert child_keys[: len(seed)] == base_keys[: len(seed)]
    assert keys_of(from_keys("r-fork", child_keys))[:2] == [
        "step:extract",
        f"ledger;extracted:{message_id}",
    ]
    app.close()


def test_the_session_reset_refuses_an_undeclared_database(monkeypatch):
    """Pytest may run against a production DSN, where a request parked for a human sign-off is
    non-terminal and must be kept. An unmarked database yields no connection, so
    neither the reset, the lock nor the per-test sweep writes to it."""
    import conftest
    import psycopg

    class _Unmarked:
        """A connection that reports no marker and fails loudly on any write."""

        closed = False

        def execute(self, sql, params=None):
            text = str(sql)
            if "_pgtest_disposable" in text:
                return type("R", (), {"fetchone": staticmethod(lambda: (None,))})()
            raise AssertionError(f"the reset touched an undeclared database: {text!r}")

        def close(self):
            self.closed = True

    unmarked = _Unmarked()
    monkeypatch.setattr(psycopg, "connect", lambda *_a, **_k: unmarked)
    assert conftest._disposable_connection() is None
    assert unmarked.closed


def test_a_LOOPING_program_folds_back_into_a_loop_across_the_ENGINE_WRAPPER():
    """The cycle view's whole purpose, on the shape that broke it: a repeated PARK.

    A loop unrolls into a chain and `fold_cycles` recovers the loop. On the await axis that
    needs the grammar to accept Absurd's own `$awaitEvent:{name}` wrapper around a park, so that
    `split_occurrence` can peel `#2` off it. A grammar that refuses the wrapper draws a program
    that awaited one name twice as TWO nodes, each reporting occurrence 1: no error, just a
    picture of a chain that never closes.

    Measured 2026-08-09 on the identical key list, with the wrapper refused and accepted:

        refused    3 folded nodes   `$awaitEvent:ask:q` count=1, `$awaitEvent:ask:q#2` count=1
        accepted   2 folded nodes   `$awaitEvent:ask:q` count=2

    This is the substrate's Y-combinator-shaped test: one small program that exercises the
    fixpoint, the frame scope, the engine's foreign boundary and the occurrence coordinate at
    once. It either folds into a loop or one of those is subtly wrong. A parser total over
    foreign tags is what makes it fold.
    """
    from effective.graphview import fold_cycles, from_keys

    # what a recursive, parking program leaves behind: one name reached twice on BOTH axes
    keys = ["$awaitEvent:ask:q", "$awaitEvent:ask:q#2", "step;tool:work", "step;tool:work#2"]
    folded = fold_cycles(from_keys("r", keys), drop=(Index,))

    by_key = {node.key: node for node in folded.nodes}
    # `ask:{turn}` is a registered namespace whose coordinate declares `Index`, and the wrapper is
    # peeled before the payload is read, so the park's own coordinate projects too.
    assert set(by_key) == {"$awaitEvent:ask:*", "step;tool:work"}, sorted(by_key)
    # the PARK axis is the one that broke; the step axis folded all along
    assert by_key["$awaitEvent:ask:*"].count == 2
    assert by_key["step;tool:work"].count == 2  # anti-vacuity: the step axis folded all along


UNROLLED_AND_FOLDED = (
    ("a loop, on the occurrence coordinate", "03-unrolled-loop", "04-folded-loop"),
    ("a fan-out, on the branch coordinate", "05-gather-interleaved", "06-folded-gather"),
)
"""Each demo pair renders the same run twice, and the second is what `fold_cycles` recovers."""


@pytest.mark.parametrize(
    ("unrolled", "folded"),
    [(unrolled, folded) for _id, unrolled, folded in UNROLLED_AND_FOLDED],
    ids=[i for i, _u, _f in UNROLLED_AND_FOLDED],
)
def test_each_demo_pair_still_demonstrates_the_fold_it_is_named_for(unrolled, folded):
    """Key-shaped DATA in `src/`, which no gate scans, so it went stale in silence once: a corpus
    whose keys the fold cannot group folded 8 to 8 and demonstrated nothing.

    Asserted as a RELATION between the pair rather than as either one's node count. A count is the
    first thing a quotient change moves, so pinning one buys a re-pin every time the fold is
    touched and says nothing about whether the picture still shows a loop. What the gallery
    promises is that the second image is smaller and closed.

    Both pairs, because they fold on different coordinates: the loop on the occurrence suffix,
    which no role covers, and the gather on a branch coordinate declaring `Index`. Dropping the
    occurrence strip leaves the gather pair passing and turns the loop into a nine-node fan."""
    from effective.graphlayout.demo import corpus

    shipped = corpus()
    before, after = shipped[unrolled], shipped[folded]
    assert not before.cyclic
    assert after.cyclic, "a fold that recovers a loop makes the graph cyclic"
    assert len(after.nodes) < len(before.nodes)
    assert after.executions == before.executions  # fewer NODES, never fewer FACTS


def test_every_demo_key_is_in_the_language():
    """A key the grammar refuses is what stops a corpus from folding.

    `--key-literals` covers half of this already: it reads `src` and refuses a malformed literal
    under a tag the REGISTRY OWNS, so a broken `gather:` here reddens `just lint`. What it does
    not read, by the ownership rule its own docstring states, is a literal under a tag production
    does not own, which is most of this corpus (`start`, `plan`, `ask`, `fetch`, `summarize`)."""
    from effective.graphlayout.demo import GATHER, LONG, LOOP, PARKED
    from effective.keys.grammar import KeySyntaxError, parse

    for corpus in (GATHER, PARKED, LONG, LOOP):
        for key in corpus:
            try:
                parse(key)
            except KeySyntaxError as exc:  # pragma: no cover - the assertion is the message
                raise AssertionError(f"demo corpus key not in the language: {key!r}") from exc


# --- the placed-writer collision, read off the tape --------------------------------------------


def test_ledger_collisions_names_the_writers_of_one_event_id():
    """The detector's positive case, on the keys the phase-0 repros actually produced."""
    from effective.graphview import from_keys, ledger_collisions

    gather_keys = [
        "gather:0,0;step;tool:a",
        "gather:0,1;step;tool:b",
        "gather:0,0;ledger;done:m1",
        "gather:0,1;ledger;done:m1",
    ]
    (collision,) = ledger_collisions(from_keys("r", gather_keys))
    assert collision.event_id == "done:m1"
    assert collision.count == 2
    assert collision.writers == ("gather:0,0;ledger;done:m1", "gather:0,1;ledger;done:m1")

    # No gather anywhere — the engine's `#N` is the only thing separating the two writers.
    (sequential,) = ledger_collisions(from_keys("r", ["ledger;m2:done", "ledger;m2:done#2"]))
    assert sequential.event_id == "m2:done"
    assert sequential.writers == ("ledger;m2:done", "ledger;m2:done#2")


def test_hand_disambiguated_branches_are_not_a_collision():
    """The discrimination that makes the positive case mean something: identical program position,
    identical frames, DIFFERENT authored ids — which is what every in-branch append in this tree
    already does."""
    from effective.graphview import from_keys, ledger_collisions

    clean = ["gather:0,0;ledger;ka:m1", "gather:0,1;ledger;kb:m1"]
    assert ledger_collisions(from_keys("r", clean)) == ()


def test_a_collision_under_a_combinator_scope_is_still_detected():
    """A collision inside a `recurse`/`descend` scope, which the detector finds by grouping on
    `bare_name` rather than by reusing the fold.

    The two normalizations answer different questions and stay separate: `bare_name` asks *is this
    the same identity as that one*, so it looks past EVERY frame down to the op tag, while the fold
    asks *what program position is this* and drops only the frames whose coordinate is an unrolling
    index. A `sub:`-scoped pair is one node to `bare_name` and two to the fold, deliberately."""
    from effective.graphview import from_keys, ledger_collisions

    under_recurse = ["rec:0;gather:0,0;ledger;x:1", "rec:0;gather:0,1;ledger;x:1"]
    under_descend = ["d:0;ledger;x:1", "d:1;ledger;x:1"]

    for corpus in (under_recurse, under_descend):
        (collision,) = ledger_collisions(from_keys("r", corpus))
        assert collision.event_id == "x:1", corpus
        assert collision.count == 2

    # ...and distinct ids under the same scopes are still clean.
    assert ledger_collisions(from_keys("r", ["d:0;ledger;a:1", "d:1;ledger;b:1"])) == ()


def test_only_ledger_nodes_collide():
    """A repeated STEP is a loop and a repeated ARTIFACT is content-addressed dedup — neither is a
    lost row. Row 3 of the crux report's totality table is the case that proves the rule is not
    'the frame must always be there'."""
    from effective.graphview import from_keys, ledger_collisions

    assert ledger_collisions(from_keys("r", ["step;tool:work", "step;tool:work#2"])) == ()
    assert (
        ledger_collisions(from_keys("r", ["artifact:text/plain:ab", "artifact:text/plain:ab#2"]))
        == ()
    )


def test_the_parked_node_is_never_a_collision():
    """`pending=` synthesizes a node no bookkeeper wrote. It is an await, which writes no ledger
    row, so it must not be counted as a writer of anything."""
    from effective.graphview import from_keys, ledger_collisions

    graph = from_keys("r", ["ledger;a:1"], pending=Key.parse("event;review:m1"))
    assert ledger_collisions(graph) == ()


def test_a_real_but_tiny_cost_does_not_render_as_zero():
    """A live gpt-5-nano turn costs ~$0.000042 (measured 2026-08-24), which `.4f` renders
    `$0.0000`: the same "this node was free" claim the `None`/`0.0` distinction exists to avoid,
    arriving from the other side."""
    assert format_cost(0.000042) == "$0.000042"
    assert format_cost(0.0300) == "$0.0300"  # the ordinary case is unchanged
    assert format_cost(0.0) == "$0.0000"  # measured-and-free stays the plain form


def test_a_measured_zero_renders_distinguishably_from_unmeasured():
    """`Node.cost` is `float | None` so "free" and "unmeasured" cannot collapse: a model call
    whose usage carried no price is measured and free (`0.0`). A renderer testing truthiness
    collapses them anyway.

    Pinned as a rendering difference rather than as `is not None` at two line numbers, because the
    invariant is that the two states stay TELLABLE APART on the surface a human reads.

    `Usage.as_attributes()` always emits a cost, so an unpriced model produces exactly
    this node. It is the ordinary case, not a contrived one."""
    measured_free = from_keys("t", ["step;tool:a"], telemetry={"step;tool:a": (0.0, 0)})
    unmeasured = from_keys("t", ["step;tool:a"], telemetry=None)

    assert to_mermaid(measured_free) != to_mermaid(unmeasured)


# --- containment: the second multiplicity ------------------------------------------------------


def test_a_node_knows_what_it_is_INSIDE_not_only_what_it_stands_for():
    """The two multiplicities, side by side on one machine tape.

    `members` collapses peers and `frames` nests, so no consumer re-derives containment by
    re-parsing the key. The tape below is the
    shape `effective/coding/__init__.py` mints: `d:` counts the trampoline's turns, `state:`
    names which code ran, and dropping either one loses a distinction.
    """
    tape = [
        "d:0;state:test;step;tool:run_suite",
        "d:1;state:draft;step;tool:read_file",
        "d:1;state:draft;step;tool:apply_fix",
        "ledger;machine:r-1",
    ]
    graph = from_keys("r-1", tape)
    assert [(node.frames, node.order) for node in graph.nodes] == [
        (("d:0", "state:test"), 0),
        (("d:1", "state:draft"), 1),
        (("d:1", "state:draft"), 2),
        ((), 3),
    ]
    # The postamble row is FRAMELESS and ran last. A renderer that groups by `frames` and sorts by
    # `order` puts it after the visits; one that groups alone hoists it to the top, which is
    # exactly the fidelity that hoisting loses.
    assert graph.nodes[-1].frames == ()
    assert graph.nodes[-1].order == 3


def test_frames_survive_the_projections_and_agree_with_the_key_they_came_from():
    """A folded view is still a tree, and each node's frames still describe its own key.

    `project` output keeps its frames. A `regroup` that rebuilds a `Node` from some of its fields
    returns every projected node with `frames` at its default. The second assertion is the one
    with teeth: the frames are read off the name the node HAS, so a quotient that stripped a
    coordinate cannot leave a node describing the key it had before.
    """
    tape = [
        "d:0;state:test;step;tool:run_suite",
        "d:1;state:draft;step;tool:read_file",
        "d:1;state:draft;step;tool:run_suite",
    ]
    graph = from_keys("r-1", tape)

    # `project` discovers the axes and strips the coordinates; the frames remain, uncoordinated.
    assert {node.frames for node in project(graph).nodes} == {("d", "state")}
    # `fold_cycles` drops `d:` (an UNROLLING scope) and keeps `state:` (a NAMING one).
    assert {node.frames for node in fold_cycles(graph, drop=(Index,)).nodes} == {
        ("d:*", "state:test"),
        ("d:*", "state:draft"),
    }
    for projection in (project(graph), fold_cycles(graph, drop=(Index,))):
        for node in projection.nodes:
            assert node.key.startswith("".join(f"{frame};" for frame in node.frames))


def test_order_folds_as_the_minimum_so_a_node_sorts_where_it_FIRST_ran():
    """`order` is the one field a projection cannot recompute, which is why it folds rather than
    re-derives.

    `kind`, `occurrence`, `path` and `frames` are all functions of the name — hole a projection and
    they come back. Position is a fact about the run, so a fold that dropped it has destroyed it.
    The minimum (rather than last-seen) is what makes a folded node sort where its earliest
    execution ran, and it agrees with the order `regroup` already emits nodes in.
    """
    graph = from_keys("r1", ["plan", "ask", "act", "ask#2", "act#2", "ask#3"])
    folded = {node.key: node for node in fold_cycles(graph, drop=(Index,)).nodes}
    assert folded["ask"].members == ("ask", "ask#2", "ask#3")
    assert folded["ask"].order == 1  # the FIRST ask, not the third
    assert folded["act"].order == 2
    assert folded["plan"].order == 0
    # The emitted sequence and the field cannot disagree about the same graph.
    assert [node.order for node in fold_cycles(graph, drop=(Index,)).nodes] == sorted(
        node.order for node in fold_cycles(graph, drop=(Index,)).nodes
    )


def test_regroup_rebuilds_EVERY_field_of_a_node():
    """The tripwire for the defect that produced this work, stated over the type rather than over
    a list of symptoms.

    `regroup` is the aggregation half of every projection here, and it constructed a fresh `Node`
    from seven of nine keyword arguments. `occurrence` and `path` fell back to their defaults
    through both `project` and `fold_cycles`; `frames` and `order` would have been the third and
    fourth casualties, and the field after that would have been the fifth. Enumerating the field
    set here means the next person to add one meets a red test instead of a silent default.

    **If this assertion fails because you added a field: add it to `regroup` first**, then name it
    here. The set is compared against `dataclasses.fields` so it cannot go stale in the quiet
    direction.
    """
    from dataclasses import fields

    assert {f.name for f in fields(Node)} == {
        "key",
        "kind",
        "occurrence",
        "path",
        "frames",
        "order",
        "count",
        "state",
        "cost",
        "duration_ns",
        "members",
    }, "a Node field was added or removed — check `regroup` carries it before updating this set"


def test_a_folded_node_never_contradicts_its_own_key_about_its_coordinates():
    """`occurrence` and `path` are recomputed from the FOLDED name, not carried from the group's
    representative — so the node and its key cannot disagree.

    The distinction matters because a quotient may drop exactly those coordinates. `fold_cycles`
    DECLARES that it drops the branch, and `RunGraph.dropped` records it; the honest `path` for a
    node whose key no longer carries a branch is therefore the empty one, and reading it off the
    name is what guarantees that. Carrying the representative's value would have reported a branch
    coordinate on a key that has none.
    """
    graph = from_keys(
        "r1",
        ["gather:0,0;step;tool:a", "gather:0,1;step;tool:b", "gather:0,0;step;tool:c"],
    )
    assert [(n.key, n.path) for n in graph.nodes] == [
        ("gather:0,0;step;tool:a", ((0, 0),)),
        ("gather:0,1;step;tool:b", ((0, 1),)),
        ("gather:0,0;step;tool:c", ((0, 0),)),
    ]
    folded = fold_cycles(graph, drop=(Index,))
    assert "index" in folded.dropped  # the role a branch coordinate declares
    for node in folded.nodes:
        assert node.path == (), f"{node.key} claims a branch its key does not carry"


def test_a_race_edge_reads_as_forward_once_the_branch_is_folded_away():
    """Pinned as INTENDED behavior of the fold.

    `graphlayout.prepare.classify` derives `race` from `Node.path`, and its docstring says drawing
    such an edge as a plain arrow *"would assert causation the record does not contain."* On a
    FOLDED graph it does exactly that. That is not `regroup` dropping a field: the fold declares it
    removes the branch coordinate, the folded key no longer carries one, and `RunGraph.dropped`
    says so; a node reporting a branch there would be the false claim instead. A consumer that
    needs the race reads the unrolled graph, which still reports it.
    """
    from effective.graphlayout.prepare import prepare

    graph = from_keys(
        "r1",
        ["gather:0,0;step;tool:a", "gather:0,1;step;tool:b", "gather:0,0;step;tool:c"],
    )
    assert [edge.kind for edge in prepare(graph).edges] == ["commit-order", "commit-order"]
    assert [edge.kind for edge in prepare(fold_cycles(graph, drop=(Index,))).edges] == [
        "forward",
        "forward",
    ]


# --- the tree renderer -------------------------------------------------------------------------


def test_to_text_draws_the_trajectory_a_machine_run_actually_took():
    """The tree IS the trajectory, ORDER included.

    A renderer that hoists frameless nodes to the top of their level draws a postamble (the
    artifact, the bare tool call, the ledger rows) FIRST although it ran last. The tape below is
    in true commit order, and the assertion is that the postamble draws at the BOTTOM. That is
    the whole reason `Node.order` is a field: group by containment alone and the picture claims
    an order the run did not have.
    """
    tape = [
        "d:0;state:test;step;tool:run_suite",
        "d:1;state:draft;step;tool:read_file",
        "d:1;state:draft;step;tool:run_suite",
        "artifact:application/json,sha256-d47e",
        "ledger;machine:r-9",
    ]
    assert to_text(from_keys("r-9", tape)) == "\n".join(
        [
            "r-9",
            "├─ d:0",
            "│  └─ state:test",
            "│     └─ step;tool:run_suite",
            "├─ d:1",
            "│  └─ state:draft",
            "│     ├─ step;tool:read_file",
            "│     └─ step;tool:run_suite",
            "├─ artifact:application/json,sha256-d47e",
            "└─ ledger;machine:r-9",
        ]
    )


def test_a_leaf_and_a_frame_group_sort_against_EACH_OTHER_not_in_two_passes():
    """A frame that opened before a sibling leaf ran must draw above it.

    The easy implementation renders leaves, then groups — which is stable, looks right on the
    machine tape (whose postamble happens to sit at one end), and silently reorders any run that
    interleaves the two. Here `first` runs before the frame and `last` after it, so a two-pass
    renderer would put both leaves together and the run's shape would be lost.
    """
    tape = ["first", "d:0;step;tool:a", "d:0;step;tool:b", "last"]
    assert to_text(from_keys("r", tape)) == "\n".join(
        [
            "r",
            "├─ first",
            "├─ d:0",
            "│  ├─ step;tool:a",
            "│  └─ step;tool:b",
            "└─ last",
        ]
    )


def test_a_flat_tape_draws_flat_and_that_is_the_honest_picture():
    """Route T earns its keep on a framed run and must not invent structure on a flat one.

    A renderer that manufactured levels here would be claiming a shape the tape does not carry;
    `project` is what recovers structure from a flat tape, and it is the caller's choice to apply
    it."""
    tape = ["step;plan", "step;tool:list_dir", "step;review"]
    assert to_text(from_keys("run-1", tape)) == "\n".join(
        ["run-1", "├─ step;plan", "├─ step;tool:list_dir", "└─ step;review"]
    )


def test_the_badge_distinguishes_measured_zero_from_unmeasured():
    """`Node.cost`'s claim, arriving at the last step before a human reads it.

    `is not None`, not truthiness: a measured-and-free model call is `0.0` and must render, or the
    renderer collapses "this cost nothing" back into "nobody measured this" — the exact lie the
    field stopped telling at the leaf.
    """
    graph = from_keys(
        "r",
        ["step;tool:free", "step;tool:unmeasured"],
        telemetry={"step;tool:free": (0.0, None)},
    )
    lines = to_text(graph).splitlines()
    assert lines[1] == "├─ step;tool:free  ($0.0000)"
    assert lines[2] == "└─ step;tool:unmeasured"


def test_project_does_not_FORGE_the_foreign_join_the_engine_wrote():
    """`project`'s node key is the identity `regroup` groups on, so its bytes have to be real.

    A foreign term that wraps our key joins its payload with `:` (the bytes the SDK writes), so
    rebuilding a key by joining tags with `;` turns `$awaitEvent:review:m1` into
    `$awaitEvent;review:m1`. That parses, as a DIFFERENT valid key, which is why no other
    assertion sees it; it covered 1,718 of the 7,126 distinct live `absurd.c_default` names on
    2026-08-29, a store that grows with every suite run. A join from a projected node back to the
    store misses, and the label shown to a human is a name no producer ever wrote.
    """
    engine_wrote = [
        "$awaitEvent:review:m1",
        "$awaitEvent:budget-grant:mg-c0ab9919,0",
        "$awaitEvent:approve;ledger;processed:sha256-bdbddb70872b68d9",
        "step;tool:read",
    ]
    graph = from_keys("r1", engine_wrote)
    assert {node.key for node in project(graph).nodes} == set(engine_wrote), (
        "a projected key must still be bytes a producer wrote -- `;` where the engine used `:` "
        "is a different valid key, so nothing downstream can notice the substitution"
    )

    # ... and again where π actually DROPS something, since the four keys above have empty drop
    # sets and so only exercise the identity case. Here `axes` finds two, and the foreign `:`
    # has to survive the rebuild that removes them.
    fan = [  # spelled, not composed: these are the ENGINE's bytes, and a fixture states them
        "$awaitEvent:gather:0,0;review:m0",
        "$awaitEvent:gather:0,1;review:m1",
        "$awaitEvent:gather:0,2;review:m2",
        "$awaitEvent:gather:0,3;review:m3",
    ]
    assert {node.key for node in project(from_keys("r1", fan)).nodes} == {
        "$awaitEvent:gather:0;review"
    }


def test_the_branch_pattern_reads_a_frame_the_SUBSTRATE_actually_mints():
    """`BRANCH` is built from the grammar's constants; this checks the build against a real mint.

    The pattern states four facts -- the gather arm's tag and the tag, arity and term separators.
    Deriving them keeps a grammar change from having to find a regex, but a derivation can still
    be wrong in a way only a real key exposes, and asserting the pattern text against the same
    constants it was built from would be circular. So the frame here comes from `api.gather_frame`
    -- the substrate's own minter, the thing whose output `branch_path` exists to read.
    """
    from effective.api import gather_frame
    from effective.keys import compose_key

    inner = compose_key(t"step;tool:read")
    framed = gather_frame(0, 1, gather_frame(2, 3, inner)).stored()

    assert branch_path(framed) == ((0, 1), (2, 3)), "outermost first"
    assert strip_branches(framed) == inner.stored()


def test_a_parked_await_folds_across_gather_branches_like_any_other_op():
    """`fold_cycles` groups by program position, so a branch coordinate must drop wherever it sits.

    The engine's park record wraps our key (`$awaitEvent:{key}`), and a walk that reads the
    wrapper as part of the first frame cannot see the branch behind it -- so two branches of one
    gather stayed two positions while the ordinary op beside them folded to one.
    """
    ordinary = ["gather:0,1;step;tool:read", "gather:0,2;step;tool:read"]
    parked = ["$awaitEvent:gather:0,1;ev:x", "$awaitEvent:gather:0,2;ev:x"]

    assert len(fold_cycles(from_keys("r1", ordinary)).nodes) == 1
    assert len(fold_cycles(from_keys("r1", parked)).nodes) == 1, "the park folds like its sibling"


def test_a_parked_node_sits_in_the_SAME_scope_as_its_unparked_sibling():
    """`Node.frames` is the containment relation, and a park does not change what contains it.

    Reading the wrapper as part of the first frame put one scope in the tree twice -- once as
    `fork:X` and once as `$awaitEvent:fork:X` -- so a run's tree showed two levels where the run
    had one.
    """
    graph = from_keys("r1", ["$awaitEvent:fork:X;review:m1", "fork:X;step;tool:read"])
    assert {node.frames for node in graph.nodes} == {("fork:X",)}

    # ... and the leaf is still the node's own identity, not the whole key: the frames are no
    # longer a literal prefix, so a positional strip would have returned everything.
    assert {identity(node) for node in graph.nodes} == {
        "$awaitEvent:review:m1",
        "step;tool:read",
    }


def test_the_leaf_label_keeps_the_occurrence_that_tells_repeats_apart():
    """`#N` is what distinguishes three executions of one step in the unrolled view.

    Nothing else renders it there -- the badge carries count, cost and state, and count is 1 when
    a view is unrolled -- so dropping it from the label makes three nodes draw identically.
    """
    graph = from_keys("r1", ["step;tool:ask", "step;tool:ask#2", "step;tool:ask#3"])
    assert [identity(node) for node in graph.nodes] == [
        "step;tool:ask",
        "step;tool:ask#2",
        "step;tool:ask#3",
    ]
    # ... and it survives a frame the leaf does not show, which is where the strip crept in.
    parked = from_keys("r1", ["$awaitEvent:fork:X;review:m1#2"])
    assert [identity(node) for node in parked.nodes] == ["$awaitEvent:review:m1#2"]

    # `identity` is the shared seam; this is the assembled label, where a future strip could land.
    assert "step;tool:ask#3" in to_text(graph)
