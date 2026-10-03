"""The composition table and its laws.

Infra-free, on `RecordingHandler` — the same instrument `test_combinators.py` uses, for the
same reason: a combinator that assembled a wrong key would miss its canned response and fail
loudly. What is new is that no key here is *spelled*. Each member declares once what frames a
body nested in its hole will see (`tests/_composition.py`), and every expectation is computed
from that declaration through `qualified_event_name` — the importable oracle the engines are
already pinned against (`test_recurse_with_parking_leaves_serializes_wakes_through_the_scopes`).

Six laws, and what each one is FOR:

=================================  ===========================================================
frames concatenate                 the homomorphism on checkpoint keys, every ordered pair
the park name carries them too     the same law on the surface the defects actually used
both, at depth three               21 frame shapes a pair cannot reach (12 -> 33, measured)
no cell is SILENT                  each cell RUN; the outcome with no `Disposition` member
a terminal is refused in a BRANCH  `fork` and `respawn`: one discriminator, two mechanisms
a generation frames nothing        `respawn` as the OUTER — the identity element, measured
the diagonal                       A∘A, where both remaining silent defects lived
=================================  ===========================================================

**Every one of these FAILS under a mutation of the code it guards**, which is the answer to
"does a law earn its keep". Dispositions are measured (`observed_disposition` runs every cell)
rather than declared, so no law reads its own premise back: a disjunction over the full cartesian
product, or a no-silent check of `x in {x}`, would hold by construction.

**The scope caveat is the same for all of them:** this is the RECORDER, and parks are exactly
where the two engines diverge (the nested-gather frame defect and the `sleep_until` arity bug
were both Absurd-only). Green here is necessary, not sufficient. The durable half is the
conformance suite: `..._absolute_await_reroutes_around_a_scope...`,
`..._absolute_await_inside_a_gather_branch_is_refused...`,
`..._park_carries_every_frame_through_a_nested_gather`, and
`..._park_inside_recurses_fold_carries_the_fold_frames`, all on both engines. One cell is
verified there and not here on purpose: `fork ∘ fork` needs a fork child's ctx, which exists
only on the durable path (`test_fork_durable.py`).
"""

from pathlib import Path
from uuid import uuid4

import pytest
from _composition import (
    BUILDERS,
    DONE_EVENT,
    FORK_ANSWER,
    LEAF,
    PARK_EVENT,
    SPAWN_KEY,
    TERMINALS,
    ChainState,
    Disposition,
    compose,
    compose_terminal,
    enclosing_frames,
    expected_keys,
    expected_park,
    expected_responses,
    generation_outer,
    leaf,
    observable,
    observed_disposition,
    pairs,
    parking_leaf,
    spawn_at_root,
    terminal_disposition,
    triples,
)

from effective.api import GatherBranch, gather, qualified_event_name
from effective.combinators import Again, Chain, hoisted, respawn
from effective.domain import Spawned
from effective.fork import join_fork
from effective.handlers.base import step_key
from effective.handlers.recording import RecordingHandler, Respawned, Suspended, _qualified
from effective.keys import Key
from effective.ops import CompositionRefused
from effective.skills import Pin

EXPECTED_PAIRS = 49
"""7 combinator-HOLES squared. Pinned so that a builder silently failing to register — the way a
generated suite dies quietly — fails here instead of shrinking the table to nothing."""


# One model, one law, many arms: the overlap across arms IS the design, so the unit-role
# minimize-overlap rule does not apply here.
pytestmark = pytest.mark.conformance


def _ids(cases):
    return [f"{outer.name}o{inner.name}" for outer, inner in cases]


# ------------------------------------------------------- the grid: the homomorphism law


@pytest.mark.parametrize(("outer", "inner"), pairs(), ids=_ids(pairs()))
def test_frames_concatenate_in_nesting_order(outer, inner):
    """`frames(A ∘ B) = frames(A) · frames(B)` — the monoid homomorphism, per pair.

    This is the law `formal/lean/Effective/Keys.lean` proves for the structured encoding and
    explicitly DEFERS for the string serialization, which is the level everything here runs
    at. Generalizes `test_an_enclosing_scope_encloses_the_whole_recurse`'s single hand-written
    instance (`scoped ∘ recurse`) to every ordered pair."""
    handler = RecordingHandler(responses=expected_responses(outer, inner))
    result = handler.run(compose(outer, inner))
    actual = [entry.key.stored() for entry in handler.trace]

    assert actual == expected_keys(outer, inner)
    assert result == "L", "the composed workflow must return the leaf's value"
    # Non-degeneracy of the ORACLE is `test_the_oracle_is_not_degenerate`, once, rather than
    # three clauses here that hold by construction given the equality above (measured: removing
    # the equality did not make them fire).


def test_the_table_is_not_empty():
    """The failure mode a generated suite dies of: zero pairs, all laws vacuously true."""
    assert len(pairs()) == EXPECTED_PAIRS
    assert len(BUILDERS) * len(BUILDERS) == EXPECTED_PAIRS
    assert {mk(0).name for mk in BUILDERS.values()} == set(BUILDERS), (
        "a builder must produce the combinator it is registered under"
    )


def test_the_fold_hole_reaches_frames_the_leaf_hole_cannot():
    """`recurse@fold` earns its row: it exercises `fold:{level},{k}`, which nothing did.

    The one-chunk `recurse` builder never calls `combine`, so without this row
    `combinators.py`'s fold frames are neither declared nor run. Asserted through the registry
    rather than by re-deriving: the two `recurse` holes must contribute DIFFERENT frames, and the
    fold's gather ordinal must be the second one, because that ordinal is the whole
    discriminator between two gathers at one level."""
    leaf_hole, fold_hole = BUILDERS["recurse"](0), BUILDERS["recurse@fold"](0)
    assert leaf_hole.frames != fold_hole.frames
    assert fold_hole.frames[0] == GatherBranch(1, 0), "the FOLD gather is the second one issued"
    assert leaf_hole.frames[0] == GatherBranch(0, 0), "the LEAF gather is the first"
    fold_scope = fold_hole.frames[1]
    assert isinstance(fold_scope, Key), "the second frame is a scope atom, not a coordinate"
    assert "fold" in fold_scope.stored()

    # And it is REACHED: the frames appear in a real trace, not only in the declaration.
    inner = BUILDERS["route"](1)
    handler = RecordingHandler(responses=expected_responses(fold_hole, inner))
    handler.run(compose(fold_hole, inner))
    assert any("fold:0,0" in entry.key.stored() for entry in handler.trace)


OBSERVED = {
    (outer.name, inner.name): observed_disposition(outer, inner) for outer, inner in pairs()
}
"""Every cell's disposition, MEASURED once by running it — the table's own data.

Computed at import so the two laws below read the same measurement rather than each
re-deriving one. `None` is the SILENT outcome `Disposition` has no member for."""


def test_every_pair_of_combinators_has_at_least_one_OK_order():
    """The disjunction law: for members A and B, at least one of A∘B, B∘A composes.

    It reads OBSERVED dispositions, so a member for which neither order composes fails here. An
    `ok` set built from `pairs()`, the full cartesian product, would assert only that every pair
    is in the set of all pairs: a member whose hole raises in every position would stay green."""
    ok = {cell for cell, disposition in OBSERVED.items() if disposition is Disposition.OK}
    assert ok, "anti-vacuity: at least one pair must actually compose"
    for a in BUILDERS:
        for b in BUILDERS:
            assert (a, b) in ok or (b, a) in ok, f"neither {a}o{b} nor {b}o{a} composes"


# --------------------------------------- the park-name law: the surface that was at risk


@pytest.mark.parametrize(("outer", "inner"), pairs(), ids=_ids(pairs()))
def test_a_relative_park_name_carries_exactly_the_frames_that_enclose_it(outer, inner):
    """The SAME homomorphism, measured where the defects actually were.

    The grid above runs a leaf that always COMPLETES, so it exercises the checkpoint-key path,
    while every frame defect this table cites lived on the park/event-name path, which travels
    by a different mechanism (`_scope_path` + park-as-value re-arms at the barrier). A grid that
    is green everywhere on its first run is what you get when the grid and the risk are disjoint.

    So the leaf parks instead, and the assertion is that the engine is reached with
    `frames(A) · frames(B) · name`. It is the same law and the same oracle; only the measurement
    point moves, which is the argument that a park name is not a second naming scheme.

    **Its own scope, stated:** this runs on the recorder, and parks are precisely where the two
    engines diverge; the nested-gather frame defect and the `sleep_until` arity bug were both
    Absurd-only. The durable half is `test_conformance.py`'s park cases, and the standing triple
    below is the nested-gather shape. Green here is necessary, not sufficient."""
    handler = RecordingHandler(responses=expected_responses(outer, inner))
    parked = handler.run(compose(outer, inner, body=parking_leaf()))

    assert isinstance(parked, Suspended), f"{outer.name}o{inner.name} must reach the await"
    assert _qualified(parked) == expected_park(outer, inner)
    # Anti-vacuity: an unframed park would pass a comparison of two bare names. At least one
    # frame-bearing pair must actually carry frames, and this asserts it per pair.
    assert bool(enclosing_frames(outer, inner)) == (_qualified(parked).stored() != PARK_EVENT)


def test_both_laws_hold_at_depth_three_which_reaches_frame_shapes_a_pair_cannot():
    """Every ordered TRIPLE, both laws.

    **Why depth three.** A pair already reaches the nested-gather shape
    `gather(scoped(gather(await)))`: `recurse` contributes `(GatherBranch, rec:{i})`, so
    `recurse ∘ gather` puts the leaf under gather-scope-gather. The justification is a
    measurement: pairs reach **12** distinct frame shapes at a maximum depth of 4; triples reach
    **33** at a maximum depth of 6. Twenty-one shapes, `(G,G,G)`, `(G,S,G,S,G)` and so on, exist
    only at depth three. That is what the 343 cells buy.

    One test that loops rather than 686 parametrized cases, deliberately. The value is the
    *closure* (no triple is unclassified), not per-cell reporting, and every failure is
    collected so a frame bug shows its whole footprint instead of one arbitrary first case."""
    key_failures: list[str] = []
    park_failures: list[str] = []

    for chain in triples():
        ids = "o".join(built.name for built in chain)

        # Each run is caught, because a wrong frame's DOMINANT failure mode is a raise, not a
        # mismatch: the qualified canned key misses and the bare fallback misses too, so
        # `_canned` raises. An uncaught one would abort the loop at the first bad triple and
        # leave the other 342 unmeasured, making the "no triple is unclassified" claim false in
        # exactly the case it exists for.
        try:
            handler = RecordingHandler(responses=expected_responses(*chain))
            result = handler.run(compose(*chain))
            actual = [entry.key.stored() for entry in handler.trace]
            if actual != expected_keys(*chain) or result != "L":
                key_failures.append(f"{ids}: {actual} != {expected_keys(*chain)}")
        except BaseException as exc:
            key_failures.append(f"{ids}: raised {exc!r}")

        try:
            parking = RecordingHandler(responses=expected_responses(*chain))
            parked = parking.run(compose(*chain, body=parking_leaf()))
            if not isinstance(parked, Suspended):
                park_failures.append(f"{ids}: did not park ({parked!r})")
            elif _qualified(parked) != expected_park(*chain):
                park_failures.append(
                    f"{ids}: {_qualified(parked).display()!r} != "
                    f"{expected_park(*chain).display()!r}"
                )
        except BaseException as exc:
            park_failures.append(f"{ids}: raised {exc!r}")

    assert not key_failures, f"{len(key_failures)} triples broke the key law: {key_failures[:5]}"
    assert not park_failures, f"{len(park_failures)} broke the park law: {park_failures[:5]}"
    assert len(triples()) == EXPECTED_PAIRS * len(BUILDERS)  # the table did not shrink to nothing

    # The nested-gather shape is IN here, named so a reader can see the generalization
    # contains it.
    assert ("gather", "scoped", "gather") in {tuple(b.name for b in c) for c in triples()}


# ------------------- ABSOLUTE awaits: rerouted under a scope, refused in a branch


TASK_CASES = [(mk(0), terminal_disposition(mk(0))) for mk in BUILDERS.values()]


@pytest.mark.parametrize(("outer", "disposition"), TASK_CASES, ids=[o.name for o, _ in TASK_CASES])
def test_an_absolute_await_reroutes_under_a_scope_and_is_refused_in_a_branch(outer, disposition):
    """`outer ∘ join_fork`, with the disposition DERIVED from `is_branch`.

    The defect this pins is silent. `spawn_fork` mints `fork-done:{child}`; the CHILD emits that
    name from its own params and has never seen a frame of this task's. What gets rescoped away
    from the emitter is therefore the **join's await**, not the spawn, so guarding the spawn both
    misses the real hazard (spawn at root, join in a frame: reproduced parking on
    `s:0;fork-done:c1` and `gather:0,0;fork-done:c1`, no error at any moment) and refuses a shape
    that works end-to-end (spawn in a frame, join at root: reproduced resuming to completion).

    Note the disposition does NOT key on "has frames". `scoped` and `descend` contribute frames
    and are OK, because a scope completes RELATIVE names and an absolute one is already whole,
    so the handler resolves the await at the ctx it was constructed with, as it does for the
    other absolute name (`budget-grant`). `gather` and `recurse`
    are LOUD because their bodies run in a gather BRANCH, where the coordinate is a concurrency
    slot rather than a naming choice and there is nothing to reroute to."""
    responses: dict[str, object] = {
        **expected_responses(outer, outer),  # outer's own overhead names
        # the spawn runs at the ROOT, so unframed
        SPAWN_KEY: Spawned(task_id=uuid4(), done_event=Key.parse(DONE_EVENT)),
    }
    handler = RecordingHandler(responses=responses)

    def workflow():
        handle = yield from spawn_at_root()
        return (yield from outer.hole(lambda: join_fork(handle)))

    if disposition is Disposition.LOUD:
        # Caught explicitly rather than with `pytest.raises(..., match=...)`: a gather branch
        # wraps the refusal in an `ExceptionGroup` (the documented L5 shape) whose own `str`
        # is "unhandled errors in a TaskGroup", so a `match` would test the wrapper, not the
        # message.
        #
        # Parenthesized deliberately: PEP 758 drops the parens for `except A, B:` but NOT when
        # binding with `as` — "multiple exception types must be parenthesized when using 'as'"
        # is a 3.14 SyntaxError, verified here rather than assumed.
        caught: BaseException | None = None
        try:
            handler.run(workflow)
        except (ValueError, BaseExceptionGroup) as exc:
            caught = exc
        assert caught is not None, (
            f"{outer.name} runs its body in a gather branch and must refuse an ABSOLUTE await"
        )
        message = _flatten(caught)
        assert "is an ABSOLUTE event name" in message
        assert DONE_EVENT in message, "the message must name the culprit"
        # The fix is a different SHAPE of program, so the message has to say so — an author
        # told only "refused" would reach for a smaller edit that cannot work.
        assert "loop" in message
        assert "marginal_sweep" in message
    else:
        parked = handler.run(workflow)
        assert isinstance(parked, Suspended), f"{outer.name} should park on the child's event"
        # THE REROUTE, measured: the frame `outer` contributes does NOT appear, so the parent
        # waits on exactly the name the child emits.
        assert parked.awaiting.stored() == DONE_EVENT, (
            f"{outer.name} contributes {outer.frames} — an ABSOLUTE await must ignore them"
        )
        assert parked.resume(FORK_ANSWER) is not None


def test_a_CANNED_absolute_await_resolves_instead_of_parking():
    """The recorder's absolute-await lookup fires.

    `_interpret` reroutes an absolute await to the bare name and then asks the response table
    for it with `.stored()`. `Key` is opaque, so asking with the `Key` itself compares a frozen
    dataclass against `str` keys and is silently always False: every absolute await parks,
    canned or not.

    The sibling test above cans no answer for `fork-done:c1` and asserts the PARK, which is the
    same observable either way, so only this test tells the two apart. Opacity failing QUIETLY
    is the hazard: returning `False` reads as 'not canned'."""
    handler = RecordingHandler(
        responses={
            SPAWN_KEY: Spawned(task_id=uuid4(), done_event=Key.parse(DONE_EVENT)),
            DONE_EVENT: FORK_ANSWER,  # the half that was unreachable
        }
    )

    def workflow():
        handle = yield from spawn_at_root()
        return (yield from join_fork(handle))

    outcome = handler.run(workflow)
    assert not isinstance(outcome, Suspended), "a canned absolute await must not park"
    assert [entry.key.stored() for entry in handler.trace][-1] == f"event;{DONE_EVENT}"


def _leaves(exc: BaseException) -> list[BaseException]:
    """The non-group exceptions inside a possibly-nested `ExceptionGroup`.

    The class matters here, not the message: `CompositionRefused` is what makes a refusal an
    ANSWER a fork child relays rather than a crash its parent hangs on, so a test that matched
    text would pass for an untyped refusal with the right words in it."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


def _flatten(exc: BaseException) -> str:
    """A gather branch's failure arrives inside an `ExceptionGroup` (the documented L5 shape),
    so a message assertion has to look through one."""
    if isinstance(exc, BaseExceptionGroup):
        return " | ".join(_flatten(sub) for sub in exc.exceptions)
    return f"{exc}"


def test_the_frames_that_make_a_spawn_unwakeable_are_exactly_the_ones_declared():
    """The canary: THIS is the divergence the refusal prevents, kept checkable after the
    refusal makes it unreachable — the "commit the counterexample beside the guard" shape
    `test_op_key_injectivity.py:110` uses.

    Computed through the same oracle the handlers are pinned against, so it cannot drift
    from what a frame actually does."""
    for mk in BUILDERS.values():
        built = mk(0)
        awaited = qualified_event_name(*built.frames, name=DONE_EVENT).stored()
        if built.frames:
            # Every frame-bearing combinator WOULD rescope the join away from its emitter.
            # That is the hazard; the policy decides what to do about it, and the two answers
            # differ — reroute where the frame is a name, refuse where it is a branch slot.
            assert awaited != DONE_EVENT, f"{built.name} would rescope the join"
        else:
            assert awaited == DONE_EVENT, f"{built.name} applies no frame"


# ------------------------- the terminals: one discriminator, two mechanisms, and the DIAGONAL


TERMINAL_CASES = [
    (mk(0), terminal, terminal_disposition(mk(0)))
    for mk in BUILDERS.values()
    for terminal in TERMINALS.values()
]


@pytest.mark.parametrize(
    ("outer", "terminal", "disposition"),
    TERMINAL_CASES,
    ids=[f"{o.name}o{t.name}" for o, t, _ in TERMINAL_CASES],
)
def test_a_task_addressing_terminal_is_refused_exactly_where_the_body_runs_in_a_branch(
    outer, terminal, disposition
):
    """The COLUMN law: `outer ∘ fork` and `outer ∘ respawn`, disposition DERIVED from `is_branch`.

    The two members with no Effect-hole get a column the way a hole gets a row, so each is a
    cell the table ranges over.

    What the parametrization asserts that two separate tests could not: **one discriminator
    governs two entirely different mechanisms.** A fork's join is refused because a branch
    coordinate would rescope an ABSOLUTE name away from the task that emits it; a `Respawn` is
    refused because a generation boundary ends the whole task, so a branch that respawned would
    end its siblings' task too — and on the concurrent path kill the worker, since
    `_ChainContinues` cannot become a value at the barrier the way `_GatherPark` does. Different
    op kind, different failure, same predicate. Measured for both rather than assumed from one.

    And the four OK cells are load-bearing in the other direction: `scoped ∘ respawn` is LEGAL
    (ordinary namespacing; the task ends and the interpreters agree via `Ended`), so a guard that
    refused every frame would be the F4 mistake — refusing name-correct compositions."""
    handler = RecordingHandler(
        responses={**expected_responses(outer, outer), **terminal.responses}
    )
    workflow = compose_terminal(outer, terminal)

    if disposition is Disposition.LOUD:
        # A gather branch wraps the refusal in an `ExceptionGroup` (the documented L5 shape)
        # whose own `str` is "unhandled errors in a TaskGroup", so `pytest.raises(match=...)`
        # would test the wrapper rather than the message.
        caught: BaseException | None = None
        try:
            handler.run(workflow)
        except (ValueError, BaseExceptionGroup) as exc:
            caught = exc
        assert caught is not None, (
            f"{outer.name} runs its body in a gather branch and must refuse {terminal.name}"
        )
        message = _flatten(caught)
        for named in terminal.refusal_names:
            assert named in message, f"the refusal must name {named!r}"
    else:
        result = handler.run(workflow)
        assert isinstance(result, terminal.outcome), (
            f"{outer.name} ∘ {terminal.name} must reach the boundary, not {result!r}"
        )
        if isinstance(result, Respawned):
            # The INNER-direction dual of the row law's identity element: a generation boundary
            # reached from inside a frame must be the SAME boundary as one reached bare. The
            # frames namespace the ops, not the chain. Measured identical across all four OK
            # cells and a bare respawn.
            bare = RecordingHandler(responses={}).run(terminal.body(None))
            assert isinstance(bare, Respawned)
            assert (result.task, result.run_id, result.generation) == (
                bare.task,
                bare.run_id,
                bare.generation,
            )
            assert result.next_params() == bare.next_params()


def test_a_generation_frames_nothing_and_everything_composes_inside_one():
    """The ROW law, `respawn ∘ X`, and the asymmetry it records.

    `respawn` is in two registries on purpose. As the OUTER it is an ordinary hole contributing
    NO frames: a generation is a fresh task with a fresh checkpoint store, and `Turn` carries no
    scope: a per-generation scope would break the acceptance test that the cycle view at
    `generations=20` equals the one at `generations=None`. As the INNER it is a `Terminal` that
    ends the task and hands its enclosing combinator nothing.

    "Anything may sit inside a generation" is what this test measures. The converse does not
    hold: four of the seven holes take a `respawn` inside them (the column law above), and
    `scoped ∘ respawn` is legal."""
    generation = generation_outer()
    assert generation.frames == (), "a generation is a fresh task; it namespaces nothing"

    for mk in BUILDERS.values():
        inner = mk(1)
        handler = RecordingHandler(responses=expected_responses(generation, inner))
        result = handler.run(compose(generation, inner))
        actual = [entry.key.stored() for entry in handler.trace]

        assert result == "L", f"respawn ∘ {inner.name} must return its body's value"
        assert actual == expected_keys(generation, inner), inner.name
        # The homomorphism's identity element, measured: enclosing a body in a generation leaves
        # every key exactly as it was without one.
        assert actual == expected_keys(inner), f"a generation must not reframe {inner.name}"


def test_the_diagonal_is_where_the_open_defects_live():
    """A∘A for every member: the cells a table indexed by pairs of DISTINCT members would miss.

    The two silent defects of task-addressing composition both sit on the diagonal:

    * `fork ∘ fork`, a counterfactual that spawns a counterfactual. The child's event world
      (`RenamedAwaitCtx`) prepends `fork:{child};` to the grandchild's ABSOLUTE done event, so
      the join would park forever on a name nothing produces. It is refused, and verified in
      `test_fork_durable.py` rather than here, because a fork child's ctx exists only on the
      durable path: the recorder has no rename to compose with itself.
    * `respawn ∘ respawn`, refused LOUD; the test below pins it.

    The seven combinator-hole diagonals are all OK, which is worth asserting rather than
    assuming: it is what makes "the diagonal is dangerous" a claim about task-addressing members
    specifically, not about self-composition in general."""
    diagonal = [cell for cell in OBSERVED if cell[0] == cell[1]]
    assert len(diagonal) == len(BUILDERS), "every member must have a diagonal cell"
    # Read off the measured table: re-running the cells would be a strict subset of
    # `test_frames_concatenate_in_nesting_order`. What is NOT covered elsewhere is the two
    # terminal diagonals.
    assert all(OBSERVED[cell] is Disposition.OK for cell in diagonal), (
        "the combinator-hole diagonals are all OK, which is what makes 'the diagonal is "
        "dangerous' a claim about task-addressing members, not about self-composition"
    )

    # `fork ∘ fork`'s cell, recorded as a pointer to where it is verified: the table says which
    # cells exist and what they mean; it does not pretend to a ctx it cannot build.
    assert "fork" in TERMINALS
    assert (
        Path("tests/test_fork_durable.py")
        .read_text()
        .count("test_a_fork_child_that_spawns_and_joins_a_grandchild_is_refused_not_deadlocked")
    ), "the fork diagonal's verification must exist where the fork-child ctx does"


def test_respawn_refuses_to_nest_and_the_refusal_names_the_product_carry():
    """`respawn ∘ respawn` is refused LOUD.

    It is the same category error as `fork ∘ respawn`: **one task has one lifecycle, so exactly
    one chain can own it.** A generation boundary ends the whole task, so an inner chain's
    boundary ends the OUTER chain's task, and there is no inner task for it to end.

    Without the refusal, the escaping `Respawned` carries the INNER chain's `task`, `run_id` and
    carry, so the outer chain's next generation resumes under the inner chain's name with the
    inner chain's state: three silent wrong answers and no error at any moment.

    The refusal has to name what the author wanted, because nesting is not a missing feature but
    a mis-factored one: two loop axes on one task is ONE chain with a product carry, and an inner
    loop that genuinely needs its own task boundary needs its own TASK (a spawned child chain).

    **This does not settle chain identity.** Two *sequential* chains in one run (`respawn(A)`
    completing with `Done`, then `respawn(B)`) still share one params namespace and still need a
    discriminator."""
    outer_chain = Chain(task="outer-task", state=ChainState(n=100), run_id="outer-run")
    inner_chain = Chain(task="inner-task", state=ChainState(n=1), run_id="inner-run")

    def nested():
        def outer_step(_state, _turn):
            def inner_step(state, _t):
                return Again(ChainState(n=state.n + 1))
                yield  # pragma: no cover - unreachable

            yield from respawn(inner_step, inner_chain)
            raise AssertionError("the outer step cannot resume past its inner boundary")

        return (yield from respawn(outer_step, outer_chain))

    with pytest.raises(CompositionRefused) as caught:
        RecordingHandler(responses={}).run(nested)

    message = str(caught.value)
    assert "inner-task" in message, "the refusal must name the chain it refused"
    assert "one lifecycle" in message  # the reason
    assert "product" in message  # the thing the author actually wanted
    assert "its own chain" in message  # and the other one
    # The escape hatch must be honest about what is BUILT: joining a spawned chain needs a
    # chain-done event, which does not exist. A refusal must not recommend an unbuilt path.
    assert "cannot JOIN one yet" in message

    # It is a `CompositionRefused`, so a fork child hitting it ANSWERS its parent rather than
    # crashing — the property the whole family shares, asserted rather than assumed.
    assert isinstance(caught.value, ValueError)  # unchanged for every existing call site


@pytest.mark.parametrize("between", list(BUILDERS), ids=list(BUILDERS))
def test_respawn_refuses_to_nest_THROUGH_any_combinator(between):
    """`respawn ∘ A ∘ respawn` for every hole A — the transitive question, answered by test.

    The question is whether the refusal holds through an intervening sequence rather than only at
    depth 2. It does, and the mechanism is why: the flag is a `ContextVar` set around the step's
    BODY, so every combinator that is ordinary workflow code inside that body (`scoped`, `route`,
    `descend`, `hoisted`) is transparent to it at any depth — and `gather`/`recurse` interpose a
    branch, where a `Respawn` is refused earlier by the branch guard. Two different refusals
    cover the seven cells, which is why this asserts the CLASS (`CompositionRefused`) rather than
    one message.

    The `fork ∘ A ∘ fork` twin travels by a different mechanism (the rename lives on the ctx and
    `_root_ctx` retains it), and this recorder suite cannot pin it, because a fork child's ctx
    exists only on the durable path."""
    hole = BUILDERS[between](1)
    inner_chain = Chain(task="inner-task", state=ChainState(), run_id="inner-run")
    outer_chain = Chain(task="outer-task", state=ChainState(), run_id="outer-run")

    def inner_body():
        def inner_step(state, _t):
            return Again(ChainState(n=state.n + 1))
            yield  # pragma: no cover - unreachable

        return (yield from respawn(inner_step, inner_chain))

    def nested():
        def outer_step(_state, _turn):
            yield from hole.hole(lambda: inner_body())
            raise AssertionError("the outer step cannot resume past its inner boundary")

        return (yield from respawn(outer_step, outer_chain))

    handler = RecordingHandler(responses=expected_responses(hole, hole))
    caught: BaseException | None = None
    try:
        handler.run(nested)
    except BaseException as exc:
        caught = exc

    assert caught is not None, f"respawn ∘ {between} ∘ respawn must be refused"
    assert all(isinstance(leaf, CompositionRefused) for leaf in _leaves(caught)), (
        f"the refusal must be the typed class so a fork child ANSWERS: got {caught!r}"
    )


# -------------------------------------------------- COSTLY: both orders right, one cheap


FANOUT = 3


def test_hoisting_over_a_gather_costs_one_disclosure_and_under_it_costs_N():
    """`hoisted ∘ gather` is OK; `gather ∘ hoisted` is COSTLY — measured, not asserted by fiat.

    This is the pair where the disjunction law says something. Both orders are *correct* —
    same result, same leaf keys — so no guard should refuse either: a skill only some
    branches need genuinely belongs inside them. What differs is price, which is why
    `Disposition` has a third value rather than a boolean.

    Rule 4 (`combinators.py`: "never call `activate_skill` inside a branch") is the
    combinator's whole reason for existing and is pure author discipline — nothing enforces
    it, and nothing should. What the table adds is that the cost is now a number in a test
    instead of a sentence in a docstring. SkillsBench M1 measured this burn on real tokens.

    No engine divergence here, and the neighbouring one is worth NOT confusing it with: two
    activations at the SAME level diverge (the recorder cans by name and returns one pin, the
    durable engine's `name#2` suffix performs a fresh disclose — `skills.py:103`). Per-branch
    activation is not that case: each branch's frame makes the key distinct, so both engines
    disclose N times and agree."""
    pin = Pin(name="sk", content_hash="h", body="b")
    # Composed, not spelled, per the module docstring's discipline.
    responses: dict[str, object] = {"skill:sk,activate": pin, LEAF: "L"}
    for i in range(FANOUT):
        branch = GatherBranch(0, i)
        responses[qualified_event_name(branch, name=LEAF).stored()] = "L"
        responses[qualified_event_name(branch, name="skill:sk,activate").stored()] = pin

    def branches():
        return [(lambda: observable(LEAF)) for _ in range(FANOUT)]

    def outside():
        return (yield from hoisted(("sk",), lambda _pins: gather(branches())))

    def inside():
        def branch():
            return (yield from hoisted(("sk",), lambda _pins: observable(LEAF)))

        return (yield from gather([branch for _ in range(FANOUT)]))

    def activations(workflow):
        handler = RecordingHandler(responses=responses)
        result = handler.run(workflow)
        keys = [entry.key.stored() for entry in handler.trace]
        return result, sum(k.endswith("skill:sk,activate") for k in keys)

    out_result, out_activations = activations(outside)
    in_result, in_activations = activations(inside)

    assert out_result == in_result == ["L"] * FANOUT, "both orders must be CORRECT"
    assert out_activations == 1
    assert in_activations == FANOUT
    assert in_activations > out_activations, "the COSTLY order is the one that pays per branch"


# --------------------------------------------------------------- the no-silent law


def test_no_pair_is_silent():
    """The table's headline law: no cell composes, is wrong, and says nothing.

    `observed_disposition` RUNS each cell, and returns `None` for the silent outcome: composed,
    returned something other than the leaf's value, raised nothing. `Disposition`
    having no `SILENT` member is the design; this is that absence made checkable.

    Its own scope stays the honest caveat: `BUILDERS` squared plus the two `TERMINALS`, not
    `run_agent` / `improve` / `run_code`, each excluded for a reason stated at
    `BUILDERS`. `Disposition.COSTLY` is measured by its own test rather than tagged per cell:
    both orders of the costly pair are `OK` here, which is exactly the point of having a third
    value at all."""
    assert len(OBSERVED) == EXPECTED_PAIRS
    silent = [cell for cell, disposition in OBSERVED.items() if disposition is None]
    assert not silent, f"cells that composed, were wrong, and said nothing: {silent}"
    assert not hasattr(Disposition, "SILENT")


def test_the_oracle_is_not_degenerate():
    """The oracle's degeneracy, bounded and named.

    Leaf-suffix and length checks after `actual == expected_keys(...)` hold by construction,
    since `expected_keys` always appends the leaf, so they cannot guard against an empty trace.
    The ceiling is here instead, stated as the number it is: 45 of the 49 park cells carry
    at least one frame, and the 4 that do not are the frameless corner (`route` and `hoisted`
    contribute nothing), where the park assertion genuinely does reduce to `ev:1 == ev:1`. That
    is a bounded, named degeneracy rather than an unexamined one."""
    framed = [(o.name, i.name) for o, i in pairs() if enclosing_frames(o, i)]
    bare = [(o.name, i.name) for o, i in pairs() if not enclosing_frames(o, i)]

    assert len(framed) == 45
    assert set(bare) == {(a, b) for a in ("route", "hoisted") for b in ("route", "hoisted")}
    for outer, inner in pairs():
        # `.stored()` once: this test is ABOUT the oracle's text form — that the composed name
        # ends in the awaited name — so it reads the wire spelling rather than the identity.
        expected = expected_park(outer, inner).stored()
        assert (expected != PARK_EVENT) == bool(enclosing_frames(outer, inner))
        assert expected.endswith(PARK_EVENT), "the oracle must always end in the awaited name"


def test_a_bare_leaf_needs_no_frames():
    """The identity end of the algebra: with nothing wrapping it, the leaf carries frames from
    nowhere — only its own arm term.

    Not "the leaf is its own key" — the arm term means it never is. The identity element of the
    algebra is what this test is about: no FRAMES, not no wrapping at all."""
    handler = RecordingHandler(responses={LEAF: "L"})
    assert handler.run(leaf()) == "L"
    assert [entry.key.stored() for entry in handler.trace] == [step_key(LEAF).stored()]
