"""`depth-grant:`'s aliasing: one answer settles exactly ONE op-occurrence.

`depth-grant:` is declared `Scope.SETTLEMENT` (`budget.py:43`), which promises that one answer
settles exactly ONE op-occurrence. `depth-grant:{run_id},{generation},depth={depth}` cannot keep
it alone: it is a complete function of its three fields and of nothing that says *which* descent
is asking, so without a further coordinate two questions compose one name and the first answer
settles the second.

**The four cases, which is the fastest way to read this file:**

| case | engine | outcome |
|---|---|---|
| two descents in one run | SQLite | separated: B parks |
| the same, through a `Grantor` | SQLite | separated: B parks |
| a real `respawn` chain | in-process | separated: each generation names its own |
| two hand-spawned tasks, one `run_id` | Absurd | **aliases, and should** |

The first two are separated by one mechanism, because the coordinate is applied by the WALK
(`placing`), which sits above both answerers. The third is separated differently: two
generations are two tasks, so no within-frame ordinal reaches them and the GENERATION goes into
the name instead, threaded by `respawn` through `ops.CHAIN_GENERATION` so a call site cannot
forget it.

**The fourth is a SIMULATION of the third, and the distinction is the point.** It spawns two
tasks by hand sharing a `run_id` (the shape respawn produces) but never goes through `respawn`,
so nothing tells those tasks which generation they are and both name generation 0. It aliases,
correctly. Keeping it stops the third test from being read as coverage of something it does not
touch: a green simulation is evidence about a hand-rolled pair of tasks, not about respawn.

The cases are not redundant; each refutes a different candidate placement of the coordinate. The
grantor case decides it: `human_grant` composes the name in author-reachable code, so a
coordinate applied to the built-in arm's name (`refill_levels`' `run_id` branch) never reaches
it, and only a handler-applied one does. The walk is the mint because the await occurrence is
decided where the op is gated.
"""

import uuid
from typing import Any

import pytest
from _durable import absurd, pg_ready

from effective.api import step
from effective.budget import Grant, depth_grant_name
from effective.combinators import Answered, Deeper, descend, grant_cascade, human_grant
from effective.domain import CallTool, DomainOp
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler

pytestmark = pytest.mark.adversarial


class _Tool:
    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        return "deeper"


def _judge_for(tag: str, levels: list[str]):
    """A judge that drills one level past its budget, then answers — so a descent granted
    nothing terminates and a descent granted more drills again. `levels` records every entry,
    which is how the test sees an unasked-for authorization being spent."""

    def judge(ctx, level):
        levels.append(f"{tag}@{level.depth}")
        yield from step("judge", CallTool(name="op", args={"t": tag}, result_schema=str))
        if level.depth >= 1:
            return Answered(f"{tag}@{level.depth}")
        return Deeper(f"{tag}:{level.depth}")

    return judge


_ALIASED = "aliased: B consumed A's grant"
_PARKED = "parked: B asked its own question"


def _outcome(snapshot: Any, levels: list[str]) -> str:
    """Which of the two states the run ended in, as a value rather than a pile of assertions.

    Naming both is what lets the transitional pins accept either without going vague: the test
    still says exactly what it saw, and the tightening step is one edit to the expected value
    rather than a rewrite."""
    entered = [x for x in levels if x.startswith("B@")] or [
        x for x in levels if x.startswith("g@")
    ]
    if snapshot is not None and snapshot.state == "completed" and len(entered) > 1:
        return _ALIASED
    if len(entered) == 1:
        return _PARKED
    return f"neither: state={None if snapshot is None else snapshot.state} levels={levels}"


def _two_descents(descent_kwargs: dict[str, Any]) -> tuple[Any, list[str]]:
    """Run two independent `descend(...)` calls in ONE task and ONE run, emit exactly one grant
    (addressed to the first), and return the final snapshot plus the levels each judge entered.

    SQLite suffices because both awaits are in one task — its events are keyed
    `(task_id, name)`, so this is the within-task alias, not the cross-task case."""
    levels: list[str] = []
    app = SqliteApp(":memory:")

    @app.register_task("two-descents")
    def task(params, ctx):
        def wf():
            a = yield from descend("ctx", _judge_for("A", levels), budget=1, **descent_kwargs)
            b = yield from descend("ctx", _judge_for("B", levels), budget=1, **descent_kwargs)
            return [a, b]

        return DurableHandler(ctx, _Tool()).run(wf)

    task_id = app.spawn("two-descents", {"run_id": "r1"})
    app.work_batch()  # descent A runs to exhaustion and parks
    app.emit_event(
        depth_grant_name("r1", depth=1, generation=0).stored(), Grant(add_depth=2).model_dump()
    )
    snapshot = app.run_until_result(task_id)
    app.close()
    return snapshot, levels


def test_two_descents_in_one_run_do_not_share_one_grant():
    """The hazard, on the substrate's built-in arm (`refill_levels`' `run_id` branch).

    Descent A exhausts at depth 1 and parks. A human grants `add_depth=2`, once, to A. Were
    descent B to compose the byte-identical name when it exhausts at the same depth,
    Absurd/SQLite would answer it from the recorded fact: B would drill a second level on an
    authorization it never asked for, and nothing would record that two descents shared one
    answer.

    `A@0` appears twice because the park replays depth 0 from its checkpoint. That is ordinary
    replay, not part of the defect.

    B parks: `snapshot.state` is not `completed` until a SECOND grant is emitted, and B's
    levels are `['B@0']` alone."""
    snapshot, levels = _two_descents({"run_id": "r1"})

    assert _outcome(snapshot, levels) == _PARKED, (snapshot.state, levels)
    assert levels == ["A@0", "A@0", "A@1", "B@0"], levels
    # B is waiting on ITS OWN question — the same base name, second occurrence — so the run does
    # not finish until someone answers it. That is the whole point: a second authorization now
    # has to be given rather than inherited.
    assert snapshot.state == "waiting", snapshot.state


def test_the_grantor_path_is_separated_identically():
    """The same hazard through an author-supplied `Grantor`.

    `descend`'s two arms are "two answerers, one park" BY DESIGN: `human_grant` composes exactly
    the name the built-in arm does, so the two agree by construction rather than by contract
    (`combinators.refill_levels`). That is why a coordinate applied to the built-in arm's name
    (`refill_levels`' `run_id` branch) is not a fix: the grantor arm composes its own. Whatever
    assigns the coordinate has to sit where BOTH arms pass, which is the handler.

    The outcome is identical to the built-in arm's: B parks, same assertions. This is the row
    that matters: the coordinate is applied by the handler, so it reaches a name composed in
    author code that the substrate never sees."""
    snapshot, levels = _two_descents({"grantor": grant_cascade([human_grant("r1")])})

    assert _outcome(snapshot, levels) == _PARKED, (snapshot.state, levels)
    assert levels == ["A@0", "A@0", "A@1", "B@0"], levels


@pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")
def test_two_generations_share_one_grant_across_tasks():
    """The cross-TASK half — what no within-frame ordinal reaches.

    A respawn chain keeps `run_id` stable across generations by design, and each generation is a
    fresh task. Absurd's events are queue-global and its checkpoint counter is keyed
    `(task_id, checkpoint_name)`, so generation 1 restarts at ordinal 1 exactly as generation 0
    did and composes the identical `depth-grant:{run}:1`. It is answered instantly from
    generation 0's grant.

    The run id folds a constant discriminator in exactly where an author-supplied descent
    `Segment` would sit, which shows that a call-site constant cannot separate generations. Only
    threading the GENERATION into the name does, and these hand-spawned tasks are never told
    theirs, so generation 1 completes on generation 0's grant with levels `['g@0', 'g@1']`."""
    app = absurd()
    suffix = uuid.uuid4().hex[:8]
    run = f"r-{suffix}-triage"
    task_name = f"depth-gen-{suffix}"
    levels: list[str] = []

    @app.register_task(task_name, default_max_attempts=1)
    def task(params, ctx):
        def wf():
            return (yield from descend("ctx", _judge_for("g", levels), budget=1, run_id=run))

        return DurableHandler(ctx, _Tool()).run(wf)

    def spawn(generation: int):
        spawned = app.spawn(task_name, {"run_id": run, "g": generation})
        return spawned["task_id"] if isinstance(spawned, dict) else spawned

    generation_0 = spawn(0)
    app.work_batch()  # generation 0 exhausts and parks
    app.emit_event(
        depth_grant_name(run, depth=1, generation=0).stored(), Grant(add_depth=2).model_dump()
    )
    result_0 = app.run_until_result(generation_0)
    assert result_0 is not None
    assert result_0.state == "completed"
    generation_0_levels = list(levels)

    result_1 = app.run_until_result(spawn(1))
    generation_1_levels = levels[len(generation_0_levels) :]

    assert result_1 is not None, "generation 1 produced no result at all"
    assert result_1.state == "completed", (
        "generation 1 was expected to be answered instantly by generation 0's grant; a park "
        "here means the generation reached the name — invert this test"
    )
    assert generation_1_levels == ["g@0", "g@1"], (
        f"generation 1 drilled on generation 0's authorization: {generation_1_levels}"
    )


def test_a_real_respawn_chain_separates_each_generation_s_depth_grant():
    """The cross-generation case through the machinery that actually threads the coordinate.

    **The reproduction above is a SIMULATION, and this is what it does not cover.** It spawns two
    tasks by hand sharing a `run_id` — the shape respawn produces — but it never goes through
    `respawn`, so `ops.CHAIN_GENERATION` is unset and both generations name generation 0. It
    still aliases, correctly: nothing told those tasks which generation they were. A real chain
    is told, by `respawn`, and that is the difference this test pins.

    Worth stating plainly because it is the instrument-measures-a-copy shape: a green simulation
    would have been evidence about a hand-rolled pair of tasks, not about respawn.

    In-process on the recording handler, because what is being asserted is the NAME each
    generation composes — `Respawned.next_params()` is how a generation boundary is expressed
    there, and the durable engines' event semantics are already pinned elsewhere."""
    from effective.combinators import Again, Chain, Done, Turn, respawn
    from effective.handlers.recording import RecordingHandler, Respawned, Suspended

    RUN = "chain-r"
    asked: list[str] = []

    def judge(ctx, level):
        yield from step("judge", CallTool(name="op", args={}, result_schema=str))
        return Answered("done") if level.depth >= 1 else Deeper("deeper")

    def chain_step(state, turn: Turn):
        # A `descend` inside a chain generation, written EXACTLY as it is outside one — no
        # generation argument at the call site, which is the property being tested.
        yield from descend("ctx", judge, budget=1, run_id=RUN)
        return Done("finished") if turn.final else Again(state)

    def run_generation(generation: int, params: dict[str, Any]):
        chain = Chain(task="w", state=None, run_id=RUN, generation=generation, params=params)
        outcome = RecordingHandler(responses={"d:0;judge": "v", "d:1;judge": "v"}).run(
            lambda: respawn(chain_step, chain, budget=None)
        )
        if isinstance(outcome, Suspended):
            asked.append(outcome.awaiting.stored())
        return outcome

    parked_0 = run_generation(0, {})
    assert isinstance(parked_0, Suspended), parked_0
    parked_1 = run_generation(1, {})
    assert isinstance(parked_1, Suspended), parked_1

    # Two generations, two DISTINCT questions. Without the generation coordinate both would
    # read `depth-grant:chain-r:1`.
    assert asked == ["depth-grant:chain-r,0,depth=1", "depth-grant:chain-r,1,depth=1"], asked
    assert len(set(asked)) == 2

    # And the boundary machinery is untouched: a chain that is NOT parked still respawns.
    assert Respawned is not None


def test_a_chain_generation_recomposes_the_same_grant_name_on_REPLAY():
    """Generation 1's grant name, re-derived by the replay walk rather than the recording one.

    **The soundness argument for `CHAIN_GENERATION` was generic until this test.** That a
    workflow-side contextvar survives replay was established on a bare probe (`CHAIN_DEPTH`'s
    precedent, a fork's P1 case, and the `layer_run_state` divergence that motivated
    `ReplayHandler` entering the run scope). None of those is *this* ambient, in a chain,
    composing an authority name — which is the thing that breaks silently if the argument is
    wrong, because a replay that composes a different name binds a different park.

    **Generation 1, not 0, and that is the whole point of the fixture.** `depth_grant_name`
    defaults the generation to 0, so a replay that failed to re-derive the ambient would compose
    the byte-identical name at generation 0 and this test would pass while proving nothing. At
    generation 1 the recorded name is `depth-grant:chain-r,1,depth=1` and the un-derived one is
    `…:0:1`, so a failure surfaces as a `ReplayMismatch` naming both.

    The park is RESUMED before the trace is taken: a trace that ends at a park has no recorded
    result for the await, so replay would report an extra op rather than compare the name."""
    from effective.combinators import Chain, Done, Turn, respawn
    from effective.handlers.recording import RecordingHandler, Suspended
    from effective.handlers.replay import ReplayHandler

    RUN = "chain-r"

    def judge(ctx, level):
        yield from step("judge", CallTool(name="op", args={}, result_schema=str))
        return Answered("done") if level.depth >= 1 else Deeper("deeper")

    def chain_step(state, turn: Turn):
        yield from descend("ctx", judge, budget=1, run_id=RUN)
        return Done("finished")

    def program():
        chain = Chain(task="w", state=None, run_id=RUN, generation=1, params={})
        return (yield from respawn(chain_step, chain, budget=None))

    recorder = RecordingHandler(responses={"d:0;judge": "v", "d:1;judge": "v"})
    parked = recorder.run(program)
    assert isinstance(parked, Suspended), parked
    assert parked.awaiting == depth_grant_name(RUN, depth=1, generation=1), parked.awaiting
    assert parked.resume(Grant(add_depth=1)) == "finished"

    recorded = [entry.key.stored() for entry in recorder.trace]
    assert f"event;{depth_grant_name(RUN, depth=1, generation=1).stored()}" in recorded, recorded

    # The walk that re-derives rather than records. A `ReplayMismatch` here means the ambient
    # did not survive, and its message names the two keys.
    assert ReplayHandler(recorder.trace).run(program) == "finished"
