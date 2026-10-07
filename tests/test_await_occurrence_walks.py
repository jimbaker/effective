"""What per-occurrence identity does each WALK surface for two awaits of one name?

The conformance table any candidate await-occurrence coordinate has to pass, and the reason the
coordinate is more than an engine fix. `depth-grant:` is declared `Scope.SETTLEMENT`: one answer
settles one op-occurrence, and the substrate cannot keep that promise unless every interpreter
agrees on *which* occurrence is asking.

It is **one walk of six** once the fork drivers are counted:

===========================  =========================================  ==========
walk                         per-await identity                         distinct
===========================  =========================================  ==========
recording                    ``['event;q', 'event;q']``                 1 of 2
replay                       none of its own — matches POSITIONALLY     --
SQLite durable               ``[]`` (no await checkpoint at all)        0 of 0
Absurd durable               ``['$awaitEvent:q', '$awaitEvent:q#2']``   **2 of 2**
``fork.live_drive``          ``['event;q', 'event;q']``                 1 of 2
``fork.measured_drive``      ``['event;q', 'event;q']``                 1 of 2
===========================  =========================================  ==========

Only Absurd has it, and it has it by accident of storage: the SDK checkpoints an await's
delivered payload under ``$awaitEvent:{name}``, so its general duplicate-*checkpoint-name* rule
yields ``#2`` for the second asker. Nobody decided awaits need occurrences. SQLite's
``await_event`` (``sqlite.py:293``) peeks the events table and raises ``_Suspend``, checkpointing
nothing, while its ``step()`` (``:270``) *does* apply ``name.occurrence(count)``, so the port
took the shared dup-name rule where it needed it and skipped it where it did not.

**The two fork drivers are the rows a smaller count misses.** ``fork._replay_scoped`` carries a
``FramePosition`` (``fork.py:233``, ``placing`` at ``:245``), as does ``replay_prefix``
(``:209``). The two that carry none are ``live_drive`` (``:303``) and ``_MeasuredRun.drive``
(``:560``), and **both resolve await names**, through ``sandbox.inspect_only`` (``:113``),
which looks up the BARE op name. So a candidate that passes the four-row table can still
compose an unqualified name in a fork tail and miss its grant.

That shared lookup is also the good news: ``inspect_only`` is one seam both fork drivers go
through, the same "one transition, many drivers" shape as ``enforce_measured`` and
``permission.decide``.

**The table above is the state for a name that declares NO reach**, and it stays true: the
coordinate is applied only where a namespace declared `Scope.SETTLEMENT`, so a plain `event;q`
is untouched by design. The five tests below the divider are the same six
walks asked the same question about a `depth-grant:` name, and it reads 2 distinct of 2
everywhere. Keeping both is the point: one pins that the mechanism fires, the other
pins that it does not fire anywhere it was not asked to.
"""

import uuid
from typing import Any

import psycopg
import pytest
from _durable import DSN, absurd, pg_ready, run_until_result

from effective.api import await_event, qualified_event_name
from effective.budget import BUDGET_GRANT, Grant, MeasuredBudget, depth_grant_name
from effective.cost import Usage
from effective.domain import DomainOp
from effective.fork import live_drive, measured_drive
from effective.govern import GOVERN
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import placed_await_name, placing
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import FramePosition, Key, Scope, Segment, Tag, compose_key
from effective.ops import AwaitEvent
from effective.parked import read_sqlite_parked_conn
from effective.sqlite import SqliteApp

pytestmark = pytest.mark.adversarial

NAME = Key.parse("q")
ANSWER = {"n": 1}

GRANT = depth_grant_name("r1", depth=1, generation=0)
"""A name whose namespace DECLARED `Scope.SETTLEMENT` — so the walk qualifies it."""

GRANT_ANSWER = {"add_depth": 0}


class _Tool:
    def run(self, op: DomainOp) -> Any:
        return 1

    def run_metered(self, op: DomainOp) -> tuple[Any, Usage]:
        return 1, Usage()


def _two_awaits():
    """One name, asked twice. The whole probe — anything more would measure something else."""
    a = yield from await_event(NAME, dict)
    b = yield from await_event(NAME, dict)
    return [a, b]


def test_the_recorder_serves_one_canned_answer_to_both_asks():
    """The recorder has no store to read a coordinate off, so it would have to COUNT — and its
    canned-answer lookup has no occurrence in it (`recording.py:706-709`: frame-qualified name,
    then bare name, stop). One `responses={"q": ...}` entry silently serves both asks.

    This is why the recorder is the hard case: 113 `responses=` sites across the corpus, and
    the failure mode of getting it wrong is a fixture that serves the *wrong occurrence* rather
    than one that errors."""
    handler = RecordingHandler(responses={"q": ANSWER})
    result = handler.run(_two_awaits)

    keys = [entry.key.stored() for entry in handler.trace]
    assert keys == ["event;q", "event;q"], keys
    assert len(set(keys)) == 1, "the recorder distinguishes the two asks — invert this test"
    assert result == [ANSWER, ANSWER], "one canned entry served both asks"


def test_replay_has_no_identity_of_its_own_and_matches_positionally():
    """Replay carries the ordinal only as trace POSITION and never names it, so it inherits
    whatever the recorder assigned. It cannot be the walk that fixes this, and it cannot be
    the walk that breaks it either — but it is the walk that will disagree loudly if the
    recorder and the durable handler are given different counters."""
    recorded = RecordingHandler(responses={"q": ANSWER})
    recorded.run(_two_awaits)

    assert ReplayHandler(recorded.trace).run(_two_awaits) == [ANSWER, ANSWER]


def test_sqlite_writes_no_await_checkpoint_at_all():
    """SQLite parks in `tasks.waiting_event` and checkpoints nothing, so there is no row for a
    coordinate to live in. Defensible minimalism rather than an oversight: with events keyed
    `(task_id, name)`, a re-executing attempt just re-peeks and gets the same row.

    It is also why the byte-safety worry about making SQLite checkpoint awaits is small — an
    await has no thunk, so a missing await checkpoint means re-reading an event the table still
    holds, not re-running work."""
    app = SqliteApp(":memory:")

    @app.register_task("two")
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(_two_awaits)

    task_id = app.spawn("two", {"run_id": "r1"})
    app.work_batch()
    app.emit_event(NAME.stored(), ANSWER)
    snapshot = app.run_until_result(task_id)
    names = [
        row[0]
        for row in app.conn.execute(
            "SELECT name FROM checkpoints WHERE task_id=?", (str(task_id),)
        )
    ]
    app.close()

    # The anti-vacuity pair, and it is load-bearing: `names == []` is ALSO what a workflow that
    # never ran produces, which is exactly the assertion-the-defect-satisfies shape. Pin that
    # both awaits resolved first, so the empty list is evidence about the await path rather
    # than about nothing having happened. `run_until_result` is `TaskSnapshot | None` — a run
    # that never reached a terminal state is precisely the vacuous case, so narrow it here.
    assert snapshot is not None, "the task never reached a terminal state"
    assert snapshot.state == "completed", snapshot.state
    assert snapshot.result == [ANSWER, ANSWER], snapshot.result
    assert names == [], f"SQLite grew an await checkpoint: {names}"


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(
            lambda: live_drive(_two_awaits(), None, _Tool(), {NAME: ANSWER}).trace,
            id="live_drive",
        ),
        pytest.param(
            lambda: (
                measured_drive(
                    _two_awaits,
                    MeasuredBudget(run_id="r1", overall=100.0),
                    _Tool(),
                    {NAME: ANSWER},
                ).trace
            ),
            id="measured_drive",
        ),
    ],
)
def test_both_fork_drivers_alias_the_two_asks(drive):
    """The two walks a four-row table misses: neither carries a `FramePosition`, and both
    answer an await from `sandbox.inspect_only`, which looks the BARE name up in `grants`
    (`sandbox.py:113`). One grant entry answers both asks, in a driver whose whole job is to
    explore a counterfactual accurately.

    Parametrized over both because they are the same defect at one shared seam; a fix applied
    at `inspect_only` reaches both, and a fix applied in either driver alone would not."""
    keys = [entry.key.stored() for entry in drive()]

    assert keys == ["event;q", "event;q"], keys
    assert len(set(keys)) == 1, "a fork driver distinguishes the two asks — invert this test"


@pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")
def test_absurd_is_the_ONE_walk_that_already_has_the_coordinate():
    """And it has it for its own reasons, which is why "bring the others to parity" understates
    the work. Absurd checkpoints the delivered payload under `$awaitEvent:{name}`, so two asks
    on one name become two checkpoint rows and the SDK's duplicate-name rule disambiguates them
    exactly as it disambiguates two `step("tool:a")` calls. The coordinate falls out of storing
    the payload, not out of anyone deciding awaits need occurrences.

    **It is per-task, so it does not reach the cross-generation case** — `c_default` is keyed
    `(task_id, checkpoint_name)`, and a fresh generation restarts the counter. See
    `test_grant_aliasing.py::test_two_generations_share_one_grant_across_tasks`."""
    app = absurd()
    suffix = uuid.uuid4().hex[:8]
    name = Key.parse(f"q-{suffix}")
    task_name = f"two-awaits-{suffix}"

    def workflow():
        a = yield from await_event(name, dict)
        b = yield from await_event(name, dict)
        return [a, b]

    @app.register_task(task_name, default_max_attempts=1)
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(workflow)

    spawned = app.spawn(task_name, {"run_id": "r1"})
    task_id = spawned["task_id"] if isinstance(spawned, dict) else spawned
    app.work_batch()
    app.emit_event(name.stored(), ANSWER)
    run_until_result(app, task_id)

    with psycopg.connect(DSN) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT checkpoint_name FROM absurd.c_default WHERE task_id=%s "
                "ORDER BY checkpoint_name",
                (str(task_id),),
            )
        ]

    assert names == [f"$awaitEvent:{name.stored()}", f"$awaitEvent:{name.stored()}#2"], names
    assert len(set(names)) == 2, "Absurd stopped distinguishing the two asks"


# --- the same six walks, on a name that DECLARES a reach ------------------------------------


def _two_settlement_awaits():
    """Two asks on one `Scope.SETTLEMENT` name — the shape a repeated `descend` grant makes."""
    a = yield from await_event(GRANT, Grant)
    b = yield from await_event(GRANT, Grant)
    return [a, b]


def _grants() -> dict[Key, Any]:
    """One answer per OCCURRENCE. Two entries, not one, is the whole behavioural change: before
    the coordinate a single entry served both asks, which is the aliasing in miniature."""
    return {GRANT: GRANT_ANSWER, GRANT.occurrence(2): GRANT_ANSWER}


def test_the_recorder_distinguishes_two_asks_on_a_settlement_name():
    """The hard case: no store to read a coordinate off, so the recorder would otherwise have to
    count. It does not: it reads the mint `placing` published, and its canned lookup is untouched.

    So a fixture needs a second `responses` entry only if its workflow asks one SETTLEMENT name
    twice, and the suite has none."""
    handler = RecordingHandler(
        responses={GRANT.stored(): GRANT_ANSWER, GRANT.occurrence(2).stored(): GRANT_ANSWER}
    )
    handler.run(_two_settlement_awaits)

    keys = [entry.key.stored() for entry in handler.trace]
    assert keys == [f"event;{GRANT.stored()}", f"event;{GRANT.occurrence(2).stored()}"], keys
    assert len(set(keys)) == 2


def test_replay_rederives_the_same_occurrence_the_recorder_assigned():
    """Replay has no counter of its own — it re-runs the deterministic computation and `placing`
    mints again. If the two disagreed, `ReplayHandler` would raise `ReplayMismatch` on the key
    comparison rather than fail quietly, which is why this is a one-line assertion: the walk
    itself is the check."""
    recorded = RecordingHandler(
        responses={GRANT.stored(): GRANT_ANSWER, GRANT.occurrence(2).stored(): GRANT_ANSWER}
    )
    recorded.run(_two_settlement_awaits)

    assert ReplayHandler(recorded.trace).run(_two_settlement_awaits) is not None


def test_ONE_answer_no_longer_settles_both_asks_on_sqlite():
    """The property, stated the only way that cannot be faked: emit **one** answer and the run
    must NOT finish, because the second ask is a different question nobody has answered.

    A first draft of this test emitted both answers and asserted the events table held two rows —
    which asserts what the TEST emitted, not what the run asked, and passed with the coordinate
    mutated away. Asking "does one answer still settle two questions?" is the actual defect, and
    nothing but the coordinate makes it come out `waiting`."""
    app = SqliteApp(":memory:")

    @app.register_task("two-grants")
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(_two_settlement_awaits)

    task_id = app.spawn("two-grants", {"run_id": "r1"})
    app.work_batch()
    app.emit_event(GRANT.stored(), GRANT_ANSWER)  # ONE answer, to the first ask
    parked = app.run_until_result(task_id)

    # ... and the second occurrence is answerable, so this is a distinct live question rather
    # than a name nothing can ever satisfy.
    app.emit_event(GRANT.occurrence(2).stored(), GRANT_ANSWER)
    finished = app.run_until_result(task_id)
    app.close()

    assert parked is None or parked.state != "completed", f"one answer settled both asks: {parked}"
    assert finished is not None, "the task never reached a terminal state"
    assert finished.state == "completed", finished.state


def test_an_emitter_composes_the_second_asks_registration_rather_than_writing_it():
    """What an emitter does now that `qualified_event_name` refuses `#2` in its `name`.

    The sanctioned spelling is the composed key's own `occurrence`, and it has to equal the
    registration the run waits on, or waking the second ask is guesswork."""
    composed = qualified_event_name(name=GRANT.stored()).occurrence(2)

    app = SqliteApp(":memory:")

    @app.register_task("two-grants")
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(_two_settlement_awaits)

    task_id = app.spawn("two-grants", {"run_id": "r1"})
    app.work_batch()
    app.emit_event(GRANT.stored(), GRANT_ANSWER)
    app.run_until_result(task_id)
    waiting = [park.wake_event for park in read_sqlite_parked_conn(app.conn)]
    app.emit_event(composed.stored(), GRANT_ANSWER)
    finished = app.run_until_result(task_id)
    app.close()

    assert waiting == [composed.stored()], waiting
    assert finished is not None, "the task never reached a terminal state"
    assert finished.state == "completed", finished.state


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(
            lambda: live_drive(_two_settlement_awaits(), None, _Tool(), _grants()).trace,
            id="live_drive",
        ),
        pytest.param(
            lambda: (
                measured_drive(
                    _two_settlement_awaits,
                    MeasuredBudget(run_id="r1", overall=100.0),
                    _Tool(),
                    _grants(),
                ).trace
            ),
            id="measured_drive",
        ),
    ],
)
def test_both_fork_drivers_carry_the_occurrence_into_their_grant_lookup(drive):
    """The two walks that had no `FramePosition` at all. Both resolve an await through
    `sandbox.inspect_only`, which looks the name up in `grants` — so the occurrence has to be on
    the op BEFORE that lookup, and a driver that skipped it would compose an unqualified name and
    silently take the first occurrence's grant into a counterfactual."""
    keys = [entry.key.stored() for entry in drive()]

    assert keys == [f"event;{GRANT.stored()}", f"event;{GRANT.occurrence(2).stored()}"], keys


@pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")
def test_absurds_own_duplicate_name_rule_does_not_fire():
    """Absurd's own `#N` is not bypassed; it never triggers for two settlement-scoped awaits.

    The two awaits carry DIFFERENT names, so the SDK's duplicate-checkpoint-name rule never
    fires. The `#2` in the second checkpoint is the walk's suffix riding inside the checkpoint
    name, which is why there is no `#2#2`."""
    app = absurd()
    suffix = uuid.uuid4().hex[:8]
    grant = depth_grant_name(f"r-{suffix}", depth=1, generation=0)
    task_name = f"two-grants-{suffix}"

    def workflow():
        a = yield from await_event(grant, Grant)
        b = yield from await_event(grant, Grant)
        return [a, b]

    @app.register_task(task_name, default_max_attempts=1)
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(workflow)

    spawned = app.spawn(task_name, {"run_id": "r1"})
    task_id = spawned["task_id"] if isinstance(spawned, dict) else spawned
    app.work_batch()
    app.emit_event(grant.stored(), GRANT_ANSWER)
    app.emit_event(grant.occurrence(2).stored(), GRANT_ANSWER)
    run_until_result(app, task_id)

    with psycopg.connect(DSN) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT checkpoint_name FROM absurd.c_default WHERE task_id=%s "
                "ORDER BY checkpoint_name",
                (str(task_id),),
            )
        ]

    assert names == [
        f"$awaitEvent:{grant.stored()}",
        f"$awaitEvent:{grant.occurrence(2).stored()}",
    ], names
    assert not any(name.endswith("#2#2") for name in names), (
        f"the engine's dup-name rule fired ON TOP of the walk's suffix: {names}"
    )
    # **The checkpoint names alone cannot tell the two mechanisms apart** — with the coordinate
    # removed, two asks on ONE name produce the very same two strings via the SDK's dup rule.
    # So the discriminator is behavioural: emit one answer to a FRESH run and it must park.
    assert _one_answer_leaves_it_parked(app), (
        "one answer settled both asks — the suffix above is the engine's, not the walk's"
    )


def _one_answer_leaves_it_parked(app: Any) -> bool:
    """A fresh task on a fresh grant name, answered ONCE — still waiting?

    Fresh BOTH ways, and the name is the one that matters: Absurd's events are queue-global, so
    reusing the caller's grant name would let the answers it already emitted resolve this run and
    the probe would report `completed` no matter what the coordinate does. (It did, on the first
    draft.) A new suffix is the only way to ask the question again."""
    suffix = uuid.uuid4().hex[:8]
    grant = depth_grant_name(f"r-probe-{suffix}", depth=1, generation=0)
    task_name = f"one-answer-{suffix}"

    def workflow():
        a = yield from await_event(grant, Grant)
        b = yield from await_event(grant, Grant)
        return [a, b]

    @app.register_task(task_name, default_max_attempts=1)
    def task(params, ctx):
        return DurableHandler(ctx, _Tool()).run(workflow)

    spawned = app.spawn(task_name, {"run_id": "r-probe"})
    task_id = spawned["task_id"] if isinstance(spawned, dict) else spawned
    app.work_batch()
    app.emit_event(grant.stored(), GRANT_ANSWER)  # ONE answer
    snapshot = run_until_result(app, task_id, max_batches=4)
    return snapshot is None or snapshot.state != "completed"


# --- `Key.scope`'s three journeys ---------------------------------------------------------------
#
# The property's docstring SHOWS three rows of walk output, and these are the rows. One test per
# journey rather than one asserting "the table is true", because a table-is-true test says only
# that a doc matches code — where a journey says what a caller may rely on, and its NAME is the
# claim. The rule these keep: the case set is named in prose, and every case has a name
# appearing in both the prose and the test.


def _asked_twice(name: Key) -> list[str]:
    """The same key awaited twice under one walk — what each row of the table runs."""
    position, seen = FramePosition(), []
    for _ in range(2):
        op = AwaitEvent(name=name, schema=dict)
        with placing(op, position):
            seen.append(placed_await_name(op).display())
    return seen


def test_asking_twice_at_a_settlement_namespace_is_two_questions():
    # A `govern:` approval settles ONE op-occurrence, so a second ask must not be answerable by
    # the first answer — the walk gives it a name of its own. Byte-preserving at the first ask,
    # so no recorded run is orphaned by the numbering existing.
    name = compose_key(t"{GOVERN}:{Segment('r1')}")
    assert name.scope is Scope.SETTLEMENT
    assert _asked_twice(name) == ["govern:r1", "govern:r1#2"]


def test_asking_twice_at_an_accrual_namespace_is_one_question_asked_again():
    # `budget-grant:`'s first-emit-wins is the FEATURE: a grant raises the run's ceiling and must
    # not be re-requested. Both asks wait on one name and one answer serves both.
    name = compose_key(t"{BUDGET_GRANT}:{Segment('r1')}")
    assert name.scope is Scope.ACCRUAL
    assert _asked_twice(name) == ["budget-grant:r1", "budget-grant:r1"]


def test_a_key_that_declared_no_reach_is_never_numbered():
    # `None` takes the same un-numbered path as ACCRUAL and is NOT the same fact — nothing
    # declared a reach. This is the row that makes `is not Scope.ACCRUAL` wrong as a test for
    # "settles": it is true here, and this key settles nothing.
    name = compose_key(t"{Tag('review')}:{Segment('r1')}")
    assert name.scope is None
    assert _asked_twice(name) == ["review:r1", "review:r1"]
    assert name.scope is not Scope.ACCRUAL
    assert name.scope is not Scope.SETTLEMENT


def test_the_numbering_rule_reads_the_scope_and_not_the_tag_text():
    # The guard is `name.scope is Scope.SETTLEMENT`, not a match on the leading bytes. A namespace
    # nobody registered would read as `None` under a text-keyed lookup, silently the un-numbered
    # answer, which is the failure a key's retained `scope` exists to prevent.
    settles = compose_key(t"{GOVERN}:{Segment('r1')}")
    looks_the_same = compose_key(t"{Tag('govern-ish')}:{Segment('r1')}")
    assert settles.scope is Scope.SETTLEMENT
    assert looks_the_same.scope is None
    assert _asked_twice(looks_the_same) == ["govern-ish:r1", "govern-ish:r1"]
