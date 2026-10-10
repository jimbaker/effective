"""Cross-backend durable conformance: ONE harness, both engines.

Parameterized over the embedded SQLite engine (0↔1, always) and Absurd/Postgres
(0↔N, skipped without a reachable PG). The same ``DurableHandler`` + ``TaskContext``
contract runs the same self-contained workflow with the same assertions on both,
so green-on-both is the engine-independence proof. The SQLite cases run in
``test-core`` (infra-free); the Postgres cases run under ``just pgt-test``.
"""

import time
from datetime import datetime, timedelta
from typing import Any, ClassVar, assert_never
from uuid import uuid4

import pytest
from _conformance import (
    RULING_GOAL,
    Approval,
    CodeActionDomain,
    CodingSuiteDomain,
    CountingDomain,
    Fault,
    FaultPosition,
    FlakyMeteringDomain,
    MeteringDomain,
    PeekRace,
    PerEventPeekRace,
    RaceOnEveryFirstPeek,
    RaceOnFirstPeek,
    RulingDeployment,
    RulingState,
    StubbornSuiteDomain,
    branch_absolute_await_wf,
    branched,
    coding_machine_wf,
    colliding_gather_wf,
    descend_recurse_wf,
    done_id,
    double_await_wf,
    duplicate_run_code_wf,
    ev_name,
    gated_wf,
    gather_await_wf,
    gather_run_code_wf,
    gather_wf,
    make_gather_sleep_wf,
    make_top_level_sleep_wf,
    metered_gather_trip_wf,
    metered_trip_wf,
    multi_park_wf,
    nested_gather_park_wf,
    note_row_id,
    parking_ruling_machine_wf,
    recurse_fold_park_wf,
    review_name,
    ruling_machine_wf,
    ruling_row_id,
    ruling_specs,
    run_code_wf,
    scoped_absolute_await_wf,
    scoped_await_wf,
    scoped_by,
    scoped_metered_trip_wf,
    scoped_nested_gather_await_wf,
    scoped_wf,
    sequential_collision_wf,
    sub_scope,
    two_gathers_wf,
    two_parking_gathers_wf,
    two_step_wf,
    wake_race_collision_wf,
)
from pydantic import BaseModel

from effective.api import (
    GatherBranch,
    await_event,
    await_until,
    call_tool,
    direct_tool_key,
    qualified_event_name,
    sleep_until,
)
from effective.budget import Grant, depth_grant_name
from effective.cost import Contract, Usage, serve
from effective.domain import CallTool, DomainOp
from effective.govern import GateState, Proceed, govern
from effective.graphview import PARKED, from_keys, to_mermaid
from effective.handlers.base import op_key, step_key
from effective.handlers.durable import DurableHandler, SeedingCtx, fork_event_name
from effective.keys import Key, Segment, compose_key, frame_path
from effective.layers import retry, retry_domain
from effective.machine.spec import Ctx
from effective.machine.trampoline import appending_states, canonical_violations
from effective.ops import (
    AppendLedgerRow,
    Arrived,
    Expired,
    LedgerRow,
    PlacedWriterCollision,
    Step,
    WaitOutcome,
    Writer,
)
from effective.parked import answer, pending_key
from effective.permission import APPROVE, Allow, Deny, Escalate, as_policy, cascade, human, rules
from effective.viewing import OutranTheTape, ViewingCtx

# The cross-engine `backend` fixture now lives in `conftest.py` — two modules need it
# (this one and `test_carrier_audit.py`), and pytest finds it by ordinary lookup.


# One model, one law, many arms: the overlap across arms IS the design, so the unit-role
# minimize-overlap rule does not apply here.
pytestmark = pytest.mark.conformance


def _run_two_step(
    backend, crash_at=None, layers=(), flaky=False, position=FaultPosition.BEFORE_OP
):
    """Spawn one two_step_wf run on `backend`, returning (snap, domain, fault, run_id)."""
    domain = CountingDomain(flaky=flaky)
    fault = Fault(crash_at, position=position)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"two-{run_id}"
    backend.register(name, two_step_wf, domain, fault, tuple(layers))
    snap = backend.run_until_result(backend.spawn(name, run_id))
    return snap, domain, fault, run_id


def test_clean_run_is_the_reference_outcome(backend):
    snap, domain, fault, run_id = _run_two_step(backend)
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == {"a": 10, "b": 20}
    assert domain.calls == ["a", "b"]
    assert fault.count == 4  # the 4 ctx ops
    assert backend.ledger_kinds(run_id) == ["k1", "k2"]


def test_crash_at_every_op_replays_to_identical_outcome(backend):
    """The headline gate on both engines: a crash at any op boundary resumes to the
    same committed outcome by replaying checkpoints — exactly-once, idempotent ledger."""
    for k in range(1, 5):  # crash before each of the 4 ops
        snap, domain, fault, run_id = _run_two_step(backend, crash_at=k)
        assert fault.armed is False, f"k={k}: fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"a": 10, "b": 20}, f"k={k}: {snap.result}"
        assert domain.calls == ["a", "b"], f"k={k}: {domain.calls}"  # exactly once across crash
        assert backend.ledger_kinds(run_id) == ["k1", "k2"], f"k={k}"


# The k-th step execution of `two_step_wf`, and what the domain sees when the process dies
# AFTER that step's thunk ran and BEFORE its checkpoint committed. Spelled out per k rather than
# counted, because which ops double is the whole finding: the two tool ks re-run a landed effect,
# and the two ledger ks re-append a row that `event_id` then dedupes.
_POST_THUNK_CALLS = {
    1: ["a", "a", "b"],  # tool "a" ran twice — the window, visible
    2: ["a", "b"],  # ledger e1 re-appends; "a" re-serves from its committed checkpoint
    3: ["a", "b", "b"],  # tool "b" ran twice
    4: ["a", "b"],  # ledger e2 re-appends
}


def test_post_thunk_crash_re_runs_a_landed_effect(backend):
    """The other half of the sweep above: a crash AFTER the thunk and before the commit.

    `test_crash_at_every_op_replays_to_identical_outcome` kills the process BEFORE each op, so its
    only death mode is "the tool never ran" and its `domain.calls == ["a", "b"]` is exactly-once by
    construction. This kills it AFTER the thunk and before the commit, where the effect has landed
    and the record has not: the tool then runs TWICE, on both engines.

    **Two claims, and they must be read together.** The outcome is identical to a clean run, which
    is a fact about `CountingDomain` and today's op alphabet being idempotent, NOT about the
    substrate preventing anything. `domain.calls` is where the substrate's actual behavior shows,
    and asserting it is what stops this test from passing vacuously: without that half it would be
    green even if the injector never fired.

    **What makes it a gate rather than a curiosity.** Every tool in the tree today is pure,
    in-memory or temp-dir, so re-execution is harmless and this is green. It goes RED the day
    someone adds a tool whose second execution is not free. A failure here is not a regression in
    the substrate; it is the arrival of the first op that needs a replay policy for an unknown
    outcome. Read it that way before you "fix" it.
    """
    for k, expected_calls in _POST_THUNK_CALLS.items():
        snap, domain, fault, run_id = _run_two_step(
            backend, crash_at=k, position=FaultPosition.AFTER_THUNK
        )
        assert fault.armed is False, f"k={k}: the fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"a": 10, "b": 20}, f"k={k}: {snap.result}"
        assert backend.ledger_kinds(run_id) == ["k1", "k2"], f"k={k}"
        assert domain.calls == expected_calls, f"k={k}: {domain.calls}"


def test_suspend_resume_via_event(backend):
    """await_event parks the task durably; emit_event resumes it; the pre-await step
    is not re-run, and the delivered payload binds by (run-scoped) name."""
    domain = CountingDomain()
    fault = Fault(None)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"gated-{run_id}"
    backend.register(name, gated_wf, domain, fault, ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state != "completed"  # parked on the event (engine-specific state name)
    assert domain.calls == ["a"]

    backend.emit_event(task_id, review_name(run_id).stored(), {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"a": 10, "decision": {"ok": True}}
    assert domain.calls == ["a"]  # pre-await step did NOT re-run on resume
    assert backend.ledger_kinds(run_id) == ["done"]


def test_parked_reader_finds_the_park_and_drops_it_on_resume(backend):
    """The generic task/park reader on both engines.

    Only what is genuinely engine-independent is asserted here: that the reader *finds* the
    parked task, reports the wake registration **verbatim** (so a caller can emit on the name
    the run registered rather than one it reconstructed), carries the spawn params, and stops
    listing the task once it resumes. `state` belongs here because the readers NORMALIZE it to
    `graphview.PARKED`: the engines' own words differ (`'waiting'` vs `'sleeping'`) and carrying
    either would make the field's meaning depend on which engine answered. `parked_since` stays
    out: only one engine records a park time, which is per-engine fact pinned in
    `test_parked_reader.py`, not a parity to force."""
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"gated-{run_id}"
    backend.register(name, gated_wf, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(name, run_id)
    backend.run_until_result(task_id)

    parked = backend.parked(task_id)
    assert len(parked) == 1, parked
    (park,) = parked
    # A `UUID` on both engines, compared RAW — which is the parity this assertion is for.
    # It read `== str(task_id)` while `SqliteApp.spawn` composed `f"{name}-{seq}"` and the record
    # normalized both engines down to text; converging them on `uuid7` removed the conversion,
    # and a raw `==` is now the stronger assertion because a regression to `str` on either side
    # fails it rather than being absorbed by the stringify.
    assert park.task_id == task_id
    assert park.task_name == name
    assert (
        park.wake_event == review_name(run_id).stored()
    )  # RAW: no `event:`/`$awaitEvent:` spelling
    assert park.params["run_id"] == run_id  # run_id is a params convention, not a column (L-2)
    assert park.state == PARKED  # normalized: 'waiting' on SQLite, 'sleeping' on Absurd

    backend.emit_event(task_id, review_name(run_id).stored(), {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert backend.parked(task_id) == []


def test_the_parked_run_projects_the_SAME_pending_node_on_both_engines(backend):
    """The pending-node bridge as a PARITY.

    The projection is a pure function of strings, so the engine-independence claim is testable:
    the same workflow parked at the same point must produce a byte-identical graph on both
    engines, pending node included. Spelling the pending node `event;{wake_event}` buys that
    property; `$awaitEvent:` is one engine's SDK-internal name, so a graph keyed on it would
    differ per engine at the one node a user clicks.

    The default (view) checkpoint projection is used, not `exclude=()`: an unfiltered read is
    *not* cross-engine comparable (3 vs 4 keys on the same resumed workflow), and the point here
    is that the pending node is comparable even though the engines' internals are not."""
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"gated-{run_id}"
    backend.register(name, gated_wf, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(name, run_id)
    backend.run_until_result(task_id)

    (park,) = backend.parked(task_id)
    graph = from_keys(run_id, backend.checkpoint_keys(task_id), pending=pending_key(park))
    assert [(n.key, n.kind, n.state) for n in graph.nodes] == [
        ("step;tool:a", "step", "committed"),
        # lint: terminal-hole — a `Key` splice
        (compose_key(t"event;{review_name(run_id):domain=address}").stored(), "await", "parked"),
    ]
    assert f'{{{{"event;review:{run_id}<br/>(parked)"}}}}' in to_mermaid(graph)

    backend.emit_event(task_id, review_name(run_id).stored(), {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    # Resumed: no park, so no pending node — and NO committed counterpart on either engine. The
    # await's checkpoint row exists on Absurd and not on SQLite, but it is engine-internal and the
    # view filters it out, so the two engines agree on what a user sees (per-engine detail:
    # `test_parked_reader.py` / `test_graphview.py`).
    assert backend.parked(task_id) == []
    resumed = from_keys(run_id, backend.checkpoint_keys(task_id))
    assert [n.key for n in resumed.nodes] == [
        "step;tool:a",
        # lint: terminal-hole — a `Key` splice
        compose_key(t"ledger;{done_id(run_id):domain=address}").stored(),
    ]
    assert "{{" not in to_mermaid(resumed)


def test_parked_reader_reports_one_branch_of_a_parked_gather(backend):
    """Two branches of a gather park; the engine arms only the LOWEST index's event (`_join`'s
    deterministic re-arm), so the reader reports 1 park where 2 exist, and answering that one
    yields a NEW park rather than a completion. That is substrate-correct, since wakes are
    serialized; a page built on this reader has to say so out loud."""
    run_id = f"mp-{uuid4().hex[:8]}"
    name = f"multi-park-{run_id}"
    backend.register(name, multi_park_wf, CountingDomain(), Fault(), ())
    task_id = backend.spawn(name, run_id)
    backend.run_until_result(task_id)

    (park,) = backend.parked(task_id)
    assert (
        park.wake_event == branched(0, 0, compose_key(t"ev0:{Segment(run_id)}")).stored()
    )  # branch 2 is parked too, and unlisted

    backend.emit_event(
        task_id, branched(0, 0, compose_key(t"ev0:{Segment(run_id)}")).stored(), {"n": 0}
    )
    backend.run_until_result(task_id)
    (park,) = backend.parked(task_id)
    assert (
        park.wake_event == branched(0, 2, compose_key(t"ev2:{Segment(run_id)}")).stored()
    )  # a new park, not a completion

    # Finish the run: the shared queue rule is that a spawned task reaches a TERMINAL state.
    backend.emit_event(
        task_id, branched(0, 2, compose_key(t"ev2:{Segment(run_id)}")).stored(), {"n": 2}
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert backend.parked(task_id) == []


def test_event_emission_is_first_write_wins(backend):
    """Events are immutable durable FACTS on both engines: a re-emission with a
    different payload is a no-op, so replay's re-read of the event binds the
    SAME payload a committed step already used. An upsert would let a re-emission
    rewrite replay history."""
    domain = CountingDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"gated-{run_id}"
    backend.register(name, gated_wf, domain, Fault(None), ())
    task_id = backend.spawn(name, run_id)
    backend.run_until_result(task_id)  # park on the review event
    backend.emit_event(task_id, review_name(run_id).stored(), {"n": 1})
    backend.emit_event(task_id, review_name(run_id).stored(), {"n": 999})  # must NOT overwrite
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"a": 10, "decision": {"n": 1}}  # first write won


def test_retry_op_layer_recovers_transient_failure(backend):
    """A retry op-layer recovers a transient step failure within one attempt; the
    thunk ran twice (fail, then success) but the durable record committed once."""
    snap, domain, _, run_id = _run_two_step(backend, layers=[retry(attempts=2)], flaky=True)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"a": 10, "b": 20}
    assert domain.calls == ["a", "a", "b"]  # 'a' re-run by retry, then 'b'
    assert backend.ledger_kinds(run_id) == ["k1", "k2"]


def test_durable_scoped_namespaces_two_same_shaped_bodies(backend):
    """`scoped(...)` runs durably on both engines, and two same-shaped bodies under different
    scopes do NOT share checkpoints. Without the handler-applied prefix the second body's
    `tool:a` would re-bind the first's committed checkpoint and return 10 twice while calling
    the tool once — the aliasing a handler-applied prefix exists to make impossible.

    `CountingDomain(incrementing=True)` is what makes the failure visible: a stale re-bind
    returns the FIRST value, a real execution returns the next one."""
    domain = CountingDomain(incrementing=True)
    run_id = f"sc-{uuid4().hex[:8]}"
    name = f"scoped-{run_id}"
    backend.register(name, scoped_wf, domain, Fault(), ())
    snap = backend.run_until_result(backend.spawn(name, run_id))
    assert snap is not None
    assert snap.state == "completed", snap
    # 11 then 12 — `incrementing` returns base + the call count, so two DISTINCT values prove
    # the second body executed. A collision would return 11 twice off one committed checkpoint.
    assert snap.result == {"first": 11, "second": 12}
    assert domain.calls == ["a", "a"]  # the same bare tool name, twice, under two scopes
    assert sorted(backend.ledger_kinds(run_id)) == ["k0", "k1"]


def test_durable_scoped_await_parks_on_the_scoped_event_name(backend):
    """An await inside a scope parks on the SCOPED name on both engines. The author wrote
    `review:{run_id}`; the engine must be waiting on `rec:0;review:{run_id}`, so emitting the
    bare name must NOT wake it and emitting the scoped one must.

    This is the assertion that proves the scope is structural rather than cosmetic — the park
    is the one place the prefix has to reach all the way into the engine's own event table."""
    domain = CountingDomain()
    run_id = f"sa-{uuid4().hex[:8]}"
    name = f"scopedawait-{run_id}"
    backend.register(name, scoped_await_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state != "completed"  # parked (engine-specific state name)

    # The BARE name is not what it parked on — the scope is real, not decoration.
    backend.emit_event(task_id, review_name(run_id).stored(), {"ok": False})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state != "completed", "the bare event name must not wake a scoped await"

    backend.emit_event(
        task_id,
        # lint: terminal-hole — a `Key` splice
        scoped_by(compose_key(t"rec:{0}"), review_name(run_id)).stored(),
        {"ok": True},
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"decision": {"ok": True}}
    assert backend.ledger_kinds(run_id) == ["done"]


def test_durable_park_carries_every_frame_through_a_nested_gather(backend):
    """A park inside gather → scope → gather must reach the engine with ALL its frames.

    Regression for the fourth frame-carrying bug. The park name is COMPOSED
    here, not spelled, so this asserts the engine agrees with the emitter contract rather than
    with a literal that could drift alongside a bug."""
    contract_for = lambda run_id: qualified_event_name(  # noqa: E731 — one expression, used twice
        GatherBranch(0, 0), compose_key(t"s:{0}"), GatherBranch(0, 0), name=f"ev:{run_id}"
    ).stored()
    # ARMED with a fault: a crash before each ctx op, so the frames are re-derived by
    # RE-EXECUTION on every resume rather than only on the first pass. k=0 is the no-crash control.
    for k in range(0, 4):
        domain = CountingDomain()
        fault = Fault(k) if k else Fault()
        run_id = f"sn-{uuid4().hex[:8]}"
        name = f"scopednested-{run_id}"
        backend.register(name, scoped_nested_gather_await_wf, domain, fault, ())
        task_id = backend.spawn(name, run_id)

        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state != "completed", f"k={k}: {snap}"

        # The name a frame-dropping park would have used — must NOT wake it.
        backend.emit_event(
            task_id, branched(0, 0, branched(0, 0, ev_name(run_id))).stored(), {"ok": False}
        )
        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state != "completed", f"k={k}: a frame-short name must not wake a scoped park"

        backend.emit_event(task_id, contract_for(run_id), {"ok": True})
        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"results": [[{"ok": True}]]}, f"k={k}"


def test_scoped_crash_at_every_op_replays_to_identical_outcome(backend):
    """Crash x SCOPE on both engines: the fault no other scoped case arms.

    Every other scoped test runs `Fault()`, i.e. no crash, so only this one asks: does re-entering
    a `scoped(...)` by RE-EXECUTION land on the same keys the crashed attempt committed? It has
    to, because that is the only mechanism: a scope is re-derived on replay, never restored (no
    captured continuation).

    `scoped_wf` is the right subject: two same-shaped bodies whose keys differ ONLY by the
    handler-applied frame, so a frame that failed to re-derive after a crash shows up as a stale
    re-bind (11 twice) rather than as an error."""
    for k in range(1, 5):  # crash before each of the 4 ctx ops (2 bodies x step + ledger)
        domain = CountingDomain(incrementing=True)
        fault = Fault(k)
        run_id = f"sc-{uuid4().hex[:8]}"
        name = f"scoped-crash-{run_id}"
        backend.register(name, scoped_wf, domain, fault, ())
        snap = backend.run_until_result(backend.spawn(name, run_id))
        assert snap is not None, f"k={k}"
        assert fault.armed is False, f"k={k}: fault never fired"
        assert snap.state == "completed", f"k={k}: {snap}"
        # The frames re-derived: two DISTINCT values, so neither body re-bound the other's
        # checkpoint across the crash.
        assert snap.result == {"first": 11, "second": 12}, f"k={k}: {snap.result}"
        # exactly once across the crash
        assert domain.calls == ["a", "a"], f"k={k}: {domain.calls}"
        assert sorted(backend.ledger_kinds(run_id)) == ["k0", "k1"], f"k={k}"


def test_scoped_park_crash_at_every_op_resumes_on_the_scoped_name(backend):
    """Park x scope x crash — the composition where F1 lived, now armed.

    A scoped park's name is re-derived by re-executing into the scope, so a crash before OR after
    the park must leave the engine waiting on the same frame-qualified event. If a frame failed to
    re-derive, the resumed attempt would park on a different name and the emit below would not
    wake it — which is the failure mode, not an exception."""
    for k in range(1, 4):
        domain = CountingDomain()
        fault = Fault(k)
        run_id = f"sp-{uuid4().hex[:8]}"
        name = f"scoped-park-crash-{run_id}"
        backend.register(name, scoped_await_wf, domain, fault, ())
        task_id = backend.spawn(name, run_id)

        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state != "completed", f"k={k}: {snap}"

        backend.emit_event(
            task_id,
            # lint: terminal-hole — a `Key` splice
            scoped_by(compose_key(t"rec:{0}"), review_name(run_id)).stored(),
            {"ok": True},
        )
        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"decision": {"ok": True}}, f"k={k}: {snap.result}"
        assert backend.ledger_kinds(run_id) == ["done"], f"k={k}"


def _run_gather_wf(backend, fault=None, delay=0.0):
    domain = CountingDomain(delay=delay)
    fault = fault if fault is not None else Fault()
    run_id = f"g-{uuid4().hex[:8]}"
    name = f"gather-{run_id}"
    backend.register(name, gather_wf, domain, fault, ())
    snap = backend.run_until_result(backend.spawn(name, run_id))
    return snap, domain, fault, run_id


def test_durable_gather_clean_run(backend):
    """gather runs durably on both engines, each branch's ops checkpointed under gather:{i}:.
    Results join in branch index order (deterministic); the ledger holds both branches'
    events but their relative order is a race — sequential consistency, a partial order —
    so assert it as a set."""
    snap, domain, _, run_id = _run_gather_wf(backend)
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == {"results": [10, 20]}  # branch index order — deterministic
    assert sorted(domain.calls) == ["a", "b"]  # each tool once (partial order)
    assert sorted(backend.ledger_kinds(run_id)) == ["ka", "kb"]  # partial order on the ledger


def test_durable_gather_keys_distinct_across_two_same_shaped_gathers(backend):
    """Two same-shaped gathers in one workflow must NOT share checkpoint keys: the second
    must execute, not silently read the first's checkpoints and return a stale result. The
    `gather:{g},{i};` discriminator ({g} = the gather's position in the op stream) is what
    prevents the collision."""
    domain = CountingDomain(incrementing=True)
    run_id = f"tg-{uuid4().hex[:8]}"
    name = f"twogather-{run_id}"
    backend.register(name, two_gathers_wf, domain, Fault(), ())
    snap = backend.run_until_result(backend.spawn(name, run_id))
    assert snap is not None
    assert snap.state == "completed", snap
    # 4 distinct calls (a,a,b,b); a key collision would skip the 2nd gather (only 2 calls):
    assert sorted(domain.calls) == ["a", "a", "b", "b"], domain.calls
    # the gathers are sequential at the workflow level, so g1's calls precede g2's:
    assert snap.result["g1"] == [11, 21]  # 1st call: a->10+1, b->20+1
    assert snap.result["g2"] == [12, 22]  # 2nd call: a->10+2, b->20+2
    assert snap.result["g1"] != snap.result["g2"]  # not a stale re-read


def test_durable_gather_branch_await_parks_and_resumes(backend):
    """V1: a gather branch's await parks the WHOLE task on the branch's prefixed
    event; the sibling's committed step holds across the park (round-barrier);
    the emission resumes by name and the payload joins in branch order."""
    domain = CountingDomain()
    run_id = f"ga-{uuid4().hex[:8]}"
    name = f"gather-await-{run_id}"
    backend.register(name, gather_await_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # parked, not dead
    assert domain.calls == ["a"]  # the sibling ran to completion in the round

    backend.emit_event(
        task_id, branched(0, 1, ev_name(run_id)).stored(), {"ok": True}
    )  # the QUALIFIED name
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [10, {"ok": True}]}
    assert domain.calls == ["a"]  # exactly-once: the sibling re-bound, never re-ran


def test_durable_gather_two_parked_branches_serialized_wakes(backend):
    """V1 serialized wakes: with branches 0 and 2 parked, the task waits on the
    LOWEST index. Emitting branch 2's event first does NOT wake it; branch 0's
    does — and that wake's replay finds branch 2's payload already delivered and
    completes without another park (the re-arm/peek resolves it)."""
    domain = CountingDomain()
    run_id = f"mp-{uuid4().hex[:8]}"
    name = f"multi-park-{run_id}"
    backend.register(name, multi_park_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap
    assert domain.calls == ["a"]  # the tool branch committed in the first round

    backend.emit_event(
        task_id, branched(0, 2, compose_key(t"ev2:{Segment(run_id)}")).stored(), {"n": 2}
    )  # out of order: no wake
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # still parked on branch 0

    backend.emit_event(
        task_id, branched(0, 0, compose_key(t"ev0:{Segment(run_id)}")).stored(), {"n": 0}
    )  # the armed event
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [{"n": 0}, 10, {"n": 2}]}
    assert domain.calls == ["a"]  # exactly-once across both wakes


def test_durable_gather_park_keys_distinct_across_two_same_shaped_gathers(backend):
    """Injectivity of {g} on EVENT names: two same-shaped gathers each park on the
    same BARE event name; the qualified names differ (gather:0,1; vs gather:1,1;),
    so each needs its own emission and each payload lands in its own gather."""
    domain = CountingDomain(incrementing=True)
    run_id = f"tp-{uuid4().hex[:8]}"
    name = f"two-park-{run_id}"
    backend.register(name, two_parking_gathers_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap

    backend.emit_event(task_id, branched(0, 1, ev_name(run_id)).stored(), {"g": 1})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # now parked on gather 1's await

    backend.emit_event(task_id, branched(1, 1, ev_name(run_id)).stored(), {"g": 2})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result["g1"] == [11, {"g": 1}]
    assert snap.result["g2"] == [12, {"g": 2}]  # distinct payloads, distinct tool calls


def test_durable_nested_gather_park_composes_the_prefix(backend):
    """Injectivity under composition: a park inside a nested gather waits on the
    fully-composed path (gather:0,0;gather:0,0;ev:...)."""
    domain = CountingDomain()
    run_id = f"np-{uuid4().hex[:8]}"
    name = f"nested-park-{run_id}"
    backend.register(name, nested_gather_park_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap
    assert domain.calls == ["b"]  # the outer sibling committed

    backend.emit_event(
        task_id, branched(0, 0, branched(0, 0, ev_name(run_id))).stored(), {"deep": True}
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [{"deep": True}, 20]}
    assert domain.calls == ["b"]


def test_durable_descend_leaf_parks_on_the_gather_qualified_grant(backend):
    """The composition contract: a descend leaf
    inside a recurse fan-out parks on its GATHER-QUALIFIED, run-scoped grant.
    The sharp edge pinned here: the bare name a descend call-site reader would
    guess does NOT wake it: the emitter composes gather:{g},{i}; (the ctx
    prefix, invisible at the call site) with the run-scoped grant name."""
    domain = CountingDomain()
    run_id = f"dg-{uuid4().hex[:8]}"
    name = f"descend-{run_id}"
    backend.register(name, descend_recurse_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap
    assert sorted(domain.calls) == ["a", "a"]  # one committed judge per leaf in the round

    # The unqualified name a call-site reader would guess: no wake.
    backend.emit_event(
        task_id,
        f"rec:1;{depth_grant_name(run_id, depth=1, generation=0).stored()}",
        {"add_depth": 0},
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap

    backend.emit_event(
        task_id,
        f"gather:0,1;rec:1;{depth_grant_name(run_id, depth=1, generation=0).stored()}",
        {"add_depth": 0},
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"total": 21}  # easy 10+0, hard final-judge 10+1
    assert sorted(domain.calls) == ["a", "a", "a"]  # exactly-once: only the final judge ran live


def test_durable_descend_multi_grant_round_trip(backend):
    """The multi-grant flavor, durably: the same drill parks TWICE. It grants 2
    more levels at d1 (a fresh park at d3, a DISTINCT :d3 event name), then 0 at
    d3 (the final nudge). Exactly-once at every stage: committed judges re-bind
    across both wakes."""
    domain = CountingDomain()
    run_id = f"dm-{uuid4().hex[:8]}"
    name = f"descend-multi-{run_id}"
    backend.register(name, descend_recurse_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap
    assert sorted(domain.calls) == ["a", "a"]  # both leaves' d0 judges committed

    backend.emit_event(
        task_id,
        f"gather:0,1;rec:1;{depth_grant_name(run_id, depth=1, generation=0).stored()}",
        {"add_depth": 2},
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # re-parked on the d3 grant
    assert sorted(domain.calls) == ["a"] * 4  # d1, d2 ran live; d0 pair re-bound

    backend.emit_event(
        task_id,
        f"gather:0,1;rec:1;{depth_grant_name(run_id, depth=3, generation=0).stored()}",
        {"add_depth": 0},
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"total": 23}  # easy 10+0, hard final-judge at d3 10+3
    assert sorted(domain.calls) == ["a"] * 5  # only the final judge ran live on the last wake


def _wait_past(deadline: datetime, *, margin: float = 0.05, timeout: float = 30.0) -> None:
    """Block until `time.time()` is past `deadline` — the ENGINE's clock, not elapsed sleep.

    A durable deadline is compared against `time.time()` (`SqliteApp._claim`; Absurd's own SQL
    does the equivalent), so that is the only clock whose reading settles "has it expired?".
    `time.sleep(n)` measures MONOTONIC time and the two diverge whenever the wall clock steps —
    which it does on this host, backward, by ~0.95s (see the caller). Polling the guard's clock
    is total under a step in either direction: forward and we return at once, backward and we
    simply wait longer.

    Bounded on `monotonic` rather than on the clock being waited for, so a clock that never
    reaches the deadline raises instead of hanging the suite."""
    started = time.monotonic()
    while time.time() < deadline.timestamp() + margin:
        if time.monotonic() - started > timeout:
            raise AssertionError(
                f"wall clock never reached {deadline.isoformat()} in {timeout}s of monotonic "
                f"time (now={time.time()}); the host clock is stepping, not merely slow"
            )
        time.sleep(0.02)


def test_durable_gather_sleep_parks_then_wakes(backend):
    """V1 for SleepUntil: a branch's durable sleep parks the task 'sleeping';
    the deadline passing completes it (and exercises the durable-sleep path on
    Postgres).

    **Both halves are timed against the ENGINE's clock, not against elapsed sleep.** A durable
    deadline is a wall-clock instant (`available_at`, compared to `time.time()` in `_claim`),
    and it has to be, because a durable timer must survive process death and a monotonic clock
    restarts with the process. Wall clocks **step**: on a WSL2 host `time.time()` can step
    **backward** by most of a second during a `time.sleep(0.1)`. So `sleep(deadline + margin)` does
    not imply the deadline passed: a backward step un-expires it, the task stays `sleeping`, and
    the test fails while the engine is behaving exactly as specified. A test that slept the
    margin failed one run in twelve on that host, and it is not a race: no ordering makes it go
    away.

    The rule this test follows: **wait on the same clock the guard reads.** `_wait_past`
    polls `time.time()` until it is genuinely past the deadline (bounded on `monotonic`, so a
    pathological clock fails loudly instead of hanging). The park half is the mirror image: the
    reading only means something if it happened strictly *before* the deadline, so if the clock
    reached it during the drain the attempt proves nothing and is retried with a wider budget.
    Neither loosening can hide a defect: a sleep that never parks still fails, and a sleep that
    never wakes still fails."""
    for budget in (1.0, 4.0, 16.0):
        domain = CountingDomain()
        run_id = f"gs-{uuid4().hex[:8]}"
        name = f"gather-sleep-{run_id}"
        wake_at = datetime.now().astimezone() + timedelta(seconds=budget)
        backend.register(name, make_gather_sleep_wf(wake_at), domain, Fault(), ())
        task_id = backend.spawn(name, run_id)

        snap = backend.run_until_result(task_id)
        assert snap is not None
        if time.time() < wake_at.timestamp():
            break  # read strictly before the deadline, so 'is it parked?' is a real question
    else:
        pytest.fail("the clock passed a 16s deadline mid-drain three times — not an artifact")

    assert snap.state not in ("completed", "failed"), snap  # parked sleeping
    assert domain.calls == ["a"]  # the sibling committed in the round

    _wait_past(wake_at)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": ["woke", 10]}
    assert domain.calls == ["a"]


def test_durable_gather_park_crash_at_every_op(backend):
    """Park x crash: a one-shot crash injected at the k-th ctx-op touch (steps,
    peeks, the re-arm await; k is a benign race under concurrent branches, and a
    k beyond the touch count injects nothing) still converges — park, emit,
    resume, identical outcome, exactly-once side effects, for every k."""
    for k in range(1, 6):
        domain = CountingDomain()
        fault = Fault(k)
        run_id = f"gc-{uuid4().hex[:8]}"
        name = f"gather-crash-{run_id}"
        backend.register(name, gather_await_wf, domain, fault, ())
        task_id = backend.spawn(name, run_id)
        snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        if snap.state not in ("completed", "failed"):  # parked — deliver and drain
            backend.emit_event(task_id, branched(0, 1, ev_name(run_id)).stored(), {"ok": True})
            snap = backend.run_until_result(task_id)
        assert snap is not None, f"k={k}"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"results": [10, {"ok": True}]}, f"k={k}"
        assert domain.calls == ["a"], f"k={k}: {domain.calls}"  # exactly once, every k


def _run_raced_gather(backend, *, max_attempts=None, fault=None):
    """One gather_await_wf run under the deterministic wake race: the branch's
    first peek emits its own awaited event mid-round, so the post-barrier re-arm
    finds every wake condition satisfied and _join must repark (no attempt burn).
    Returns (snap, domain, fault, task_id)."""
    domain = CountingDomain()
    fault = fault or Fault()
    run_id = f"wr-{uuid4().hex[:8]}"
    name = f"wake-race-{run_id}"
    box: list[str] = []
    race = PeekRace(lambda event: backend.emit_event(box[0], event, {"ok": True}))
    backend.register(
        name, gather_await_wf, domain, fault, (), wrap=lambda ctx: RaceOnFirstPeek(ctx, race)
    )
    task_id = backend.spawn(name, run_id, max_attempts=max_attempts)
    box.append(task_id)
    snap = _drain_terminal(backend, task_id)
    assert race.fired, "the race instrument never fired"
    return snap, domain, fault, task_id


def _drain_terminal(backend, task_id, deadline: float = 8.0):
    """Drain a task to a terminal state on a wall-clock deadline, not a fixed
    retry count: a reparked task sleeps until now + epsilon (and a raced sleep
    branch until its own wake time), so under suite load a single fixed retry
    is a flake — the deadline absorbs the jitter without changing what the
    test asserts."""
    snap = backend.run_until_result(task_id)
    start = time.monotonic()
    while (
        snap is not None
        and snap.state not in ("completed", "failed")
        and time.monotonic() - start < deadline
    ):
        time.sleep(0.3)
        snap = backend.run_until_result(task_id)
    return snap


def test_durable_gather_wake_race_reparks_and_completes_without_burning_an_attempt(backend):
    """The no-burn repark on a real engine: the raced task converges to the
    identical outcome, exactly-once, with the SAME attempt count as an unraced
    park/resume run — the race re-queued via the engine's park path, not its
    retry path (the baseline is read, not assumed)."""
    # Baseline: the ordinary park -> emit -> resume run (no race).
    domain = CountingDomain()
    run_id = f"bl-{uuid4().hex[:8]}"
    name = f"baseline-{run_id}"
    backend.register(name, gather_await_wf, domain, Fault(), ())
    baseline_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(baseline_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed")
    backend.emit_event(baseline_id, branched(0, 1, ev_name(run_id)).stored(), {"ok": True})
    snap = backend.run_until_result(baseline_id)
    assert snap is not None
    assert snap.state == "completed"

    raced_snap, raced_domain, _, raced_id = _run_raced_gather(backend)
    assert raced_snap is not None
    assert raced_snap.state == "completed", raced_snap
    assert raced_snap.result == {"results": [10, {"ok": True}]}
    assert raced_domain.calls == ["a"]  # exactly-once across the repark round-trip
    assert backend.task_attempts(raced_id) == backend.task_attempts(baseline_id)


def test_durable_gather_wake_race_max_attempts_1_completes(backend):
    """A HEALTHY max_attempts=1 task hitting the wake race must complete. Repark
    burns no attempt; a retry fallback would spend the only one and fail the task
    permanently."""
    snap, domain, _, _ = _run_raced_gather(backend, max_attempts=1)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [10, {"ok": True}]}
    assert domain.calls == ["a"]


def test_durable_wake_race_step_name_collision_is_impossible(backend):
    """An author step legally named ``wake-race`` inside the raced branch must
    NOT alias the repark checkpoint. A repark minted as ``gather:{g},{i};wake-race``
    would collide with it, and the second run's begin_step would return the
    repark's wake TIME as the step's value: silent wrong data, the tool never
    executed, the engines diverging."""
    domain = CountingDomain()
    run_id = f"wc-{uuid4().hex[:8]}"
    name = f"wake-collide-{run_id}"
    box: list[str] = []
    race = PeekRace(lambda event: backend.emit_event(box[0], event, {"ok": True}))
    backend.register(
        name,
        wake_race_collision_wf,
        domain,
        Fault(),
        (),
        wrap=lambda ctx: RaceOnFirstPeek(ctx, race),
    )
    task_id = backend.spawn(name, run_id)
    box.append(task_id)
    snap = _drain_terminal(backend, task_id)
    assert race.fired
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [[{"ok": True}, 10]]}  # the TOOL's value, not a timestamp
    assert domain.calls == ["a"]  # exactly-once includes "once", not zero


def test_durable_double_wake_race_still_burns_no_attempt(backend):
    """A branch with two sequential awaits can race the same gather TWICE, so
    'at most once' must hold per branch, not only per await. With the wake
    condition in the repark name each race mints a fresh checkpoint, so no-burn
    holds unconditionally: a max_attempts=1 task completes and the attempt count
    matches an unraced baseline."""
    # Baseline: unraced park/resume of the same double-await shape.
    domain = CountingDomain()
    run_id = f"db-{uuid4().hex[:8]}"
    name = f"double-baseline-{run_id}"
    backend.register(name, double_await_wf, domain, Fault(), ())
    baseline_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(baseline_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed")
    backend.emit_event(
        baseline_id, branched(0, 0, compose_key(t"ev1:{Segment(run_id)}")).stored(), {"n": 1}
    )
    snap = backend.run_until_result(baseline_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed")  # now parked on the second await
    backend.emit_event(
        baseline_id, branched(0, 0, compose_key(t"ev2:{Segment(run_id)}")).stored(), {"n": 2}
    )
    snap = backend.run_until_result(baseline_id)
    assert snap is not None
    assert snap.state == "completed"

    # Raced: every await's first peek emits mid-round; max_attempts=1 must survive both races.
    domain = CountingDomain()
    run_id = f"dr-{uuid4().hex[:8]}"
    name = f"double-race-{run_id}"
    box: list[str] = []
    race = PerEventPeekRace(lambda event: backend.emit_event(box[0], event, {"ok": True}))
    backend.register(
        name,
        double_await_wf,
        domain,
        Fault(),
        (),
        wrap=lambda ctx: RaceOnEveryFirstPeek(ctx, race),
    )
    task_id = backend.spawn(name, run_id, max_attempts=1)
    box.append(task_id)
    snap = _drain_terminal(backend, task_id)  # two repark wakes (now + epsilon each)
    assert len(race.raced) == 2, race.raced
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [[{"ok": True}, {"ok": True}], 10]}
    assert domain.calls == ["a"]
    assert backend.task_attempts(task_id) == backend.task_attempts(baseline_id)


def test_durable_sleep_wake_race_reparks_and_completes(backend):
    """The sleep flavor of the wake race: a branch parks on an UNDUE deadline,
    the slow sibling delays the barrier past it, and the re-arm's sleep_until
    finds it already due — nothing to park on. Repark's name derives from
    bp.until here (the isoformat arm), so this pins that derivation on both
    engines. (If suite load delays the round past wake_at, the branch's own
    sleep is already due and no race occurs — the same assertions hold on that
    path, so the test is load-tolerant in both directions.)"""
    domain = CountingDomain(delay=1.0)
    run_id = f"sr-{uuid4().hex[:8]}"
    name = f"sleep-race-{run_id}"
    wake_at = datetime.now().astimezone() + timedelta(seconds=0.3)
    backend.register(name, make_gather_sleep_wf(wake_at), domain, Fault(), ())
    task_id = backend.spawn(name, run_id, max_attempts=1)
    snap = _drain_terminal(backend, task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": ["woke", 10]}
    assert domain.calls == ["a"]


def test_a_top_level_durable_sleep_wakes_and_completes(backend):
    """A durable sleep at the TASK ROOT, on both engines: the `sleep` pair, not `sleep ∘ gather`.

    A sleep inside a gather branch routes to `_branch_sleep`, a pure clock compare that calls
    neither ctx, so the top-level path needs a cross-engine cell of its own.

    **Scope.** This does NOT pin `SdkCtx.sleep_until`'s two-argument forward to the SDK:
    `AbsurdBackend` wraps `ConcurrentAbsurdCtx` rather than `SdkCtx` (which `_adapt_ctx`
    installs on a raw SDK ctx, the deployed worker's path), so breaking that forward leaves this
    test GREEN. `tests/test_durable_sleep_and_drain_depth.py` pins it. What this adds is
    top-level durable sleep interpreted identically on SQLite and Absurd.

    Both steps must commit, one either side of the sleep, so the sleep is proved to be
    traversed rather than skipped."""
    domain = CountingDomain()
    run_id = f"tls-{uuid4().hex[:8]}"
    name = f"top-sleep-{run_id}"
    wake_at = datetime.now().astimezone() + timedelta(seconds=0.3)
    backend.register(name, make_top_level_sleep_wf(wake_at), domain, Fault(), ())
    task_id = backend.spawn(name, run_id, max_attempts=1)
    snap = _drain_terminal(backend, task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"before": 10, "after": 20}
    assert domain.calls == ["a", "b"], "the sleep must be crossed, not skipped"


def test_an_absolute_await_reroutes_around_a_scope_on_both_engines(backend):
    """`scoped ∘ (ABSOLUTE await)` — the frame is REROUTED, not applied.

    A scope completes RELATIVE names; an absolute one is already whole, and its emitter is a
    different task that has never seen
    this scope. So the run must park on the BARE name; the test emits exactly that, and a
    regression would park forever on `s:0;sub:{run_id}` and never complete.

    This is the `_root_ctx` rule `budget-grant` follows, generalized from a wired call-site to
    a declared property of the name."""
    domain = CountingDomain()
    run_id = f"sar-{uuid4().hex[:8]}"
    name = f"scoped-subject-{run_id}"
    backend.register(name, scoped_absolute_await_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id, max_attempts=1)
    backend.run_until_result(task_id)

    backend.emit_event(task_id, sub_scope(run_id).stored(), {"ok": True})
    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"decision": {"ok": True}}


def test_an_absolute_await_inside_a_gather_branch_is_refused_on_both_engines(backend):
    """`gather ∘ (ABSOLUTE await)` — LOUD, and the message names the loop as the fix.

    A branch coordinate is a concurrency slot rather than a naming choice: a branch await peeks
    and parks as a value the barrier re-arms with the coordinate re-added, so there is nothing
    to reroute to. One message, both interpreters, by construction
    (`ops.refuse_absolute_await_in_branch`) — two interpreters that decide this separately drift
    (F3)."""
    domain = CountingDomain()
    run_id = f"bsa-{uuid4().hex[:8]}"
    name = f"branch-subject-{run_id}"
    backend.register(name, branch_absolute_await_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id, max_attempts=1)

    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "failed", snap
    # Absurd stores `{"name", "message"}`, SQLite a bare string — normalize rather than pick
    # one shape and silently pass on the other engine for the wrong reason.
    failure = str(snap.failure)
    assert "is an ABSOLUTE event name" in failure, failure
    # The refusal must name the FIX — an author told only "refused" reaches for a smaller
    # edit, and no smaller edit can work here.
    assert "loop" in failure, failure
    assert "marginal_sweep" in failure, failure


def test_durable_wake_race_crash_at_repark_still_converges(backend):
    """Crash-at-repark is a real death mode: the fault lands exactly on the
    repark touch, the engine's ordinary retry re-queues (this one legitimately
    burns an attempt; the default budget absorbs it), and the wake replay
    resolves every branch at its peek — identical outcome, exactly-once."""
    fault = Fault(on_name="repark:")
    snap, domain, fault, _ = _run_raced_gather(backend, fault=fault)
    assert fault.armed is False, "the crash never landed on the repark touch"
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [10, {"ok": True}]}
    assert domain.calls == ["a"]


def test_durable_gather_crash_resume(backend):
    """Per-branch crash-resume: crash branch 1's ledger commit once. The retry keeps branch
    1's committed tool-step (and all of branch 0) and re-runs only the rest — exactly-once,
    on both paths, which this harness wraps concurrently whichever engine is under it. A
    name-targeted fault is well-defined regardless of branch interleaving."""
    snap, domain, fault, run_id = _run_gather_wf(backend, Fault(on_name="gather:0,1;ledger"))
    assert fault.armed is False, "fault never fired"
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"results": [10, 20]}
    assert sorted(domain.calls) == ["a", "b"]  # exactly once across the crash
    assert sorted(backend.ledger_kinds(run_id)) == ["ka", "kb"]


def test_durable_gather_runs_tools_concurrently(backend):
    """Parallel tools, serialized writes on BOTH concurrent paths (SQLite's write-lock,
    Postgres's begin/complete lock): the two tools are IN FLIGHT AT THE SAME TIME — the lock
    is released during the tool work, GIL-style.

    An interval intersection, not a wall-clock bound. The bound asserted "finished fast enough
    to have been concurrent", which is a claim about the machine rather than the code: it read
    1.975s against a 0.55s ceiling at load average 15, and 0.49s alone, on the same commit."""
    snap, domain, _, _ = _run_gather_wf(backend, delay=0.3)
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == {"results": [10, 20]}
    spans = {name: (enter, done) for name, enter, done in domain.spans}
    assert set(spans) == {"a", "b"}, f"expected both tools to run: {domain.spans}"
    (a_in, a_out), (b_in, b_out) = spans["a"], spans["b"]
    overlap = min(a_out, b_out) - max(a_in, b_in)
    assert overlap > 0, f"tools did not overlap (serialized?): {domain.spans}"


def test_ledger_is_append_only(backend):
    """The two-bookkeepers invariant on both engines: idempotent by event_id, and the
    DB blocks UPDATE (RAISE(ABORT) on SQLite, the plpgsql trigger on Postgres)."""
    run_id = f"r-{uuid4().hex[:8]}"
    event_id = f"{run_id}:x"
    backend.ledger_append(run_id, event_id, "k")
    backend.ledger_append(run_id, event_id, "k")  # idempotent: ON CONFLICT DO NOTHING
    assert backend.ledger_kinds(run_id) == ["k"]
    with pytest.raises(backend.mutation_error):
        backend.raw_update_kind(event_id)


# ── run_code: the segment machine on the durable path ──────────────


def _run_code_run(backend, fault=None, layers=(), factory=run_code_wf):
    """One run_code workflow on `backend` -> (snap, domain, fault, run_id, task_id)."""
    domain = CodeActionDomain()
    fault = fault or Fault(None)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"code-{run_id}"
    backend.register(name, factory, domain, fault, tuple(layers))
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    return snap, domain, fault, run_id, task_id


def test_durable_run_code_clean_run(backend):
    """The reference outcome — and the within-run re-execution proof: segment 2
    re-runs the code with llm_query re-bound from the threaded fn_log, so the live
    function fires ONCE even though the code executed twice."""
    snap, domain, fault, run_id, _ = _run_code_run(backend)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"summary": f"summary({run_id})", "ack": {"id": "msg-1"}}
    assert domain.fn_calls == [run_id]  # live once; seg:1 re-bound, never called
    assert len(domain.action_calls) == 1
    assert fault.count == 4  # seg:0, action, seg:1, ledger
    assert backend.ledger_kinds(run_id) == ["code-done"]


def test_durable_run_code_crash_at_every_op(backend):
    """The run_code gate: a crash at any op boundary replays to the identical
    outcome; the action fires exactly once and the function's live call count
    stays one. Committed segments re-bind as checkpoint lookups (the engine
    never re-executes them), uncommitted ones re-execute with recorded results."""
    for k in range(1, 5):
        snap, domain, fault, run_id, _ = _run_code_run(backend, fault=Fault(k))
        assert fault.armed is False, f"k={k}: fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {
            "summary": f"summary({run_id})",
            "ack": {"id": "msg-1"},
        }, f"k={k}"
        assert domain.fn_calls == [run_id], f"k={k}: {domain.fn_calls}"
        assert len(domain.action_calls) == 1, f"k={k}: {domain.action_calls}"
        assert backend.ledger_kinds(run_id) == ["code-done"], f"k={k}"


def _gate_send_email(op):
    """Escalate the send_email action to the next tier; allow everything else."""
    if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "send_email":
        return Escalate("world-mutating: needs a ruling")
    return Allow()


def test_durable_run_code_action_parks_for_human_approval(backend):
    """Mid-code HITL: the cascade's human tier injects an AwaitEvent at the action
    op — the task parks durably *between two code segments*, with the function
    already run and the action NOT yet fired. Approval resumes it: the action runs
    exactly once and the code completes with the ack.

    The approval event is **run-scoped** (`approve;{run_id}:{op_key}`): Absurd
    events are global by NAME, not task-scoped, so the human tier's op_key-derived
    default collides across runs of a same-shaped workflow — a persisted approval
    from an earlier run resumes the next run's park instantly (caught live: this
    test passed standalone and failed on the suite's second run). Production
    cascades must scope the event name the same way (the gated_wf rule)."""
    domain = CodeActionDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    approve_event = f"approve:{run_id};step;code:action,0,c;tool:send_email"
    scoped = human(
        Approval,
        # Composed, not f-stringed: a `Key` has no `__str__`, so an f-string here would
        # bake `Key(_value=...)` into the park name; and the composer is where a
        # run-scoped approval name belongs regardless.
        event_name=lambda op: compose_key(
            # lint: terminal-hole — `op_key` returns a `Key`, which the composer splices by
            # INDUCTION; wrapping it in a `Segment` would be a regression, and since `Segment`
            # takes a `str` it now refuses one outright.
            t"{APPROVE}:{Segment(run_id)};{op_key(op):domain=identity}"
        ),
    )
    name = f"code-{run_id}"
    gate = cascade([rules(_gate_send_email), scoped])
    backend.register(name, run_code_wf, domain, Fault(None), (gate,))
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state != "completed", snap  # parked on the approval event
    assert domain.fn_calls == [run_id]  # seg:0 ran (the function fired) ...
    assert domain.action_calls == []  # ... but the world is untouched

    backend.emit_event(task_id, approve_event, {"decision": "approve"})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"summary": f"summary({run_id})", "ack": {"id": "msg-1"}}
    assert domain.fn_calls == [run_id]  # replay re-bound, never re-called
    assert len(domain.action_calls) == 1
    assert backend.ledger_kinds(run_id) == ["code-done"]


def test_durable_run_code_denied_action_routes_in_sandbox(backend):
    """Denial-as-routing-signal inside the REPL: a rules tier denies the action;
    the combinator threads the refusal into the next segment as PermissionError;
    the code catches it and completes on its fallback path. The world is untouched."""

    def deny_send(op):
        if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "send_email":
            return Deny("outbound email blocked")
        return Allow()

    snap, domain, _, run_id, _ = _run_code_run(backend, layers=[cascade([rules(deny_send)])])
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {
        "summary": f"summary({run_id})",
        "ack": "denied: outbound email blocked",
    }
    assert domain.action_calls == []  # the op never forwarded
    assert backend.ledger_kinds(run_id) == ["code-done"]


def test_durable_run_code_in_gather_keys_disjoint(backend):
    """Injectivity under composition: the SAME run_code name in two gather branches
    — only the gather:{g},{i}; prefix separates their keys. Distinct summaries per
    branch and two distinct acks prove no branch read the other's checkpoints."""
    snap, domain, _, run_id, _ = _run_code_run(backend, factory=gather_run_code_wf)
    assert snap is not None
    assert snap.state == "completed", snap
    results = snap.result["results"]
    assert [r["summary"] for r in results] == [
        f"summary({run_id}-b0)",
        f"summary({run_id}-b1)",
    ]
    # acks are distinct; WHICH branch got msg-1 is a benign race (assert as a set)
    assert {r["ack"]["id"] for r in results} == {"msg-1", "msg-2"}
    assert sorted(domain.fn_calls) == sorted([f"{run_id}-b0", f"{run_id}-b1"])
    assert len(domain.action_calls) == 2


def test_durable_run_code_in_gather_crash_resume(backend):
    """Composition x crash: a name-targeted fault at one branch's action op. The
    retry keeps every committed segment/action and re-runs only the rest — each
    branch's action still fires exactly once, with correct per-branch results."""
    snap, domain, fault, run_id, _ = _run_code_run(
        backend, fault=Fault(on_name="code:action,0,c"), factory=gather_run_code_wf
    )
    assert fault.armed is False, "fault never fired"
    assert snap is not None
    assert snap.state == "completed", snap
    results = snap.result["results"]
    assert [r["summary"] for r in results] == [
        f"summary({run_id}-b0)",
        f"summary({run_id}-b1)",
    ]
    assert {r["ack"]["id"] for r in results} == {"msg-1", "msg-2"}
    assert sorted(domain.fn_calls) == sorted([f"{run_id}-b0", f"{run_id}-b1"])
    assert len(domain.action_calls) == 2  # exactly once per branch, across the crash


def test_two_gather_branches_hand_an_action_tool_distinct_idempotency_keys(backend):
    """Exactly-once for an action tool that dedupes on its key, under gather: two branches
    requesting the same action hand the tool two keys, one per branch, so a deduping tool sends
    two emails where the workflow asked for two."""
    _, domain, _, _, _ = _run_code_run(backend, factory=gather_run_code_wf)
    keys = sorted(call["idempotency_key"] for call in domain.action_calls)
    # each key carries its own branch's frame
    assert [(";gather:0,0;" in k, ";gather:0,1;" in k) for k in keys] == [
        (True, False),
        (False, True),
    ], keys


def test_durable_run_code_duplicate_names_agree_across_engines(backend):
    """Two same-named run_code calls behave identically on both engines: the
    second EXECUTES
    (occurrence-suffixed checkpoints), never a silent stale read of the first."""
    snap, domain, _, _, _ = _run_code_run(backend, factory=duplicate_run_code_wf)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {
        "first": {"who": "summary(first)"},
        "second": {"who": "summary(second)"},
    }
    assert domain.fn_calls == ["first", "second"]  # both ran live — no stale read


def test_durable_run_code_gather_with_human_tier_parks_per_branch(backend):
    """`run_code`'s HITL actions COMPOSE with gather on the durable path. Each
    branch's human tier parks on its own QUALIFIED approval event, so two gated
    branches need TWO approvals (the permission-granularity pin: one emission
    must never approve a sibling); approvals resolve lowest-index-first
    (serialized wakes), each action fires exactly once, and the world is untouched while parked."""
    domain = CodeActionDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    scoped = human(
        Approval,
        # Composed, not f-stringed: a `Key` has no `__str__`, so an f-string here would
        # bake `Key(_value=...)` into the park name; and the composer is where a
        # run-scoped approval name belongs regardless.
        event_name=lambda op: compose_key(
            # lint: terminal-hole — `op_key` returns a `Key`, which the composer splices by
            # INDUCTION; wrapping it in a `Segment` would be a regression, and since `Segment`
            # takes a `str` it now refuses one outright.
            t"{APPROVE}:{Segment(run_id)};{op_key(op):domain=identity}"
        ),
    )
    layers = (cascade([rules(_gate_send_email), scoped]),)
    name = f"code-{run_id}"
    backend.register(name, gather_run_code_wf, domain, Fault(None), layers)
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # parked on branch 0's approval
    assert domain.action_calls == []  # the world untouched while parked

    approve = f"approve:{run_id};step;code:action,0,c;tool:send_email"
    backend.emit_event(task_id, f"gather:0,0;{approve}", {"decision": "approve"})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), snap  # branch 1 still needs ITS approval
    assert len(domain.action_calls) == 1  # branch 0's action fired, exactly once

    backend.emit_event(task_id, f"gather:0,1;{approve}", {"decision": "approve"})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    results = snap.result["results"]
    assert [r["summary"] for r in results] == [
        f"summary({run_id}-b0)",
        f"summary({run_id}-b1)",
    ]
    assert {r["ack"]["id"] for r in results} == {"msg-1", "msg-2"}
    assert len(domain.action_calls) == 2  # exactly once per branch, across both parks


# ── measured-spend accrual: the definition-of-done pins, both engines ─────
def _drain_with_grants(backend, task_id, run_id, grant, *, max_parks=6):
    """Drive to a terminal state, delivering one grant per park. Returns
    ``(snap, parks)`` — ``parks`` is the number of times the task ASKED, which by
    construction equals the number of grants emitted (the count(parks)==count(emits) pin:
    every trip parks, every park is answered once)."""
    parks = 0
    for _ in range(max_parks + 1):
        snap = backend.run_until_result(task_id)
        assert snap is not None
        if snap.state in ("completed", "failed", "cancelled"):
            return snap, parks
        backend.emit_event(task_id, f"budget-grant:{run_id},{parks}", grant)
        parks += 1
    return backend.run_until_result(task_id), parks


def test_measured_trip_parks_grants_and_resumes(backend):
    """Park → grant → resume, total real spend within the granted ceiling, and one park
    per trip (count(parks)==count(emits)). The ceiling (0.0015) sits between 2x and 3x the
    per-ask cost, so the 3rd sequential ask trips exactly once."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mt-{uuid4().hex[:8]}"
    name = f"metered-trip-{run_id}"
    backend.register(
        name,
        metered_trip_wf,
        domain,
        Fault(),
        (),
        budget_limit=0.0015,
        on_exhaust="park",
    )
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.01})

    assert snap.state == "completed", snap
    assert snap.result == {"asks": ["ans", "ans", "ans"]}  # the 3rd ask ran after the grant
    assert parks == 1  # exactly one trip, one grant
    real_spend = len(domain.calls) * domain.cost  # LIVE calls only — replay re-binds
    assert real_spend == pytest.approx(0.003)  # exactly-once: 3 asks charged
    # A generous grant, so the one-ask pre-check overshoot is invisible here; the tight,
    # overshoot-aware bound (spend <= limit + grants + one ask) is pinned separately by
    # test_measured_trip_just_sufficient_grant_respects_the_overshoot_bound.
    assert real_spend <= 0.0015 + 0.01  # within the granted ceiling (with headroom)


def test_measured_trip_inside_a_scope_parks_on_the_UNSCOPED_grant_name(backend):
    """A measured trip inside a `scoped(...)` parks on `budget-grant:{run_id},{trip_n}` with no
    scope frame — so the SAME emitter drives it.

    The pin is that `_drain_with_grants` is reused UNCHANGED: it composes the bare name, exactly
    as every real emitter does (`effective.voi`, the grant-injection seam). If the park were
    scope-qualified this test would hang at the first trip.

    Everything else must match `test_measured_trip_parks_grants_and_resumes` — same park count,
    same spend — so the scope changes the KEYS INSIDE the body and nothing about the authority
    boundary."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"smt-{uuid4().hex[:8]}"
    name = f"scoped-metered-trip-{run_id}"
    backend.register(
        name,
        scoped_metered_trip_wf,
        domain,
        Fault(),
        (),
        budget_limit=0.0015,
        on_exhaust="park",
    )
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.01})

    assert snap.state == "completed", snap
    assert snap.result == {"asks": ["ans", "ans", "ans"]}
    assert parks == 1
    assert len(domain.calls) * domain.cost == pytest.approx(0.003)


def test_measured_trip_fail_fast_aborts_within_the_ceiling(backend):
    """`on_exhaust="fail"` refuses the over-budget ask without parking — the unattended
    default. The trip is a PRE-check, so the last allowed ask can push spend one ask past
    the limit (an ask's cost is unknown until it runs); the crisp invariant is that the
    over-budget ask itself is refused, and spend stays within one ask of the ceiling."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mf-{uuid4().hex[:8]}"
    name = f"metered-fail-{run_id}"
    backend.register(
        name,
        metered_trip_wf,
        domain,
        Fault(),
        (),
        budget_limit=0.0015,
        on_exhaust="fail",
    )
    task_id = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)

    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "failed", snap
    assert len(domain.calls) == 2  # the 3rd ask (which would cross the ceiling) was refused
    assert len(domain.calls) * domain.cost <= 0.0015 + domain.cost  # limit + one in-flight ask


def test_measured_trip_survives_worker_death(backend):
    """Park x crash: a one-shot crash at the k-th ctx-op touch (steps, the grant await)
    during the park→grant→resume cycle still converges to the identical outcome, exactly
    once, within the granted ceiling — for every k."""
    for k in range(1, 6):
        domain = MeteringDomain(cost=0.001)
        run_id = f"mk-{uuid4().hex[:8]}"
        name = f"metered-crash-{run_id}"
        backend.register(
            name,
            metered_trip_wf,
            domain,
            Fault(k),
            (),
            budget_limit=0.0015,
            on_exhaust="park",
        )
        task_id = backend.spawn(name, run_id, contract=Contract.V1)

        snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.01})

        assert snap is not None, f"k={k}"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result == {"asks": ["ans", "ans", "ans"]}, f"k={k}"
        assert parks == 1, f"k={k}: crash must not spawn spurious trips"  # trip re-derives once
        assert len(domain.calls) == 3, f"k={k}: {domain.calls}"  # exactly once despite crash
        assert len(domain.calls) * domain.cost <= 0.0015 + 0.01, f"k={k}"


def test_measured_concurrent_gather_runaway_trips_deterministically(backend):
    """The §7 concurrent pin: two branches spend CONCURRENTLY, their subtotals fold into
    the parent meter (index order — confluent, schedule-independent), and the following
    sequential ask trips deterministically. Branches never trip (sequential-only, §5), so
    exactly one park, and real spend stays within the granted ceiling."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mg-{uuid4().hex[:8]}"
    name = f"metered-gather-{run_id}"
    backend.register(
        name,
        metered_gather_trip_wf,
        domain,
        Fault(),
        (),
        budget_limit=0.0015,
        on_exhaust="park",
    )
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.01})

    assert snap.state == "completed", snap
    assert parks == 1  # only the sequential tail ask trips; the gather branches do not
    assert len(domain.calls) == 3  # g0, g1, tail — exactly once
    assert len(domain.calls) * domain.cost <= 0.0015 + 0.01


def test_measured_v0_task_stays_v0_under_a_v1_capable_worker(backend):
    """The migration guard, end-to-end: a task spawned with NO contract param
    (indistinguishable from one in flight before the envelope existed) resolves to v0 via
    `Contract.from_params`, so the SAME v1-capable worker runs it on the bare path: no
    envelope, and the measured ceiling is inert (the trip lives on the v1 arm only). It
    completes without ever parking, even with a budget that WOULD trip under v1."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mv0-{uuid4().hex[:8]}"
    name = f"metered-v0-{run_id}"
    backend.register(name, metered_trip_wf, domain, Fault(), (), budget_limit=0.0015)
    task_id = backend.spawn(name, run_id)  # NO contract param → v0

    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap  # never parked — the trip is inert on v0
    assert len(domain.calls) == 3  # all three asks ran; the ceiling did not gate them


def test_measured_trip_just_sufficient_grant_respects_the_overshoot_bound(backend):
    """The corrected §7 bound. The trip is a PRE-check, so a barely-sufficient grant lets the
    over-budget ask run and its cost lands PAST `limit + Σ grants` — the true invariant is
    `spend <= limit + Σ grants + one ask`, non-compounding. A grant of 0.0006 clears the
    check (0.002 < 0.0015 + 0.0006) but not the spend (0.003 > 0.0021)."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mj-{uuid4().hex[:8]}"
    name = f"metered-just-{run_id}"
    backend.register(name, metered_trip_wf, domain, Fault(), (), budget_limit=0.0015)
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.0006})

    assert snap.state == "completed", snap
    assert parks == 1
    real_spend = len(domain.calls) * domain.cost
    limit_plus_grants = 0.0015 + 0.0006
    assert real_spend > limit_plus_grants  # the one-ask overshoot is REAL (the naive pin is false)
    assert real_spend <= limit_plus_grants + domain.cost  # the true, overshoot-aware bound


def test_measured_trip_multiple_grants_re_derive_trip_n_across_replays(backend):
    """Insufficient grants force several parks at ONE gate: `trip_n` increments 0,1,2 and,
    because each grant delivery replays the task from scratch, re-derives deterministically
    (a park re-binds `budget-grant:{run_id},{trip_n}` by name). Three 0.0002 grants clear
    0.0015 → 0.0021."""
    domain = MeteringDomain(cost=0.001)
    run_id = f"mm-{uuid4().hex[:8]}"
    name = f"metered-multi-{run_id}"
    backend.register(name, metered_trip_wf, domain, Fault(), (), budget_limit=0.0015)
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.0002})

    assert snap.state == "completed", snap
    assert parks == 3  # trip_n 0,1,2 — each a distinct grant name, re-derived on every replay
    assert len(domain.calls) == 3  # exactly-once despite the repeated park/resume replays


def test_measured_concurrent_gather_trip_survives_worker_death(backend):
    """The cross-product: the CONCURRENT metered gather plus a one-shot crash at the k-th
    ctx op still converges. The confluent per-branch fold and the deterministic trip
    re-derive through the crash, exactly once, every k."""
    for k in range(1, 7):
        domain = MeteringDomain(cost=0.001)
        run_id = f"mgc-{uuid4().hex[:8]}"
        name = f"metered-gather-crash-{run_id}"
        backend.register(name, metered_gather_trip_wf, domain, Fault(k), (), budget_limit=0.0015)
        task_id = backend.spawn(name, run_id, contract=Contract.V1)

        snap, parks = _drain_with_grants(backend, task_id, run_id, {"add_dollars": 0.01})

        assert snap is not None, f"k={k}"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert parks == 1, f"k={k}"  # the tail ask trips once; branches never trip
        assert len(domain.calls) == 3, f"k={k}: {domain.calls}"  # g0, g1, tail — exactly once


def test_a_flaky_metered_service_recovers_under_serve_on_both_engines(backend):
    """The DURABLE v1 arm under a `serve(retry_domain(...))` stack, on both engines: the arm
    production runs, and the one the in-process `measured_drive` pin does not cover.

    Retry lives BELOW the checkpoint seam, so the recovery is invisible above it: the workflow
    completes, and the meter charges exactly one usage per ask because a failed attempt yields
    no usage to fold. (This is also the V1-metered-only path `_check_domain_reachable` cannot
    prove safe at assembly, so execution proves it here.)"""
    base = FlakyMeteringDomain(cost=0.001, fail_first=2)
    domain = serve(retry_domain(2), base=base)
    run_id = f"flaky-{uuid4().hex[:8]}"
    name = f"flaky-metered-{run_id}"
    backend.register(name, metered_trip_wf, domain, Fault(), (), budget_limit=1.0)
    task_id = backend.spawn(name, run_id, contract=Contract.V1)

    snap = backend.run_until_result(task_id)

    assert snap.state == "completed", snap
    assert snap.result == {"asks": ["ans", "ans", "ans"]}
    assert base.failures == 2  # the transients really fired...
    assert len(base.calls) == 3  # ...and exactly three calls were CHARGED, not five


def test_op_seam_retry_then_crash_calls_the_domain_once_per_try(backend):
    """A flaky first try makes op-seam `retry` re-forward the Step, which lands at the step's own
    placement. A crash after the retried success commits replays and is served it, so the domain
    sees one call per try and none from the replay."""
    domain = CountingDomain(flaky=True)
    run_id = f"rtc-{uuid4().hex[:8]}"
    name = f"retry-crash-{run_id}"
    # Crash right after the retried step commits, forcing a replay that must find it.
    backend.register(name, two_step_wf, domain, Fault(k=3), (retry(2),))
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)

    assert snap.state == "completed", snap
    # The workflow yields 'a' once. A flaky first attempt costs one extra LIVE call; anything
    # beyond that is the replay re-executing a step it should have re-bound from a checkpoint.
    assert domain.calls.count("a") == 2, domain.calls


# ── the checkpoint READER: one key sequence, two engines ─────────────────────
#
# The seed reader is what a fork is built on (`fork_seed` maps key -> raw state). Without an
# Absurd reader a durable fork could be proven on the 0<->1 engine and remain unported on the
# deployed 0<->N one; `read_absurd_task` is that half. Its correctness claim is not
# "it returns rows" but "it returns the SAME key sequence its SQLite sibling does", which is
# exactly what an engine-parametrized assertion against a literal expectation pins: the literal is
# engine-independent, so both engines must produce it.
#
# The two engines' checkpoint TABLES genuinely differ: Absurd persists `$awaitEvent:` freezes and
# `sleep_until` wake times as checkpoint rows and orders by `updated_at` with no ordinal; SQLite
# parks via `tasks.waiting_event`, writes nothing for a sleep, and has `rowid`. `ENGINE_INTERNAL`
# is the enumeration that reconciles them, and this case is what keeps that enumeration honest.


def test_checkpoint_reader_projects_one_key_sequence_on_both_engines(backend):
    """The reader pair's contract: op keys, unfiltered except for the engine's own suspension
    bookkeeping. A `gather` run covers the three key shapes a fork seed must carry (`Step`,
    `ledger;`, and branch-prefixed keys).

    ORDER IS ASSERTED AS THE SUBSTRATE ACTUALLY GUARANTEES IT. Branch events keep *within*-branch
    order, and cross-branch interleaving is a race, so the order is partial. Concurrent branches
    commit as their tool work finishes, so commit order across branches is a race on BOTH engines
    and pinning a total order would pin a schedule. What is guaranteed, and what a fork seed
    depends on, is the multiset of keys and each branch's INTERNAL order, so that is what this
    asserts. `fork_seed` cuts the prefix at the first occurrence of `through` in commit order, so
    a `through` inside a gather region is not a stable cut."""
    domain = CountingDomain()
    run_id = f"ck-{uuid4().hex[:8]}"
    name = f"ckpt-{run_id}"
    backend.register(name, gather_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap

    keys = backend.checkpoint_keys(task_id)
    assert sorted(keys) == sorted(
        [
            "gather:0,0;step;tool:a",
            f"gather:0,0;ledger;{run_id}:ka",
            "gather:0,1;step;tool:b",
            f"gather:0,1;ledger;{run_id}:kb",
        ]
    )
    for branch, tool, kind in ((0, "a", "ka"), (1, "b", "kb")):
        within = [k for k in keys if k.startswith(f"gather:0,{branch};")]
        assert within == [
            f"gather:0,{branch};step;tool:{tool}",
            f"gather:0,{branch};ledger;{run_id}:{kind}",
        ]


def test_checkpoint_reader_excludes_the_engines_await_bookkeeping(backend):
    """The divergence that would otherwise make the two engines incomparable — and make a fork
    unrunnable on Absurd. A parked-then-resumed await leaves a `$awaitEvent:` CHECKPOINT on
    Absurd and no checkpoint at all on SQLite; the reader must show neither, or `fork_seed` would
    carry a key that nothing replays through `ctx.step` and `run_fork` would refuse a valid fork
    for a leftover `unconsumed()` entry."""
    domain = CountingDomain()
    run_id = f"aw-{uuid4().hex[:8]}"
    name = f"gated-{run_id}"
    backend.register(name, gated_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)
    backend.run_until_result(task_id)  # parks on review:{run_id}
    backend.emit_event(task_id, review_name(run_id).stored(), {"decision": "approve"})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap

    keys = backend.checkpoint_keys(task_id)
    assert keys == [
        "step;tool:a",
        # lint: terminal-hole — a `Key` splice
        compose_key(t"ledger;{done_id(run_id):domain=address}").stored(),
    ]
    assert not any(k.startswith("$") for k in keys)


class _NestedPayloadDomain:
    """Returns one deliberately awkward value — the shape a store normalizes if it is going to.

    Keys out of alphabetical order, a float, and an integer past 2^53 (where a JSON reader that
    round-trips through a double would lose the low bits). Postgres `jsonb` really does normalize:
    it reorders keys, drops duplicates and re-renders numbers, so a seeded copy on that engine is
    only ever value-identical, never byte-identical. This payload is chosen so a normalization
    that *changed the value* would show up."""

    payload: ClassVar[dict[str, Any]] = {
        "zeta": 1,
        "alpha": {"nested": [1, 2, {"deep": True}]},
        "cost": 0.010,
        "big": 9007199254740993,  # 2**53 + 1
    }

    def run(self, op: DomainOp) -> Any:
        return dict(self.payload)

    def run_metered(self, op: DomainOp) -> tuple[Any, Usage]:
        return self.run(op), Usage()


def test_a_seeded_prefix_value_survives_this_ENGINE_s_own_store(backend):
    """A fork child's seeded checkpoint holds the base's recorded value — through the STORE.

    **Why this is a conformance case and not a unit test.** `test_fork_seeding.py` already pins
    `SeedingCtx`'s contract against a spy ctx: a seeded step returns the raw value, never calls
    the domain, and commits through the inner ctx. That proves the logic and touches no database.
    The failure this case exists for happens one layer down, in serialization — and the two
    engines serialize differently *in kind*: SQLite writes checkpoint state as TEXT via
    `json.dumps`, Absurd writes `jsonb`, which is explicitly not byte-preserving (it reorders
    keys, drops duplicates, re-renders numbers). The other tests that
    reach `SeedingCtx.step` on Absurd (`scripts/cov_contexts.py` lists them) assert fork outcomes,
    results and ledger kinds, never a seeded value. Without this case the round-trip is pinned on
    one engine of two, and a green SQLite pass does not verify a port.

    **Why it matters that the copy is faithful.** A fork makes a counterfactual durable by
    re-committing the base's values into the CHILD's own checkpoints, so a crash mid-tail replays
    the prefix from the child and never re-reads the base. `test_fork_sweep_cost.py` measures what
    that costs (rows = forks x prefix_len). This asserts the bill buys something: a lossy copy
    would be cheaper and wrong, and wrong *quietly*: the v1 meter folds usage out of the raw
    `{result, usage}` envelope after `ctx.step` returns, so a mangled seed replays green while the
    meter under-derives cost.

    **Value identity, not byte identity.** `jsonb` does not preserve bytes, so the assertion
    compares DECODED values: canonical value identity."""
    domain = _NestedPayloadDomain()
    subject = f"m{uuid4().hex[:8]}"  # authored names scope on the SUBJECT, so they are fork-stable

    def make_wf(await_name: str | Key):
        """The child awaits a RENAMED event, which `run_fork` does for real via
        `fork_event_name`. Not cosmetic: Absurd delivers events by NAME across the queue, so a
        child sharing the base's await name receives the base's answer and silently replays the
        decision the counterfactual exists to replace. SQLite addresses events by `(task_id,
        name)` and hides that entirely — the sharpest little instance of why this case is
        parametrized. The prefix op key (`tool:a`) is unaffected, so the seed still matches."""

        def seeded_wf(_rid: str):
            recorded = yield from call_tool("a", {}, dict)
            approval = yield from await_event(await_name, dict)
            return {"a": recorded, "decision": approval["decision"]}

        return seeded_wf

    base_await = compose_key(t"review:{Segment(subject)}").stored()
    # `Key.parse`: this name is the SUBSTRATE's (`fork_event_name` composes it on the real
    # path), hand-built here to stand in for the ctx rename. Author text may not name `fork:`.
    child_await = fork_event_name(subject, Key.parse(base_await))
    seeded_wf = make_wf(base_await)
    base_run = f"seedbase-{subject}"
    base_name = f"seed-base-{subject}"
    backend.register(base_name, seeded_wf, domain, Fault(), ())
    base_id = backend.spawn(base_name, base_run)
    backend.run_until_result(base_id)
    backend.emit_event(
        base_id, compose_key(t"review:{Segment(subject)}").stored(), {"decision": "reject"}
    )
    snap = backend.run_until_result(base_id)
    assert snap is not None
    assert snap.state == "completed", snap

    base_states = backend.checkpoint_states(base_id)
    # `Key.parse`, not `"tool:a"`: `checkpoint_states` is keyed by IDENTITY, and a `str`
    # lookup in a `Key`-keyed dict simply MISSES — no error, no match. That is the one silent
    # hazard the opaque type introduces, and the guard below is what turns it loud here.
    prefix_key = Key.parse("step;tool:a")
    assert prefix_key in base_states, base_states  # the seed has something to carry

    # The child replays the prefix from the seed and runs its tail live — the fork shape, minus
    # the fork spine (spawn/join/ledger sealing are pinned by `test_fork_sweep*`, so re-asserting
    # them here would be the overlap this case is scoped to avoid).
    seed = {prefix_key: base_states[prefix_key]}
    child_run = f"seedchild-{subject}"
    child_name = f"seed-child-{subject}"
    backend.register(
        child_name,
        make_wf(child_await),
        domain,
        Fault(),
        (),
        wrap=lambda ctx: SeedingCtx(ctx, seed, fork_point=child_await),
    )
    child_id = backend.spawn(child_name, child_run)
    backend.run_until_result(child_id)
    backend.emit_event(child_id, child_await.stored(), {"decision": "approve"})
    snap = backend.run_until_result(child_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"a": domain.payload, "decision": "approve"}  # the tail diverged

    child_states = backend.checkpoint_states(child_id)
    # the copy is faithful THROUGH THE STORE, and faithful to what the domain produced
    assert child_states[prefix_key] == base_states[prefix_key]
    assert base_states[prefix_key] == domain.payload


def test_checkpoint_reader_preserves_duplicate_occurrence_suffixes(backend):
    """`SeedingCtx` is occurrence-aware and looks up `name#2` verbatim, so — unlike
    `export_measured_prefix`, which normalizes them away for the positionally-keyed measured
    driver — the seed reader must keep the suffix the engine wrote."""
    domain = CountingDomain(incrementing=True)
    run_id = f"dup-{uuid4().hex[:8]}"
    name = f"dup-{run_id}"

    def repeated_wf(rid: str):
        first = yield from call_tool("a", {}, int)
        second = yield from call_tool("a", {}, int)  # the SAME step name, twice
        return {"first": first, "second": second}

    backend.register(name, repeated_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"first": 11, "second": 12}  # each occurrence ran its own call

    assert backend.checkpoint_keys(task_id) == ["step;tool:a", "step;tool:a#2"]


def test_durable_park_inside_recurses_fold_carries_the_fold_frames(backend):
    """The fold frames on a real engine, both engines: the durable half of `recurse@fold`.

    `recurse`'s `combine` runs under `gather:{g},{k};fold:{level},{k}`. `descend_recurse_wf` runs
    a fold gather with a PURE combine, so its frames never reach a key, and a recording-path law
    cannot stand in, because parks are exactly where the two engines diverge.

    Two things are asserted, and the second is the sharp one: the committed checkpoint key
    carries the fold frames, and the PARK reaches the engine with them — composed through
    `qualified_event_name`, so this agrees with the emitter contract rather than with a literal
    that could drift alongside a bug. The frame-short name a call-site reader would guess must
    NOT wake it.

    The coordinate is `gather:1,0`; `recurse` issues its leaf gather first and its fold gather
    second at the same level, so the `{g}` ordinal is the only discriminator between them."""
    domain = CountingDomain()
    run_id = f"fold-{uuid4().hex[:8]}"
    name = f"recursefold-{run_id}"
    backend.register(name, recurse_fold_park_wf, domain, Fault(), ())
    task_id = backend.spawn(name, run_id)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    # Positively parked, not merely "not finished": `pending` (the shared-queue starvation state)
    # also satisfies a negation, so a task that was never claimed would pass one. Both engines
    # produce a parked state of their own name.
    assert snap.state in ("waiting", "sleeping"), snap
    assert sorted(domain.calls) == ["a", "a", "a"]  # two leaves committed, then the merge

    fold_frames = (GatherBranch(1, 0), compose_key(t"fold:{0},{0}"))
    # The FRAMES are what this asserts, and since step 8 the frames are all an address and an
    # identity share — `qualified_event_name` composes the ADDRESS an emitter delivers to, while
    # a committed step's key is its op key, which opens with the `step` arm. Under one set of
    # frames they now read:
    #
    #     gather:1,0;fold:0,0;merge          the park's wire name (address)
    #     gather:1,0;fold:0,0;step:merge     the merge step's checkpoint (identity)
    #
    # They were one string before the arm, which is why this assertion could compare them. Ask
    # `step_key` for the leaf rather than re-spelling it, so the two stay derived from the same
    # minters the emitter contract uses.
    committed = qualified_event_name(*fold_frames, name=step_key("merge").stored())
    assert committed.stored() in set(backend.checkpoint_keys(task_id)), (
        "the merge step must commit UNDER the fold frames"
    )
    # And the two differ ONLY by the arm — the frames are shared, which is the property that
    # makes an emitter's name and the step's checkpoint locatable from one another.
    address = qualified_event_name(*fold_frames, name="merge")
    assert committed.stored() == address.stored().replace(";merge", ";step:merge")

    # The name a fold-frame-blind park would have used — must not wake it.
    backend.emit_event(task_id, branched(1, 0, ev_name(run_id)).stored(), {"extra": 99})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed"), (
        "a frame-short name must not wake a fold park"
    )

    backend.emit_event(
        task_id, qualified_event_name(*fold_frames, name=f"ev:{run_id}").stored(), {"extra": 5}
    )
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"total": 10 + 10 + 5}  # both leaves, then the awaited answer


# --- event delivery: a STICKY latch, and what its key includes ---------------------------------
#
# **An event is a `threading.Event`**: a manual-reset latch nobody resets. Emitting SETS it, and
# it stays set: immutable, first-write-wins, and it satisfies every await that arrives afterwards.
# A fork child emits its done-event when it finishes, which may be before the parent has reached
# its await; the parent's await then reads the already-set latch and returns without parking. If
# the latch did not stay set, that emit would land with no waiter and the parent would park
# forever. (`threading.Condition` is the near neighbor to keep straight: `notify_all` has no
# memory, so a waiter arriving after the notify blocks.)
#
# The stickiness is shared, and SO IS THE LATCH'S KEY:
#
#     engine   latch key               one latch per
#     SQLite   events(name)            name
#     Absurd   e_{queue}(event_name)   name, queue-wide
#
# The three tests below are SHARED assertions of that one fact, which is stronger than a pin: a
# third engine cannot state its own answer, it has to meet this.
#
# The vendored schema keys an event by name alone. `absurd.emit_event(p_queue_name, p_event_name,
# p_payload)` takes no task while `absurd.await_event` takes `p_task_id`: the id is available at
# wait time and deliberately absent at emit, because one emit is meant to wake every waiter.
# Queues are physical tables (`create table … absurd.%I` over `e_`/`w_`/`r_`/`t_` + queue name),
# so a per-task queue is no workaround either. The SDK and `absurd.sql` are CO-VERSIONED
# (`infra/absurd/PIN.txt`), so an upgrade could still change the reference under us; it must
# redden here and not in a production run.
#
# Consequence for the substrate, on BOTH engines: a park's identity cannot rely on the engine
# separating it. The NAME is the whole address, which is why the
# authority namespaces carry a generation coordinate and why `test_grant_aliasing`'s fourth case is
# documented as "still aliases, and should".


def _fresh_token() -> str:
    """A letter-led unique atom. The leading letter is not cosmetic: a bare `uuid4().hex[:8]` is
    digit-led half the time, and the atom rule refuses a digit-led NAME with a `KeySyntaxError`
    that takes the whole run down."""
    return "t" + uuid4().hex[:8]


def _awaiting_shared_name(token: str):
    def wf(_run_id: str):
        answer = yield from await_event(compose_key(t"shared:{Segment(token)}"), _Ack)
        return {"got": answer.ok}

    return wf


class _Ack(BaseModel):
    ok: bool = True


def test_an_event_emitted_before_the_await_is_still_found_on_both_engines(backend):
    """The SHARED half: on both engines the latch stays set, so an emit that lands before the
    await is read by that await rather than lost.

    Asserted rather than assumed, because it is what a fork child relies on when it emits its
    done-event without knowing whether the parent has reached its await."""
    token = _fresh_token()
    backend.register(f"early-{token}", _awaiting_shared_name(token), None, Fault(), ())
    task_id = backend.spawn(f"early-{token}", "run-early")
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})  # nothing is parked yet
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"got": True}


def test_one_emit_wakes_EVERY_waiter_on_the_name_on_both_engines(backend):
    """One emit, two waiters on one name, both woken: one SHARED assertion for both engines.

    A latch keyed on an addressee wakes only that one; Absurd wakes both because the name is the
    whole key. A shared assertion is strictly stronger than a per-engine pin: a third engine
    cannot state its own answer here, it has to meet this one.

    The property: a park's identity cannot rely on the engine separating
    it. That is why the authority namespaces carry a generation coordinate, and why
    `test_grant_aliasing`'s fourth case is documented as "still aliases, and should"."""
    token = _fresh_token()
    backend.register(f"shared-{token}", _awaiting_shared_name(token), None, Fault(), ())
    first = backend.spawn(f"shared-{token}", "run-a")
    second = backend.spawn(f"shared-{token}", "run-b")
    parked = [backend.run_until_result(t).state for t in (first, second)]
    assert all(state in ("waiting", "sleeping") for state in parked), parked

    backend.emit_event(first, f"shared:{token}", {"ok": True})  # ONE emit, at `first`
    assert backend.run_until_result(first).state == "completed"
    assert backend.run_until_result(second).state == "completed"


def _waiting_until(token: str, deadline: datetime):
    """One bounded wait, and the arm it took as the run's answer."""

    def wf(_run_id: str):
        outcome = yield from await_until(
            compose_key(t"shared:{Segment(token)}"), _Ack, deadline=deadline
        )
        match outcome:
            case Arrived(payload=ack):
                return {"arrived": ack.ok}
            case Expired():
                return {"arrived": None}

    return wf


def test_a_bounded_wait_takes_an_event_already_emitted_on_both_engines(backend):
    """An event in the store answers the wait, and the deadline is never consulted.

    The unbounded half is `test_an_event_emitted_before_the_await_is_still_found_on_both_engines`;
    a deadline must not cost a workflow the latch that case relies on.
    """
    token = _fresh_token()
    deadline = datetime.now().astimezone() + timedelta(hours=1)
    backend.register(f"bw-early-{token}", _waiting_until(token, deadline), None, Fault(), ())
    task_id = backend.spawn(f"bw-early-{token}", "run-bw-early")
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})  # nothing is parked yet
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": True}


def test_a_bounded_wait_parks_then_takes_its_event_on_both_engines(backend):
    """The park half: the deadline is an hour out, so the wait is still open when the emit lands.

    The park is asserted before the emit, or a wait that answered without ever suspending would
    satisfy the rest of this and the deadline column would go untested.
    """
    token = _fresh_token()
    deadline = datetime.now().astimezone() + timedelta(hours=1)
    backend.register(f"bw-park-{token}", _waiting_until(token, deadline), None, Fault(), ())
    task_id = backend.spawn(f"bw-park-{token}", "run-bw-park")
    parked = backend.run_until_result(task_id)
    assert parked.state in ("waiting", "sleeping"), parked

    backend.emit_event(task_id, f"shared:{token}", {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": True}


def test_a_bounded_wait_whose_deadline_passed_expires_on_both_engines(backend):
    """No event, a deadline already behind: the run ends at the clock and says so.

    This is the answer a deadline-free await has no way to give, so it is the whole reason the
    outcome is a union rather than a payload.
    """
    token = _fresh_token()
    deadline = datetime.now().astimezone() - timedelta(seconds=1)
    backend.register(f"bw-late-{token}", _waiting_until(token, deadline), None, Fault(), ())
    task_id = backend.spawn(f"bw-late-{token}", "run-bw-late")
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": None}


def _expiring_then_parking(token: str, deadline: datetime):
    """A wait the clock ends, then a park, so the run REPLAYS the first wait after the park.

    The park is what makes this test say anything. A task that ran straight to completion could
    not re-execute its wait at all, so a late event would be ignored whether or not the outcome
    was ever recorded: the assertion would hold over an empty domain. Measured by mutation —
    dropping the expiry's settle on the Absurd side left the straight-line version green.
    """

    def wf(_run_id: str):
        first = yield from await_until(
            compose_key(t"shared:{Segment(token)}"), _Ack, deadline=deadline
        )
        yield from await_event(compose_key(t"release:{Segment(token)}"), _Ack)
        match first:
            case Arrived(payload=ack):
                return {"arrived": ack.ok}
            case Expired():
                return {"arrived": None}

    return wf


def test_a_bounded_wait_that_expired_ignores_a_later_event_on_both_engines(backend):
    """One wait, one answer: the expiry is recorded, so a replay serves it rather than waiting.

    The late event is delivered to the SAME name the expired wait held, and the release wakes the
    task into a replay that reaches that wait again with the event now in the store. An engine
    that recorded nothing answers `Arrived` on that pass.
    """
    token = _fresh_token()
    deadline = datetime.now().astimezone() - timedelta(seconds=1)
    body = _expiring_then_parking(token, deadline)
    backend.register(f"bw-stale-{token}", body, None, Fault(), ())
    task_id = backend.spawn(f"bw-stale-{token}", "run-bw-stale")
    parked = backend.run_until_result(task_id)
    assert parked.state in ("waiting", "sleeping"), parked

    backend.emit_event(task_id, f"shared:{token}", {"ok": True})  # the late event
    backend.emit_event(task_id, f"release:{token}", {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": None}, "a late event reopened a decided wait"


def _waiting_while_the_deadline_passes(token: str, deadline: datetime):
    """One bounded wait whose deadline passes WHILE the task is parked."""

    def wf(_run_id: str):
        outcome = yield from await_until(
            compose_key(t"shared:{Segment(token)}"), _Ack, deadline=deadline
        )
        match outcome:
            case Arrived(payload=ack):
                return {"arrived": ack.ok}
            case Expired():
                return {"arrived": None}

    return wf


def test_an_event_after_the_deadline_leaves_an_expired_wait_alone_on_both_engines(backend):
    """A wake is owed to the waits the clock has not passed, which is what the reference does.

    `absurd.emit_event` deletes waits whose `timeout_at` has passed and wakes only those still
    ahead of it (`infra/absurd/absurd.sql`), so the late event reaches nobody. The embedded
    engine's own docstring claims the same behavior, and the sibling case cannot check it: there
    the deadline is already past at spawn, so the expiry is SETTLED before the event lands. Here
    nothing is settled when the emit arrives, which is the only window where the wake predicate
    decides the answer.
    """
    token = _fresh_token()
    deadline = datetime.now().astimezone() + timedelta(seconds=1)
    body = _waiting_while_the_deadline_passes(token, deadline)
    backend.register(f"bw-wake-{token}", body, None, Fault(), ())
    task_id = backend.spawn(f"bw-wake-{token}", "run-bw-wake")
    parked = backend.run_until_result(task_id)
    assert parked.state in ("waiting", "sleeping"), parked

    _wait_past(deadline, margin=0.5)
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})  # after the deadline
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": None}, "a late event woke a waiter the clock had passed"


def test_an_event_a_wait_never_parked_for_is_taken_on_both_engines(backend):
    """A wait that never parked has no wake to read, so the store answers it.

    The reference decides by WHY the task woke: the SDK raises its timeout from `wake_event`
    before it consults the events table, and on a wait's first call there is nothing there to
    read. So an event already in the store answers, whatever the deadline says — and the sibling
    case, where the task DID park and the clock woke it, is where the deadline wins.

    Reachable without contrivance: a task that fails before its wait and retries meets the wait
    again with its deadline (absolute, recorded) behind it and the event since landed.
    """
    token = _fresh_token()
    behind = datetime.now().astimezone() - timedelta(seconds=2)
    backend.register(f"bw-first-{token}", _waiting_until(token, behind), None, Fault(), ())
    task_id = backend.spawn(f"bw-first-{token}", "run-bw-first")
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})  # before the wait is ever run
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"arrived": True}, "a wait that never parked read a wake it never had"


def _parking_then_sleeping_then_waiting(
    token: str, deadline: datetime, wake_at: datetime, ahead: datetime
):
    """A wait the clock ends, a sleep, then a second wait on the same name.

    The sleep is not incidental: it ends the first park by a route that is not an emit, which is
    where a registration that outlives its park shows up. A failing step's retry re-queue and a
    worker death reach the same row state.
    """

    def wf(_run_id: str):
        name = compose_key(t"shared:{Segment(token)}")
        first = yield from await_until(name, _Ack, deadline=deadline)
        yield from sleep_until(wake_at)
        second = yield from await_until(name, _Ack, deadline=ahead)
        return {"first": _arm(first), "second": _arm(second)}

    return wf


def test_a_wake_does_not_outlive_the_park_it_ended_on_both_engines(backend):
    """A wake answers the wait it woke, and nothing later.

    The first wait parks and the clock ends it. The event then arrives, too late for that wait
    and in good time for the second — which has an hour of headroom and must read the store.
    An engine that keeps the first park's registration answers the second from a wake spent
    several claims earlier.
    """
    token = _fresh_token()
    now = datetime.now().astimezone()
    deadline, wake_at = now + timedelta(seconds=1), now + timedelta(seconds=2)
    body = _parking_then_sleeping_then_waiting(token, deadline, wake_at, now + timedelta(hours=1))
    backend.register(f"bw-stale-wake-{token}", body, None, Fault(), ())
    task_id = backend.spawn(f"bw-stale-wake-{token}", "run-bw-stale-wake")
    parked = backend.run_until_result(task_id)
    assert parked.state in ("waiting", "sleeping"), parked

    _wait_past(deadline, margin=0.2)
    backend.run_until_result(task_id)  # the clock ends the first wait; the sleep begins
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})
    _wait_past(wake_at, margin=0.2)

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"first": "expired", "second": "arrived"}, (
        "the second wait was answered by a wake the first one spent"
    )


def _two_waits_on_one_name(token: str, first_deadline: datetime, second_deadline: datetime):
    """Two bounded waits on ONE name: the shape a receive loop has."""

    def wf(_run_id: str):
        name = compose_key(t"shared:{Segment(token)}")
        first = yield from await_until(name, _Ack, deadline=first_deadline)
        second = yield from await_until(name, _Ack, deadline=second_deadline)
        return {"first": _arm(first), "second": _arm(second)}

    return wf


def _arm(outcome: WaitOutcome[Any]) -> str:
    """Which arm a bounded wait took, as a value a task result can carry."""
    match outcome:
        case Arrived():
            return "arrived"
        case Expired():
            return "expired"
        case unreachable:
            assert_never(unreachable)


def test_two_bounded_waits_on_one_name_answer_independently_on_both_engines(backend):
    """A wait records ITS outcome, so a second ask of the same name decides for itself.

    The first deadline is behind and the second is an hour ahead with its event emitted, so the
    two answers differ and a shared record shows up as the second inheriting the first. This is
    the aliasing the authority namespaces already carry a coordinate against: a park's identity
    cannot rely on the engine separating it.
    """
    token = _fresh_token()
    behind = datetime.now().astimezone() - timedelta(seconds=1)
    ahead = datetime.now().astimezone() + timedelta(hours=1)
    body = _two_waits_on_one_name(token, behind, ahead)
    backend.register(f"bw-two-{token}", body, None, Fault(), ())
    task_id = backend.spawn(f"bw-two-{token}", "run-bw-two")
    backend.run_until_result(task_id)
    backend.emit_event(task_id, f"shared:{token}", {"ok": True})
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"first": "expired", "second": "arrived"}, (
        "the second wait was answered by the first wait's record"
    )


# ── the bounded-wait table, as one assertion ─────────────────────────────────
#
# `wiki/concepts/bounded-wait.md` enumerates what decides a wait that names a deadline. Four
# review rounds each found a defect of one shape: an arm the two engines answered differently,
# in a cell nobody had written down. So the cells are the subject here rather than any one of
# them. A row is a HISTORY, because that is what the two engines disagree about: the same
# (event, deadline) pair answers differently depending on how the task got here.


def _answering(token: str, deadline: datetime):
    """One bounded wait, reporting the arm it took."""

    def wf(_run_id: str):
        outcome = yield from await_until(
            compose_key(t"shared:{Segment(token)}"), _Ack, deadline=deadline
        )
        return {"answer": _arm(outcome)}

    return wf


def _answering_then_parking(token: str, deadline: datetime):
    """A wait that REPLAYS: the release park is what sends the run back through it."""

    def wf(_run_id: str):
        name = compose_key(t"shared:{Segment(token)}")
        first = yield from await_until(name, _Ack, deadline=deadline)
        yield from await_event(compose_key(t"release:{Segment(token)}"), _Ack)
        return {"answer": _arm(first)}

    return wf


def _answering_after_a_wait(token: str, first: datetime, second: datetime):
    """Two waits on one name, reporting the SECOND; one claim decides both."""

    def wf(_run_id: str):
        name = compose_key(t"shared:{Segment(token)}")
        yield from await_until(name, _Ack, deadline=first)
        answer = yield from await_until(name, _Ack, deadline=second)
        return {"answer": _arm(answer)}

    return wf


def _answering_after_a_sleep(token: str, first: datetime, wake_at: datetime, second: datetime):
    """The same pair with a sleep between, so the second wait is decided on a LATER claim."""

    def wf(_run_id: str):
        name = compose_key(t"shared:{Segment(token)}")
        yield from await_until(name, _Ack, deadline=first)
        yield from sleep_until(wake_at)
        answer = yield from await_until(name, _Ack, deadline=second)
        return {"answer": _arm(answer)}

    return wf


PAST_THE_SKEW = 0.2
"""How long after a deadline an emit has to land for BOTH engines to agree it is late.

Each engine reads its OWN clock to decide: `time.time()` on the embedded engine,
`absurd.current_time()` on the reference. They share a host here and the gap between them is
sub-millisecond, so this is three orders of room. An empirical allowance, and not a bound for a
loaded runner or a managed database: a history asserting the INSTANT rather than the rule puts
both sides on the engine's clock instead (`tests/test_absurd_deadline_park.py`).
`wiki/concepts/bounded-wait.md` carries the measurement and what widens it."""


def _start(backend, token: str, body, tag: str):
    backend.register(f"bw-{tag}-{token}", body, None, Fault(), ())
    return backend.spawn(f"bw-{tag}-{token}", f"run-{tag}-{token}")


PARKED_STATES = ("waiting", "sleeping")
"""What a park looks like on either engine, asked by one predicate so two cannot drift.

`pending` is deliberately absent: it means CLAIMABLE, which is the opposite of parked. Counting it
would turn a drain that ran out of batches into a green park."""


def _answer(backend, task_id) -> str:
    """The arm the wait took, or ``"parks"`` where it took neither.

    A park is one of the decision table's five outcomes, so it is a value here rather than an
    assertion: a history that parks where it should have answered then reads as an offender row
    beside the rest, and one whose whole point is the park can say so."""
    snap = backend.run_until_result(task_id)
    assert snap is not None, "the run produced no snapshot"
    assert snap.state != "pending", f"the drain ran out of batches with the task claimable: {snap}"
    if snap.state in PARKED_STATES:
        return "parks"
    assert snap.state == "completed", snap
    return snap.result["answer"]


def _parks(backend, task_id) -> None:
    """A history that says the wait PARKED has to check it, or it measures another history."""
    snap = backend.run_until_result(task_id)
    assert snap.state in PARKED_STATES, f"the wait did not park: {snap}"


def _parked(backend, task_id) -> bool:
    """Whether the wait parked, for a history that can widen its budget and try again."""
    return backend.run_until_result(task_id).state in PARKED_STATES


PARK_BUDGETS = (0.5, 2.0, 8.0)
"""Deadlines to try for a history whose wait must PARK before the clock reaches it.

The deadline is fixed before the spawn, so a loaded server can reach the wait after it has already
passed: the wait expires on its first call, never parks, and the history quietly becomes the one
where nothing parked. Widening on a miss is what `test_durable_gather_sleep_parks_then_wakes`
does, and for the same reason — the fast path stays fast and a slow host costs seconds rather than
a wrong measurement. Measured: 0.5 missed once in roughly four full-suite runs under `-n auto`."""


def _settled_keeps_its_answer(backend, token, now):
    body = _answering_then_parking(token, now - timedelta(seconds=2))
    task = _start(backend, token, body, "settled")
    _parks(backend, task)  # on the release, with the first wait already decided
    backend.emit_event(task, f"shared:{token}", {"ok": True})
    backend.emit_event(task, f"release:{token}", {"ok": True})
    return _answer(backend, task)


def _the_clock_reached_a_parked_wait(backend, token, now):
    for budget in PARK_BUDGETS:
        token = _fresh_token()
        deadline = datetime.now().astimezone() + timedelta(seconds=budget)
        task = _start(backend, token, _answering(token, deadline), f"clock{budget}")
        if not _parked(backend, task):
            continue
        _wait_past(deadline, margin=PAST_THE_SKEW)
        backend.emit_event(task, f"shared:{token}", {"ok": True})  # too late for this wait
        return _answer(backend, task)
    raise AssertionError(f"the wait never parked, on any of {PARK_BUDGETS}s")


def _its_event_reached_a_parked_wait(backend, token, now):
    task = _start(backend, token, _answering(token, now + timedelta(hours=1)), "event")
    _parks(backend, task)
    backend.emit_event(task, f"shared:{token}", {"ok": True})
    return _answer(backend, task)


def _never_parked_with_its_event(backend, token, now):
    task = _start(backend, token, _answering(token, now + timedelta(hours=1)), "stored")
    backend.emit_event(task, f"shared:{token}", {"ok": True})
    return _answer(backend, task)


def _never_parked_past_its_deadline_with_its_event(backend, token, now):
    task = _start(backend, token, _answering(token, now - timedelta(seconds=2)), "late-stored")
    backend.emit_event(task, f"shared:{token}", {"ok": True})
    return _answer(backend, task)


def _never_parked_past_its_deadline(backend, token, now):
    task = _start(backend, token, _answering(token, now - timedelta(seconds=2)), "late")
    return _answer(backend, task)


def _a_later_wait_on_the_same_claim(backend, token, now):
    for budget in PARK_BUDGETS:
        token = _fresh_token()
        started = datetime.now().astimezone()
        deadline = started + timedelta(seconds=budget)
        body = _answering_after_a_wait(token, deadline, started + timedelta(hours=1))
        task = _start(backend, token, body, f"again{budget}")
        if not _parked(backend, task):
            continue
        _wait_past(deadline, margin=PAST_THE_SKEW)
        backend.emit_event(task, f"shared:{token}", {"ok": True})
        return _answer(backend, task)
    raise AssertionError(f"the first wait never parked, on any of {PARK_BUDGETS}s")


def _a_later_wait_on_the_same_claim_without_one(backend, token, now):
    """History 7 with the emit removed, which is the whole difference.

    Nothing is ever emitted and the second deadline is an hour out, so the second ask has neither
    ending available to it and parks. An engine that reads the FIRST ask's wake here answers a
    wait nobody could have answered."""
    for budget in PARK_BUDGETS:
        token = _fresh_token()
        started = datetime.now().astimezone()
        deadline = started + timedelta(seconds=budget)
        body = _answering_after_a_wait(token, deadline, started + timedelta(hours=1))
        task = _start(backend, token, body, f"nofab{budget}")
        if not _parked(backend, task):
            continue
        _wait_past(deadline, margin=PAST_THE_SKEW)
        answered = _answer(backend, task)
        # WHICH ask is parked, not merely that something is. The two asks park identically, so
        # without this the row reads the same whether the run got past the first one or not.
        settled = backend.checkpoint_states(task)
        first = compose_key(t"event;shared:{Segment(token)}")
        assert settled.get(first) == ["expired"], (
            f"the run never got past its first ask: {sorted(k.stored() for k in settled)}"
        )
        return answered
    raise AssertionError(f"the first wait never parked, on any of {PARK_BUDGETS}s")


def _a_later_wait_across_a_sleep(backend, token, now):
    for budget in PARK_BUDGETS:
        token = _fresh_token()
        started = datetime.now().astimezone()
        deadline = started + timedelta(seconds=budget)
        wake_at = deadline + timedelta(seconds=budget + PAST_THE_SKEW + 4)
        body = _answering_after_a_sleep(token, deadline, wake_at, started + timedelta(hours=1))
        task = _start(backend, token, body, f"slept{budget}")
        if not _parked(backend, task):
            continue
        _wait_past(deadline, margin=PAST_THE_SKEW)
        backend.run_until_result(task)  # the clock ends the first wait, and the sleep begins
        # The sleep has to still be AHEAD here, or the second ask is decided on the same claim
        # and this history is its sibling wearing a longer name. A slow drain is exactly what the
        # ladder absorbs, so this widens rather than raising.
        if time.time() >= wake_at.timestamp():
            continue
        backend.emit_event(task, f"shared:{token}", {"ok": True})
        _wait_past(wake_at, margin=0.2)
        return _answer(backend, task)
    raise AssertionError(f"the first wait never parked, on any of {PARK_BUDGETS}s")


BOUNDED_WAIT_TABLE: tuple[tuple[str, str, Any], ...] = (
    ("it had already settled", "expired", _settled_keeps_its_answer),
    ("the clock reached a parked wait", "expired", _the_clock_reached_a_parked_wait),
    ("its event reached a parked wait", "arrived", _its_event_reached_a_parked_wait),
    ("it never parked, and its event is stored", "arrived", _never_parked_with_its_event),
    (
        "it never parked, its deadline is behind, and its event is stored",
        "arrived",
        _never_parked_past_its_deadline_with_its_event,
    ),
    ("it never parked and its deadline is behind", "expired", _never_parked_past_its_deadline),
    (
        "a later wait on the same claim, its event stored",
        "arrived",
        _a_later_wait_on_the_same_claim,
    ),
    ("a later wait across a sleep, its event stored", "arrived", _a_later_wait_across_a_sleep),
    (
        "a later wait on the same claim, no event ever emitted",
        "parks",
        _a_later_wait_on_the_same_claim_without_one,
    ),
)
"""Every history in `wiki/concepts/bounded-wait.md`'s decision table, and the one answer each has.

A history rather than a state, because history is what the engines can disagree about: the same
event and deadline answer differently depending on whether the task parked, what woke it, and
whether a claim intervened. `parks` is an answer like the other two, so a row can assert that
neither ending was reached."""


def _reached(backend, run) -> str:
    """The answer a history reached, or the reason it could not be measured.

    A history that checks it measured the history it names raises when it did not, and a raise
    ends the whole table at the first offender. Reported as an answer instead, so the row joins
    the others and the run below still reads every remaining cell.

    Every `Exception`, not an `AssertionError` alone: a driver error is exactly as good a reason
    to keep reading the other cells, and narrowing the catch put the hiding back for every
    history that fails some other way. `BaseException` keeps propagating, so a `KeyboardInterrupt`
    still stops the run."""
    try:
        return run(backend, _fresh_token(), datetime.now().astimezone())
    except Exception as why:
        return f"unmeasurable: {type(why).__name__}: {why}"


def test_a_history_that_raises_joins_the_offenders_rather_than_ending_the_table(backend):
    """The driver's promise, asked directly: one bad cell must not hide the cells after it."""

    def raises(_backend, _token, _now) -> str:
        raise ValueError("this history could not run")

    assert "unmeasurable: ValueError" in _reached(backend, raises)


def test_every_history_in_the_bounded_wait_table_answers_alike_on_both_engines(backend):
    """The whole table, as ONE assertion, so a cell is a row here rather than a gap.

    Each history is run and its answer collected; the offenders are what is asserted empty. A
    per-history test reports the first divergence and hides the rest, and a defect in a history
    nobody listed reports nothing at all. The second is why this exists: four review rounds each
    found one of these, one cell at a time.
    """
    offenders = [
        (history, expected, answered)
        for history, expected, run in BOUNDED_WAIT_TABLE
        if (answered := _reached(backend, run)) != expected
    ]
    assert offenders == [], f"{len(offenders)} of {len(BOUNDED_WAIT_TABLE)} histories: {offenders}"


def test_an_event_OUTLIVES_its_task_on_both_engines(backend):
    """The other face of the same key, and the one that bites an operator rather than a workflow.

    An event emitted while one task was waiting is found by a LATER, unrelated task's await: the
    queue keeps it and it is immutable, on both engines. So a test that reuses an event name is
    contaminated by its own earlier run: its tasks can complete before anything is emitted. That
    is what `_fresh_token` exists for."""
    token = _fresh_token()
    backend.register(f"outlive-{token}", _awaiting_shared_name(token), None, Fault(), ())
    done = backend.spawn(f"outlive-{token}", "run-first")
    backend.emit_event(done, f"shared:{token}", {"ok": True})
    assert backend.run_until_result(done).state == "completed"

    later = backend.spawn(f"outlive-{token}", "run-later")  # never emitted to
    state = backend.run_until_result(later).state
    assert state == "completed", state


class _Charges:
    """What a gated workflow actually charged — the observable the isolation test needs."""

    def __init__(self) -> None:
        self.amounts: list[int] = []

    def run(self, op):
        self.amounts.append(op.args.get("amount"))
        return {"ok": True}

    def run_metered(self, op):
        return self.run(op), Usage()


def _one_gated_charge(_run_id: str):
    return (yield from call_tool("charge_card", {"amount": 5}, dict))


def _escalating_gate(gate: str, run_id: str):
    """A `govern` gate that escalates every charge — one park, one approval."""

    def policy(op, state):
        if not (isinstance(op, Step) and getattr(op.op, "name", None) == "charge_card"):
            return Proceed()
        return as_policy([lambda _op: Escalate()])(op, state)

    return govern(policy, gate=gate, run_id=run_id)


def test_two_runs_under_distinct_gates_are_isolated_on_both_engines(backend):
    """Two runs are isolated by their NAMES, on both engines.

    Two tasks sharing one `run_id` are isolated on the embedded engine only because its events
    table is keyed `(task_id, name)`; on the deployed broadcast engine both tasks compose one park
    name and one approval settles both, as reproduced on real Absurd.

    The property that survives parity is this one, and it lives in the NAME rather than in the
    latch: two runs carrying DISTINCT run coordinates ask distinct questions, so an answer to one
    is not an answer to the other. Asserted on both engines because it is the positive statement
    of what broadcast event delivery relies on.
    """
    token = _fresh_token()
    parks, tasks, domains = {}, {}, {}
    for run in (f"{token}a", f"{token}b"):
        gate = f"spend-{token}"
        domains[run] = _Charges()
        backend.register(
            f"gated-{run}",
            _one_gated_charge,
            domains[run],
            Fault(),
            (_escalating_gate(gate, run),),
        )
        tasks[run] = backend.spawn(f"gated-{run}", run)
        parks[run] = GateState(
            run_id=run, gate=gate, op_key=step_key("tool:charge_card").stored(), occurrence=0
        ).park_name.stored()

    first, second = (f"{token}a", f"{token}b")
    assert parks[first] != parks[second], "the run coordinate did not reach the park name"
    for run in (first, second):
        assert backend.run_until_result(tasks[run]).state in ("waiting", "sleeping")

    approve = {"answers": {"permission": {"decision": "approve"}}}
    backend.emit_event(tasks[first], parks[first], approve)
    assert backend.run_until_result(tasks[first]).state == "completed"
    assert domains[first].amounts == [5]
    assert domains[second].amounts == [], (
        f"run {second} charged on run {first}'s approval: {domains[second].amounts}"
    )

    backend.emit_event(tasks[second], parks[second], approve)  # its OWN question
    assert backend.run_until_result(tasks[second]).state == "completed"
    assert domains[second].amounts == [5]


# --- answering a park you READ, rather than a name you composed --------------------------------
#
# `answer` exists because a wake registration carries coordinates the caller cannot see: the
# enclosing frames, and `Key.occurrence`'s `#N` for a `Scope.SETTLEMENT` namespace at its second
# ask. The second test below is the one that would fail for an emitter composing the name — the
# park it must settle is `depth-grant:…#2`, and `qualified_event_name` has no way to say that.


def _twice_on_one_grant(token: str):
    """A workflow that asks the SAME settlement question twice. The second ask is a different
    op-occurrence, so the walk qualifies it — which is exactly what one answer must not settle."""

    def wf(_run_id: str):
        name = depth_grant_name(token, generation=0, depth=1)
        first = yield from await_event(name, Grant)
        second = yield from await_event(name, Grant)
        return {"first": first.add_depth, "second": second.add_depth}

    return wf


def test_answer_settles_a_park_read_off_the_engine(backend):
    """The round trip, on both engines: park, read the registration, answer it, RESUME.

    The caller composes nothing — it hands back the `ParkedTask` the reader gave it.

    **The resume is the assertion, and this test did not have it.** It stopped at "the emit
    returned the name it was given", which `answer` does unconditionally — it is the last line of
    the function and no engine has to agree. A verb named `answer` that delivered nothing would
    have passed. So the park is now read back and found GONE, and the value the workflow received
    is checked: those are the two facts that say the answer landed.
    """
    token = _fresh_token()
    backend.register(f"grant-{token}", _twice_on_one_grant(token), None, Fault(), ())
    task_id = backend.spawn(f"grant-{token}", "run-answer")
    backend.run_until_result(task_id)

    (park,) = backend.parked(task_id)
    assert park.wake_event == depth_grant_name(token, generation=0, depth=1).stored()
    assert answer(backend.app, park, {"add_depth": 1}) == park.wake_event

    # Answering marks the task claimable; it does not resume it. Nothing polls, so the drain is
    # the caller's — which is the sentence the verb's docstring ends on, asserted here.
    backend.run_until_result(task_id)
    (moved,) = backend.parked(task_id)
    assert moved.wake_event != park.wake_event, (
        "the run is still parked on the name that was answered — the emit did not land"
    )


def test_answer_settles_the_SECOND_ask_without_the_caller_knowing_about_occurrence(backend):
    """The case the composed path cannot express.

    Two asks on one settlement name are two op-occurrences, so `placing` qualifies the second with
    `#2`. An emitter composing the name would address the bare form and settle nothing — and
    `qualified_event_name`, the documented anchor for completing a park's name, carries frames but
    has no occurrence parameter at all. Reading the registration needs to know none of that, which
    is the whole argument for the verb: the caller passes back what it was handed, twice."""
    token = _fresh_token()
    bare = depth_grant_name(token, generation=0, depth=1)
    backend.register(f"grant2-{token}", _twice_on_one_grant(token), None, Fault(), ())
    task_id = backend.spawn(f"grant2-{token}", "run-answer-twice")
    backend.run_until_result(task_id)

    (first,) = backend.parked(task_id)
    assert first.wake_event == bare.stored()
    answer(backend.app, first, {"add_depth": 1})
    backend.run_until_result(task_id)

    # The SECOND park is a different name, and the caller never had to work that out.
    (second,) = backend.parked(task_id)
    assert second.wake_event == bare.occurrence(2).stored()
    assert second.wake_event != first.wake_event, "one answer settled both asks"
    answer(backend.app, second, {"add_depth": 2})

    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == {"first": 1, "second": 2}


# --- the placed-writer collision: two placed ops, one `event_id` -------------------------------
#
# Reproduced in a gather, sequentially, and inside a fork child, and again with the fork
# sealing over the corrupted lineage. The collision is REFUSED now, so these are plain tests;
# they were strict xfails while the defect stood, and are kept because a refusal that stops
# firing is exactly as silent as the defect it replaced.
#
# Both engines, deliberately. The defect reproduced on Absurd/Postgres as well as SQLite, which
# is why both run: a green SQLite pass is not a port.


def test_two_gather_branches_appending_one_event_id_are_refused(backend):
    """The easiest way to write the collision by accident: a fan-out whose branches run the same
    source, so they author the same id. A programming error, so the task fails on its first
    attempt, reported as the collision itself rather than the group the gather raised."""
    domain = CountingDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"collide-{run_id}"
    backend.register(name, colliding_gather_wf, domain, Fault(), ())
    task = backend.spawn(name, run_id)
    snap = backend.run_until_result(task)

    assert snap is not None
    assert snap.state == "failed", (
        f"the run reported {snap.state!r} having silently dropped a canonical row"
    )
    assert backend.failure_kind(snap) == "PlacedWriterCollision"
    assert backend.task_attempts(task) == 1


def test_two_sequential_appends_of_one_event_id_are_refused(backend):
    """The same defect with NO gather anywhere — which is why the family is named for the WRITER,
    not for the combinator that makes it easy to hit."""
    domain = CountingDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"seq-{run_id}"
    backend.register(name, sequential_collision_wf, domain, Fault(), ())
    task = backend.spawn(name, run_id)
    snap = backend.run_until_result(task)

    assert snap is not None
    assert snap.state == "failed", (
        f"the run reported {snap.state!r} having silently dropped a canonical row"
    )
    assert backend.failure_kind(snap) == "PlacedWriterCollision"
    assert backend.task_attempts(task) == 1


# The two arms the refusal must NOT take. Written after a mutation round showed the whole suite
# — all 2564 tests — passing with each of them deleted: the refusal was pinned and its
# PERMISSIONS were not, which is the more dangerous half. A refusal that over-fires breaks
# cross-generation idempotency and poisons crash recovery, and both fail closed and silently.


def _writer(task: str, placement: str) -> Writer:
    return Writer(task=task, placement=Key.parse(placement))


def test_the_same_event_id_from_a_DIFFERENT_task_is_allowed_silently(backend):
    """Cross-generation idempotency is the FEATURE (`handlers/durable.py`): one message triaged in
    generation 0 and again in 3 is ONE row, and generations share a run id — so only the task
    tells them apart. The second append must be a silent no-op, not a refusal."""
    run_id = f"r-{uuid4().hex[:8]}"
    event_id = f"{run_id}:done"
    backend.ledger_append(run_id, event_id, "gen0", writer=_writer("task-0", "ledger;x:1"))
    # A different task, a different placement, the same id — the shape the feature is made of.
    backend.ledger_append(run_id, event_id, "gen3", writer=_writer("task-3", "ledger;x:9"))
    assert backend.ledger_kinds(run_id) == ["gen0"], (
        "the first write wins and the second is a no-op"
    )


def test_the_same_placement_re_appending_is_allowed(backend):
    """The crash window. An attempt can die between the store write and the checkpoint commit, so
    the retry re-executes THIS placement and re-writes THIS row. Refusing here would make a
    surviving ledger row poison its own run on every subsequent attempt — a durable substrate
    turning its own crash recovery into a permanent failure."""
    run_id = f"r-{uuid4().hex[:8]}"
    event_id = f"{run_id}:done"
    writer = _writer("task-0", "gather:0,0;ledger;x:1")
    backend.ledger_append(run_id, event_id, "first", writer=writer)
    backend.ledger_append(run_id, event_id, "retry", writer=writer)  # byte-identical writer
    assert backend.ledger_kinds(run_id) == ["first"]


def test_an_unknown_writer_is_allowed_on_either_side(backend):
    """A fork's genesis and seal bypass `ctx.step` and re-append live on every attempt, so they
    carry no placement; rows written before the columns existed carry none either. Unknown must
    read as ALLOW in both directions — as the holder, and as the appender."""
    run_id = f"r-{uuid4().hex[:8]}"
    known, unknown = f"{run_id}:a", f"{run_id}:b"
    backend.ledger_append(run_id, known, "held-unknown")  # holder has no writer
    backend.ledger_append(run_id, known, "second", writer=_writer("t", "ledger;x:1"))
    backend.ledger_append(run_id, unknown, "held-known", writer=_writer("t", "ledger;x:1"))
    backend.ledger_append(run_id, unknown, "second")  # appender has no writer
    assert sorted(backend.ledger_kinds(run_id)) == ["held-known", "held-unknown"]


def test_a_different_placement_in_the_SAME_task_is_refused(backend):
    """The one arm that refuses, at the store rather than through a workflow — so the decision
    table is pinned cell by cell and not only through the shape that happens to reach it."""
    run_id = f"r-{uuid4().hex[:8]}"
    event_id = f"{run_id}:done"
    backend.ledger_append(run_id, event_id, "first", writer=_writer("t", "gather:0,0;ledger;x:1"))
    with pytest.raises(PlacedWriterCollision, match="two placed ops in task"):
        backend.ledger_append(
            run_id, event_id, "second", writer=_writer("t", "gather:0,1;ledger;x:1")
        )


# ── the coding machine (`effective.coding`) ──────────────────────────────────
#
# It had only ever run under `RecordingHandler`/`ReplayHandler`. A hand-driven pass on both
# engines said it worked; these two cases are that pass made repeatable, which is the difference
# between a measurement and a gate.

MACHINE_PATH = ["test", "draft", "finalize", "review"]
MACHINE_CALLS = [
    "run_suite",
    "read_file",
    "apply_fix",
    "run_suite",
    "run_suite",
    "run_suite",
    "run_suite",
]
MACHINE_OPS = 10
"""Ctx-op touches on a clean run — 6 worker steps + the postamble's artifact, predicate and two
ledger appends. Discovered from the reference case below, not assumed; it is the bound the crash
sweep loops over, and `test_the_coding_machine_runs_the_same_on_both_engines` re-derives it every
run so a walk that grows an op cannot leave the sweep quietly covering a prefix."""


def _run_machine(backend, crash_at=None, domain=None):
    """Spawn one coding-machine run on `backend`. Fresh run id and task name per call — the
    Absurd queue and the `ledger` table are shared across cases."""
    domain = domain if domain is not None else CodingSuiteDomain()
    fault = Fault(crash_at)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"mach-{run_id}"
    backend.register(name, coding_machine_wf, domain, fault, ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    return snap, domain, fault, run_id, task_id


def test_the_coding_machine_runs_the_same_on_both_engines(backend):
    """The reference outcome: the composition, on a durable engine.

    The PATH is asserted, not only the terminal outcome. With a referee mutated to return GREEN
    unconditionally, the machine still detours through FINALIZE and parks, so an outcome cannot
    tell a working referee from a broken one, and a trajectory can.

    The artifact id is the sharpest line here. It is content-addressed over the tree the machine
    committed, so it is identical across engines only if both re-derived the same workspace from
    recorded results — the property `Evidence.tree` exists for."""
    snap, domain, fault, run_id, task_id = _run_machine(backend)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result["path"] == MACHINE_PATH, snap.result
    assert snap.result["stopped"] == "machine-finished"
    assert snap.result["passed"] is True
    assert snap.result["artifact_id"] == "application/json,sha256-d47e76c79bd609d2"
    assert domain.calls == MACHINE_CALLS, domain.calls
    assert fault.count == MACHINE_OPS, "the walk changed shape — re-derive the crash sweep's bound"
    assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-finished"]

    # The KEYS, because everything above is checkpoint-blind: with the `d:` frame deleted, every
    # assertion above still passes, across the full k=1..10 crash sweep, on both engines.
    # An outcome cannot tell a walk that framed its ops from one that did not; the tape can.
    assert backend.checkpoint_keys(task_id) == [
        "d:0;state:test;step;tool:run_suite",
        "d:1;state:draft;step;tool:read_file",
        "d:1;state:draft;step;tool:apply_fix",
        "d:1;state:draft;step;tool:run_suite",
        "d:2;state:finalize;step;tool:run_suite",
        "d:3;state:review;step;tool:run_suite",
        "artifact:application/json,sha256-d47e76c79bd609d2",
        "step;tool:run_suite",  # the postamble's own predicate — outside every visit frame
        # COMPOSED, not spelled: the two ledger ids go through the composer `machine.py` uses and
        # wear the `ledger;` arm `op_key` puts on them, so a change to either travels here rather
        # than leaving a stale literal behind. (`Key` has no `__str__` — an f-string of one yields
        # the repr, which is PEP 750's protection working, so `.stored()` is the explicit flatten.)
        compose_key(t"ledger;machine:{Segment(run_id)};commit").stored(),
        compose_key(t"ledger;machine:{Segment(run_id)}").stored(),
    ]


DRAFT_TOOLS = ("read_file", "apply_fix", "run_suite")


def _visit_step(visit: int | None, state: str, tool: str) -> str:
    """One of the machine's tool checkpoints, ASKED FOR rather than spelled.

    Every piece goes through the minter that produces it — `step_key` for the `step;` arm,
    `frame_path` for each scope frame — so this is the key as it is CANONICALLY supposed to be
    stored, and a change to the arm or to the frame delimiter travels here. Splitting the stored
    bytes on `;` instead would be the nominal move where a structural one belongs: it would keep
    passing the day either spelling changed. `visit=None` composes the same key WITHOUT the visit
    frame, which is what makes the counter's contribution assertable rather than assumed."""
    # lint: terminal-hole — `visit` is an `int` here, narrowed by the guard, so it needs no wrapper
    path = "" if visit is None else frame_path("", compose_key(t"d:{visit}"))
    path = frame_path(path, compose_key(t"state:{Segment(state)}"))
    # `direct_tool_key` is PRODUCTION's minter for a tool called outside a loop, which is what
    # this walk's hand-written workers do; `.stored()` is the explicit flatten at the seam where
    # `step_key` requires a `str`, the layering rule rather than a dodge. `react.tool_key` is
    # the in-loop minter and carries a turn sub-term, so it is the wrong one here.
    return step_key(direct_tool_key(tool).stored()).prefixed(path).stored()


def test_a_re_entered_state_stays_injective_on_both_engines(backend):
    """The case `d:` exists for: a re-entered state, durably.

    `MACHINE_PATH` is four DISTINCT states, so the reference run above never re-enters a `state:`
    frame and its keys would stay disjoint on the state atom alone. `StubbornSuiteDomain` takes
    the `route_draft` self-edge — `STILL_RED -> Advance(State.DRAFT)` — so DRAFT runs twice and
    the two visits mint the same three tool ops under the same state atom.

    Asserted as the INVARIANT rather than as the byte strings: the tape is injective, and the six
    DRAFT ops separate into two visits of three. Delete `d:` and each of those three pairs
    collapses: a repeated scope frame gets no occurrence suffix on the in-memory tape, which is
    exactly what `d:{n}`'s declared `Index` role says the counter is for."""
    domain = StubbornSuiteDomain()
    snap, domain, _fault, _run_id, task_id = _run_machine(backend, domain=domain)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result["path"] == ["test", "draft", "draft", "finalize", "review"], snap.result
    assert snap.result["stopped"] == "machine-finished"

    expected = [_visit_step(v, "draft", tool) for v in (1, 2) for tool in DRAFT_TOOLS]
    keys = backend.checkpoint_keys(task_id)
    assert [k for k in keys if k in set(expected)] == expected
    # And the visit ordinal is what separates them: compose the same six WITHOUT that frame and
    # six distinct keys become three. Stated over the COMPOSITION rather than as a set-size check
    # on the tape, because a set-size check here would prove nothing: with `d:` deleted the
    # durable tape stays 13-of-13 distinct, since both engines apply the SDK's `name#N`
    # duplicate-step rule. The list above reddens.
    assert len(set(expected)) == 6
    assert len({_visit_step(None, "draft", tool) for tool in DRAFT_TOOLS}) == 3


def test_the_coding_machine_survives_a_crash_at_every_op(backend):
    """Crash-at-every-op over the whole walk, POSTAMBLE INCLUDED — the commitment tail is the
    durability-critical part, and it is where the two engines differ most (an Absurd park writes a
    NULL-payload marker row SQLite has no counterpart to).

    `domain.calls` is the exactly-once proof and it is not a tautology: the domain mutates on
    `apply_fix`, so a resume that wrongly re-ran a committed step would show the edit twice."""
    for k in range(1, MACHINE_OPS + 1):
        snap, domain, fault, run_id, _ = _run_machine(backend, crash_at=k)
        assert fault.armed is False, f"k={k}: the fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result["path"] == MACHINE_PATH, f"k={k}: {snap.result['path']}"
        assert snap.result["stopped"] == "machine-finished", f"k={k}"
        assert snap.result["artifact_id"] == "application/json,sha256-d47e76c79bd609d2", (
            f"k={k}: the committed tree re-derived differently across the crash"
        )
        assert domain.calls == MACHINE_CALLS, f"k={k}: {domain.calls}"
        assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-finished"], f"k={k}"


# ── the ruling machine: a judge that YIELDS, on both engines ─────────────────
#
# The coding-machine cases above are green over the EMPTY SET for this property. `machine_specs()`
# wires `mechanical_judges` + `_judge_review`, every one of them `return X; yield`, so they mint no
# judge-side op and read no `incoming` — an assertion about where a judge's ops land holds
# vacuously when there are none. `tests/test_machine_fuse_or_split.py` witnesses a `Judge`'s own
# `Ctx` on one engine, with no crash injection; these three cases are that witness on both
# engines and under the sweep.

RULING_PATH = ["work", "review", "rule", "work", "review", "rule"]
RULING_VERDICTS = ["done", "gathered", "breach", "done", "gathered", "approved"]
RULING_KINDS = ["noted", "ruled", "noted", "ruled", "machine-committed", "machine-finished"]
RULING_ARTIFACT = "application/json,sha256-97339bb2ab6b6b69"
RULING_FILES = ["ruling.txt"]
RULING_GATE_CALLS = 5
"""Once per WORK and once per REVIEW, plus the postamble's predicate — and never in RULE, which
reads its predecessor's measurement out of `ctx.incoming` instead of taking it again."""

RULING_SAW = [
    "gathered at visit 1 after clean at visit 0",
    "gathered at visit 4 after clean at visit 3",
]
"""What RULE read out of `ctx.incoming.summary` at each of its two visits, recorded handler-side.

Both carries in one string. `gathered at visit N` is REVIEW's own coordinate reaching RULE across
a state boundary, the SPLIT half; `after clean at visit N-1` is what REVIEW read out of WORK's
report, which only exists because `fuse` lifted a worker's `Evidence` into one, the FUSED half.
Without the second clause, deleting `measured` from `Report.of` would leave every case here green.

`incoming` is not checkpointed: the trampoline re-derives it from the previous visit's `Report`,
itself re-derived from replayed op results. So a resume that rebuilt it differently, or not at
all, changes these strings, and `RulingDeployment` refuses an empty one at the tool boundary.

**Scope.** These are handler-side lists, so they record the ops that ran IN THIS PROCESS. That is
exactly right for the in-process crash sweep below (the same shape as the coding machine's
`domain.calls`), and it is not a claim about a fresh worker, which would have executed fewer ops
and recorded a shorter list. What IS process-independent is the CONTENT: every stamp is a function
of the visit the workflow was on, so it re-derives identically anywhere. A live counter in this
object would not: a cross-process resume renumbers it and reddens the assertion with nothing
wrong."""

RULING_CARRIED = ["clean at visit 1", "clean at visit 4"]
"""What RULE read out of `ctx.incoming.measured` at each of its two visits.

`measured` is the field the RECORD parameter types, so this is what fails if a `Ctx` carries a
`Report` whose typed half was dropped. The gate stamps each measurement with the VISIT it was
taken at, which is what makes this discriminating rather than a shape check: RULE at visit 2 must
see what REVIEW measured at visit 1, and at visit 5 what it measured at visit 4. A carried record
rebuilt from the wrong visit agrees on the type and disagrees here.

See `RULING_SAW` for the scope these two share, and for why the stamp is a function of the visit
rather than of a call counter."""

RULING_OPS = 14
"""Ctx-op touches on a clean run: WORK 2 (gate + its judge's row) + REVIEW 1 + RULE 2, twice around
the BREACH back edge, + 4 postamble. Discovered from the reference case below, not assumed; it is
the bound the crash sweep loops over, and the reference case re-derives it every run so a walk that
grows an op cannot leave the sweep quietly covering a prefix."""


def _run_ruling(backend, crash_at=None):
    """Spawn one ruling-machine run on `backend`. Fresh run id and task name per call, for the
    reason `_run_machine` gives — the Absurd queue and the `ledger` table are shared."""
    deployment = RulingDeployment()
    fault = Fault(crash_at)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"rule-{run_id}"
    backend.register(name, ruling_machine_wf, deployment, fault, ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    return snap, deployment, fault, run_id, task_id


def _judged_rows(run_id: str) -> list[Key]:
    """The four ids the two judgements author, in walk order — ASKED FOR, not spelled.

    Each goes through the minter the machine itself calls, from a `Ctx` carrying the coordinates
    that judgement had — so this list tracks a change in the machine's COORDINATES.

    **It is deliberately blind to a change in the MINTER**: both sides go through `note_row_id`,
    so renaming its tag moves them together and every case here stays green. The spelled tape in
    `test_a_judges_ledger_op_is_framed_by_its_own_state_on_both_engines` is the half that is not
    blind, and it is spelled for exactly that reason. Spell what the FIXTURE mints, compose what
    PRODUCTION mints."""

    def at(state: RulingState, visit: int) -> Ctx[RulingState]:
        return Ctx(run_id=Segment(run_id), goal=RULING_GOAL, state=state, visit=visit)

    return [
        note_row_id(at(RulingState.WORK, 0)),
        ruling_row_id(at(RulingState.RULE, 2)),
        note_row_id(at(RulingState.WORK, 3)),
        ruling_row_id(at(RulingState.RULE, 5)),
    ]


def _ruling_ledger_ids(run_id: str, reported: dict[str, Any]) -> list[str]:
    """Every row the run puts on the canonical record: the judgements', then the postamble's.

    The postamble's two come from the run's own account. Composing them here made this a second
    speller of an identity the trampoline mints, and one that omitted the generation coordinate,
    so the two agreed only at generation 0. The BYTES are pinned as literals in
    `test_machine_in_machine`, which fixes its run id and so can spell them."""
    return [
        *(row.stored() for row in _judged_rows(run_id)),
        reported["commit_id"],
        reported["outcome_id"],
    ]


def _visit_ledger(visit: int, state: str, event_id: Key) -> str:
    """A judge's ledger checkpoint, composed the way `_visit_step` composes a tool's.

    `op_key` is production's minter for the `ledger;` arm, so the arm and its `domain=address`
    treatment of the id travel here; `frame_path` supplies the two scope frames the trampoline
    wraps a visit in. Nothing about this key is spelled."""
    # lint: terminal-hole -- `visit` is an `int`, so it needs no wrapper
    path = frame_path("", compose_key(t"d:{visit}"))
    path = frame_path(path, compose_key(t"state:{Segment(state)}"))
    op = AppendLedgerRow(row=LedgerRow(event_id=event_id, kind="unread-here"))
    return op_key(op).prefixed(path).stored()


def test_a_yielding_judge_appends_the_same_rows_on_both_engines(backend):
    """The reference outcome for a walk whose judgements mint ops — the first on a durable engine.

    The two `note:` rows are the case a `Ctx`-bearing judge exists for. WORK is visited twice, and
    before a judge had coordinates the only ones in reach were the run id and a literal, so both
    visits composed ONE `event_id`, the second was refused, and the run died taking the postamble's
    rows with it. The kinds cannot see any of that — both rows are `noted` — so it takes the ids.

    What the ids here CANNOT see is a change to the minter itself, since they go through it: that
    is the next case's spelled tape, and the distinction is written down there."""
    snap, deployment, fault, run_id, _ = _run_ruling(backend)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result["path"] == RULING_PATH, snap.result
    assert snap.result["verdicts"] == RULING_VERDICTS, snap.result
    assert snap.result["stopped"] == "machine-finished"
    # The POSTAMBLE's two answers. Without them, every gate call can exit 1 and every test stays
    # green: `passed` goes into the projection and nothing else reads it. `files` is non-empty
    # only because REVIEW returns a tree; without one the artifact is the digest of `{}`, and
    # "the committed tree re-derived identically" is a claim over the empty set.
    assert snap.result["passed"] is True
    assert snap.result["files"] == RULING_FILES, snap.result["files"]
    assert snap.result["artifact_id"] == RULING_ARTIFACT
    assert fault.count == RULING_OPS, "the walk changed shape — re-derive the crash sweep's bound"
    assert deployment.gate_calls == RULING_GATE_CALLS, deployment.gate_calls
    assert deployment.saw == RULING_SAW, deployment.saw
    assert deployment.carried_outputs == RULING_CARRIED, deployment.carried_outputs
    assert deployment.rulings == [], "both rulings were consumed, exactly once each"

    # BOTH bookkeepers. `checkpoint_keys` (the next case) says the op was recorded; this says the
    # row reached the append-only record, which is the one that is canonical.
    assert backend.ledger_kinds(run_id) == RULING_KINDS
    assert backend.ledger_ids(run_id) == _ruling_ledger_ids(run_id, snap.result)


def test_a_judges_ledger_op_is_framed_by_its_own_state_on_both_engines(backend):
    """Where a judgement's ops LAND, read off the durable tape by the substrate's own readers.

    `appending_states` and `canonical_violations` are `StateSpec.canonical`'s only readers, and
    nothing has ever run them against a tape a durable engine wrote. They answer two different
    questions: which state each canonical row was minted under, and whether any state reached the
    ledger without declaring that it would. This run's postamble is minted outside every visit
    frame, so its two rows appear under no state: they belong to the run. A run nested in a state
    sits inside that state's frames instead, and `test_machine_in_machine` is where that lands."""
    snap, _deployment, _fault, run_id, task_id = _run_ruling(backend)
    assert snap is not None
    assert snap.state == "completed", snap
    keys = backend.checkpoint_keys(task_id)
    note_0, ruling_2, note_3, ruling_5 = _judged_rows(run_id)

    # THE WHOLE TAPE, and mostly SPELLED — the hybrid the coding machine above already uses.
    #
    # Which half to spell is not a style call, it is about who MINTS the id. The coding machine
    # composes its two postamble ids because `machine.py` mints them: a production change should
    # travel into the assertion rather than leave a stale literal. These four `note:`/`ruling:` ids
    # are minted by the FIXTURE, so composing them would have the fixture assert against itself —
    # and it did: renaming the minter's tag `note:` -> `annot:` left all six of these cases green,
    # because both sides moved together. Only the SQLite witness, which spells its bytes, reddened.
    # So: spell what the fixture mints, compose what production mints.
    assert backend.checkpoint_keys(task_id) == [
        "d:0;state:work;step;tool:gate",
        f"d:0;state:work;ledger;note:{run_id},work,0",
        "d:1;state:review;step;tool:gate",
        "d:2;state:rule;step;tool:ruling",
        f"d:2;state:rule;ledger;ruling:{run_id},rule,2",
        "d:3;state:work;step;tool:gate",
        f"d:3;state:work;ledger;note:{run_id},work,3",
        "d:4;state:review;step;tool:gate",
        "d:5;state:rule;step;tool:ruling",
        f"d:5;state:rule;ledger;ruling:{run_id},rule,5",
        f"artifact:{RULING_ARTIFACT}",
        "step;tool:gate",  # the postamble's predicate — outside every visit frame
        compose_key(t"ledger;machine:{Segment(run_id)};commit").stored(),
        compose_key(t"ledger;machine:{Segment(run_id)}").stored(),
    ]

    appended = appending_states(RulingState, keys)
    assert appended, "no judgement minted a ledger op — the rest of this test is vacuous"
    assert appended == {
        RulingState.WORK: [
            _visit_ledger(0, "work", note_0),
            _visit_ledger(3, "work", note_3),
        ],
        RulingState.RULE: [
            _visit_ledger(2, "rule", ruling_2),
            _visit_ledger(5, "rule", ruling_5),
        ],
    }, appended
    assert canonical_violations(RulingState, ruling_specs(), keys) == {}


def test_the_ruling_machine_survives_a_crash_at_every_op(backend):
    """Crash-at-every-op over a walk whose JUDGEMENTS mint ops, postamble included.

    `deployment.saw` is the exactly-once proof. Its CONTENT is what carries it: each entry is what
    `ctx.incoming.summary` held, and `incoming` is re-derived across the crash rather than
    checkpointed, so a resume that rebuilt the carried `Report` differently shows up here.

    `carried_outputs` is the same question asked of the TYPED half, and it is the sharper one: the
    gate stamps every measurement with its visit, so an entry names the exact one it came from. A
    resume that rebuilt the carried record from the wrong visit still type-checks and reads
    `visit 1` where `visit 4` belongs.

    Two limits. The bound this loops over is `RULING_OPS`, re-derived by the reference case ABOVE
    rather than here: a walk that
    grew an op would redden there while this quietly covered a prefix, so the two travel together.
    And a duplicate `ruling` execution is caught by `saw`'s CONTENT and by the path, not by the pop
    list running dry: duplicating the FIRST of the two pops `approved` early, and the walk finishes
    in three turns with `len(saw) == 2`. Only duplicating the last one raises."""
    for k in range(1, RULING_OPS + 1):
        snap, deployment, fault, run_id, _ = _run_ruling(backend, crash_at=k)
        assert fault.armed is False, f"k={k}: the fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result["path"] == RULING_PATH, f"k={k}: {snap.result['path']}"
        assert snap.result["verdicts"] == RULING_VERDICTS, f"k={k}: {snap.result['verdicts']}"
        assert snap.result["artifact_id"] == RULING_ARTIFACT, (
            f"k={k}: the committed tree re-derived differently across the crash"
        )
        assert deployment.saw == RULING_SAW, f"k={k}: {deployment.saw}"
        assert deployment.carried_outputs == RULING_CARRIED, f"k={k}: {deployment.carried_outputs}"
        assert deployment.gate_calls == RULING_GATE_CALLS, f"k={k}: {deployment.gate_calls}"
        assert backend.ledger_kinds(run_id) == RULING_KINDS, f"k={k}"
        assert backend.ledger_ids(run_id) == _ruling_ledger_ids(run_id, snap.result), f"k={k}"


# ── the other exit path: a machine that EXHAUSTS still commits ───────────────

PARKING_PATH = ["work", "review", "rule", "work", "review"]
PARKING_KINDS = ["noted", "ruled", "noted", "machine-committed", "machine-parked"]
PARKING_ARTIFACT = "application/json,sha256-aa3c7c10996e12f3"
PARKING_OPS = 11
"""Three fewer than the finishing walk: the budget stops it before the second RULE's two ops, and
the exhausted visit itself mints none — `_visit` returns `Exhausted` without consulting the state.
Re-derived by the reference case below, as `RULING_OPS` is."""


def _run_parking(backend, crash_at=None):
    """Always-breaching rulings, so the back edge is taken until the budget runs out."""
    deployment = RulingDeployment(rulings=["breach"] * 6)
    fault = Fault(crash_at)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"park-{run_id}"
    backend.register(name, parking_ruling_machine_wf, deployment, fault, ())
    task_id = backend.spawn(name, run_id)
    return backend.run_until_result(task_id), deployment, fault, run_id, task_id


def test_a_machine_that_exhausts_still_commits_on_both_engines(backend):
    """`run_machine`'s postamble has ONE call site and no `if` above it, so a run that exhausted
    reaches the canonical record exactly as one that finished does, on both engines.

    Every other machine case on both engines walks to a `Finish`, and the only `machine-parked`
    rows anywhere in the suite are recorder-based. The commitment-on-every-exit-path claim was
    therefore green over the empty set on the two engines it matters on: the empty-set shape this
    section closes, one exit path over.

    The judgement rows are what make it more than a status check: three of them, not four, because
    the walk stops before the second RULE, so the record says where the machine got to."""
    snap, _deployment, fault, run_id, _ = _run_parking(backend)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result["path"] == PARKING_PATH, snap.result
    assert snap.result["stopped"] == "machine-parked", snap.result
    assert snap.result["artifact_id"] == PARKING_ARTIFACT
    assert snap.result["files"] == RULING_FILES, snap.result["files"]
    assert fault.count == PARKING_OPS, "the walk changed shape — re-derive the crash sweep's bound"
    assert backend.ledger_kinds(run_id) == PARKING_KINDS, backend.ledger_kinds(run_id)


def test_the_parking_machine_commits_after_a_crash_at_every_op(backend):
    """The durability-critical half. A run that exhausted its budget has, by construction, been
    going wrong for a while — so a crash in its commitment tail is the case where losing the
    record costs most, and it is the one no durable test covered."""
    for k in range(1, PARKING_OPS + 1):
        snap, _deployment, fault, run_id, _ = _run_parking(backend, crash_at=k)
        assert fault.armed is False, f"k={k}: the fault never fired"
        assert snap is not None, f"k={k}: no result"
        assert snap.state == "completed", f"k={k}: {snap}"
        assert snap.result["path"] == PARKING_PATH, f"k={k}: {snap.result['path']}"
        assert snap.result["stopped"] == "machine-parked", f"k={k}: {snap.result['stopped']}"
        assert snap.result["artifact_id"] == PARKING_ARTIFACT, f"k={k}"
        assert backend.ledger_kinds(run_id) == PARKING_KINDS, f"k={k}"


# --- a viewer may READ a run it does not own, and may not advance it ---------------------------


def _viewer(backend, task_id, domain):
    """A `DurableHandler` bound to a task this process never claimed, guarded — the decisions-pane
    shape. `ledger=None` is the durable no-commit mode, which is right and was never sufficient."""
    tape = set(backend.checkpoint_states(task_id))
    ctx = ViewingCtx(backend.unclaimed_ctx(task_id), tape)
    return DurableHandler(ctx, domain, ledger=None), ctx


def test_a_viewer_replays_a_complete_run_without_touching_it(backend):
    """Replay IS the read, and on a complete run it is a genuine one.

    A decision is not on the tape — `Checkpoint` is `(key, state)`, so verdicts and prompts are
    absent by construction — and that is the substrate working, because both are a function of the
    tape: re-run the deterministic loop with recorded results replayed and it rebuilds them. The
    two assertions that make this a read rather than a re-run are the delta and the call list.
    """
    domain = CountingDomain()
    fault = Fault(None)
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"view-{run_id}"
    backend.register(name, two_step_wf, domain, fault, ())
    task_id = backend.spawn(name, run_id)
    assert backend.run_until_result(task_id).state == "completed"
    before = backend.checkpoint_keys(task_id)
    kinds_before = backend.ledger_kinds(run_id)

    viewing_domain = CountingDomain()
    handler, ctx = _viewer(backend, task_id, viewing_domain)
    assert handler.run(lambda: two_step_wf(run_id)) == {"a": 10, "b": 20}

    # The domain was never consulted: every op re-bound from the record.
    assert viewing_domain.calls == []
    assert len(ctx.reached) == len(before)
    # Neither bookkeeper moved.
    assert backend.checkpoint_keys(task_id) == before
    assert backend.ledger_kinds(run_id) == kinds_before


def _two_step_plus_one(run_id: str):
    """`two_step_wf` with one op more than the recorded run has.

    The realistic shape of outrunning, and the one both engines permit: the tape is COMPLETE and
    the code the viewer holds has moved past it — a workflow edited since the run, or a pane
    driving a variant. `call_tool("a", ...)` again rather than a new tool name, so `CountingDomain`
    still answers and the extra op lands as `step;tool:a#2`, an occurrence the tape does not have.
    """
    result = yield from two_step_wf(run_id)
    result["c"] = yield from call_tool("a", {}, int)
    return result


def test_a_viewer_refuses_to_outrun_the_tape_and_the_unguarded_drive_writes(backend):
    """The exposure and the guard on one run, on BOTH engines.

    Replayed past its frontier, an op with no checkpoint has its thunk run and committed into a
    run this process does not own. With `ledger=None` the canonical appends are suppressed, so the
    run gains checkpoints and no ledger rows — and when the real worker resumes it finds those
    checkpoints present, honours at-most-once, skips the thunks, and the appends never happen.

    **A COMPLETED run, deliberately.** Crashing the run is the more obvious way to get a partial
    tape, and Postgres REFUSES the unguarded write with `FailedTask`, because
    `absurd.set_task_checkpoint_state` raises on a run already failed. That is one of the three
    arms it does check (run-not-found, cancelled, failed), and a crashed run sits squarely inside
    it. So a crash scenario cannot demonstrate the hazard on Absurd: it demonstrates the one state
    Absurd already handles, and a rejection there is easy to misread as "unreachable".

    What the guard does NOT check is claim ownership, so a completed run — id present, not
    cancelled, not failed — takes the write from anyone holding the real `owner_run_id`, which is
    one `SELECT` away. That is the window, and it is where this test stands.

    The last block is the anti-vacuity half, and it is what the paragraph above earns: the refusal
    means nothing unless the same drive without the wrapper really writes, on this engine, now.
    """
    domain = CountingDomain()
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"outrun-{run_id}"
    backend.register(name, two_step_wf, domain, Fault(None), ())
    task_id = backend.spawn(name, run_id)
    assert backend.run_until_result(task_id).state == "completed"
    recorded = backend.checkpoint_keys(task_id)
    kinds = backend.ledger_kinds(run_id)

    handler, ctx = _viewer(backend, task_id, CountingDomain())
    with pytest.raises(OutranTheTape) as raised:
        handler.run(lambda: _two_step_plus_one(run_id))
    # It stopped AT the frontier, not before it: everything recorded was re-bound first.
    assert len(ctx.reached) == len(recorded)
    assert raised.value.key.display() not in recorded
    assert backend.checkpoint_keys(task_id) == recorded
    assert backend.ledger_kinds(run_id) == kinds

    # ANTI-VACUITY. Without the wrapper the same drive commits the op it had no record of.
    raw = backend.unclaimed_ctx(task_id)
    DurableHandler(raw, CountingDomain(), ledger=None).run(lambda: _two_step_plus_one(run_id))
    after = backend.checkpoint_keys(task_id)
    assert len(after) > len(recorded), (
        "the unguarded viewer wrote nothing, so this test proves nothing about the guard"
    )
    # And the run now carries a checkpoint with no ledger row behind it — the shape that makes
    # the real worker skip the append when it resumes.
    assert backend.ledger_kinds(run_id) == kinds
