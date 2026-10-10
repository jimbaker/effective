"""`SeedingCtx`: seeding a fork child's PREFIX from the base's raw recorded state.

A durable fork re-runs the base workflow, replaying the prefix's recorded results through
`ctx.step` (so the tail is durable by construction) without calling the domain. `SeedingCtx` is the
ctx decorator that does the replay: a `name` in the seed map returns the seeded RAW value
(committed through the inner ctx, no domain thunk); a `name` outside it runs live. The seed map's
keys are the fork boundary, and `unconsumed()` is the seed-exhausted-at-the-boundary check.

Unit-level: the seeding CONTRACT, with a spy ctx. The end-to-end durable proof, a fork child that
survives crash-at-every-op and replays its prefix from its OWN checkpoints, is a conformance case,
where `SeedingCtx` wraps a real engine ctx.
"""

import pytest

from effective.counterfactual import ForkedPrefixAwait
from effective.handlers.durable import SeedBoundaryError, SeedingCtx
from effective.keys import Key, Segment


class _SpyCtx:
    """A minimal TaskContext that RUNS each step's thunk and records `(name, result)` — standing in
    for a real ctx whose checkpoint would hold that result."""

    def __init__(self) -> None:
        self.committed: list[tuple[str, object]] = []
        self.awaited: list[str] = []
        self.task_id = "t-child"

    def step(self, name, thunk, /):
        result = thunk()  # a real ctx runs the thunk once; replay returns the saved value
        self.committed.append((name, result))
        return result

    def await_event(self, name, /):
        # `.stored()` because this spy stands in for the ENGINE boundary, which speaks text —
        # the same unwrap `SdkCtx.await_event` makes. Recording the `Key` instead would let an
        # f-string below compose its repr.
        self.awaited.append(name.stored())
        return f"answer:{name.stored()}"

    def sleep_until(self, when, /, *, name: Key | None = None) -> None:
        pass


def test_a_seeded_step_returns_the_raw_seed_and_never_runs_the_domain_thunk():
    """The free prefix: a seeded op replays the base's RAW checkpoint value (verbatim, not decoded;
    the meter folds usage from the raw envelope after `ctx.step`), and the domain is never called.
    It is committed THROUGH the inner ctx so the child's own checkpoint holds it (durable-by-
    construction: a crash mid-tail replays the prefix from the child, not the base)."""
    spy = _SpyCtx()
    raw = {"result": "extracted", "usage": {"cost": 0.01}}  # the raw {result, usage} envelope
    ctx = SeedingCtx(spy, {Key.parse("extract"): raw}, fork_point=Key.parse("review:m1"))
    domain_ran: list[int] = []

    got = ctx.step(Key.parse("extract"), lambda: domain_ran.append(1) or "LIVE")

    assert got == raw  # the raw seed, verbatim
    assert domain_ran == []  # the domain thunk is dead on the prefix
    assert spy.committed == [(Key.parse("extract"), raw)]  # committed through the inner ctx
    assert ctx.consumed == {Key.parse("extract")}


def test_an_unseeded_step_runs_live_only_in_the_LIVE_phase():
    """A step outside the seed is the TAIL, but only past the fork point.

    Before the fork point every op is by definition a recorded prefix op, so a live one there is
    a divergence, not a tail. "Outside the seed means tail" is true only in `Live`."""
    ctx = _SpyCtx()
    seeding = SeedingCtx(ctx, {Key.parse("a"): 1}, fork_point=Key.parse("go"))

    # in Seeding, an unseeded step is a boundary error, not the tail
    with pytest.raises(SeedBoundaryError, match="ran LIVE during the Seeding phase"):
        seeding.step(Key.parse("b"), lambda: "live")

    # past the fork point it runs live, as the tail
    seeding.await_event(Key.parse("go"))
    assert seeding.step(Key.parse("b"), lambda: "live") == "live"
    assert (Key.parse("b"), "live") in ctx.committed


def test_unconsumed_flags_a_prefix_key_the_child_never_reached():
    """The seed-exhausted-at-the-boundary check: a leftover seed key means the child's op
    sequence diverged from the base's prefix, a bug the driver asserts against."""
    spy = _SpyCtx()
    ctx = SeedingCtx(
        spy, {Key.parse("a"): 1, Key.parse("b"): 2}, fork_point=Key.parse("review:m1")
    )
    ctx.step(Key.parse("a"), lambda: "x")
    assert ctx.unconsumed() == frozenset({Key.parse("b")})  # b never reached
    ctx.step(Key.parse("b"), lambda: "y")
    assert ctx.unconsumed() == frozenset()  # fully consumed at the boundary


def test_await_and_attributes_delegate_unchanged():
    """Seeding touches only `step`. An await is NOT rescoped here (that is `RenamedAwaitCtx`'s job;
    the two compose); everything else delegates via `__getattr__`."""
    spy = _SpyCtx()
    ctx = SeedingCtx(spy, {}, fork_point=Key.parse("review:m1"))
    assert ctx.await_event(Key.parse("review:m1")) == "answer:review:m1"
    assert spy.awaited == ["review:m1"]  # unscoped
    assert ctx.task_id == "t-child"  # __getattr__ delegation


def test_seeding_is_occurrence_aware():
    """Each occurrence of a repeated op name replays ITS OWN base value; replaying every
    occurrence with the first's silently diverges an agent-loop fork's prefix."""
    spy = _SpyCtx()
    ctx = SeedingCtx(
        spy,
        {Key.parse("extract"): "first", Key.parse("extract#2"): "second"},
        fork_point=Key.parse("review:m1"),
    )
    assert ctx.step(Key.parse("extract"), lambda: "LIVE") == "first"
    assert ctx.step(Key.parse("extract#2"), lambda: "LIVE") == "second"
    assert ctx.unconsumed() == frozenset()  # both occurrences consumed


def test_a_tail_seed_after_the_fork_point_await_is_refused():
    """The phase crosses to `Live` at the fork-point await; a seeded `step` in `Live` is a TAIL op
    the driver should not have seeded (`through` reached past the fork point), so it raises
    rather than silently dropping the divergent row."""
    spy = _SpyCtx()
    ctx = SeedingCtx(
        spy, {Key.parse("ledger;reviewed:m1"): None}, fork_point=Key.parse("review:m1")
    )
    ctx.await_event(Key.parse("review:m1"))  # the fork point — seals
    with pytest.raises(SeedBoundaryError, match="TAIL op"):
        ctx.step(Key.parse("ledger;reviewed:m1"), lambda: None)


def test_a_handler_internal_await_does_not_seal():
    """The seal is for the WORKFLOW fork-point await; a handler or layer park does not seal.
    A measured trip parks mid-prefix on `budget-grant:` (via `_enforce_measured`'s direct
    `ctx.await_event`); sealing there would falsely reject the next seeded prefix op.

    That pass-through must be EARNED: the same await deadlocks a real child, so it is refused
    unless `transplanted` names it (see the test below). The two rules compose: naming it
    buys pass-through, and pass-through still does not seal."""
    spy = _SpyCtx()
    ctx = SeedingCtx(
        spy,
        {Key.parse("ask3"): "seeded", Key.parse("tail"): "x"},
        fork_point=Key.parse("review:m1"),
        transplanted=frozenset({Key.parse("budget-grant:r,0")}),
    )
    ctx.await_event(Key.parse("budget-grant:r,0"))  # handler-internal — must NOT seal
    assert (
        ctx.step(Key.parse("ask3"), lambda: "LIVE") == "seeded"
    )  # still seeds; no false SeedBoundaryError
    ctx.await_event(Key.parse("review:m1"))  # a WORKFLOW await DOES seal
    with pytest.raises(SeedBoundaryError):
        ctx.step(
            Key.parse("tail"), lambda: "LIVE"
        )  # after the real fork point, a seed match is refused


def test_a_prefix_await_that_is_not_the_fork_point_is_refused():
    """An unearned pass-through is a silent forever-park on a real engine: the child awaits in
    its own namespace, where the base's answer does not exist, and no driver emits there.
    Refuse at the await instead. The message names the culprit and the two ways out."""
    ctx = SeedingCtx(_SpyCtx(), {Key.parse("ask3"): "seeded"}, fork_point=Key.parse("review:m1"))
    with pytest.raises(ForkedPrefixAwait) as caught:
        ctx.await_event(Key.parse("budget-grant:r,0"))
    assert "budget-grant:r,0" in str(caught.value)
    assert "fork_point='budget-grant:r,0'" in str(caught.value)  # fork AT it...
    assert "transplanted" in str(caught.value)  # ...or promise you delivered the answer


def test_a_tail_await_past_the_fork_point_is_untouched():
    """The refusal is PHASE-scoped, like `ForkedSleep`'s. A counterfactual's own tail may park
    and be answered — that is how the driver delivers the delta at the fork point to begin with."""
    spy = _SpyCtx()
    ctx = SeedingCtx(spy, {}, fork_point=Key.parse("review:m1"))
    ctx.await_event(Key.parse("review:m1"))  # crosses to Live
    assert (
        ctx.await_event(Key.parse("approve;m1")) == "answer:approve;m1"
    )  # a tail await: delegated
    assert spy.awaited == ["review:m1", "approve;m1"]


def test_an_event_name_is_a_fact_not_a_queue_so_awaits_are_name_keyed(sqlite_app):
    """6.2. `SeedingCtx.step` counts occurrences (`name#k`) and `.await_event` does not, which
    reads like an omission and is not: the ENGINE answers one name once. Measured here, because
    the alternative — keying awaits by occurrence — would contradict the engine rather than fix
    anything.

    The cost is real and bounded: a `fork_point` naming a repeated await crosses `Seeding → Live`
    on the FIRST occurrence, so a fork at "the second one" cannot be expressed."""
    import uuid as _uuid

    from effective.api import await_event, step
    from effective.domain import CallTool
    from effective.handlers.durable import DurableHandler
    from effective.keys import compose_key

    def twice():
        a = yield from await_event(compose_key(t"rev:{Segment('r2')},{Segment('m1')}"), dict)
        yield from step("tool:mid", CallTool(name="m", result_schema=str))
        b = yield from await_event(compose_key(t"rev:{Segment('r2')},{Segment('m1')}"), dict)
        return [a, b]

    class Dom:
        def run(self, op):
            return "m"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        app = sqlite_app(f"{tmp}/once.db")

        @app.register_task("w")
        def task(params, ctx):
            return DurableHandler(ctx, Dom(), params=params).run(twice)

        tid = _uuid.UUID(str(app.spawn("w", {"run_id": "r1"})))
        for _ in range(6):
            app.work_batch()
        app.emit_event("rev:r2,m1", {"n": 1})  # ONE emit
        for _ in range(6):
            app.work_batch()

        snap = app.fetch_task_result(tid)
        assert snap is not None
        assert snap.state == "completed", snap
        assert snap.result == [{"n": 1}, {"n": 1}], "one emit answered both awaits"
