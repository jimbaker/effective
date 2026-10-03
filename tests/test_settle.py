"""A settled checkpoint holds the first value written under its name, on both engines.

`settle` is the write a race's choice goes through and `peek_step` the read its losers make, so
both take a checkpoint's full name and neither counts an occurrence."""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import private

from effective import sqlite
from effective.handlers.absurd import SeedingCtx, _PrefixedCtx
from effective.keys import Key, gather_prefix, race_choice
from effective.sqlite import ClaimLost, SqliteApp
from effective.steering import SteeringCtx

CHOICE = race_choice(0)


def _settles(params: Any, ctx: Any) -> dict[str, Any]:
    """Everything one attempt can observe about settling, as the task's result."""
    before = ctx.peek_step(CHOICE)
    first = ctx.settle(CHOICE, {"winners": [0]})
    second = ctx.settle(CHOICE, {"winners": [1]})
    branch = _PrefixedCtx(ctx, gather_prefix(0, 1))
    framed = branch.settle(CHOICE, {"winners": [2]})
    counted = ctx.step(Key.parse("step;after"), lambda: "ran")
    return {
        "before": list(before),
        "first": first,
        "second": second,
        "after": list(ctx.peek_step(CHOICE)),
        "framed": framed,
        "framed_at": list(ctx.peek_step(Key.parse(gather_prefix(0, 1) + CHOICE.stored()))),
        "counted": counted,
    }


@pytest.mark.parametrize("deployed", [False, True], ids=["concurrent-ctx", "deployed-ctx"])
def test_the_first_value_settled_stands(backend, deployed):
    """Reddens if a second settle overwrites the first, if a peek counts an occurrence (the step
    after it would read as `#2` and miss), or if a frame's settle lands outside its frame."""
    name = private("settles")
    backend.register_body(name, _settles, deployed=deployed)
    snapshot = backend.run_until_result(backend.spawn(name, str(uuid4()), max_attempts=1))
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == {
        "before": [False, None],
        "first": {"winners": [0]},
        "second": {"winners": [0]},
        "after": [True, {"winners": [0]}],
        "framed": {"winners": [2]},
        "framed_at": [True, {"winners": [2]}],
        "counted": "ran",
    }


@pytest.mark.parametrize("deployed", [False, True], ids=["concurrent-ctx", "deployed-ctx"])
def test_a_retry_reads_the_value_its_crashed_attempt_settled(backend, deployed):
    """Reddens if a settled checkpoint is attempt-local: the second attempt settles a different
    value and must be handed the first attempt's."""
    name = private("resettles")
    seen: list[Any] = []

    def body(params: Any, ctx: Any) -> Any:
        seen.append(ctx.settle(CHOICE, {"attempt": len(seen) + 1}))
        if len(seen) == 1:
            raise RuntimeError("the worker dies after settling")
        return "ok"

    backend.register_body(name, body, deployed=deployed)
    snapshot = backend.run_until_result(backend.spawn(name, str(uuid4()), max_attempts=2))
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert seen == [{"attempt": 1}, {"attempt": 1}]


def test_a_stale_claim_settles_nothing_and_reads_what_is_there(tmp_path, monkeypatch):
    """SQLite's fence on `settle`: a claim the task has moved past raises when the name is
    unsettled, and is handed the settled value when it is not, as a read on Absurd would be."""
    monkeypatch.setattr(sqlite, "CLAIM_LEASE_SECONDS", 0.0)
    path = str(tmp_path / "settle.db")
    stale, fresh = SqliteApp(path), SqliteApp(path)
    outcomes: list[Any] = []

    @stale.register_task("t")
    def stale_task(params: Any, ctx: Any) -> Any:
        assert fresh.work_batch(), "the fresh worker found nothing to reclaim"
        for name in (Key.parse("step;unsettled"), CHOICE):
            try:
                outcomes.append(ctx.settle(name, "stale"))
            except ClaimLost:
                outcomes.append(ClaimLost)
        return "stale"

    @fresh.register_task("t")
    def fresh_task(params: Any, ctx: Any) -> Any:
        return ctx.settle(CHOICE, "fresh")

    task = stale.spawn("t", {})
    stale.work_batch()
    assert outcomes == [ClaimLost, "fresh"]
    snapshot = fresh.fetch_task_result(task)
    assert snapshot is not None
    assert (snapshot.state, snapshot.result) == ("completed", "fresh")
    stale.close()
    fresh.close()


@pytest.mark.parametrize("wrapper", ["seeding", "steering"])
def test_a_fork_refuses_to_settle(wrapper):
    """A fork seeds and steers steps by name, and neither can say what a race chose, so both
    wrappers refuse rather than let the call reach the engine beneath them."""
    app = SqliteApp()
    base = sqlite.SqliteTaskContext(app.conn, uuid4(), app.write_lock)
    ctx = (
        SeedingCtx(base, {}, fork_point=Key.parse("never"))
        if wrapper == "seeding"
        else SteeringCtx(base, {})
    )
    for call in (lambda: ctx.peek_step(CHOICE), lambda: ctx.settle(CHOICE, 1)):
        with pytest.raises(NotImplementedError, match="race cannot run under"):
            call()
    app.close()


@pytest.mark.parametrize("deployed", [False, True], ids=["concurrent-ctx", "deployed-ctx"])
def test_step_resolved_hands_the_thunk_the_name_its_checkpoint_lands_under(backend, deployed):
    """Reddens if the name handed to the thunk is not the one the checkpoint is written under, at
    either occurrence, in a frame or out of one, or if a retry's hit runs the thunk."""
    name, seen, calls = private("resolves"), list[Any](), list[str]()
    op = Key.parse("step;tool:a")

    def body(params: Any, ctx: Any) -> Any:
        branch = _PrefixedCtx(ctx, gather_prefix(0, 1))

        def thunk(resolved: Key) -> str:
            calls.append(resolved.stored())
            return resolved.stored()

        handed = [
            ctx.step_resolved(op, thunk),
            ctx.step_resolved(op, thunk),
            branch.step_resolved(op, thunk),
        ]
        landed = [
            ctx.peek_step(Key.parse(stored))[0]
            for stored in ("step;tool:a", "step;tool:a#2", gather_prefix(0, 1) + "step;tool:a")
        ]
        seen.append(handed)
        if len(seen) == 1:
            raise RuntimeError("the worker dies after its steps")
        return {"handed": handed, "landed": landed}

    backend.register_body(name, body, deployed=deployed)
    snapshot = backend.run_until_result(backend.spawn(name, str(uuid4()), max_attempts=2))
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == {
        "handed": ["step;tool:a", "step;tool:a#2", "step;tool:a"],
        "landed": [True, True, True],
    }
    assert calls == ["step;tool:a", "step;tool:a#2", "step;tool:a"]
