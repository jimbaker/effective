"""A claim the task has moved past changes nothing in the task, as on the reference engine.

A claim is stale once its task is no longer running as the attempt it claimed. Another worker
reclaiming an expired lease starts the next attempt; the claim sweep fails a task whose expired
lease was its last attempt. Absurd fails the stale run in both cases, and its checkpoint and
end-of-run writes raise. On SQLite the reclaim is that sweep, so the task's row decides.
"""

import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from effective import sqlite
from effective.keys import Key
from effective.ops import DONE_EVENT_PARAM, Unretryable
from effective.sqlite import ClaimLost, SqliteApp


class _Refuses(Unretryable):
    """A failure no retry would fix, so the task fails for good and answers its parent."""


def _ends(how: str, ctx: Any) -> Any:
    """End a claim the way `how` names; each is a different write to the task's row."""
    match how:
        case "complete":
            return "stale"
        case "retry":
            raise RuntimeError("a retry would run this again")
        case "fail":
            raise _Refuses("no retry fixes this")
        case "park":
            return ctx.await_event(Key.parse("never-emitted"))
        case "sleep":
            return ctx.sleep_until(datetime.now(UTC) + timedelta(days=1), name=Key.parse("nap"))
        case "group":
            # A gather's barrier groups a refused write with a sibling's own crash.
            try:
                return ctx.step(Key.parse("late"), lambda: "late")
            except ClaimLost as lost:
                group = [lost, RuntimeError("a sibling")]
                raise ExceptionGroup("gather branches raised", group) from None
    raise AssertionError(how)


def _store(app: SqliteApp) -> tuple[Any, ...]:
    """Everything a stale claim could write: the task row, its checkpoints, and the events."""
    return (
        app.conn.execute(
            "SELECT state, attempt, result, failure, waiting_event, available_at, claimed_by "
            "FROM tasks"
        ).fetchall(),
        sorted(app.conn.execute("SELECT name, state FROM checkpoints").fetchall()),
        sorted(app.conn.execute("SELECT name FROM events").fetchall()),
    )


def _stale_claim(
    tmp_path: Any, stall: str, stale_end: str, fresh_end: str | None
) -> tuple[Any, Any, dict[str, Any]]:
    """The store just after the second worker's batch, again after the stale claim ended, and
    what the stale claim itself saw.

    The stale worker claims first and holds inside its step's thunk (`stall="step"`, so its
    checkpoint write is the late one) or after its step committed (`stall="end"`, so the late
    write is the one that ends its claim). The lease is spent when taken. With a `fresh_end`, the
    second worker reclaims the task as the next attempt and ends it that way; with none, the task
    was on its last attempt and the second worker's batch only sweeps it."""
    path = str(tmp_path / "fence.db")
    stale, fresh = SqliteApp(path), SqliteApp(path)
    entered, release = threading.Event(), threading.Event()
    seen: dict[str, Any] = {}

    @stale.register_task("t")
    def stale_task(params: Any, ctx: Any) -> Any:
        def work() -> str:
            if stall == "step":
                entered.set()
                release.wait(10)
            return "stale"

        ctx.step(Key.parse("x"), work)
        if stall == "end":
            entered.set()
            release.wait(10)
        seen["attempt"] = ctx.attempt.number
        return _ends(stale_end, ctx)

    @fresh.register_task("t")
    def fresh_task(params: Any, ctx: Any) -> Any:
        ctx.step(Key.parse("x"), lambda: "fresh")
        return _ends(fresh_end or "complete", ctx)

    def work_batch() -> None:
        try:
            seen["batch"] = stale.work_batch()
        except BaseException as raised:
            seen["batch"] = raised

    stale.spawn("t", {DONE_EVENT_PARAM: "parent-hears"}, max_attempts=3 if fresh_end else 1)
    worker = threading.Thread(target=work_batch)
    worker.start()
    try:
        assert entered.wait(10), "the stale worker never claimed the task"
        assert fresh.work_batch() is (fresh_end is not None)
        after_fresh = _store(fresh)
    finally:
        release.set()
        worker.join(10)
    seen["alive"] = worker.is_alive()
    after_stale = _store(fresh)
    stale.close()
    fresh.close()
    return after_fresh, after_stale, seen


STALLS = ("step", "end")
STALE_ENDS = ("complete", "retry", "fail", "park", "sleep", "group")
FRESH_ENDS = ("complete", "park")
"""A reclaiming worker's ends; `None` in the product below is the sweep, which ends nothing."""


def _answered(stall: str, stale_end: str, fresh_end: str | None) -> bool:
    """Whether the stale claim answers the parent: only a swept last attempt whose own body
    failed after its checkpoint, since the task is still at its attempt. A refused write grouped
    with a real failure still answers, for the real failure."""
    return fresh_end is None and stall == "end" and stale_end in ("retry", "fail", "group")


def test_a_stale_claim_changes_nothing(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """For every place the stale worker can stall and every way either batch ends, the task row
    and its checkpoints are the ones the second worker left. Events change only by the parent's
    answer from a swept last attempt, which Absurd's worker also sends. The stale worker's batch
    returns, and its ctx reports the attempt it claimed."""
    monkeypatch.setattr(sqlite, "CLAIM_LEASE_SECONDS", 0.0)
    wrong = {}
    cells = [(s, se, fe) for s in STALLS for se in STALE_ENDS for fe in (*FRESH_ENDS, None)]
    for n, (stall, stale_end, fresh_end) in enumerate(cells):
        cell = tmp_path / str(n)
        cell.mkdir()
        after_fresh, after_stale, seen = _stale_claim(cell, stall, stale_end, fresh_end)
        rows, checkpoints, events = after_fresh
        answer = [("parent-hears",)] if _answered(stall, stale_end, fresh_end) else []
        expected = (rows, checkpoints, sorted(events + answer))
        held = {"batch": True, "alive": False} | ({"attempt": 1} if stall == "end" else {})
        if after_stale != expected or seen != held:
            wrong[(stall, stale_end, fresh_end)] = (expected, after_stale, seen)
    assert wrong == {}


def test_a_checkpoint_extends_the_claim(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that commits a checkpoint before its lease runs out keeps its task, as Absurd's
    checkpoint write extends its claim, so a second worker polling after the first lease would
    have expired finds nothing to reclaim."""
    now = [0.0]
    monkeypatch.setattr(sqlite, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(sqlite, "CLAIM_LEASE_SECONDS", 10.0)
    path = str(tmp_path / "lease.db")
    live, poller = SqliteApp(path), SqliteApp(path)
    turns = {n: (threading.Event(), threading.Event()) for n in (1, 2)}

    def held(n: int) -> str:
        entered, go = turns[n]
        entered.set()
        go.wait(10)
        return f"s{n}"

    @live.register_task("t")
    def live_task(params: Any, ctx: Any) -> Any:
        return [ctx.step(Key.parse(f"s{n}"), lambda n=n: held(n)) for n in (1, 2)]

    @poller.register_task("t")
    def poller_task(params: Any, ctx: Any) -> Any:
        return "reclaimed"

    task = live.spawn("t", {})
    worker = threading.Thread(target=live.work_batch)
    worker.start()
    try:
        assert turns[1][0].wait(10)
        now[0] = 8.0  # inside the first lease, which runs to 10
        turns[1][1].set()
        assert turns[2][0].wait(10)  # the first checkpoint committed at 8
        now[0] = 12.0  # past the first lease, inside the one the checkpoint extended to 18
        assert poller.work_batch() is False
    finally:
        for _, go in turns.values():
            go.set()
        worker.join(10)
    snapshot = live.fetch_task_result(task)
    assert snapshot is not None
    assert (snapshot.state, snapshot.result) == ("completed", ["s1", "s2"])
    live.close()
    poller.close()
