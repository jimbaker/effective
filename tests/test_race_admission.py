"""Race admission: the pins of the admission build. A loser stopping at a structure lives
with the structure pins in `test_race.py` and `test_race_durable.py`.

A loser checks twice. Before the layers, its walk position against the horizon saved in the choice;
where an op reaches the engine, whether the engine can serve it. Each pin runs on SQLite and on
Absurd over both ctx shapes unless its name says otherwise.
"""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import pytest
from _conformance import private, review_name
from pydantic import BaseModel, Field, PrivateAttr
from test_race import _AnyTool
from test_race_durable import SHAPES, _described, _Sequential, _Settled, _Tools

from effective.api import (
    append_ledger,
    await_event,
    call_tool,
    gather,
    quorum,
    race,
    scoped,
    store_artifact,
)
from effective.choice import Chosen, EndingLost, Stopped, Unchosen, Won
from effective.combinators import recurse
from effective.domain import CallTool
from effective.engines.sqlite import SqliteTaskContext
from effective.govern import Refused
from effective.handlers.admission import digest, observed
from effective.handlers.base import Racing, Stopping
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Key, compose_key
from effective.ops import AppendLedgerRow, LedgerRow, Step, StoreArtifact

type Layers = Callable[[int], list[Callable[[Any], Any]]]


def _task(
    backend: Any,
    shape: str,
    program: Callable[[], Any],
    *,
    layers: Layers = lambda attempt: [],
    tools: Any = None,
    wrap: Callable[[Any, int], Any] = lambda ctx, attempt: ctx,
) -> tuple[Any, Any, Any]:
    """Run `program` as one task that dies once after its first attempt returns, and retries."""
    name, seen = private("admission"), {"attempt": 0}
    tools = _Tools() if tools is None else tools

    def body(params: Any, ctx: Any) -> Any:
        seen["attempt"] += 1
        attempt = seen["attempt"]
        ctx = wrap(
            _Sequential(ctx) if shape == "sequential" and backend.name == "sqlite" else ctx,
            attempt,
        )
        result = DurableHandler(ctx, tools, op_layers=layers(attempt)).run(program)
        if attempt == 1:
            raise RuntimeError("the worker dies after the race")
        return result

    backend.register_body(name, body, deployed=shape == "sequential")
    task = backend.spawn(name, str(uuid4()), max_attempts=2)
    return backend.run_until_result(task), task, tools


def _plain(key: str) -> bool:
    """Whether a checkpoint name is an op's own record, not a race's or a refusal's."""
    return not any(tag in key for tag in ("refusal;", "gated;", "race:0;choice", "race:0;endings"))


# --- A2: a caught gate refusal, then the same call ---------------------------------------------


def _repeated() -> Any:
    def retrying() -> Any:
        with suppress(Refused):
            yield from call_tool("b", {"deny": True}, str)
        return (yield from call_tool("b", {"deny": False}, str))

    return (yield from quorum(2, [retrying, lambda: call_tool("deny", {}, str)]))


def _repeated_gate() -> Callable[[Any], Any]:
    completed = threading.Event()

    def gate(op: Any) -> Any:
        if isinstance(op, Step) and op.op.name == "b" and op.op.args["deny"]:
            raise Refused(op, "first b refused")
        if isinstance(op, Step) and op.op.name == "deny":
            assert completed.wait(10)
            raise Refused(op, "no quorum")
        result = yield op
        if isinstance(op, Step) and op.op.name == "b":
            completed.set()
        return result

    return gate


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("placement", ["plain", "scoped", "gather", "nested"])
def test_A2_a_caught_gate_refusal_then_the_same_call_retries_to_its_endings(
    backend, shape, placement
):
    def program() -> Any:
        match placement:
            case "scoped":
                answer = yield from scoped(Key.parse("s"), _repeated)
            case "gather":
                answer = (yield from gather([_repeated]))[0]
            case "nested":
                outer = yield from race([_repeated])
                assert isinstance(outer, Chosen)
                answer = outer.winners[0].value
            case _:
                answer = yield from _repeated()
        return _described(answer)

    recorded = RecordingHandler(responses={"tool:b": "b"}, op_layers=[_repeated_gate()])
    first = recorded.run(program)
    assert ReplayHandler(recorded.trace).run(program) == first
    gate = _repeated_gate()  # one across both attempts: its event stays set once `b` has answered
    snapshot, _, _ = _task(backend, shape, program, layers=lambda attempt: [gate])
    assert snapshot.state == "completed", snapshot
    assert snapshot.result[0] == "impossible"


# --- A3, A4: a gate that decides differently on the retry --------------------------------------


def _store(leaf: str) -> Any:
    match leaf:
        case "artifact":
            return store_artifact("payload", "text/plain")
        case _:
            return append_ledger(LedgerRow(event_id=Key.parse("led:flip"), kind="k"))


def _caught(leaf: str) -> Callable[[], Any]:
    def branch() -> Any:
        try:
            yield from _store(leaf)
            return "allowed"
        except Refused:
            return "refused"

    return branch


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("leaf", ["artifact", "ledger", "rewritten artifact"])
def test_A3_a_gate_that_refused_a_store_write_and_forwards_it_fails_before_the_write(
    backend, shape, leaf
):
    """The retry's gate forwards what the first attempt's refused, or rewrites it first: the task
    fails, and no artifact or ledger row is written."""
    kind = StoreArtifact if "artifact" in leaf else AppendLedgerRow

    def layers(attempt: int) -> list[Callable[[Any], Any]]:
        def gate(op: Any) -> Any:
            if isinstance(op, kind) and attempt == 1:
                raise Refused(op, "vetoed")
            if isinstance(op, StoreArtifact) and leaf == "rewritten artifact":
                return (yield replace(op, value="rewritten"))
            return (yield op)

        return [gate]

    def program() -> Any:
        return _described((yield from race([_caught(leaf.removeprefix("rewritten "))])))

    snapshot, task, _ = _task(backend, shape, program, layers=layers)
    assert snapshot.state == "failed", snapshot
    assert not [
        k
        for k in backend.checkpoint_keys(task)
        if _plain(k) and ("artifact:" in k or "ledger;" in k)
    ]


class _DiesBeforeEndings:
    """A ctx whose worker dies as the race's endings are about to be saved."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def settle(self, name: Key, value: Any) -> Any:
        if name.stored().endswith(";endings"):
            raise RuntimeError("the worker dies before the endings")
        return self._ctx.settle(name, value)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


def _refuses_then_answers(attempt: int) -> list[Callable[[Any], Any]]:
    """A gate that refuses on the first attempt and answers every op itself on the next."""

    def gate(op: Any) -> Any:
        if attempt == 1:
            raise Refused(op, "no")
        return "answered"
        yield

    return [gate]


def _refused_or_cached() -> Any:
    try:
        return (yield from call_tool("cached", {}, str))
    except Refused:
        return "refused"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dies", ["after the endings", "before the endings"])
def test_A4_a_gate_that_refused_and_answers_on_the_retry_fails_the_task(backend, shape, dies):
    def wrap(ctx: Any, attempt: int) -> Any:
        return _DiesBeforeEndings(ctx) if dies == "before the endings" and attempt == 1 else ctx

    def program() -> Any:
        return _described((yield from race([_refused_or_cached])))

    snapshot, _, tools = _task(backend, shape, program, layers=_refuses_then_answers, wrap=wrap)
    assert snapshot.state == "failed", snapshot
    assert tools.calls == []


# --- A5: the stored choice is the one acted on -------------------------------------------------


def test_A5_the_handler_acts_on_the_choice_the_store_returned(sqlite_app):
    """The store answers `settle` with a choice other than the one proposed, as it would when an
    earlier incarnation saved it: the race answers with the stored winners."""
    app, calls = sqlite_app(), list[str]()

    class _Overruled:
        concurrent_safe = False

        def __init__(self, ctx: Any) -> None:
            self._ctx = ctx

        def settle(self, name: Key, value: Any) -> Any:
            if name.stored() == "race:0;choice":
                value = {**value, "winners": [1], "batch": [1]}
            return self._ctx.settle(name, value)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._ctx, attr)

    class _Calls:
        def run(self, op: Any) -> Any:
            calls.append(op.name)
            return op.name

    def program() -> Any:
        answer = yield from race(
            [lambda: call_tool("a", {}, str), lambda: call_tool("b", {}, str)]
        )
        assert isinstance(answer, Chosen)
        return [winner.index for winner in answer.winners]

    ctx: Any = _Overruled(SqliteTaskContext(app.conn, uuid4(), app.write_lock))
    assert DurableHandler(ctx, _Calls()).run(program) == [1]
    assert calls == ["a", "b"]


# --- A6, A7: a stopped loser that layers keep busy ---------------------------------------------


def _cached(op: Any) -> Any:
    """Answers every `cached` call itself; forwards the rest."""
    if isinstance(op, Step) and op.op.name == "cached":
        return "hit"
    return (yield op)


def test_A6_a_loser_looping_on_cached_answers_stops_on_every_interpreter(sqlite_app):
    looped = [0]

    def loser() -> Any:
        while looped[0] < 10_000:
            looped[0] += 1
            yield from call_tool("cached", {}, str)
        return "ran out"

    def program() -> Any:
        return _described((yield from race([lambda: call_tool("won", {}, str), loser])))

    recorded = RecordingHandler(responses={"tool:won": "won"}, op_layers=[_cached])
    assert recorded.run(program) == ["chosen", [["won", 0, "won"], ["stopped", 1, None]]]
    looped[0] = 0
    app = sqlite_app()
    answer = DurableHandler(
        SqliteTaskContext(app.conn, uuid4(), None), _Tools(), op_layers=[_cached]
    ).run(program)
    assert answer[1][1] == ["stopped", 1, None]
    assert looped[0] < 10_000


def test_A7_an_upper_layer_looping_over_a_cache_after_the_choice_stops_at_once(sqlite_app):
    """The loser's op is admitted before the choice; after the engine answers, an upper layer
    loops over ops a lower cache layer answers, so nothing reaches the engine. The first new op it
    yields after the choice stops the loser (a step budget of 0)."""
    app, settled, started, looped = sqlite_app(), threading.Event(), threading.Event(), [0]

    def upper(op: Any) -> Any:
        value = yield op
        if isinstance(op, Step) and op.op.name == "loser":
            for _ in range(10_000):
                looped[0] += 1
                yield Step(name="cached", op=CallTool(name="cached", args={}, result_schema=str))
        return value

    tools = _Tools(
        {"won": lambda: started.wait(10), "loser": lambda: [started.set(), settled.wait(10)]}
    )

    def program() -> Any:
        branches = [lambda: call_tool("won", {}, str), lambda: call_tool("loser", {}, str)]
        return _described((yield from race(branches)))

    ctx = _Settled(SqliteTaskContext(app.conn, uuid4(), app.write_lock), settled)
    answer = DurableHandler(ctx, tools, op_layers=[upper, _cached]).run(program)
    assert answer[1][1] == ["stopped", 1, None]
    assert looped[0] <= 1


# --- A1, A11: an op a layer injects after the choice -------------------------------------------


class _Injector:
    """A layer that yields its own store write after the loser's op, once the choice is saved,
    and records whether a stop ever reached it as an exception and whether its `finally` ran."""

    def __init__(self, injected: str, catches: bool) -> None:
        self.started, self.chosen = threading.Event(), threading.Event()
        self.caught: list[str] = []
        self.closed: list[str] = []
        self._catches = catches
        self._op = (
            StoreArtifact("secret", "text/plain")
            if injected == "artifact"
            else AppendLedgerRow(LedgerRow(event_id=Key.parse("led:injected"), kind="k"))
        )

    def reset(self) -> None:
        self.started.clear()
        self.chosen.clear()
        self.closed.clear()

    def layer(self, step: Any) -> Any:
        if isinstance(step, Step) and step.op.name == "won":
            assert self.started.wait(10)
        try:
            value = yield step
            if isinstance(step, Step) and step.op.name == "loser":
                self.started.set()
                assert self.chosen.wait(10)
                try:
                    yield self._op
                except Exception as raised:
                    self.caught.append(type(raised).__name__)
                    if self._catches:
                        yield self._op
            return value
        finally:
            self.closed.append("closed")


def _catches_the_stop() -> Any:
    try:
        return (yield from call_tool("loser", {}, str))
    except Stopping:
        return "caught the stop"


def _injected_race() -> Any:
    return (yield from race([lambda: call_tool("won", {}, str), _catches_the_stop]))


@pytest.mark.parametrize("injected", ["artifact", "ledger"])
@pytest.mark.parametrize("catches", [False, True])
def test_A11_an_op_injected_after_the_choice_is_stopped_and_never_thrown_into_the_layer(
    backend, monkeypatch, injected, catches
):
    """The write never lands, the stop never reaches the layer as an exception (a layer that
    catches broadly sees nothing), the layer's `finally` runs, and the loser ends `Stopped`, even
    though the workflow itself catches `Stopping`."""
    injector = _Injector(injected, catches)
    read = Racing.read

    def reading(self: Racing, *args: Any) -> bool:
        failing = read(self, *args)
        if self.decided():
            injector.chosen.set()
        return failing

    monkeypatch.setattr(Racing, "read", reading)
    recorded = RecordingHandler(
        responses={"tool:won": "won", "tool:loser": "loser"}, op_layers=[injector.layer]
    )
    answer = recorded.run(_injected_race)
    assert isinstance(answer, Chosen)
    assert answer.endings[1] == Stopped(1)
    assert ReplayHandler(recorded.trace).run(_injected_race) == answer
    assert not recorded.artifacts
    assert not recorded.ledger
    injector.reset()
    name = private("injected")
    backend.register_body(
        name,
        lambda params, ctx: DurableHandler(ctx, _Tools(), op_layers=[injector.layer]).run(
            _injected_race
        ),
    )
    task = backend.spawn(name, str(uuid4()), max_attempts=1)
    snapshot = backend.run_until_result(task)
    assert snapshot.state == "completed", snapshot
    assert snapshot.result["endings"][1] == {"index": 1}
    assert not [
        k
        for k in backend.checkpoint_keys(task)
        if _plain(k) and ("artifact:" in k or "ledger;" in k)
    ]
    assert injector.caught == []
    assert injector.closed


_DEATH = os.path.join(os.path.dirname(__file__), "_admission_death.py")


def test_A1_an_injected_op_a_worker_death_interrupts_after_the_choice_is_not_rerun(tmp_path):
    """SQLite, by a real `os._exit`: the process dies inside an injected `audit` call after the
    choice is saved, and recovery does not call `audit` again."""
    db, calls, task = str(tmp_path / "run.db"), str(tmp_path / "calls"), str(uuid4())
    died = subprocess.run(
        [sys.executable, _DEATH, db, task, calls, "dying"], timeout=60, check=False
    )
    assert died.returncode == 9
    subprocess.run([sys.executable, _DEATH, db, task, calls, "recovering"], timeout=60, check=True)
    with open(calls) as log:
        assert log.read().splitlines() == ["audit"]


# --- A8: a retry whose inputs changed ----------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("change", ["value"])
def test_A8_a_loser_or_winner_whose_inputs_change_on_the_retry_fails_with_ending_lost(
    backend, shape, change
):
    def layers(attempt: int) -> list[Callable[[Any], Any]]:
        def answers(op: Any) -> Any:
            if isinstance(op, Step):
                match change:
                    case "type":
                        return (1,) if attempt == 1 else [1]
                    case "order":
                        return {"a": 1, "b": 2} if attempt == 1 else {"b": 2, "a": 1}
                    case _:
                        return "old" if attempt == 1 else "changed"
            return (yield op)

        return [answers]

    def branch() -> Any:
        value = yield from call_tool("cached", {}, str)
        return [type(value).__name__, list(value)]

    def program() -> Any:
        return _described((yield from race([branch])))

    snapshot, _, _ = _task(backend, shape, program, layers=layers)
    assert snapshot.state == "failed", snapshot
    assert "EndingLost" in str(snapshot.failure)


def test_A8_an_ending_saved_without_an_input_digest_fails_loudly_on_a_retry(sqlite_app):
    app, task, answer = sqlite_app(), uuid4(), ["first"]

    def cached(op: Any) -> Any:
        return answer[0]
        yield

    def program() -> Any:
        return _described((yield from race([lambda: call_tool("cached", {}, str)])))

    def run() -> Any:
        return DurableHandler(
            SqliteTaskContext(app.conn, task, app.write_lock), _Tools(), op_layers=[cached]
        ).run(program)

    run()
    row = app.conn.execute("SELECT state FROM checkpoints WHERE name='race:0;endings'").fetchone()
    endings = json.loads(row[0])
    endings[0].pop("inputs_sha256")
    app.conn.execute(
        "UPDATE checkpoints SET state=? WHERE name='race:0;endings'", (json.dumps(endings),)
    )
    answer[0] = "changed"
    with pytest.raises(EndingLost):
        run()


# --- A9: what a horizon holds ------------------------------------------------------------------


def _horizon(app: Any, name: str = "race:0;choice") -> dict[str, int]:
    (state,) = app.conn.execute("SELECT state FROM checkpoints WHERE name=?", (name,)).fetchone()
    return json.loads(state)["horizon"]


def _pure(value: Any) -> Any:
    return value
    yield


@pytest.mark.parametrize(
    ("loser_does", "entries"),
    [
        ("1,000 scoped enrichments", 1),
        ("recurse over 64, open at the choice", 65),
        ("recurse over 64, finished", 1),
    ],
)
def test_A9_a_horizon_holds_one_entry_per_live_loser_handler(sqlite_app, loser_does, entries):
    app, started, settled = sqlite_app(), threading.Event(), threading.Event()
    held = {"held"} | ({"chunk"} if "open" in loser_does else set())

    class _Domain:
        def run(self, op: Any) -> Any:
            if op.name == "won":
                assert started.wait(20)
            if op.name in held:
                started.set()
                assert settled.wait(20)
            return 1

    def loser() -> Any:
        if "scoped" in loser_does:
            for i in range(1000):
                yield from scoped(
                    compose_key(t"rec:{Index(i)}"), lambda: call_tool("enrich", {}, int)
                )
        else:
            yield from recurse(
                64,
                lambda n: _pure(list(range(n))),
                lambda i: call_tool("chunk", {}, int),
                lambda v: _pure(sum(v)),
            )
        return (yield from call_tool("held", {}, int))

    def program() -> Any:
        return (yield from race([lambda: call_tool("won", {}, int), loser]))

    ctx = _Settled(SqliteTaskContext(app.conn, uuid4(), app.write_lock), settled)
    DurableHandler(ctx, _Domain()).run(program)
    horizon = _horizon(app)
    assert len(horizon) == entries
    assert not [frame for frame in horizon if frame.startswith("race:0,0;")]


# --- A10: the refusal records a first attempt reads --------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_A10_a_first_attempt_reads_no_refusal_record(backend, shape):
    peeks: dict[int, list[str]] = {1: [], 2: []}

    class _Counting:
        def __init__(self, ctx: Any, attempt: int) -> None:
            self._ctx, self._attempt = ctx, attempt

        def peek_step(self, name: Key) -> Any:
            if "refusal;" in name.stored() or "gated;" in name.stored():
                peeks[self._attempt].append(name.stored())
            return self._ctx.peek_step(name)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._ctx, attr)

    def branch(tag: str) -> Callable[[], Any]:
        def calls() -> Any:
            for n in range(3):
                yield from call_tool(f"{tag}{n}", {}, str)
            return tag

        return calls

    def program() -> Any:
        return _described((yield from quorum(3, [branch("a"), branch("b"), branch("c")])))

    snapshot, _, _ = _task(
        backend, shape, program, wrap=lambda ctx, attempt: _Counting(ctx, attempt)
    )
    assert snapshot.state == "completed", snapshot
    assert peeks[1] == []


# --- A13: an admitted op finishes through its layers -------------------------------------------


def test_A13_an_op_admitted_before_the_choice_finishes_through_its_layers(sqlite_app):
    """The loser's op was admitted before the choice and answers after it. Its layer runs the code
    after its `yield`, and the loser returns the value: a stop counts only new work."""
    app, settled, started, after = sqlite_app(), threading.Event(), threading.Event(), list[str]()

    def layer(op: Any) -> Any:
        value = yield op
        after.append(op.op.name)
        return value

    tools = _Tools(
        {"won": lambda: started.wait(10), "held": lambda: [started.set(), settled.wait(10)]}
    )

    def program() -> Any:
        branches = [lambda: call_tool("won", {}, str), lambda: call_tool("held", {}, str)]
        return _described((yield from race(branches)))

    ctx = _Settled(SqliteTaskContext(app.conn, uuid4(), app.write_lock), settled)
    answer = DurableHandler(ctx, tools, op_layers=[layer]).run(program)
    assert answer == ["chosen", [["won", 0, "won"], ["unchosen", 1, "held"]]]
    assert sorted(after) == ["held", "won"]


# --- the fold: a finished structure still bounds its replay ------------------------------------


class _FinishedGather:
    """A loser whose gather loops over a cache until its hundredth `done`, then holds its last op
    until the choice is saved. `changed` makes the cache answer `loop` on the retry."""

    def __init__(self, changed: bool) -> None:
        self.changed, self.attempt, self.iterations = changed, 1, [0, 0]
        self.started, self.settled = threading.Event(), threading.Event()

    def cached(self, op: Any) -> Any:
        if isinstance(op, Step) and op.op.name == "cached":
            return "loop" if self.changed and self.attempt == 2 else "done"
        return (yield op)

    def run(self, op: Any) -> Any:
        if op.name == "won":
            assert self.started.wait(10)
        if op.name == "held":
            self.started.set()
            if self.attempt == 1:
                assert self.settled.wait(10)
        return op.name

    def body(self) -> Any:
        for i in range(10_000):
            self.iterations[self.attempt - 1] += 1
            if (yield from call_tool("cached", {}, str)) == "done" and i == 99:
                return "done"
        return "ran out"

    def program(self) -> Any:
        def loser() -> Any:
            yield from gather([self.body])
            return (yield from call_tool("held", {}, str))

        return _described((yield from race([lambda: call_tool("won", {}, str), loser])))


@pytest.mark.parametrize("changed", [False, True])
def test_a_finished_gathers_replay_is_capped_by_its_parents_count(sqlite_app, changed):
    """A loser's gather finishes before the choice; a retry replays it. Its children are no longer
    in the horizon, so the parent's count, with the finished children folded in, caps them. A
    cache that answers the same way replays to the same value; one that now answers so the body
    loops is stopped within that cap, and the retry fails with `EndingLost`."""
    app, task, run = sqlite_app(), uuid4(), _FinishedGather(changed)

    def attempt() -> Any:
        ctx = _Settled(SqliteTaskContext(app.conn, task, app.write_lock), run.settled)
        return DurableHandler(ctx, run, op_layers=[run.cached]).run(run.program)

    first = attempt()
    run.attempt = 2
    if changed:
        with pytest.raises(EndingLost):
            attempt()
        assert run.iterations[1] < 400
    else:
        assert attempt() == first
        assert run.iterations == [100, 100]


def test_A11_a_stop_from_below_never_reaches_a_layer_above(sqlite_app, monkeypatch):
    """The stop arises under an outer layer that catches broadly and would answer in its place.
    It escapes the driver instead, so the outer layer sees nothing and the loser ends `Stopped`."""
    app, injector, caught = sqlite_app(), _Injector("ledger", catches=False), list[str]()
    read = Racing.read

    def reading(self: Racing, *args: Any) -> bool:
        failing = read(self, *args)
        if self.decided():
            injector.chosen.set()
        return failing

    def swallowing(op: Any) -> Any:
        try:
            return (yield op)
        except Exception as raised:
            caught.append(type(raised).__name__)
            return "swallowed"

    def program() -> Any:
        return _described((yield from _injected_race()))

    monkeypatch.setattr(Racing, "read", reading)
    ctx = SqliteTaskContext(app.conn, uuid4(), app.write_lock)
    answer = DurableHandler(ctx, _Tools(), op_layers=[swallowing, injector.layer]).run(program)
    assert answer[1][1] == ["stopped", 1, None]
    assert caught == []


# --- A14: a park after the race resumes on the same attempt ------------------------------------


class _Denying(_Tools):
    """Refuses `deny` and `no` in the domain and answers the rest with their names. `no` waits
    until `after` has answered, so a branch calling it loses after its sibling has a value."""

    def __init__(self) -> None:
        super().__init__()
        self._after = threading.Event()

    def run(self, op: Any) -> Any:
        self.calls.append(op.name)
        match op.name:
            case "deny":
                raise Refused(Step(name="deny", op=op), "denied by the domain")
            case "no":
                assert self._after.wait(10)
                raise Refused(Step(name="no", op=op), "no quorum")
            case "after":
                self._after.set()
        return op.name


def _denies_gate(op: Any) -> Any:
    if isinstance(op, Step) and op.op.name == "deny":
        raise Refused(op, "denied by a gate")
    return (yield op)


def _caught_then_value() -> Any:
    with suppress(Refused):
        yield from call_tool("deny", {}, str)
    yield from call_tool("after", {}, str)
    return "value"


def _parking(backend: Any, shape: str, race_of: Callable[[], Any], layers: list[Any], tools: Any):
    """Run a race, then park on an event, then resume: the resume runs on the same attempt."""
    name = private("park")

    def program(run_id: str) -> Any:
        answer = yield from race_of()
        return [_described(answer), (yield from await_event(review_name(run_id), dict))]

    def body(params: Any, ctx: Any) -> Any:
        run: Any = _Sequential(ctx) if shape == "sequential" and backend.name == "sqlite" else ctx
        return DurableHandler(run, tools, op_layers=layers).run(lambda: program(params["run_id"]))

    backend.register_body(name, body, deployed=shape == "sequential")
    run_id = str(uuid4())
    task = backend.spawn(name, run_id, max_attempts=3)
    parked = backend.run_until_result(task)
    assert parked is None or parked.state not in ("completed", "failed"), parked
    at_park = list(tools.calls)
    backend.emit_event(task, review_name(run_id).stored(), {"ok": True})
    return backend.run_until_result(task), at_park


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    ("refused_by", "race_of", "calls"),
    [
        ("the domain, in a winner", lambda: race([_caught_then_value]), ["deny", "after"]),
        (
            "the domain, in an unchosen branch",
            lambda: quorum(2, [_caught_then_value, lambda: call_tool("no", {}, str)]),
            ["deny", "after", "no"],
        ),
        (
            "a gate, in an unchosen branch",
            lambda: quorum(2, [_caught_then_value, lambda: call_tool("no", {}, str)]),
            ["after", "no"],
        ),
    ],
    ids=["domain-winner", "domain-unchosen", "gate-unchosen"],
)
def test_A14_a_race_before_a_park_replays_its_refusals_on_the_resume(
    backend, shape, refused_by, race_of, calls
):
    gated = refused_by.startswith("a gate")
    tools = _Denying()
    snapshot, at_park = _parking(backend, shape, race_of, [_denies_gate] if gated else [], tools)
    assert sorted(at_park) == sorted(calls)
    assert snapshot.state == "completed", (snapshot.state, snapshot.failure)
    assert sorted(tools.calls) == sorted(calls)


# --- A15: every value a step can hand a branch has a witness -----------------------------------


class _Kind(Enum):
    A = "a"


class _Stamped(BaseModel):
    at: datetime


_HANDED: dict[str, tuple[Any, Any]] = {
    "datetime": (datetime, datetime(2026, 9, 19, tzinfo=UTC)),
    "date": (date, date(2026, 9, 19)),
    "stamped-model": (_Stamped, _Stamped(at=datetime(2026, 9, 19, tzinfo=UTC))),
    "enum": (_Kind, _Kind.A),
    "decimal": (Decimal, Decimal("1.10")),
    "uuid": (UUID, UUID(int=7)),
    "bytes": (bytes, b"utf-8 text"),
}


class _Handing(_Tools):
    def run(self, op: Any) -> Any:
        self.calls.append(op.name)
        return _HANDED[op.name][1]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("handed", list(_HANDED))
def test_A15_a_branch_handed_any_checkpointable_value_retries_to_its_ending(
    backend, shape, handed
):
    def branch() -> Any:
        value = yield from call_tool(handed, {}, _HANDED[handed][0])
        return repr(value)

    def program() -> Any:
        return _described((yield from race([branch])))

    snapshot, _, tools = _task(backend, shape, program, tools=_Handing())
    assert snapshot.state == "completed", (snapshot.state, snapshot.failure)
    assert tools.calls == [handed]


# --- A16: in-memory replay of a loser that finished after a leafless structure ------------------


def _seven() -> Any:
    return 7
    yield


@pytest.mark.parametrize("structure", ["scoped", "gather"])
def test_A16_a_loser_that_returned_after_a_leafless_structure_replays_to_its_value(
    monkeypatch, structure
):
    """The loser has its value before the choice, and its thread returns only after the
    winner's, so the winner is chosen whichever batch the parent reads first."""
    finished, won = threading.Event(), threading.Event()
    concurrently = Racing.concurrently

    def ordered(self: Racing) -> Any:
        def run(i: int) -> Any:
            slot = self.run(i)
            if i == 0:
                won.set()
            else:
                assert won.wait(10)
            return slot

        return concurrently(replace(self, run=run))

    monkeypatch.setattr(Racing, "concurrently", ordered)

    def holds(op: Any) -> Any:
        if isinstance(op, Step) and op.name == "tool:a":
            assert finished.wait(10)
        return (yield op)

    def loser() -> Any:
        yield from call_tool("b", {}, str)
        if structure == "scoped":
            value = yield from scoped(Key.parse("s"), _seven)
        else:
            (value,) = yield from gather([_seven])
        finished.set()
        return value

    def program() -> Any:
        return (yield from race([lambda: call_tool("a", {}, str), loser]))

    handler = RecordingHandler(responses=_AnyTool(), op_layers=[holds])
    answer = handler.run(program)
    assert answer == Chosen((Won(0, "a"),), (Won(0, "a"), Unchosen(1, 7)))
    assert ReplayHandler(handler.trace).run(program) == answer


# --- A17: a retry whose branch was handed something else fails, whatever the difference --------


class _Map(dict):
    pass


class _Coded(Refused):
    def __init__(self, code: int) -> None:
        super().__init__(
            Step(name="cached", op=CallTool(name="cached", args={}, result_schema=object)),
            "the same message",
        )
        self.code = code


class _Opaque:
    def __init__(self, code: int) -> None:
        self.code = code


class _Detailed(Refused):
    """A refusal carrying an attribute no encoding takes, which two attempts differ in."""

    def __init__(self, code: int) -> None:
        super().__init__(
            Step(name="cached", op=CallTool(name="cached", args={}, result_schema=object)),
            "the same message",
        )
        self.detail = _Opaque(code)


def _handed_changes(change: str) -> tuple[Callable[[], Any], Layers]:
    """A branch reading what its layer hands it, and the layers that hand it something else on
    the retry, differing in the way `change` names."""

    def layers(attempt: int) -> list[Callable[[Any], Any]]:
        def answers(op: Any) -> Any:
            match change:
                case "a refusal's attribute":
                    raise _Coded(attempt)
                case "a refusal's attribute with no encoding":
                    raise _Detailed(attempt)
                case "a value mutated after it was handed":
                    return {"n": attempt}
                case _:
                    return _Opaque(attempt)
            yield

        return [answers]

    def branch() -> Any:
        if change == "an unverifiable gather child":
            children: list[Any] = yield from gather([lambda: call_tool("cached", {}, object)])
            return children[0].code
        try:
            value: Any = yield from call_tool("cached", {}, object)
        except _Detailed as refused:
            return refused.detail.code
        except _Coded as refused:
            return refused.code
        return value.pop("n")

    return branch, layers


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "change",
    [
        "a refusal's attribute",
        "a refusal's attribute with no encoding",
        "a value mutated after it was handed",
        "an unverifiable gather child",
    ],
)
def test_A17_a_retry_handed_something_else_fails_with_ending_lost(backend, shape, change):
    branch, layers = _handed_changes(change)

    def program() -> Any:
        return _described((yield from race([branch])))

    snapshot, _, _ = _task(backend, shape, program, layers=layers)
    assert snapshot.state == "failed", snapshot
    assert "EndingLost" in str(snapshot.failure)


# --- A18: what counts as the same input ---------------------------------------------------------


class _Private(BaseModel):
    field: int = 1
    _hidden: int = PrivateAttr(default=0)


def _hiding(hidden: int) -> _Private:
    model = _Private()
    model._hidden = hidden
    return model


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        (datetime(2026, 9, 19, fold=0), datetime(2026, 9, 19, fold=1), True),
        (_hiding(0), _hiding(1), True),
        (datetime(2026, 9, 19), datetime(2026, 9, 20), False),
        (_Private(field=1), _Private(field=2), False),
        ({"a": 1}, _Map({"a": 1}), True),
        (0.0, -0.0, True),
        (2**100, 2**100 + 1, False),
        (10**29 + 7, 10**29 + 8, False),
        (float("nan"), float("nan"), True),
    ],
    ids=[
        "fold",
        "private",
        "a kept field",
        "a model's field",
        "a mapping's type",
        "zero",
        "nan",
        "wide integers",
        "wide integers, near the context",
    ],
)
def test_A18_two_inputs_are_the_same_when_a_checkpoint_encodes_them_alike(left, right, same):
    """Durability decides what a retry is held to: a difference the checkpoint encoding drops is
    not one a branch can be asked to reproduce."""
    assert (digest([observed(left)]) == digest([observed(right)])) is same


# --- A19: a value the store reshapes is the same input ------------------------------------------


class _Returning(_Tools):
    """A domain answering every tool with one value, so the retry is served the store's."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value

    def run(self, op: Any) -> Any:
        self.calls.append(op.name)
        return self.value


_RESHAPED: dict[str, tuple[Any, Any]] = {
    "a longer key first": (dict, {"bb": 1, "a": 2}),
    "keys of one length, unsorted": (dict, {"b": 1, "a": 2}),
    "a dict inside a list": (object, [{"zz": 1, "a": 2}]),
    "a typed mapping": (dict[str, int], {"bb": 1, "a": 2}),
    "keys already sorted": (dict, {"a": 2, "bb": 1}),
    "a negative zero": (float, -0.0),
    "a negative zero, untyped": (object, -0.0),
    "an exponent": (float, 1e16),
    "an exponent, untyped": (object, 1e16),
    "an integer past 2**63": (object, 2**63),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("handed", list(_RESHAPED))
def test_A19_a_value_the_store_reshapes_retries_to_its_ending(backend, shape, handed):
    """Absurd keeps a checkpoint as jsonb, which sorts an object's keys, drops a zero's sign and
    reads `1e+16` back as an integer. The first attempt is handed the live value and the retry
    what the store holds, so a witness finer than the store would fail a retry handed the same
    input."""
    schema, value = _RESHAPED[handed]

    def branch() -> Any:
        got: Any = yield from call_tool("t", {}, schema)
        return sorted(got) if isinstance(got, dict) else repr(got)

    def program() -> Any:
        return _described((yield from race([branch])))

    snapshot, _, tools = _task(backend, shape, program, tools=_Returning(value))
    assert snapshot.state == "completed", (snapshot.state, snapshot.failure)
    assert tools.calls == ["t"]


class _Excluding(BaseModel):
    field: int
    dropped: int = Field(0, exclude=True)


_ANSWERED: dict[str, Callable[[int], Any]] = {
    "a field the encoding drops": lambda attempt: _Excluding(field=1, dropped=attempt),
    "a tuple, then a list": lambda attempt: (1, 2) if attempt == 1 else [1, 2],
    "a mapping, then its subclass": lambda attempt: {"a": 1} if attempt == 1 else _Map({"a": 1}),
    "the same members, reordered": lambda attempt: (
        {"a": 1, "b": 2} if attempt == 1 else {"b": 2, "a": 1}
    ),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("answered", list(_ANSWERED))
def test_A19_a_layer_answering_one_encoding_two_ways_retries_to_its_ending(
    backend, shape, answered
):
    """A layer's answer never reaches the store, and the same rule holds of it: what the branch
    was handed is what a checkpoint would keep of it."""

    def layers(attempt: int) -> list[Callable[[Any], Any]]:
        def answers(op: Any) -> Any:
            return _ANSWERED[answered](attempt)
            yield

        return [answers]

    def branch() -> Any:
        got: Any = yield from call_tool("t", {}, object)
        return str(sorted(got)) if isinstance(got, dict) else str(got)

    def program() -> Any:
        return _described((yield from race([branch])))

    snapshot, _, _ = _task(backend, shape, program, layers=layers)
    assert snapshot.state == "completed", (snapshot.state, snapshot.failure)


# --- A20: the freshness rule's defense ----------------------------------------------------------


class _BoomingFirst(_Denying):
    """`boom` raises a retryable error before `deny` proceeds, so the error comes before any
    choice in both shapes."""

    def __init__(self) -> None:
        super().__init__()
        self._boomed = threading.Event()

    def run(self, op: Any) -> Any:
        if op.name == "boom":
            self.calls.append(op.name)
            self._boomed.set()
            raise RuntimeError("before any choice")
        if op.name == "deny":
            assert self._boomed.wait(10)
        return super().run(op)


@pytest.mark.parametrize("shape", SHAPES)
def test_A20_a_race_that_left_records_without_a_choice_fails_its_attempt(backend, shape):
    """What makes a fresh walk safe: a race that wrote records and saved no choice cannot be
    survived, since no branch was left to win it and only refusals are delivered to the
    workflow. So no resume at attempt 1 reaches such a race's records."""
    tools = _BoomingFirst()

    def caught_then_raises() -> Any:
        with suppress(Refused):
            yield from call_tool("deny", {}, str)
        raise LookupError("the branch's own code raised after the refusal")

    def program(run_id: str) -> Any:
        with suppress(Exception):  # the workflow tries to survive the race's error
            yield from race([lambda: call_tool("boom", {}, str), caught_then_raises])
        return (yield from await_event(review_name(run_id), dict))

    name = private("fresh")

    def body(params: Any, ctx: Any) -> Any:
        run: Any = _Sequential(ctx) if shape == "sequential" and backend.name == "sqlite" else ctx
        return DurableHandler(run, tools).run(lambda: program(params["run_id"]))

    backend.register_body(name, body, deployed=shape == "sequential")
    task = backend.spawn(name, str(uuid4()), max_attempts=2)
    snapshot = backend.run_until_result(task)
    assert snapshot.state == "failed", (snapshot.state, tools.calls)
    assert tools.calls.count("boom") == 2, tools.calls
    assert tools.calls.count("deny") == 1, tools.calls


# --- A21: a witness does not depend on the process that takes it --------------------------------


_SEEDED = """
import sys
sys.path.insert(0, "src")
from pydantic import BaseModel
from effective.handlers.admission import digest, observed


class Tagged(BaseModel):
    tags: set[str]


held = {"alpha", "beta", "gamma", "delta", "epsilon"}
print(digest([observed(["value", HELD])]))
"""


@pytest.mark.parametrize(
    "held",
    ["held", "[held]", "{'t': held}", "Tagged(tags=held)"],
    ids=["a set", "in a list", "in a mapping", "in a model"],
)
def test_A21_a_set_witnesses_the_same_in_another_process(held):
    """A retry runs in the process that claimed the task next, where `PYTHONHASHSEED` differs, so
    a witness that followed a set's iteration order would fail a retry handed the same set. The
    witness runs over what the handler records, `["value", ...]`, not over the set alone: the
    encoding turns a set nested anywhere into a list before a bare-set arm could see it."""
    seeds = [
        subprocess.run(
            [sys.executable, "-c", _SEEDED.replace("HELD", held)],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2", "3")
    ]
    assert len(set(seeds)) == 1, seeds


def test_A22_a_witness_holds_whatever_the_workflow_asks_of_its_decimal_context():
    """`normalize` rounds to the ambient decimal context, which a workflow sets for its own
    arithmetic, so two numbers the store keeps apart would witness alike under it."""
    with localcontext() as context:
        context.prec = 2
        assert digest([observed(1.21)]) != digest([observed(1.24)])
        assert digest([observed(1.21e16)]) == digest([observed(12100000000000000)])
        narrow = digest([observed(1.21)])
    assert narrow == digest([observed(1.21)])


@pytest.mark.parametrize("held", ["bytes", "cyclic"])
def test_A22_a_value_whose_encoding_raises_leaves_no_witness(held):
    """`observed` answers with no witness where the encoding raises, so the ending is
    unverifiable and its retry fails: an encoder's own error is not the branch's."""
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    assert observed(b"\xff" if held == "bytes" else cyclic) is None
