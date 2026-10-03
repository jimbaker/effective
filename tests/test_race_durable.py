"""`race` and `quorum` on the durable handler, over both engines.

Each pin is one of the durable predictions made before the build, named by its id.
A pin runs on each engine over two ctx shapes:

| shape        | SQLite                                  | Absurd                                |
|--------------|-----------------------------------------|---------------------------------------|
| `concurrent` | the claimed ctx, which holds a lock     | `ConcurrentAbsurdCtx`                 |
| `sequential` | the claimed ctx, declining concurrency  | the adapted SDK ctx a deployed worker |
|              |                                         | hands its handler                     |

A sequential race runs its branches in index order, so every row there has one answer. A
concurrent row holds a loser inside a domain call until the choice is in the store, which is how
it puts that loser's next admission after the flag without a timer.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from _conformance import private

from effective.api import append_ledger, ask_llm, call_tool, gather, quorum, race, scoped
from effective.checkpoints import keys, read_sqlite_conn
from effective.choice import (
    Answer,
    Chosen,
    Raised,
    Refusal,
    Stopped,
    TimedOut,
    Unchosen,
    Won,
)
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import CallTool, DomainOp
from effective.govern import Refused
from effective.handlers import base
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import (
    Key,
    Segment,
    compose_key,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
    scope_prefix,
)
from effective.ops import (
    AwaitEvent,
    CompositionRefused,
    LedgerRow,
    Step,
    Unretryable,
    leaves,
    unretryable,
)
from effective.sqlite import SqliteApp, SqliteLedger, SqliteTaskContext
from effective.viewing import ViewingCtx

SHAPES = ["concurrent", "sequential"]


class _Sequential:
    """A claimed SQLite ctx that declines concurrency, so a race runs in branch-index order."""

    concurrent_safe = False

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class _Tools:
    """A domain answering each tool with its own name, counting calls, and running `before`
    for a tool when one is registered: a hold, or a raise."""

    def __init__(self, before: dict[str, Callable[[], object]] | None = None) -> None:
        self.calls: list[str] = []
        self._before = before or {}

    def run(self, op: DomainOp[Any]) -> Any:
        assert isinstance(op, CallTool)
        self.calls.append(op.name)
        if (hold := self._before.get(op.name)) is not None:
            hold()
        return op.name


class _Settled:
    """Wraps a ctx and sets `settled` each time a checkpoint is settled through it."""

    def __init__(self, ctx: Any, settled: threading.Event) -> None:
        self._ctx, self._settled = ctx, settled

    def settle(self, name: Key, value: Any) -> Any:
        stored = self._ctx.settle(name, value)
        self._settled.set()
        return stored

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        return self._ctx.step(name, thunk)

    def await_event(self, name: Key, /) -> Any:
        return self._ctx.await_event(name)

    def sleep_until(self, when: Any, /, *, name: Key) -> None:
        return self._ctx.sleep_until(when, name=name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


def _calls(*names: str):
    def branch():
        for name in names:
            last = yield from call_tool(name, {}, str)
        return last

    return branch


def _refuses(reason: str):
    """A branch that refuses before its first op, so no flag can stop it first."""

    def branch():
        gate = Step(name="gate", op=CallTool(name="gate", args={}, result_schema=str))
        raise Refused(gate, reason)
        yield  # a generator, which never reaches its first yield

    return branch


def _described(answer: Answer[Any]) -> list[Any]:
    """An answer as plain data: its kind, then each ending as `(kind, index, detail)`."""
    match answer:
        case Chosen():
            kind = "chosen"
        case TimedOut():
            kind = "timeout"
        case _:
            kind = "impossible"
    endings: list[Any] = []
    for ending in answer.endings:
        match ending:
            case Won(index=i, value=v):
                endings.append(["won", i, v])
            case Unchosen(index=i, value=v):
                endings.append(["unchosen", i, v])
            case Refusal(index=i, reason=r):
                endings.append(["refusal", i, r])
            case Stopped(index=i):
                endings.append(["stopped", i, None])
            case Raised(index=i):
                endings.append(["raised", i, None])
    return [kind, endings]


def _run(
    backend: Any,
    shape: str,
    program: Callable[[], Any],
    domain: Any,
    *,
    attempts: int = 1,
    wrap: Callable[[Any], Any] = lambda ctx: ctx,
    after: Callable[[Any, int], None] = lambda ctx, attempt: None,
) -> Any:
    """Run `program` under a durable handler as one task, and return its result. `after` runs
    once the handler returns, with the ctx and the attempt, and may raise to fail the attempt."""
    name = private("race")
    seen = {"attempt": 0}

    def body(params: Any, ctx: Any) -> Any:
        seen["attempt"] += 1
        ctx = wrap(_Sequential(ctx) if shape == "sequential" and backend.name == "sqlite" else ctx)
        result = DurableHandler(ctx, domain).run(program)
        after(ctx, seen["attempt"])
        return result

    backend.register_body(name, body, deployed=shape == "sequential")
    snapshot = backend.run_until_result(backend.spawn(name, str(uuid4()), max_attempts=attempts))
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    return snapshot.result


@pytest.mark.parametrize("shape", SHAPES)
def test_D1_D2_a_race_answers_by_index_and_refusals_lose(backend, shape):
    """A refusal loses; the successes that complete the quorum win in branch-index order; every
    branch refusing makes the race impossible."""

    def program():
        chosen = yield from quorum(2, [_refuses("no"), _calls("b"), _calls("c")])
        hopeless = yield from race([_refuses("x"), _refuses("y")])
        return [_described(chosen), _described(hopeless)]

    assert _run(backend, shape, program, _Tools()) == [
        ["chosen", [["refusal", 0, "no"], ["won", 1, "b"], ["won", 2, "c"]]],
        ["impossible", [["refusal", 0, "x"], ["refusal", 1, "y"]]],
    ]


def test_D1_a_sequential_loser_admits_nothing(backend):
    """In branch-index order the winner's choice is saved before the loser starts, so the loser
    stops at its first admission and its tool is never called."""
    tools = _Tools()

    def program():
        return _described((yield from race([_calls("a"), _calls("b0", "b1")])))

    assert _run(backend, "sequential", program, tools) == [
        "chosen",
        [["won", 0, "a"], ["stopped", 1, None]],
    ]
    assert tools.calls == ["a"]


@pytest.mark.parametrize("shape", SHAPES)
def test_D3_a_retry_after_the_choice_serves_it(backend, shape):
    """The second attempt is served the choice and every op the first attempt recorded."""
    tools = _Tools()

    def program():
        return _described((yield from race([_calls("a"), _refuses("no")])))

    def crash_once(ctx: Any, attempt: int) -> None:
        if attempt == 1:
            raise RuntimeError("the worker dies after the race")

    assert _run(backend, shape, program, tools, attempts=2, after=crash_once) == [
        "chosen",
        [["won", 0, "a"], ["refusal", 1, "no"]],
    ]
    assert tools.calls == ["a"]


def test_D4_a_crash_between_the_save_and_the_flag_serves_the_saved_choice(backend):
    """Sequential: the first attempt settles the choice and dies before any loser hears it. The
    retry reads the choice from the store, serves the winner, and stops the loser at its first
    op, which no attempt ever calls."""
    tools = _Tools()
    died = {"yet": False}

    class _DiesAfterSettle(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            stored = self._ctx.settle(name, value)
            if not died["yet"]:
                died["yet"] = True
                raise RuntimeError("the worker dies after the choice is saved")
            return stored

    def program():
        return _described((yield from race([_calls("a"), _calls("b")])))

    result = _run(
        backend,
        "sequential",
        program,
        tools,
        attempts=2,
        wrap=lambda ctx: _DiesAfterSettle(ctx, threading.Event()),
    )
    assert result == ["chosen", [["won", 0, "a"], ["stopped", 1, None]]]
    assert tools.calls == ["a"]


@pytest.mark.parametrize("backend_shape", ["concurrent"])
def test_D6_D7_a_loser_admits_nothing_after_the_flag(backend, backend_shape):
    """The winner's call holds until the loser's first call has started, and that call holds
    until the choice is settled, so the loser's second op is admitted, or not, after the flag: it
    must not be. The race returns only once the loser's held op has committed."""
    started, settled = threading.Event(), threading.Event()

    def first_loser_call() -> None:
        started.set()
        settled.wait(10)

    tools = _Tools({"a": lambda: started.wait(10), "b0": first_loser_call})

    def program():
        return _described((yield from race([_calls("a"), _calls("b0", "b1")])))

    def committed(ctx: Any, attempt: int) -> None:
        found, _ = ctx.peek_step(Key.parse(race_prefix(0, 1) + "step;tool:b0"))
        assert found, "the race returned before the loser's admitted op committed"

    result = _run(
        backend,
        "concurrent",
        program,
        tools,
        wrap=lambda ctx: _Settled(ctx, settled),
        after=committed,
    )
    assert result == ["chosen", [["won", 0, "a"], ["stopped", 1, None]]]
    assert sorted(tools.calls) == ["a", "b0"]


def test_D9_a_race_inside_a_flagged_loser_settles_nothing(backend):
    """The outer winner's call holds until the inner race has started, and each
    inner branch's first call holds until a choice is settled, which only the outer race can do,
    so the flag reaches an inner race that is running and has chosen nothing."""
    inner_started, settled = threading.Event(), threading.Event()

    def first_inner_call() -> None:
        inner_started.set()
        settled.wait(10)

    tools = _Tools(
        {
            "won": lambda: inner_started.wait(10),
            "p0": first_inner_call,
            "q0": first_inner_call,
        }
    )

    def inner():
        return (yield from race([_calls("p0", "p1"), _calls("q0", "q1")]))

    def program():
        return _described((yield from race([_calls("won"), inner])))

    def no_inner_choice(ctx: Any, attempt: int) -> None:
        found, _ = ctx.peek_step(Key.parse(race_prefix(0, 1) + race_choice(0).stored()))
        assert not found, "an inner race settled a choice after its enclosing loser was flagged"

    result = _run(
        backend,
        "concurrent",
        program,
        tools,
        wrap=lambda ctx: _Settled(ctx, settled),
        after=no_inner_choice,
    )
    assert result == ["chosen", [["won", 0, "won"], ["stopped", 1, None]]]
    assert "p1" not in tools.calls
    assert "q1" not in tools.calls


@pytest.mark.parametrize("shape", SHAPES)
def test_D10_races_in_two_gather_branches_settle_apart(backend, shape):
    """Each gather branch counts its races from 0, so the first race in each branch settles
    `race:0;choice` under its own branch frame; a settle that dropped the frame would hand the
    second branch the first branch's choice. Races under two scopes continue one count."""
    a, b = compose_key(t"arm:{Segment('a')}"), compose_key(t"arm:{Segment('b')}")

    def program():
        in_branches = yield from gather(
            [
                lambda: race([_calls("a0"), _refuses("a1")]),
                lambda: race([_refuses("b0"), _calls("b1")]),
            ]
        )
        first = yield from scoped(a, lambda: race([_calls("c0"), _refuses("c1")]))
        second = yield from scoped(b, lambda: race([_refuses("d0"), _calls("d1")]))
        return [*map(_described, in_branches), _described(first), _described(second)]

    def apart(ctx: Any, attempt: int) -> None:
        stored = [
            ctx.peek_step(Key.parse(frame + race_choice(r).stored()))
            for frame, r in (
                (gather_prefix(0, 0), 0),
                (gather_prefix(0, 1), 0),
                (scope_prefix(a), 0),
                (scope_prefix(b), 1),
            )
        ]
        assert [found for found, _ in stored] == [True] * 4
        assert [choice["winners"] for _, choice in stored] == [[0], [1], [0], [1]]

    assert _run(backend, shape, program, _Tools(), after=apart) == [
        ["chosen", [["won", 0, "a0"], ["refusal", 1, "a1"]]],
        ["chosen", [["refusal", 0, "b0"], ["won", 1, "b1"]]],
        ["chosen", [["won", 0, "c0"], ["refusal", 1, "c1"]]],
        ["chosen", [["refusal", 0, "d0"], ["won", 1, "d1"]]],
    ]


def test_D11_a_race_that_fails_before_choosing_saves_nothing(backend):
    """Sequential: a programming error before any choice fails the race and saves no choice, so
    the retry reads completions afresh and chooses from them."""
    fails = {"left": 1}

    def boom() -> None:
        if fails["left"]:
            fails["left"] -= 1
            raise ValueError("a programming error before any choice")

    tools = _Tools({"b": boom})

    def program():
        return _described((yield from race([_refuses("no"), _calls("b")])))

    seen: list[bool] = []

    def before(ctx: Any) -> Any:
        seen.append(ctx.peek_step(race_choice(0))[0])
        return ctx

    result = _run(backend, "sequential", program, tools, attempts=2, wrap=before)
    assert result == ["chosen", [["refusal", 0, "no"], ["won", 1, "b"]]]
    assert seen == [False, False], "a choice was saved by the attempt that failed"


class _Broken(Unretryable):
    """A programming error: a retry would raise it again, so it is a loser's `Raised` ending."""


def test_D12_endings_are_served_from_the_store(backend):
    """A loser's held op raises a programming error after the choice, so the first attempt ends
    it `Raised`; the winner's call waits until that op has started, so it is admitted before the
    flag. The retry finds that op unrecorded and stops the loser there, but the endings it answers
    with are the ones the first attempt stored."""
    started, settled = threading.Event(), threading.Event()

    def raise_after_the_choice() -> None:
        started.set()
        settled.wait(10)
        raise _Broken("the loser's op fails after the choice")

    tools = _Tools({"a": lambda: started.wait(10), "b0": raise_after_the_choice})

    def program():
        return _described((yield from race([_calls("a"), _calls("b0")])))

    def crash_once(ctx: Any, attempt: int) -> None:
        found, _ = ctx.peek_step(race_endings(0))
        assert found
        if attempt == 1:
            raise RuntimeError("the worker dies after the race")

    result = _run(
        backend,
        "concurrent",
        program,
        tools,
        attempts=2,
        wrap=lambda ctx: _Settled(ctx, settled),
        after=crash_once,
    )
    assert result == ["chosen", [["won", 0, "a"], ["raised", 1, None]]]


def test_D8_a_losers_spend_and_ledger_row_reach_the_record(sqlite_app):
    """On SQLite. The loser appends a row and makes one metered call before the
    flag; the winner's call holds until that call is running, and the loser's call holds until
    the choice is settled. The root meter folds both calls' spend, and the loser's row names it."""
    app = sqlite_app()
    started, settled = threading.Event(), threading.Event()
    spend = {"w": 0.001, "l0": 0.01, "l1": 0.1}

    def llm(op: Any) -> tuple[str, Usage]:
        match op.messages:
            case "w":
                started.wait(10)
            case "l0":
                started.set()
                settled.wait(10)
        return op.messages, Usage(cost=spend[op.messages])

    def loser():
        yield from append_ledger(LedgerRow(event_id=Key.parse("led:loser"), kind="k"))
        yield from ask_llm("l0", "l0", str)
        return (yield from ask_llm("l1", "l1", str))

    def program():
        return _described((yield from race([lambda: ask_llm("w", "w", str), loser])))

    handler = DurableHandler(
        _Settled(SqliteTaskContext(app.conn, uuid4(), app.write_lock), settled),
        MeteredInterpreter(llm=llm, tools=lambda op: None),
        ledger=SqliteLedger(app.conn, "r1", app.write_lock),
        contract=Contract.V1,
    )
    assert handler.run(program) == ["chosen", [["won", 0, "w"], ["stopped", 1, None]]]
    assert handler.meter.cost == pytest.approx(spend["w"] + spend["l0"])
    placements = app.conn.execute("SELECT writer_placement FROM ledger").fetchall()
    assert [p for (p,) in placements] == [race_prefix(0, 1) + "ledger;led:loser"]


def _metered_race(app: SqliteApp, settled: threading.Event, llm: Callable[[Any], Any]) -> Any:
    """D8's race over a fresh ctx of one task, so a second call is the task's next attempt."""

    def loser():
        yield from ask_llm("l0", "l0", str)
        return (yield from ask_llm("l1", "l1", str))

    def program():
        return _described((yield from race([lambda: ask_llm("w", "w", str), loser])))

    handler = DurableHandler(
        _Settled(SqliteTaskContext(app.conn, _TASK, app.write_lock), settled),
        MeteredInterpreter(llm=llm, tools=lambda op: None),
        contract=Contract.V1,
    )
    return handler.run(program), handler.meter.cost


_TASK = uuid4()


def test_D3_a_retry_serves_the_ops_a_loser_recorded(sqlite_app):
    """The loser's first call commits before the flag. On the next attempt the choice is read
    back and that call is served from its checkpoint, so its spend reaches the meter again and
    nothing is called twice; the loser still stops at its second call."""
    app = sqlite_app()
    started, settled = threading.Event(), threading.Event()
    spend = {"w": 0.001, "l0": 0.01}

    def llm(op: Any) -> tuple[str, Usage]:
        match op.messages:
            case "w":
                started.wait(10)
            case "l0":
                started.set()
                settled.wait(10)
        return op.messages, Usage(cost=spend[op.messages])

    def never(op: Any) -> Any:
        raise AssertionError(f"a retry called {op.messages!r}, which the record serves or stops")

    first = _metered_race(app, settled, llm)
    assert _metered_race(app, threading.Event(), never) == first
    assert first == (
        ["chosen", [["won", 0, "w"], ["stopped", 1, None]]],
        pytest.approx(spend["w"] + spend["l0"]),
    )


def test_D4_a_loser_stops_only_for_a_choice_the_store_holds(sqlite_app):
    """The save of the choice fails before it writes, while the loser is
    held until the race has decided: the loser must go on to its next call, since the store holds
    no choice to stop it for."""
    app = sqlite_app()
    started, deciding = threading.Event(), threading.Event()

    def first_loser_call() -> None:
        started.set()
        deciding.wait(10)

    tools = _Tools({"a": lambda: started.wait(10), "b0": first_loser_call})

    class _SaveFails(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            deciding.set()
            raise RuntimeError("the store refused the choice")

    def program():
        return (yield from race([_calls("a"), _calls("b0", "b1")]))

    ctx = _SaveFails(SqliteTaskContext(app.conn, uuid4(), app.write_lock), threading.Event())
    with pytest.raises(RuntimeError, match="refused the choice"):
        DurableHandler(ctx, tools).run(program)
    assert sorted(tools.calls) == ["a", "b0", "b1"]
    assert ctx.peek_step(race_choice(0)) == (False, None)


def _parks_before_a(op: Any) -> Any:
    """A layer that parks before tool `a`, as a gate asking a human does."""
    if isinstance(op, Step) and getattr(op.op, "name", "") == "a":
        yield AwaitEvent(name=Key.parse("ev:approve"), schema=str)
    return (yield op)


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "concurrent"])
def test_a_park_a_layer_injects_in_a_race_branch_is_refused(concurrent, sqlite_app):
    """An await a layer yields reaches the handler past admission. It is refused by its kind,
    where it would otherwise park the branch and let a gated op that never ran win the race."""
    app = sqlite_app()
    tools = _Tools()
    ctx = SqliteTaskContext(app.conn, uuid4(), app.write_lock if concurrent else None)

    def program():
        return (yield from race([_calls("a"), _calls("b")]))

    with pytest.raises(ExceptionGroup) as raised:
        DurableHandler(ctx, tools, op_layers=[_parks_before_a]).run(program)
    assert [type(leaf) for leaf in leaves(raised.value)] == [CompositionRefused]
    assert "a" not in tools.calls


def test_a_losers_value_survives_a_retry_that_stops_it(sqlite_app):
    """The loser's op, admitted before the flag, is refused after the choice; the loser catches
    the refusal and returns a value. The refusal is recorded beside the op, so the next attempt
    re-raises it there and the loser replays to the same value."""
    app, task = sqlite_app(), uuid4()

    def attempt() -> Any:
        started, settled = threading.Event(), threading.Event()

        def refuse_after_the_choice() -> None:
            started.set()
            settled.wait(10)
            gate = Step(name="b0", op=CallTool(name="b0", args={}, result_schema=str))
            raise Refused(gate, "no")

        tools = _Tools({"a": lambda: started.wait(10), "b0": refuse_after_the_choice})

        def loser():
            try:
                return (yield from call_tool("b0", {}, str))
            except Refused:
                return "fallback"

        def program():
            return _described((yield from race([_calls("a"), loser])))

        ctx = _Settled(SqliteTaskContext(app.conn, task, app.write_lock), settled)
        return DurableHandler(ctx, tools).run(program)

    first = attempt()
    assert first == ["chosen", [["won", 0, "a"], ["unchosen", 1, "fallback"]]]
    assert attempt() == first


def test_a_viewer_reads_a_race_without_writing_it(sqlite_app):
    """A viewer over a finished run's tape serves the race's choice, its winner and its endings,
    and answers as the run did."""
    app, task = sqlite_app(), uuid4()

    def program():
        return _described((yield from race([_calls("a"), _refuses("no")])))

    ran = DurableHandler(SqliteTaskContext(app.conn, task, app.write_lock), _Tools()).run(program)
    tape = {Key.parse(name) for name in keys(read_sqlite_conn(app.conn, task))}
    viewer = ViewingCtx(SqliteTaskContext(app.conn, task, app.write_lock), tape)
    assert DurableHandler(viewer, _Tools()).run(program) == ran


class _Opaque:
    """A branch value with no JSON form."""

    def __init__(self, label: str) -> None:
        self.label = label


def test_a_race_answers_with_values_that_have_no_json_form(sqlite_app):
    """A branch value is handed to the workflow, never stored, so a value with no JSON form wins
    on both handlers and replays to the same object type."""

    def program():
        answer = yield from race([lambda: _labelled("a"), _refuses("no")])
        return [
            type(ending.value).__name__ for ending in answer.endings if isinstance(ending, Won)
        ]

    def _labelled(name: str):
        return _Opaque((yield from call_tool(name, {}, str)))

    app, task = sqlite_app(), uuid4()
    for _ in range(2):
        ctx = SqliteTaskContext(app.conn, task, app.write_lock)
        assert DurableHandler(ctx, _Tools()).run(program) == ["_Opaque"]
    recorder = RecordingHandler(responses={"tool:a": "a"})
    assert recorder.run(program) == ["_Opaque"]
    assert ReplayHandler(recorder.trace).run(program) == ["_Opaque"]


def _refusing_b0(source: str, refused: threading.Event) -> tuple[_Tools, list[Any]]:
    """Tools and layers under which tool `b0` is refused by `source`, setting `refused` first,
    while tool `a` waits until it has been."""
    gate = Step(name="b0", op=CallTool(name="b0", args={}, result_schema=str))

    def refuse() -> None:
        refused.set()
        raise Refused(gate, f"the {source} says no")

    def refusing_layer(op: Any) -> Any:
        if isinstance(op, Step) and getattr(op.op, "name", "") == "b0":
            refuse()
        return (yield op)

    before: dict[str, Callable[[], object]] = {"a": lambda: refused.wait(10)}
    if source == "domain":
        before["b0"] = refuse
    return _Tools(before), [refusing_layer] if source == "gate" else []


@pytest.mark.parametrize("source", ["domain", "gate"])
def test_a_refusal_survives_a_crash_before_the_endings(sqlite_app, source):
    """A branch's op is refused before the choice, and the worker dies
    after the choice is saved and before the endings are. The retry reads the recorded refusal,
    so the branch ends a `Refusal` with its reason, whether the domain or a gate refused."""

    class _DiesAtTheEndings(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            if "endings" in name.stored():
                raise RuntimeError("the worker dies before the endings are saved")
            return self._ctx.settle(name, value)

    def program():
        return _described((yield from race([_calls("a"), _calls("b0")])))

    def attempt(app: Any, task: Any, wrap: Callable[[Any], Any]) -> Any:
        tools, layers = _refusing_b0(source, threading.Event())
        ctx = wrap(SqliteTaskContext(app.conn, task, app.write_lock))
        return DurableHandler(ctx, tools, op_layers=layers).run(program)

    app, task = sqlite_app(), uuid4()
    with pytest.raises(RuntimeError, match="before the endings"):
        attempt(app, task, lambda ctx: _DiesAtTheEndings(ctx, threading.Event()))
    assert attempt(app, task, lambda ctx: ctx) == [
        "chosen",
        [["won", 0, "a"], ["refusal", 1, f"the {source} says no"]],
    ]


class _Vetoed(Refused):
    """A gate's refusal of its own type, which a workflow may catch by name."""


def test_a_gates_refusal_keeps_its_type_across_a_retry(sqlite_app):
    """A gate refuses the loser's op with a `Refused` subclass before the choice, and the loser
    catches that subclass by name. The worker dies before the endings are saved. On the retry the
    gate runs again and re-derives its own refusal, type and all, so the loser takes the same path
    and the domain is never called for the refused op."""
    calls: list[str] = []

    def vetoing(op: Any) -> Any:
        if isinstance(op, Step) and getattr(op.op, "name", "") == "b0":
            raise _Vetoed(op, "vetoed")
        return (yield op)

    def loser():
        try:
            return (yield from call_tool("b0", {}, str))
        except _Vetoed:
            return "caught the veto"

    def program():
        return _described((yield from quorum(2, [_calls("a"), loser])))

    class _DiesAtTheEndings(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            if "endings" in name.stored():
                raise RuntimeError("the worker dies before the endings are saved")
            return self._ctx.settle(name, value)

    app, task = sqlite_app(), uuid4()

    def attempt(wrap: Callable[[Any], Any]) -> Any:
        tools = _Tools()
        ctx = wrap(SqliteTaskContext(app.conn, task, app.write_lock))
        try:
            return DurableHandler(ctx, tools, op_layers=[vetoing]).run(program)
        finally:
            calls.extend(tools.calls)

    with pytest.raises(RuntimeError, match="before the endings"):
        attempt(lambda ctx: _DiesAtTheEndings(ctx, threading.Event()))
    assert attempt(lambda ctx: ctx) == [
        "chosen",
        [["won", 0, "a"], ["won", 1, "caught the veto"]],
    ]
    assert calls == ["a"], "a tool the first attempt recorded, or the vetoed one, was called again"


def test_a_served_refusal_keeps_the_count_of_its_name(sqlite_app):
    """A branch calls `b`, is refused, and calls `b` again. On the retry the refusal is served
    from the record, and the engine still counts that first `b`, so the second is read from its
    own checkpoint and the domain is not called again."""
    app, task = sqlite_app(), uuid4()
    calls: list[str] = []

    asked: list[str] = []

    def refused_first() -> None:
        asked.append("b")
        if len(asked) == 1:
            gate = Step(name="b", op=CallTool(name="b", args={}, result_schema=str))
            raise Refused(gate, "not yet")

    def retrying():
        try:
            return (yield from call_tool("b", {}, str))
        except Refused:
            return (yield from call_tool("b", {}, str))

    def program():
        return _described((yield from race([retrying])))

    def attempt() -> Any:
        tools = _Tools({"b": refused_first})
        try:
            ctx = SqliteTaskContext(app.conn, task, app.write_lock)
            return DurableHandler(ctx, tools).run(program)
        finally:
            calls.extend(tools.calls)

    first = attempt()
    assert attempt() == first == ["chosen", [["won", 0, "b"]]]
    assert calls == ["b", "b"]


def test_a_gate_that_decides_differently_on_a_retry_fails_before_the_call(sqlite_app):
    """A gate refuses the loser's op on the first attempt and forwards it on the next. The
    record says the op was refused, so the retry fails, `Unretryable`, without the domain call a
    flagged loser would otherwise make after the choice."""
    app, task = sqlite_app(), uuid4()
    decisions = iter([True, False])
    calls: list[str] = []

    def flipping(op: Any) -> Any:
        if isinstance(op, Step) and getattr(op.op, "name", "") == "b0" and next(decisions):
            raise Refused(op, "no")
        return (yield op)

    def loser():
        try:
            return (yield from call_tool("b0", {}, str))
        except Refused:
            return (yield from call_tool("c", {}, str))

    def program():
        return _described((yield from quorum(2, [_calls("a"), loser])))

    def attempt() -> Any:
        tools = _Tools()
        try:
            ctx = SqliteTaskContext(app.conn, task, app.write_lock)
            return DurableHandler(ctx, tools, op_layers=[flipping]).run(program)
        finally:
            calls.extend(tools.calls)

    assert attempt() == ["chosen", [["won", 0, "a"], ["won", 1, "c"]]]
    with pytest.raises((Unretryable, ExceptionGroup)) as raised:
        attempt()
    assert unretryable(raised.value) is not None, raised.value
    assert "b0" not in calls


def test_an_op_a_layer_yields_runs_in_a_race_branch(sqlite_app):
    """A layer may yield an op of its own before forwarding; the walk never placed it, so it is
    run as it is outside a race, with no refusal record to read or write."""

    def announces(op: Any) -> Any:
        if isinstance(op, Step) and getattr(op.op, "name", "") == "a":
            yield Step(name="note", op=CallTool(name="note", args={}, result_schema=str))
        return (yield op)

    def program():
        return _described((yield from race([_calls("a")])))

    ctx = SqliteTaskContext(sqlite_app().conn, uuid4(), None)
    tools = _Tools()
    assert DurableHandler(ctx, tools, op_layers=[announces]).run(program) == [
        "chosen",
        [["won", 0, "a"]],
    ]
    assert tools.calls == ["note", "a"]


def test_a_transient_error_after_the_choice_retries_the_attempt(sqlite_app):
    """A loser's held op fails after the choice with an error a
    retry could clear, so the attempt fails at the barrier and saves no endings; the retry stops
    the loser at that op, which it never recorded."""
    app, task = sqlite_app(), uuid4()
    started, settled = threading.Event(), threading.Event()

    def fails_after_the_choice() -> None:
        started.set()
        settled.wait(10)
        raise TimeoutError("the provider did not answer")

    def attempt(tools: _Tools, wrap: Callable[[Any], Any]) -> Any:
        def program():
            return _described((yield from race([_calls("a"), _calls("b0")])))

        ctx = wrap(SqliteTaskContext(app.conn, task, app.write_lock))
        return DurableHandler(ctx, tools).run(program)

    held = _Tools({"a": lambda: started.wait(10), "b0": fails_after_the_choice})
    with pytest.raises(ExceptionGroup) as raised:
        attempt(held, lambda ctx: _Settled(ctx, settled))
    assert [type(leaf) for leaf in leaves(raised.value)] == [TimeoutError]
    ctx = SqliteTaskContext(app.conn, task, app.write_lock)
    assert ctx.peek_step(race_endings(0)) == (False, None)
    retry = _Tools()
    assert attempt(retry, lambda ctx: ctx) == ["chosen", [["won", 0, "a"], ["stopped", 1, None]]]
    assert retry.calls == []


def test_an_inner_races_save_in_flight_lands_before_the_outer_flag(sqlite_app):
    """The inner race has decided and its save is slow; the
    outer race decides meanwhile. The race tree's lock holds the outer race's save until the inner
    save is done, so an inner choice, when one is saved, is saved before the outer one."""
    inner_saving, outer_saved = threading.Event(), threading.Event()
    order: list[str] = []
    inner_choice = race_prefix(0, 1) + race_choice(0).stored()

    class _SlowInnerSave(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            if name.stored() == inner_choice:
                inner_saving.set()
                outer_saved.wait(0.5)  # returns at once without the lock; times out with it
            stored = self._ctx.settle(name, value)
            order.append("inner" if name.stored() == inner_choice else name.stored())
            if name == race_choice(0):
                outer_saved.set()
            return stored

    def inner():
        return (yield from race([_calls("p0"), _calls("q0")]))

    def program():
        return _described((yield from race([_calls("won"), inner])))

    tools = _Tools({"won": lambda: inner_saving.wait(10)})
    app = sqlite_app()
    ctx = _SlowInnerSave(SqliteTaskContext(app.conn, uuid4(), app.write_lock), threading.Event())
    DurableHandler(ctx, tools).run(program)
    choices = [name for name in order if name in ("inner", race_choice(0).stored())]
    assert choices[0] == "inner" or "inner" not in choices, choices


def test_a_viewer_reads_a_race_whose_branch_caught_a_domain_refusal(sqlite_app):
    """A branch's first call is refused by the domain and its second answered, under one name.
    A viewer over the tape passes the recorded refusal to the engine, which counts it, so the
    second call reads its own checkpoint and the view answers as the run did."""
    app, task = sqlite_app(), uuid4()
    asked: list[str] = []

    def refused_first() -> None:
        asked.append("r")
        if len(asked) == 1:
            gate = Step(name="r", op=CallTool(name="r", args={}, result_schema=str))
            raise Refused(gate, "not yet")

    def retrying():
        try:
            return (yield from call_tool("r", {}, str))
        except Refused:
            return (yield from call_tool("r", {}, str))

    def program():
        return _described((yield from race([retrying])))

    ctx = SqliteTaskContext(app.conn, task, app.write_lock)
    ran = DurableHandler(ctx, _Tools({"r": refused_first})).run(program)
    tape = {Key.parse(name) for name in keys(read_sqlite_conn(app.conn, task))}
    viewer = ViewingCtx(SqliteTaskContext(app.conn, task, app.write_lock), tape)
    assert DurableHandler(viewer, _Tools()).run(program) == ran


def test_a_gate_that_flips_after_a_crash_before_the_endings_fails_the_task(sqlite_app):
    """The worker dies before the endings are saved, and the gate then forwards what it refused.
    With no endings to rebuild, the divergence is the loser's error, and it fails the task at the
    barrier rather than becoming the loser's ending."""
    app, task = sqlite_app(), uuid4()
    decisions = iter([True, False])
    calls: list[str] = []

    def flipping(op: Any) -> Any:
        if isinstance(op, Step) and getattr(op.op, "name", "") == "b0" and next(decisions):
            raise Refused(op, "no")
        return (yield op)

    def loser():
        try:
            return (yield from call_tool("b0", {}, str))
        except Refused:
            return (yield from call_tool("c", {}, str))

    def program():
        return _described((yield from quorum(2, [_calls("a"), loser])))

    class _DiesAtTheEndings(_Settled):
        def settle(self, name: Key, value: Any) -> Any:
            if "endings" in name.stored():
                raise RuntimeError("the worker dies before the endings are saved")
            return self._ctx.settle(name, value)

    def attempt(wrap: Callable[[Any], Any]) -> Any:
        tools = _Tools()
        try:
            ctx = wrap(SqliteTaskContext(app.conn, task, app.write_lock))
            return DurableHandler(ctx, tools, op_layers=[flipping]).run(program)
        finally:
            calls.extend(tools.calls)

    with pytest.raises(RuntimeError, match="before the endings"):
        attempt(lambda ctx: _DiesAtTheEndings(ctx, threading.Event()))
    with pytest.raises(ExceptionGroup) as raised:
        attempt(lambda ctx: ctx)
    assert [type(leaf).__name__ for leaf in leaves(raised.value)] == ["RefusalDiverged"]
    assert "b0" not in calls


def _pure():
    """A body that returns without yielding an op."""
    return 7
    yield  # a generator, which never reaches its first yield


@pytest.mark.parametrize(
    "structure",
    ["scoped", "gather", "race"],
)
def test_a_flagged_loser_stops_at_a_structure(backend, structure):
    """The recorder's pin of the same name, run sequentially so the loser is flagged before its
    first op."""

    def loser():
        match structure:
            case "scoped":
                return (yield from scoped(Key.parse("s"), _pure))
            case "gather":
                return (yield from gather([_pure]))[0]
            case _:
                return (yield from race([_pure]))  # an inner race in a loser saves no choice

    def program():
        return _described((yield from race([_calls("a"), loser])))

    assert _run(backend, "sequential", program, _Tools()) == [
        "chosen",
        [["won", 0, "a"], ["stopped", 1, None]],
    ]


class _Veto(Refused):
    """A domain's own refusal type, which a record cannot rebuild."""


@pytest.mark.parametrize("shape", SHAPES)
def test_a_domain_refusal_subclass_calls_the_domain_again_on_a_retry(backend, shape):
    """A domain that refuses with its own subclass of `Refused` is called again on the next
    attempt, as it is outside a race; nothing records it as a gate's refusal."""

    class _Vetoing(_Tools):
        def run(self, op: DomainOp[Any]) -> Any:
            super().run(op)
            raise _Veto(Step(name="a", op=op), "no")

    def caught():
        try:
            return (yield from call_tool("a", {}, str))
        except _Veto:
            return "caught"

    def program():
        return _described((yield from race([caught])))

    def crash(ctx: Any, attempt: int) -> None:
        if attempt == 1:
            raise RuntimeError("the worker dies after the race")

    tools = _Vetoing()
    assert _run(backend, shape, program, tools, attempts=2, after=crash) == [
        "chosen",
        [["won", 0, "caught"]],
    ]
    assert tools.calls == ["a", "a"]


DEADLINE = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
"""The instant the deadline rows race against. A race reads one clock, `base.race_clock`, so a
row holds that clock rather than waiting for a real one."""


@pytest.mark.parametrize("shape", SHAPES)
def test_D26_a_deadline_already_behind_the_race_stops_it_before_a_branch_runs(
    backend, shape, monkeypatch
):
    """A race reads its deadline before it launches a branch, so one whose deadline is already
    gone stops every branch at its first admission and calls no tool. This is the restart case:
    a crash before the choice can leave the retry starting with the deadline long behind it."""
    monkeypatch.setattr(base, "race_clock", lambda: DEADLINE.timestamp() + 1.0)
    tools = _Tools()

    def program():
        return _described((yield from race([_calls("a"), _calls("b")], deadline=DEADLINE)))

    assert _run(backend, shape, program, tools) == [
        "timeout",
        [["stopped", 0, None], ["stopped", 1, None]],
    ]
    assert tools.calls == []


def test_D27_a_sequential_race_records_a_late_success_as_unchosen(backend, monkeypatch):
    """A branch already running when the deadline arrives finishes, which is the promise a
    sequential ctx can keep. Branch 0 moves the clock past the deadline on its way out, so its
    success is on record and late: unchosen rather than stopped, and branch 1 never starts."""
    held = {"at": (DEADLINE - timedelta(hours=1)).timestamp()}
    monkeypatch.setattr(base, "race_clock", lambda: held["at"])

    def past_the_deadline() -> object:
        held["at"] = DEADLINE.timestamp() + 1.0
        return None

    tools = _Tools({"a": past_the_deadline})

    def program():
        return _described((yield from race([_calls("a"), _calls("b")], deadline=DEADLINE)))

    assert _run(backend, "sequential", program, tools) == [
        "timeout",
        [["unchosen", 0, "a"], ["stopped", 1, None]],
    ]
    assert tools.calls == ["a"]


@pytest.mark.parametrize("shape", SHAPES)
def test_D28_a_retry_is_served_the_timeout_with_the_deadline_ahead_of_it(
    backend, shape, monkeypatch
):
    """The code-side tooth for `deadlineNotExtended`: the first attempt times out and dies, and
    the retry runs with the clock held an hour BEFORE the deadline. The stored choice stands, so
    a handler that re-decided would reopen a race the first attempt had closed."""
    held = {"at": DEADLINE.timestamp() + 1.0}
    monkeypatch.setattr(base, "race_clock", lambda: held["at"])

    def program():
        return _described((yield from race([_calls("a"), _calls("b")], deadline=DEADLINE)))

    def crash_once(ctx: Any, attempt: int) -> None:
        if attempt == 1:
            held["at"] = DEADLINE.timestamp() - 3600.0
            raise RuntimeError("the worker dies after the race")

    kind, endings = _run(backend, shape, program, _Tools(), attempts=2, after=crash_once)
    assert kind == "timeout"
    assert [ending for ending in endings if ending[0] == "won"] == []


@pytest.mark.parametrize("shape", SHAPES)
def test_D29_a_crash_before_the_choice_meets_its_deadline_already_gone(
    backend, shape, monkeypatch
):
    """The state a restart is really in. The first attempt starts with the deadline an hour ahead
    and dies before saving a choice, so nothing durable says how the race went; the retry starts
    with it an hour behind. The deadline is the workflow's own instant either way, so the retry
    times out at once and calls no tool, where D28 could have been served a stored choice."""
    held = {"at": (DEADLINE - timedelta(hours=1)).timestamp()}
    monkeypatch.setattr(base, "race_clock", lambda: held["at"])
    calls_by_attempt: dict[int, list[str]] = {}

    def die() -> object:
        held["at"] = (DEADLINE + timedelta(hours=1)).timestamp()
        raise ZeroDivisionError("a programming error before any choice")

    tools = _Tools({"a": die})

    def program():
        return _described((yield from race([_calls("a"), _calls("b")], deadline=DEADLINE)))

    def fresh(ctx: Any) -> Any:
        tools.calls.clear()  # `after` does not run on the attempt that raises, so clear here
        return ctx

    def record(ctx: Any, attempt: int) -> None:
        calls_by_attempt[attempt] = list(tools.calls)

    result = _run(backend, shape, program, tools, attempts=2, wrap=fresh, after=record)
    assert result == ["timeout", [["stopped", 0, None], ["stopped", 1, None]]]
    assert calls_by_attempt[2] == [], "the retry called a tool under a deadline already gone"
