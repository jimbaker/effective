"""A handler runs under the format its task's record marks, however the handler is built.

| question | the pin                                                                          |
|----------|----------------------------------------------------------------------------------|
| build    | every way of building a handler refuses a run whose record marks no format       |
| store    | a store error while reading the format keeps its kind                            |
| reads    | a handler reads its run's format once, and a gather branch reads none            |
| fork     | a fork whose base or own record is unread is refused before it writes            |
| handed   | no argument to a handler names a format                                          |
| decode   | a recorded entry decodes at its op                                               |
"""

import ast
import inspect
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints
from uuid import UUID, uuid4

import pytest
from _conformance import PG_DSN, private
from _steps import placed_step
from psycopg.types.json import Jsonb

from effective import bridge_sqlite
from effective.api import ask_llm, await_event, call_tool, gather
from effective.budget import MeasuredBudget, as_policy
from effective.checkpoints import Checkpoint
from effective.contexts import LocalCtx
from effective.cost import (
    FORMAT_PARAM,
    NEWEST_FORMAT,
    MeteredInterpreter,
    RecordedFormat,
    Usage,
    marked,
)
from effective.domain import AskLLM, CallTool
from effective.engines import absurd as absurd_engine
from effective.engines import sqlite as sqlite_engine
from effective.engines.absurd import queue_table
from effective.engines.sqlite import SqliteLedger, SqliteTaskContext
from effective.fork import (
    Bridged,
    Driven,
    MeteredEntry,
    decode_checkpoint,
    fork_seed,
    run_fork,
    run_fork_as_task,
)
from effective.govern import govern
from effective.handlers.durable import DurableHandler, fork_event_name
from effective.keys import Key, compose_key
from effective.keys.processor import gather_prefix, race_prefix
from effective.ops import DONE_EVENT_PARAM, UnreadFormat
from effective.spawning import ChildAnswer, Failed

PREFIX = placed_step("prefix", "llm")
"""The key of the model call every workflow here makes first."""

COLLISION = {"result": {"amount": 1}, "usage": {"cost": 7}, "business": "keep-me"}
"""A model call's bare result, as a release before the format recorded it, shaped like an
envelope."""


def _unmarked(backend, name: str) -> UUID:
    """A task spawned as a release before the format spawned it: its params mark nothing."""
    if backend.name == "postgres":
        return UUID(str(backend.app.spawn(name, {}, queue=backend.engine.queue)["task_id"]))
    task = backend.engine.spawn(name, {})
    backend.app.conn.execute("UPDATE tasks SET params='{}' WHERE task_id=?", (task,))
    return task


class _Forwarding:
    """A ctx forwarding `TaskContext`'s other members, and not the format."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        return self._inner.step(name, thunk)

    def await_event(self, name: Key, /) -> Any:
        return self._inner.await_event(name)

    def sleep_until(self, when: Any, /, *, name: Key) -> None:
        return self._inner.sleep_until(when, name=name)

    def clock(self) -> float:
        return self._inner.clock()


type Build = Callable[[dict[str, Any], Any, list[str]], DurableHandler]

BUILDS: dict[str, Build] = {
    "ctx alone": lambda params, ctx, calls: DurableHandler(ctx, _domain(calls)),
    "its params": lambda params, ctx, calls: DurableHandler(ctx, _domain(calls), params=params),
    "a forwarding wrapper": lambda params, ctx, calls: DurableHandler(
        _Forwarding(ctx),  # ty: ignore[invalid-argument-type] - the wrapper an untyped caller builds
        _domain(calls),
    ),
    "a body that marks its params": lambda params, ctx, calls: DurableHandler(
        ctx, _domain(calls), params=_marked_in_place(params)
    ),
}


def _marked_in_place(params: dict[str, Any]) -> dict[str, Any]:
    params[FORMAT_PARAM] = NEWEST_FORMAT.value
    return params


def _domain(calls: list[str]) -> MeteredInterpreter:
    def unasked(op: Any) -> Any:
        raise AssertionError(f"asked {op!r}")

    return MeteredInterpreter(llm=unasked, tools=lambda op: calls.append(op.name) or "ok")


def _resumed(backend, build: Build, *, deployed: bool) -> tuple[Any, list[str], bool]:
    """An unmarked run parked after its model call, resumed by a handler built as `build`:
    its snapshot, the tools it called, and whether its checkpoints stayed as they were."""
    name, wake = private(f"record-{uuid4().hex[:8]}"), f"wake-{uuid4().hex}"
    calls: list[str] = []

    def legacy(params: dict[str, Any], ctx: Any) -> Any:
        ctx.step(Key.parse(PREFIX), lambda: COLLISION)
        return ctx.await_event(Key.parse(wake))

    def workflow() -> Any:
        found = yield from ask_llm("prefix", "x", dict)
        yield from await_event(wake, dict)
        yield from call_tool("effect", {}, str)
        return found

    backend.register_body(name, legacy, deployed=deployed)
    task = _unmarked(backend, name)
    assert backend.engine.run_until_result(task).state == "waiting"
    backend.register_body(
        name,
        lambda params, ctx: build(params, ctx, calls).run(workflow),
        deployed=deployed,
    )
    backend.engine.emit_event(wake, {})
    before = backend.checkpoint_states(task)  # the engine's delivery writes the wait's own
    snapshot = backend.engine.run_until_result(task)
    return snapshot, calls, backend.checkpoint_states(task) == before


def _adapters(backend) -> list[bool]:
    """The ctxs a body sees: on Absurd the concurrent adapter and the deployed one."""
    return [False, True] if backend.name == "postgres" else [False]


@pytest.mark.adversarial
def test_every_build_of_a_handler_refuses_a_run_whose_record_marks_no_format(backend):
    accepted = []
    for deployed in _adapters(backend):
        for build in BUILDS:
            snapshot, calls, kept = _resumed(backend, BUILDS[build], deployed=deployed)
            refused = snapshot.failure and snapshot.failure.kind
            if refused != "UnreadFormat" or calls or not kept:
                accepted.append((build, deployed, snapshot.state, calls, kept))
    assert accepted == [], "\n".join(map(str, accepted))


class _Unreadable(LocalCtx):
    def recorded_format(self) -> RecordedFormat:
        raise OSError("the store is unreachable")


def test_a_context_that_cannot_answer_its_format_is_refused_by_name_once(backend):
    name, attempts = private(f"forwarding-{uuid4().hex[:8]}"), []

    def body(params: dict[str, Any], ctx: Any) -> Any:
        attempts.append(1)
        return BUILDS["a forwarding wrapper"](params, ctx, []).run(_branches)

    backend.register_body(name, body, deployed=True)
    snapshot = backend.engine.run_until_result(backend.engine.spawn(name, {}))
    kind = snapshot.failure and snapshot.failure.kind
    assert (snapshot.state, kind, len(attempts)) == ("failed", "UnreadFormat", 1), snapshot


def test_a_store_error_while_reading_the_format_keeps_its_kind():
    with pytest.raises(OSError, match="unreachable"):
        DurableHandler(_Unreadable(), _domain([]))


def _branches():
    def branch(name: str):
        def thunk():
            return (yield from call_tool(name, {}, str))

        return thunk

    return (yield from gather([branch("a"), branch("b")]))


def test_a_run_reads_its_format_from_the_store_once_and_a_branch_never(backend, monkeypatch):
    reads: Counter[str] = Counter()
    for module, reader in (
        (sqlite_engine, "read_sqlite_params"),
        (absurd_engine, "read_absurd_params"),
    ):
        read = getattr(module, reader)

        def counted(*args: Any, _read: Any = read, _reader: str = reader) -> Any:
            reads[_reader] += 1
            return _read(*args)

        monkeypatch.setattr(module, reader, counted)
    name, calls = private(f"reads-{uuid4().hex[:8]}"), []
    backend.register_body(
        name, lambda params, ctx: DurableHandler(ctx, _domain(calls)).run(_branches)
    )
    snapshot = backend.engine.run_until_result(backend.engine.spawn(name, {}))
    assert (snapshot.state, sorted(calls)) == ("completed", ["a", "b"]), snapshot
    assert sum(reads.values()) == 1, reads


# --- a fork reads its base's format before it writes ---------------------------------------


def _delivered(backend, name: str) -> Any:
    """The payload delivered under event `name`, or `None` if none was."""
    if backend.name == "postgres":
        import psycopg

        events = queue_table("e", backend.engine.queue)
        with psycopg.connect(PG_DSN) as conn:
            row = conn.execute(
                t"SELECT payload FROM absurd.{events:i} WHERE event_name = {name}"
            ).fetchone()
        return None if row is None else row[0]
    row = backend.app.conn.execute("SELECT payload FROM events WHERE name=?", (name,)).fetchone()
    return None if row is None else json.loads(row[0])


def _hypothetical(backend, child: str) -> Any:
    if backend.name == "postgres":
        from effective.ledger import PostgresLedger

        return PostgresLedger(PG_DSN, workflow_run_id=child, hypothetical=True)
    return SqliteLedger(backend.app.conn, child, backend.app.write_lock, hypothetical=True)


UNREAD_BASES: dict[str, tuple[dict[str, Any], Any]] = {
    "collision seed": ({}, COLLISION),
    "bare seed": ({}, {"amount": 1}),
    "unknown mark": ({FORMAT_PARAM: "v0"}, COLLISION),
}
"""A base whose record this release does not read: its params, and its model call's state."""


@dataclass(frozen=True)
class Fork:
    """A fork of `base` from `UNREAD_BASES`, or of a v1 base when `base` is empty."""

    base: str
    delta: dict[str, Any] | None
    direct: bool = False
    task_marked: bool = True
    point: str = "review"
    two_grant_books: bool = False
    """A handler budget beside a budget policy, which the handler refuses when it is built."""


def _forked(backend, fork: Fork) -> list[str]:
    """What `fork` did that a refusal does not."""
    base, delta, direct = fork.base, fork.delta, fork.direct
    base_params, raw = (
        UNREAD_BASES[base]
        if base
        else (
            marked({}),
            {
                "result": {"amount": 1},
                "usage": {"cost": 0.0},
            },
        )
    )
    name, child, done = private(f"fork-{uuid4().hex[:8]}"), f"c-{uuid4().hex[:8]}", uuid4().hex
    seed = [Checkpoint(Key.parse(PREFIX), raw)]

    def workflow(_run_id: str) -> Any:
        found = yield from ask_llm("prefix", "x", dict)
        yield from await_event("review", dict)
        return found

    ceiling = MeasuredBudget(overall=1.0, run_id=child) if fork.two_grant_books else None
    layers = () if ceiling is None else (govern(as_policy(ceiling), gate="spend", run_id=child),)

    def body(params: dict[str, Any], ctx: Any) -> Any:
        ledger = _hypothetical(backend, child)
        try:
            if direct:
                return run_fork(
                    ctx,
                    lambda: workflow("base"),
                    child_run_id=child,
                    seed=fork_seed(seed, through=PREFIX),
                    fork_point=Key.parse(fork.point),
                    hypothetical_ledger=ledger,
                    domain=_domain([]),
                    forked_from="base",
                    forked_at_event="review",
                    base_params=base_params,
                    op_layers=layers,
                    budget=ceiling,
                )
            return run_fork_as_task(
                params,
                ctx,
                workflow=workflow,
                domain=_domain([]),
                hypothetical_ledger=lambda _: ledger,
                read_base=lambda _: seed,
                read_base_params=lambda _: base_params,
                op_layers=layers,
                budget=ceiling,
            )
        finally:
            if backend.name == "postgres":
                ledger.close()

    backend.register_body(name, body, deployed=True)
    params = {
        "child_run_id": child,
        "forked_from": "base",
        "base_task_id": str(uuid4()),
        "through": PREFIX,
        "fork_point": fork.point,
        "forked_at_event": "review",
        "delta": delta,
        DONE_EVENT_PARAM: done,
    }
    if fork.task_marked:
        task = backend.engine.spawn(name, params)
    else:
        task = _unmarked(backend, name)
        _restamp(backend, task, params)
    snapshot = backend.engine.run_until_result(task)
    done_answer = _delivered(backend, done)
    answer_at = fork_event_name(child, Key.parse(fork.point))
    effects = [
        "completed" if snapshot.state != "failed" else "",
        "delta" if _delivered(backend, answer_at.stored()) is not None else "",
        "emit"
        # lint: terminal-hole: `answer_at` is the `Key` `fork_event_name` mints, composed as
        # the fork composes its delta's checkpoint.
        if compose_key(t"emit;{answer_at:domain=address}") in backend.checkpoint_states(task)
        else "",
        "ledger" if backend.hypothetical_payloads(child) else "",
        "answer"
        if done_answer is not None
        and not isinstance(ChildAnswer.model_validate(done_answer).answer, Failed)
        else "",
    ]
    return [effect for effect in effects if effect]


def _restamp(backend, task: UUID, params: dict[str, Any]) -> None:
    """Store `params`, unmarked, as the params `task` was spawned with."""
    if backend.name == "postgres":
        import psycopg

        tasks = queue_table("t", backend.engine.queue)
        with psycopg.connect(PG_DSN) as conn:
            conn.execute(
                t"UPDATE absurd.{tasks:i} SET params = {Jsonb(params)} WHERE task_id = {task}"
            )
        return
    backend.app.conn.execute(
        "UPDATE tasks SET params=? WHERE task_id=?", (json.dumps(params), task)
    )


FORKS = [
    *(Fork(base, delta) for base in UNREAD_BASES for delta in ({"decision": "approve"}, None)),
    Fork("", {"decision": "approve"}, task_marked=False),
    Fork("", None, direct=True, task_marked=False),
    *(Fork(base, None, direct=True) for base in UNREAD_BASES),
]


@pytest.mark.adversarial
def test_a_fork_of_a_run_whose_format_is_unread_is_refused_before_it_writes(backend):
    wrote = [(fork, effects) for fork in FORKS if (effects := _forked(backend, fork))]
    assert wrote == [], "\n".join(map(str, wrote))


REGIONS = [
    Fork("", delta, point=prefix(0, 0) + "review")
    for prefix in (gather_prefix, race_prefix)
    for delta in ({"decision": "approve"}, None)
]


@pytest.mark.adversarial
def test_a_fork_point_inside_a_region_is_refused_before_the_fork_writes(backend):
    """The refusal is the fork's answer; what it may not do first is write."""
    wrote = [
        (fork, writes)
        for fork in REGIONS
        if (writes := [e for e in _forked(backend, fork) if e in ("delta", "emit", "ledger")])
    ]
    assert wrote == [], "\n".join(map(str, wrote))


def test_a_handler_is_handed_no_format():
    """Its run's record answers the format, so no argument a caller passes names one."""
    hints = get_type_hints(DurableHandler.__init__)
    assert [
        name for name, hint in hints.items() if RecordedFormat in (hint, *get_args(hint))
    ] == []


ENVELOPE = {"result": "r", "usage": {"cost": 0.5}}

DECODED: dict[str, tuple[MeteredEntry, AskLLM[Any] | CallTool[Any], tuple[Any, Usage]]] = {
    "driven": (
        Driven(Key.parse(placed_step("a", "tool")), result="r", usage=Usage(cost=0.5)),
        CallTool("a", dict),
        ("r", Usage(cost=0.5)),
    ),
    "bridged model call": (
        Bridged(Key.parse(placed_step("a", "llm")), ENVELOPE),
        AskLLM([], str),
        ("r", Usage(cost=0.5)),
    ),
    "bridged tool call": (
        Bridged(Key.parse("tool:a"), ENVELOPE),
        CallTool("a", dict),
        (ENVELOPE, Usage()),
    ),
}
"""An entry, the op it is decoded at, and what it decodes to."""


@pytest.mark.parametrize("entry", DECODED)
def test_an_entry_decodes_at_its_op(entry):
    """A bridged checkpoint is an envelope only at a model call: the op decides, not the shape."""
    recorded, op, decoded = DECODED[entry]
    assert decode_checkpoint(op, recorded) == decoded


def _heard_at_the_point(backend, *, direct: bool) -> set[str]:
    """The answers a fork of a v1 base delivering `approve` hears at its point, when `reject` is
    also emitted there once the fork has run. A replay hears its answer again, so each answer
    counts once."""
    name, child, heard = private(f"answer-{uuid4().hex[:8]}"), f"c-{uuid4().hex[:8]}", []
    seed = [Checkpoint(Key.parse(PREFIX), {"result": {}, "usage": {"cost": 0.0}})]

    def workflow(_run_id: str) -> Any:
        yield from ask_llm("prefix", "x", dict)
        heard.append((yield from await_event("review", dict)))

    def body(params: dict[str, Any], ctx: Any) -> Any:
        ledger = _hypothetical(backend, child)
        try:
            if direct:
                return run_fork(
                    ctx,
                    lambda: workflow("base"),
                    child_run_id=child,
                    seed=fork_seed(seed, through=PREFIX),
                    fork_point=Key.parse("review"),
                    hypothetical_ledger=ledger,
                    domain=_domain([]),
                    forked_from="base",
                    forked_at_event="review",
                    delta={"decision": "approve"},
                    base_params=marked({}),
                )
            return run_fork_as_task(
                params,
                ctx,
                workflow=workflow,
                domain=_domain([]),
                hypothetical_ledger=lambda _: ledger,
                read_base=lambda _: seed,
                read_base_params=lambda _: marked({}),
            )
        finally:
            if backend.name == "postgres":
                ledger.close()

    backend.register_body(name, body, deployed=True)
    task = backend.engine.spawn(
        name,
        {
            "child_run_id": child,
            "forked_from": "base",
            "base_task_id": str(uuid4()),
            "through": PREFIX,
            "fork_point": "review",
            "forked_at_event": "review",
            "delta": {"decision": "approve"},
            DONE_EVENT_PARAM: uuid4().hex,
        },
    )
    backend.engine.run_until_result(task)
    backend.engine.emit_event(fork_event_name(child, Key.parse("review")), {"decision": "reject"})
    backend.engine.run_until_result(task)
    return {answer["decision"] for answer in heard}


def test_a_fork_answers_its_point_with_its_delta_alone(backend):
    """The delta the genesis records is the answer the child hears, whoever emits after it."""
    heard = {direct: _heard_at_the_point(backend, direct=direct) for direct in (False, True)}
    assert heard == {direct: {"approve"} for direct in (False, True)}


REFUSED_WHEN_BUILT = [
    Fork("", delta, direct=direct, two_grant_books=True)
    for direct in (False, True)
    for delta in ({"decision": "approve"}, None)
]


@pytest.mark.adversarial
def test_a_fork_whose_handler_refuses_when_built_writes_nothing(backend):
    wrote = [(fork, effects) for fork in REFUSED_WHEN_BUILT if (effects := _forked(backend, fork))]
    assert wrote == [], "\n".join(map(str, wrote))


def test_a_sqlite_context_reads_only_the_store_it_was_built_for():
    """Its format is read once, so the store it reads is fixed with it."""
    match inspect.getattr_static(SqliteTaskContext, "conn", None):
        case property(fset=None):
            pass
        case bound:
            pytest.fail(f"a SQLite context's store can be reassigned through {bound!r}")


def test_a_sqlite_context_reads_only_the_task_it_was_built_for():
    """Its format is read once, so the task it reads is fixed with it."""
    match inspect.getattr_static(SqliteTaskContext, "task_id", None):
        case property(fset=None):
            pass
        case bound:
            pytest.fail(f"a SQLite context's task can be reassigned through {bound!r}")


def _parked_base(backend, *, marked: bool) -> UUID:
    name, wake = private(f"measured-{uuid4().hex[:8]}"), f"wake-{uuid4().hex}"

    def legacy(params: dict[str, Any], ctx: Any) -> Any:
        ctx.step(Key.parse(PREFIX), lambda: COLLISION)
        return ctx.await_event(Key.parse(wake))

    backend.register_body(name, legacy, deployed=True)
    task = backend.engine.spawn(name, {}) if marked else _unmarked(backend, name)
    assert backend.engine.run_until_result(task).state == "waiting"
    return task


def _exported(backend, task: UUID) -> Sequence[MeteredEntry]:
    if backend.name == "postgres":
        import psycopg

        from effective import bridge_absurd

        with psycopg.connect(PG_DSN) as conn:
            return bridge_absurd.export_measured_prefix(
                conn, task, "r", queue=backend.engine.queue
            )[0]
    return bridge_sqlite.export_measured_prefix(backend.app.conn, str(task), "r")[0]


def test_a_measured_export_refuses_a_base_whose_format_is_unread(backend):
    entries = _exported(backend, _parked_base(backend, marked=True))
    assert [e.key.stored() for e in entries] == [PREFIX]
    with pytest.raises(UnreadFormat):
        _exported(backend, _parked_base(backend, marked=False))


# --- a producer outside the engines marks what it spawns ------------------------------------


def unmarked_spawns(path: str) -> list[int]:
    """The lines of `path` that hand a raw Absurd app's `spawn` params not wrapped in `marked`."""
    unmarked = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        match node:
            case ast.Call(func=ast.Attribute(attr="spawn"), args=[_, params, *_]):
                match params:
                    case ast.Call(func=ast.Name(id="marked")):
                        pass
                    case _:
                        unmarked.append(node.lineno)
    return unmarked


def test_the_smoke_script_marks_the_run_it_spawns():
    assert unmarked_spawns("scripts/absurd_smoke.py") == []
