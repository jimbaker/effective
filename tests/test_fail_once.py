"""A programming error fails its task on the attempt that raised it, on both engines.

A retry re-derives a programming error, so one among a task's errors is enough: the task fails
once, reported as that error, with the others as notes. A task that is only refused fails once
too. A runtime refusal beside a crash leaves the crash its retries, and a gather reports every
branch's error rather than the first to arrive.
"""

import ast
import threading
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from _conformance import Fault, private
from _durable import DSN, IMMEDIATE_RETRY, pg_ready

from effective.api import ask_llm, await_event, call_tool, gather
from effective.budget import MeasuredBudget
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, CallTool, DomainOp
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Run, compose_key
from effective.ops import CompositionRefused, leaves
from effective.spawning import (
    ChildAnswer,
    ChildFailed,
    ChildRefused,
    Returned,
    answer_parent,
    join_answer,
    run_child,
    spawn_child,
)


class Ordered:
    """Tools that answer or crash in a fixed order across concurrent branches: a `first` tool
    marks that it ran, and an `after` tool waits for that mark, so no timing is left to chance."""

    def __init__(self) -> None:
        self.first_ran = threading.Event()

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="first"):
                self.first_ran.set()
                return 1
            case CallTool(name="first_crash"):
                self.first_ran.set()
                raise ValueError("a crash")
            case CallTool(name="after"):
                assert self.first_ran.wait(30)
                return 1
            case CallTool(name="after_crash"):
                assert self.first_ran.wait(30)
                raise ValueError("a crash")
        raise TypeError(op)


def refusing(tool: str, error: Exception):
    def branch():
        yield from call_tool(tool, {}, int)
        raise error

    return branch


def crashing(tool: str):
    def branch():
        return (yield from call_tool(tool, {}, int))

    return branch


PROGRAMMING_ERROR_BESIDE_A_FASTER_CRASH = {
    "error first": lambda: [
        refusing("after", CompositionRefused("a programming error")),
        crashing("first_crash"),
    ],
    "crash first": lambda: [
        crashing("first_crash"),
        refusing("after", CompositionRefused("a programming error")),
    ],
}


@pytest.mark.parametrize("order", list(PROGRAMMING_ERROR_BESIDE_A_FASTER_CRASH))
def test_a_programming_error_beside_a_crash_fails_its_task_once(backend, order):
    """Reddens if the faster crash hides the programming error: the task must fail on its first
    attempt, reported as the programming error, whichever branch raised first."""
    name = private("mixed")
    branches = PROGRAMMING_ERROR_BESIDE_A_FASTER_CRASH[order]

    def workflow(_run_id: str):
        return (yield from gather(branches()))

    backend.register(name, workflow, Ordered(), Fault(), [])
    task = backend.spawn(name, str(uuid4()), max_attempts=3)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "failed", snapshot
    assert backend.failure_kind(snapshot) == "CompositionRefused"
    assert "ValueError" in str(snapshot.failure)  # the crash, kept as a note
    assert backend.task_attempts(task) == 1


REFUSAL_BESIDE_A_SLOWER_CRASH = {
    "refusal first": lambda: [
        refusing("first", ChildRefused("a refusal")),
        crashing("after_crash"),
    ],
    "crash first": lambda: [
        crashing("after_crash"),
        refusing("first", ChildRefused("a refusal")),
    ],
}


@pytest.mark.parametrize("order", list(REFUSAL_BESIDE_A_SLOWER_CRASH))
def test_a_refusal_beside_a_crash_leaves_the_crash_its_retries(backend, order):
    """Reddens if a faster refusal answers for a child whose other branch crashed: the crash
    might pass on a retry, so the child is retried to its limit and answers nobody meanwhile."""
    name = private("refused-beside")
    branches = REFUSAL_BESIDE_A_SLOWER_CRASH[order]

    def workflow(_params: dict[str, Any]):
        return (yield from gather(branches()))

    backend.register_child(name, str(uuid4()), workflow, Ordered(), Fault())
    task = backend.spawn(name, str(uuid4()), max_attempts=3)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "failed", snapshot
    assert backend.task_attempts(task) == 3


ONLY_REFUSED = {
    "bare": lambda: refusing("first", ChildRefused("a refusal"))(),
    "gathered": lambda: gather(
        [refusing("first", ChildRefused("one")), refusing("after", ChildRefused("two"))]
    ),
}


@pytest.mark.parametrize("shape", list(ONLY_REFUSED))
def test_a_task_that_is_only_refused_fails_once(backend, shape):
    """A refusal re-derives on a retry, so the attempt that raised it answers."""
    name = private("refused")

    def workflow(_run_id: str):
        return (yield from ONLY_REFUSED[shape]())

    backend.register(name, workflow, Ordered(), Fault(), [])
    task = backend.spawn(name, str(uuid4()), max_attempts=3)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "failed", snapshot
    assert backend.failure_kind(snapshot) == "ChildRefused"
    assert backend.task_attempts(task) == 1


def test_a_programming_error_in_a_nested_gather_fails_its_task_once(backend):
    """Reddens if an exception group is read one level deep: the leaf inside a nested gather is
    what decides, and what the task is reported as."""
    name = private("nested")

    def workflow(_run_id: str):
        inner = refusing("first", CompositionRefused("a programming error"))
        return (yield from gather([lambda: gather([inner, crashing("after")])]))

    backend.register(name, workflow, Ordered(), Fault(), [])
    task = backend.spawn(name, str(uuid4()), max_attempts=3)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "failed", snapshot
    assert backend.failure_kind(snapshot) == "CompositionRefused"
    assert backend.task_attempts(task) == 1


class OrderedResponses(Mapping[str, Any]):
    """The recorder's response table over `Ordered`'s tools."""

    def __init__(self) -> None:
        self.tools = Ordered()

    def __getitem__(self, name: str) -> Any:
        tool = name.rsplit(";", 1)[-1].removeprefix("tool:")
        return self.tools.run(CallTool(name=tool, args={}, result_schema=int))

    def __iter__(self) -> Iterator[str]:
        return iter(["tool:after", "tool:first_crash"])

    def __len__(self) -> int:
        return 2


def supervising(backend, child: str, max_attempts: int | None = None):
    """A parent task that spawns `child`, joins it, and returns what it heard."""

    def supervises():
        spawned = yield from spawn_child(child, "c", {}, max_attempts=max_attempts)
        try:
            return ["returned", (yield from join_answer(spawned))]
        except ChildFailed as failed:
            return ["failed", str(failed)]

    parent = private("parent")
    backend.register_body(
        parent,
        lambda params, ctx: DurableHandler(ctx, spawning(backend), params=params).run(supervises),
    )
    return parent


def spawning(backend) -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(backend.spawner)}),
    )


def heard(backend, parent: str, child: str) -> Any:
    """Run `parent` until it parks on its child, run the child to its end, resume the parent."""
    parent_id = backend.spawn(parent, str(uuid4()))
    backend.run_until_result(parent_id)
    [(child_id, _params)] = backend.enqueued(child)
    ended = backend.run_until_result(child_id)
    assert ended is not None
    assert ended.state in ("completed", "failed"), ended
    snapshot = backend.run_until_result(parent_id)
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    return snapshot.result


def test_a_child_that_crashes_on_every_attempt_answers_its_parent_failed(backend):
    """Reddens if a task body that never answers its parent itself leaves the parent waiting once
    its last attempt crashes: the worker that fails the task answers for it."""
    child = private("crashes")

    def body(params, ctx):
        raise ValueError("a crash on every attempt")

    backend.register_body(child, body)
    kind, message = heard(backend, supervising(backend, child), child)

    assert kind == "failed"
    assert "ValueError: a crash on every attempt" in message


def test_a_child_whose_measured_budget_trips_answers_its_parent_refused(backend):
    """Reddens if a spend ceiling reached in a child is reported as a crash: a trip re-derives on
    every retry, so the child completes on its first attempt and its parent hears a refusal."""
    child, parent = private("overspends"), private("parent")

    def overspends():
        for n in range(3):
            yield from ask_llm(f"ask-{n}", "x", str)

    def child_body(params, ctx):
        budget = MeasuredBudget(overall=0.0015, run_id=params["run_id"], on_exhaust="fail")
        domain = MeteredInterpreter(llm=lambda _op: ("", Usage(cost=0.001)), tools=lambda _op: 0)
        handler = DurableHandler(ctx, domain, params=params, budget=budget, contract=Contract.V1)
        return run_child(ctx, params, handler, overspends)

    def supervises():
        spawned = yield from spawn_child(child, "c", {"run_id": "overspent"})
        try:
            return ["returned", (yield from join_answer(spawned))]
        except ChildRefused as refused:
            return ["refused", str(refused)]

    backend.register_body(child, child_body)
    backend.register_body(
        parent,
        lambda params, ctx: DurableHandler(ctx, spawning(backend), params=params).run(supervises),
    )
    kind, message = heard(backend, parent, child)

    assert kind == "refused"
    assert "BudgetRefused: measured budget exceeded" in message
    [(child_id, _params)] = backend.enqueued(child)
    assert backend.task_attempts(child_id) == 1


def test_a_crash_on_an_earlier_attempt_does_not_answer_its_parent(backend):
    """Reddens if a crash that a retry recovers from is answered: the parent must hear the value
    the retry returned, and the done event takes only its first answer."""
    child = private("recovers")

    def body(params, ctx):
        if ctx.attempt.number == 1:
            raise ValueError("a crash the retry recovers from")
        answer_parent(ctx, params, ChildAnswer(answer=Returned(value="recovered")).model_dump())
        return "recovered"

    backend.register_body(child, body)

    assert heard(backend, supervising(backend, child), child) == ["returned", "recovered"]


def test_a_child_parked_on_its_last_attempt_answers_its_value_once_woken(backend):
    """Reddens if a park is taken for a failure: a child on its only attempt that waits for an
    event is suspended, and answers its parent with the value it returns once the event lands."""
    child, wake = private("parks"), compose_key(t"wake:{Run(str(uuid4()))}")

    def body(params, ctx):
        def waits():
            return (yield from await_event(wake, dict))

        payload = DurableHandler(ctx, spawning(backend), params=params).run(waits)
        answer_parent(ctx, params, ChildAnswer(answer=Returned(value=payload)).model_dump())
        return payload

    backend.register_body(child, body)
    parent = supervising(backend, child, max_attempts=1)
    parent_id = backend.spawn(parent, str(uuid4()))
    backend.run_until_result(parent_id)
    [(child_id, _params)] = backend.enqueued(child)
    parked = backend.run_until_result(child_id)
    assert parked is not None
    assert parked.state not in ("completed", "failed"), parked
    backend.emit_event(child_id, wake, {"woke": True})
    backend.run_until_result(child_id)
    snapshot = backend.run_until_result(parent_id)

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == ["returned", {"woke": True}]


def notes_heard(backend, child: str) -> list[str]:
    """What a parent that joined `child` on its only attempt heard beside the failure: the notes on
    the `ChildFailed` it caught."""

    def supervises():
        spawned = yield from spawn_child(child, "c", {}, max_attempts=1)
        try:
            yield from join_answer(spawned)
        except ChildFailed as failed:
            return [str(failed), *getattr(failed, "__notes__", ())]
        return None

    parent = private("parent")
    backend.register_body(
        parent,
        lambda params, ctx: DurableHandler(ctx, spawning(backend), params=params).run(supervises),
    )
    parent_id = backend.spawn(parent, str(uuid4()))
    backend.run_until_result(parent_id)
    [(child_id, _params)] = backend.enqueued(child)
    backend.run_until_result(child_id)
    snapshot = backend.run_until_result(parent_id)
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    return snapshot.result


BESIDE_A_CRASH = {
    # (the branches, what the child failed of, what else it raised)
    "a refusal first": (
        lambda: [refusing("first", ChildRefused("a refusal")), crashing("after_crash")],
        "ValueError: a crash",
        "ChildRefused: a refusal",
    ),
    "a refusal second": (
        lambda: [crashing("first_crash"), refusing("after", ChildRefused("a refusal"))],
        "ValueError: a crash",
        "ChildRefused: a refusal",
    ),
    "a programming error": (
        lambda: [
            refusing("first", CompositionRefused("a programming error")),
            crashing("after_crash"),
        ],
        "CompositionRefused: a programming error",
        "ValueError: a crash",
    ),
}


@pytest.mark.parametrize("beside", list(BESIDE_A_CRASH))
def test_a_parent_hears_what_a_failed_child_failed_of_and_every_error_beside_it(backend, beside):
    """Reddens if a failed child is reported as anything but its cause, or an error beside the
    cause is lost: a programming error fails a child on its first attempt, and a crash fails it on
    its last where a refusal alone would have completed it, whichever branch raised each."""
    child = private("beside")
    branches, cause, beside_it = BESIDE_A_CRASH[beside]

    def workflow(_params: dict[str, Any]):
        return (yield from gather(branches()))

    backend.register_child(child, str(uuid4()), workflow, Ordered(), Fault())
    message, *notes = notes_heard(backend, child)

    assert message.endswith(f"failed of {cause}"), message
    assert notes == [beside_it]


@pytest.mark.skipif(not pg_ready(), reason="no Postgres with Absurd (just pgt-up)")
def test_an_absurd_worker_fails_and_answers_through_its_hook():
    """Reddens if the app every queue worker builds stops installing the hook that fails a task
    once and answers its parent."""
    from effective.absurd_worker import absurd_worker, fail_terminally

    app = absurd_worker(DSN)
    try:
        assert app._hooks["wrap_task_execution"] is fail_terminally
    finally:
        app.close()


def called_names(source: str) -> set[str]:
    """The name every call in `source` calls: a function's, or a method's attribute."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        match node:
            case ast.Call(func=ast.Name(id=name)) | ast.Call(func=ast.Attribute(attr=name)):
                names.add(name)
    return names


def test_every_process_that_works_an_absurd_queue_builds_its_app_through_absurd_worker():
    """Reddens if a module outside the tests builds an `Absurd(...)` and works a queue with it:
    that worker would retry a programming error and answer no parent. The census must see a
    shipped worker, or it proves nothing, and it names the one file this interpreter cannot
    read."""
    workers, plain, unread = [], [], []
    for root in ("src", "scripts", "examples"):
        for path in sorted(Path(root).rglob("*.py")):
            if path.name == "absurd_worker.py":
                continue
            try:
                calls = called_names(path.read_text(encoding="utf-8"))
            except SyntaxError:
                unread.append(str(path))
                continue
            if calls & {"work_batch", "start_worker"}:
                (plain if "Absurd" in calls else workers).append(str(path))

    assert "examples/smol_durable.py" in workers
    assert plain == []
    assert unread == []


def test_the_recorder_reports_every_gather_branchs_error():
    """Reddens if the infra-free harness drops a slower branch's error: an author developing
    against the recorder must see the programming error the engines fail the task of."""
    handler = RecordingHandler(responses=OrderedResponses())

    def workflow():
        return (
            yield from gather(
                [
                    refusing("after", CompositionRefused("a programming error")),
                    crashing("first_crash"),
                ]
            )
        )

    with pytest.raises(ExceptionGroup) as raised:
        handler.run(workflow)

    assert [type(leaf).__name__ for leaf in leaves(raised.value)] == [
        "CompositionRefused",
        "ValueError",
    ]


@pytest.mark.skipif(not pg_ready(), reason="no Postgres with Absurd (just pgt-up)")
def test_a_run_whose_lease_expired_cannot_spend_its_successors_attempts():
    """Reddens if a superseded run's programming error lowers the task's limit: its successor,
    claimed after the lease expired, crashes and must keep its retry.

    Run A claims and waits; run B's worker, its clock past A's lease, sweeps it and claims the next
    attempt; A then raises a programming error while B runs, and B crashes once A is done."""
    from effective.absurd_worker import absurd_worker

    queue, name = f"q{uuid.uuid4().hex[:8]}", private("leased")
    tasks = f"t_{queue}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("SELECT absurd.create_queue(%s)", (queue,))
    # Any: the SDK types `retry_strategy` strictly, and the dict-shaped IMMEDIATE_RETRY is the test
    # convention.
    first: Any = absurd_worker(DSN, queue_name=queue)
    second: Any = absurd_worker(DSN, queue_name=queue)
    a_running, b_running = threading.Event(), threading.Event()

    def run_a(params, ctx):
        a_running.set()
        assert b_running.wait(30)
        raise CompositionRefused("a programming error, raised after its lease expired")

    def run_b(params, ctx):
        b_running.set()
        a.join(30)
        raise ValueError("a crash on the successor's attempt")

    first.register_task(name)(run_a)
    second.register_task(name)(run_b)
    a = threading.Thread(target=lambda: first.work_batch(claim_timeout=30))
    try:
        spawned: Any = first.spawn(name, {}, max_attempts=3, retry_strategy=IMMEDIATE_RETRY)
        a.start()
        assert a_running.wait(30)
        # B's worker reads a clock past A's lease; Absurd's claim sweep reads it from the session.
        past_the_lease = datetime.now(UTC) + timedelta(minutes=5)
        second._conn.execute(
            "SELECT set_config('absurd.fake_now', %s, false)", (past_the_lease.isoformat(),)
        )
        for _ in range(5):  # the sweep that fails A's run need not claim its successor
            second.work_batch(claim_timeout=30)
            if b_running.is_set():
                break
        a.join(30)
        with psycopg.connect(DSN) as conn:
            row = conn.execute(
                t"SELECT state, attempts, max_attempts FROM absurd.{tasks:i} "
                t"WHERE task_id = {spawned['task_id']}"
            ).fetchone()

        assert b_running.is_set()
        assert row == ("pending", 3, 3)
    finally:
        b_running.set()
        if a.ident is not None:
            a.join(30)
        first.close()
        second.close()
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("SELECT absurd.drop_queue(%s)", (queue,))
