"""A spawn's name to the engine: one child per placed spawn, on both engines.

The engine enqueues at most once per idempotency key. So a crash between the enqueue and the
spawn's checkpoint must retry under the same key, and two spawns placed apart must never send
one key: the second would be handed the first one's child and its own params dropped.
"""

import uuid
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultCtx, FaultPosition, private

from effective.api import call_tool, gather, scoped
from effective.budget import BUDGET_DEPTH_PARAM
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.compose import spawn_subagent_task
from effective.contexts import LocalCtx
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, SpawnArgs, Spawned, SpawnResult
from effective.fork import spawn_fork
from effective.govern import Refused
from effective.handlers.absurd import DurableHandler, respawn_name, spawned_name
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Key, compose_key
from effective.ops import Writer
from effective.spawning import (
    ChildAnswer,
    ChildFailed,
    Failed,
    answer_parent,
    join_answer,
    join_child,
    run_child,
    spawn_child,
)
from effective.viewing import ViewingCtx

pytestmark = pytest.mark.adversarial


def spawning(backend, spawner=None) -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner or backend.spawner)}),
    )


def completes(backend, task_id: Any) -> None:
    snapshot = backend.run_until_result(task_id)
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot


def child(params, ctx):
    """A child that answers the parent waiting on it through the substrate's child side."""
    answer_parent(ctx, params, {"answer": "ok"})
    return "ok"


def subagent(child_task: str, correlation: str):
    return spawn_subagent_task(child_task, "go", correlation=correlation)


def fork_child(child_task: str, child_run_id: str):
    return spawn_fork(
        child_task,
        child_run_id=child_run_id,
        base_task_id=uuid.uuid4(),
        through="ledger;extracted:m1",
        fork_point=Key.parse("review:m1"),
        forked_from="r-base",
        forked_at_event="extracted:m1",
        delta={},
    )


def after_the_spawn(on_name: str) -> Fault:
    """One crash, after the spawn lands and before its checkpoint commits."""
    return Fault(on_name=on_name, position=FaultPosition.AFTER_THUNK)


def test_a_spawn_that_lands_before_its_checkpoint_enqueues_one_child(backend):
    """Reddens if a retried spawn is named differently from the attempt that crashed."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, child)
    fault = after_the_spawn("tool:spawn")
    correlation = str(uuid4())
    backend.register(
        parent, lambda _rid: subagent(child_task, correlation), spawning(backend), fault, layers=[]
    )

    completes(backend, backend.spawn(parent, "r-parent"))

    assert not fault.armed, "the crash never fired, so this pinned nothing"
    assert len(backend.enqueued(child_task)) == 1


def test_a_respawn_that_lands_before_its_checkpoint_enqueues_one_successor(backend):
    """Reddens if a generation that crashed after spawning its successor spawns another, or names
    its successor by anything but its own task and the generation cut."""
    chain_task, run = private("chain"), str(uuid4())
    fault = after_the_spawn("respawn:")
    names: list[str] = []

    def spawner(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        names.append(idempotency_key)
        return backend.spawner(
            task_name, params, idempotency_key, queue, max_attempts=max_attempts
        )

    def step(state: int, turn: Turn):
        yield from ()
        return Done("finished") if turn.generation >= 1 else Again(state + 1)

    def body(params, ctx):
        carried = Chain.from_params(params, task=chain_task, schema=int, initial=0, run_id=run)
        chain = Chain(
            task=chain_task, state=carried.state, run_id=run, generation=carried.generation
        )
        return DurableHandler(FaultCtx(ctx, fault), spawning(backend, spawner), params=params).run(
            lambda: respawn(step, chain)
        )

    backend.register_body(chain_task, body)
    first = backend.spawn(chain_task, "r-chain")
    completes(backend, first)
    for successor, _params in backend.enqueued(chain_task)[1:]:
        completes(backend, successor)

    assert not fault.armed, "the crash never fired, so this pinned nothing"
    assert len(backend.enqueued(chain_task)) == 2
    placed = Writer(task=str(first), placement=respawn_name(run, 1))
    assert set(names) == {spawned_name(placed).stored()}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(subagent, subagent, id="two-subagents-one-correlation"),
        pytest.param(fork_child, subagent, id="a-fork-and-a-subagent-one-string"),
    ],
)
@pytest.mark.parametrize("child_tasks", [1, 2], ids=["one-child-task", "two-child-tasks"])
def test_two_spawners_that_choose_one_string_each_get_their_own_child(
    backend, first, second, child_tasks
):
    """Reddens if a string two spawners happen to share names both spawns to the engine: the
    second is handed the first's child, of whatever task, and its own params are dropped."""
    shared = str(uuid4())
    names = [private("child") for _ in range(child_tasks)]
    for name in names:
        backend.register_body(name, child)
    parents = []
    for index, spawns in enumerate((first, second)):
        parent = private("parent")
        target = names[index % child_tasks]
        backend.register(
            parent,
            lambda _rid, spawns=spawns, target=target: spawns(target, shared),
            spawning(backend),
            Fault(),
            layers=[],
        )
        parents.append(backend.spawn(parent, "r-parent"))
    for parent in parents:
        backend.run_until_result(parent)

    assert sum(len(backend.enqueued(name)) for name in names) == 2
    for parent in parents:
        completes(backend, parent)


def test_spawns_placed_apart_in_one_task_each_get_their_own_child(backend):
    """Reddens if a spawn is named without its placement: two in sequence and two gather branches
    in one task would share a child."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, child)

    def spawn():
        args = SpawnArgs(task_name=child_task, params={}).model_dump(exclude_none=True)
        return call_tool(SPAWN_TOOL, args, SpawnResult)

    def workflow(_rid):
        yield from spawn()
        yield from spawn()
        yield from gather([spawn, spawn])

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    completes(backend, backend.spawn(parent, "r-parent"))

    assert len(backend.enqueued(child_task)) == 4


def test_a_spawn_that_names_itself_is_refused(backend):
    """Reddens if a workflow, or a model writing its tool request, can choose the name the engine
    deduplicates a spawn by: the handler names every spawn."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, child)
    supplied = SpawnArgs(task_name=child_task, params={}, idempotency_key=str(uuid4()))

    def workflow(_rid):
        try:
            yield from call_tool(SPAWN_TOOL, supplied.model_dump(exclude_none=True), SpawnResult)
        except Refused:
            return "refused"
        return "spawned"

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))

    assert snapshot is not None
    assert snapshot.result == "refused"
    assert backend.enqueued(child_task) == []


def test_spawns_in_a_scope_entered_twice_each_get_their_own_child(backend):
    """Reddens while a gather inside a scope entered twice names its branch's spawn the same both
    times: the second is handed the first's child. The two spawns are identical, so only their
    placement can tell them apart."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, child)
    scope = compose_key(t"round")
    args = SpawnArgs(task_name=child_task, params={}).model_dump(exclude_none=True)

    def spawned():
        (result,) = yield from gather([lambda: call_tool(SPAWN_TOOL, args, SpawnResult)])
        return str(result.task_id)

    def workflow(_rid):
        first = yield from scoped(scope, spawned)
        second = yield from scoped(scope, spawned)
        return [first, second]

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))

    assert snapshot is not None
    assert len(backend.enqueued(child_task)) == 2
    assert len(set(snapshot.result)) == 2


def unenqueued() -> tuple[MeteredInterpreter, list[str]]:
    """A spawn tool that records a call and enqueues nothing, for a ctx with no engine."""
    calls: list[str] = []

    def spawner(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        calls.append(idempotency_key)
        return str(uuid.uuid4())

    domain = MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
    )
    return domain, calls


def test_a_spawn_with_no_task_to_name_it_by_is_refused_and_says_so():
    """Reddens if a ctx without a task can spawn, or is refused without naming what is missing."""
    domain, calls = unenqueued()

    def workflow():
        args = SpawnArgs(task_name="child", params={}).model_dump(exclude_none=True)
        try:
            yield from call_tool(SPAWN_TOOL, args, SpawnResult)
        except Refused as refused:
            return refused.reason
        return "spawned"

    assert "no task" in DurableHandler(LocalCtx(), domain).run(workflow)
    assert calls == []


def test_a_respawn_with_no_task_to_name_it_by_is_refused():
    """Reddens if a respawn under a ctx without a task crashes on the missing attribute where a
    spawn is refused."""
    domain, calls = unenqueued()

    def step(state: int, turn: Turn):
        yield from ()
        return Again(state + 1)

    with pytest.raises(Refused, match="no task"):
        DurableHandler(LocalCtx(), domain).run(
            lambda: respawn(step, Chain(task="chain", state=0, run_id="r"))
        )
    assert calls == []


def test_a_viewer_replays_a_generation_that_respawned(backend):
    """Reddens if replaying a recorded generation cut asks the ctx for a task: a viewer passes
    none through, and the tape already holds the successor the cut spawned."""
    chain_task, run = private("chain"), str(uuid4())

    def step(state: int, turn: Turn):
        yield from ()
        return Again(state + 1) if state < 1 else Done(state)

    def body(params, ctx):
        carried = Chain.from_params(params, task=chain_task, schema=int, initial=0, run_id=run)
        chain = Chain(
            task=chain_task, state=carried.state, run_id=run, generation=carried.generation
        )
        return DurableHandler(ctx, spawning(backend), params=params).run(
            lambda: respawn(step, chain)
        )

    backend.register_body(chain_task, body)
    first = backend.spawn(chain_task, "r-chain")
    recorded = backend.run_until_result(first)
    assert recorded is not None
    for successor, _params in backend.enqueued(chain_task)[1:]:
        completes(backend, successor)
    calls: list[str] = []

    def spawner(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        calls.append(idempotency_key)
        return str(uuid.uuid4())

    tape = set(backend.checkpoint_states(first))
    viewer = ViewingCtx(backend.unclaimed_ctx(first), tape)
    generation = Chain(task=chain_task, state=0, run_id=run, generation=0)

    replayed = DurableHandler(viewer, spawning(backend, spawner), ledger=None).run(
        lambda: respawn(step, generation)
    )

    assert replayed == recorded.result
    assert calls == []


def test_a_viewer_replays_a_refused_spawn_from_its_record(backend):
    """Reddens if a refused spawn is decided again on replay: the tape must hold the refusal, so a
    viewer with depth to spare is served it and enqueues nothing."""
    parent = private("parent")

    def refused_spawn():
        try:
            yield from call_tool(
                SPAWN_TOOL,
                SpawnArgs(task_name="child", params={}).model_dump(exclude_none=True),
                SpawnResult,
            )
        except Refused as refused:
            return refused.reason
        return "spawned"

    def body(params, ctx):
        spent = {**params, BUDGET_DEPTH_PARAM: 0}
        return DurableHandler(ctx, spawning(backend), params=spent).run(refused_spawn)

    backend.register_body(parent, body)
    first = backend.spawn(parent, "r-refused")
    recorded = backend.run_until_result(first)
    assert recorded is not None
    assert str(recorded.result).startswith("spawn depth exhausted"), recorded
    calls: list[str] = []

    def spawner(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        calls.append(idempotency_key)
        return str(uuid.uuid4())

    viewer = ViewingCtx(backend.unclaimed_ctx(first), set(backend.checkpoint_states(first)))
    replayed = DurableHandler(viewer, spawning(backend, spawner), ledger=None).run(refused_spawn)

    assert replayed == recorded.result
    assert calls == []


def answering_with_its_task(params, ctx):
    """A child whose answer is its own task id, so a parent can tell whose answer it got."""
    answer_parent(ctx, params, {"answer": str(ctx.task_id)})
    return "ok"


def test_a_parent_catches_a_failed_child_and_recovers(backend):
    """Reddens if a `Failed` answer reaches its parent as anything but a `ChildFailed` naming the
    child's task: a supervising parent catches it and completes."""
    parent, child_task = private("parent"), private("child")

    def failing(params, ctx):
        failed = Failed(error=("ValueError", "a crash"), notes=[("KeyError", "beside it")])
        answer_parent(ctx, params, ChildAnswer(answer=failed).model_dump(mode="json"))
        return "answered"

    def supervises():
        spawned = yield from spawn_child(child_task, "c", {})
        try:
            yield from join_answer(spawned)
        except ChildFailed as failed:
            return [str(spawned.task_id), str(failed)]
        return None

    backend.register_body(child_task, failing)
    backend.register_body(
        parent,
        lambda params, ctx: DurableHandler(ctx, spawning(backend), params=params).run(supervises),
    )
    parent_id = backend.spawn(parent, str(uuid4()))
    backend.run_until_result(parent_id)
    [(child_id, _params)] = backend.enqueued(child_task)
    completes(backend, child_id)
    snapshot = backend.run_until_result(parent_id)

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    task_id, message = snapshot.result
    assert task_id == str(child_id)
    assert message == f"child task {child_id} failed of ValueError: a crash"


def test_a_parent_is_answered_only_by_the_child_it_spawned(backend):
    """Reddens if a done event is named by anything a second run can repeat: the second parent,
    spawning under the same author name, would take the first child's answer."""
    child_task = private("child")
    backend.register_body(child_task, answering_with_its_task)
    correlation = str(uuid4())
    answers = []
    for _ in range(2):
        parent = private("parent")
        backend.register(
            parent,
            lambda _rid: subagent(child_task, correlation),
            spawning(backend),
            Fault(),
            layers=[],
        )
        snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))
        assert snapshot is not None
        assert snapshot.state == "completed", snapshot
        answers.append(snapshot.result["content"])

    assert answers == [str(task_id) for task_id, _params in backend.enqueued(child_task)]


def test_every_placed_spawn_is_answered_on_a_name_the_handler_gives_it(backend):
    """Reddens if two placed spawns share a done event, if the name a workflow is handed differs
    from the one its child answers on, or if a workflow's own `done_event` param reaches the
    child."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, child)
    scope = compose_key(t"round")
    args = SpawnArgs(task_name=child_task, params={"done_event": "chosen"})

    def spawn():
        return call_tool(SPAWN_TOOL, args.model_dump(exclude_none=True), Spawned)

    def in_a_gather():
        return gather([spawn])

    def workflow(_rid):
        first = yield from spawn()
        second = yield from spawn()
        branches = yield from gather([spawn, spawn])
        entered = yield from scoped(scope, in_a_gather)
        again = yield from scoped(scope, in_a_gather)
        spawned = [first, second, *branches, *entered, *again]
        return [spawn.done_event.stored() for spawn in spawned]

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    handed = snapshot.result
    assert len(set(handed)) == 6
    assert sorted(handed) == sorted(p["done_event"] for _id, p in backend.enqueued(child_task))


def test_children_spawned_twice_under_one_name_each_answer_their_parent(backend):
    """Reddens if a done event carries its spawn's occurrence as a suffix: the child's emit step
    composes the name, and an engine that counts step names refuses a second suffix."""
    child_task, parent = private("child"), private("parent")
    backend.register_body(child_task, answering_with_its_task)

    def workflow(_rid):
        first = yield from spawn_child(child_task, "c", {})
        second = yield from spawn_child(child_task, "c", {})
        answers = []
        for spawned in (first, second):
            answers.append((yield from join_child(spawned, dict))["answer"])
        return answers

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == [str(task_id) for task_id, _p in backend.enqueued(child_task)]


def test_a_child_that_respawns_answers_its_parent_once_it_finishes(backend):
    """Reddens if a generation boundary answers the parent: the first generation's marker would
    take the done event, and the finished chain's answer would lose to it."""
    chain_task, parent = private("chain"), private("parent")

    def step(state: int, turn: Turn):
        yield from ()
        return Done(f"finished-at-{state}") if turn.generation >= 1 else Again(state + 1)

    def body(params, ctx):
        carried = Chain.from_params(params, task=chain_task, schema=int, initial=0, run_id="r")
        chain = Chain(
            task=chain_task,
            state=carried.state,
            run_id="r",
            generation=carried.generation,
            params=carried.params,
        )
        handler = DurableHandler(ctx, spawning(backend), params=params)
        return run_child(ctx, params, handler, lambda: respawn(step, chain))

    backend.register_body(chain_task, body)

    def workflow(_rid):
        spawned = yield from spawn_child(chain_task, "c", {})
        return (yield from join_answer(spawned))

    backend.register(parent, workflow, spawning(backend), Fault(), layers=[])
    snapshot = backend.run_until_result(backend.spawn(parent, "r-parent"))
    for generation, _params in backend.enqueued(chain_task):
        completes(backend, generation)

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == "finished-at-1"
    assert len(backend.enqueued(chain_task)) == 2


class Recorded(LocalCtx):
    """A ctx whose every step is served from a record, as a replayed spawn's is."""

    task_id = uuid.uuid4()

    def __init__(self, value: Any) -> None:
        self.value = value

    def step(self, name, thunk):
        return self.value


def test_a_spawn_recorded_without_a_done_event_is_refused_by_name():
    """Reddens if a spawn checkpoint written before done events were named fails as a bare lookup:
    an operator draining parked parents has to tell it from a bug."""
    domain, _calls = unenqueued()
    old_checkpoint = {"task_id": str(uuid.uuid4())}

    def workflow():
        args = SpawnArgs(task_name="child", params={}).model_dump(exclude_none=True)
        return (yield from call_tool(SPAWN_TOOL, args, Spawned))

    with pytest.raises(Exception, match="no done event") as caught:
        DurableHandler(Recorded(old_checkpoint), domain).run(workflow)
    assert not isinstance(caught.value, KeyError)
