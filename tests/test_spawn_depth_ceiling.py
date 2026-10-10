"""The spawn depth ceiling holds at the enqueue, whatever yields the spawn.

Depth is refused before enqueue. These pin it at the one door every spawn passes,
on the embedded engine: a raw `call_tool("spawn")`, a spawned child's own body, a gather branch,
and the respawn that continues a chain at the same depth.
"""

from contextlib import suppress
from functools import partial
from typing import Any

import pytest
from _spawning import sqlite_spawner

from effective.api import append_ledger, await_event, call_tool, gather
from effective.budget import BUDGET_DEPTH_PARAM
from effective.checkpoints import read_sqlite_conn
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.cost import LLMCall, MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, AskLLM, SpawnArgs, SpawnResult
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.fork import ForkOutcome, join_fork, run_fork_as_task, spawn_fork
from effective.govern import Refused
from effective.handlers.base import op_key
from effective.handlers.durable import DurableHandler
from effective.interpreters.scripted import scripted_caller
from effective.interpreters.tools import make_tool_runner, run_subagent_as_task, spawn_tool
from effective.keys import Key, Segment, compose_key
from effective.ops import ACCRUAL_PARAM, CARRY_PARAM, GENERATION_PARAM, AppendLedgerRow, LedgerRow
from effective.react import AssistantTurn, ToolRequest
from effective.spawning import Returned

pytestmark = pytest.mark.adversarial


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def no_model(op: AskLLM[Any]) -> tuple[Any, Usage]:
    raise AssertionError("these workflows ask no model")


def spawning_domain(app: SqliteApp, llm: LLMCall = no_model) -> MeteredInterpreter:
    agents = {SPAWN_TOOL: spawn_tool(sqlite_spawner(app))}
    return MeteredInterpreter(llm=llm, tools=make_tool_runner({}, agents=agents))


def spawn_args(task_name: str, **params: Any) -> dict[str, Any]:
    return SpawnArgs(task_name=task_name, params=params).model_dump(exclude_none=True)


def enqueued(app: SqliteApp, name: str) -> list[dict]:
    import json

    rows = app.conn.execute("SELECT params FROM tasks WHERE name = ? ORDER BY rowid", (name,))
    return [json.loads(row[0]) for row in rows]


def raw_spawn(**params: Any):
    """What a model's tool request named `spawn` becomes: no wrapper, no budget."""
    try:
        yield from call_tool(SPAWN_TOOL, spawn_args("child", **params), SpawnResult)
    except Refused:
        return "refused"
    return "spawned"


def register_noop_child(app: SqliteApp) -> None:
    @app.register_task("child")
    def child(params, ctx):
        return "child"


def test_a_raw_spawn_at_depth_zero_is_refused_into_the_workflow(app):
    """Reddens if a spawn from a task whose depth is spent reaches the enqueue. The refusal is a
    `Refused` the workflow can route around, and no child exists."""
    register_noop_child(app)

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(lambda: raw_spawn())

    snap = app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: 0}))

    assert snap is not None
    assert snap.result == "refused"
    assert enqueued(app, "child") == []


@pytest.mark.parametrize("requested", [None, 7], ids=["unbounded", "deeper"])
def test_a_spawned_childs_depth_is_stamped_below_its_parent(app, requested):
    """Reddens if a spawn can hand its child more depth than the parent has left: the child's
    params are what the model wrote, so the handler sets the depth it may carry."""
    register_noop_child(app)

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(
            lambda: raw_spawn(**{BUDGET_DEPTH_PARAM: requested})
        )

    app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: 2}))

    assert [child[BUDGET_DEPTH_PARAM] for child in enqueued(app, "child")] == [1]


def test_the_shipped_subagent_body_carries_its_depth_to_its_own_spawns(app):
    """Reddens if `run_subagent_as_task` builds its handler without its params, which is how a
    depth-0 child enqueued a grandchild."""
    usage = Usage(cost=0.0)
    delegate = AssistantTurn(
        thought="delegate",
        tool=ToolRequest(
            name=SPAWN_TOOL,
            args=spawn_args("child", task="deeper", done_event="child-done:g"),
        ),
    )
    finish = AssistantTurn(thought="done", answer="ok")
    domain = spawning_domain(app, llm=scripted_caller([(delegate, usage), (finish, usage)]))
    app.register_task("child")(partial(run_subagent_as_task, domain=domain, max_iters=3))

    app.run_until_result(
        app.spawn("child", {"task": "t", "done_event": "child-done:c", BUDGET_DEPTH_PARAM: 0})
    )

    assert len(enqueued(app, "child")) == 1  # the child itself, and no grandchild


def test_a_gather_branch_is_held_to_its_task_depth(app):
    """Reddens if a branch handler loses the depth: a branch's spawn is its task's spawn."""
    register_noop_child(app)

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(
            lambda: gather([raw_spawn, raw_spawn])
        )

    snap = app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: 0}))

    assert snap is not None
    assert snap.result == ["refused", "refused"]
    assert enqueued(app, "child") == []


def test_a_task_without_a_depth_spawns_unbounded(app):
    """An absent depth is unbounded, so a task nobody gave a ceiling spawns as it always has."""
    register_noop_child(app)

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(lambda: raw_spawn())

    snap = app.run_until_result(app.spawn("parent", {}))

    assert snap is not None
    assert snap.result == "spawned"
    assert [child.get(BUDGET_DEPTH_PARAM) for child in enqueued(app, "child")] == [None]


def test_a_respawn_successor_keeps_its_depth_and_is_not_refused(app):
    """A chain's next generation is the same task at the same depth: a depth-0 leaf may still
    continue its own chain, and the successor carries the depth unchanged."""

    def step(state: int, turn: Turn):
        yield from ()
        return Done("finished") if turn.generation >= 1 else Again(state + 1)

    @app.register_task("chain")
    def chain_task(params, ctx):
        chain = Chain.from_params(params, task="chain", schema=int, initial=0, run_id="r-chain")
        return DurableHandler(ctx, spawning_domain(app), params=params).run(
            lambda: respawn(step, chain)
        )

    app.spawn("chain", {BUDGET_DEPTH_PARAM: 0})
    for _ in range(6):
        app.work_batch()

    assert [task.get(BUDGET_DEPTH_PARAM) for task in enqueued(app, "chain")] == [0, 0]


# ---------------------------------------------------- forks, reserved params, names and shapes


def extracted(mid: str) -> LedgerRow:
    return LedgerRow(event_id=compose_key(t"extracted:{Segment(mid)}"), kind="extracted")


def review(mid: str) -> Key:
    return compose_key(t"review:{Segment(mid)}")


def spawning_base(mid: str):
    """Spawns a leaf, or routes around the refusal when its depth is spent, then parks."""
    with suppress(Refused):
        yield from call_tool(SPAWN_TOOL, spawn_args("leaf"), SpawnResult)
    yield from append_ledger(extracted(mid))
    approval = yield from await_event(review(mid), dict)
    return approval["decision"]


def fork_of_a_spawning_base(tmp_path, *, base_depth, sweep_params):
    """A base that spawned and parked, forked at the park with a delta: the fork's outcome."""
    app = SqliteApp(str(tmp_path / "fork.db"))
    mid = "m1"

    @app.register_task("leaf")
    def leaf(params, ctx):
        return "leaf"

    @app.register_task("base")
    def base(params, ctx):
        ledger = SqliteLedger(app.conn, "r-base", app.write_lock)
        return DurableHandler(ctx, spawning_domain(app), ledger=ledger, params=params).run(
            lambda: spawning_base(mid)
        )

    base_id = app.spawn("base", {} if base_depth is None else {BUDGET_DEPTH_PARAM: base_depth})
    app.run_until_result(base_id)
    app.emit_event(review(mid).stored(), {"decision": "approve"})
    assert (snap := app.run_until_result(base_id)) is not None
    assert snap.state == "completed"

    app.register_task("child")(
        partial(
            run_fork_as_task,
            workflow=lambda _run: spawning_base(mid),
            domain=spawning_domain(app),
            hypothetical_ledger=lambda run: SqliteLedger(
                app.conn, run, app.write_lock, hypothetical=True
            ),
            read_base=lambda task: read_sqlite_conn(app.conn, task),
        )
    )

    @app.register_task("sweep")
    def sweep(params, ctx):
        def workflow():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-a",
                base_task_id=base_id,
                through=op_key(AppendLedgerRow(row=extracted(mid))).stored(),
                fork_point=review(mid),
                forked_from="r-base",
                forked_at_event=extracted(mid).event_id.stored(),
                delta={"decision": "reject"},
            )
            return (yield from join_fork(handle))

        return DurableHandler(ctx, spawning_domain(app), params=params).run(workflow)

    outcome = app.run_until_result(app.spawn("sweep", sweep_params), max_batches=256)
    app.close()
    assert outcome is not None
    return ForkOutcome.model_validate(outcome.result)


@pytest.mark.parametrize(
    "sweep_params",
    [
        pytest.param({}, id="unbounded"),
        pytest.param({BUDGET_DEPTH_PARAM: 1}, id="sweeper-depth"),
    ],
)
def test_a_fork_of_a_spawning_base_is_not_refused_by_the_forkers_depth(tmp_path, sweep_params):
    """Reddens if a fork child re-decides a spawn its base already made: the spawn is replayed
    from the seed and enqueues nothing, so the forker's depth has nothing to refuse."""
    outcome = fork_of_a_spawning_base(tmp_path, base_depth=1, sweep_params=sweep_params)

    assert outcome.answer == Returned(value="reject")


def test_an_unbounded_fork_of_a_base_whose_spawn_was_refused_replays_the_refusal(tmp_path):
    """Reddens if a fork child decides its base's refused spawn again: the refusal is the step's
    recorded value, so seeding serves it. An unbounded fork is the case a re-decision cannot pass,
    since it would spawn where the base was refused."""
    outcome = fork_of_a_spawning_base(tmp_path, base_depth=0, sweep_params={})

    assert outcome.answer == Returned(value="reject")


def test_a_spawned_child_cannot_carry_the_substrates_reserved_params(app):
    """Reddens if a spawn's params can seed its child's spend meter or chain position: the
    substrate sets those, so a model-written `__accrual__` must not reach the child."""
    register_noop_child(app)
    forged = {ACCRUAL_PARAM: [-1000.0, 50.0, 7], GENERATION_PARAM: 9, CARRY_PARAM: "x"}

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(
            lambda: raw_spawn(**forged)
        )

    app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: 3}))

    (child,) = enqueued(app, "child")
    assert set(child) & set(forged) == set()


def test_the_spawn_capability_answers_only_its_reserved_name(app):
    """Reddens if `spawn_tool` mounted under another name enqueues: the ceiling holds at the spawn
    name, so the capability must not answer any other."""
    register_noop_child(app)

    aliased = MeteredInterpreter(
        llm=no_model,
        tools=make_tool_runner({}, agents={"delegate": spawn_tool(sqlite_spawner(app))}),
    )

    @app.register_task("parent")
    def parent(params, ctx):
        # Named, so the only door left to refuse it is the capability's own name check.
        named = SpawnArgs(task_name="child", params={}, idempotency_key="chosen")
        return DurableHandler(ctx, aliased, params=params).run(
            lambda: call_tool("delegate", named.model_dump(), SpawnResult)
        )

    app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: 0}))

    assert enqueued(app, "child") == []


@pytest.mark.parametrize("depth", ["1", 1.5, [1], True], ids=["string", "float", "list", "bool"])
def test_a_malformed_depth_is_a_refusal_not_a_crash(app, depth):
    """Reddens if a depth that is not an int crashes the task or bounds nothing: it is treated as
    spent, so the spawn is refused as a value."""
    register_noop_child(app)

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(lambda: raw_spawn())

    snap = app.run_until_result(app.spawn("parent", {BUDGET_DEPTH_PARAM: depth}))

    assert snap is not None
    assert snap.result == "refused"
    assert enqueued(app, "child") == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"task_name": "child", "params": None}, id="params-none"),
        pytest.param({"params": {}}, id="no-task-name"),
    ],
)
def test_a_spawn_with_malformed_args_is_refused(app, args):
    """Reddens if a spawn whose args declare no child reaches the enqueue or crashes there."""
    register_noop_child(app)

    def workflow():
        try:
            yield from call_tool(SPAWN_TOOL, args, SpawnResult)
        except Refused:
            return "refused"
        return "spawned"

    @app.register_task("parent")
    def parent(params, ctx):
        return DurableHandler(ctx, spawning_domain(app), params=params).run(workflow)

    snap = app.run_until_result(app.spawn("parent", {}))

    assert snap is not None
    assert snap.result == "refused"
    assert enqueued(app, "child") == []


@pytest.mark.parametrize(
    "chain_params", [{}, {BUDGET_DEPTH_PARAM: 9}], ids=["bare", "asks-deeper"]
)
def test_a_respawn_successor_is_held_to_its_task_depth(app, chain_params):
    """Reddens if `_respawn` lets a chain's params choose the successor's depth: the successor is
    the same task at the same depth, whatever the chain carries."""

    def step(state: int, turn: Turn):
        yield from ()
        return Done("finished") if turn.generation >= 1 else Again(state + 1)

    @app.register_task("chain")
    def chain_task(params, ctx):
        carried = Chain.from_params(params, task="chain", schema=int, initial=0, run_id="r-chain")
        chain = Chain(
            task="chain",
            state=carried.state,
            run_id="r-chain",
            generation=carried.generation,
            params=chain_params,
        )
        return DurableHandler(ctx, spawning_domain(app), params=params).run(
            lambda: respawn(step, chain)
        )

    app.spawn("chain", {BUDGET_DEPTH_PARAM: 0})
    for _ in range(6):
        app.work_batch()

    assert [task.get(BUDGET_DEPTH_PARAM) for task in enqueued(app, "chain")] == [0, 0]
