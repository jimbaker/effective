"""A run's contract is fixed at spawn and read back from its params, on both engines.

| question  | the pin                                                                            |
|-----------|------------------------------------------------------------------------------------|
| handler   | a handler runs under its params' contract, and refuses a `contract` they contradict |
| unmarked  | an example's run spawned before it marked its params reads as the v1 it recorded   |
| child     | a harness child spawned under v1 records v1                                        |
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from _conformance import Fault, private
from pydantic import BaseModel

from effective.api import ask_llm, await_event
from effective.checkpoints import read_sqlite_task
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.durable import DurableHandler
from examples.agent_firm import critic
from examples.agent_firm.engage import CritiqueWire
from examples.agent_firm.world import World
from examples.startup import engine as startup

V1 = Contract.V1.spawn_params({"run_id": "r"})


def _unasked(op: Any) -> Any:
    raise AssertionError(f"asked {op!r}")


@pytest.mark.parametrize(
    ("contract", "params", "runs"),
    [
        (None, None, Contract.V0),
        (None, {"run_id": "r"}, Contract.V0),
        (None, V1, Contract.V1),
        (Contract.V1, None, Contract.V1),
        (Contract.V1, {"run_id": "r"}, Contract.V1),
        (Contract.V1, V1, Contract.V1),
        (Contract.V0, V1, None),
    ],
    ids=["neither", "unmarked", "params", "contract", "unmarked-v1", "agree", "contradicted"],
)
def test_a_handler_runs_under_the_contract_its_run_was_spawned_with(
    tmp_path, contract, params, runs
):
    app = SqliteApp(str(tmp_path / "task.db"))
    try:
        ctx = SqliteTaskContext(app.conn, app.spawn("t", {}))
        try:
            domain = MeteredInterpreter(llm=_unasked, tools=_unasked)
            ran = DurableHandler(ctx, domain, contract=contract, params=params)
        except ValueError:
            ran = None
        assert (ran and ran._contract) == runs
    finally:
        app.close()


def _capped(op: Any) -> tuple[Any, Usage]:
    return CritiqueWire.model_validate(
        {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}
    ), Usage(cost=0.02)


def test_a_critique_banked_before_its_params_were_marked_replays_as_v1(tmp_path):
    db = tmp_path / "task.db"
    heard = critic.run(World(critic=_capped, critiques=frozenset({CritiqueWire})), db)
    with closing(sqlite3.connect(db)) as unmarked:
        unmarked.execute("UPDATE tasks SET params = '{}'")
        unmarked.commit()
    assert critic.replay(db) == heard


class _Reviewed(BaseModel):
    first: dict
    second: dict


def _reviewed():
    first = yield from ask_llm("first", "first", dict)
    yield from await_event("review", dict)
    second = yield from ask_llm("second", "second", dict)
    return _Reviewed(first=first, second=second)


def test_a_startup_run_in_flight_before_its_params_were_marked_resumes_as_v1(tmp_path):
    db = tmp_path / "startup.db"
    world = startup.World(llm=lambda prompt: {"answer": "ok"}, tools={})
    app = SqliteApp(str(db))
    try:

        @app.register_task("startup")
        def unmarked(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, world.domain(), ledger=None, contract=Contract.V1)
            return handler.run(_reviewed)

        task_id = app.spawn("startup", {})
        waiting = app.run_until_result(task_id)
        assert (waiting and waiting.state) == "waiting", waiting
        app.emit_event("review", {})
    finally:
        app.close()
    startup.run(_reviewed, world, store=db)
    app = SqliteApp(str(db))
    try:
        resumed = app.fetch_task_result(task_id)
    finally:
        app.close()
    recorded = {c.key.stored(): c.state for c in read_sqlite_task(Path(db), task_id)}
    assert (resumed and resumed.result) == {
        "first": {"answer": "ok"},
        "second": {"answer": "ok"},
    }
    assert {
        key: "usage" in state for key, state in recorded.items() if key.startswith("step:")
    } == {
        "step:first": True,
        "step:second": True,
    }


@pytest.mark.parametrize("contract", list(Contract), ids=str)
def test_a_harness_child_records_the_contract_it_was_spawned_with(backend, contract):
    name = private("marked-child")
    domain = MeteredInterpreter(
        llm=lambda op: ({"answer": "ok"}, Usage(cost=0.01)), tools=_unasked
    )

    def program(params: dict[str, Any]):
        return (yield from ask_llm("think", "x", dict))

    backend.register_child(name, name, program, domain, Fault())
    task_id = backend.spawn(name, name, contract=contract)
    assert backend.run_until_result(task_id).state == "completed"
    (recorded,) = (state for key, state in backend.checkpoint_states(task_id).items())
    assert ("usage" in json.loads(json.dumps(recorded))) is (contract is Contract.V1)
