"""A run's recorded format is fixed at spawn and read back from its params, on both engines.

| question | the pin                                                                        |
|----------|--------------------------------------------------------------------------------|
| unmarked | a run whose params mark no format is refused by name                           |
| resume   | a run parked by one body resumes under the example with its usage recorded     |
| child    | a harness child's model call records its usage                                 |
| spawn    | an engine marks every run it spawns                                            |
| one      | this release records one format                                                |
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
from effective.cost import FORMAT_PARAM, NEWEST_FORMAT, MeteredInterpreter, RecordedFormat, Usage
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.ops import UnreadFormat
from examples.agent_firm import critic
from examples.agent_firm.engage import CritiqueWire
from examples.agent_firm.world import World
from examples.startup import engine as startup


def _unasked(op: Any) -> Any:
    raise AssertionError(f"asked {op!r}")


def _capped(op: Any) -> tuple[Any, Usage]:
    return CritiqueWire.model_validate(
        {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}
    ), Usage(cost=0.02)


def test_a_critique_whose_params_mark_no_format_is_refused_by_name(tmp_path):
    db = tmp_path / "task.db"
    critic.run(World(critic=_capped, critiques=frozenset({CritiqueWire})), db)
    with closing(sqlite3.connect(db)) as unmarked:
        unmarked.execute("UPDATE tasks SET params = '{}'")
        unmarked.commit()
    with pytest.raises(UnreadFormat):
        critic.replay(db)


class _Reviewed(BaseModel):
    first: dict
    second: dict


def _reviewed():
    first = yield from ask_llm("first", "first", dict)
    yield from await_event("review", dict)
    second = yield from ask_llm("second", "second", dict)
    return _Reviewed(first=first, second=second)


def test_a_startup_run_parked_by_another_body_resumes_with_its_usage_recorded(tmp_path):
    db = tmp_path / "startup.db"
    world = startup.World(llm=lambda prompt: {"answer": "ok"}, tools={})
    app = SqliteApp(str(db))
    try:

        @app.register_task("startup")
        def unmarked(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, world.domain(), ledger=None)
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


def test_a_record_its_contract_cannot_read_fails_naming_the_contract(tmp_path):
    world = startup.World(llm=lambda prompt: {"answer": "ok"}, tools={})
    app = SqliteApp(str(tmp_path / "task.db"))
    try:

        @app.register_task("startup")
        def task(params: dict[str, Any], ctx: Any) -> Any:
            return DurableHandler(ctx, world.domain(), ledger=None).run(_reviewed)

        task_id = app.spawn("startup", {})
        waiting = app.run_until_result(task_id)
        assert (waiting and waiting.state) == "waiting", waiting
        app.conn.execute(
            "UPDATE checkpoints SET state = ? WHERE name = ?",
            (json.dumps({"answer": "ok"}), "step:first"),
        )
        app.emit_event("review", {})
        failed = app.run_until_result(task_id)
    finally:
        app.close()
    assert failed is not None
    assert failed.failure is not None
    assert failed.failure.kind == "RecordedValueRejected", failed
    assert "{result, usage}" in failed.failure.text


def test_a_harness_childs_model_call_records_its_usage(backend):
    name = private("marked-child")
    domain = MeteredInterpreter(
        llm=lambda op: ({"answer": "ok"}, Usage(cost=0.01)), tools=_unasked
    )

    def program(params: dict[str, Any]):
        return (yield from ask_llm("think", "x", dict))

    backend.register_child(name, name, program, domain, Fault())
    task_id = backend.spawn(name, name)
    assert backend.run_until_result(task_id).state == "completed"
    (recorded,) = (state for key, state in backend.checkpoint_states(task_id).items())
    assert "usage" in json.loads(json.dumps(recorded))


def test_an_engine_marks_every_run_it_spawns(backend):
    name = private("unmarked-spawn")
    backend.engine.spawn(name, {"run_id": "r-unmarked"})
    ((_task_id, params),) = backend.enqueued(name)
    assert params[FORMAT_PARAM] == RecordedFormat.V1.value


def test_an_engine_marks_a_run_whose_params_carry_a_mark(backend):
    name = private("carried-spawn")
    backend.engine.spawn(name, {"run_id": "r-carried", FORMAT_PARAM: "v0"})
    ((_task_id, params),) = backend.enqueued(name)
    assert params[FORMAT_PARAM] == NEWEST_FORMAT.value


def test_this_release_records_one_format():
    """A second format has three things to decide before this test may change: whether a child or
    a fork records under its parent's format, how each decoder chooses by format, and what format
    an entry bridged from a base's record carries."""
    assert list(RecordedFormat) == [RecordedFormat.V1]
