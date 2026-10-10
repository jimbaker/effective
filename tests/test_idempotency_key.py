"""A step's idempotency key is minted by the handler from its task and placement, or refused."""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault

from effective.api import step
from effective.budget import MeasuredBudget
from effective.contexts import LocalCtx, RecordingCtx, ReplayCtx, ResumeCtx
from effective.cost import Usage
from effective.domain import AskLLM, CallTool, DomainOp, SpawnArgs
from effective.fork import live_drive, measured_drive
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import Run, compose_key
from effective.layers import op_layer
from effective.ops import Minted, Step
from effective.permission import Refused
from effective.viewing import ViewingCtx


class _TaskCtx(LocalCtx):
    task_id = "t1"


class _Echo:
    """Answers a tool call with the args it reached the domain with."""

    def run(self, op: Any) -> Any:
        return op.args


def _asks(op: DomainOp[Any]):
    def workflow():
        try:
            return (yield from step("send", op, idempotency_key=Minted()))
        except Refused as refused:
            return refused.reason

    return workflow


def test_a_tool_call_in_a_task_is_handed_its_minted_key():
    sent = CallTool(name="send", args={"to": "a"}, result_schema=dict)
    assert DurableHandler(_TaskCtx(), _Echo()).run(_asks(sent)) == {
        "to": "a",
        "idempotency_key": "idempotency:t1;step:send",
    }


@pytest.mark.parametrize(
    ("ctx", "op", "reason"),
    [
        (
            _TaskCtx,
            CallTool(name="send", args={"idempotency_key": "mine"}, result_schema=dict),
            "minted by the handler",
        ),
        (LocalCtx, CallTool(name="send", args={}, result_schema=dict), "no task"),
        (_TaskCtx, AskLLM(messages=[], response_schema=str), "AskLLM has none"),
    ],
    ids=["a-key-in-the-args", "outside-a-task", "a-model-call"],
)
def test_a_key_the_handler_cannot_mint_is_refused(ctx, op, reason):
    assert reason in DurableHandler(ctx(), _Echo()).run(_asks(op))


@op_layer
def _audits_first(op):
    """Injects a keyed audit step ahead of the op it wraps."""
    audit = CallTool(name="audit", args={}, result_schema=dict)
    yield Step(name="audit", op=audit, idempotency_key=Minted())
    return (yield op)


def test_a_step_a_layer_injects_is_keyed_by_its_own_placement():
    seen: list[str] = []

    class Recording(_Echo):
        def run(self, op: Any) -> Any:
            seen.append(op.args["idempotency_key"])
            return op.args

    sent = CallTool(name="send", args={}, result_schema=dict)
    DurableHandler(_TaskCtx(), Recording(), op_layers=[_audits_first]).run(_asks(sent))
    assert seen == ["idempotency:t1;step:audit", "idempotency:t1;step:send"]


def test_a_key_an_untyped_caller_supplies_is_refused():
    supplied: Any = "mine"
    sent = CallTool(name="send", args={}, result_schema=dict)

    def workflow():
        try:
            return (yield Step(name="send", op=sent, idempotency_key=supplied))
        except Refused as refused:
            return refused.reason

    assert "not supplied: 'mine'" in DurableHandler(_TaskCtx(), _Echo()).run(workflow)


def _keyed_send(run_id: str):
    sent = CallTool(name="send", args={"v": 1}, result_schema=dict)
    return (yield from step("send", sent, idempotency_key=Minted()))


def test_a_viewer_replays_a_completed_keyed_step_without_calling_the_tool(backend):
    """A viewer runs as the task it views, so a keyed step is served from the tape."""
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, _keyed_send, _Echo(), Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot

    ctx = backend.unclaimed_ctx(task_id)
    try:
        viewer = ViewingCtx(ctx, backend.checkpoint_states(task_id))
        assert DurableHandler(viewer, _Refuses()).run(lambda: _keyed_send(run_id)) == (
            snapshot.result
        )
    finally:
        if backend.name == "postgres":
            ctx._conn.close()


class _Refuses:
    def run(self, op: Any) -> Any:
        raise AssertionError(f"a viewer called the tool: {op!r}")


class _Metered(_Echo):
    def run_metered(self, op: Any) -> tuple[Any, Usage]:
        return self.run(op), Usage()


DRIVERS = {
    "live": lambda: live_drive(_keyed_send("r"), None, _Metered(), {}),
    "measured": lambda: measured_drive(
        lambda: _keyed_send("r"), MeasuredBudget(overall=1, run_id="r"), _Metered(), {}
    ),
}


@pytest.mark.parametrize("drive", DRIVERS.values(), ids=DRIVERS.keys())
def test_an_in_process_fork_refuses_a_keyed_step_it_would_run_live(drive):
    """A fork driver has no task to mint a key from."""
    with pytest.raises(Refused, match="no task"):
        drive()


def _supplies_its_own_key(run_id: str):
    sent = CallTool(name="send", args={"idempotency_key": "mine"}, result_schema=dict)
    try:
        return (yield from step("send", sent))
    except Refused as refused:
        return refused.reason


def test_idempotency_key_is_reserved_in_a_tool_calls_args_without_asking():
    reason = DurableHandler(_TaskCtx(), _Echo()).run(lambda: _supplies_its_own_key("r"))
    assert "not by its args: 'mine'" in reason


def test_a_live_fork_refuses_a_key_in_a_tool_calls_args():
    with pytest.raises(Refused, match="not by its args: 'mine'"):
        live_drive(_supplies_its_own_key("r"), None, _Metered(), {})


def _spawns(run_id: str):
    request = SpawnArgs(task_name="child", params={}).call()
    return (yield from step(compose_key(t"tool:spawn,{Run(run_id)}").stored(), request))


def test_a_live_fork_refuses_a_spawn_it_has_no_task_to_name():
    with pytest.raises(Refused, match="a spawn is named by its task"):
        live_drive(_spawns("r"), None, _Metered(), {})


class _RecordingTask(RecordingCtx):
    task_id = "t1"


@pytest.mark.parametrize("replaying", [ReplayCtx, ResumeCtx], ids=["replay", "resume"])
def test_an_in_process_replay_serves_a_keyed_step_from_its_log(replaying):
    """A ctx with no task replays a keyed step it logged, and never asks for a key."""
    recorded = _RecordingTask()
    sent = CallTool(name="send", args={"v": 1}, result_schema=dict)
    first = DurableHandler(recorded, _Echo()).run(_asks(sent))
    assert DurableHandler(replaying(recorded.log), _Refuses()).run(_asks(sent)) == first


def test_the_recording_core_refuses_a_key_nobody_but_the_handler_may_write():
    handler = RecordingHandler(responses={"send": {"canned": True}})
    match handler.run(lambda: _supplies_its_own_key("r")):
        case str() as reason:
            assert "not by its args: 'mine'" in reason
        case unexpected:
            raise AssertionError(unexpected)
