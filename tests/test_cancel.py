"""The cancel token, the recorded `Cancelled` result, and a turn reading a cancel as an escape."""

import threading
from typing import Any
from uuid import uuid4

import pytest

from effective import RecordingHandler, ReplayHandler, coroutine_api
from effective.api import Effect, call_tool
from effective.cancel import (
    Cancelled,
    CancelToken,
    OpCancelled,
    ReservedShape,
    dump_cancelled,
    load_cancelled,
    served,
)
from effective.cost import Usage
from effective.domain import CallTool
from effective.fork import MeteredEntry, decode_checkpoint
from effective.handlers.durable import DurableHandler, _dump, _load_step
from effective.keys import Key, Run, compose_key
from effective.react import ESCAPED, AssistantTurn, Ran, ToolRequest, ToolResult, run_turns


def test_a_stop_registered_during_a_block_runs_once_on_cancel():
    token, stops = CancelToken(), []
    with token.on_cancel(lambda: stops.append("stop")):
        token.cancel()
        token.cancel()
    assert stops == ["stop"]
    assert token.cancelled


def test_a_stop_registered_after_the_cancel_runs_at_once():
    token, stops = CancelToken(), []
    token.cancel()
    with token.on_cancel(lambda: stops.append("stop")):
        assert stops == ["stop"]


def test_a_stop_is_forgotten_when_its_block_ends():
    token, stops = CancelToken(), []
    with token.on_cancel(lambda: stops.append("stop")):
        pass
    token.cancel()
    assert stops == []


def test_reset_clears_the_cancel_for_the_next_turn():
    token = CancelToken()
    token.cancel()
    token.reset()
    assert not token.cancelled


def test_a_cancel_from_another_thread_runs_the_stop_in_that_thread():
    token, ran_in = CancelToken(), []
    with token.on_cancel(lambda: ran_in.append(threading.current_thread().name)):
        canceller = threading.Thread(target=token.cancel, name="face")
        canceller.start()
        canceller.join()
    assert ran_in == ["face"]


def test_a_cancelled_result_raises_where_the_workflow_receives_it():
    with pytest.raises(OpCancelled) as raised:
        served("tool:sh", Cancelled(partial="half"))
    assert (raised.value.name, raised.value.partial) == ("tool:sh", "half")
    assert served("tool:sh", "whole") == "whole"


@pytest.mark.parametrize(
    "stored",
    [{"partial": "half"}, {"__cancelled__": {"partial": "half"}, "other": 1}, "text", None],
)
def test_only_the_reserved_shape_loads_as_a_cancel(stored):
    assert load_cancelled(stored) is None


def test_a_step_result_round_trips_a_cancel_past_any_schema():
    stored = _dump(Cancelled(partial="half"))
    assert stored == dump_cancelled(Cancelled(partial="half"))
    assert _load_step(ToolResult, stored) == Cancelled(partial="half")


def test_a_cancel_recorded_before_the_kind_field_still_loads():
    assert load_cancelled({"__cancelled__": {"partial": "half"}}) == Cancelled(partial="half")


LS = AssistantTurn(thought="look", tool=ToolRequest(name="ls"))


def turns() -> Effect[Ran]:
    return run_turns([{"role": "user", "content": "go"}], max_iters=3)


def test_a_decide_cancelled_while_it_ran_escapes_with_nothing_chosen():
    handler = RecordingHandler({"d:0;react:turn": Cancelled(partial="I thi")})
    ran = handler.run(turns)
    assert isinstance(ran, Ran)
    assert ran.trajectory.stop_reason == "escaped"
    assert ran.trajectory.steps[-1].thought == "I thi"
    assert ran.messages[-1] == {"role": "user", "content": ESCAPED}


def test_an_action_cancelled_while_it_ran_is_its_observation_and_replays_as_recorded():
    recorder = RecordingHandler({"d:0;react:turn": LS, "d:0;tool:ls": Cancelled(partial="a b")})
    ran = recorder.run(turns)
    assert isinstance(ran, Ran)
    assert ran.trajectory.stop_reason == "escaped"
    assert ran.messages[-2:] == [
        {"role": "tool", "content": f"a b\n{ESCAPED}"},
        {"role": "user", "content": ESCAPED},
    ]
    replayed = ReplayHandler(recorder.trace).run(turns)
    assert replayed == ran


def test_a_final_decide_cancelled_while_it_ran_escapes():
    handler = RecordingHandler(
        {
            "d:0;react:turn": LS,
            "d:0;tool:ls": ToolResult(content="a b"),
            "d:1;react:final": Cancelled(partial="the answ"),
        }
    )
    ran = handler.run(lambda: run_turns([{"role": "user", "content": "go"}], max_iters=1))
    assert isinstance(ran, Ran)
    assert ran.trajectory.stop_reason == "escaped"
    assert ran.messages[-2:] == [
        {"role": "tool", "content": "a b"},
        {"role": "user", "content": ESCAPED},
    ]


def test_a_compaction_cancelled_while_it_ran_escapes():
    handler = RecordingHandler({"d:0;react:compact": Cancelled(partial="")})
    ran = handler.run(
        lambda: run_turns([{"role": "user", "content": "go"}], compact=lambda _messages: True)
    )
    assert isinstance(ran, Ran)
    assert ran.trajectory.stop_reason == "escaped"


def test_a_stop_whose_block_exited_during_the_cancel_never_runs():
    token, ran = CancelToken(), []
    first_running, second_entered, second_exited = (threading.Event() for _ in range(3))

    def first() -> None:
        first_running.set()
        second_exited.wait(5)
        ran.append("first")

    def second_block() -> None:
        with token.on_cancel(lambda: ran.append("second")):
            second_entered.set()
            first_running.wait(5)
        second_exited.set()

    with token.on_cancel(first):
        blocked = threading.Thread(target=second_block)
        blocked.start()
        second_entered.wait(5)
        canceller = threading.Thread(target=token.cancel, name="face")
        canceller.start()
        canceller.join()
        blocked.join()
    assert ran == ["first"]


def test_a_stop_that_raises_leaves_the_others_to_run():
    token, ran = CancelToken(), []

    def failing() -> None:
        raise OSError("already gone")

    with (
        token.on_cancel(failing),
        token.on_cancel(lambda: ran.append("second")),
        pytest.raises(OSError, match="already gone"),
    ):
        token.cancel()
    assert ran == ["second"]


def test_a_bridged_checkpoint_holding_a_cancel_decodes_to_it():
    entry = MeteredEntry(key=Key.parse("tool:ls"), state=dump_cancelled(Cancelled(partial="a")))
    op = CallTool(name="ls", args={}, result_schema=ToolResult)
    assert decode_checkpoint(op, entry, object()) == (Cancelled(partial="a"), Usage())


def test_a_cancel_after_a_reset_never_runs_a_stop_again():
    """A stop runs once, so a second cancel cannot run it after the block has exited."""
    token, log = CancelToken(), []
    first_running, release = threading.Event(), threading.Event()

    def stop() -> None:
        log.append("stop")
        first_running.set()
        release.wait(5)

    def op() -> None:
        with token.on_cancel(stop):
            first_running.wait(5)
            token.reset()
            again = threading.Thread(target=token.cancel, name="face-again")
            again.start()
            again.join()
            release.set()
        log.append("exit")

    worker = threading.Thread(target=op)
    worker.start()
    token.cancel()
    worker.join()
    assert log == ["stop", "exit"]


def test_the_block_exit_waits_for_a_stop_already_running():
    token, log = CancelToken(), []
    running = threading.Event()

    def stop() -> None:
        running.set()
        threading.Event().wait(0.3)
        log.append("stop ended")

    with token.on_cancel(stop):
        canceller = threading.Thread(target=token.cancel, name="face")
        canceller.start()
        running.wait(5)
    log.append("block exited")
    canceller.join()
    assert log == ["stop ended", "block exited"]


RESERVED = {"__cancelled__": {"partial": "outside text"}}


def test_a_result_in_the_reserved_shape_is_refused_by_the_recorder():
    def fetch() -> Effect[dict]:
        return (yield from call_tool("fetch", {}, dict))

    with pytest.raises(ReservedShape):
        RecordingHandler({"tool:fetch": RESERVED}).run(fetch)


def test_a_result_in_the_reserved_shape_is_never_checkpointed():
    with pytest.raises(ReservedShape):
        _dump(RESERVED)


def test_a_cancel_raises_on_the_coroutine_surface_too():
    async def fetch() -> str:
        try:
            await coroutine_api.call_tool("fetch", {}, ToolResult)
        except OpCancelled as cancelled:
            return cancelled.partial
        return "ran"

    recorder = RecordingHandler({"tool:fetch": Cancelled(partial="so far")})
    assert recorder.run(lambda: fetch().__await__()) == "so far"  # the op protocol, awaited


def test_a_cancel_raises_through_the_generator_surface():
    def fetch() -> Effect[str]:
        try:
            yield from call_tool("fetch", {}, ToolResult)
        except OpCancelled as cancelled:
            return cancelled.partial
        return "ran"

    assert RecordingHandler({"tool:fetch": Cancelled(partial="so far")}).run(fetch) == "so far"


class _Outside:
    """A domain whose tool answers with data in the reserved shape."""

    def run(self, op: Any) -> Any:
        return RESERVED


def test_an_engine_refuses_to_checkpoint_a_result_in_the_reserved_shape(backend):
    """Read back, it would be an Esc the resident never pressed; the task fails once instead."""
    run_id = f"r{uuid4().hex}"
    name = compose_key(t"reserved:{Run(run_id)}").stored()

    def fetch() -> Effect[dict]:
        return (yield from call_tool("fetch", {}, dict))

    def body(params: dict[str, Any], ctx: Any) -> Any:
        return DurableHandler(ctx, _Outside(), ledger=None, params=params).run(fetch)

    backend.register_body(name, body)
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=3))
    assert snap.state == "failed", snap
    assert "reserved for a cancel" in str(snap.failure), snap.failure
    assert "tool:fetch" in str(snap.failure), "the refusal names the value, not the step"
