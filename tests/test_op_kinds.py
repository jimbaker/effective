"""Two kinds of op under one name stay apart: a step's key names its op's kind."""

from _conformance import Fault, private
from _steps import called_step, placed_step

from effective.api import step
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool, Judge
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler


def _three_kinds():
    asked = yield from step("x", AskLLM(messages="m", response_schema=str))
    judged = yield from step("x", Judge(state={}, questions={}, response_schema=str))
    called = yield from step("x", CallTool(name="x", args={}, result_schema=str))
    return [asked, judged, called]


ANSWERS = {"llm:x": "asked", "judge:x": "judged", "tool:x": "called"}


def test_three_kinds_under_one_name_take_three_keys_and_three_answers():
    recorder = RecordingHandler(responses=ANSWERS)
    assert recorder.run(_three_kinds) == ["asked", "judged", "called"]
    assert len({entry.key for entry in recorder.trace}) == 3
    assert ReplayHandler(recorder.trace).run(_three_kinds) == ["asked", "judged", "called"]


def _durable_three_kinds(_run_id: str):
    return (yield from _three_kinds())


def test_three_kinds_under_one_name_take_three_keys_on_the_engines(backend):
    name = private("three-kinds")
    domain = MeteredInterpreter(
        llm=lambda op: ("asked", Usage()),
        tools=lambda op: "called",
        judge=lambda op: ("judged", Usage()),
    )
    backend.register(name, _durable_three_kinds, domain, Fault(), [])
    task_id = backend.spawn(name, name)
    assert backend.run_until_result(task_id).result == ["asked", "judged", "called"]
    assert list(backend.checkpoint_keys(task_id)) == [
        placed_step("x", "llm"),
        placed_step("x", "judge"),
        called_step("x"),
    ]
