"""The interrupt signal, the phases a poll subscribes to, and a turn's answer to each signal."""

from typing import Any

import pytest
from _steps import placed_step

from effective import RecordingHandler
from effective.api import Effect
from effective.interrupts import (
    EVERY_PHASE,
    Escape,
    Interrupted,
    Phase,
    Quiet,
    Redirect,
    Signal,
    interrupt_key,
    signal_of,
    tool_interrupt,
)
from effective.react import (
    ESCAPED,
    AssistantTurn,
    Escaped,
    Ran,
    ToolRequest,
    ToolResult,
    Trajectory,
    react_key,
    run_turns,
)

ACT = AssistantTurn(thought="look", tool=ToolRequest(name="ls"))
ANSWER = AssistantTurn(thought="done", answer="answered")


@pytest.mark.parametrize(
    ("answer", "signal"),
    [
        (Interrupted(), Quiet()),
        (Interrupted(redirect="left"), Redirect("left")),
        (Interrupted(escape=True), Escape()),
        (Interrupted(redirect="left", escape=True), Escape()),
    ],
)
def test_the_wire_answer_reads_as_one_signal(answer: Interrupted, signal: Signal):
    assert signal_of(answer) == signal


def poll_key(depth: int, phase: Phase) -> str:
    """The canned key of the poll a run makes at `depth` and `phase`."""
    return interrupt_key(phase).prefixed(f"d:{depth};").stored()


def poll_name(phase: Phase) -> str:
    """The authored name of a phase's poll, which an escape it reports carries."""
    return interrupt_key(phase).stored()


def poll_step(phase: Phase) -> str:
    """The recorded key of the first turn's poll at `phase`."""
    return placed_step(interrupt_key(phase), "tool", "d:0;")


TURN = placed_step(react_key("turn"), "llm", "d:0;")


def quiet_polls(*depths: int) -> dict[str, Interrupted]:
    return {poll_key(depth, phase): Interrupted() for depth in depths for phase in EVERY_PHASE}


def run(responses: dict[str, Any], phases=EVERY_PHASE) -> tuple[Ran, RecordingHandler]:
    handler = RecordingHandler(responses)
    ran = handler.run(
        lambda: run_turns(
            [{"role": "user", "content": "go"}], max_iters=3, interrupt=tool_interrupt(phases)
        )
    )
    assert isinstance(ran, Ran)
    return ran, handler


def keys(handler: RecordingHandler) -> list[str]:
    return [entry.key.stored() for entry in handler.trace]


def test_a_poll_records_only_the_phases_it_subscribes_to():
    _, between = run({"d:0;react:turn": ANSWER, **quiet_polls(0)}, phases={"pre", "post"})
    assert keys(between) == [poll_step("pre"), TURN, poll_step("post")]
    _, every = run(
        {"d:0;react:turn": ACT, "d:0;tool:ls": ToolResult(content="a b"), "d:1;react:turn": ANSWER}
        | quiet_polls(0, 1)
    )
    assert poll_step("act") in keys(every)


ESCAPES = {
    "pre": ([poll_step("pre")], [("user", "go"), ("user", ESCAPED)]),
    "post": (
        [poll_step("pre"), TURN, poll_step("post")],
        [("user", "go"), ("assistant", "look"), ("user", ESCAPED)],
    ),
    "act": (
        [poll_step("pre"), TURN, poll_step("post"), "d:0;step;tool:ls", poll_step("act")],
        [
            ("user", "go"),
            ("assistant", "look\nAction: ls({})"),
            ("tool", "a b"),
            ("user", ESCAPED),
        ],
    ),
}
"""An escape at each phase of the first turn: what the run recorded, and the transcript it hands
back. After `post` the chosen action is dropped with its thought kept; after `act` it has run."""


@pytest.mark.parametrize("phase", ["pre", "post", "act"])
def test_an_escape_ends_the_run_at_the_phase_it_arrives(phase: Phase):
    responses = {"d:0;react:turn": ACT, "d:0;tool:ls": ToolResult(content="a b"), **quiet_polls(0)}
    responses[poll_key(0, phase)] = Interrupted(escape=True)
    ran, handler = run(responses)
    recorded, transcript = ESCAPES[phase]
    assert keys(handler) == recorded
    assert [(m["role"], m["content"]) for m in ran.messages] == transcript
    dropped = "look" if phase == "post" else ""  # the thought whose action never ran
    ending = ran.trajectory
    assert isinstance(ending, Escaped)
    assert (ending.origin, ending.name, ending.partial) == (
        "escape",
        poll_name(phase),
        dropped,
    )


def escape_at_once(_turn: int, _phase: Phase) -> Effect[Signal]:
    """A poll that records nothing."""
    yield from ()
    return Escape()


def test_an_escape_from_a_poll_that_records_nothing_has_no_name():
    handler = RecordingHandler({})
    ran = handler.run(
        lambda: run_turns([{"role": "user", "content": "go"}], interrupt=escape_at_once)
    )
    assert isinstance(ran, Ran)
    assert handler.trace == []
    ending = ran.trajectory
    assert isinstance(ending, Escaped)
    assert (ending.origin, ending.name, ending.partial) == ("escape", None, "")


def test_a_redirect_after_the_action_follows_its_observation():
    responses = {
        "d:0;react:turn": ACT,
        "d:0;tool:ls": ToolResult(content="a b"),
        "d:1;react:turn": ANSWER,
        **quiet_polls(0, 1),
        poll_key(0, "act"): Interrupted(redirect="only txt files"),
    }
    ran, _ = run(responses)
    ending = ran.trajectory
    assert isinstance(ending, Trajectory)
    assert ending.stop_reason == "finish"
    assert [(m["role"], m["content"]) for m in ran.messages][2:4] == [
        ("tool", "a b"),
        ("user", "[interrupt] only txt files"),
    ]
