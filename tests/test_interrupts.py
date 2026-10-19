"""The interrupt signal, the phases a poll subscribes to, and a turn's answer to each signal."""

from typing import Any

import pytest

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


POLL_KEYS = {
    (0, "pre"): "d:0;tool:interrupt,pre",
    (0, "post"): "d:0;tool:interrupt,post",
    (0, "act"): "d:0;tool:interrupt,act",
    (1, "pre"): "d:1;tool:interrupt,pre",
    (1, "post"): "d:1;tool:interrupt,post",
    (1, "act"): "d:1;tool:interrupt,act",
}
"""The canned key of each poll a two-turn run makes, by depth and phase."""

POLL_NAMES = {
    "pre": "tool:interrupt,pre",
    "post": "tool:interrupt,post",
    "act": "tool:interrupt,act",
}
"""The authored name of each phase's poll, which an escape it reports carries."""


def poll_key(depth: int, phase: str) -> str:
    return POLL_KEYS[depth, phase]


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
    assert keys(between) == [
        "d:0;step;tool:interrupt,pre",
        "d:0;step;react:turn",
        "d:0;step;tool:interrupt,post",
    ]
    _, every = run(
        {"d:0;react:turn": ACT, "d:0;tool:ls": ToolResult(content="a b"), "d:1;react:turn": ANSWER}
        | quiet_polls(0, 1)
    )
    assert "d:0;step;tool:interrupt,act" in keys(every)


ESCAPES = {
    "pre": (["d:0;step;tool:interrupt,pre"], [("user", "go"), ("user", ESCAPED)]),
    "post": (
        ["d:0;step;tool:interrupt,pre", "d:0;step;react:turn", "d:0;step;tool:interrupt,post"],
        [("user", "go"), ("assistant", "look"), ("user", ESCAPED)],
    ),
    "act": (
        [
            "d:0;step;tool:interrupt,pre",
            "d:0;step;react:turn",
            "d:0;step;tool:interrupt,post",
            "d:0;step;tool:ls",
            "d:0;step;tool:interrupt,act",
        ],
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
def test_an_escape_ends_the_run_at_the_phase_it_arrives(phase: str):
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
        POLL_NAMES[phase],
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
        "d:0;tool:interrupt,act": Interrupted(redirect="only txt files"),
    }
    ran, _ = run(responses)
    ending = ran.trajectory
    assert isinstance(ending, Trajectory)
    assert ending.stop_reason == "finish"
    assert [(m["role"], m["content"]) for m in ran.messages][2:4] == [
        ("tool", "a b"),
        ("user", "[interrupt] only txt files"),
    ]
