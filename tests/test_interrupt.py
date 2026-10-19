"""Interrupt as a race around the turn, recorded.

The poll runs twice per iteration, pre-turn and post-turn, modeling an interrupt
concurrent with the in-flight turn without cancelling it:

- a *pending* interrupt wins **pre-turn**: the model is never called for that iteration
  (the saving a race buys over a post-only poll);
- an interrupt that lands while deciding wins **post-turn**: the turn's action is dropped.

Both are recorded `CallTool` polls, so the winner is a checkpoint and replay re-derives it.
"""

from _steps import placed_step

from effective import RecordingHandler, ReplayHandler
from effective.compose import Interrupted, tool_interrupt
from effective.interrupts import Phase, interrupt_key
from effective.react import (
    AssistantTurn,
    ToolRequest,
    ToolResult,
    Trajectory,
    run_agent,
)

ANSWER = AssistantTurn(thought="redirected", answer="answered per the interrupt")
ACT_SEARCH = AssistantTurn(thought="act", tool=ToolRequest(name="search", args={}))


def _poll(frame: str, phase: Phase) -> str:
    """The canned-table key that answers the interrupt poll at `phase` under `frame`."""
    return interrupt_key(phase).prefixed(frame).stored()


def counting_decide(calls, *turns):
    """A pure scripted step that records how many times it was consulted (so a skipped
    iteration is observable — proving a pre-turn interrupt skips the model call)."""
    script = iter(turns)

    def decide(request):
        level = request.level
        yield from ()
        calls.append(level.depth)
        return next(script)

    return decide


def test_pending_interrupt_wins_pre_turn_and_skips_the_model_call():
    calls: list[str] = []
    responses = {
        _poll("d:0;", "pre"): Interrupted(redirect="stop, just answer"),  # pending: wins
        _poll("d:1;", "pre"): Interrupted(),
        _poll("d:1;", "post"): Interrupted(),
    }
    h = RecordingHandler(responses)
    # iteration 0 is skipped before decide, so only one turn (the answer) is ever consumed
    decide = counting_decide(calls, ANSWER)
    traj = h.run(lambda: run_agent("q", decide=decide, interrupt=tool_interrupt()))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "answered per the interrupt"
    # iteration 0's decide was NEVER called — the pre-turn interrupt won first
    assert calls == [1]
    assert [e.key.stored() for e in h.trace] == [
        placed_step(interrupt_key("pre"), "tool", "d:0;"),  # pending -> wins, turn skipped
        placed_step(interrupt_key("pre"), "tool", "d:1;"),  # quiet
        placed_step(interrupt_key("post"), "tool", "d:1;"),  # quiet -> the turn proceeds to finish
    ]


def test_interrupt_during_the_turn_wins_post_turn_and_drops_the_action():
    calls: list[str] = []
    responses = {
        _poll("d:0;", "pre"): Interrupted(),  # quiet before the turn
        _poll("d:0;", "post"): Interrupted(redirect="stop searching"),  # landed during
        _poll("d:1;", "pre"): Interrupted(),
        _poll("d:1;", "post"): Interrupted(),
        # NO "d:0;tool:search" — the action must never run, or this KeyErrors
    }
    h = RecordingHandler(responses)
    decide = counting_decide(calls, ACT_SEARCH, ANSWER)
    traj = h.run(lambda: run_agent("q", decide=decide, interrupt=tool_interrupt()))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "answered per the interrupt"
    assert traj.steps[0].observation == "[interrupted] stop searching"
    assert calls == [0, 1]  # the turn ran (we can't cancel) but its action was dropped


def test_quiet_channel_leaves_loop_behavior_unchanged():
    def acting_decide():
        script = iter(
            [
                AssistantTurn(thought="act", tool=ToolRequest(name="search", args={})),
                AssistantTurn(thought="done", answer="found it"),
            ]
        )

        def decide(request):
            yield from ()
            return next(script)

        return decide

    quiet = {
        _poll("d:0;", "pre"): Interrupted(),
        _poll("d:0;", "post"): Interrupted(),
        _poll("d:1;", "pre"): Interrupted(),
        _poll("d:1;", "post"): Interrupted(),
        "d:0;tool:search": ToolResult(content="a result"),
    }
    h = RecordingHandler(quiet)
    traj = h.run(lambda: run_agent("q", decide=acting_decide(), interrupt=tool_interrupt()))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "found it"
    assert traj.steps[0].tool is not None
    assert traj.steps[0].tool.name == "search"
    assert traj.steps[0].observation == "a result"


def test_interrupted_trajectory_replays_deterministically():
    responses = {
        _poll("d:0;", "pre"): Interrupted(redirect="stop, just answer"),
        _poll("d:1;", "pre"): Interrupted(),
        _poll("d:1;", "post"): Interrupted(),
    }
    rec = RecordingHandler(responses)
    live = rec.run(
        lambda: run_agent("q", decide=counting_decide([], ANSWER), interrupt=tool_interrupt())
    )

    replayed = ReplayHandler(rec.trace).run(
        lambda: run_agent("q", decide=counting_decide([], ANSWER), interrupt=tool_interrupt())
    )

    assert replayed == live
    assert isinstance(replayed, Trajectory)
    assert replayed.answer == "answered per the interrupt"
