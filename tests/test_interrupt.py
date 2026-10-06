"""Interrupt as a race around the turn, recorded.

The poll runs twice per iteration, pre-turn and post-turn, modeling an interrupt
concurrent with the in-flight turn without cancelling it:

- a *pending* interrupt wins **pre-turn**: the model is never called for that iteration
  (the saving a race buys over a post-only poll);
- an interrupt that lands while deciding wins **post-turn**: the turn's action is dropped.

Both are recorded `CallTool` polls, so the winner is a checkpoint and replay re-derives it.
"""

from effective import RecordingHandler, ReplayHandler
from effective.compose import Interrupted, tool_interrupt
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent

ANSWER = AssistantTurn(thought="redirected", answer="answered per the interrupt")
ACT_SEARCH = AssistantTurn(thought="act", tool=ToolRequest(name="search", args={}))


def counting_decide(calls, *turns):
    """A pure scripted step that records how many times it was consulted (so a skipped
    iteration is observable — proving a pre-turn interrupt skips the model call)."""
    script = iter(turns)

    def decide(messages, level):
        yield from ()
        calls.append(level.depth)
        return next(script)

    return decide


def test_pending_interrupt_wins_pre_turn_and_skips_the_model_call():
    calls: list[str] = []
    responses = {
        "d:0;tool:interrupt,pre": Interrupted(redirect="stop, just answer"),  # pending: wins
        "d:1;tool:interrupt,pre": Interrupted(),
        "d:1;tool:interrupt,post": Interrupted(),
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
        "d:0;step;tool:interrupt,pre",  # pending -> wins, turn skipped
        "d:1;step;tool:interrupt,pre",  # quiet
        "d:1;step;tool:interrupt,post",  # quiet -> the turn proceeds to finish
    ]


def test_interrupt_during_the_turn_wins_post_turn_and_drops_the_action():
    calls: list[str] = []
    responses = {
        "d:0;tool:interrupt,pre": Interrupted(),  # quiet before the turn
        "d:0;tool:interrupt,post": Interrupted(redirect="stop searching"),  # landed during
        "d:1;tool:interrupt,pre": Interrupted(),
        "d:1;tool:interrupt,post": Interrupted(),
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

        def decide(messages, tag):
            yield from ()
            return next(script)

        return decide

    quiet = {
        "d:0;tool:interrupt,pre": Interrupted(),
        "d:0;tool:interrupt,post": Interrupted(),
        "d:1;tool:interrupt,pre": Interrupted(),
        "d:1;tool:interrupt,post": Interrupted(),
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
        "d:0;tool:interrupt,pre": Interrupted(redirect="stop, just answer"),
        "d:1;tool:interrupt,pre": Interrupted(),
        "d:1;tool:interrupt,post": Interrupted(),
    }
    rec = RecordingHandler(responses)
    live = rec.run(
        lambda: run_agent("q", decide=counting_decide([], ANSWER), interrupt=tool_interrupt())
    )

    replayed = ReplayHandler(rec.trace).run(
        lambda: run_agent("q", decide=counting_decide([], ANSWER), interrupt=tool_interrupt())
    )

    assert replayed == live
    assert replayed.answer == "answered per the interrupt"
