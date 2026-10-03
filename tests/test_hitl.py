"""HITL through the agent loop: an Ask tool is a tool that suspends.

The loop parks on `await_event` when the step calls the Ask tool, returns a
`Suspended`, and resumes when the human answer is delivered — threading the answer
back as the next observation. The whole trajectory (including the human turn) then
replays deterministically with no model and no human.
"""

from agent.compose import ASK_HUMAN, HumanAnswer, make_act
from effective import RecordingHandler, ReplayHandler, Suspended
from effective.react import AssistantTurn, ToolRequest, Trajectory, run_agent


def make_decide():
    """A fresh scripted step each call (a pure policy; the iterator must not be shared
    across a record + replay run)."""
    script = iter(
        [
            AssistantTurn(thought="I should ask", tool=ToolRequest(name=ASK_HUMAN, args={})),
            AssistantTurn(thought="got it", answer="the user said: ship it"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(script)

    return decide


def test_loop_suspends_on_ask_and_resumes_with_the_human_answer():
    h = RecordingHandler()  # no canned `ask:0` event -> the loop must park inside turn 0's frame
    parked = h.run(
        lambda: run_agent("decide whether to ship", decide=make_decide(), act=make_act())
    )

    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "d:0;ask:0"

    resumed = parked.resume(HumanAnswer(text="ship it"))

    assert isinstance(resumed, Trajectory)
    assert resumed.answer == "the user said: ship it"
    # the human answer is threaded in as the observation of the ask step
    assert resumed.steps[0].observation == "ship it"
    assert [e.key.stored() for e in h.trace] == ["d:0;event;ask:0"]


def test_hitl_trajectory_replays_without_model_or_human():
    rec = RecordingHandler()
    parked = rec.run(lambda: run_agent("q", decide=make_decide(), act=make_act()))
    assert isinstance(parked, Suspended)
    live = parked.resume(HumanAnswer(text="ship it"))

    # replay gets only the recorded trace — no canned event, no human in the loop
    replayed = ReplayHandler(rec.trace).run(
        lambda: run_agent("q", decide=make_decide(), act=make_act())
    )

    assert replayed == live
    assert replayed.steps[0].observation == "ship it"
