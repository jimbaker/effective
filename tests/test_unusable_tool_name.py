"""A model-chosen tool name that cannot BE a key routes like a denial.

`default_act` composes `tool:{name}` from a MODEL-controlled value. The composer refuses a name
it cannot make an atom of: that guarantee is the reason the site composes instead of
f-stringing. The refusal needs somewhere to go: `compose_key` raises `KeySyntaxError`, a bare
`ValueError`, and an `except Refused` arm catches permission denials only, so without its own
arm a tool named `7zip` takes the whole run down.

**The refusal class is wider than "digit-led", which is the part worth pinning.** `Segment`
refuses a *delimiter*; the atom rule refuses anything that is not a well-formed atom. So a
leading digit, a space, and an interior `;` all arrive here by different doors and must route
identically — *"no delimiter" was never the safety property.*

**Where this differs from `[denied]`, and why the trace assertion below is the one that
matters.** A `Refused` is delivered INTO the workflow by a layer while the op runs, so the
denied op is in the trace as an error entry (`test_refused_routing`). `UnusableToolName` is
raised while evaluating `step`'s *argument*, before any op is yielded, so a bad name leaves
NO trace entry at all. Asserting the observation alone would pass on an implementation that
recorded a phantom op; asserting the trace is what pins the shape.

Role: adversarial. Every name here is an attack on the composer, and a pass means only that it
failed. Mutation check: delete the `except UnusableToolName` arm in `react._acted` and this
module must redden (5 tests fail).
"""

import pytest

from effective import RecordingHandler, ReplayHandler
from effective.keys.grammar import parse
from effective.ops import CompositionRefused, Unretryable
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent, tool_key

pytestmark = pytest.mark.adversarial

# Each is refused by the composer, and they do NOT all fail the same way — which is the point.
UNUSABLE = [
    ("7zip", "a NAME must open with a letter"),
    ("2fa_check", "digit-led, and the underscore is legal — the leading `2` is not"),
    ("rm -rf", "a space is in no atom kind"),
    ("a;b", "the frame delimiter, refused one layer earlier by `Segment`"),
]

USABLE = ["sh", "s3-sync", "x264"]


def _decide(bad_name: str):
    """Ask for `bad_name`, then recover with `sh`, then answer."""
    script = iter(
        [
            AssistantTurn(thought="reach for it", tool=ToolRequest(name=bad_name, args={})),
            AssistantTurn(thought="rerouting", tool=ToolRequest(name="sh", args={})),
            AssistantTurn(thought="done", answer="used the shell instead"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(script)

    return decide


@pytest.mark.parametrize(("name", "why"), UNUSABLE, ids=[n for n, _ in UNUSABLE])
def test_an_unusable_tool_name_is_an_observation_not_a_crash(name: str, why: str) -> None:
    h = RecordingHandler({"d:1;tool:sh": ToolResult(content="shell result")})
    traj = h.run(lambda: run_agent("solve it", decide=_decide(name)))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "used the shell instead", f"the loop did not survive {name!r} ({why})"

    observation = traj.steps[0].observation or ""
    assert observation.startswith("[unusable tool name] "), observation
    assert repr(name) in observation, "the observation must name the culprit the model chose"

    assert traj.steps[1].observation == "shell result"

    # The refusal happens while evaluating `step`'s argument, so NO op was ever yielded for it.
    assert [e.key.stored() for e in h.trace] == ["d:1;step;tool:sh"]


@pytest.mark.parametrize("name", USABLE)
def test_an_ordinary_tool_name_still_composes(name: str) -> None:
    """The anti-vacuity leg: a rule that refused everything would pass the test above.

    Asserted by parsing: the key is one `tool` term whose coordinate is the name."""
    terms = parse(tool_key(name).stored()).terms
    assert [t.tag for t in terms] == ["tool"]
    assert terms[0].coordinates[0].atoms[0].text == name


def test_the_refusal_is_a_programming_error() -> None:
    """It subclasses `CompositionRefused`, so it is `Unretryable` by construction.

    Not decoration: a ReAct loop runs inside spawned children, and a child that raises an
    unretryable error answers its parent `Failed` and fails once, where a crash is retried to
    death and parks the joining parent. A bare `ValueError` would be that crash.
    """
    with pytest.raises(CompositionRefused) as caught:
        tool_key("7zip")

    assert isinstance(caught.value, Unretryable)


def test_the_routed_trajectory_replays_deterministically() -> None:
    """Nothing is recorded for the bad turn, so replay must RE-DERIVE the refusal.

    That is the property the pure-function argument buys: `tool_key` is a function of the name
    alone, so re-executing the workflow reaches the same refusal at the same place with no
    recorded entry to consult.
    """
    rec = RecordingHandler({"d:1;tool:sh": ToolResult(content="shell result")})
    live = rec.run(lambda: run_agent("solve it", decide=_decide("7zip")))

    replayed = ReplayHandler(rec.trace).run(lambda: run_agent("solve it", decide=_decide("7zip")))

    assert replayed == live
    assert (replayed.steps[0].observation or "").startswith("[unusable tool name] ")
