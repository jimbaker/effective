"""The scripted path through `_funnel` — the combinator triage funnel, end to end.

Pins the properties the workflow shows: the emergent tape, the barrier park, the closed-book
lanes. Its sibling `test_funnel_sweep.py` covers the same workflow's SPACE.

**Nothing here spells a key.** Key text in a fixture is a maintenance cost when it is right and
a silent pass when it is wrong — a predicate that matches no key asserts nothing and still goes
green. So the assertions are derived two ways instead.

The **FOLD** carries the shape: project the tape onto the program and the counts *are* the funnel
— one op per lane per talk, the route's split across rubrics, one park per material flag, and no
frame spelling anywhere. The **properties** carry what a fold necessarily drops: which lane an op
sat in, the park's address, replay. Everything is computed from `_funnel.SUBMISSIONS`, so adding
a talk changes no expectation here.
"""

import _funnel as funnel
import pytest
from _keymap import including

from effective.api import GatherBranch, qualified_event_name
from effective.graphview import bare_name, fold_cycles, from_keys
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Segment, compose_key

# Journey role: one realistic path end to end, over the deepest composition the suite drives.
pytestmark = pytest.mark.journey


RULING = "uphold — the anecdote reads as an anecdote"


def _material() -> list[tuple[int, str]]:
    """`(lane index, talk ident)` for every talk the skeptic flags — the lanes that park."""
    return [(i, s.ident) for i, s in enumerate(funnel.SUBMISSIONS) if s.audit.startswith("flag")]


def _run() -> tuple[RecordingHandler, object, list[str]]:
    """Drive the funnel to completion, resuming each park in turn. Returns the handler, the
    result, and the tape — every expectation below is computed from these, never from a literal."""
    handler = RecordingHandler(responses=funnel.Answers(funnel.SUBMISSIONS))
    outcome = handler.run(lambda: funnel.triage("cfp-2026"))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(RULING)
    return handler, outcome, [entry.key.stored() for entry in handler.trace]


def _program(tape: list[str]):
    """The tape folded onto the program. `talk:` is this workflow's own unrolling scope (three
    lanes running identical code), so the CALLER declares it."""
    return fold_cycles(from_keys("cfp-2026", tape), keymap=including("tests/_funnel.py"))


def test_scripted_spine_runs_clean():
    """The scripted spine — parked, resumed, and asserted end to end."""
    result = funnel.scripted_run()
    assert result.review.startswith("uphold")


def test_material_flag_parks_in_the_lane():
    """The skeptic's material flag escalates IN THE LANE (the V1 await-in-gather park): the run
    parks on the lane's fully-qualified event, mid-fan-out.

    The expected address is COMPOSED by the same function an emitter would use, not spelled —
    so this compares identities rather than wire text."""
    handler = RecordingHandler(responses=funnel.Answers(funnel.SUBMISSIONS))
    parked = handler.run(lambda: funnel.triage("cfp-2026"))
    assert isinstance(parked, Suspended)

    (index, ident), *_ = _material()
    assert parked.awaiting == qualified_event_name(
        GatherBranch(0, index),
        compose_key(t"talk:{Segment(ident)}"),
        name="review:cfp-2026",
    )
    # nothing merges while parked, and the rank barrier certainly hasn't run
    assert not any("rank;" in entry.key.stored() for entry in handler.trace)


def test_the_tape_folds_back_into_the_program_and_replays():
    """The run writes its own graph, and folding it recovers the funnel: the counts are the
    structure. Nobody declared either — and the resumed tape REPLAYS, re-binding every op
    including the in-lane event to the identical result."""
    from effective.handlers.replay import ReplayHandler

    handler, result, tape = _run()
    counts = {node.key: node.count for node in _program(tape).nodes}
    n = len(funnel.SUBMISSIONS)

    assert counts["step;tool:inbox"] == 1  # the profile, once
    assert counts["step;skill:rubric,activate"] == 1  # rule 4: ONE disclosure above the fan-out
    assert counts["gather:*,*;talk:*;step:enrich"] == n  # one lane per talk...
    assert counts["gather:*,*;talk:*;step:classify"] == n
    assert counts["gather:*,*;talk:*;step:audit"] == n
    assert counts["gather:*,*;talk:*;event;review:cfp-2026"] == len(
        _material()
    )  # park iff escalation
    assert counts["rank;step:chunk"] == 1  # the decompose commits before the fan-out
    assert _program(tape).executions == len(tape)  # fewer nodes, never fewer facts

    assert ReplayHandler(handler.trace).run(lambda: funnel.triage("cfp-2026")) == result


def test_the_route_splits_the_lanes_across_the_rubrics_it_chose():
    """The classifier's label is in the tape (`score:{label}`), so replay re-dispatches
    identically — the routed path is data, not control state.

    Asserted as a fold: one `score:{track}` node per track actually used, each counting the
    talks that chose it. That is the dispatch being TOTAL over what the classifier produced."""
    _, _, tape = _run()
    # `bare_name` rather than a prefix: the question is which op ran, and the fold keeps the
    # lane frames in front of it.
    scored = {
        bare_name(node.key).removeprefix("step;score:"): node.count
        for node in _program(tape).nodes
        if bare_name(node.key).startswith("step;score:")
    }
    expected: dict[str, int] = {}
    for submission in funnel.SUBMISSIONS:
        expected[submission.track] = expected.get(submission.track, 0) + 1
    assert scored == expected
    assert sum(scored.values()) == len(funnel.SUBMISSIONS)  # every talk was scored exactly once


def test_answers_can_be_iterated_and_then_indexed():
    """`Answers` COMPUTES its values, so the Mapping half is the part a run never touches: the
    handler only ever subscripts it, so a `KeyError` from `dict(Answers(...))` would go unnoticed
    by every run.

    The lane ops are the interesting absence. `enrich` needs a `talk:` frame to say which lane is
    asking, so no bare key answers it — enumerating one would hand a caller the `KeyError` this
    test rules out for every name that IS enumerated."""
    answers = funnel.Answers(funnel.SUBMISSIONS)
    materialized = dict(answers)
    assert set(materialized) == set(answers)
    assert len(materialized) == len(answers)
    assert materialized["tool:inbox"] == [s.talk for s in funnel.SUBMISSIONS]

    assert "enrich" not in answers
    with pytest.raises(KeyError):
        answers["enrich"]


def test_lanes_are_closed_book_by_construction():
    """Structural isolation: every op inside a lane's sub-tape carries ONLY that lane's scope,
    so a lane physically cannot read a sibling's evidence.

    Every assertion sits inside a `continue` filter, so a filter that matches nothing would
    pass by examining nothing. The trailing count says how much the loop actually looked at:
    four ops per lane, plus a park in each material one."""
    _, _, tape = _run()
    idents = {s.ident for s in funnel.SUBMISSIONS}
    examined = 0
    for key in tape:
        if not (lanes := funnel.lanes_in(key)):
            continue
        examined += 1
        assert len(lanes) == 1, f"{key} sits in {len(lanes)} lanes"
        assert lanes[0] in idents, key
    # four ops per lane, plus the park in each material one
    assert examined == 4 * len(funnel.SUBMISSIONS) + len(_material()), tape
