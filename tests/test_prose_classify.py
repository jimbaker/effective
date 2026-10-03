"""`CLASSIFY` must ENUMERATE, and `READ` must gather evidence before it asks.

Both are answers to the same defect, measured on this repo's own prose: a `CLASSIFY` verdict that
SUMMARIZED ("3 stay-compressed, 1 to the commit") let a pass through every gate while taking 12%
of a 55% cut, and nothing anywhere could disagree with it — `verdict_words` scored the shallow and
the deep pass identically, `doc_bloat` moved 33.0x to 29.0x and was read as a win, and the skeleton
check and `just check` are indifferent to how much prose survives.
"""

import pytest

from effective.domain import CallTool
from effective.keys import Segment
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx
from effective.ops import AwaitEvent, Step, StoreArtifact
from effective.prose import DESTINATIONS, State, build_prose_specs
from effective.prose.runners import MAX_ANSWER_ATTEMPTS, AnswerRefused, answered, decode


def test_a_classified_answer_must_name_a_destination_for_every_block():
    with pytest.raises(AnswerRefused, match="needs `blocks`"):
        decode(State.CLASSIFY, {"verdict": "classified", "summary": "3 stay, 1 to the commit"})


def test_an_invented_destination_is_refused():
    # Seven destinations, and only one of them deletes. A free-text answer would let a pass invent
    # "compressed" and never notice it had used two of the seven.
    with pytest.raises(AnswerRefused, match="not a destination"):
        decode(
            State.CLASSIFY,
            {"verdict": "classified", "blocks": [{"block": "intro", "to": "compressed"}]},
        )


def test_a_block_missing_its_destination_is_refused():
    with pytest.raises(AnswerRefused, match="needs 'block' and 'to'"):
        decode(State.CLASSIFY, {"verdict": "classified", "blocks": [{"block": "intro"}]})


def test_a_full_enumeration_is_accepted_and_lands_on_the_tape():
    verdict, summary, _ = decode(
        State.CLASSIFY,
        {
            "verdict": "classified",
            "summary": "Key: 8 blocks",
            "blocks": [
                {"block": "contract", "to": "shows"},
                {"block": "not-a-str-subclass", "to": "point"},
                {"block": "four-bites", "to": "commit"},
                {"block": "no-__str__", "to": "shows"},
                {"block": "two-text-positions", "to": "shows"},
                {"block": "named-exits", "to": "shows"},
                {"block": "source-map-someday", "to": "wiki8"},
                {"block": "the-pleasant-part", "to": "cut"},
            ],
        },
    )
    assert verdict.value == "classified"
    assert summary == "Key: 8 blocks"


def test_no_destination_needs_no_enumeration():
    # The whole point of NO_DESTINATION is that a block did not fit — demanding a destination for
    # it would make the arm unreachable.
    verdict, _, _ = decode(State.CLASSIFY, {"verdict": "no-destination", "summary": "block 7"})
    assert verdict.value == "no-destination"


def test_the_seven_destinations_are_the_rubrics_seven():
    assert len(DESTINATIONS) == 7
    assert set(DESTINATIONS) == {"shows", "101", "commit", "wiki8", "adr", "cut", "point"}


def test_READ_gathers_callers_before_it_parks():
    """The evidence CLASSIFY never had, as a RECORDED op rather than something run beside the run.

    Two ops in one state: the caller query, then the park. A replay re-serves the caller list, so
    the tape says which evidence the classification was made against.
    """
    spec = build_prose_specs("probe")[State.READ]
    # A real `Ctx` rather than `None` with a suppression: the repo's rule is to sharpen the
    # contract, and `None` here would also have been the mypy spelling of the pragma, which `ty`
    # ignores — so the suppression would not have suppressed anything either.
    ctx = Ctx(run_id=Segment("r1"), goal="de-essay", state=State.READ, visit=0)
    gen = spec.run(ctx)
    # `call_tool` wraps its op in a `Step`, so the tool name rides one level in. Asserting on the
    # WRAPPER would have passed against any tool at all.
    first = next(gen)
    assert isinstance(first, Step)
    assert isinstance(first.op, CallTool)
    assert first.op.name == "find_callers"
    second = gen.send(CommandRun(exit_code=0, output="handlers/base.py:390"))
    assert isinstance(second, AwaitEvent)
    assert "read" in second.name.display()


def test_a_refused_answer_asks_again_instead_of_killing_the_run():
    """The defect a designed refusal made worse than an accidental one.

    `decode` raising propagates out of the workflow, the postamble never runs, and the run dies
    with its work. Measured twice — a payload-schema mismatch, then the CLASSIFY enumeration this
    machine refuses ON PURPOSE. Rejecting an answer by destroying the run is a punishment, not a
    rejection.
    """
    gen = answered("probe", State.CLASSIFY)
    first = next(gen)
    assert isinstance(first, AwaitEvent)
    assert first.name.display() == "prose:probe,classify"  # byte-preserving at the first ask

    # a summary, which this state refuses
    recorded = gen.send({"verdict": "classified", "summary": "4 stay, 1 cut"})
    # Narrowed with an assert rather than read off the union: `WorkflowOp` has no `.value`, and
    # the repo's rule is to sharpen rather than suppress.
    assert isinstance(recorded, StoreArtifact)
    stored = recorded.value
    assert isinstance(stored, dict)
    assert "needs `blocks`" in stored["refused"]
    assert stored["attempt"] == 1

    second = gen.send("artifact-id")
    assert isinstance(second, AwaitEvent)
    # a DIFFERENT name, because the first event is durable and a replay would re-serve it forever
    assert second.name.display() == "prose:probe,classify#2"

    with pytest.raises(StopIteration) as done:
        gen.send({"verdict": "classified", "blocks": [{"block": "intro", "to": "shows"}]})
    assert done.value.value[0].value == "classified"


def test_the_asking_is_bounded():
    """A human who cannot produce a well-formed answer in five tries has a problem the machine
    cannot fix by asking again — the reason `Exhausted` is a bound rather than a loop."""
    gen = answered("probe", State.READ)
    next(gen)
    for attempt in range(1, MAX_ANSWER_ATTEMPTS + 1):
        recorded = gen.send({"nonsense": True})
        assert isinstance(recorded, StoreArtifact)
        stored = recorded.value
        assert isinstance(stored, dict)
        assert stored["attempt"] == attempt
        if attempt < MAX_ANSWER_ATTEMPTS:
            gen.send("artifact-id")
    with pytest.raises(AnswerRefused, match="answers refused in a row"):
        gen.send("artifact-id")
