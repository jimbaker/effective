"""Reconciling the two records of one run — the tape, and the telemetry beside it.

A durable run leaves two independent traces. The **tape** is `(key, result)` checkpoints and is
replayable; the **telemetry** is spans carrying cost, duration and the wire messages, and is not.
They share an address space — the placed key — which is what lets one be asked about the other.

The reason to bother is that **the tape cannot witness itself**. Replay reads the checkpoints, so
it agrees with them by construction: an op that ran live and never checkpointed is invisible to
every replay-based check, because replay never re-mints it. Telemetry was written at a different
seam while the run was live, so it sees what the record omits.

Both walks here are members of the tape-test family and return a `TapeVerdict`, so both carry
`witnessed` — the count that separates "the property held" from "the property never looked".
"""

from typing import Any

import pytest

from agent.tape_bank import bank_refusal
from effective.domain import AskLLM
from effective.layers import compose_domain
from effective.tape import (
    TapeVerdict,
    VacuousTape,
    Violation,
    collecting_layer,
    unmeasured_steps,
    wire_divergences,
)


class _K:
    """The whole of what these walks ask of a key: `.stored()`."""

    def __init__(self, text: str) -> None:
        self._t = text

    def stored(self) -> str:
        return self._t


class _Entry:
    """A hand-built `TraceEntry` stand-in — the walks duck-type `key`/`op`/`result`."""

    def __init__(self, key: str, op: Any, result: Any = None) -> None:
        self.key, self.op, self.result = _K(key), op, result


def _turn(key: str, *contents: str) -> _Entry:
    messages = [
        {"role": "user" if i == 0 else "assistant", "content": c} for i, c in enumerate(contents)
    ]
    return _Entry(key, AskLLM(messages=messages, response_schema=dict))


def _wire(key: str, *contents: str) -> dict[str, list[dict[str, str]]]:
    return {
        key: [
            {"role": "user" if i == 0 else "assistant", "content": c}
            for i, c in enumerate(contents)
        ]
    }


# --- A. did replay rebuild what was actually sent? -------------------------------------------


def test_a_faithful_replay_diverges_nowhere():
    """The premise the whole tape technique rests on, stated as a test.

    Everything else reads ops replay re-minted, on the grounds that the prompt is a function of
    the tape. That is derived from determinism; this is what measures it."""
    trace = [_turn("d:0;step;react:turn", "go"), _turn("d:1;step;react:turn", "go", "did a thing")]
    wire = {
        **_wire("d:0;step;react:turn", "go"),
        **_wire("d:1;step;react:turn", "go", "did a thing"),
    }
    wire_divergences(trace, wire).require_clean(at_least=2)


def test_a_changed_message_is_named_by_position():
    """A divergence says WHICH message, because "the transcripts differ" is not actionable on a
    nine-message turn."""
    trace = [_turn("d:1;step;react:turn", "go", "did a thing")]
    wire = _wire("d:1;step;react:turn", "go", "did something ELSE")
    verdict = wire_divergences(trace, wire)
    assert verdict.witnessed == 1
    assert [v.why for v in verdict.violations] == [
        "replay and the wire disagree at message(s) [1]"
    ]


def test_a_dropped_message_is_reported_as_a_count():
    """Length and content are different failures. A prefix that was truncated on the wire is not
    "message 2 differs"; there is no message 2."""
    trace = [_turn("d:1;step;react:turn", "go", "did a thing")]
    wire = _wire("d:1;step;react:turn", "go")
    verdict = wire_divergences(trace, wire)
    assert [v.why for v in verdict.violations] == [
        "replay rebuilt 2 message(s), the wire recorded 1"
    ]


def test_an_emitter_is_not_compared_at_all():
    """`write:`/`edit:` calls are CHANNEL renders, not transcripts.

    Measured on a real tape: the emitter's `messages` is a `dict`, while the wire records the
    span file's text rendering of it. Comparing those is comparing two renderings and calling the
    difference a defect — the same mistake that made the span sidecar unusable for the transcript
    property in the first place. So the emitter is out of scope by ADDRESS, and a wire entry for
    it cannot manufacture a violation."""
    trace = [_turn("d:0;step;react:turn", "go"), _turn("step;write:0", "totally different")]
    wire = {**_wire("d:0;step;react:turn", "go"), **_wire("step;write:0", "not this at all")}
    verdict = wire_divergences(trace, wire)
    assert verdict.witnessed == 1  # the emitter was not counted, not merely forgiven
    assert verdict.violations == ()


def test_a_sidecar_from_another_run_is_vacuous_rather_than_clean():
    """The failure this walk is most likely to have in practice.

    A sidecar whose keys do not match the tape compares nothing and returns no violations — which
    reads exactly like success. `witnessed` is what turns it into a loud failure instead."""
    trace = [_turn("d:0;step;react:turn", "go")]
    verdict = wire_divergences(trace, _wire("d:9;step;react:turn", "go"))
    assert verdict.violations == ()
    assert verdict.witnessed == 0
    with pytest.raises(VacuousTape):
        verdict.require_clean()


def test_a_transcript_that_stops_being_a_message_list_goes_vacuous_not_green():
    """A shape guard that fails loudly. If a turn's messages become a `Template` or a dict, this
    walk cannot compare them — and the honest outcome is a vacuous verdict the caller's floor
    rejects, not a quiet pass."""
    trace = [
        _Entry("d:0;step;react:turn", AskLLM(messages={"prompt": "go"}, response_schema=dict))
    ]
    verdict = wire_divergences(trace, _wire("d:0;step;react:turn", "go"))
    assert verdict.witnessed == 0
    with pytest.raises(VacuousTape):
        verdict.require_clean()


# --- B. did a domain op run unmeasured? -------------------------------------------------------


def test_every_step_measured_is_clean():
    tape = ["d:0;step;react:turn", "d:0;step;tool:run", "ledger;commit:run-1"]
    measured = ["d:0;step;react:turn", "d:0;step;tool:run"]
    unmeasured_steps(tape, measured).require_clean(at_least=2)


def test_a_step_with_no_span_is_a_defect():
    """The cost leak nothing else looks for.

    `traced` sits on the domain seam, so every `step` node had a domain op run under it. A step
    the telemetry never saw is either an unmetered model call — money spent that no cost report
    can show — or one op minting two different addresses, which is the join silently breaking."""
    tape = ["d:0;step;react:turn", "d:1;step;react:turn"]
    verdict = unmeasured_steps(tape, ["d:0;step;react:turn"])
    assert verdict.witnessed == 2
    assert [v.where for v in verdict.violations] == ["d:1;step;react:turn"]


def test_ledger_artifact_sleep_and_await_nodes_are_excused_by_kind():
    """Not forgiven case by case — excluded by KIND, because `traced` is a domain layer and none
    of these is a domain op. Verified on a real run: every left orphan was an artifact or a
    ledger row."""
    tape = [
        "ledger;commit:run-1",
        "artifact:text/plain,sha256-bc6f2dfef7b413bd",
        "sleep:epoch-1787614493",
        "event;review:m1",
        "d:0;step;react:turn",
    ]
    verdict = unmeasured_steps(tape, ["d:0;step;react:turn"])
    assert verdict.violations == ()
    assert verdict.witnessed == 1  # only the step was ever a candidate


def test_scope_frames_do_not_change_the_kind():
    """A step inside a state visit or a gather branch is still a step. `kind_of` strips frames
    first, which is why this walk asks it rather than reading the leading term."""
    tape = ["visit:1;state:draft;d:2;step;react:turn", "gather:0,0;step;tool:x"]
    verdict = unmeasured_steps(tape, [])
    assert verdict.witnessed == 2
    assert len(verdict.violations) == 2


def test_a_tape_of_no_steps_is_vacuous_rather_than_clean():
    """A tape with nothing on the domain seam proves nothing about metering, and says so."""
    verdict = unmeasured_steps(["ledger;commit:run-1", "artifact:text/plain,sha256-aa"], [])
    assert verdict.violations == ()
    with pytest.raises(VacuousTape):
        verdict.require_clean()


# --- what the banker refuses -----------------------------------------------------------------


def test_a_tape_that_violates_the_property_is_refused_not_stamped():
    """The banker's claim about re-banking, made checkable.

    Everywhere this tape is described — `tape.py`, `tape_bank.py`, `test_banked_tape.py`,
    `bank_tape.py` — it is called an INPUT rather than a golden output, and re-banking is called
    safe because a freshly banked tape that violates the property still fails. That is only true
    if BANKING checks, and it did not: `bank()` computed the whole verdict, read `.witnessed` off
    it for the census, and discarded the violations. A tape that failed the walk was stamped and
    copied over the good one.

    The census and the property are different questions — *is this worth walking?* and *did it
    pass?* — and the first was standing in for both."""
    failing = TapeVerdict((Violation("d:1;step;react:turn", "cannot see the action it chose"),), 5)

    refusal = bank_refusal(failing, actions=7)

    assert refusal is not None
    assert "VIOLATES" in refusal
    assert "d:1;step;react:turn" in refusal, "the refusal must name the site, not just the count"


def test_a_clean_and_substantial_tape_is_banked():
    """The positive arm, so the refusal cannot pass by refusing everything."""
    assert bank_refusal(TapeVerdict((), 5), actions=7) is None


def test_a_vacuous_tape_is_refused_for_being_vacuous_not_for_failing():
    """A give-up run has turns and no actions. It is refused, and the reason distinguishes it from
    a defective loop — one is a re-run, the other is a bug."""
    refusal = bank_refusal(TapeVerdict((), 5), actions=0)

    assert refusal is not None
    assert "vacuous" in refusal
    assert "VIOLATES" not in refusal


class _NoInterpreter:
    """A base the guard never reaches — it refuses at assembly, before anything runs."""

    def run(self, op: object) -> object:
        raise AssertionError("the seam guard should refuse before any op is interpreted")


def test_an_op_layer_installed_at_the_domain_seam_is_refused():
    """`collecting_layer`'s docstring says "IT MUST BE AN OP LAYER" and explains why at length.
    That was a sentence; this is the guard.

    The wrong install is silent in the worst direction. An op-layer composed into the domain stack
    is accepted and observes everything on a LIVE pass, so the run you would sanity-check it with
    looks perfect; on REPLAY it collects nothing, because the domain call sits inside the
    checkpoint thunk and `ctx.step` returns the committed row without running it. Measured: op
    seam 2 entries, domain seam 0, same script. And replay is the pass the tape walk
    exists to read.

    The marker `@op_layer` stamps is read in exactly one place in the tree."""
    with pytest.raises(TypeError, match="op_layer"):
        compose_domain((collecting_layer([]),), _NoInterpreter())


def test_an_unmarked_callable_is_still_accepted_at_either_seam():
    """The guard refuses a POSITIVE mismatch, not the absence of a claim.

    A layer is an ordinary generator function and plenty are built without the decorator — a
    closure returned by a factory, a test double. Refusing those would make the marker mandatory,
    which is a much larger change than closing the silent-wrong-seam hole."""

    def undecorated(op):
        result = yield op
        return result

    assert compose_domain((undecorated,), _NoInterpreter()) is not None
