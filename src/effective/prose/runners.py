"""What each state DOES — and here, almost every state asks a human through a terminal.

**The worker is a person, and that is the design rather than a placeholder.** The coding machine's
workers are ReAct loops driving a model; this machine's are parks. A de-essaying judgment is what
a rubric exists to make repeatable and has not yet made mechanical, so the honest embodiment is one
where every judgment arrives from outside and lands on the tape, verbatim, under a key that says
which state and which region asked. That is what makes a later claim about the rubric checkable
instead of remembered — and it is why this machine is worth running before any of it is automated.

**`VERIFY` is the exception, and the asymmetry is the point.** Its verdict is DERIVED from a
measurement rather than asserted by anyone, exactly as `coding.verdicts` derives three of its own:
a `match` over a `CommandRun`, not an opinion. A machine where every verdict is a human's is a
transcript; one where the mechanical half is mechanical is a machine.

**Every park name is composed, never spelled.** `compose_key(t"prose:{…},{…}")` rather than an
f-string, because a park name is an IDENTITY: it is what replay binds to and what a driver has to
address from outside the run. It is also subject-scoped rather than run-scoped, which is
`await_event`'s one authoring rule — a name carrying the run id is refused on any fork.
"""

from collections.abc import Callable
from typing import Any

from effective.api import Effect, await_event, call_tool, store_artifact
from effective.keys import Name, Subject, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Report
from effective.prose.states import VERDICTS, State, Verdict, VerifyVerdict


class AnswerRefused(ValueError):
    """A payload delivered to a park did not decode into that state's fibre.

    Refused loudly rather than defaulted, for the reason `VerdictOutOfDomain` is: the incumbent's
    defect was that an unrecognised outcome silently took an edge, and the edge it took appended
    to the canonical record every visit. A driver that mistypes a verdict gets an error, not a
    lap."""


MAX_ANSWER_ATTEMPTS = 5
"""How many refused answers one state will take before the run gives up.

A bound rather than a loop, for the reason `Exhausted` is one: a human who cannot produce a
well-formed answer in five tries has a problem the machine cannot fix by asking again."""


def answered(subject: str, state: State) -> Effect[tuple[Verdict, str, str]]:
    """Ask, and keep asking while the answer is refused — WITHOUT killing the run.

    **A refused answer is not fatal, and a designed refusal is why.** If `decode` raised out of
    the workflow, the trampoline's postamble would never run and the whole run would die with its
    work, both on a payload-schema mismatch and on the `CLASSIFY` enumeration this machine refuses
    ON PURPOSE. Rejecting a bad answer by destroying the run punishes rather than rejects.

    Re-awaiting the SAME name cannot work: the event is durable, so a replay re-serves the bad
    payload forever. `Key.occurrence` is what makes a second ask a different question, and it is
    byte-preserving at the first ask, so a run that answers
    correctly composes the name it always composed.

    The refusal itself is recorded with `store_artifact`, so a driver reads WHY in the run view
    instead of in the database. Deterministic: the value is derived from a recorded answer, so a
    replay re-derives the same artifact."""
    for attempt in range(1, MAX_ANSWER_ATTEMPTS + 1):
        answer = yield from await_event(park_name(subject, state).occurrence(attempt), dict)
        try:
            return decode(state, answer)
        except AnswerRefused as exc:
            yield from store_artifact(
                {"state": state.value, "attempt": attempt, "refused": str(exc)},
                "application/json",
            )
    raise AnswerRefused(
        f"{state.value}: {MAX_ANSWER_ATTEMPTS} answers refused in a row — the run stops rather "
        f"than asking a sixth time"
    )


def park_name(subject: str, state: State):
    """The event a driver answers to advance this state.

    Arity two rather than a path, so the two coordinates stay separately addressable: a surface
    listing every open question for one region filters on the subject, and the grammar gives it
    that for free. A second visit to the same state under the same subject is distinguished by the
    key's own `#N` occurrence suffix, which is exactly what that suffix is for — so a back edge
    needs no coordinate of its own here."""
    return compose_key(t"prose:{Subject(subject)},{Name(state.value)}")


def decode(state: State, body: Any) -> tuple[Verdict, str, str]:
    """Turn what a human answered into this state's verdict, plus what they said about it.

    Takes the DECODED payload, not a JSON string, and the difference cost a whole run: a client
    that parses the payload before appending it — which the terminal run view does, so that a
    malformed answer is refused at the keyboard rather than inside the workflow — delivers a
    `dict`, and a park awaiting a `str` refuses it at the schema. The run died with a Pydantic
    validation error and the view rendered it as an empty tree. So the seam is a mapping, and this
    function owns the shape.

    The verdict is looked up through `VERDICTS`, so the vocabulary has ONE mint: a state that grows
    an arm is answerable the same day, and a state that loses one refuses the stale spelling here
    rather than routing it."""
    if not isinstance(body, dict) or "verdict" not in body:
        raise AnswerRefused(f"{state.value}: answer needs a 'verdict' key, got {body!r}")
    if state is State.CLASSIFY and body["verdict"] == "classified":
        _require_blocks(body)
    try:
        verdict = VERDICTS[state](body["verdict"])
    except ValueError as exc:
        allowed = ", ".join(v.value for v in VERDICTS[state])
        raise AnswerRefused(f"{state.value}: {body['verdict']!r} is not one of {allowed}") from exc
    return verdict, str(body.get("summary", "")), str(body.get("detail", ""))


def read_run(subject: str) -> Callable[[Ctx[State, CommandRun]], Effect[Any]]:
    """`READ` runs the caller query, THEN parks — so the human inventories with the evidence.

    Two ops in one state rather than a state of its own, because the caller list is not a
    judgment anyone rules on: it is what the next judgment is made against. Splitting it would
    buy an edge in the transition table and a budget line for a lookup."""

    def run(ctx: Ctx[State, CommandRun]) -> Effect[Any]:
        callers = yield from call_tool(CALLERS_TOOL, {"subject": subject}, CommandRun)
        verdict, summary, detail = yield from answered(subject, State.READ)
        return Report(
            verdict=verdict,
            summary=summary,
            detail=f"{detail}\n\ncallers:\n{callers.output}".strip(),
            measured=callers,
        )

    return run


def _require_blocks(body: dict[str, Any]) -> None:
    """A `CLASSIFIED` answer must name a destination for EVERY block, and the refusal is the point.

    Both looking-back passes in this repo work by forcing an enumeration — a total `match` makes
    you name every arm, and showing makes you name the mechanism — and failing to produce one IS
    the finding. A `CLASSIFY` verdict that summarizes ("3 stay-compressed, 1 to the commit") lets a
    shallow pass through every gate: measured, one such answer produced a 12% cut where 55% was
    available, and `verdict_words`, `doc_bloat`, the skeleton check and `just check` all said yes.

    So the answer carries `blocks`, and the machine refuses anything else. The enumeration lands on
    the tape, which is what makes a later claim about the rubric checkable."""
    blocks = body.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise AnswerRefused(
            "classify: a 'classified' answer needs `blocks`: a list of "
            "{'block': <name>, 'to': <destination>} naming EVERY block in the region. "
            f"Destinations: {', '.join(DESTINATIONS)}"
        )
    for entry in blocks:
        if not isinstance(entry, dict) or "block" not in entry or "to" not in entry:
            raise AnswerRefused(f"classify: each block needs 'block' and 'to', got {entry!r}")
        if entry["to"] not in DESTINATIONS:
            raise AnswerRefused(
                f"classify: {entry['to']!r} is not a destination; "
                f"use one of {', '.join(DESTINATIONS)}"
            )


def park_run(subject: str, state: State) -> Callable[[Ctx[State, CommandRun]], Effect[Any]]:
    """A state whose whole body is one park. The human is the worker AND the judge.

    Fused by hand rather than through `specs.fuse`, because there is nothing to fuse: a park does
    not produce evidence for someone else to rule on, it produces the ruling. `specs.fuse` is a
    construction an embodiment may decline, and this is an embodiment declining it."""

    def run(ctx: Ctx[State, CommandRun]) -> Effect[Any]:
        verdict, summary, detail = yield from answered(subject, state)
        return Report(verdict=verdict, summary=summary, detail=detail)

    return run


CALLERS_TOOL = "find_callers"
"""The tool `READ` runs before it parks — who actually uses the unit under the docstring.

**The evidence CLASSIFY never had.** A docstring answers *how and why would I use this*; the type
says a little and the code says what; nothing in the rubric or the gates supplies the why. Three
passes over one property cut its caveat down and none noticed the USE was absent, because nobody
had asked who calls it.

A recorded op rather than something the driver runs beside the machine, for R6's reason: the
caller list is what the classification was made against, so a replay re-serves it and the tape says
which evidence judged which region."""

DESTINATIONS = (
    "shows",
    "101",
    "commit",
    "wiki8",
    "adr",
    "cut",
    "point",
)
"""The rubric's seven, as data — so `CLASSIFY` can be made to ENUMERATE rather than to summarize.

`shows` stays in place compressed; `101`, `commit`, `wiki8`, `adr` and `point` RELOCATE; only `cut`
deletes. That five of seven relocate is why a good pass drops a third of the words without losing
an argument, and why naming a destination per BLOCK is the work — a per-region verdict records none
of it. `Key`'s own docstring needs six of the seven across eight blocks."""

VERIFY_TOOL = "prose_gates"
"""The tool a handler serves for `VERIFY` — the five mechanical checks, run as one command.

One tool rather than five, because the verdict is a fold over all of them and a state that yielded
five ops would put the fold in the workflow, where a replay has to re-derive it. The handler runs
them and returns one `CommandRun`; which checks it ran is the deployment's business, since a
tunable seam takes data."""


def verify_run(subject: str) -> Callable[[Ctx[State, CommandRun]], Effect[Any]]:
    """Run the gates and DERIVE a verdict. The one state no human answers."""

    def run(ctx: Ctx[State, CommandRun]) -> Effect[Any]:
        measured = yield from call_tool(VERIFY_TOOL, {"subject": subject}, CommandRun)
        return Report(
            verdict=verdict_for_verify(measured),
            summary=f"gates exit {measured.exit_code}",
            detail=measured.output[:400],
            measured=measured,
        )

    return run


def verdict_for_verify(run: CommandRun) -> VerifyVerdict:
    """`CommandRun` -> `VerifyVerdict`, and the ORDER is the content.

    A collection error comes first because it is a fact about the environment and says nothing
    about the prose — reading it as "the checks failed" is how a vanished container gets mistaken
    for a bad docstring and sends the machine to DRAFT to fix writing that is fine.

    Then the skeleton, before the rewordable failures: an edit that moved code is not a smaller
    version of a line that is too long, and telling them apart is what stops a driver from tidying
    prose around a change it should be reverting."""
    if run.collection_error is not None:
        return VerifyVerdict.BROKEN_ENV
    if run.failed("skeleton"):
        return VerifyVerdict.NOT_PROSE_ONLY
    if not run.green:
        return VerifyVerdict.MECHANICAL
    return VerifyVerdict.CLEAN


__all__ = [
    "VERIFY_TOOL",
    "AnswerRefused",
    "answered",
    "decode",
    "park_name",
    "park_run",
    "read_run",
    "verdict_for_verify",
    "verify_run",
]
