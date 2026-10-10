"""Tape tests — a declared property, read off a completed tape, checked offline.

**The family, and why it has a home now.** Two of these arrived independently:
`machine.canonical_violations` (which states touched the append-only ledger without declaring
they would) and the transcript walk below (could turn `i` see what turn `i-1` did). Both take a
DECLARATION and a finished tape and return the places the tape contradicts the declaration; both
are checked offline rather than at run time for the same reason — the tape is only complete once
the run is over, so a runtime check could only fire after the damage.

The nearest established name is **runtime verification** (checking an execution trace against a
specification), except offline and repeatable. It is adjacent to golden/approval testing and
differs on the point that matters: the assertion is a PROPERTY every valid trace must satisfy, not
equality with a blessed artifact. That is what makes re-banking an input safe here — a freshly
banked tape that violates the property still fails, where re-blessing a golden file would silently
redefine correct.

**What this module adds over the two functions it unifies is `witnessed`.** A walk that returns no
violations has two very different meanings — the property held, or the property never got to look —
and a bare `list[str]` cannot tell them apart. The transcript walk scopes by address, so a
give-up trajectory with no ReAct turns scored nothing and returned `[]`, reading as clean. Counting
the places the property actually looked is what makes `TapeVerdict.require_clean` able to refuse
that, and it is the wiki §8 rule in the property rather than only in a fixture: **assert a count,
not a presence.**
"""

from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, assert_never

from effective.domain import AskLLM, CallTool
from effective.graphview import kind_of
from effective.handlers.base import TraceEntry
from effective.keys.grammar import KeySyntaxError, parse
from effective.layers import current_placement, op_layer
from effective.react import action_line, summary_message


@dataclass(frozen=True, slots=True)
class Violation:
    """One place a completed tape contradicted the declaration.

    `where` is an ADDRESS the tape actually carries — a placed key, or a state name — so a reader
    can go look at it. `why` is the sentence; keeping the two apart is what lets a caller group by
    site without parsing a message back apart."""

    where: str
    why: str

    def __str__(self) -> str:
        return f"{self.where}: {self.why}"


class VacuousTape(AssertionError):
    """A walk that found nothing because there was nothing to find.

    An `AssertionError` because that is what a suite expects to see, and named because "no
    violations" is exactly what a passing run looks like — the whole hazard is that vacuity is
    indistinguishable from success unless something says so."""


@dataclass(frozen=True, slots=True)
class TapeVerdict:
    """What a walk found, and how much of the tape it got to look at.

    The second field is the point. `violations` empty is not the same claim as "the property
    held"; it is only that claim when `witnessed` is greater than zero."""

    violations: tuple[Violation, ...]
    witnessed: int

    def require_clean(self, *, at_least: int = 1) -> None:
        """Raise unless the property both HELD and had something to hold over.

        `at_least` is deliberately required to be positive and deliberately has no `0` escape: a
        caller that genuinely expects an empty tape is asserting something else and should say so
        directly. This is the anti-vacuity guard living in the property rather than in one
        fixture, which is what the give-up run showed was necessary."""
        if at_least < 1:
            raise ValueError(f"at_least must be positive; {at_least} would assert nothing")
        if self.violations:
            raise AssertionError(
                f"{len(self.violations)} violation(s) over {self.witnessed} witnessed site(s):\n"
                + "\n".join(f"  {violation}" for violation in self.violations)
            )
        if self.witnessed < at_least:
            raise VacuousTape(
                f"the property witnessed {self.witnessed} site(s), fewer than the {at_least} "
                f"required — it returned no violations because it never got to look, not because "
                f"the tape was clean"
            )


class TurnKind(StrEnum):
    """What an `AskLLM` on a ReAct tape IS, told apart by its address.

    Three arms rather than a predicate, because the middle one has behaviour of its own: a
    compaction is neither scored nor transparent, it is a BOUNDARY. Spelling this as an enum makes
    the walk's match total, so a fourth kind of `AskLLM` is a type error rather than a silent fall
    into `OTHER`."""

    REACT = "react"
    """A turn the model drove: `react:turn`, `react:final`. Scored."""

    COMPACTION = "compaction"
    """The summariser — `react:compact`. Rewrites the transcript, so it ends the obligation the
    prior turn created and starts one of its own."""

    OTHER = "other"
    """An `AskLLM` that is not a ReAct turn at all — a `write:`/`edit:` emitter is a separate
    metered call with its own prompt that legitimately never saw the prior tool call. Transparent:
    the obligation survives past it to the next real turn."""


def turn_kind(key: str) -> TurnKind:
    """Classify a placed key by READING THE GRAMMAR, not by testing its text.

    The predicate this replaces was `rsplit(";", 1)[-1]`, `startswith("react:")` and
    `removeprefix("react:").startswith("compact")` — a parse spelled as three string operations,
    in a consumer, outside the key machinery. That is the shape the wiki §8 `startswith` audit
    names: an unanchored prefix test aliases across a delimiter it does not know about, and
    `just key-check` scans COMPOSITION sites so nothing gated it.

    The grammar answers it as a field read. `react:compact` is a `react` term whose FIRST
    COORDINATE is `compact` — a fact about arity and position, which is exactly what a substring
    test cannot see and why `react:` needed a second string operation to disambiguate.

    An unparseable key is `OTHER`: a foreign or malformed address is certainly not one of our
    ReAct turns, and refusing to guess is cheaper than a heuristic here."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return TurnKind.OTHER
    if not terms or terms[-1].tag != "react":
        return TurnKind.OTHER
    coordinates = terms[-1].coordinates
    if coordinates and coordinates[0].atoms[0].text == "compact":
        return TurnKind.COMPACTION
    return TurnKind.REACT


def _turn_violations(
    where: str,
    contents: list[str],
    prior_action: str | None,
    prior_observation: str | None,
    spliced_summary: str | None,
) -> Iterator[Violation]:
    """What ONE scored turn's transcript fails to show, given what the tape says preceded it.

    Split out of the walk deliberately: the walk is a state machine over the tape and this is a
    pure question about a single turn, so keeping them apart means the compaction arm can grow
    without the loop's branching growing with it."""
    if spliced_summary and not any(spliced_summary in content for content in contents):
        yield Violation(where, "cannot see the summary that replaced the compacted prefix")

    # THE RENDERER'S OWN LINE, verbatim, in ONE message — not the name and the values looked for
    # separately. Checking them separately is what a defect can satisfy: this fixture's args are
    # `q="acme"` and `id=7`, and "acme" is already in the question while "7" is already in the
    # observation, so a walk that searched everywhere passed happily with the arguments dropped.
    #
    # Splitting the check was ALSO wrong in the other direction, which is the part that argues for
    # asking rather than rebuilding: the renderer writes `json.dumps`, so a `café` argument reaches
    # the model as `caf\u00e9` and a checker looking for `café` reported a violation against a
    # correct run. One string, minted once, compared once — and a rendering change moves both sides
    # together by construction.
    if prior_action is not None and not any(prior_action in content for content in contents):
        yield Violation(where, f"cannot see the action it chose — {prior_action!r}")

    if prior_observation is not None and not any(prior_observation in c for c in contents):
        yield Violation(where, "cannot see the observation it got back")


def transcript_violations(trace: Iterable[Any]) -> TapeVerdict:
    """Walk a recorded agent tape and check what the model was SHOWN at each turn.

    The generic invariant, independent of any fixture: **turn `i` must be able to see what turn
    `i-1` did and what came back.** A loop that satisfies it can attribute an observation to an
    action; one that does not is asking the model to infer causation from adjacency, and the
    observed consequence is that it repeats the call.

    Expressed over the TAPE rather than inside the loop, so it costs nothing to run and applies to
    any recorded trajectory — a `RecordingHandler` trace, or one replayed from a durable engine.
    `TraceEntry.op` carries the `AskLLM`, and `AskLLM.messages` is precisely the transcript the
    caller will send, so this reads the real thing rather than a reconstruction of it.
    """
    problems: list[Violation] = []
    witnessed = 0
    prior_action: str | None = None
    prior_observation: str | None = None
    # Set at a compaction and consumed by the next scored turn. `None` means "no compaction is
    # outstanding", which is the state every non-compacting run stays in.
    pending_summary: Any | None = None

    for entry in trace:
        # A recorded op arrives WRAPPED — the walk mints `Step(op=…)` and the domain op is its
        # `.op`. Unwrapping by attribute rather than by isinstance keeps this working on a bare
        # domain op too, which is what a differently-driven tape carries.
        op = getattr(entry.op, "op", entry.op)
        if isinstance(op, CallTool):
            # ONLY the observation. The action is taken from the turn that CHOSE it, above — a
            # `CallTool`'s args are the loop's, not the model's, and reading them here is what
            # produced four false violations against a succeeding tape.
            prior_observation = getattr(entry.result, "content", None)
            continue
        if not isinstance(op, AskLLM):
            continue

        where = entry.key.stored()
        match turn_kind(where):
            case TurnKind.OTHER:
                # Transparent — the obligation survives to the next real turn. Measured on a live
                # sidecar: without this scoping the walk reported 5 of 24 runs clean; with it,
                # 24 of 24.
                continue
            case TurnKind.COMPACTION:
                # `KEEP_TAIL` is about to rewrite the transcript, so the next turn genuinely cannot
                # see the prior one verbatim and scoring it would be a FALSE violation. But an
                # exemption alone would let a compaction that dropped the summary entirely pass
                # clean, so the obligation is REPLACED rather than waived: the next scored turn
                # must be able to see the summary that took the prefix's place.
                prior_action, prior_observation = None, None
                pending_summary = entry.result
                continue
            case TurnKind.REACT:
                witnessed += 1
                # ASK the renderer, never re-implement it. A walk that spelled this format would
                # drift from `react.summary_message` and then report confident false violations —
                # the failure the span-sidecar experiment measured, where an unwrap heuristic had
                # to mirror a tool-specific render table and could not.
                spliced = (
                    None
                    if pending_summary is None
                    else str(summary_message(pending_summary).get("content", ""))
                )
                problems.extend(
                    _turn_violations(
                        where,
                        [str(message.get("content", "")) for message in op.messages],
                        prior_action,
                        prior_observation,
                        spliced,
                    )
                )
                # THE OBLIGATION FOR THE NEXT TURN COMES FROM THE MODEL'S OWN REQUEST, not from
                # the `CallTool` the loop went on to execute. Those are not the same arguments and
                # assuming they were produced four confident false violations against a real,
                # SUCCEEDING tape: a coding agent spliced its workflow-local file `tree` into every
                # op's args (`args={**request.args, "tree": tree}`), so the executed op
                # carried an entire source tree the model never chose and could not be shown. The
                # canned fixture could not see this, because there the two are equal.
                #
                # Reading the request is also what makes the check agree with the renderer:
                # `react._assistant` writes `json.dumps(turn.tool.args)`, so this compares the
                # transcript against exactly what put text into it.
                chosen = getattr(entry.result, "tool", None)
                prior_action = None if chosen is None else action_line(chosen)
                prior_observation, pending_summary = None, None
            case unreachable:
                assert_never(unreachable)

    return TapeVerdict(tuple(problems), witnessed)


def action_count(trace: Iterable[Any]) -> int:
    """How many tool calls a tape carries — the other half of "is this tape worth walking?".

    `TapeVerdict.witnessed` counts scored TURNS, which is not sufficient on its own: the property
    is that a turn can see the ACTION before it, so a trajectory with turns and no actions has
    nothing for the action half to be shown. A give-up run is exactly that shape, and it is why
    the banker refuses to bank one."""
    return sum(1 for entry in trace if isinstance(getattr(entry.op, "op", entry.op), CallTool))


def wire_divergences(
    trace: Iterable[Any], transcripts: Mapping[str, Sequence[Mapping[str, str]]]
) -> TapeVerdict:
    """Did REPLAY reconstruct the prompt that was actually sent? Ask the wire, not the tape.

    **This closes the one assumption the rest of this module rests on.** Everything else here
    reads ops that replay re-minted, on the premise that *the prompt is a function of the tape* —
    a durable record holds `(key, result)` and never the transcript, so replay rebuilding
    `messages` identically is what makes the transcript inspectable at all. That premise is
    derived from determinism and, until this walk, was never MEASURED.

    It cannot be measured against the tape, because **the tape cannot witness itself**: replay
    reads the checkpoints, so it agrees with them by construction. An independent record of the
    same run is needed, and telemetry is exactly that — written at a different seam, while the run
    was live, by a layer that observed the call rather than replaying it.

    `transcripts` comes from `telemetry.sidecar_transcripts`, which reads LLM spans only. That
    scoping is what keeps this clear of the sidecar's known defect: a TOOL span records the domain
    result and the loop feeds back something tool-specific, so nothing here goes near a tool span.

    **A divergence is a FINDING, not automatically a defect.** A layer may legitimately rewrite an
    op in flight (wire is not workflow), in which case the wire SHOULD differ from what the
    workflow composed. No such layer sits on this path today, so equality is the expected answer
    and a difference is something to explain.

    Only keys present on BOTH sides are compared. A re-minted turn with no span is a left-outer
    miss (a keyless emitter, a span dropped for having no address) and is counted in neither
    `violations` nor `witnessed`, so a sidecar from a different run cannot manufacture a pass —
    it manufactures a vacuous verdict, which `require_clean` refuses.
    """
    problems: list[Violation] = []
    witnessed = 0
    for entry in trace:
        op = getattr(entry.op, "op", entry.op)
        if not isinstance(op, AskLLM):
            continue
        where = entry.key.stored()
        # SCOPE TO THE ReAct FAMILY, for the same reason the sidecar cannot carry the transcript
        # property: an emitter (`write:`/`edit:`) is a CHANNEL call whose prompt is a rendered
        # structure — measured here, a `dict` rather than a message list — so comparing it against
        # the flattened wire dialect would be comparing two renderings and calling a difference a
        # defect. Only what is a transcript on both sides is compared.
        if turn_kind(where) is TurnKind.OTHER:
            continue
        sent = transcripts.get(where)
        if sent is None:
            continue
        # A SHAPE guard as well as an address one, written as a pattern rather than an isinstance
        # chain: `AskLLM.messages` is `Any`, so there is no closed union to be total over, and a
        # structural pattern says what is actually required — a non-empty sequence of role-bearing
        # mappings. Anything else (a `Template`, the emitter's dict) is not a transcript on this
        # side and is not comparable; `witnessed` then falls, so the caller's floor turns a shape
        # change into a vacuity failure rather than a silent pass.
        match op.messages:
            case [{"role": _}, *_]:
                rebuilt = [
                    (str(m.get("role", "")), str(m.get("content", ""))) for m in op.messages
                ]
            case _:
                continue
        witnessed += 1
        recorded = [(str(m.get("role", "")), str(m.get("content", ""))) for m in sent]
        if rebuilt == recorded:
            continue
        if len(rebuilt) != len(recorded):
            problems.append(
                Violation(
                    where,
                    f"replay rebuilt {len(rebuilt)} message(s), the wire recorded {len(recorded)}",
                )
            )
            continue
        differing = [i for i, (a, b) in enumerate(zip(rebuilt, recorded, strict=True)) if a != b]
        problems.append(
            Violation(where, f"replay and the wire disagree at message(s) {differing}")
        )
    return TapeVerdict(tuple(problems), witnessed)


def unmeasured_steps(tape_keys: Iterable[str], measured_keys: Iterable[str]) -> TapeVerdict:
    """Tape nodes that a LIVE run left unmeasured, minus the ones nothing could measure.

    The right-orphan direction — a span addressing no tape node — is already asserted empty: it
    would mean telemetry observed an op the durable record does not know about. This is the other
    direction: a tape node with no span.

    It is normal for most kinds and a DEFECT for one. `traced` is a DOMAIN layer, so it sees
    `AskLLM` and `CallTool` and nothing else; a `ledger`, `artifact`, `sleep` or `await` node is
    structurally unmeasurable and its absence says nothing. A **`step`** node is a domain op that
    ran, so a live run that produced no span for one either failed to meter a model call — a cost
    leak, invisible to every cost report — or minted two different addresses for one op, which is
    the join breaking.

    Kind comes from `graphview.kind_of`, which strips scope frames and gather coordinates first,
    so a step inside `d:1;state:draft;` or `gather:0,0;` classifies as a step rather than as
    its frame. That normalization already exists precisely so a second spelling cannot drift from
    it.

    **Only meaningful on a LIVE run's pair.** Replay calls no model and emits no telemetry, so
    running this over a replayed trace would report every step as unmeasured. `witnessed` is the
    number of step nodes examined, so pointing it at the wrong pair yields a vacuous verdict
    rather than a false pass.
    """
    measured = set(measured_keys)
    problems: list[Violation] = []
    witnessed = 0
    for key in tape_keys:
        if kind_of(key) != "step":
            continue  # a ledger/artifact/sleep/await node is not on the domain seam at all
        witnessed += 1
        if key not in measured:
            problems.append(
                Violation(key, "a domain op with no span — unmetered, or the join is broken")
            )
    return TapeVerdict(tuple(problems), witnessed)


def collecting_layer(seen: list[TraceEntry]) -> Callable[[Any], Generator[Any, Any, Any]]:
    """An OP-seam layer that records every op the walk drives, with its placed key and its result.

    **It must be an OP layer, and the reason is the whole point of pointing this at a banked
    tape.** The obvious reading — probe the DOMAIN — collects nothing on replay: the domain call
    sits inside the checkpoint thunk (`handlers/durable.py`, `ctx.step(op_key(op), lambda: …
    self.domain.run(inner))`) and `SqliteTaskContext.step` returns the committed row *without
    running the thunk* on a hit. The op seam is the one that "re-fires per op on replay"
    (`handlers/durable.py`'s own comment), and the drive loop enters `placement_scope` AROUND the
    layer stack, so a probe here can read `current_placement()` and emit exactly a `TraceEntry`.

    That asymmetry is also why this is useful at all: replay re-runs the deterministic loop, so the
    loop RECONSTRUCTS `messages` and yields a fresh `AskLLM` carrying them. A durable tape cannot
    store the prompt — `Checkpoint` is `(key, state)`, the op's key and its RESULT — but the prompt
    is a function of the tape, and this is how you get it back: free, no model, and exact rather
    than approximated.

    Install as `DurableHandler(ctx, interp, op_layers=[collecting_layer(seen)])`. Note the mirror
    of the trap in `tests/test_placement_visible_to_domain_layer.py`, which warns against reaching
    for `layers=` when probing the domain seam; here the op seam is the correct one and a domain
    layer is what would measure nothing.
    """

    @op_layer
    def run(op: Any) -> Generator[Any, Any, Any]:
        result = yield op
        placement = current_placement()
        if placement is not None:
            seen.append(TraceEntry(key=placement, op=op, result=result))
        return result

    return run
