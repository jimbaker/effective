"""The trampoline: drive the states, and commit on every outcome.

**The budget is counted by the interpreter.** Each visit is a `descend` level, and at the final
level the trampoline mints `Exhausted` instead of running the state's spec. An exhaustion a judge
supplied would move the livelock and keep it: a judge that never says so loops forever. A
`grantor` adds visits before the final one.

**The postamble runs once, after the walk, with no branch above it, for every outcome.** Approved,
exhausted or rejected, the run reaches the canonical record, and at small-model scale exhaustion is
the common stop.

**A judge, worker or grantor that raises skips the postamble, and the run stays uncommitted.** A
refusal climbs to the caller, which may catch it: a parent running this machine inside a state
closes its own run either way. What stays on the record is the run's middle, since rows a
canonical state appended before the refusal are not withdrawn, with no commit row, no outcome row
and no artifact after them. A caller that wants a refused run committed says so where it catches
the refusal.

**Two scope tags.** `state:{name}` names which code runs. `d:{n}`, the level, counts
re-executions of one position: a re-entered `state:` frame gets no occurrence suffix, so
`TEST -> DRAFT -> TEST` under `state:` alone would mint two identical keys on the tape. The durable
engines number repeated steps `name#N`; `await_event` has no such rule.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from itertools import pairwise
from string.templatelib import Template
from typing import Any, assert_never

from pydantic_core import to_jsonable_python

from effective.api import Effect, append_ledger, call_tool, scoped, store_artifact
from effective.combinators import Answered, Deeper, Grantor, Level, descend
from effective.domain import ToolRefused
from effective.govern import ChildRefused, all_refusals
from effective.keys import Index, Key, Name, Run, Segment, compose_key
from effective.keys.grammar import KeySyntaxError, Term, parse
from effective.machine.evidence import CommandRun, Commitment, Measured, Predicate
from effective.machine.outcomes import Advance, Exhausted, Finish, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Evidence, Fused, Report, StateSpec
from effective.ops import LedgerRow, current_generation, refuse_an_unportable_value
from effective.tape import Violation

SUITE_TOOL = "run_suite"
"""The DEFAULT success-predicate tool name — the coding embodiment's, kept as the default because
it is the one every existing walk names. One constant, two consumers — `run_machine` defaults to
it and `coding.runners.TOOLS` serves it — so the op the machine names and the body a deployment
provides cannot drift apart by a typo.

**It is a default, and a machine over anything else names its own.** A postamble that yielded
this name unconditionally would ask a non-coding machine for a pytest runner at its commitment
point. `tests/test_machine_second_embodiment.py` walks a `just docs-check` embodiment on
`link_check` and `repair_link`, and every tool it asks for, the commitment's included, is one
that embodiment declares."""

DEFAULT_PREDICATE: Predicate[CommandRun] = Predicate(SUITE_TOOL, CommandRun)
"""What `run_machine` uses when a caller names no predicate — the coding machine's."""


@dataclass(frozen=True, slots=True)
class Under:
    """Where a nested run sits in its parent: the state that ran it, and that state's visit.

    These are COORDINATES, framing the record's address as an `under:` term, rather than packed
    into the `run_id` as one atom. Packing them hides them inside one `Run`: `--coordinate-roles`
    sees one role, `registry.explain` decodes one field, and the `-` joining them is a delimiter
    this grammar does not know. `under:` is provisional as a tag and as a name; the shape is what
    is settled.

    **A nesting deeper than one passes a SEQUENCE of these, outermost first**, and the address
    carries a frame per pair, in that order: `under:work,0;under:work,0;machine:r1;commit`. A
    machine that runs itself reads the same coordinates at every level, and a second task's write
    of an id already on the record is dropped, so every ancestor is part of the address.

    Two child machines run from one visit compose one address. In one task the store refuses the
    second rather than dropping its rows; addressing sibling nestings is open work."""

    state: str
    visit: int


type Placement = Under | Sequence[Under]
"""Where a nested run sits: one pair, or one per enclosing machine, outermost first."""


def running_under(under: Placement | None, ctx: Ctx[Any, Any]) -> tuple[Under, ...]:
    """Where a machine that THIS state runs will sit: this run's own placement, then here.

    Takes the `Ctx` rather than a state and a visit so the coordinates cannot be invented. A worker
    holds the state that is running and the visit that is running it; a caller outside the walk
    holds neither, and the pair it supplies instead names a state that never ran the machine."""
    return (*placement(under), Under(ctx.state.value, ctx.visit))


def placed(address: Key, steps: Sequence[Under]) -> Key:
    """`address`, framed by where its run sits, outermost first."""
    for step in reversed(steps):
        address = compose_key(
            t"under:{Name(step.state)},{Index(step.visit)};{address:domain=address}"
        )
    return address


def placement(under: Placement | None) -> tuple[Under, ...]:
    """One pair, a sequence of them, or nothing, read as the sequence all three mean.

    Taking all three is what keeps the sequence additive: a caller that nests once writes the pair
    it always wrote, and the address it composes is unchanged."""
    match under:
        case None:
            return ()
        case Under():
            return (under,)
        case Sequence():
            return tuple(under)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class Turn[S: StrEnum, V: StrEnum, R: Measured = CommandRun]:
    """One visit, recorded: where it ran, what it produced, and where that sent it."""

    visit: int
    state: S
    verdict: V | Exhausted[S]
    outcome: Outcome[S]
    report: Report[V, R] | None = None
    """The evidence the verdict stands on, `None` only at the final level, where `Exhausted` is
    minted without running the spec.

    Kept because a finished session is what a CALLER reads, and the verdict alone cannot say what
    a run did. A machine composed inside another found this: the child returned a verdict and an
    artifact id and no account of itself, so the parent had nothing to compose a next goal from."""


@dataclass(frozen=True, slots=True)
class Session[S: StrEnum, V: StrEnum, R: Measured = CommandRun]:
    """The finished run. `stopped` is `Finish` or `Park`, because `_step` answers only on those."""

    run_id: Segment
    turns: tuple[Turn[S, V, R], ...]
    stopped: Finish | Park[S]

    commit_id: Key
    outcome_id: Key
    """The two rows this run appends at its postamble, as the addresses it composed them from.

    Appended when `commitment` is set, and only composed when it is `None`, as on a `RunRefused`.

    A run that cannot name itself is a run nothing can cite, and the two ids were local to the
    walk: a caller that wanted one composed the skeleton again, which is a second speller of an
    identity. A ledger id is frame-free, so these are valid to any reader, and a second task
    writing the same id is dropped by design (`ops.refuse_placed_writer_collision`), which is why
    a citation names a row that may not be there if the runs were not in one task."""

    commitment: Commitment[R] | None = None
    """What the postamble committed and measured. `None` only while the session is being built —
    a returned `Session` always carries one, because the postamble is unconditional."""

    @property
    def path(self) -> tuple[S, ...]:
        return tuple(turn.state for turn in self.turns)

    @property
    def concluded(self) -> Report[V, R] | None:
        """What the last visit that RAN reported, which is not always the last turn's.

        The final level mints `Exhausted` without running the spec, so an exhausted run's last
        turn carries no report, and exhaustion is the common stop at small-model scale. `None`
        only when no visit ran at all, which a budget of zero produces."""
        reported = (turn.report for turn in reversed(self.turns) if turn.report is not None)
        return next(reported, None)


def machine_frames(terms: Sequence[Term]) -> tuple[Under, ...]:
    """The machine visits a key was minted in, outermost first: each `state:{name}` term that
    follows a `d:{visit}` term, read up to the `ledger` term where an op's own identity begins."""
    frames: list[Under] = []
    for before, term in pairwise(terms):
        match before, term:
            case (_, Term(tag="ledger")) | (Term(tag="ledger"), _):
                break
            case (Term(tag="d", coordinates=[visit]), Term(tag="state", coordinates=[state])) if (
                visit.atoms[0].text.isdecimal()
            ):
                frames.append(Under(state.atoms[0].text, int(visit.atoms[0].text)))
            case _:
                pass
    return tuple(frames)


def appending_states[S: StrEnum](
    states: type[S], keys: Iterable[str], *, under: Placement | None = None
) -> dict[S, list[str]]:
    """Which state of the machine placed at `under` each ledger append happened under, READ OFF THE
    TAPE.

    This is what the `state:` scope frame buys beyond readability: a ledger op minted inside a
    state carries `d:{n};state:{name};ledger;…`, so *which bookkeeper a state reached* is a
    recorded fact rather than something a reader infers from the call graph.

    **A key carries one machine frame per machine it ran in, outermost first** (`machine_frames`),
    so the machine `under` places owns the appends whose frames extend its placement by one, and
    that last frame names the state. The whole placement decides, states and visits alike, since
    two machines' state types may share a value, and no `under` places the machine at the top.

    A run's own postamble sits outside every state scope of that run, so a top-level run's two
    rows appear under no state: they belong to the run rather than to a visit. A NESTED run's
    postamble still sits inside the enclosing state's scope, so it appears under that state. The
    state ran the machine that wrote those rows, so the canonical record was reached from inside
    it, which is what `StateSpec.canonical` is for declaring."""
    members = {member.value for member in states}
    placed = placement(under)
    found: dict[S, list[str]] = {}
    for key in keys:
        try:
            terms = parse(key).terms
        except KeySyntaxError:
            continue
        if not any(term.tag == "ledger" for term in terms):
            continue
        match machine_frames(terms):
            case (*outer, Under(state=name)) if tuple(outer) == placed and name in members:
                found.setdefault(states(name), []).append(key)
            case _:
                pass
    return found


def canonical_violations[S: StrEnum](
    states: type[S],
    specs: Mapping[S, StateSpec[S, Any]],
    keys: Iterable[str],
    *,
    under: Placement | None = None,
) -> dict[S, list[Violation]]:
    """States that reached the canonical record without DECLARING that they would.

    This is the reader `StateSpec.canonical` has; without it the field is written and never
    consulted. The declaration says which states touch the append-only ledger; this reads the tape
    and says which ones did. A mismatch is the two-bookkeepers rule breaking silently: a back
    edge into an appending phase nobody marked as one.

    Checked in tests rather than enforced at run time on purpose: the tape is only complete once
    the run is over, so a runtime check could only fire after the damage.

    Reported as `effective.tape.Violation` because this is one member of that family — a declared
    property read off a completed tape — and the sites are ADDRESSES a reader can go look at
    rather than a prose list. The mapping is still keyed by state: this walk attributes each
    violation to the state that caused it, which is the coordinate its callers group by."""
    return {
        state: [
            Violation(site, f"{state} appended without declaring `canonical`") for site in sites
        ]
        for state, sites in appending_states(states, keys, under=under).items()
        if not (spec := specs.get(state)) or not spec.canonical
    }


@dataclass(frozen=True, slots=True)
class Stop:
    """How a run ended, as the canonical record spells it: a category, and why.

    Two fields rather than a wider `kind` vocabulary, and the choice was MEASURED rather than
    argued. The alternative was `machine-parked-exhausted` / `machine-parked-rejected`, whose whole
    claim was that a new `ParkReason` could not be added without deciding what it is called —
    `assert_never` forcing the arm. It cannot: `ty` does not narrow a nested enum pattern inside a
    class pattern, and infers `Park[S] & ~Finish` rather than `Never` (measured 2026-08-19). So
    that shape buys no compile-time totality over the reason, and costs a consumer filtering on
    `machine-parked` its answer.

    Here the totality that exists is kept — `assert_never` still closes the OUTCOME, which is the
    union `ty` does narrow — and the reason rides as data, so a third `ParkReason` reaches the
    record with no arm to remember."""

    kind: str
    reason: str | None = None


def stop_record[S: StrEnum](stopped: Finish | Park[S]) -> Stop:
    """What the outcome row says about how the run ended.

    **The reason has to survive to here**, which is `ParkReason`'s own stated requirement: a bare
    `kind` would record both reasons as `machine-parked`. `route_plan` declines to route a
    rejected plan to `Finish` because *"'we shipped it' and 'a human said no'"* must differ on the
    tape, and the same distinction has to hold on the canonical record."""
    match stopped:
        case Finish():
            return Stop("machine-finished")
        case Park(why=why):
            return Stop("machine-parked", why.value)
        case unreachable:
            assert_never(unreachable)


def changed_paths(seed: Mapping[str, str], tree: Mapping[str, str]) -> tuple[str, ...]:
    """The paths added, edited or removed between `seed` and `tree`, sorted. An empty file and an
    absent one are different trees."""
    return tuple(sorted(p for p in seed.keys() | tree.keys() if seed.get(p) != tree.get(p)))


def commitment_postamble[S: StrEnum, V: StrEnum, R: Measured](
    session: Session[S, V, R],
    tree: Mapping[str, str],
    *,
    seed: Mapping[str, str],
    commit_id: Key,
    outcome_id: Key,
    predicate: Predicate[R],
) -> Effect[Commitment[R]]:
    """Commit what exists, run the predicate, record what it said — in that order, on EVERY
    exit path. This is the coding agent's unconditional tail, and the
    order is what carries the property: committing FIRST means a run whose predicate fails still
    has its work on the record, so a failure is recorded rather than lost.

    **`predicate` is the tool a deployment must serve, and it is a PARAMETER rather than a
    constant** — the last coding-ism in the generic walk. What makes something a success predicate
    is that its answer is a measurement rather than a judgment, which is a property of the
    contract and not of the word "suite": a docs machine reads `just docs-check`, and its exit code
    and named failures are the same record. `CommandRun` stays as the record's type for exactly
    that
    reason; what was domain-specific was the NAME.

    **`store_artifact` is the commit**, not a git call. The op alphabet already content-addresses a
    value, so the substrate needs no VCS to have a commitment point; turning that content id into a
    real git history is a projection a deployment may build (the example does), and keeping it out
    of here is what keeps the machine engine- and vendor-neutral.

    **The predicate runs on no branch at all.** Not `if finished`, not `if the tree changed` — at
    small-model scale the common stop is exhaustion, so a predicate gated on success would usually
    not run, and the ledger would record only the runs that went well. Two rows, always: what was
    committed, and what the predicate said about it.

    **The two ids are handed in, not composed here.** They are minted at `run_machine` entry so a
    `run_id` that cannot form an atom is refused before any op — see that docstring for what it
    cost when they were composed at this line. Taking them as parameters is what makes "always"
    true of the address as well as of the call."""
    artifact_id = yield from store_artifact(dict(tree), "application/json")
    # A union is no `type[T]`, so the schema is `Any`, as the ReAct loop's own tool call does.
    record: Any = ToolRefused | predicate.record
    run: R | ToolRefused = yield from call_tool(predicate.tool, {"tree": dict(tree)}, record)
    commitment: Commitment[R] = Commitment(
        artifact_id=artifact_id,
        measured=run,
        files=tuple(sorted(tree)),
        changed=changed_paths(seed, tree),
    )
    yield from append_ledger(
        LedgerRow(
            event_id=commit_id,
            kind="machine-committed",
            artifact_id=artifact_id,
            files=list(commitment.files),
            changed=list(commitment.changed),
        )
    )
    stop = stop_record(session.stopped)
    concluded = session.concluded
    yield from append_ledger(
        LedgerRow(
            event_id=outcome_id,
            kind=stop.kind,
            reason=stop.reason,
            visits=len(session.turns),
            path=[state.value for state in session.path],
            passed=commitment.passed,
            # WHAT THE RUN CONCLUDED, so this address decodes to the words a caller quotes. A
            # parent composing a goal from a child reads the child's summary; without it here the
            # canonical record holds the measurement and not the sentence, and a citation would
            # name a row that does not contain what it is cited for. `None` where no visit ran.
            summary=None if concluded is None else concluded.summary,
            # THE RECORD, WHOLE — not two of its fields by name. `exit_code` and `failures` are a
            # COMMAND's vocabulary, and writing them here made every predicate a command: `ty`
            # refuses a bench machine's `Pareto(score, cost_usd)` against a bound wide enough to
            # supply them. Carrying the record keeps `Measured` at one member, which is what lets
            # a non-command predicate exist at all, and makes the row self-describing besides.
            measured=to_jsonable_python(run),
        )
    )
    return commitment


class RunRefused(ChildRefused):
    """A machine run stopped at a refusal, carrying what it had built.

    Raised where `run_machine` was called, from the refusal that stopped it. The run is
    uncommitted: `session.commitment` is `None`, its two ids are composed and not appended, and
    `session.stopped` parks at the refused state. `tree`, `seed` and `predicate` are what the
    skipped postamble would have used, so a caller can run it with `committing`.

    A refusal, so a handler delivers it across a scope or a gather, and a spawned task raising it
    answers its parent `Refusal`. It is not a `Refused`, so a loop that turns a `Refused` into an
    observation lets this one climb: a nested run's ending is not an op a loop can route around."""

    def __init__(
        self,
        message: str,
        session: Session[Any, Any, Any],
        tree: Mapping[str, str],
        predicate: Predicate[Any],
        *,
        seed: Mapping[str, str],
    ) -> None:
        super().__init__(message)
        self.session = session
        self.tree = tree
        self.seed = seed
        self.predicate = predicate


def committing[T](body: Callable[[], Effect[T]]) -> Effect[T]:
    """Run `body`, and commit a machine refused inside it before the refusal climbs on.

    The postamble the refused run skipped, run by the caller that chose to have it: the artifact,
    the predicate and the two rows, with the outcome row reading `machine-parked` and `refused`.

    **A run commits once however many frames catch it.** The refusal that climbs on is the same
    object, so the commitment is recorded on the session it carries, and a frame further out finds
    it set and passes the refusal on. Two commits would append one run's ids from two placements in
    one task, which the store refuses."""
    try:
        return (yield from body())
    except RunRefused as refused:
        if refused.session.commitment is None:
            commitment = yield from commitment_postamble(
                refused.session,
                refused.tree,
                seed=refused.seed,
                commit_id=refused.session.commit_id,
                outcome_id=refused.session.outcome_id,
                predicate=refused.predicate,
            )
            refused.session = replace(refused.session, commitment=commitment)
        raise


@dataclass(frozen=True, slots=True)
class _Walk[S: StrEnum, V: StrEnum, R: Measured]:
    """What one visit hands the next: the state to run, the turns so far, the last report, and
    the workspace as the workflow knows it."""

    state: S
    turns: tuple[Turn[S, V, R], ...]
    incoming: Report[Any, R] | None
    carried: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Walked[S: StrEnum, V: StrEnum, R: Measured = CommandRun]:
    """A walk that reached a terminal outcome."""

    turns: tuple[Turn[S, V, R], ...]
    stopped: Finish | Park[S]
    carried: Mapping[str, str]
    refusal: Exception | None = None
    """What stopped the walk, when a visit was refused rather than judged."""


def _step[S: StrEnum, V: StrEnum, R: Measured](
    run_id: Run,
    goal: str | Template,
    specs: Mapping[S, StateSpec[S, Any, R]],
    transition: Callable[[S, V | Exhausted[S]], Outcome[S]],
    walk: _Walk[S, V, R],
    level: Level,
) -> Effect[Answered[_Walked[S, V, R]] | Deeper[_Walk[S, V, R]]]:
    """One visit as a `descend` level: run the state under `state:{name}`, then follow its
    transition. `Advance` goes a level deeper; `Finish` and `Park` answer."""
    state = walk.state
    ctx = Ctx[S, R](
        run_id=run_id,
        goal=goal,
        state=state,
        visit=level.depth,
        tree=walk.carried,
        incoming=walk.incoming,
    )
    # A refusal ends the walk where it stands, as `Finish` and `Park` do, so the caller learns
    # what the run had built. The visit has no verdict and no turn; the turns before it stand.
    try:
        visited: _Visited[S, V, R] = yield from scoped(
            compose_key(t"state:{Name(state.value)}"),
            lambda: _visit(specs[state], ctx, final=level.final),
        )
    except Exception as raised:
        if not all_refusals(raised):
            raise
        return Answered(_refused(walk, raised, walk.carried))
    match visited:
        case _JudgedRefused(raised, left):
            built = walk.carried if left.tree is None else dict(left.tree)
            return Answered(_refused(walk, raised, built))
        case (verdict, produced):
            pass
        case unreachable:
            assert_never(unreachable)
    incoming, carried = walk.incoming, walk.carried
    if produced is not None:
        incoming = produced
        if produced.tree is not None:
            carried = _storable(produced.tree, "a tree a worker produced")
    outcome: Outcome[S] = transition(state, verdict)
    turns = (
        *walk.turns,
        Turn[S, V, R](
            visit=level.depth, state=state, verdict=verdict, outcome=outcome, report=produced
        ),
    )
    match outcome:
        case Advance(to):
            return Deeper(
                _Walk[S, V, R](state=to, turns=turns, incoming=incoming, carried=carried)
            )
        case Finish() | Park():
            return Answered(_Walked[S, V, R](turns=turns, stopped=outcome, carried=carried))
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class _JudgedRefused[R: Measured = CommandRun]:
    """A fused state's judge was refused after its worker returned what `left` holds."""

    raised: Exception
    left: Evidence[R]


type _Visited[S: StrEnum, V: StrEnum, R: Measured] = (
    tuple[V | Exhausted[S], Report[V, R] | None] | _JudgedRefused[R]
)
"""A visit's result: its verdict and report, or its judge's refusal with what the worker left."""


def _refused[S: StrEnum, V: StrEnum, R: Measured](
    walk: _Walk[S, V, R], raised: Exception, carried: Mapping[str, str]
) -> _Walked[S, V, R]:
    return _Walked[S, V, R](
        turns=walk.turns,
        stopped=Park(walk.state, ParkReason.REFUSED),
        carried=carried,
        refusal=raised,
    )


def _visit[S: StrEnum, V: StrEnum, R: Measured](
    spec: StateSpec[S, V, R], ctx: Ctx[S, R], *, final: bool
) -> Effect[_Visited[S, V, R]]:
    """Run one state and return its verdict and report, or at the final level mint `Exhausted`
    without running it.

    One call per state: a worker and judge pair is a construction an embodiment may choose
    (`specs.fuse`), and one that wants its judge as a state of its own writes two states. The
    report's tree is what the state derived from its recorded ops, which a replay reconstructs.
    A `Fused` run is run phase by phase, so a refusal of its own judge returns with what the
    worker left, where a refusal anywhere else raises. Any other callable, a subclass or a wrapper
    of one included, is run as it is written."""
    if final:
        return Exhausted(ctx.state, level=ctx.visit), None
    match spec.run:
        case Fused(worker, judge) if type(spec.run) is Fused:
            evidence = yield from worker(ctx)
            try:
                verdict = yield from judge(ctx, evidence)
            except Exception as raised:
                if not all_refusals(raised):
                    raise
                return _JudgedRefused(raised, evidence)
            report = Report.of(verdict, evidence)
        case run:
            report = yield from run(ctx)
    return report.verdict, report


def _storable(tree: Mapping[str, str], what: str) -> dict[str, str]:
    """`tree`, when every store holds it alike (`ops.refuse_an_unportable_value`)."""
    refuse_an_unportable_value(dict(tree), what)
    return dict(tree)


def run_machine[S: StrEnum, V: StrEnum, R: Measured = CommandRun](
    run_id: Run,
    goal: str | Template,
    specs: Mapping[S, StateSpec[S, Any, R]],
    transition: Callable[[S, V | Exhausted[S]], Outcome[S]],
    *,
    start: S,
    budget: int = 12,
    tree: Mapping[str, str] | None = None,
    predicate: Predicate[R] | None = None,
    grantor: Grantor | None = None,
    under: Placement | None = None,
) -> Effect[Session[S, V, R]]:
    """Drive the machine from `start` until it finishes or parks, then commit, on every outcome.

    Each visit is a `descend` level under `d:{n}` and runs its state under `state:{name}`. The
    final level mints `Exhausted` without running the spec, and `grantor` may add visits before it.

    A visit that is refused raises `RunRefused`, carrying the run as far as it got, and the
    postamble does not run. Anything else a judge or worker raises propagates as it was raised, and
    so does a refusal out of `grantor`, which arrives between visits and has no state to park at.

    `specs` must be total over the state type, which `Mapping[S, StateSpec]` does not say, so a
    partial map is refused here, naming every missing state, before the first op. `build_specs`
    builds a total map. `tree` is the workspace to commit, empty by default: a run that edited
    nothing commits an empty tree and still has a predicate verdict. A seed or a produced tree
    that no store holds alike is refused before it is carried.

    The record's address is composed before the first op, so a `run_id` that cannot form an atom is
    refused before anything is stored. It carries the chain generation, omitted at 0, because the
    generations of a `respawn` chain share a `run_id`. The commit row is the sub-term `;commit`: as
    a coordinate, `commit` and the optional generation would both reach arity 2, and a decode could
    not tell which produced a key.

    Two runs sharing a `run_id` in one generation compose one address. In different tasks the
    second run's rows are dropped, by the allowance in `ops.refuse_placed_writer_collision`; in one
    task, as when a worker runs an inner `run_machine`, the collision is refused.

    **A nested run says where it sits with `under`**, and the coordinates are composed into the
    address as a sub-term. Packing them into the `run_id` instead hides three coordinates inside
    one `Run` atom and leaves the only legal spelling an f-string in an identity position, which
    this repo's gates refuse. A grantor the nested run parks grants under that run's own `run_id`,
    which a nested run shares with the run above it, so two nested runs of one state that both
    park for a grant want distinct ids there."""
    # `None` rather than a concrete default: a generic function's parameter cannot TAKE a
    # concrete default (`ty` refuses `Predicate[R] = Predicate(SUITE_TOOL, CommandRun)` — the
    # gotcha this arc recorded), so the PEP 696 default on `R` supplies the type and this
    # supplies the value.
    # BEFORE ANY OP, and beside the address for the same reason: a run that cannot complete should
    # refuse while refusing is still free. `type(start)` is the embodiment's `StrEnum` class, so
    # the domain is the enum itself rather than a list someone passed alongside it.
    match run_id:
        case Run():
            pass
        case _:
            # Stated here rather than left to the composer, which sees only what the mint hands
            # it: `compose_key(t"machine:{Run(run_id)}")` wraps its own value, so it accepts
            # whatever a `Run` accepts. The seam asks for more than delimiter-freedom, and `Run`
            # is the word for it: the caller has decided this names an execution.
            raise ValueError(
                f"run_id {run_id!r} is {type(run_id).__name__}, not a `Run`. A machine's run "
                f"identity addresses its ledger rows, so it is promoted at the boundary that "
                f"knows it names an execution, before any op is performed."
            )
    if missing := sorted(state.value for state in type(start) if state not in specs):
        raise ValueError(
            f"no spec for {', '.join(missing)} — a spec map must be total over "
            f"{type(start).__name__}, because the trampoline indexes it directly. "
            f"`build_specs` iterates the state type and so cannot produce a partial map."
        )
    chosen: Predicate[Any] = predicate if predicate is not None else DEFAULT_PREDICATE
    generation = current_generation()
    steps = placement(under)
    commit_id = placed(
        compose_key(t"machine:{Run(run_id)},generation={Index(generation):default=0};commit"),
        steps,
    )
    outcome_id = placed(
        compose_key(t"machine:{Run(run_id)},generation={Index(generation):default=0}"), steps
    )
    # `carried` is the workspace as the workflow knows it: seeded by the caller and advanced only
    # from what a worker returns (`Evidence.tree`), since a replay runs no tool and a resumed
    # worker holds nothing the crashed one built.
    seeded = _storable(tree or {}, "the seed")
    walked = yield from descend(
        _Walk[S, V, R](state=start, turns=(), incoming=None, carried=dict(seeded)),
        partial(_step, run_id, goal, specs, transition),
        budget=budget,
        grantor=grantor,
    )
    session: Session[S, V, R] = Session(
        run_id=run_id,
        turns=walked.turns,
        stopped=walked.stopped,
        commit_id=commit_id,
        outcome_id=outcome_id,
    )
    if walked.refusal is not None:
        raise RunRefused(
            f"machine {run_id} was refused before it committed: {walked.refusal}",
            session,
            walked.carried,
            chosen,
            seed=seeded,
        ) from walked.refusal
    # THE ONE POSTAMBLE CALL SITE for a run that ended, so that a run which exhausted or was
    # rejected still reaches the canonical record. A refused run ended nowhere and raised above.
    # `tests/test_coding_livelock.py` pins this structurally, because no fixture can.
    commitment = yield from commitment_postamble(
        session,
        walked.carried,
        seed=seeded,
        commit_id=commit_id,
        outcome_id=outcome_id,
        predicate=chosen,
    )
    return Session[S, V, R](
        run_id=run_id,
        turns=walked.turns,
        stopped=walked.stopped,
        commit_id=commit_id,
        outcome_id=outcome_id,
        commitment=commitment,
    )
