"""Workflow operations — the abstract syntax of the effect DSL.

A workflow *yields* these; a handler *interprets* them. They are inert
dataclasses and perform no I/O themselves. This is the reified term that makes
the embedding deep: the same op stream can be run, recorded, replayed, or
re-interpreted under a different handler.
"""

from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Never

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticSerializationError, to_jsonable_python

from effective.domain import DomainOp
from effective.keys import RESERVED_AUTHORITY_TAGS, Key, authored_key, frame_path
from effective.keys.frame import ARM_TAGS, leads_with_an_arm
from effective.keys.grammar import TERM_SEPARATOR, KeySyntaxError, parse


@dataclass(frozen=True)
class Minted:
    """Asks the handler for a step's idempotency key, minted from the task and the step's
    placement and handed to the tool in its args under `idempotency_key`.

    A receiver that remembers the key performs the step's effect once, however often a crash
    re-runs the call: this task's engine always, and an outside service for as long as it keeps
    the key."""


@dataclass(frozen=True)
class Step[T]:
    """Run a domain operation as a durable, checkpointed step."""

    name: str
    op: DomainOp[T]
    idempotency_key: Minted | None = None


class Addressing(Enum):
    """How an event name resolves: **relative to the frames**, or **absolutely**.

    Exactly the distinction a filesystem path makes, and it transfers without adaptation —
    which is the point of the word. A relative path is completed by the current directory; a
    relative name is completed by the handler's frame stack, and `absurd.py` already calls it
    that (*"branch-RELATIVE name — the parent re-adds its `gather:{g},{i};` prefix"*). An
    absolute path ignores the cwd and prepending to one is a bug; an absolute name ignores
    frames, and prepending to one is *precisely* the defect this exists to stop.

    `RELATIVE` is the default and covers everything an author writes. Two scopes awaiting a bare
    ``review:{id}`` are two different questions, and an emitter that knows the structure completes
    the name with `api.qualified_event_name`. A fork child's rename is the same mechanism doing
    load-bearing work: the child re-runs the BASE workflow, so without it the child would absorb
    the base's answer (reproduced — `RenamedAwaitCtx`).

    `ABSOLUTE` is for a name whose whole address is already in the string, because the emitter is
    a different task running from its own params (`fork.run_fork_as_task` echoes the
    ``done_event`` it was handed). It cannot complete a relative name — it has never seen a frame
    of ours, so nothing may be prepended once the name is handed over.

    **Not `await_event`'s "scope it on the SUBJECT" rule**, which is a different axis: that says
    what to EMBED in a name (the message id, never the run id) for fork-stability.
    ``review:{message_id}`` follows it and is still `RELATIVE`."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


def refuse_a_naive_instant(instant: datetime, what: str) -> None:
    """An instant the substrate takes as a bound is absolute. A naive datetime reads as the local
    time of whichever host converts it, so an attempt that retries in another zone would wait on
    a different instant."""
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise CompositionRefused(
            f"{what} is an absolute instant and a naive datetime is not one: it reads as the "
            "local time of whichever host converts it. Pass an aware datetime"
        )


def refuse_an_unportable_value(value: Any, what: str) -> None:
    """Every string in `value`, keys included, has UTF-8 bytes and no NUL. Postgres's `jsonb`
    holds no other string where SQLite's JSON text holds both escaped, so a value that has one
    would be stored by one engine and retried to failure by the other."""
    try:
        jsonable = to_jsonable_python(value)
    except UnicodeError as unencodable:
        raise CompositionRefused(f"{what} has a key with no UTF-8 bytes") from unencodable
    except PydanticSerializationError as unserializable:
        raise CompositionRefused(f"{what} is no JSON value: {unserializable}") from unserializable
    if (text := next(_unportable(jsonable), None)) is not None:
        raise CompositionRefused(
            f"{what} holds the string {text!a}, which has no UTF-8 bytes or holds a NUL, so no "
            "store holds it alike"
        )


def _unportable(jsonable: Any) -> Iterator[str]:
    """The strings in a JSON value that `jsonb` cannot hold."""
    match jsonable:
        case str() as text:
            try:
                text.encode()
            except UnicodeError:
                yield text
            else:
                if "\x00" in text:
                    yield text
        case dict() as mapping:
            for key, item in mapping.items():
                yield from _unportable(key)
                yield from _unportable(item)
        case list() as items:
            for item in items:
                yield from _unportable(item)
        case _:
            pass


@dataclass(frozen=True)
class AwaitEvent[T]:
    """Suspend until a named external event arrives; validate it to T.

    `addressing` declares whether enclosing frames complete `name` (see `Addressing`). It
    defaults to `RELATIVE`, so every await an author writes behaves exactly as before; the
    substrate sets `ABSOLUTE` on the awaits whose emitter is another task (`fork.join_fork`,
    `compose.spawn_subagent_task`).

    **`name` is a `Key`.** An event name is what a park BINDS to, the position `Step.name` holds
    on the checkpoint axis, and it is minted by `event_name`, which every author surface funnels
    through. Typing the field makes a hand-rolled `f"{prefix}{name}"` at a ctx boundary a `ty`
    error; the named exit is `Key.prefixed`."""

    name: Key
    schema: type[T]
    addressing: Addressing = Addressing.RELATIVE
    deadline: datetime | None = None
    """When this wait gives up, and so how many answers it has.

    ==========  ==================  ==================================================
    deadline    the author writes   the wait answers with
    ==========  ==================  ==================================================
    ``None``    ``await_event``     the payload, once an emitter delivers it
    an instant  ``await_until``     ``Arrived(payload)`` or ``Expired`` (`WaitOutcome`)
    ==========  ==================  ==================================================

    Absolute, so every attempt waits on the instant the first one chose and a restart shortens
    the wait. A ctx that cannot wait on a clock refuses by name, since dropping the deadline
    would park a workflow that asked to be released."""

    def __post_init__(self) -> None:
        if self.deadline is not None:
            refuse_a_naive_instant(self.deadline, "a wait's deadline")


@dataclass(frozen=True)
class Writer:
    """WHO wrote a ledger row — the task, and the placed op within it.

    The ledger is `UNIQUE(event_id)` and `ON CONFLICT DO NOTHING`, which is exactly right for the
    two cases it was built for (a crash-window re-append, and the same message triaged in two
    generations) and silently wrong for a third: two ops the substrate places DISTINCTLY writing
    one `event_id`, where the second row vanishes from the canonical record and the run reports
    success. Telling those apart needs the one fact the row never carried — who wrote it.

    `task` scopes the question, and the scoping is the whole predicate rather than a detail.
    ACROSS tasks a repeated `event_id` is the documented idempotency FEATURE, so it must stay
    silent; WITHIN one task two different placements are a lost row. Attempt is deliberately NOT
    part of this: a retry re-executes the same placement, which is how the crash-window re-append
    stays permitted.

    `placement` is the fully placed, occurrence-qualified name (`effective.layers.placement_scope`)
    — frames included, because a gather branch's identity is exactly its frame."""

    task: str
    placement: Key


class LedgerRow(BaseModel):
    """One append to the canonical record: **the typed shape of a ledger insert**.

    A model gives the requirement that a ledger insert refuse anything other than a `Key`
    somewhere to attach, twice over: `ty` rejects a `str` at the call site, and Pydantic rejects
    one at runtime (`Key.__get_pydantic_core_schema__`'s python arm is `is_instance`). The
    ledger is the canonical append-only record with `UNIQUE(event_id)`, so this is the identity
    that matters most and the one whose corruption is quietest: a bad `event_id` is accepted,
    committed, and permanent.

    **`extra="allow"`, and the wire format is unchanged.** Both writers store
    `to_jsonable_python(row)` as the payload — the WHOLE row, `event_id` and `kind` inline
    alongside the domain fields. So the model has to serialize flat, not nested, or every
    recorded row would change shape. It does: extras ride beside the two declared fields and
    `event_id` serializes as its `stored()` text, byte-identical to the equivalent `dict`.
    Verified against both engines' payload columns rather than assumed.

    Frozen because the ledger is append-only in the small as well as the large: a row that could
    be mutated between construction and commit is a second bookkeeper by accident."""

    model_config = ConfigDict(extra="allow", frozen=True)

    event_id: Key
    kind: str = "event"

    def get(self, name: str, default: Any = None) -> Any:
        """Read a field by name — the DOMAIN half (`decision`, `total_amount`, …) as readily as
        the two declared ones.

        Kept because the consumers that read a row generically are *policies* and *projections*
        (a domain policy gates on `kind`/`status`; the lineage readers key on `event_id`), and
        they should not have to know which fields the model declares and which arrived as extras.
        `getattr` handles both uniformly under `extra="allow"`."""
        return getattr(self, name, default)


@dataclass(frozen=True)
class AppendLedgerRow:
    """Append one row to the append-only decision ledger."""

    row: LedgerRow


@dataclass(frozen=True)
class StoreArtifact[T]:
    """Persist an immutable artifact; resolves to a content id (str)."""

    value: T
    content_type: str

    def __post_init__(self) -> None:
        refuse_an_unportable_value(self.value, "an artifact")


@dataclass(frozen=True)
class SleepUntil:
    """Release the worker and resume at a wall-clock time."""

    when: datetime

    def __post_init__(self) -> None:
        refuse_a_naive_instant(self.when, "a sleep's end")


@dataclass(frozen=True)
class Arrived[T]:
    """The event was delivered."""

    payload: T


@dataclass(frozen=True)
class Expired:
    """The deadline passed first."""


type WaitOutcome[T] = Arrived[T] | Expired
"""How a wait that named a deadline ended. A deadline-free wait has one outcome, so it returns
its payload directly.

A wait answers once: the outcome is recorded when it is first reached, so a replay serves it
rather than deciding again. Absurd signals a deadline by raising ``absurd_sdk.TimeoutError``,
which descends straight from ``Exception``, so only a caller naming that class catches it, and
the ctx adapting the SDK turns it into an ``Expired``.
"""


def settled_wait(stored: Any) -> WaitOutcome[Any]:
    """Read back what a bounded wait settled: the durable form both engines write.

    ``["arrived", payload]`` and ``["expired"]`` are that form, and this is the one reader, so a
    writer that drifts from it raises here rather than answering a workflow with a shape it has
    never seen. The store holds the FIRST answer reached, which is where "one wait, one answer"
    is spent."""
    match stored:
        case ["arrived", payload]:
            return Arrived(payload)
        case ["expired"]:
            return Expired()
        case other:
            raise ValueError(f"a settled wait holds neither an arrival nor an expiry: {other!r}")


@dataclass(frozen=True)
class Gather:
    """Run independent sub-workflows concurrently; join results in branch order.

    The applicative dual of sequential ``yield from``: declaring the branches
    independent is what licenses both concurrent execution *and* order-independent
    replay, since results bind by **branch index**, never wall-clock completion
    order. Each ``branches[i]`` is a thunk returning an ``Effect``. (``fork`` is
    reserved for counterfactuals.)
    """

    branches: tuple[Callable[[], Any], ...]


@dataclass(frozen=True)
class Race:
    """Run branches concurrently, record which `want` of them succeeded first, and stop the rest
    at their next op admission.

    `race` is `want=1`. Each ``branches[i]`` is a thunk returning an ``Effect``, as for `Gather`,
    and a race keeps gather's barrier: it returns once every branch has ended. A race of zero
    winners starts no branch, so `api.quorum` answers it without yielding this op."""

    want: int
    branches: tuple[Callable[[], Any], ...]
    deadline: datetime | None = None
    """When the race stops waiting for the winners it does not have, and so how many answers it
    has.

    ==========  ==================  ==================================================
    deadline    the author writes   the race answers with
    ==========  ==================  ==================================================
    ``None``    ``race(branches)``  ``Chosen`` or ``Impossible``
    an instant  ``deadline=at``     those, or ``TimedOut`` (`choice.Answer`)
    ==========  ==================  ==================================================

    Absolute, as `AwaitEvent.deadline` is and for the same reason: the caller reads its clock
    through a `step` and hands the instant in, so a restart races against the instant the first
    attempt chose. The clock never stops a branch. A deadline the handler has reached forces the
    parent to decide, and the choice it saves stops the losers the way any other choice does."""

    def __post_init__(self) -> None:
        if not 1 <= self.want <= len(self.branches):
            raise CompositionRefused(
                f"a race over {len(self.branches)} branches cannot want {self.want} winners: "
                "want between 1 and the number of branches"
            )
        if self.deadline is not None:
            refuse_a_naive_instant(self.deadline, "a race's deadline")


@dataclass(frozen=True)
class Scoped[T]:
    """Run ``body`` with every key it mints namespaced under ``scope``.

    A scope is a prefix the **handler** applies while walking the execution tree, never text an
    author splices into a name. It is the mechanism a gather branch's `gather:{g},{i};` frame
    uses (`_PrefixedCtx` on the durable path, the handler's `_prefix` in memory), for a scope an
    author chooses.

    **`scope` is a `Key`.** Every scope atom is a tag plus fields (``rec:0``, ``fold:1,2``,
    ``d:3``), so it is composed by the one composer (``compose_key(t"rec:{i}")``) and is
    injective by construction. The handler owns the ``;`` join (`scope_prefix`); nesting composes
    by nesting `Scoped`.

    Pure structure, like `Gather`: it has no standalone `op_key` (see `op_key`'s arm) and it
    checkpoints nothing of its own. Its *effect* is entirely on the names of the ops inside it.

    An atom may not open with a term the substrate mints. An op arm (`ARM_TAGS`) names frames and
    ops, `gather:{g},{i}` among them, so an author's `gather:1,0` scope would compose the path of a
    different nesting; a scope that places its body where the checkpoint readers see engine
    bookkeeping (`checkpoints.is_engine_internal`: `$awaitEvent:…`, `wake-race:…`) would hide the
    steps under it from them.
    """

    scope: Key
    body: Callable[[], Any]  # () -> Effect[T]

    def __post_init__(self) -> None:
        from effective.checkpoints import is_engine_internal

        if (tag := self.scope.terms()[0].tag) in ARM_TAGS:
            raise CompositionRefused(
                f"scope atom {self.scope.display()!r} opens with the op arm {tag!r}: a scope "
                f"names a namespace, and the arms ({', '.join(ARM_TAGS)}) name frames and ops "
                f"the substrate mints. Choose a tag of your own."
            )
        # Under an enclosing frame, which is where an infix marker such as `;wake-race:` sits.
        if is_engine_internal(frame_path("scope;", self.scope)):
            raise CompositionRefused(
                f"scope atom {self.scope.display()!r} names engine bookkeeping: the checkpoint "
                f"readers set aside a name carrying it, so the steps in this scope would vanish "
                f"from them. Choose a tag of your own."
            )


GENERATION_PARAM = "__generation__"
CARRY_PARAM = "__carry__"
ACCRUAL_PARAM = "__accrual__"
"""The SUBSTRATE's own carry across a generation boundary: `(spent, granted, trips)` for the
measured ceiling. Invisible to an author, exactly as checkpoint ids are — two carries, one per
bookkeeper. What a chain has SPENT is a ledger question (the `respawned` row); this is only
what the next handler needs to keep enforcing. It rides params because a chain is strictly
SEQUENTIAL, one generation at a time, so it needs no budget pool shared across tasks; a
concurrent fleet would. Do not extend this past that boundary."""
"""Spawn-params keys for the two things that survive a generation boundary. Everything else
about the previous task — its checkpoints, its meter, its layer run-scope — is gone by design,
which is the whole point of the cut. They live here beside `Respawn` rather than in
`combinators`, because both handlers read them and a handler must not import a combinator."""

DONE_EVENT_PARAM = "done_event"
"""The event a spawned child answers its parent on, named by the handler that spawned it."""

SUBSTRATE_PARAMS = frozenset({GENERATION_PARAM, CARRY_PARAM, ACCRUAL_PARAM, DONE_EVENT_PARAM})
"""The spawn params only the substrate writes. A spawn a workflow yields may not carry them."""


@dataclass(frozen=True)
class Respawn:
    """End this task and continue the chain in a fresh one — the generation boundary.

    **Authored only by the `respawn` combinator, never by a workflow.** The three names sit on
    three wires: an author writes `respawn(...)` at the call site and returns
    `Again(state)`; the combinator reifies that into this op; the handler acts on it. The author
    never says "respawn" to the interpreter.

    It exists because there is no other way to COMPLETE a task from inside a combinator —
    `DurableHandler._run` exits only by `StopIteration`, an exception, or the ctx's suspend, and
    parking would leave a zombie waiting-task per generation. So the handler arm spawns
    generation *n+1* under a deterministic idempotency key, appends the `respawned` ledger
    event, completes this task, and abandons the generator.

    What survives the cut is `state` and nothing else — the whole point of the boundary is that
    the next generation replays no history. `task` is the workflow's own registered task name,
    because the substrate has to re-spawn the same program.
    """

    task: str
    generation: int
    state: Any
    params: dict[str, Any]
    run_id: str
    granted: int = 0
    """How many generations the grant at THIS boundary authorized; 0 when none was asked for.

    Carried on the op because only the combinator knows it — the handler sees a `Respawn`, not
    the park that answered it — and the `respawned` ledger row has to record it: a granted
    generation is a governance decision, and deriving it from checkpoint rotation would derive
    the canonical bookkeeper from the disposable one."""


type WorkflowOp = (
    Step[Any]
    | AwaitEvent[Any]
    | AppendLedgerRow
    | StoreArtifact[Any]
    | SleepUntil
    | Gather
    | Race
    | Scoped[Any]
    | Respawn
)


class Unretryable(Exception):
    """An error a retry would raise again, so its task fails on the attempt that raised it."""


def leaves(raised: BaseException) -> Iterator[BaseException]:
    """The exceptions inside `raised`, every exception group opened, in order."""
    match raised:
        case BaseExceptionGroup(exceptions=inner):
            for exception in inner:
                yield from leaves(exception)
        case _:
            yield raised


def unretryable(raised: BaseException) -> Unretryable | None:
    """The first leaf of `raised` whose type a retry would raise again. One is enough: the task
    cannot succeed on a retry, whatever the other leaves are."""
    return next((leaf for leaf in leaves(raised) if isinstance(leaf, Unretryable)), None)


_REDERIVED = "_effective_rederived"


def mark_rederived(error: BaseException, rederived: bool = True) -> None:
    """Mark whether `error`, as just raised, is one a retry would raise again: its frame ran
    nothing fresh and read nothing the record does not hold, so a retry replays the same record to
    the same raise. Each raise sets the mark, so an instance raised again carries no earlier
    raise's mark."""
    setattr(error, _REDERIVED, rederived)


class RaceUndecided(ExceptionGroup):
    """A race's branches raised before any choice, and no branch is left to win it. A retry may run
    a branch fresh and win, so a typed error inside is not, on its own, one a retry would raise
    again."""

    def derive(self, excs: Sequence[Exception]) -> RaceUndecided:
        return RaceUndecided(self.message, excs)


def placed_leaves(
    raised: BaseException, *, undecided: bool = False
) -> Iterator[tuple[BaseException, bool]]:
    """Each leaf of `raised`, in order, and whether an undecided race holds it."""
    match raised:
        case BaseExceptionGroup(exceptions=inner):
            within = undecided or isinstance(raised, RaceUndecided)
            for exception in inner:
                yield from placed_leaves(exception, undecided=within)
        case _:
            yield raised, undecided


def futile_leaf(raised: BaseException, *, delayed: bool = False) -> BaseException | None:
    """The first leaf of `raised` a retry would raise again: an `Unretryable` one outside an
    undecided race, or one marked rederived unless the retry is `delayed`, which leaves time to
    deploy a fix to the code that raised it."""
    return next(
        (
            leaf
            for leaf, undecided in placed_leaves(raised)
            if (isinstance(leaf, Unretryable) and not undecided)
            or (getattr(leaf, _REDERIVED, False) and not delayed)
        ),
        None,
    )


def noted[E: BaseException](leaf: E, raised: BaseException) -> E:
    """`leaf`, reported alone, carrying every other leaf of `raised` as a note."""
    for other in leaves(raised):
        if other is not leaf:
            leaf.add_note(repr(other))
    return leaf


class RecordedValueRejected(ValueError, Unretryable):
    """A recorded value its schema rejects: an event's payload or a step's result. The record is
    first-write-wins, so a retry would load the same value and reject it again, and a retry that
    waits keeps no attempt for it: a run that would rescue the value catches the error."""


class CompositionRefused(ValueError, Unretryable):
    """The substrate refused a COMPOSITION: the `LOUD` cell of the composition table, typed.

    A programming error: the workflow composed something the substrate cannot run, and replay
    makes the refusal deterministic, so a retry would raise it again. Its task fails on the attempt
    that raised it, reported as this error, and a spawned child answers its parent `Failed` first,
    so a joining parent learns which child failed and why instead of waiting on it.

    One base for every composition refusal, so a refusal added later is `Unretryable` by
    construction. It subclasses `ValueError`, so an `except ValueError` at a call site still
    catches one."""


class PlacedWriterCollision(CompositionRefused):
    """Two placed ops in one task wrote one canonical `event_id`, so the second row was lost.

    A member of the composition-refusal family, so a collision inside a forked tail fails its
    child once and answers the joining sweep `Failed`, where a crash would park the sweep.

    **Refusal rather than rescoping**, and the asymmetry with the fork axis is principled. A
    canonical `event_id` is the DOMAIN's public address — projections read it, a fork's
    `forked_at_event` points at one — so the substrate silently rewriting it would mutate domain
    meaning. A `hyp:` id is substrate-owned and fenced out of every projection, so `ForkLedger`
    rescoping one is free. Same defect, different owner, different remedy.

    Raised AT THE STORE, after the conflict is observed, which is the only place that can tell a
    collision from the idempotency it must not break: the discriminator is the writer already on
    the row, and nothing above the store has read it."""


def refuse_placed_writer_collision(
    event_id: Key, writer: Writer | None, held: tuple[str | None, str | None] | None
) -> None:
    """Decide what an `event_id` conflict MEANS, given who is writing and who already wrote.

    ONE function because two stores must agree and there is no way to check that they do — a
    `SqliteLedger` and a `PostgresLedger` reaching the same verdict by two hand-written `if`
    chains is the fork-scope-token shape, where two spellings of one rule drifted and nothing
    raised. The stores own the read-back (it has to happen inside their own critical section);
    they do not own the ruling.

    The table is total over (writer known?) x (holder known?) x (same task?) x (same placement?),
    and every row that ALLOWS is a case the substrate documents as correct:

    - either side unknown -> allow. The genesis and seal appends in `run_fork` bypass `ctx.step`
      and re-append live on every attempt; their ids are deterministic and child-scoped, and
      idempotence-by-`event_id` is their stated safety argument. A row written before this column
      existed is unknown for the same reason and must not become a landmine.
    - different task -> allow, SILENTLY. This is the documented feature, not a tolerated
      collision: one message triaged in generation 0 and again in 3 is ONE row, and generations
      share a run id, so nothing but the task tells them apart.
    - same task, same placement -> allow. The crash window: an attempt died between the store
      write and the checkpoint commit, so the retry re-executes this exact placement and re-writes
      this exact row. Refusing here would make a surviving ledger row poison its own run.
    - same task, different placement -> REFUSE. Two ops the substrate placed distinctly wrote one
      address; the second row is already lost."""
    match writer, held:
        case (None, _) | (_, None) | (_, (None, _)) | (_, (_, None)):
            return
        case Writer(task=task), (held_task, _) if task != held_task:
            return
        case Writer(placement=placement), (_, held_placement) if (
            placement.stored() == held_placement
        ):
            return
        case Writer(task=task, placement=placement), (_, held_placement):
            raise PlacedWriterCollision(
                f"two placed ops in task {task} both appended the canonical event_id "
                f"{event_id.display()!r}: {held_placement!r} wrote it and {placement.stored()!r} "
                f"lost the conflict, so ITS ROW IS NOT ON THE RECORD. The ledger is "
                f"UNIQUE(event_id) and append-only, so this cannot be repaired after the fact. "
                f"An event_id is composed by the WORKFLOW and the frames are applied by the "
                f"handler, so two branches of a gather — or two calls in sequence — author the "
                f"same id without seeing that they collide. Give each write an id that says "
                f"which one it is (the branch index, the item it is about); if you meant ONE row, "
                f"write it once, above the fan-out."
            )


def awaits_an_absolute_name(op: WorkflowOp) -> bool:
    """True when this op awaits a name that carries its whole address already.

    The whole guard turns on one question: **can the party that emits this name see the frames
    the handler is applying to my await?**

    For a `RELATIVE` await the answer is yes and the frames complete the question — two scopes
    awaiting a bare ``review:{id}`` are two different questions
    (`_conformance.scoped_await_wf`), and an emitter that knows the workflow's structure
    completes the name with `api.qualified_event_name`. That is the default, and it is why
    nothing an author writes needs to change.

    For an `ABSOLUTE` await the answer is no: the emitter is a **different task** running from
    its own params (`fork.run_fork_as_task` emits the ``done_event`` it was handed, verbatim),
    so it cannot complete a name it has never seen a frame of.

    **The criterion, stated once so a new name can be classified without re-deriving it:**
    *an emitter that composes the name from its OWN params needs `ABSOLUTE`; an emitter that
    reads the parked name off the engine can complete a `RELATIVE` one.* Swept over every
    `await_event` site in `src/` plus the handler-internal parks:

    =======================================  ==========  =========================================
    name                                     addressing  emitter
    =======================================  ==========  =========================================
    ``review:{message_id}``                  RELATIVE    a human/UI told the qualified form
    ``ask:{tag}``                            RELATIVE    reads the park
    ``depth-grant:{run},{gen},depth={d}``    RELATIVE    reads the park (`combinators.human_grant`)
    ``spawn-done:{task},{occ};{placement}``  ABSOLUTE    a spawned child echoing its own params
    =======================================  ==========  =========================================

    Note this is the AWAIT, not the spawn. A `spawn_fork` `Step`'s own key is a place (a
    checkpoint key); framing it is harmless and deterministic, and refusing it would reject a
    composition that works end-to-end. What gets rescoped away from its emitter is the join."""
    return isinstance(op, AwaitEvent) and op.addressing is Addressing.ABSOLUTE


def refuse_respawn_in_branch(frame: str) -> Never:
    """The one refusal both interpreters raise for a `Respawn` inside a gather branch.

    **Raised BEFORE the spawn commits.** A guard downstream of `ctx.step` would let the branch
    enqueue a real durable child and *then* fail, leaving an orphan running detached.

    It also closes a liveness defect. `_ChainContinues` is a
    `BaseException` so an author's `except Exception` cannot swallow it — but on the concurrent
    path an `asyncio.TaskGroup` sits between the raise and `_run`'s catch and wraps it into a
    `BaseExceptionGroup`, which `except _ChainContinues` does not match. Measured: the group
    escapes `work_batch` (both engines catch `Exception`), so the WORKER DIES, no failure is
    recorded, and the task holds a live lease — `sqlite.py`'s own "poison task" comment names the
    shape — while the successor is already committed and the chain runs on. `_GatherPark` avoids
    this by becoming a `BranchParked` VALUE before the TaskGroup sees it; `Respawn` cannot,
    because it has no value to become. So it must never reach a branch at all.

    Respawn is asymmetric in the composition table, and the two directions are named rather than
    called a row and a column, since the two conventions are easy to confuse.
    **As the OUTER** (`respawn ∘ X`) it is unrestricted and contributes no frames: a generation is
    a fresh task with a fresh checkpoint store, so everything composes inside one, leaving every
    key exactly as it was without the generation (measured, `test_composition.py`).
    **As the INNER** (`X ∘ respawn`) it is refused in a gather branch and legal everywhere else —
    four of the seven combinator-holes take a `respawn` inside them, `scoped` among them (an
    ordinary namespacing; the task ends and both interpreters agree via `Ended`)."""
    raise CompositionRefused(
        f"respawn cannot run inside the gather branch {frame!r}: a generation boundary ends the "
        "whole TASK, so a branch that respawns would end its siblings' task too — silently "
        "duplicating or skipping their work depending on branch order, and on the concurrent "
        "path killing the worker outright. respawn is the OUTER loop: put the gather inside a "
        "generation, not a generation inside a branch."
    )


CHAIN_DEPTH: ContextVar[int] = ContextVar("effective_chain_depth", default=0)
"""How many `respawn` chains lexically enclose the code now running, in THIS task run.

Lives here rather than in `combinators` for the reason `GENERATION_PARAM` does: both handlers
read it and **a handler must not import a combinator**. The combinator sets it; the handlers
reset it at a task boundary.

A `ContextVar` rather than a plain global because a gather branch runs in its own thread and
`asyncio.to_thread` copies the context — so a branch correctly inherits "a chain encloses you"
instead of racing a shared counter. Deterministic under replay: it is re-derived by
re-executing the same generators, never read from the world (the determinism boundary is about
I/O; this is in-process control state, like a frame's `FramePosition`).

**Its reset is at the TASK RUN, not in a `finally`, and that was measured the hard way.** The
first version reset in a `finally` around the step's body — which never runs when a gather
branch's refusal propagates out of the handler without being thrown back into the workflow
generator, because the generator is abandoned for the GC. Abandonment is not exceptional here:
it is how a park and a generation boundary both work. So the flag leaked across runs, and in a
long-lived WORKER that means task B refusing because task A left it dirty. Bounding it to the
run makes a leak unobservable — a run that ends dirty has ended — and gives replay the same
zero it started with."""


CHAIN_GENERATION: ContextVar[int | None] = ContextVar("effective_chain_generation", default=None)
"""Which GENERATION of a `respawn` chain the code now running belongs to, or `None` outside one.

`CHAIN_DEPTH`'s sibling, set by the same combinator around the same call, for the same reasons —
and read for a different kind of purpose, which is the part to be careful about. That one is
*control* (refuse a nested respawn); this one reaches an **identity**: `descend`'s grant park
names the generation, so one generation's grant cannot settle the next one's question.

**Why an identity may be derived from it, when identity usually may not.** A name has to be
re-derivable by every walk, and this value is: it is set by ordinary workflow code
(`combinators.respawn`) from `chain.generation`, which the author's task function built from the
task's own params. Replay re-executes that same code and sets the same value; it never reads the
world. That is the same argument `CHAIN_DEPTH`'s docstring makes, and it is why the generation
could NOT come from the handler instead: `ReplayHandler` has no params at all, so a
handler-published generation would be absent exactly where the name has to be recomputed.

`None` rather than `0` for "no chain", because a chain's generation 0 is a real generation and
must be distinguishable from a `descend` that is not in a chain at all — the second composes the
name it always composed, so nothing outside a respawn moves a byte.

Reset with its sibling at the task run, not in a `finally`, for the reason recorded above: a
gather branch's refusal abandons the generator and no `finally` runs."""


def current_generation() -> int:
    """Which generation of a `respawn` chain is asking — `0` outside a chain.

    The read side of `CHAIN_GENERATION`, and it lives here rather than in any one caller because
    it now has four answerers across three modules: `descend`'s two grant answerers
    (`combinators`), `govern`'s gate, and `permission`'s human tier. Those are the substrate's
    authority namespaces, and the coordinate only separates generations if every one of them
    applies it — a namespace that reads the ambient in its own way is a namespace that will read
    it differently.

    **`0` outside a chain is a real generation, not a sentinel**, exactly as `Chain.from_params`
    treats a missing `GENERATION_PARAM`: an unchained run IS generation 0. The `None` the
    ContextVar carries distinguishes "no chain" so an unchained run composes the byte-identical
    name it always did; by the time a name is being composed, that distinction has done its work
    and the answer is a number."""
    return CHAIN_GENERATION.get() or 0


def enter_task_run() -> None:
    """Reset the per-run control state a handler owns at a task boundary.

    Called by both interpreters at the top of `run`. One function so the two cannot drift, and
    named for the boundary rather than the variable so the next piece of per-run state has an
    obvious home."""
    CHAIN_DEPTH.set(0)
    CHAIN_GENERATION.set(None)


def refuse_nested_respawn(task: str) -> Never:
    """A `respawn` lexically inside another `respawn`, in one task — refused, and the refusal
    names the thing the author actually wanted.

    **One task has ONE lifecycle, so exactly one chain can own it.** A generation boundary ends
    the whole task; an inner chain's boundary would therefore end the OUTER chain's task, and
    there is no inner task for it to end. Unrefused, the escaping boundary carries the INNER
    chain's `task`, `run_id` and carry, and `next_params()` writes `__generation__`/`__carry__`
    with no discriminator, so the outer chain's next generation would resume under the inner
    chain's name with the inner chain's state: three silent wrong answers, no error at any
    moment.

    **And it is not a missing feature — it is a mis-factored one.** Two loop axes on one task
    (respawn per turn inside respawn per day) is expressible today as ONE chain whose carry is a
    product and whose step decides which axis advanced. The product of two chains is one chain
    with a product carry; that is what the substrate can actually run, because it is what one
    task's lifecycle can express.

    The other real wish behind nesting — *"my inner loop needs bounded replay history too"* — is a
    different mechanism: a spawned CHILD task chain. That is `fork`/subagent territory, where
    there is a second task to own a second lifecycle, and it works today (measured: three
    generations, each its own task, the answer produced in the last).

    **JOINING one does not work, and the message says so.** The handle a spawner holds
    completes at generation 0 with `{"next_generation": 1, "task_id": …}`; the chain's real
    answer lands in a later task whose id nobody outside holds. So a parent awaiting that handle
    wakes EARLY with a boundary marker instead of an answer: a silent wrong answer, where a hang
    would at least be visible. A chain-done event would repair it; it is unbuilt, so a spawned
    chain is followed by its `workflow_run_id` in the ledger rather than by the task id you
    spawned.

    Transitive by construction: the flag lives on a `ContextVar` set around the step's body, so
    `respawn ∘ scoped ∘ respawn`, `∘ route ∘`, `∘ descend ∘` and `∘ hoisted ∘` all refuse for the
    same reason at any depth — those combinators are ordinary workflow code inside the step. A
    gather or a `recurse` leaf between them is refused earlier, by the branch guard."""
    raise CompositionRefused(
        f"respawn cannot nest inside another respawn: a generation boundary ends the whole TASK, "
        f"so the inner chain would end the outer chain's task and continue under its own "
        f"identity — the outer chain's next generation resuming with the inner chain's carry, "
        f"silently. One task has one lifecycle, so one chain owns it (this one declared "
        f"task={task!r}). If you want two loop axes, make ONE chain whose carry is a product and "
        f"let the step decide which axis advanced. If the inner loop needs a task boundary of "
        f"its OWN, it needs its own TASK: spawn a child that runs its own chain. Note you "
        f"cannot JOIN one yet: the spawned handle completes at generation 0 with a boundary "
        f"marker, so follow the child by its `workflow_run_id` in the ledger (a chain-done "
        f"event is unbuilt)."
    )


def refuse_respawn_under_a_rename(rename: str, task: str) -> Never:
    """A generation boundary inside a FORK CHILD — refused, and the refusal names PROMOTION.

    `fork ∘ respawn` is a LOUD cell whose message names promotion.

    **Everything that makes a counterfactual a counterfactual is built per TASK by
    `fork.run_fork`** — the `ForkLedger` writing to a hypothetical lineage, the `RenamedAwaitCtx`
    giving the child its own event world, the `DryRun` sandbox refusing world writes, the
    `SeedingCtx` seed/phase boundary. A respawn is *defined* as ending the task. So the
    composition means "keep being a counterfactual after the machinery that makes you one is
    gone": the next generation is spawned as the workflow's OWN registered task name, with none
    of the four, and it writes to the canonical ledger, awaits unrenamed names (absorbing the
    base's answers — the defect the rename exists to prevent) and mutates the world for real.

    A second, independent reason: a fork's contract is a **marginal** — a terminating answer the
    parent joins on, with `sealed => valid marginal` as the invariant. A chain has no answer until
    its last generation, and the chain-done event is unbuilt, so a forked chain
    structurally cannot produce the one thing a fork exists to produce.

    **The asymmetry is the tell**, and it matches the other terminals: `respawn ∘ fork` is fine —
    a generation runs a counterfactual inside itself, every fence intact, the marginal computed
    and joined within the generation. Only this order is broken.

    **What the author actually wants is PROMOTION.** A counterfactual that becomes a real,
    long-lived interaction (parking on a human for a week) legitimately wants bounded replay
    history — but then it is not hypothetical any more, and the fences *should* drop. So crossing
    out of a fork's fences is a promotion, and a promotion is an explicit act with a name of its
    own, never a side effect of a generation boundary.

    Without this refusal the composition is guarded only by ACCIDENT: `DryRun` refuses the spawn
    as a `WorldMutation`, on a rule about world writes rather than about counterfactual scope, and
    its message recommends adding `spawn` to `allow`, which removes the guard."""
    raise CompositionRefused(
        f"respawn cannot run inside a fork child (this run's event world is {rename!r}): a "
        f"generation boundary ends the task, and every fence that makes this a counterfactual — "
        f"the hypothetical ledger, the event rename, the DryRun sandbox — is built per TASK. The "
        f"next generation would run as an ordinary {task!r} task: canonical ledger, unrenamed "
        f"awaits, real world writes. A fork must also ANSWER, and a chain has no answer until its "
        f"last generation. If you want this counterfactual to become a real long-lived run, that "
        f"is a PROMOTION and it needs to be an explicit act — not a side effect of a generation "
        f"boundary. Run the chain from the base run, or fork INSIDE one generation "
        f"(`respawn` outside, `fork` inside) — that order composes."
    )


def refuse_a_park_in_a_race_branch(op: WorkflowOp, frame: str) -> Never:
    """The one refusal every interpreter raises for an await or a sleep inside a race branch.

    Decided by the op's kind when the branch yields it, so the same program is refused whether or
    not its event has arrived. A race branch that parks is the arrivals bridge's to rule."""
    raise CompositionRefused(
        f"{type(op).__name__} cannot run inside the race branch {frame!r}: a race branch may "
        "not park, since a race over parked branches is not built yet. Await or sleep before "
        "the race, or race branches that do not wait."
    )


def refuse_a_bounded_wait_in_a_branch(op: WorkflowOp, frame: str) -> Never:
    """The one refusal every interpreter raises for a deadlined wait inside a gather branch.

    A branch resolves its wait by PEEKING — a park is a value the barrier re-arms, and the peek
    surface reads the event alone. A deadline needs the engine's own clock-and-event wait, which
    is the suspend machinery a branch must not touch mid-round. Decided by the op's kind, so the
    same program is refused whether or not the event has arrived."""
    raise CompositionRefused(
        f"a wait naming a deadline cannot run inside the gather branch {frame!r}: a branch peeks "
        "rather than parking, and a peek reads the event alone. Wait on the deadline before the "
        "gather, or gather branches that wait on their events alone."
    )


def refuse_absolute_await_in_branch(op: WorkflowOp, frame: str) -> Never:
    """The one refusal both interpreters raise, so they cannot drift apart.

    Raised only for a gather BRANCH. A scope frame is reroutable: a scope completes RELATIVE
    names and an absolute one needs no completing, so the handler resolves the await at the ctx
    it was constructed with, as for `budget-grant` (`absurd.py`'s `_root_ctx`). A branch
    coordinate is not a naming choice: it is a concurrency slot, and a branch await peeks and
    parks as a *value* that the barrier re-arms with the coordinate re-added, so there is nothing
    to reroute to.

    The message names the FIX, not just the fault, because the fix is a different shape of
    program rather than a smaller edit, and it names the ORPHAN, because the refusal fires
    downstream of the spawn. The guard sits at the await because refusing the spawn would reject
    a working composition, which puts the refusal *after* a side effect: a branch that spawns and
    joins enqueues a real durable child, then fails (`children ENQUEUED before the refusal:
    ['child-task']`). An earlier refusal would re-refuse the legal shapes, so the message says
    what was already enqueued."""
    # `.display()` — the name is going into a refusal an operator reads, and a `Key` has no
    # `__str__`, so `{name!r}` would print the repr instead of the name they must grep for.
    name = op.name.display() if isinstance(op, AwaitEvent) else type(op).__name__
    raise CompositionRefused(
        f"{name!r} is an ABSOLUTE event name, and this gather branch would prepend "
        f"{frame!r} to it — so the await would park forever on a name nothing produces. "
        "The task that emits it runs from its own params and has never seen a frame of ours, "
        "so it cannot complete a relative name. A cross-task fan-in does not need a gather: "
        "the children are already concurrent, each in its own task. Spawn them in a loop, then "
        "join them in a loop, as `effective.fork.marginal_sweep` does. NOTE: if this branch also "
        "spawned the task it "
        "is joining, that child is ALREADY enqueued and will run to completion detached — the "
        "refusal stops the join, not the spawn."
    )


def refuse_absolute_await_under_a_rename(op: WorkflowOp, rename: str) -> Never:
    """The third frame that stands between a composer and an emitter: a fork child's EVENT WORLD.

    `RenamedAwaitCtx` rescopes every awaited name to ``fork:{child_run_id};{name}`` so a child
    re-running the base workflow parks in its own namespace instead of absorbing the base's answer
    (`absurd.py`). That is a frame on the **event axis only** — steps pass through — and it is
    invisible to the branch guard, which reads `prefix`. So a fork child that spawns a
    counterfactual of its own and joins it would await ``fork:{child};{done_event}`` while the
    grandchild emits the bare ``{done_event}`` from its own params: a silent forever-park on both
    engines.

    Reroute is not available here the way it is for a scope. The rename is what the handler was
    BORN with rather than something `_run_scoped` pushed, so `_await_absolute` keeps it *on
    purpose* — bypassing it would let a base's answer resolve a child's park, which is the defect
    the rename exists to prevent. The two frames both hold and they disagree, so the answer is
    to refuse until **rename-aware emit** exists: the grandchild's
    ``done_event`` param would have to carry the child's ``fork:{cid}:`` scope, composed at the
    spawn where both halves are known.

    This enforces `Addressing`'s rule, *an absolute name ignores frames, and prepending to one
    is a bug*, for the rename ctx.

    **Durable-only, by construction and not by omission.** A fork child exists only under
    `fork.run_fork`, which runs on `DurableHandler`; `RecordingHandler` has no child ctx and so
    no state to diverge in. That is the `LAYERED_OPS_DURABLE_ONLY` shape — a NAMED asymmetry —
    not the interpreter drift `refuse_absolute_await_in_branch` exists to prevent."""
    # `.display()` — the name is going into a refusal an operator reads, and a `Key` has no
    # `__str__`, so `{name!r}` would print the repr instead of the name they must grep for.
    name = op.name.display() if isinstance(op, AwaitEvent) else type(op).__name__
    raise CompositionRefused(
        f"{name!r} is an ABSOLUTE event name, and this run's event world would prepend "
        f"{rename!r} to it — so the await would park forever on a name nothing produces. The "
        "task that emits it composes the name from its OWN params and has never seen this "
        "rename. Spawn it from the run that owns the event world instead (the BASE run, not a "
        "child of it) — or, if you DO emit under the renamed form, name it in `transplanted`, "
        "which is the substrate's word for 'the caller has delivered this under the child's "
        "names'. NOTE: the task you are joining is ALREADY enqueued and will run to completion "
        "detached — the refusal stops the join, not the spawn."
    )


def _refuse_an_op_identity_as_an_address(name: str) -> None:
    """An await's name is an ADDRESS; a key that opens with an ARM is an op's own identity.

    `event;step;tool:foo` is well formed and unmintable — it says "await an event whose name is a
    step's checkpoint key", and nothing emits one. The grammar cannot tell those apart on its own
    because the arm's payload is typed `Key`, i.e. ANY key; this is where the domain narrows it.

    Reachability is decidable here precisely because the arm set is CLOSED (`ARM_TAGS`) — the key
    space is a qualification scheme over a finite tag alphabet, not a Turing-complete one, so
    "could anything mint this?" is a check rather than a search."""
    if leads_with_an_arm(name):
        raise ValueError(
            f"await name {name!r} opens with an op ARM ({' '.join(ARM_TAGS)}), so it is an op's "
            f"own identity and not an ADDRESS. Nothing emits an event named after a checkpoint, "
            f"so this await could never be answered. Await the name an emitter actually delivers "
            f"to — an author subject, or a substrate authority key."
        )


def _leading_tag(name: str) -> str | None:
    """The tag of an author name's first term, or `None` when it is not in the language.

    PARSED, not `startswith`. The prefix form went stale the moment the arms began minting `;`,
    and it could not have done otherwise: a prefix carries a separator, and the separator is
    exactly what moved. A tag has no spelling to drift."""
    try:
        return parse(name).terms[0].tag
    except KeySyntaxError, IndexError:
        return None


def event_name(name: str | Key) -> Key:
    """Validate an author-supplied await name, or accept an already-composed one.

    The await axis is the other place an author names an identity, and the only one where a
    reserved namespace has to be REFUSED rather than fenced off structurally. A step name
    composes under its own arm — `step_key("approve;tool:act")` is `step;approve;tool:act` — so
    it is disjoint from the permission layer's `approve` park by construction. An await name
    carries no arm: it is delivered as written. Absurd delivers events by name, first-emit-wins,
    across the whole queue, so an unguarded `await_event("approve;tool:act")` would park on the
    name the permission layer's `human` tier parks on and consume the approval meant for the
    gate. These names ARE the authorization.

    **A `Key` is trusted and a `str` is checked**, which is the marker-type idiom rather than a
    trust-by-location rule: a `Key` came from `compose_key`, so it is registry-visible and the
    substrate's own parks (`depth_grant_name`, an `approve:` gate) pass one. Author text has made
    no such promise. So the substrate's call sites pass their composed names as `Key`s and never
    launder them back to text to satisfy a `str` parameter.

    **It returns a `Key`, so it is a constructor and not a validator that hands back text.** This
    is the await axis's `op_key`: every await writer goes through here the way every checkpoint
    writer goes through `op_key`, which is what licenses `TaskContext.await_event` to type its
    parameter (`handlers.base`, and see `step` there for the same argument on the checkpoint
    axis). The `str` arm ends at `Key.parse` — the named read boundary, and a cast
    consciously accepted — but it is a *checked* one: the two refusals below run first, so author
    text reaches the typed world only after the reserved-namespace and frame-delimiter guards
    have passed on it."""
    # **The DECISION TABLE, not an `isinstance` split.** An `isinstance` chain is
    # the smell here; what this wants is a total structural pattern match — a decision table. An
    # `if isinstance(name, Key): … return name` short-circuit is exactly where a defect hides:
    # the `Key` arm answers "trusted" without ever naming the case where a `Key` is NOT
    # trustworthy. A table has to name its cases, which is what makes a missing arm visible.
    #
    # Order is load-bearing and the comments say why, arm by arm. Total by the wildcard.
    match name:
        # An op's own identity is never an ADDRESS, whichever type it arrives as. FIRST for the
        # `str` arms too: a `;`-bearing identity would otherwise be refused as a forged frame,
        # which is true and unhelpful — it names the wrong defect.
        case Key() if leads_with_an_arm(name.stored()):
            _refuse_an_op_identity_as_an_address(name.stored())
            raise AssertionError("unreachable — the call above raises")  # pragma: no cover
        # A reserved-authority `Key` carrying NO scope was PARSED, not composed. Every authority
        # namespace is declared as an `AuthorityTag`, so the substrate's own park names carry a
        # `Scope` and `Key.parse` never attaches one — which is what separates "I am the
        # substrate" from an author who reached for the read boundary as a shortcut.
        case Key(scope=None) if _leading_tag(name.stored()) in RESERVED_AUTHORITY_TAGS:
            raise ValueError(
                f"await name {name.stored()!r} is in a reserved authority namespace "
                f"({' '.join(RESERVED_AUTHORITY_TAGS)}) and carries no scope, so it was PARSED "
                f"rather than composed. An event delivered by that name answers a substrate gate, "
                f"and Absurd delivers first-emit-wins across the queue — so this await would "
                f"consume the answer meant for it. Compose it through its `AuthorityTag` if you "
                f"ARE the substrate; `Key.parse` is the READ boundary and confers no authority."
            )
        # Any other composed `Key` — including a scoped authority name, which is the substrate's
        # own park, and including `depth_grant_name(...).occurrence(2)`.
        case Key():
            return name
        case str() if leads_with_an_arm(name):
            _refuse_an_op_identity_as_an_address(name)
            raise AssertionError("unreachable — the call above raises")  # pragma: no cover
        case str() if _leading_tag(name) in RESERVED_AUTHORITY_TAGS:
            raise ValueError(
                f"await name {name!r} is in a reserved authority namespace "
                f"({' '.join(RESERVED_AUTHORITY_TAGS)}): an event delivered by that name "
                f"answers a substrate gate — an approval, a grant, a park — and Absurd delivers "
                f"by name "
                f"first-emit-wins across the queue, so this await would consume the answer meant "
                f"for it. Name the await on your own subject, or compose a `Key` if you ARE the "
                f"substrate."
            )
        case str() if TERM_SEPARATOR in name:
            raise ValueError(
                f"await name {name!r} contains the frame delimiter {TERM_SEPARATOR!r}, which "
                f"forges a `scoped(...)` boundary: inside one scope this name parks where a real "
                f"nested scope would, so two different parks share one name. Express nesting with "
                f"`scoped(...)` and keep the await's own name flat."
            )
        # The author boundary: `authored_key` refuses an occurrence, which is the RUNTIME's to
        # assign. The `Key` arms are deliberately NOT checked for one — the substrate's own second
        # ask of an authority name IS `depth_grant_name(...).occurrence(2)`.
        case str():
            return authored_key(name)
        case _:
            raise TypeError(
                f"an await name is a `str` or a composed `Key`, not {type(name).__name__}."
            )
