"""The counterfactual lineage's genesis: how a fork is RECORDED.

A fork is a **sibling lineage** that shares a prefix and diverges. Its recording has two parts,
both here or in `effective.ledger`:

1. **The marker**: every row of a hypothetical run carries `ledger.hypothetical = True` (set
   on the writer), so the canonical view is `WHERE NOT hypothetical`: one predicate, no join.
2. **The genesis**: the lineage's *first* row is a `Forked` event (`kind="forked"`) in the CHILD's
   run id, carrying provenance: which run it forked from, at which parent ledger event, and the
   substitution. The branch graph is then derivable from `forked_from` pointers alone; a fork is
   never an UPDATE, always a new run id with a parent pointer (append-only loves this).

**A fork stores only the divergence** and never a copy of history: the shared prefix
already exists in the parent lineage, so the genesis *references* it (`forked_at_event`) not
copying it. The prefix *results* for replay come from the parent's checkpoints (the disposable
bookkeeper), never the ledger — two bookkeepers, cleanly.

This is substrate, not domain: any workflow can be forked, so `Forked` lives in `effective`,
deliberately NOT a member of any domain's event union, so a domain projection filters those
rows out and never sees a `forked` genesis. The complement to `effective.lineage`, which COMPARES
two recorded lineages; this one RECORDS the fork that produced the second.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from effective.keys import AuthorityTag, Key, Run, Scope, Segment, compose_key
from effective.keys.grammar import KeySyntaxError, ParsedKey, Term, parse
from effective.ops import LedgerRow, Writer

if TYPE_CHECKING:
    from effective.handlers.absurd import LedgerWriter

FORKED_KIND = "forked"


class ForkedDeadline(RuntimeError):
    """A counterfactual reached a wait that named a deadline — refused, for `ForkedSleep`'s reason.

    A bounded wait IS a clock, wearing an event's name. Both honest readings are the ones that
    argument already rules out: park the true run until the dream's deadline, or answer from a
    clock that never ran. The fork's own grants map cannot decide it either, because the grant
    says what arrived and the deadline asks whether it arrived IN TIME — a question about the
    real elapsed time the counterfactual is not spending.

    Fork a region whose waits name no deadline, or answer the wait before the fork point and
    fork on what it returned."""

    def __init__(self, name: Key, deadline: datetime) -> None:
        super().__init__(
            f"a fork reached a wait on {name.display()!r} bounded by {deadline.isoformat()}: a "
            "counterfactual explores a decision, not a clock, so there is no honest answer to "
            "'did it arrive in time' in a branch that spends no time. Fork a region whose waits "
            "name no deadline, or await it before the fork point and fork on the outcome."
        )
        self.name = name
        self.deadline = deadline


class ForkedSleep(RuntimeError):
    """A counterfactual reached a `SleepUntil`: refused by design, never handled.

    A fork is a *dream* — a hypothetical branch off the real run — and a durable sleep inside it
    is a dream-within-a-dream whose clock runs against reality's (the Inception problem). Both
    ways of honoring it are incoherent:

    - **Park it for real** and you sedate the true run for a day so the dream can reach its own
      morning — then the fork wakes under `DryRun`, where world reads are refused, and cannot see
      the world the wait was for. Real time spent, nothing learned (and an N-fork sweep would
      suspend N real days).
    - **Skip it** and you report a marginal for timing that never elapsed.

    A counterfactual explores a decision, not a clock, so the only honest kick is to stop here.
    Fork a region with no durable sleep, or move the time dependency out of the forked tail.

    **Lives HERE, apart from the rest of the sandbox vocabulary in `effective.sandbox`** (which
    re-exports it): the DURABLE fork raises it from `SeedingCtx` (`handlers/absurd.py`), so the
    refusal has to sit where the handler can reach it. One definition governs both driver
    families, the in-process drivers and the durable child; without the durable arm, a forked
    tail's sleep would park the child against the wall clock.

    PHASE-SENSITIVE on the durable path. The refusal fires only in the `Live` phase — the forked
    tail. A sleep in the replayed PREFIX happened in reality, is always already elapsed, and is a
    durable no-op, so it passes through (the same distinction `fork_at` makes: it replays a
    recorded prefix sleep structurally and completes).

    DRIVER NOTE: `measured_drive` REFUSES an already-elapsed
    prefix sleep too — its prefix is the Step-only projection (`NON_STEP` filters `sleep:` out),
    so the recorded sleep is met "live" and trips this guard. That over-refusal is latent (no
    workflow sleeps) and is the measured driver's own Step-only contract — fork a sleeping run
    with `fork_at` or `run_fork`, not `measured_drive`. Pinned on all three drivers
    (`test_measured_fork_ops.py`, `test_fork_durable.py`)."""

    def __init__(self, when: datetime) -> None:
        super().__init__(
            f"a fork reached sleep_until({when.isoformat()}): a counterfactual explores a "
            f"decision, not a clock, so a fork never sleeps. Fork a region with no durable "
            f"sleep, or move the sleep out of the forked tail."
        )
        self.when = when


class ForkedPrefixAwait(RuntimeError):
    """A fork's replayed PREFIX reached an `await_event` that is not the fork point — refused.

    The sibling of `ForkedSleep`, and the same shape of failure caught one phase earlier: a
    counterfactual meeting a *wait* it has no answer for. The two differ in what makes them
    incoherent — a fork's sleep runs a dream's clock against reality's, while a fork's prefix
    await asks reality a question that was **already answered**, in an event namespace the child
    cannot see.

    Mechanically: the child re-runs the base workflow in its OWN event world (`RenamedAwaitCtx`
    rewrites every await to `fork:{child_run_id};{name}`), so the base's delivered answer —
    recorded under the base's name, on the base's task — is not there. The child parks. Nobody
    emits, because the only event a fork driver emits is the delta at the fork point. The task
    sits in `waiting` forever: no exception, no attempt burned, no failure row. The one failure
    mode a durable substrate cannot shrug off, since a liveness bug leaves nothing behind to read.

    **The motivating case is handler-internal.** A `budget-grant:{run_id},{trip}` park is
    injected by the budget layer during the replay itself: the seeded prefix re-folds each raw
    envelope's usage, so the meter re-crosses the same ceiling at the same op and the same park is
    re-yielded. A base that parked for a grant once, and got it, would fork into a child that
    parks for it forever (`waiting` on `fork:r-fork:budget-grant:r-base:0`, `attempt=0`, no
    failure, and emitting the delta at the fork point changes nothing).

    So the refusal deliberately covers every namespace: any non-fork-point await in the
    `Seeding` phase deadlocks by the same mechanism, whoever authored it.

    **The opt-out is `transplanted`** (`SeedingCtx`/`run_fork`). Naming the awaits you have
    delivered into the child's namespace is the promise this refusal is checking for, and it is
    the seam the unbuilt prefix-await transplant will arrive through: when a reader can re-emit
    the base's recorded answers under the child's names, it passes those names here and the
    pass-through is *earned*. Without it the fork refuses: a fork of a region with a prefix await
    is unsupported, and saying so at the await beats discovering it as a task that never finishes.

    The `Live` phase never raises it. A tail await parks the child on purpose: that is how the
    driver delivers the delta at the fork point, and how a counterfactual can itself be
    answered."""

    def __init__(self, name: Key, *, fork_point: Key, scoped_name: Key) -> None:
        # All three render through `.display()`: they are `Key`s, which have no `__str__`, and
        # this message exists to hand an operator the names they must grep for and pass back as
        # `fork_point=`. A repr here would print `Key(_value='…')` at exactly that moment.
        shown, point, scoped = name.display(), fork_point.display(), scoped_name.display()
        super().__init__(
            f"a fork's replayed prefix awaited {shown!r}, which is not the fork point "
            f"{point!r}. The child would park on {scoped!r}, its own event namespace, "
            f"where the base's recorded answer does not exist; nothing emits there, so the "
            f"task would wait forever with no error, no failure and no attempt burned. "
            f"Fork AT this await instead (`fork_point={shown!r}`), or fork a region "
            f"that does not contain it; if you have delivered the base's answer into the child's "
            f"namespace, name {shown!r} in `transplanted`."
        )
        self.name = name
        self.fork_point = fork_point
        self.scoped_name = scoped_name


class ForkPointInGather(RuntimeError):
    """A fork point named an await INSIDE a `gather` region, refused.

    `ForkPointRefused` is about forking at a non-DECISION op (a write, a clock); here the op *is*
    a decision and the region is the problem.

    **A branch's await is resolved by `peek_event`.** A branch peeks and parks as a value, and
    only `_join`'s re-arm issues the run's one real `await_event`. So the fork point fires on the
    first pass, at the re-arm; on every replay the delivered payload comes back through the peek,
    `await_event` is never called, and the phase never crosses. A durable fork *is* replay, so a
    boundary that fires only on the first pass is no boundary, and the resume would die with a
    `SeedBoundaryError` that blames `through`.

    **A gather region is ONE node in the fork's order.** Fork at an await before it, or after the
    whole region. The same rule governs `fork_seed`'s `through`, which cuts by commit order;
    cross-branch commit order is a race by design, so a cut inside the region is not even stable
    run to run."""


class Forked(BaseModel):
    """The genesis event of a counterfactual lineage — its provenance, event-sourced.

    `forked_from` is the parent run id; `forked_at_event` is the parent ledger `event_id` the fork
    branches at: the human-meaningful **address** of the fork point (the ledger event is the
    address; the trace index is the seek). `at_op_index` is that seek once resolved, an
    **all-ops** index, the same convention as the fork's `at`. It is optional because resolving a
    ledger event to a trace position is the fork DRIVER's job, so the event can record the address
    and leave the seek unfilled.

    `delta` describes the substitution: what the fork replaced at the fork point. It rides the
    JSONB payload rather than a column, so its shape can sharpen (a first-class `ForkDelta`)
    without a migration."""

    event_id: Key
    forked_from: str
    forked_at_event: str
    delta: dict[str, Any] = Field(default_factory=dict)
    at_op_index: int | None = None
    kind: Literal["forked"] = FORKED_KIND


FORK_SEALED_KIND = "fork_sealed"


class ForkSealed(BaseModel):
    """The **terminal attestation** of a counterfactual lineage — "this marginal is valid".

    A fork's genesis is appended BEFORE the handler runs and its boundary checks run AFTER it
    completes, so a fork `run_fork` declares corrupt has already durably committed its whole tail —
    byte-identical in shape to a valid one. The refusal was loud to the caller and **silent in the
    canonical bookkeeper**, and `effective.lineage` reads the ledger, not the task row. One
    bookkeeper's truth is never derived from the other's, so without the seal a VOI consumer has
    no way to know the marginal it reads is invalid.

    **The attestation is POSITIVE, written on the clean path.** The refusal that most needs
    recording is a worker death mid-fork, and a dying worker structurally cannot write "I
    failed". A seal written only once every boundary check has passed is therefore the only
    crash-safe polarity: absence means "not known valid", which covers refusal, crash, and
    still-running alike. It also gives fork selection its predicate (`sealed ⇒ valid marginal`)
    from the canonical bookkeeper alone.

    An event-vocabulary addition, not a schema migration: no DDL, `kind` is the open grammar
    `Forked` already relies on, and the `hypothetical` fence keeps it out of every projection."""

    event_id: Key
    forked_from: str
    kind: Literal["fork_sealed"] = FORK_SEALED_KIND


def _as_row(event: Any, **override: Any) -> LedgerRow:
    """A typed event (or another row) as a `LedgerRow`, with the identity passed as an identity.

    The same split a domain's event-to-row helper makes, for the substrate's own fork events: dump
    the domain fields to JSON-able data, hand `event_id` over as a `Key`. One helper because both
    genesis/seal construction and the fork rescope need it."""
    payload = event.model_dump(mode="json")
    payload.pop("event_id", None)
    payload.pop("kind", None)
    return LedgerRow(event_id=override.get("event_id", event.event_id), kind=event.kind, **payload)


def fork_sealed_name(child_run_id: str) -> Key:
    """The seal's ledger id — `fork-sealed:{child_run_id}`.

    Distinct from `FORK_SEALED_KIND`, which is the ledger row's stored DISCRIMINATOR and a
    different axis entirely: the tag names the identity, the kind names the row type. A grep for
    `fork_sealed` sees both and overstates any rename by about 3x, which is why they are separated
    here. One place composes it."""
    return compose_key(t"fork-sealed:{Run(child_run_id)}")


def sealed_row(child_run_id: str, *, forked_from: str) -> LedgerRow:
    """The lineage's LAST row — appended only once every boundary check has passed.

    Deterministic id (`fork-sealed:{child_run_id}`), so it is idempotent by `event_id` exactly like
    the genesis and a crash-replay of the fork's tail cannot write a second seal."""
    return _as_row(ForkSealed(event_id=fork_sealed_name(child_run_id), forked_from=forked_from))


def genesis_row(
    child_run_id: str,
    *,
    forked_from: str,
    forked_at_event: str,
    delta: dict[str, Any] | None = None,
    at_op_index: int | None = None,
) -> LedgerRow:
    """Build the ledger row for a fork's genesis — the child's hypothetical ledger's FIRST append.

    Written by the child (not the parent), so the whole child lineage is marked hypothetical and
    the genesis names its own event id deterministically (`forked:{child_run_id}`) — idempotent by
    `event_id` like every other ledger append, so a crash-replay of the fork's first step does not
    write a second genesis."""
    return _as_row(
        Forked(
            event_id=compose_key(t"forked:{Run(child_run_id)}"),
            forked_from=forked_from,
            forked_at_event=forked_at_event,
            delta=delta or {},
            at_op_index=at_op_index,
        )
    )


# --- the child lineage's IDENTITY: rescoping event ids -------------------------------------
#
# A fork RE-RUNS the base workflow, so its divergent tail authors the SAME event ids the base did:
# a ticket workflow authors `reviewed:{ticket_id}` / `committed:{ticket_id}`,
# which embed no run id. Those ids collide with the base's CANONICAL rows on the ledger's global
# `UNIQUE(event_id)`, and `ON CONFLICT DO NOTHING` would silently DROP the fork's divergent row
# (on both engines). Hence a lineage-scoped identity:
# every hypothetical row's event id carries the child run id, so a fork's `reviewed:m1` and the
# base's `reviewed:m1` are distinct rows that `effective.lineage.marginal` re-aligns by stripping
# the scope token (the divergence then surfaces at the event's *content*, not its id).
#
# ONE scheme, two directions: `fork_scoped` WRITES the scope, `fork_unscoped` REMOVES it. If the
# two drifted, alignment would fail *silently*, so both read the same `FORK_SCOPE` tag off the
# STRUCTURE, and are pinned together (`test_fork_ledger.py`). A string PREFIX on the strip side
# would stop matching on a separator change with nothing raised.

FORK_SCOPE = AuthorityTag("hyp", scope=Scope.QUALIFIED)
"""The tag marking a hypothetical row's event id as lineage-scoped (vs a bare `event_id`)."""


def fork_scoped(child_run_id: Segment, event_id: Key) -> Key:
    """Rescope one event id into the child lineage's namespace: `hyp:{child_run_id};{event_id}`.

    Composed through `compose_key`: `child_run_id` is an interior hole and therefore a `Segment`,
    delimiter-free by type, so `event_id` rides the terminal position verbatim and stays readable
    (`hyp:cf-a;reviewed:m1`, where escaping would give `hyp:cf-a:reviewed%3Am1`). A `:`-bearing
    lineage id is refused at `Segment(...)` construction.

    The tag rides an interpolation (`FORK_SCOPE`, a `Tag`), so this namespace has exactly ONE
    spelling. A `Tag` interpolation is a runtime value that a simplistic source scan cannot read;
    the registry lint resolves the bounded case, a same-file module-level constant, and reports
    anything more dynamic as *unregistrable*."""
    return compose_key(t"{FORK_SCOPE}:{Run(child_run_id)};{event_id:domain=address}")


def fork_unscoped(event_id: str, child_run_id: str) -> str:
    """`fork_scoped`'s inverse: `hyp:{child_run_id};{id}` back to `{id}`, and anything else
    unchanged.

    What alignment needs — a fork's `hyp:cf-a;reviewed:m1` has to compare equal to the base's
    `reviewed:m1` by IDENTITY, which is `effective.lineage.marginal`'s contract.

    Matches the FRAME. The scope is one leading term whose tag is `FORK_SCOPE` and whose
    coordinate is this child, so recognizing it is a `match` over two fields. Stripping a built
    string `hyp:{cid};` asks the same question after the answer is flattened, and a separator
    change then silently stops the match: `marginal` reports `shared_prefix=0` with unstripped
    rows in the tail, and nothing raises.

    `str` in and out because a ledger row's `event_id` arrives from storage as text, and a row
    whose id is not in the language still has to project rather than raise."""
    try:
        terms = parse(event_id).terms
    except KeySyntaxError:
        return event_id
    match terms:
        case [Term(tag=str(tag), coordinates=(coordinate,)), *rest] if (
            tag == FORK_SCOPE and coordinate.render() == child_run_id and rest
        ):
            return ParsedKey(tuple(rest)).render()
        case _:
            return event_id


class ForkLedger:
    """A `LedgerWriter` that gives a fork child its own LEDGER IDENTITY.

    Wraps the child's `hypothetical=True` base writer, which marks the whole lineage, and
    rescopes every appended row's `event_id` with `fork_scoped`, so a divergent tail row can never
    collide with the base's canonical row on the global `UNIQUE(event_id)`. Engine-agnostic: it
    wraps the one-method `LedgerWriter` protocol, so the same wrapper serves Postgres and SQLite.

    Refuses a non-hypothetical base *loudly*: a fork never appends a canonical row, and rescoping
    a canonical writer's ids would be exactly that mistake."""

    def __init__(self, base: LedgerWriter, *, child_run_id: Segment) -> None:
        if isinstance(base, ForkLedger):
            raise ValueError(
                "ForkLedger cannot wrap another ForkLedger: the id would be rescoped twice "
                "(`hyp:{outer};hyp:{inner};{id}`), and a fork's marginal is defined against ONE "
                "child. Wrap the underlying hypothetical writer once."
            )
        if not getattr(base, "hypothetical", False):
            raise ValueError(
                "ForkLedger requires a hypothetical base writer (construct it with "
                "hypothetical=True); a fork must never append a canonical row."
            )
        self._base = base
        self._child_run_id = Segment(child_run_id)
        self.hypothetical = True  # so a nested ForkLedger is refused by its OWN check, below

    def append(self, row: LedgerRow, *, writer: Writer | None = None) -> None:
        # Rebuilt rather than mutated: the caller's row is also the payload source, and `LedgerRow`
        # is frozen. Going back through the constructor (not `model_copy`) keeps the rescoped id
        # under the same `is_instance(Key)` check every other append passes — a rescope is still
        # a write, and this is the one place a child's ids are minted.
        #
        # `writer` rides through UNTOUCHED, and that is what makes the collision check cover a
        # counterfactual for free: the rescope happens here, above the store, so by the time the
        # store reads back a conflict it is comparing two writers within one hypothetical lineage
        # exactly as it would within the canonical one. A fork that lost a tail row is the case
        # that made `sealed => valid marginal` false.
        rescoped = fork_scoped(self._child_run_id, row.event_id)
        self._base.append(_as_row(row, event_id=rescoped), writer=writer)


__all__ = [
    "FORKED_KIND",
    "FORK_SCOPE",
    "FORK_SEALED_KIND",
    "ForkLedger",
    "ForkPointInGather",
    "ForkSealed",
    "Forked",
    "ForkedSleep",
    "fork_scoped",
    "fork_unscoped",
    "genesis_row",
    "sealed_row",
]
