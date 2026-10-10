"""fork-as-replay: the in-process VOI probe driver.

`fork(run, at, delta) ≡ replay(trace[:at]) then interpret(tail under delta)`: the
fork-counterfactual model, built as an in-process driver over the recording/replay core.

| piece           | does                                                                     |
|-----------------|--------------------------------------------------------------------------|
| `replay_prefix` | drives a workflow generator through `trace[:at]` feeding **only recorded |
|                 | results**, op-key aligned. The prefix is free: the tail domain need      |
|                 | carry no prefix responses, so a re-executed prefix would fail.           |
| `fork_at`       | replays the prefix, guards the fork point, feeds a **counterfactual**    |
|                 | result for the op at `at`, and hands the live generator to `live_drive`. |
| `live_drive`    | the tail interpreter: runs each `Step` against a real metered domain,    |
|                 | folding `run_metered` usage, and answers or parks on `AwaitEvent`.       |

`fork_at` takes `at` two ways: `at < len(trace)` forks a *recorded* op (validated against
`trace[at]`), and `at == len(trace)` forks a *pending park*, which has no trace entry (it is a
`Suspended`), so it is validated against the pending event name. A tail that re-parks returns
`ForkTail(parked_at=…)`: the cheap **probe**, one level deep, that informs a grant of N.

`live_drive` forks **workflow-yielded** parks (the count budget). A *measured* dollars park is a
handler-internal trip around a `Step`, so `measured_drive` (below) folds the meter and enforces
the trip itself: the dollars fork. Both handle `Step`/`AwaitEvent`
only (a world-inert tail); neither applies `op_layers`, recurses on `Gather`, or replays a
recorded `TraceEntry.error`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, NewType, Protocol, assert_never

from pydantic import BaseModel, Field

from effective.api import Effect
from effective.budget import (
    Cleared,
    Exceeded,
    MeasuredBudget,
    Parked,
    enforce_measured,
)
from effective.budget import Grant as Grant  # explicit re-export (pinned: test_measured_fork)
from effective.budget import TripOutcome as TripOutcome  # explicit re-export (same pin)
from effective.cancel import cancelled_or
from effective.checkpoints import Checkpoint
from effective.cost import Contract, Usage
from effective.counterfactual import (
    ForkedPrefixAwait,
    ForkLedger,
    ForkPointInGather,
    genesis_row,
    sealed_row,
)
from effective.domain import DomainOp, Spawned
from effective.engines.absurd import _adapt_ctx
from effective.govern import BudgetRefused
from effective.handlers.absurd import (
    DurableHandler,
    LedgerWriter,
    Live,
    RenamedAwaitCtx,
    SeedBoundaryError,
    Seeding,
    SeedingCtx,
    _decode_usage_envelope,
    fork_event_name,
    metered_call,
)
from effective.handlers.base import (
    TaskContext,
    TraceEntry,
    op_key,
    placed_await_name,
    placed_key,
    placing,
    walk_run,
)
from effective.handlers.replay import ReplayMismatch
from effective.keys import FramePosition, Key, Segment, compose_key, frame_path
from effective.ops import (
    AwaitEvent,
    Gather,
    Race,
    Scoped,
    SleepUntil,
    Step,
    WorkflowOp,
)
from effective.sandbox import (
    DryRun,
    ForkedSleep,
    ForkPointInsideScope,
    ForkPointRefused,
    Observed,
    Unanswered,
    WorldMutation,
    inspect_only,
)
from effective.spawning import REFUSALS as CHILD_REFUSALS
from effective.spawning import (
    ChildAnswer,
    Crashed,
    Failed,
    Refusal,
    Returned,
    answer_with,
    deliver,
    join_child,
    spawn_child,
    stopped_at,
)

# --- the `at` index convention --------------------------------------------------------------
#
# "Fork a run AT op N" needs N in one coordinate system, and the substrate carries an op index in
# two:
#
# | index       | counts           | used by                                         |
# |-------------|------------------|-------------------------------------------------|
# | `OpIndex`   | every yielded op | the public `at` ("fork at the review decision") |
# | `StepIndex` | `Step` ops only  | the bridge and meter prefix (the rest is free)  |
#
# `to_step_index` is the one conversion. Each is a `NewType`, so ty refuses passing one where the
# other is wanted. Only the `OpIndex` direction is checked: no signature wants a `StepIndex`, so an
# all-ops index slicing a Step-only prefix (`prefix[:at]`) still type-checks. `run_fork` anchors
# its seed boundary on a key (`fork_seed`'s `through`) rather than a `StepIndex` count.

OpIndex = NewType("OpIndex", int)
"""A position in the FULL op stream: the public `at`, what an operator's fork point means."""

StepIndex = NewType("StepIndex", int)
"""A position among `Step` ops only, internal to the bridge and meter (the prefix length)."""


def to_step_index(trace: Sequence[TraceEntry], at: OpIndex) -> StepIndex:
    """The ONE all-ops -> Step-only conversion, at the ONE boundary: an all-ops fork point becomes
    the Step-prefix length the measured machinery and the bridge both index by. Total and pure.

    Counts `Step` ops in `trace[:at]`. The boundary invariant, that this agrees with the bridge's
    own Step filter (`checkpoints.is_step_checkpoint`), is
    pinned over an in-process trace. The durable record cannot host the pin: SQLite leaves
    awaits and sleeps unpositioned, and Absurd would need the forbidden `updated_at` order."""
    return StepIndex(sum(1 for e in trace[:at] if isinstance(e.op, Step)))


GATHER_PREFIX = "gather:"
"""The branch-key namespace a `gather` composes (`_PrefixedCtx`). Named because the fork has one
rule about it (a fork point may not live inside a gather region), and a bare literal at that
check would drift from the handler that mints the prefix."""

RACE_PREFIX = "race:"
"""The branch-key namespace a `race` composes, under the same rule as `GATHER_PREFIX`."""


class MeteredDomain(Protocol):
    """A domain that reports usage per op — `MeteredInterpreter` satisfies it
    (`cost.py:267`), and so does any test double with the same shape."""

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]: ...


@dataclass
class ForkTail:
    """The result of interpreting a forked tail. `result` is set when the tail ran to
    completion; `parked_at` is set when it re-parked (the probe case) — exactly one of the
    two. `trace`/`usage` cover the tail ONLY (the free prefix is excluded by construction —
    that exclusion is the measured free-prefix payoff)."""

    trace: list[TraceEntry] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    result: Any = None
    parked_at: Key | None = None


@dataclass(frozen=True)
class Delivery:
    """What a replayed prefix hands the generator to advance it: a recorded VALUE, or a recorded
    REFUSAL to throw.

    A value, not a bare `send` argument, because the distinction has to cross a function
    boundary — `replay_prefix` stops one op short of the fork point, so the last delivery is made
    by its caller. Returning the value and leaving the caller to remember the `error` arm is what
    silently turned a refused prefix op into `None`: the fork then explored the branch the base
    run did not take, with no mismatch to catch it.

    `ReplayHandler` makes the same distinction inline (`replay.py`), where it never leaves the
    loop."""

    value: Any = None
    refusal: BaseException | None = None

    @classmethod
    def of(cls, entry: TraceEntry) -> Delivery:
        return cls(refusal=entry.error) if entry.error is not None else cls(value=entry.result)

    def into(self, gen: Any) -> Any:
        """Advance `gen` past this op, re-raising a recorded refusal inside the workflow so a
        `try/except` that caught it on the record run catches it again."""
        return gen.throw(self.refusal) if self.refusal is not None else gen.send(self.value)


def replay_prefix[T](
    program: Callable[[], Effect[T]], trace: list[TraceEntry], at: OpIndex
) -> tuple[Any, Delivery]:
    """Drive `program()` through `trace[:at]`, feeding recorded results only (never the
    domain). Returns `(gen, next_send)` positioned so the next `gen.send(next_send)` yields
    the op at `at`. `at` in `[0, len(trace)]`. Raises `ReplayMismatch` on a divergent key.

    A `Scoped` in the prefix is REPLAYED THROUGH: its body is driven to completion against the
    recorded entries (which carry the scope prefix, so the keys still match) and its return value
    is sent into the parent. What is refused is a fork point *inside* a scoped body — `at` would
    then name an op whose parent frame this function cannot hand back, since the body is a
    sub-generator the driver runs rather than one the workflow `yield from`-ed."""
    if not 0 <= at <= len(trace):
        raise ValueError(f"fork index {at} out of range [0, {len(trace)}]")
    gen = program()
    delivery = Delivery()
    # A plain running index, not an `OpIndex`: this walks the trace, it does not NAME a fork
    # point. `at` stays the typed one — it is the caller's coordinate (the at-index convention).
    i = 0
    # This walk's positional ordinals, exactly as a handler keeps them: a replayed prefix has
    # to re-derive the same `sleep:{n}` the record assigned, and it can only do that by counting
    # the same way.
    position = FramePosition()
    while i < at:
        op = delivery.into(gen)
        if isinstance(op, Scoped):
            delivery, i = _replay_scoped(op, trace, i, at, prefix="", position=position)
            continue
        with placing(op, position):
            if (actual := placed_key(op)) != trace[i].key:
                raise ReplayMismatch(f"prefix divergence at {i}: {actual!r} != {trace[i].key!r}")
        delivery = Delivery.of(trace[i])
        i += 1
    return gen, delivery


def _replay_scoped(
    op: Scoped[Any],
    trace: list[TraceEntry],
    i: int,
    at: OpIndex,
    *,
    prefix: str,
    position: FramePosition,
) -> tuple[Delivery, int]:
    """Replay a `Scoped` body against `trace`, returning `(body_result, next_index)`.

    The recorded keys already carry the scope, so matching stays exact — this only has to keep
    the flat trace index in step with a nested execution. Nested scopes recurse, continuing the
    enclosing walk's `position`, as every walk counts a scope's ordinals."""
    scope = frame_path(prefix, op.scope)
    body = op.body()
    delivery = Delivery()
    while True:
        try:
            inner = delivery.into(body)
        except StopIteration as done:
            return Delivery(value=done.value), i
        if i >= at:
            # the fork point is INSIDE this scope — see `replay_prefix`'s docstring
            raise ForkPointInsideScope(inner, scope)
        if isinstance(inner, Scoped):
            delivery, i = _replay_scoped(inner, trace, i, at, prefix=scope, position=position)
            continue
        with placing(inner, position):
            if (actual := placed_key(inner).prefixed(scope)) != trace[i].key:
                raise ReplayMismatch(f"prefix divergence at {i}: {actual!r} != {trace[i].key!r}")
        delivery = Delivery.of(trace[i])
        i += 1


def fork_at[T](
    program: Callable[[], Effect[T]],
    trace: list[TraceEntry],
    at: OpIndex,
    substitute: Any,
    domain: MeteredDomain,
    *,
    pending_name: Key | None = None,
    grants: dict[Key, Any] | None = None,
) -> ForkTail:
    """Fork `program` at op `at`, substituting `substitute` as that op's result, and run the
    tail live under `domain`. For a pending-park fork (`at == len(trace)`) pass `pending_name`
    (the parked event name) — there is no `trace[at]` to validate against."""
    gen, delivery = replay_prefix(program, trace, at)
    op = delivery.into(gen)  # the op AT `at` (the fork point) — consumed here, guarded below
    # The fork point must be a DECISION to substitute. Guard the op CLASS before the key check:
    # a fork AT a sleep, write or concurrent op is refused by class, and op_key is not defined for
    # a `Gather` (it raises, masking the typed refusal). Only `Step`/`AwaitEvent` reach the key
    # check, and the same guard covers the fork point and the tail.
    match op:
        case Step() | AwaitEvent():
            pass  # substitutable — a decision result / an event answer; validated below
        case SleepUntil(when=when):
            raise ForkedSleep(when)  # the Inception refusal, at the fork point too
        case Gather() | Race():
            raise TypeError(  # a concurrent op: the same refusal as in the tail arm
                f"cannot fork at a {type(op).__name__}: `fork_at` replays one sequential "
                f"schedule and cannot interpret a concurrent op. Fork at a decision before or "
                f"after it, or use the durable driver (`run_fork`), which runs a `Gather` in "
                f"its forked tail."
            )
        case Scoped():
            raise TypeError(  # structure, not a decision — and it has no key to match `at` on
                "cannot fork at a Scoped: a fork point is a DECISION to substitute (a step's "
                "result, an event's answer), and a scope is pure structure with no op key of "
                "its own. Fork at the Step or AwaitEvent inside the body — its key carries the "
                "scope prefix, so it names the point precisely."
            )
        case _:
            raise ForkPointRefused(op)  # AppendLedgerRow / StoreArtifact — a write, not a decision
    if at < len(trace):
        expected = trace[at].key
    elif pending_name is not None:
        expected = op_key(AwaitEvent(pending_name, object))  # `event:{pending_name}`
    else:
        raise ValueError("pending-park fork (at == len(trace)) requires pending_name")
    if op_key(op) != expected:
        raise ReplayMismatch(f"fork point {at}: {op_key(op)!r} != {expected!r}")
    return live_drive(gen, substitute, domain, grants or {})


def _as_placed(op: WorkflowOp) -> WorkflowOp:
    """An await re-stated under the name the walk placed it at; every other op unchanged.

    Both fork drivers answer an await through `sandbox.inspect_only`, which looks the name up in
    a `grants` map — so the occurrence has to be ON the op before that lookup, not applied to the
    result. Restating the op is the smallest way to say that, and it keeps `inspect_only` a pure
    function of what it is handed."""
    if isinstance(op, AwaitEvent):
        return AwaitEvent(
            placed_await_name(op), op.schema, addressing=op.addressing, deadline=op.deadline
        )
    return op


def _live_loop(
    gen: Any,
    send: Any,
    domain: MeteredDomain,
    grants: dict[Key, Any],
    *,
    prefix: str = "",
    position: FramePosition | None = None,
) -> ForkTail:
    """Interpret a live tail: each `Step` runs against `domain.run_metered` (usage folded into
    the returned `ForkTail`); an `AwaitEvent` is answered from `grants` or re-parks (returning
    a `ForkTail` with `parked_at`). `send` seeds the first `gen.send` — the substituted result
    of the fork-point op.

    `prefix` is the scope path this generator runs under, so a tail that enters a `scoped(...)`
    keys its ops exactly as the base run did. A park inside a scope ends the tail, as any park
    does — a fork tail is one-shot, so no parent frame has to survive it."""
    # This driver's ordinals. It refuses `SleepUntil` (`ForkedSleep`), so it needs no positional
    # coordinate, but an await's OCCURRENCE is one: the fork tail composes the same name the base
    # run parked on. A scoped body is handed its parent's `position`, so a scope entered twice
    # counts on.
    position = FramePosition() if position is None else position
    tail = ForkTail()
    while True:
        try:
            op = gen.send(send)
        except StopIteration as done:
            tail.result = done.value
            return tail
        match op:
            case Step(op=inner):
                result, usage = domain.run_metered(inner)
                tail.usage = tail.usage + usage
                tail.trace.append(TraceEntry(op_key(op).prefixed(prefix), op, result))
                send = result
            case Scoped(scope=scope):
                # Structure: drive the body under the extended path and splice its trace in.
                # A park inside ends the whole tail (one-shot), so no frame has to survive.
                inner_tail = _live_loop(
                    op.body(),
                    None,
                    domain,
                    grants,
                    prefix=frame_path(prefix, scope),
                    position=position,
                )
                tail.trace.extend(inner_tail.trace)
                tail.usage = tail.usage + inner_tail.usage
                if inner_tail.parked_at is not None:
                    tail.parked_at = inner_tail.parked_at
                    return tail
                send = inner_tail.result
            case _:
                # Every non-Step op — the await INCLUDED — goes through the one shared
                # inspect-only policy, so `live_drive` and `measured_drive` cannot disagree
                # about what a counterfactual may touch.
                with placing(op, position):
                    probe = _as_placed(op)
                    match inspect_only(probe, grants):
                        case Unanswered(name=name):
                            tail.parked_at = name.prefixed(prefix)
                            return tail
                        case Observed(value=value):
                            tail.trace.append(
                                TraceEntry(op_key(probe).prefixed(prefix), op, value)
                            )
                            send = value
                        case unreachable:
                            assert_never(unreachable)  # pragma: no cover - proven dead


def live_drive(
    gen: Any,
    send: Any,
    domain: MeteredDomain,
    grants: dict[Key, Any],
    *,
    prefix: str = "",
) -> ForkTail:
    """Interpret a live tail, under the same per-run ambient every other walk establishes.

    **The entry point exists to hold that ambient**, because the loop it wraps (`_live_loop`)
    RECURSES into a `scoped(...)` body — so establishing the scope there would hand each frame its
    own, which is the opposite of per-run. `RecordingHandler.run` and `DurableHandler.run` have the
    same split for the same reason.

    A fork RE-RUNS the base workflow with one substitution, so a `layer_run_state` cell must
    accumulate here exactly as on the recording, replay and durable walks. A fork whose ambient
    does not accumulate diverges from the run it is a counterfactual OF, and the difference is
    attributed to the substitution."""
    with walk_run():
        return _live_loop(gen, send, domain, grants, prefix=prefix)


# --- the MEASURED (dollars) fork -----------------------------------------------------
#
# A measured park is not a workflow-yielded op: the trip fires INSIDE the interpretation of an
# over-budget model-call Step, from the handler's folded meter. So the count fork above (which
# forks a yielded AwaitEvent) cannot express it. `measured_drive` is the dollars analog: it folds
# `run_metered` usage and enforces the measured trip, consulting a synchronous `grants` map keyed
# by the SAME production name `budget-grant:{run_id},{trip}` (the in-process grant-injection
# seam). The prefix is replayed FREE: recorded usage folds so the meter (and the trip) re-derive,
# but the domain is never called. Usage rides each recorded entry (`MeteredEntry`), the
# in-process analog of the usage-in-checkpoint envelope.
#
# The trip is ONE explicit transition (`enforce_measured`, `effective.budget`), and its PLACEMENT
# is a named guard in the driver: it fires only at a metered model call, the same op class as
# DurableHandler's `_step` v1 arm, never at an implicit "every Step". An implicit placement is
# where two interpreters drift apart while their loop arithmetic stays identical.


# The measured trip transition (`enforce_measured` + `Cleared`/`Parked`/`Exceeded`) lives in
# `effective.budget`, the shared domain module both interpreters import, so one transition has
# one definition that each driver is conformance-tested against. Re-exported here (the imports
# above); `test_measured_fork` pins the re-export.


_UNSET: Any = object()  # sentinel: "no raw state — this entry is already decoded"


@dataclass(frozen=True)
class MeteredEntry:
    """A recorded step for the measured fork — two provenances, ONE decode placement:

    - a BRIDGED entry (from a durable checkpoint, which stores no op) sets `state` to the RAW
      checkpoint value and leaves `op`/`result`/`usage` at defaults. The driver decodes `state`
      **at the op** during replay (`decode_checkpoint` → `metered_call`), so envelope-ness is
      decided by op class; a `{result, usage}` content sniff would disagree with the driver,
      which holds the op.
    - a DRIVEN entry (produced by `measured_drive`'s own trace) carries the live `op` and the
      already-decoded `result`/`usage`, and leaves `state` unset: a driver trace is re-feedable
      as a prefix without re-decoding.

    `measured_drive` keys by the *live* op (`key`) and never reads `op`, so a bridged entry
    leaves it `None`."""

    key: Key
    op: WorkflowOp | None = None
    result: Any = None
    usage: Usage = field(default_factory=Usage)
    state: Any = _UNSET


def decode_checkpoint(inner: DomainOp[Any], entry: MeteredEntry, domain: Any) -> tuple[Any, Usage]:
    """A prefix entry's `(result, usage)`, decoded **at the op**. A driven entry
    is already decoded; a bridged entry carries raw checkpoint `state` split by op class via
    the shared `metered_call` predicate: a metered model call's row is a `{result, usage}`
    envelope, every other op a bare value with zero usage. The driver, which holds the live op,
    owns the placement, and the bridge reads no shape: the loop decides where, and the
    transition decides what."""
    if entry.state is _UNSET:
        return entry.result, entry.usage
    if metered_call(inner, Contract.V1, domain):
        result, usage = _decode_usage_envelope(entry.state)
        return cancelled_or(result), usage
    return cancelled_or(entry.state), Usage()


@dataclass
class MeasuredTail:
    """The result of a measured drive. `usage` is the TOTAL folded meter (prefix + tail);
    `live_usage` is the tail-only spend (the domain-called steps) — the free-prefix payoff in
    dollars is `usage - live_usage`. `tripped_at` names the park if spend crossed the ceiling."""

    trace: list[MeteredEntry] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    live_usage: Usage = field(default_factory=Usage)
    result: Any = None
    tripped_at: Key | None = None


# --- A fork is INSPECT-ONLY -------------------------------------------------------------
#
# A real workflow does not yield Steps alone: `process_ticket` yields `call_tool`,
# `ask_llm`, AND `append_ledger`, so a fork must answer an `AppendLedgerRow` too.
# The RULE (observe the op, return what the workflow needs, write nothing) lives once in
# `effective.sandbox.inspect_only`, driven by every in-process driver here, so a counterfactual's
# safety cannot drift between them (the "one transition, many drivers" move that
# `enforce_measured`, `permission.decide` and `govern.combine` already make).
#
# These ops consume NO prefix position. That is a coupling, not a preference:
# `export_measured_prefix` filters non-Step checkpoints out of the durable prefix (by the one
# `checkpoints.NON_STEP` filter), so the prefix is Step-indexed and the driver must index it
# the same way — pinned by `test_prefix_is_step_indexed_like_the_bridge_exports_it`.


def _step_result(
    op: Step[Any], prefix: list[MeteredEntry], i: int, domain: MeteredDomain
) -> tuple[Any, Usage, bool]:
    """One Step's result: replayed FREE from the prefix, or run live. Returns
    `(result, usage, ran_live)` so the caller folds the live-only meter separately.

    The prefix decode is placed AT THE OP, never by the bridge, which cannot know the op
    class."""
    if i >= len(prefix):
        return (*domain.run_metered(op.op), True)
    entry = prefix[i]
    if entry.key != op_key(op):
        raise ReplayMismatch(f"measured prefix divergence at {i}: {op_key(op)!r}")
    return (*decode_checkpoint(op.op, entry, domain), False)


def trip_at(
    op: Step[Any],
    meter: Usage,
    budget: MeasuredBudget,
    granted: float,
    trips: int,
    grants: Mapping[Key, Any],
    domain: Any = None,
) -> tuple[float, int] | Parked:
    """The measured trip's placement, as a named rule: it fires only at a metered model call.

    Returns the advanced `(granted, trips)` when the run may proceed, a `Parked` when it must
    suspend, and raises `BudgetRefused` when there is no recourse. It asks `metered_call`, the
    predicate `decode_checkpoint` and `DurableHandler` ask, so the driver trips where it decodes
    and where the durable handler trips, including under a domain that does not meter."""
    if not metered_call(op.op, Contract.V1, domain):
        return granted, trips
    match enforce_measured(meter.cost, budget, granted, trips, grants):
        case Parked() as parked:
            return parked
        case Exceeded() as exceeded:
            raise BudgetRefused(op, exceeded)
        case Cleared(granted=advanced, trips=advanced_trips):
            return advanced, advanced_trips
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - proven dead


def measured_drive[T](
    program: Callable[[], Effect[T]],
    budget: MeasuredBudget,
    domain: MeteredDomain,
    grants: Mapping[Key, Any],
    *,
    recorded: list[MeteredEntry] | None = None,
) -> MeasuredTail:
    """Drive a measured (dollars) workflow. The first `len(recorded)` Steps replay FREE (recorded
    result + usage; domain NOT called); the tail runs live. The measured trip fires **only before a
    metered model-call step** (the same placement as `DurableHandler`'s v1 arm, an explicit
    guard here) via the `enforce_measured` transition; a `CallTool`/other
    Step runs unchecked.

    `grants` carries TWO key families, because a fork answers two different kinds of question:
    `budget-grant:{run_id},{trip}` -> a `Grant` (the measured trip's recourse), and a workflow
    event name -> that event's payload (the `await_event` the workflow itself yielded). They share
    one map because they share one property — an answer the fork was *given* rather than one it
    invented — and they cannot collide (`budget-grant:` is reserved). An absent key of either
    family parks the fork rather than guessing.

    Every non-`Step` op is INSPECT-ONLY (see `_inspect_only`): a fork observes ledger rows and
    artifacts without committing them, because a counterfactual that writes is a second run. It
    REFUSES a `SleepUntil` (`ForkedSleep`): a fork explores a decision, and a clock is not one."""
    run = _MeasuredRun(prefix=recorded or [], budget=budget, domain=domain, grants=grants)
    # The per-run ambient, established HERE rather than in `drive`, which recurses per scope —
    # see `live_drive` for the finding and why a fork needs it more than it looks.
    with walk_run():
        state = run.drive(program(), None, _MeasuredState(), prefix_path="")
    return MeasuredTail(state.trace, state.meter, state.live, state.result, state.tripped_at)


@dataclass
class _MeasuredState:
    """The measured drive's loop state, made a value so a `scoped(...)` body can carry it in and
    hand it back. Every field continues ACROSS a scope — a scope is sequential, so unlike a gather
    barrier there is nothing to fold and nothing that could double-count."""

    trace: list[MeteredEntry] = field(default_factory=list)
    meter: Usage = field(default_factory=Usage)
    live: Usage = field(default_factory=Usage)
    granted: float = 0.0
    trips: int = 0
    steps: int = 0  # prefix index — STEPS ONLY (see `inspect_only`)
    result: Any = None
    tripped_at: Key | None = None


@dataclass(frozen=True)
class _MeasuredRun:
    """The fixed configuration of a measured drive, so `drive` can recurse into a scoped body
    without re-threading four parameters that never change."""

    prefix: list[MeteredEntry]
    budget: MeasuredBudget
    domain: MeteredDomain
    grants: Mapping[Key, Any]

    def drive(
        self,
        gen: Any,
        send: Any,
        state: _MeasuredState,
        *,
        prefix_path: str,
        position: FramePosition | None = None,
    ):
        # The walk's ordinals. `drive` recurses for a `Scoped` and hands the body its own
        # `position`, so a scope entered twice counts on, as every other walk does.
        position = FramePosition() if position is None else position
        while True:
            try:
                op = gen.send(send)
            except StopIteration as done:
                state.result = done.value
                return state
            match op:
                case Scoped(scope=scope):
                    state = self.drive(
                        op.body(),
                        None,
                        state,
                        prefix_path=frame_path(prefix_path, scope),
                        position=position,
                    )
                    if state.tripped_at is not None:
                        return state
                    send = state.result
                    continue
                case Step():
                    outcome = trip_at(
                        op,
                        state.meter,
                        self.budget,
                        state.granted,
                        state.trips,
                        self.grants,
                        self.domain,
                    )
                    if isinstance(outcome, Parked):
                        state.tripped_at = outcome.name
                        return state
                    state.granted, state.trips = outcome
                    result, usage, ran_live = _step_result(
                        op, self.prefix, state.steps, self.domain
                    )
                    state.steps += 1
                    if ran_live:
                        state.live = state.live + usage
                    state.meter = state.meter + usage
                    state.trace.append(
                        MeteredEntry(op_key(op).prefixed(prefix_path), op, result, usage)
                    )
                    send = result
                case _:
                    with placing(op, position):
                        probe = _as_placed(op)
                        match inspect_only(probe, self.grants):
                            case Unanswered(name=name):
                                state.tripped_at = name.prefixed(prefix_path)
                                return state
                            case Observed(value=value):
                                state.trace.append(
                                    MeteredEntry(
                                        op_key(probe).prefixed(prefix_path), op, value, Usage()
                                    )
                                )
                                send = value
                            case unreachable:
                                assert_never(unreachable)  # pragma: no cover - proven dead


# --- the DURABLE spawned fork ---------------------------------------------------------------
#
# The durable sibling of `measured_drive`: instead of driving the tail in one process, it RE-RUNS
# the base workflow as a durable child task, seeding the recorded prefix through `ctx.step` so the
# tail is durable by construction (a crash mid-tail replays the seeded prefix free). It composes
# the pieces built above into one wired `DurableHandler`:
#
# | piece             | role                            |
# |-------------------|---------------------------------|
# | `SeedingCtx`      | replays the recorded prefix     |
# | `RenamedAwaitCtx` | the child's event world         |
# | `ForkLedger`      | the hypothetical lineage        |
# | `DryRun`          | the counterfactual-safe domain  |


def fork_seed(checkpoints: Sequence[Checkpoint], *, through: str) -> dict[Key, Any]:
    """The fork's PREFIX seed map — `{op_key: raw_value}` for every base checkpoint up to and
    INCLUDING `through` (the last prefix op's key, the fork boundary). Everything after `through`
    is the fork's tail and runs live.

    The boundary is load-bearing: a tail op MUST NOT be seeded, or the child would replay the
    base's decision instead of the counterfactual's (e.g. seeding `ledger:reviewed:*` would
    re-commit the base's *reject* even though the fork substitutes *approve*). `checkpoints` carry
    Python values (`read_sqlite_task` / the Absurd reader both decode), so a seeded value
    re-commits cleanly through `ctx.step` without double-encoding."""
    # `through` is a `str` and the comparison unwraps. Every caller composes it as an f-string
    # (`f"ledger:extracted:{message_id}"`), so typing it `Key` here would assert a guarantee no
    # writer is held to: a `Key` that was never composed is worse than a `str`. It converts
    # together with the other hand-rolled identity sites.
    seed: dict[Key, Any] = {}
    for c in checkpoints:
        seed[c.key] = c.state
        if c.key.stored() == through:
            return seed
    raise ValueError(f"fork boundary {through!r} not among the base's {len(checkpoints)} ckpts")


def run_fork(
    ctx: TaskContext,
    workflow: Callable[[], Effect[Any]],
    *,
    child_run_id: str,
    seed: Mapping[Key, Any],
    fork_point: Key,
    hypothetical_ledger: LedgerWriter,
    domain: Any,
    forked_from: str,
    forked_at_event: str,
    delta: dict[str, Any] | None = None,
    at_op_index: OpIndex | None = None,
    contract: Contract = Contract.V0,
    op_layers: Sequence[Any] = (),
    budget: MeasuredBudget | None = None,
    allow: frozenset[str] = frozenset(),
    canned: Mapping[str, Any] | None = None,
    transplanted: frozenset[Key] = frozenset(),
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Run one durable fork child: re-run `workflow` as a counterfactual over a seeded prefix.

    `fork_point` is the ONE named boundary: the await name where replay stops and the delta goes in
    (the *seek*, the checkpoint bookkeeper's handle; `forked_at_event` is the *ledger address*,
    the provenance handle). `SeedingCtx` crosses `Seeding → Live` exactly there, so a mid-prefix
    handler-internal park (`budget-grant:`) passes through and does not seal. It is the SAME
    event the driver emits the delta to: after this parks at the child's renamed
    `fork:{child_run_id}:{fork_point}`, the driver emits there.

    **`fork_point` is a POST-LAYER name**: `SeedingCtx` sees the await name as
    the *ctx* does, after every `op_layers` entry has had its say. An ordinary await-namespacing
    layer therefore makes the completion phase-match fire and blame the wrong thing ("it never
    fired") even though the fork point did fire — and leaves the `Live` seal inert for the whole
    run. Pass the name the layer stack produces, not the one the workflow authors.

    Composition: `SeedingCtx` replays the recorded prefix in the child's OWN event namespace
    (`RenamedAwaitCtx`); `ForkLedger` fences the child's lineage off as `hypothetical` with
    lineage-scoped ids, and its first row is the `Forked` genesis; `DryRun` keeps the tail from
    touching the world. A `SleepUntil` in the TAIL raises `ForkedSleep`
    (`SeedingCtx.sleep_until`; the prefix's already-elapsed sleep passes through).

    **A `Gather` in a forked tail RUNS.** Held by
    `test_fork_durable.py`: the tail fans out hypothetically and seals; a crash at each distinct
    branch op replays to the identical outcome with each tool called EXACTLY once; a nested gather
    composes under the outer branch's prefix; and a `DryRun` violation inside a branch surfaces
    wrapped in an `ExceptionGroup`, which is why refusals are classified with `except*`, matching
    the bare form a fork without a gather raises. A fork point INSIDE a gather region remains
    refused (`ForkPointInGather`): a gather is one node in the fork's order. (`fork_at`, the
    in-process driver, refuses a `Gather` at the fork point — the two drivers genuinely differ.)

    `contract`/`op_layers`/`budget` MUST match the base run's: the raw seed carries a v1 base's
    `{result, usage}` envelope, which only the `Contract.V1` arm folds, so a v1 base forked under
    v0 leaks the envelope into the workflow's values. On completion the whole prefix seed must be
    consumed, or the child diverged from the base: `run_fork` asserts `SeedingCtx.unconsumed()`
    empty.

    Idempotent under replay: the genesis append is idempotent by `event_id`, and the seeded prefix
    re-commits to the child's OWN checkpoints, so a crash mid-tail replays the prefix, not the
    base.

    **Prefix-await forks are unsupported, and REFUSED**: an await in the replayed prefix that is
    not `fork_point` raises `ForkedPrefixAwait`, so the child never parks forever in its own event
    namespace. That covers the handler-internal case too: a mid-prefix `budget-grant:` park is
    re-yielded by the replay itself, so a base that parked for a grant once would fork into a
    child that waits for it always. `transplanted`
    names awaits whose answers the caller HAS delivered under the child's names; it is the opt-out
    and the seam the unbuilt transplant will arrive behind."""
    # A fork point inside a gather region is refused. Attempted, it would survive one pass and
    # then break: a gather branch's await resolves by `peek_event`, so the phase crosses on the
    # first run and never on a replay, and a durable fork is replay. The resume would die with a
    # `SeedBoundaryError` blaming `through`. A gather region is atomic for cutting: fork before
    # it, or after it.
    # `.stored()` — a named exit, and the reason is that this asks a question about the key's
    # TEXT (does it open with the gather namespace?) that the type offers no primitive for.
    # `Key` withholds `startswith` on purpose: prefix tests are the string surgery the opaque
    # type refuses, and going through the named form keeps the one legitimate case greppable.
    point = fork_point.stored()
    if point.startswith((GATHER_PREFIX, RACE_PREFIX)):
        region = "gather" if point.startswith(GATHER_PREFIX) else "race"
        raise ForkPointInGather(
            f"fork_point {point!r} names an await INSIDE a {region} region. The fork point "
            f"is a boundary on `await_event`, but a gather branch's await resolves by "
            f"`peek_event` on every replay, so the phase would cross once and never again. A "
            f"gather is ONE node in the fork's order: fork at an await before it, or after the "
            f"whole region."
        )
    # Coerce ONCE, here, where the value enters the substrate (it arrives as a task param, i.e.
    # JSON): `Segment` refuses a `:`-bearing lineage id, and everything downstream is typed to
    # require the guarantee rather than re-checking it.
    child = Segment(child_run_id)
    # `_adapt_ctx` at the INNERMOST position, next to the foreign object — not left to
    # `DurableHandler`, which cannot reach it from outside our own wrappers. The handler adapts
    # the ctx it is HANDED; here it is handed
    # a `SeedingCtx`, which is not an SDK ctx, so the `isinstance` check passes it through and
    # `SdkCtx` is never installed. The stack then delegates our one-arg `sleep_until(when)` and
    # our `Key` step names straight into the SDK, which wants `sleep_until(step_name, wake_at)`
    # and does f-string work on the name. Measured on the deployed engine: an already-elapsed
    # PREFIX sleep died with `TypeError: ... missing 1 required positional argument: 'wake_at'`
    # and retried to death (`tests/test_fork_durable_absurd.py`). A wrapper stack can only be
    # adapted where it is BUILT.
    fork_ctx = SeedingCtx(
        RenamedAwaitCtx(_adapt_ctx(ctx), child),
        seed,
        fork_point=fork_point,
        transplanted=transplanted,
    )
    fork_ledger = ForkLedger(hypothetical_ledger, child_run_id=child)
    fork_ledger.append(
        genesis_row(
            child,
            forked_from=forked_from,
            forked_at_event=forked_at_event,
            delta=delta,
            at_op_index=at_op_index,
        )
    )
    dry = DryRun(domain, allow=allow, canned=dict(canned or {}))
    result = DurableHandler(
        fork_ctx,
        dry,
        ledger=fork_ledger,
        op_layers=op_layers,
        contract=contract,
        budget=budget,
        params=params,
    ).run(workflow)
    # On CLEAN COMPLETION the whole prefix seed must be consumed: a leftover
    # key means the child diverged from the base prefix (a stale/bogus key). Checked ONLY on the
    # return path, NOT in a finally: a crash mid-prefix (the durable retry path) legitimately
    # leaves
    # the seed partly consumed, and a finally would mask the real FaultInjected with a false
    # SeedBoundaryError. The park path is unchecked by design (the fork resumes and completes; a
    # park-forever is the documented prefix-await limitation).
    stale = fork_ctx.unconsumed()
    if stale:
        raise SeedBoundaryError(
            f"fork seed had unconsumed keys {sorted(k.display() for k in stale)}"
        )
    # A completed decision fork MUST have crossed the fork point — else `fork_point` never fired (a
    # wrong name), the phase stayed `Seeding`, and the `Live` guard was inert, so a tail seed
    # slips through silently. This validates the last boundary input, so all
    # three cross-check: `seed` vs `unconsumed()`, `through` vs the `Live` guard, `fork_point` now.
    match fork_ctx.phase:
        case Live():
            # Every boundary check has passed, so ATTEST it in the canonical bookkeeper.
            # Positive-on-the-clean-path is the only crash-safe polarity: a worker that dies
            # mid-fork cannot write a failure marker, so absence must mean "not known valid" —
            # covering refusal, crash, and still-running alike. Idempotent by `event_id`, so a
            # crash-replay of the tail cannot double-seal.
            fork_ledger.append(sealed_row(child_run_id, forked_from=forked_from))
            return result
        case Seeding():
            raise SeedBoundaryError(
                f"fork completed without crossing fork_point {fork_point!r} — it never fired "
                f"(wrong fork_point, or the workflow does not await it)."
            )
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


# --- the SPAWNED fork: a counterfactual as its OWN durable task -------------------------------
#
# `run_fork` (above) runs a counterfactual INSIDE the task that calls it. This pair runs one as a
# task of its own: the cross-run edge of the run graph. Two calls, because they are
# two EDGES: `spawn_fork` creates the child, `join_fork` waits for its answer. Keeping them
# unfused is what makes PROMOTION a composition rather than a flag: spawn a fork, don't join it,
# and it is a detached lineage that can park on a human for a week without holding the parent's
# worker: the fork you promote to a full interaction.
#
# Nothing new is interpreted here. `spawn_fork` is a checkpointed `Step` (so a parent replay never
# re-spawns) around the SAME `spawn` tool the subagent path uses, and `join_fork` is an
# ordinary `AwaitEvent` (so it suspends, releasing the worker). That is why the pair needs no new
# op-kind and no engine capability beyond an event.


@dataclass(frozen=True)
class Ask:
    """Leave the fork point OPEN — no substitution; whoever is watching answers it.

    The other inhabitant of `Substitution`, and the promoted fork's whole mechanism: with a delta
    the child pre-delivers its own answer and runs straight to a marginal; with `Ask()` it parks at
    the fork point exactly as the base did and waits for a real one — a human at `just approve`, an
    operator, another workflow.

    **A marker rather than an omitted argument**, deliberately. Inferring "ask a human" from a
    MISSING delta means a caller who forgets the argument gets a task parked forever on a question
    nobody was told to answer: the silent forever-park `ForkedPrefixAwait` refuses in the
    prefix, arriving through the front door. Required parameter + explicit marker turns that
    mistake into a `TypeError` at the call site, and gives the promoted case a name a reader can
    grep for."""


type Substitution = dict[str, Any] | Ask
"""What a fork does at its fork point: substitute this answer, or leave it open (`Ask`).

Note `{}` and `Ask()` are now different things — an empty dict is a substitution whose answer
happens to be empty, and `Ask()` is no substitution at all. Before the marker they were the same
call, which is precisely the ambiguity the marker removes."""


class SpawnedFork(Spawned):
    """The handle `spawn_fork` returns: a spawned child, and the run id it forks under.

    The three fields sort by what they are: an engine-minted task id, the join address the handler
    named, and an author-supplied run id."""

    child_run_id: str


class ForkOutcome(BaseModel):
    """One spawned fork's answer, beside the run id the parent forked it under.

    A fork child answers on `ChildAnswer`, as every spawned child does, and never with silence:
    the fork family refuses loudly by design (a tail sleep, a prefix await, a seed that crossed
    the boundary, a world mutation under `DryRun`), and a refusal that only raised would leave the
    parent parked on a join forever. A crash propagates and the engine retries it; a child that
    fails for good is answered `Failed` by its worker. `join_fork` builds this from its handle and
    the answer, so a sweep keeps every child's answer, refusals and failures included."""

    child_run_id: str
    answer: Annotated[Returned | Refusal | Failed, Field(discriminator="kind")]


REFUSALS: tuple[type[Exception], ...] = (
    *CHILD_REFUSALS,
    SeedBoundaryError,
    ForkedSleep,
    ForkedPrefixAwait,
    ForkPointRefused,
    ForkPointInGather,
    WorldMutation,
)
"""The fork family's typed refusals, the ones a child answers with and completes on: every
spawned child's (`spawning.REFUSALS`), and the fork's own.

Named once so the child body and any sweep consumer classify identically. Each is a pure function
of the op and its ctx, so a retry re-derives it. A programming error (`ops.CompositionRefused`)
is not here: a retry would raise it again too, and it fails the child once, answered as `Failed`.
Everything else (a `FaultInjected`, a bug, an engine error) is a CRASH: it propagates, the engine
retries, and replay converges.

`govern.Refused` belongs by that criterion: a `rules`-tier verdict must not read the clock, random
or live state, and a `human`-tier denial rests on a delivered event, which is durable.

A measured trip's `BudgetRefused` is a `Refused` raised above `ctx.step` against the handler's
replay-derived meter, so it belongs too.

`cost.BudgetExceeded` is not here. `CostBudget` raises it inside the checkpoint thunk against a
plain in-process accumulator: a retry builds a fresh budget while committed steps replay without
re-accruing, so the refusal is not a function of the op and its ctx."""


def spawn_fork(
    child_task: str,
    *,
    child_run_id: str,
    base_task_id: object,
    through: str,
    fork_point: Key,
    forked_from: str,
    forked_at_event: str,
    delta: Substitution,
    queue: str = "default",
) -> Effect[SpawnedFork]:
    """Spawn one counterfactual as its own durable task. Returns the join handle; does NOT wait.

    The parent's trace is a single checkpointed `Step` (a parent crash replays the spawn from its
    checkpoint and never enqueues a second child) carrying the child's whole recipe as
    serializable params: which lineage to fork (`base_task_id`, `through`), where to diverge
    (`fork_point`), and what to substitute (`delta`). The child task body (`run_fork_as_task`)
    reads the base's checkpoints through the engine's own reader and calls `run_fork`.

    `child_run_id` doubles as the correlation: it is already delimiter-free by type (`Segment`)
    and already unique per sibling, so a second identity would be a second thing to keep
    in sync. It names the spawn's op key; the handler names the join's event from where that op is
    placed, and the spawn's checkpoint re-binds it on replay.

    The handler holds the spawn to this task's depth, refuses one past it with
    `Refused` before any child is enqueued, and gives the child one level less, so a
    counterfactual that spawns counterfactuals cannot sprawl.

    **Do not call this inside a `gather` branch** unless you also compose the branch prefix into
    the join name: the child emits the unqualified `done_event`, while a branch's await is
    rescoped to `gather:{g},{i};{name}` (the qualified-emitter contract,
    `api.qualified_event_name`).
    A cross-task fan-out does not need `gather`: the children are already concurrent, each in
    its own task, so the parent's fan-out is a loop."""
    child = Segment(child_run_id)  # refuses a `:`, so the spawn's op key stays injective
    # ONE decision, named, outside the dict literal below: both arms do work at a serialization
    # boundary.
    match delta:
        case Ask():
            wire_delta = None
        case dict():
            wire_delta = dict(delta)
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    params: dict[str, Any] = {
        "child_run_id": str(child),
        # Coerced here, where the value enters the substrate, as `run_fork` does for
        # `child_run_id`: a task id is a `UUID` on both engines and spawn params must be JSON, so
        # the canonical text crosses the wire and `run_fork_as_task` parses it back. The coercion
        # is deliberately not hidden inside `spawn`: `json.dumps` refusing a `UUID` is the loud
        # failure that keeps this an explicit exit.
        "base_task_id": str(base_task_id),
        "through": through,
        # `.stored()` for the same reason `base_task_id` is `str()`-ed above: spawn params are
        # JSON, so the WIRE form is text. `run_fork_as_task` parses it back with `Key.parse`.
        "fork_point": fork_point.stored(),
        "forked_from": forked_from,
        "forked_at_event": forked_at_event,
        # `None` on the wire is `Ask()` — the child parks instead of pre-delivering. A dict (even
        # an empty one) is a substitution. The distinction survives serialization, so a reader of a
        # parked child's params can see it is waiting on purpose.
        "delta": wire_delta,
    }
    spawned = yield from spawn_child(child_task, str(child), params, queue=queue)
    return SpawnedFork(
        task_id=spawned.task_id, done_event=spawned.done_event, child_run_id=str(child)
    )


def join_fork(handle: SpawnedFork) -> Effect[ForkOutcome]:
    """Wait for one spawned fork's answer — the join edge, and the only blocking half.

    An ordinary `AwaitEvent`, so the parent SUSPENDS and releases its worker while the child runs
    (deadlock-free on one queue). Omitting this call is not an error: an unjoined `spawn_fork` is a
    DETACHED counterfactual — the promotion path. The child still runs, still seals, still writes
    its hypothetical lineage; nobody is waiting on it."""
    reply = yield from join_child(handle, ChildAnswer)
    return ForkOutcome(child_run_id=handle.child_run_id, answer=reply.answer)


def marginal_sweep(
    child_task: str,
    *,
    base_task_id: object,
    through: str,
    fork_point: Key,
    forked_from: str,
    forked_at_event: str,
    deltas: Mapping[str, Substitution],
    queue: str = "default",
) -> Effect[list[ForkOutcome]]:
    """Fork one base N ways and collect the marginals: the parallel sweep, as a workflow
    combinator.

    `deltas` maps each child's run id to what it substitutes at the fork point, so the ids are the
    caller's to choose and to recognise afterwards; results come back in `deltas` order regardless
    of which child finished first. A child whose substitution is `Ask()` parks for a human instead
    of running to a marginal, so a sweep may mix probes and questions.

    **Two loops.** The children are already concurrent (each is its own durable task with its own
    lease, attempts and worker), so the parent's wall-clock is `max(child)`. Wrapping the joins in
    a `gather` would buy nothing and would break the qualified-emitter contract (a branch's await
    is rescoped `gather:{g},{i};`, the child's emit is not). Spawning first and joining second is
    also what makes the fan-out durable at full width: every child is enqueued before the parent
    blocks on any of them, so a crash between the two loops replays N checkpointed spawns and
    enqueues nothing twice.

    A refused child answers rather than hanging the sweep, so the result is N answers, some of them
    refusals, never a parent parked forever on a join."""
    handles: list[SpawnedFork] = []
    for child_run_id, delta in deltas.items():
        handle = yield from spawn_fork(
            child_task,
            child_run_id=child_run_id,
            base_task_id=base_task_id,
            through=through,
            fork_point=fork_point,
            forked_from=forked_from,
            forked_at_event=forked_at_event,
            delta=delta,
            queue=queue,
        )
        handles.append(handle)
    outcomes: list[ForkOutcome] = []
    for handle in handles:
        outcomes.append((yield from join_fork(handle)))
    return outcomes


def run_fork_as_task(
    params: Mapping[str, Any],
    ctx: TaskContext,
    *,
    workflow: Callable[[str], Effect[Any]],
    domain: Any,
    hypothetical_ledger: Callable[[str], LedgerWriter],
    read_base: Callable[[str], Sequence[Checkpoint]],
    contract: Contract = Contract.V0,
    op_layers: Sequence[Any] = (),
    budget: MeasuredBudget | None = None,
    allow: frozenset[str] = frozenset(),
    canned: Mapping[str, Any] | None = None,
    transplanted: frozenset[Key] = frozenset(),
) -> dict[str, Any]:
    """The spawned child's task body: seed from the base, `run_fork`, then ANSWER — always.

    Returns the answer as **plain JSON**, since the Absurd SDK serializes a task result itself and
    refuses a Pydantic model after the body has already emitted.

    Register it bound to the engine-specific and workflow-specific parts, exactly as
    `run_subagent_as_task` is registered::

        app.register_task("fork-child")(partial(
            run_fork_as_task,
            workflow=process_ticket,
            domain=domain,
            hypothetical_ledger=lambda rid: SqliteLedger(conn, rid, lock, hypothetical=True),
            read_base=lambda tid: read_sqlite_conn(conn, tid),   # see the lock note below
        ))

    `read_base` is the injection seam that keeps this body engine-agnostic: `read_sqlite_conn` on
    0<->1, `read_absurd_task` on 0<->N. Both return the same `Checkpoint` sequence, so the seed is
    computed the same way on either engine.

    **Note the asymmetry in that block, because copying it somewhere else can bite.** The ledger
    is handed `lock`; `read_base` is not, because `read_sqlite_conn` takes a bare connection and
    has nowhere to put one. That is safe *here*: `read_base` runs on the worker's own thread
    while seeding a fork child, never from a `gather` branch. Elsewhere an unlocked reader sharing
    the engine's connection with a concurrent writer raises `InterfaceError` and returns short
    rows, which is why `effective.dashboard._parked` reads under the lock. If you lift this
    pattern somewhere a second thread can reach it, wrap the read in `with lock:` or open a
    read-only connection.

    A refusal raised inside a `gather` branch of the forked tail arrives wrapped in an
    `ExceptionGroup`, and one raised outside a gather arrives bare. `spawning.stopped_at` reads the
    leaves of either, at any depth, so the classification does not depend on whether the
    counterfactual happened to fan out; and it is all or nothing, so an error that is not a refusal
    reaches the worker with every leaf beside it."""
    child_run_id = str(params["child_run_id"])
    forked_from = str(params["forked_from"])
    # `Key.parse` — params arrive as JSON, so this is the read side of the identity axis.
    fork_point = Key.parse(str(params["fork_point"]))
    seed = fork_seed(read_base(str(params["base_task_id"])), through=str(params["through"]))
    # DELIVER THE SUBSTITUTION, or don't — and that choice is the whole difference between a probe
    # and a promotion. A `delta` is an answer the caller already holds, so the child pre-delivers
    # it into its own event world (`fork_event_name` — the one composer both sides use) and runs
    # straight through its fork point to a marginal. `Ask()` (a `None` delta on the wire) means the
    # caller has no answer: the child parks at the fork point exactly as the base did and waits for
    # a real one. That is the promoted fork — a counterfactual that became a live, human-in-the-
    # loop session — and it needs no separate mechanism, just an unanswered await. Checkpointed, so
    # a retry re-delivers at most once; first-write-wins would make a second emit inert anyway.
    raw_delta = params.get("delta")
    delta = None if raw_delta is None else dict(raw_delta)
    if delta is not None:
        answer_at = fork_event_name(child_run_id, fork_point)
        ctx.step(
            compose_key(t"emit;{answer_at:domain=address}"),
            lambda: deliver(ctx, answer_at, delta) or {"emitted": True},
        )
    try:
        result = run_fork(
            ctx,
            # `forked_from`: the run id the author sees is the BASE's, so every child of one base
            # sees the value the base saw. The child's own id would make any workflow that uses
            # the argument unforkable, since its authored names would not match the seed it was
            # handed (`SeedBoundaryError`) or would park on a name the base never registered
            # (`ForkedPrefixAwait`). The child's identity travels by the paths that need it
            # (`RenamedAwaitCtx`, `ForkLedger`).
            lambda: workflow(forked_from),
            child_run_id=child_run_id,
            seed=seed,
            fork_point=fork_point,
            hypothetical_ledger=hypothetical_ledger(child_run_id),
            domain=domain,
            forked_from=forked_from,
            forked_at_event=str(params["forked_at_event"]),
            delta=delta,
            contract=contract,
            op_layers=op_layers,
            budget=budget,
            allow=allow,
            canned=canned,
            transplanted=transplanted,
            params=params,
        )
        answer: Returned | Refusal = Returned(value=result)
    except Exception as raised:
        # A refusal is a deterministic ANSWER: retrying re-derives it, so report and complete.
        # Anything else propagates unchanged, and the worker retries it or answers it.
        match stopped_at(raised, REFUSALS):
            case Refusal() as refusal:
                answer = refusal
            case Crashed(errors=errors):
                raise errors from None
            case None:
                raise
            case unreachable:
                assert_never(unreachable)
    return answer_with(ctx, params, answer)
