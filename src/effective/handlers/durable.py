"""DurableHandler — interpret the op stream onto Absurd's durable TaskContext.

Runs *inside* an Absurd task (``def task(params, ctx): ...``) and maps:

    Step             -> ctx.step(name, thunk)           checkpointed, JSON-stored
    AwaitEvent       -> ctx.await_event(name)           durable suspend/resume
    AppendLedgerRow  -> ctx.step("ledger:<id>", ...)    durable, idempotent by name
    StoreArtifact    -> ctx.step("artifact:<ct>:<digest>", ...)  content-addressed
    SleepUntil       -> ctx.sleep_until(when)

Every ``ctx.step`` checkpoint name is ``op_key(op)`` (``handlers/base``), the same producer the
recording/replay core uses, so at a name's **first occurrence** the durable key and the
replay-trace key for one op are the same string by construction. (A ``Step`` keeps its bare
author name; the other arms carry their reserved tag, and a Step name that lands in a reserved
namespace is rejected by ``op_key``.)

**Occurrences:** the walk counts a repeated name (``DurableHandler._place``) and hands the engine
``name#2``, which the engine checkpoints at exactly that name. The recording core is
**positional** (no suffix; a re-yielded name re-serves the first occurrence's canned value). So
the same-string guarantee holds for first occurrences, and at occurrence 2 and above the two
sides diverge by construction: a *named* divergence, which the bridges normalize through
``grammar.split_occurrence`` for ``measured_drive``'s positional keying.

Absurd checkpoints are JSON, so a step result round-trips through ``dict``.
Serde is managed by Pydantic: every value is dumped to JSON-able data with
``to_jsonable_python`` before the checkpoint (schema-agnostic — handles models,
``Decimal``, ``datetime``, ``list[...]``, dataclasses), and loaded back into the
op's schema with a ``TypeAdapter`` after, so the workflow sees a typed object on
a fresh run and on replay alike.

This module imports no ``absurd_sdk``. The handler depends only on the ``TaskContext``
protocol (``step``/``await_event``/``sleep_until``, plus optional capabilities such as
``peek_event`` discovered by ``getattr``) that the Absurd engine's ctx
(`effective.engines.absurd`) and the embedded SQLite engine both satisfy, so the
recording/replay core never pulls the SDK and the handler is engine-independent.

When a ``LedgerWriter`` is provided, ``AppendLedgerRow`` writes to the dedicated
append-only ledger table (idempotent by ``event_id``); the ``ctx.step`` wrapper
skips it on replay. Without one, the row is just checkpointed (still works for
tests). Absurd checkpoints and the ledger stay separate: neither is ever derived
from the other.
"""

import asyncio
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cache
from typing import Any, Never, Protocol, assert_never

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from effective.api import Effect
from effective.budget import (
    BUDGET_DEPTH_PARAM,
    Budget,
    Cleared,
    Exceeded,
    Grant,
    MeasuredBudget,
    Parked,
    enforce_measured,
    refuse_an_unmetered_budget_gate,
    refuse_two_drivers,
)
from effective.cancel import Cancelled, ReservedShape, dump_cancelled, load_cancelled
from effective.choice import (
    Answer,
    Choice,
    EndingLost,
    answer,
    endings_from_stored,
    stored_endings,
)
from effective.cost import Contract, MeteredDomain, Usage
from effective.counterfactual import ForkedDeadline, ForkedPrefixAwait, ForkedSleep
from effective.domain import (
    SPAWN_TOOL,
    AsksModel,
    CallTool,
    DomainOp,
    SpawnArgs,
    Spawned,
    SpawnResult,
)
from effective.engines.absurd import _adapt_ctx, sdk_signals
from effective.govern import BudgetRefused, all_refusals, delivered
from effective.handlers.admission import (
    Cursor,
    Halt,
    RaceState,
    digest,
    drive_racing,
    observed,
    observed_error,
)
from effective.handlers.base import (
    NO_RACE,
    BranchRaised,
    BranchStopped,
    Continued,
    EngineSignal,
    Finished,
    Racing,
    RefusalDiverged,
    Stop,
    Stopping,
    TaskContext,
    _walk_run,
    artifact_id,
    barrier_errors,
    branch_slot,
    content_digest,
    deadline_of,
    ending_of,
    gated_record,
    keyed_call,
    op_key,
    placed_await_name,
    placed_key,
    placing,
    race_errors,
    refusal_entry,
    refusal_record,
    refuse_a_settled_checkpoint_under,
    served_refusal,
    settled,
    transient_errors,
)
from effective.keys import (
    AuthorityTag,
    FramePosition,
    Index,
    Key,
    Run,
    Scope,
    Segment,
    Subject,
    compose_key,
    frame_path,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
    scope_prefix,
)
from effective.keys.grammar import TERM_SEPARATOR, split_occurrence
from effective.layers import (
    CHECKPOINTED_OPS,
    OpLayer,
    check_seam,
    current_op_name,
    current_placement,
    drive_through,
    layer_routing,
    placement_scope,
    run_scope,
)
from effective.ops import (
    ACCRUAL_PARAM,
    CARRY_PARAM,
    DONE_EVENT_PARAM,
    GENERATION_PARAM,
    SUBSTRATE_PARAMS,
    AppendLedgerRow,
    Arrived,
    AwaitEvent,
    Expired,
    Gather,
    LedgerRow,
    Race,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WaitOutcome,
    WorkflowOp,
    Writer,
    awaits_an_absolute_name,
    leaves,
    refuse_a_bounded_wait_in_a_branch,
    refuse_a_park_in_a_race_branch,
    refuse_absolute_await_in_branch,
    refuse_absolute_await_under_a_rename,
    refuse_respawn_in_branch,
    refuse_respawn_under_a_rename,
)
from effective.permission import Refused
from effective.steering import SteeringCtx
from effective.viewing import ViewingCtx

FORK = AuthorityTag("fork", scope=Scope.QUALIFIED)
"""A fork child's event world — a base's answer must never resolve a child's park."""


def fork_event_name(child_run_id: str, name: Key) -> Key:
    """The fully-qualified event name an EMITTER uses to answer a fork child's await — the
    importable anchor of the fork's qualified-emitter contract (`api.qualified_event_name` is
    same idea for gather branches).

    A child re-runs the base workflow in its OWN event world: `RenamedAwaitCtx` rescopes every
    await to `fork:{child_run_id};{name}`, so the base's recorded answer can never resolve the
    child's park and two sibling forks can never share one answer. The rescoping is handler-side
    and invisible at the await's call site, which means anyone *delivering* an answer — a driver
    emitting the delta, an operator answering a promoted fork by hand, `run_fork_as_task`
    pre-delivering its own substitution — has to compose the same name. Composing it in ONE
    place is what keeps the emitter and the awaiter from drifting apart.

    Returns a ``Key``: spending a composed key to satisfy a ``str`` is the laundering
    `ops.event_name` names. Take ``.stored()`` at the wire boundary where the emitting API wants
    text.

    **``name`` is a ``Key``, and that is `Scope.QUALIFIED`'s obligation.** This namespace
    identifies no occurrence of its own: it relocates one, so the occurrence question recurses to
    the name it wraps, and that recursion is only checkable if the wrapped name came out of
    ``compose_key``. A ``str`` would let a hand-rolled identity be laundered into a child's event
    world. Its ``hyp:`` twin (``counterfactual.fork_scoped``) takes a ``Key`` too, and the two are
    one grammar apart."""
    return compose_key(t"{FORK}:{Run(child_run_id)};{name:domain=address}")


def respawn_name(run_id: str, generation: int) -> Key:
    """The step key for the generation cut a `respawn` performs."""
    return compose_key(t"respawn:{Run(run_id)},{Index(generation)}")


def idempotency_key_for(writer: Writer) -> Key:
    """A step's idempotency key: the task that runs it, and where it was placed. A retry
    re-executes the same placement, so a receiver that remembers the key does nothing new."""
    return compose_key(t"idempotency:{Run(writer.task)};{writer.placement:domain=any}")


def spawn_done_name(writer: Writer) -> Key:
    """The event a spawned child answers its parent on: the spawn's own coordinates under a tag of
    their own, so an answer reaches only the task and placement that spawned it.

    The placement's occurrence is a coordinate here, where it is a ``#N`` suffix on the spawn: a
    child composes this name into its emit step's key, where a trailing ``#N`` would read as that
    step's own occurrence."""
    base, occurrence = split_occurrence(writer.placement.stored())
    placement = Key.parse(base)
    return compose_key(
        t"spawn-done:{Run(writer.task)},{Index(occurrence or 1)};{placement:domain=any}"
    )


def respawned_name(run_id: str, generation: int) -> Key:
    """The ledger id for the row that RECORDS the cut — a sibling namespace, not a variant.

    Named minters for both, so the two spellings live one line apart where a reader meets them
    together, and so a test can ask for either instead of restating its bytes."""
    return compose_key(t"respawned:{Run(run_id)},{Index(generation)}")


def epoch_atom(when: datetime) -> Segment:
    """A wake TIME as an atom: ``epoch-{seconds}``, algorithm-prefixed like a digest.

    **The prefix is doing real work, not decorating.** An atom is an integer, a uuid, a
    ``sha256-`` digest, or a name matching ``^[A-Za-z][A-Za-z0-9_.@-]*$`` — and a stringified
    float matches none of them, because a name must OPEN WITH A LETTER and an integer admits no
    ``.``. Measured, not assumed: ``1767225600.0``, ``1786196706.123456`` and ``-14182940.0`` are
    all refused bare. `epoch-` makes each a NAME, and makes the encoding legible from the bytes
    the way `sha256-` does for a digest.

    **`repr`, so the value is the shortest string that round-trips** — Python's float repr is
    ULP-aware, so ``float(repr(x)) == x`` exactly and two wake times that differ at all give two
    different atoms. Injectivity is what the caller needs, and it holds at the precision a
    `datetime` actually carries: one microsecond apart stays distinct out to year 2262 (a float's
    ULP near 1.8e9 is ~2.4e-7 s, comfortably finer than 1e-6).

    `isoformat()` is **not in the language at all**: a timestamp reads
    ``2026-01-01T00:00:00+00:00``, whose ``:`` are separators, so it would compose a checkpoint
    name no producer could mint."""
    return Segment(f"epoch-{when.timestamp()!r}")


def wake_race_on_event(gather_index: int, branch_index: int, event: Key) -> Key:
    """The `repark` step name for a wake race whose lowest raced branch waits on an EVENT.

    The condition is a `Key`, so it extends the sequence as its own term rather than being
    flattened into a coordinate — an await name may itself be structured (`approve;r1`), and a
    structured value spliced into a coordinate is the flattening this grammar exists to refuse.

    Carrying the condition is the point: `_join` re-arms only at top level, so a branch's NEXT
    await can race the same gather again, and a name that omitted the condition would find the
    first race's checkpoint stale."""
    return compose_key(
        t"gather:{Index(gather_index)};wake-race:{Index(branch_index)};{event:domain=any}"
    )


def wake_race_at_time(gather_index: int, branch_index: int, until: datetime) -> Key:
    """The `repark` step name for a wake race whose lowest raced branch waits on a SLEEP.

    A time is a VALUE, not a key, so it rides as a coordinate atom — which is what separates this
    variant from `wake_race_on_event`'s trailing term rather than a literal discriminator having
    to be invented for it."""
    return compose_key(
        t"gather:{Index(gather_index)};wake-race:{Index(branch_index)},{Subject(epoch_atom(until))}"
    )


_PREFIX_PROBE = Key.parse("probe")
"""Any well-formed key. `fork_event_prefix` composes a real qualified name around it and keeps the
frame term, so the probe's own bytes never reach a caller."""


def fork_event_prefix(child_run_id: str) -> str:
    """The frame FRAGMENT ``fork:{child_run_id};`` that `fork_event_name` prepends.

    Not an identity — it names no event. It is the left half of every name `fork_event_name`
    composes, which is what `RenamedAwaitCtx.event_rename` reports so a refusal can name the
    right half of a mismatch.

    Rendered from decided segments, which is the f-string's correct position (a processor's
    backend, once the structure is settled). Calling the identity composer with an EMPTY name,
    ``fork_event_name(child_run_id, '').stored()``, would use it as a string builder; its ``name``
    is a ``Key``, so that call is unwritable **in a `ty`-clean tree**. The guard is the type: at
    runtime `fork_event_name("r1", "")` still composes, because the terminal hole is exempt from
    ``Segment``'s non-empty rule.

    **Derived by CALLING `fork_event_name`**, so the prefix cannot drift from the composer's
    bytes: a hand-respelled `fork:c1:` beside a composer rendering `fork:c1;` would leave the
    rename correct and the refusal message wrong."""
    head, *_ = fork_event_name(child_run_id, _PREFIX_PROBE).terms()
    return f"{head.render()}{TERM_SEPARATOR}"


def _prior_accrual(params: Mapping[str, Any] | None) -> tuple[float, float, int]:
    """`(spent, granted, trips)` carried from earlier generations, or a zero start.

    The substrate's own carry (`ops.ACCRUAL_PARAM`), written by `_respawn` and read here. It is
    a plain triple rather than a model because it crosses as JSON and has exactly one reader."""
    if params is None:
        return (0.0, 0.0, 0)
    carried = params.get(ACCRUAL_PARAM)
    if not carried:
        return (0.0, 0.0, 0)
    spent, granted, trips = carried
    return (float(spent), float(granted), int(trips))


class _ChainContinues(BaseException):
    """A `Respawn` unwinding to the task boundary — control flow, not failure.

    A `BaseException` for the same reason `_GatherPark` is one: it must cross the workflow
    generator without an author's `except Exception` swallowing it. There is no other way to
    COMPLETE a task from inside the drive loop (`_run` exits only by `StopIteration`, an
    exception, or the ctx's suspend), which is why `Respawn` had to be an op at all."""

    def __init__(self, result: Any) -> None:
        self.result = result


@dataclass
class _Forward:
    """One op entering one level of the layer stack, and the names placed below it.

    A re-forward shares its first forward's `names` and reads them from `cursor`.

    | the forward is | when                            | so it                                 |
    |----------------|---------------------------------|---------------------------------------|
    | replaying      | `cursor` is before the end      | hands the name at `cursor`            |
    | recording      | `cursor` is at the end          | appends each name placed below it     |
    | diverged       | a key other than its next name  | takes no part, and its names stay as  |
    |                | was placed while it replayed    | they were, so a later re-forward      |
    |                |                                 | replays the first forward's alone     |
    """

    level: int
    op: WorkflowOp
    names: list[tuple[Key, int]]
    cursor: int
    diverged: bool = False

    @property
    def replaying(self) -> bool:
        return not self.diverged and self.cursor < len(self.names)

    @property
    def recording(self) -> bool:
        return not self.diverged and self.cursor == len(self.names)


@dataclass
class _Reforwards:
    """The forwards one walk op's layer stack made, so a re-forward is placed as its first
    forward was.

    A layer re-forwards by yielding an op object again to the level it went to before, within one
    forward of its own, as `retry` does. Counts only grow.

    When the walk places a key, it visits the open forwards:

    ```
    each open forward, innermost first
    ├── replaying, its next name this key ─── hands its occurrence; the visit ends
    │     └── each recording forward inside it records that occurrence
    ├── replaying, its next name another key ─ diverges; the visit goes on outward
    ├── recording, or diverged ─────────────── the visit goes on outward
    └── past the outermost ─────────────────── the key is counted afresh
          └── every recording forward records the new occurrence
    ```

    The names are the walk's. A layer that numbers its own op per invocation, as the human tier and
    `govern` number their awaits, places a new key on a re-forward; on Absurd the SDK numbers an
    await itself, once per ask."""

    handed: list[_Forward] = field(default_factory=list)
    open: list[_Forward] = field(default_factory=list)

    def entering(self, level: int, op: WorkflowOp) -> None:
        """`op` enters `level` of the stack, `level` being the layers below it.

        ```
        the op entering this level, within the yielding layer's current forward
        ├── an object this level saw before ── a re-forward, replaying that forward's names
        └── an object new to this level ────── a first forward, recording
        ```
        """
        while self.open and self.open[-1].level <= level:
            self.open.pop()
        forward = _Forward(level, op, [], cursor=0)
        for earlier in reversed(self.handed):
            if earlier.level > level:
                break  # the yielding layer's own forward began there
            if earlier.level == level and earlier.op is op:
                forward = _Forward(level, op, earlier.names, cursor=0)
                break
        self.handed.append(forward)
        self.open.append(forward)

    def replayed(self, placed: Key) -> int | None:
        """The occurrence a replaying forward hands `placed`, or `None` when no open forward has
        it next; the class docstring has the visit."""
        for depth in range(len(self.open) - 1, -1, -1):
            forward = self.open[depth]
            if not forward.replaying:
                continue
            key, occurrence = forward.names[forward.cursor]
            if key != placed:
                forward.diverged = True
                continue  # a forward around it may have placed this name first
            forward.cursor += 1
            for inner in self.open[depth + 1 :]:
                if inner.recording:
                    inner.names.append((placed, occurrence))
                    inner.cursor += 1
            return occurrence
        return None

    def counted(self, placed: Key, occurrence: int) -> None:
        """`placed` was counted afresh at `occurrence`."""
        for forward in self.open:
            if forward.recording:
                forward.names.append((placed, occurrence))
                forward.cursor += 1


class _GatherPark(BaseException):
    """Control signal: a gather branch wants to park.

    Raised at the op arm (a branch's unsatisfied await peek / undue sleep) and at a
    NESTED gather's barrier (with the path-composed relative name); caught by the
    branch handler's ``run``, which returns a ``BranchParked`` sentinel — so no
    exception ever crosses the ``TaskGroup`` and the round completes. A
    ``BaseException`` deliberately: a layer's or workflow's broad ``except
    Exception`` must never swallow a park mid-flight. ``event`` is the
    branch-RELATIVE name (the parent re-adds its ``gather:{g},{i};`` prefix when
    re-arming); ``until`` is a sleep's wake time and ``name`` its branch-relative identity,
    carried for the same reason and re-prefixed the same way — a park that travels as a value
    must carry everything the re-arm needs, and a sleep's name is minted in the branch's frame,
    not the parent's."""

    def __init__(
        self,
        *,
        event: Key | None = None,
        until: datetime | None = None,
        name: Key | None = None,
    ) -> None:
        self.event = event
        self.until = until
        self.name = name


@dataclass(frozen=True)
class BranchParked:
    """A parked branch's slot value at the round barrier (never leaves _run_gather)."""

    event: Key | None = None
    until: datetime | None = None
    name: Key | None = None


class GatherWakeRace(Exception):
    """Every parked branch's wake condition was already satisfied at the re-arm —
    there is nothing left to park on, and a parked branch cannot be re-run
    in-process (the occurrence counters would shift its committed checkpoint
    names). On a ctx bearing the optional ``repark`` capability (both production
    adapters) the race re-queues via the engine's *park* path instead — no
    attempt burned — and this exception is not raised on any known path. It
    remains the loud FALLBACK for a peek-capable ctx without ``repark``, and for
    the stale-checkpoint edge where ``repark`` returns without parking: the engine's
    ordinary retry then re-queues the task and replay resolves every branch from
    the durable record. On that fallback path only, the burn caveat applies — a
    ``max_attempts=1`` task racing here fails PERMANENTLY despite being healthy,
    so give such tasks a retry budget (the default 3 is fine)."""


def _supports_peek(ctx: Any) -> bool:
    """Whether the ctx (unwrapping name-decorating wrappers) offers the optional
    ``peek_event`` capability — without it the await-in-gather wall stays up,
    legibly (a raw SDK ctx not wrapped in ``ConcurrentAbsurdCtx``).

    Unwraps EVERY ctx wrapper (``_PrefixedCtx`` for a gather branch, and
    ``RenamedAwaitCtx`` + ``SeedingCtx`` for a fork child) before probing the real
    ctx — each defines or delegates ``peek_event``, so probing a wrapper directly
    would falsely report support over a base ctx that has none, and the legible
    ``NotImplementedError`` wall in ``_branch_await`` would be replaced by an
    ``AttributeError`` from inside a gather branch.

    ``SeedingCtx`` shows why: ``run_fork`` stacks it OUTERMOST
    (``SeedingCtx(RenamedAwaitCtx(ctx, cid), …)``), so a probe that stopped there would find
    ``RenamedAwaitCtx.peek_event`` through ``__getattr__`` (it is always defined on the class)
    and report support for a peek-less base ctx. **Extend this tuple whenever a ctx wrapper is
    added.** At five names the list is itself the hazard: the structural form is
    ``while (inner := getattr(ctx, "_ctx", None)) is not None``, which every wrapper here
    already satisfies."""
    while isinstance(ctx, (_PrefixedCtx, RenamedAwaitCtx, SeedingCtx, SteeringCtx, ViewingCtx)):
        ctx = ctx._ctx
    return callable(getattr(ctx, "peek_event", None))


def _supports_await_until(ctx: Any) -> bool:
    """Whether `ctx` can wait on an event and a deadline together.

    The same unwrap `_supports_peek` makes, and for the reason its docstring gives: a wrapper
    always defines the method, so probing the outermost ctx reports support the base ctx lacks.
    SQLite answers this today and the Absurd SDK ctx does not, so a caller that needs it asks
    here and refuses legibly rather than meeting an `AttributeError` from inside a branch.
    """
    while isinstance(ctx, (_PrefixedCtx, RenamedAwaitCtx, SeedingCtx, SteeringCtx, ViewingCtx)):
        ctx = ctx._ctx
    return callable(getattr(ctx, "await_until", None))


def _attempt_of(ctx: Any) -> int | None:
    """The attempt `ctx` runs as, or `None` when it cannot tell, which a viewer never can: it
    replays another attempt's tape."""
    node = ctx
    while node is not None:
        if isinstance(node, ViewingCtx):
            return None
        node = vars(node).get("_ctx") if hasattr(node, "__dict__") else None
    try:
        return ctx.attempt.number
    except AttributeError, LookupError, NotImplementedError:
        return None


def _diverged(slots: list[Any]) -> ExceptionGroup | None:
    """The group a race raises when a branch's gate decided differently than on an earlier
    attempt, which fails the task even in a loser."""
    errors = [
        slot.error
        for slot in slots
        if isinstance(slot, BranchRaised)
        and any(isinstance(leaf, RefusalDiverged) for leaf in leaves(slot.error))
    ]
    return ExceptionGroup("race gates diverged", errors) if errors else None


def _race_capable(ctx: Any) -> Any:
    """`ctx`, once the engine beneath its wrappers offers `peek_step` and `settle`, the race's
    checkpoint surface; a legible refusal otherwise. Each wrapper states its own answer to both
    (`_PrefixedCtx` frames them, a fork refuses them), so this probes only the engine."""
    base = ctx
    while isinstance(base, (_PrefixedCtx, RenamedAwaitCtx, SeedingCtx, SteeringCtx, ViewingCtx)):
        base = base._ctx
    if not all(callable(getattr(base, name, None)) for name in ("peek_step", "settle")):
        raise NotImplementedError(
            f"a race needs a ctx that can peek and settle a checkpoint, and "
            f"{type(base).__name__} cannot: run it on SQLite, or on Absurd through `SdkCtx` or "
            "`ConcurrentAbsurdCtx`"
        )
    return ctx


class _PrefixedCtx:
    """A ``TaskContext`` that namespaces a gather branch's durable keys under a prefix.

    Branch *i* of the *g*-th gather runs over ``_PrefixedCtx(ctx, "gather:{g},{i};")`` so its
    ``step``/``await_event`` names can never collide: not with a sibling branch, and (the
    ``{g}`` discriminator) not with a *different* gather elsewhere in the same workflow that
    happens to share branch count and inner step names. On replay each branch re-binds its
    own committed checkpoints by name, independent of order. The underlying ctx (and the
    determinism boundary) is unchanged; only the key is decorated.

    ``__getattr__`` delegates everything else (e.g. ``concurrent_safe``) to the wrapped ctx,
    so a *nested* gather still sees the backend's concurrency capability one level down.
    """

    def __init__(self, ctx: TaskContext, prefix: str) -> None:
        self._ctx = ctx
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        """The frames this ctx applies, read-only — for diagnostics that must NAME the frame.

        A refusal whose message cannot say what the branch rescoped the name to leaves the
        author looking at the half of the mismatch they can already see. Read through
        `__getattr__` by anything wrapping this ctx (a test's `FaultCtx`, `SeedingCtx`).

        **Composes the whole chain, not this wrapper's own segment.** Nested frames nest the
        wrapper (`_run_scoped` pushes one per scope), so returning `self._prefix` alone reported
        `'b:0;'` for a ctx that actually applies `'a:0;b:0;'`, and the two interpreters'
        messages would differ, since the recorder's `_prefix` accumulates."""
        return f"{getattr(self._ctx, 'prefix', '')}{self._prefix}"

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        # `prefixed`: the scope is applied to an IDENTITY through a named method, so the call
        # site is greppable.
        return self._ctx.step(name.prefixed(self._prefix), thunk)

    def await_event(self, name: Key, /) -> Any:
        # `prefixed`, the same call the `step` arm above makes: one frame prefix, one spelling,
        # on both sides.
        return self._ctx.await_event(name.prefixed(self._prefix))

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> Any:
        # The ADDRESS takes the frame, as `await_event` does. `decided` does not: it arrives
        # from `current_placement`, which is the fully placed name, frames already applied.
        inner: Any = self._ctx  # an optional capability, forwarded as `peek_event` is
        return inner.await_until(name.prefixed(self._prefix), deadline, decided)

    def peek_event(self, name: Key, /) -> tuple[bool, Any]:
        # peek is an OPTIONAL ctx capability (see TaskContext's docstring — not a
        # protocol member); the handler probes support via _supports_peek before
        # calling, so the dynamic access here never raises in practice.
        inner: Any = self._ctx
        return inner.peek_event(name.prefixed(self._prefix))

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        # The same composition its `step`/`await_event`/`peek_event` arms make: applying a frame
        # to a finished identity goes through the named exit, never an f-string. A sleep whose
        # name skipped this would record under the frame
        # on the recorder and outside it here, so the two interpreters would disagree.
        return self._ctx.sleep_until(when, name=name.prefixed(self._prefix))

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        inner: Any = self._ctx
        return inner.peek_step(name.prefixed(self._prefix))

    def settle(self, name: Key, value: Any, /) -> Any:
        inner: Any = self._ctx
        return inner.settle(name.prefixed(self._prefix), value)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class RenamedAwaitCtx:
    """A ``TaskContext`` that gives a fork child its own EVENT WORLD.

    A fork child re-runs the base workflow under a FRESH run id, and a fresh run id rescopes only
    the names the *substrate* mints (a gate's ``pass_n``, a grant's ``trip_n``). A name the
    *workflow* authors embeds no run id: a ticket workflow awaits ``review:{ticket_id}``. On
    Absurd events are queue-global and first-emit-wins, so without rescoping the child's await
    would absorb the BASE's answer (the substitution at ``at`` lost), the tail await would not
    re-park, and two sweep forks would collide on one answer. (SQLite parks instead: an engine
    divergence.) This wrapper rescopes every workflow-authored event name to
    ``fork:{child_run_id};{name}`` on ``await_event``/``peek_event``, so the child parks and
    resolves in its own namespace.

    Only EVENT names are rescoped. ``step`` (durable checkpoints, already child-task-scoped) and
    ``sleep_until`` pass through unchanged: renaming a step key would break the prefix seeding
    that makes the fork's tail durable. ``peek_event`` is re-implemented explicitly rather than
    left to ``__getattr__``, so ``_supports_peek`` (which unwraps this wrapper) probes the real
    ctx and the rename applies on the gather-branch read path too. Everything else
    (``concurrent_safe``, ``repark``, ``task_id``) delegates untouched.
    """

    def __init__(self, ctx: TaskContext, child_run_id: str) -> None:
        self._ctx = ctx
        # A `Segment` (delimiter-free by type) so `fork:{cid};{name}` is injective and
        # two sibling forks cannot park on one queue-global event, while `name` rides the terminal
        # position verbatim and the wire name stays readable: `fork:r-fork;review:m1`.
        self._child_run_id = Segment(child_run_id)

    @property
    def event_rename(self) -> str:
        """The rescoping this ctx applies to EVENT names, read-only — the sibling of
        `_PrefixedCtx.prefix`, and named apart from it because the axes differ: a prefix frames
        steps *and* events, this frames events only.

        It exists so a guard can SEE this frame. The absolute-await refusal probes `prefix`, which
        `__getattr__` reports as `''` here, so without this property a fork child spawning and
        joining a counterfactual of its own would pass every check and park forever.
        The property is what `ops.refuse_absolute_await_under_a_rename` reads, and it composes
        through the wrappers above it (`SeedingCtx`, `_PrefixedCtx`) by ordinary delegation.

        **Composes the whole chain, not this wrapper's own segment** — the same property
        `_PrefixedCtx.prefix` has, for the same reason. A fork inside a fork stacks two renames,
        so reporting only the innermost would name the wrong half of a mismatch. The nested stack
        is reachable by following the refusal's own advice."""
        # A frame FRAGMENT (`fork:c1;fork:c2;`), not an identity, so the f-string is the render
        # backend — and the fragment now comes from `fork_event_prefix` rather than from calling
        # the identity composer with an empty name.
        outer = getattr(self._ctx, "event_rename", "")
        return f"{outer}{fork_event_prefix(self._child_run_id)}"

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        return self._ctx.step(name, thunk)  # steps are child-task-scoped — do NOT rename

    def _scoped(self, name: Key) -> Key:
        return fork_event_name(self._child_run_id, name)

    def await_event(self, name: Key, /) -> Any:
        return self._ctx.await_event(self._scoped(name))

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> Any:
        # `_scoped`, as `await_event` above: a child re-running the base workflow parks in its
        # own event world, or it absorbs the base's answer.
        inner: Any = self._ctx
        return inner.await_until(self._scoped(name), deadline, decided)

    def peek_event(self, name: Key, /) -> tuple[bool, Any]:
        inner: Any = self._ctx
        return inner.peek_event(self._scoped(name))

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        return self._ctx.sleep_until(when, name=name)

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        inner: Any = self._ctx
        return inner.peek_step(name)  # checkpoints are child-task-scoped, like `step`

    def settle(self, name: Key, value: Any, /) -> Any:
        inner: Any = self._ctx
        return inner.settle(name, value)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class SeedBoundaryError(RuntimeError):
    """The fork's seed boundary is wrong.

    Raised when a seed key matches a TAIL op (a ``step`` in the ``Live`` phase, after the fork
    point), so the driver's ``fork_seed`` ``through`` reached past the fork point and a divergent
    row would be silently dropped; or when ``unconsumed()`` is non-empty at the boundary (a
    stale/bogus seed key, or a ``name#k`` occurrence the child never re-yielded). Both silently
    corrupt the counterfactual, so the fork refuses loudly."""


@dataclass(frozen=True)
class Seeding:
    """Fork phase: replaying the recorded prefix, where a step consults the seed."""


@dataclass(frozen=True)
class Live:
    """Fork phase: past the fork point — the tail runs live; a seed match here is a boundary
    error."""


type Phase = Seeding | Live


class SeedingCtx:
    """A ``TaskContext`` that seeds a fork child's PREFIX from the base's recorded RAW state.

    A durable fork re-runs the base workflow. For the prefix region (ops before the fork point) the
    child must NOT call the domain: it replays the base's already-recorded results. It seeds the
    **raw** checkpoint value, because the v1 metered arm folds usage from the raw
    ``{result, usage}`` envelope *after* ``ctx.step`` returns, so a decoded seed would replay green
    while the meter under-derives.

    Two decisions, expressed as data rather than booleans (cf. ``Inspection``/``TripOutcome``): a
    **phase** (``Seeding | Live``) and a per-step ``match``.

    - ``await_event(name)`` is the ONE named boundary. When ``name == fork_point`` the phase
      crosses ``Seeding → Live`` (the tail begins). Every OTHER await (a ``budget-grant:`` park, a
      permission gate, a tail await) is ``≠ fork_point``, so a mid-prefix budget park cannot seal
      the prefix: the boundary is the named fork point, never "the first await" inferred by
      prefix.
    - ``step(name, thunk)`` is a ``match`` over ``(phase, key ∈ seed)``, one arm per corner of the
      square, which makes the boundary bidirectional.

    | phase     | seeded | the step                                                        |
    |-----------|--------|-----------------------------------------------------------------|
    | `Seeding` | yes    | replays the base's value THROUGH the inner ctx; no domain call  |
    | `Live`    | yes    | raises: the seed reached PAST the fork point                    |
    | `Seeding` | no     | raises: a prefix op the seed does not cover would run live      |
    | `Live`    | no     | runs live: the tail                                             |

    Replaying THROUGH the inner ctx means the child's own checkpoint holds the value, so a crash
    mid-tail replays from the CHILD and never re-reads the base.

    OCCURRENCE-AWARE: the handler names each occurrence of a repeated op ``name#k``, and
    ``read_sqlite_task`` returns seed keys carrying the same suffixes, so a workflow that repeats
    a step name (any agent loop) seeds each occurrence with ITS OWN base value.
    ``unconsumed()`` is the complementary check the driver asserts. Only ``step`` is seeded;
    ``sleep_until`` (and ``peek_event`` etc. via ``__getattr__``) delegate UNCHANGED.

    **The prefix-await transplant is UNBUILT, and REFUSED.** A workflow whose prefix contains an
    ``await_event`` before the fork point is not forkable: the base's recorded answer is not
    re-emitted under the child's renamed name, so a pass-through would park the child forever,
    with no exception and no attempt burned. ``await_event`` raises ``ForkedPrefixAwait`` there
    instead. ``transplanted`` names the awaits a caller has delivered into the child's namespace:
    the opt-out, and the seam the transplant will land behind. The case covers more than
    workflow-authored awaits: a handler-internal ``budget-grant:`` park is re-yielded BY THE
    REPLAY itself (the seeded prefix re-folds usage, so the meter re-crosses the same ceiling).
    """

    def __init__(
        self,
        ctx: TaskContext,
        seed: Mapping[Key, Any],
        *,
        fork_point: Key,
        transplanted: frozenset[Key] = frozenset(),
    ) -> None:
        self._ctx = ctx
        self._seed = seed
        self._fork_point = fork_point
        self._transplanted = transplanted
        self.consumed: set[Key] = set()
        self._phase: Phase = Seeding()

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        match self._phase, name in self._seed:
            case Live(), True:  # a tail op is seeded — the seed reached past the fork point
                raise SeedBoundaryError(
                    f"seed key {name!r} matched a TAIL op (past the fork point "
                    f"{self._fork_point!r}) — `fork_seed`'s `through` reached into the tail."
                )
            case Seeding(), True:  # replay the base; the domain thunk is dead
                self.consumed.add(name)
                value = self._seed[name]  # the base's RAW checkpoint state, undecoded
                return self._ctx.step(name, lambda: value)
            case Seeding(), False:
                # A step in the PREFIX that the seed does not cover. By the phase's own definition
                # every op before the fork point is a recorded op, so this cannot legitimately
                # happen. Run LIVE, it would call the domain inside the free prefix (so the fork's
                # marginal bills prefix work) and re-append the base's rows to the hypothetical
                # lineage: a copy of history. The other three boundary checks cannot see it,
                # because `unconsumed()`-empty is NECESSARY, NOT SUFFICIENT: it cannot see an
                # INSERTION.
                raise SeedBoundaryError(
                    f"step {name!r} ran LIVE during the Seeding phase, before the fork point "
                    f"{self._fork_point!r}, where every op must come from the seed. One of three "
                    f"things is wrong: `fork_seed`'s `through` stopped SHORT of the fork point "
                    f"(so "
                    f"the gap between them is unseeded); the child diverged from the base prefix "
                    f"(a workflow edited between the two runs); or {self._fork_point!r} never "
                    f"fires, so the phase never crossed and TAIL ops are being judged as prefix. "
                    f"Seeded so far: {sorted(k.display() for k in self.consumed)}."
                )
            case _:  # Live and unseeded — the tail, running live as it should
                return self._ctx.step(name, thunk)

    def unconsumed(self) -> frozenset[Key]:
        """Seed keys the child never reached — a leftover means the child diverged from the base
        prefix (a stale/bogus key, or a `name#k` occurrence the child never re-yielded). `run_fork`
        asserts this is empty."""
        return frozenset(self._seed) - self.consumed

    @property
    def phase(self) -> Phase:
        """The seeding phase as a marker (`Seeding` before the fork point, `Live` after) — for the
        caller to `match`. A completed decision fork MUST be `Live`; a completed fork still
        `Seeding` means `fork_point` never fired (a wrong `fork_point`), so the `Live` guard was
        inert. `run_fork` matches this on completion, validating the last input."""
        return self._phase

    @property
    def transplanted(self) -> frozenset[Key]:
        """The awaits whose answers the caller HAS delivered under the CHILD's names, read-only.

        Exposed for the same reason `phase` is: the absolute-await guard in `DurableHandler`
        consults it, so every refusal in the fork family honors the opt-out."""
        return self._transplanted

    def await_event(self, name: Key, /) -> Any:
        """The ONE named boundary, and before it the one place a fork can deadlock silently.

        `name == fork_point` crosses `Seeding → Live`: the tail begins, and the park that follows
        is the designed one (the driver emits the delta there). Any OTHER await *in the `Seeding`
        phase* is refused (`ForkedPrefixAwait`). The named fork point settles the SEAL, and the
        refusal settles LIVENESS: a pass-through would park the child in its own event namespace,
        where the base's answer does not exist and no driver emits, so the task would wait forever
        without raising. The motivating case is a mid-prefix `budget-grant:` park, which the
        replay re-yields by itself; `transplanted` is how a caller who HAS delivered the answer
        says so.

        **Name-keyed with no occurrence coordinate, unlike `step` above, because that is what the
        ENGINE does.** A step carries an occurrence because a repeated step name gets a `name#k`
        checkpoint per occurrence. An event name is a FACT: on the SQLite engine, a workflow
        awaiting one name twice with a single `emit_event` completes, both awaits returning the
        same payload. So two awaits of one name are one question answered once, and there is no
        k-th occurrence to key on.

        The reachable edge this costs: a `fork_point` naming a
        repeated await crosses on the FIRST occurrence, so ops between the two are judged tail.
        A fork at "the second `review:m1`" is inexpressible. Pinned in
        `tests/test_fork_seeding.py`.

        A `Live` await is untouched: the tail may legitimately park and be answered."""
        # `Key` is a frozen dataclass, so `==` and `in` are structural over the composed text —
        # the same comparison this made as a `str`, now over the typed identity.
        if name == self._fork_point:
            self._phase = Live()  # crossed THE fork point — the tail begins
        else:
            # Over the PHASE rather than a condition: an `elif isinstance(self._phase, Seeding)`
            # would leave every other phase to an implicit else, so a third arm of `Phase` would
            # silently be treated as `Live` and skip the refusal. Naming `Live` makes that a `ty`
            # error.
            match self._phase:
                case Seeding() if name not in self._transplanted:
                    # The name the child would actually park on — `RenamedAwaitCtx`'s rescoped
                    # form when one is beneath us — so the message carries the string an operator
                    # would grep for.
                    scoped = (
                        self._ctx._scoped(name) if isinstance(self._ctx, RenamedAwaitCtx) else name
                    )
                    raise ForkedPrefixAwait(name, fork_point=self._fork_point, scoped_name=scoped)
                case Seeding() | Live():
                    pass
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        return self._ctx.await_event(name)

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> WaitOutcome[Any]:
        """`sleep_until`'s rule for the other clock: a deadline in the forked TAIL is refused.

        A bounded wait IS a clock wearing an event's name, so the phase split below is the same
        one, for the same reason. Without this arm the refusal would reach the two in-process
        drivers (`sandbox.inspect_only`) and miss the durable one."""
        match self._phase:
            case Live():
                raise ForkedDeadline(name, datetime.fromtimestamp(deadline, tz=UTC))
            case Seeding():
                inner: Any = self._ctx
                return inner.await_until(name, deadline, decided)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        """On the DURABLE path, a sleep in the forked TAIL is refused rather than slept.

        The phase is what distinguishes the two cases, which is why the refusal lives here and
        not in `DurableHandler` (which cannot tell a fork from a base run):

        - ``Live``: the counterfactual's own tail. Sleeping would park a *hypothetical* branch
          against the *real* wall clock: the true run is sedated for a day so the dream can reach
          its own morning, and an N-fork sweep suspends N real days. Refuse (`ForkedSleep`).
        - ``Seeding``: the replayed prefix. That sleep happened in reality, so its wake time is
          always already past and the delegate is a no-op. Pass through, exactly as `fork_at`
          replays a recorded prefix sleep structurally.
        """
        match self._phase:
            case Live():
                raise ForkedSleep(when)
            case Seeding():
                return self._ctx.sleep_until(when, name=name)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def peek_step(self, name: Key, /) -> Never:
        refuse_a_settled_checkpoint_under("SeedingCtx")

    def settle(self, name: Key, value: Any, /) -> Never:
        refuse_a_settled_checkpoint_under("SeedingCtx")

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class DomainInterpreter(Protocol):
    """Executes a DomainOp for real, returning a typed result (e.g. a pydantic model)."""

    def run(self, op: DomainOp[Any]) -> Any: ...


class LedgerWriter(Protocol):
    """Appends one event to the durable, append-only decision ledger.

    `writer` says WHO is appending — the task and the placed op (`effective.ops.Writer`). It is
    optional because two legitimate callers have no answer: a direct append that bypasses
    `ctx.step` (a fork's genesis and seal) is not placed at all, and a writer exercised outside a
    drive loop has no walk above it. A store treats `None` as *unknown* and allows, which is what
    keeps the crash-window re-append and the cross-generation idempotency feature working.

    A store that ignores it stays correct and stays BLIND: the placed-writer collision is only
    detectable by reading back who already holds the row. `ForkLedger` passes it straight through
    because it rescopes the id BEFORE the store, so one check at the store covers the canonical
    and hypothetical lineages alike."""

    def append(self, row: LedgerRow, *, writer: Writer | None = None) -> None: ...


def _schema_of(op: DomainOp[Any]) -> type:
    match op:
        case AsksModel():
            return op.response_schema
        case CallTool(result_schema=schema):
            return schema
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


@cache
def _adapter(schema: type) -> TypeAdapter:
    return TypeAdapter(schema)


def _dump(value: Any) -> Any:
    """Schema-agnostic: produce JSON-able data for the checkpoint store. Only a `Cancelled` is
    stored in the shape reserved for a cancel."""
    if isinstance(value, Cancelled):
        return dump_cancelled(value)
    dumped = to_jsonable_python(value)
    if load_cancelled(dumped) is not None:
        raise ReservedShape(type(value).__name__)
    return dumped


def _load(schema: type, raw: Any) -> Any:
    """Schema-typed: validate a checkpointed value back into the op's type."""
    if schema is object or raw is None:
        return raw
    return _adapter(schema).validate_python(raw)


def _load_step(schema: type, raw: Any) -> Any:
    """A step's recorded result: its schema's value, or the `Cancelled` an interpreter answered
    for a step it stopped. Only a step's result may be a cancel; an event payload, a grant and a
    spawn result are outside data and load against their schema alone."""
    if (cancelled := load_cancelled(raw)) is not None:
        return cancelled
    return _load(schema, raw)


def _encode_usage_envelope(result: Any, usage: Usage) -> dict[str, Any]:
    """The v1 AskLLM checkpoint value: `{result, usage}` on the SAME row.

    A separate `usage:{op_key}` row is broken by a crash between the two commits: the
    result replays without its thunk, and the usage lived only in the now-gone provider
    response. So usage rides the same commit as the result."""
    return {"result": _dump(result), "usage": _dump(usage)}


def _decode_usage_envelope(raw: Any) -> tuple[Any, Usage]:
    """Split a v1 checkpoint back into `(dumped_result, usage)`. The contract (not a
    content sniff) selects this path, so a workflow result that happens to be a
    `{result, usage}`-shaped dict is unambiguous — it rides nested under `result`."""
    return raw["result"], _load(Usage, raw["usage"])


def metered_call(inner: DomainOp[Any], contract: Contract, domain: object) -> bool:
    """Whether an op is metered: a v1 model call (`AskLLM` or `Judge`) whose domain reports
    usage (`cost.MeteredDomain`).

    The one placement predicate for the usage envelope and the measured trip.
    `DurableHandler._step` and the in-process `measured_drive` and `decode_checkpoint`
    (`effective.fork`) all ask it, so no interpreter re-derives placement from a checkpoint's
    shape."""
    match inner:
        case AsksModel():
            return meters(contract, domain)
        case CallTool():
            return False
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def meters(contract: Contract, domain: object) -> bool:
    """Whether a handler accrues spend: a V1 contract over a domain that reports usage. A
    handler that does not keeps a `Usage` that stays zero."""
    return contract is Contract.V1 and isinstance(domain, MeteredDomain)


def _check_domain_reachable(domain: DomainInterpreter, contract: Contract) -> None:
    """Reject a domain the handler provably cannot drive, at ASSEMBLY rather than mid-checkpoint.

    A `serve(...)` stack is **metered-only**: it implements `run_metered` and its `.run` raises.
    That is correct for the v1 measured path, but the handler's non-metered arm (`CallTool`, and a
    V0 `AskLLM`) calls `domain.run`. Under `Contract.V0` EVERY model call takes that arm, so the
    combination cannot work for any workflow — a fact known at construction, and therefore one to
    refuse at construction rather than discover after the first step has committed.

    Under V1 it depends on the ops: an all-`AskLLM` workflow is fine, one `CallTool` is not, and
    which it is cannot be known here. That case keeps the runtime fence — `_ServedDomain.run`'s
    named "metered-only" error — which is legible but late. The check is therefore deliberately
    partial, and says so: prove what is provable at assembly, fail legibly for the rest.

    NB this is a BELT. `ty` already rejects the call statically — a `MeteredDomain` has no `run`,
    so it is not a `DomainInterpreter` — so a ty-clean consumer cannot reach here at all. The
    check exists for consumers outside that gate, the same rationale as `serve()`'s runtime
    rejection of an `@op_layer` service."""
    if getattr(domain, "metered_only", False) and contract is Contract.V0:
        raise TypeError(
            f"{type(domain).__name__} is metered-only (it implements run_metered; its .run "
            f"raises), but this handler runs under contract={contract.value}, where every "
            f"AskLLM takes the non-metered arm. Spawn with CONTRACT_PARAM='v1', or put the "
            f"serve(...) stack inside the metered domain rather than around the handler. "
            f"See the serve() docstring."
        )


class DurableHandler:
    def __init__(
        self,
        ctx: TaskContext,
        domain: DomainInterpreter,
        ledger: LedgerWriter | None = None,
        op_layers: Sequence[OpLayer[Any]] = (),
        in_gather: bool = False,
        contract: Contract = Contract.V0,
        budget: MeasuredBudget | None = None,
        params: Mapping[str, Any] | None = None,
        stop: Stop = NO_RACE,
    ) -> None:
        self.ctx = _adapt_ctx(ctx)
        # What the engine raises to end the walk at an op, which no layer sees.
        self._signals = (EngineSignal, *sdk_signals())
        # The ctx this handler was CONSTRUCTED at — `self.ctx` before any `scoped(...)` pushed a
        # frame onto it. `_run_scoped` swaps `self.ctx` and restores it; this never moves.
        #
        # It exists for exactly one caller: the measured trip's park (`_enforce_measured`), whose
        # grant name is `budget-grant:{run_id},{trip}` — a question about the RUN, not about a
        # place in it. Awaiting that through a scoped ctx qualified it to
        # `rec:0;budget-grant:r1,0`, which no emitter composes, so a run that tripped inside a
        # scope could not be granted more budget by anything that exists. So
        # the grant name is scope-free.
        #
        # **The discriminator, because this is NOT "authority names skip scopes":** an
        # `approve:{…}:{op_key}` or `govern:{…}:{op_key}` park names an OP OCCURRENCE, and the
        # frame is what keeps one emission from approving a sibling branch (pinned by the
        # conformance suite's `gather:0:0:{approve}`). Those stay frame-qualified. A name carrying
        # no op key addresses the run and takes no frames. Ask which one a new authority park is
        # before choosing a ctx for it.
        #
        # A fork child's rename and a gather coordinate are INSIDE the constructed ctx, so they
        # survive here — which is required: `RenamedAwaitCtx` gives a child its own event world,
        # and bypassing that would let a base's grant resolve a child's park.
        self._root_ctx = self.ctx
        self._domain = domain
        self.ledger = ledger
        self.op_layers: tuple[OpLayer[Any], ...] = tuple(op_layers)
        check_seam("op", self.op_layers)  # a domain layer here would see the live pass only
        _check_domain_reachable(domain, contract)  # assembly-time, not mid-checkpoint
        refuse_two_drivers(budget, self.op_layers)
        refuse_an_unmetered_budget_gate(meters(contract, domain), self.op_layers)
        # The AskLLM checkpoint schema, fixed at spawn. V0 checkpoints bare results; V1
        # envelopes `{result, usage}` so the meter re-derives on replay. A child gather-branch
        # handler inherits the parent's contract.
        self._contract = contract
        # The handler-owned, replay-derived meter. Accrual sits ABOVE `ctx.step` (this
        # instance folds each v1 AskLLM's usage AFTER the checkpoint returns, so replay
        # re-derives it), and a gather branch's child handler keeps its OWN subtotal, folded
        # into the parent at the barrier in branch-index order: the confluent shape
        # (`Effective.Budget.operational_run_eq_executedB`). This is the enforcement
        # bookkeeper; `MeteredInterpreter.meter` is telemetry only.
        # Seeded from the chain's accrual when this task is generation *n>0*, so `overall` binds
        # ACROSS generations instead of re-arming inside each one. `params` is optional and its
        # absence means a zero start, which is right for a fresh run and for every non-chained
        # task, and WRONG for a chain whose task function forgot to pass it. That is what
        # `test_carrier_audit.py` measures, so the omission is caught rather than assumed.
        prior_spend = _prior_accrual(params)
        self._spawn_budget = Budget.from_spawn_params(params or {})
        self._meter = Usage(cost=prior_spend[0])
        # The measured (dollar) ceiling. Enforced ABOVE `ctx.step` against the replay-derived
        # meter, at SEQUENTIAL points only, so a branch handler carries none (two branches
        # tripping in one round would compute the same trip_n, so the same grant name).
        # `_grants` accumulates granted dollars, `_trips` counts the parks so far; both
        # re-derive on replay (the meter does), so the grant event name
        # `budget-grant:{run_id},{trip}` is deterministic.
        self._budget = None if in_gather else budget
        # Seeded from the chain's accrual when this task is generation *n>0* — so `overall`
        # binds ACROSS generations rather than re-arming inside each one. Absent (a fresh run,
        # or a non-chained task) means a zero start, which is what every pre-chain run had.
        self._grants: float = prior_spend[1]
        self._trips = int(prior_spend[2])
        # True for a child handler running a gather branch:
        # the branch's await/sleep arms PEEK instead of touching the engine's suspend
        # machinery (an unsatisfied engine await flips the run state as a SQL side effect
        # and forbids a second await per run), an unsatisfied park raises _GatherPark,
        # and `run` converts it to a BranchParked sentinel at the round barrier.
        self._in_gather = in_gather
        # When this handler, as a race loser or inside one, must stop at its next admission.
        self._stop = stop
        # This thread of control's positional ordinals (`effective.keys.FramePosition`). The
        # workflow yields gathers in a deterministic order, so the g-index is stable across a fresh
        # run and every crash-resume, and it discriminates one gather's checkpoint keys from
        # another's.
        self._position = FramePosition()
        # The `scoped(...)` path currently installed on `self.ctx`, RELATIVE to this handler's
        # root. The ctx already applies it to every key it mints; this exists because a gather
        # branch park travels as a VALUE (`_GatherPark` -> `BranchParked`) rather than through
        # the ctx, and `_join` re-arms it by prepending only `gather:{g},{i};`. Without this the
        # scope frames between the branch root and the await would be silently dropped from the
        # re-armed name, and the run would park on a name no emitter can produce.
        self._scope_path = ""
        # Per-execution occurrence counts for `_place`, the one minter of `#N`. Rebuilt per
        # attempt, so it re-derives from the workflow's deterministic yield order on replay.
        self._placements: dict[Key, int] = {}
        # What the walk op now running its layers has placed, so a re-forward is placed alike.
        self._reforwards: _Reforwards | None = None
        # Race admission (`effective.handlers.admission`). Every one of these is read only when
        # `self._stop` says a race encloses this handler.
        #
        # | field              | holds                                                        |
        # |--------------------|--------------------------------------------------------------|
        # | `_cursor`          | this thread of control's walk position in its race tree     |
        # | `_free`            | whether the walk op being driven was admitted with no choice |
        # |                    | naming this thread a loser                                  |
        # | `_halted`          | this thread was stopped, so every later op stops too        |
        # | `_fresh`           | no earlier execution reached this walk: attempt 1, and no    |
        # |                    | enclosing race's choice was on the store; `None` until the   |
        # |                    | first race asks                                              |
        # | `_gated`           | a layer refused the walk op being driven on an earlier       |
        # |                    | attempt                                                      |
        # | `_domain_refusals` | the domain's refusals while driving that op                  |
        # | `_inputs`          | witnesses of what this thread's workflow was handed, in      |
        # |                    | order, each `None` when it has none                          |
        # | `_children`        | child handlers whose cursors retire at their barrier         |
        self._cursor: Cursor | None = None
        self._free = True
        self._halted = False
        self._fresh: bool | None = None
        self._gated = False
        self._domain_refusals: list[BaseException] = []
        self._inputs: list[Any | None] = []
        self._children: list[DurableHandler] = []

    @property
    def domain(self) -> DomainInterpreter:
        """The domain this handler runs its steps against, fixed when the handler is built, so
        whether it meters is fixed too."""
        return self._domain

    @property
    def meter(self) -> Usage:
        """The replay-derived spend accrued by this handler, read-only for op layers and
        grantors. Includes gather branches' subtotals once folded at their
        barrier; on replay it re-derives from the recorded v1 envelopes, so a grantor at a
        park reads recorded state, not a live side-channel."""
        return self._meter

    def run[T](self, program: Callable[[], Effect[T]]) -> Any:
        """Walk ``program`` once over this handler's ctx: one run per claim.

        The handler's counts (gather ordinals, placements, awaits) belong to the run, so a second
        run over the same ctx starts them again and places its ops as the first did. The engine's
        own per-claim count still keeps the checkpoints apart."""
        # Per-attempt layer scope: an op-layer that needs cross-op state gets it from HERE, so
        # its lifetime is one attempt by construction, never a closure that can outlive its
        # task and authorize from process memory.
        #
        # A gather branch is NOT a task boundary (it is one run's structure), so only the
        # outermost handler resets. A branch inheriting "a chain encloses you" is correct.
        #
        # **This is the one place the two halves come apart**, which is why the branch path does
        # not use `walk_run()`: a branch enters a fresh layer SCOPE (its own dict, no
        # cross-branch interference) but must NOT re-enter the RUN (`enter_task_run` would reset
        # chain state mid-run). Per-RUN versus per-FRAME, stated once by this branch.
        if self._in_gather:
            with run_scope():
                try:
                    # `_drive`, not `_run`: a BRANCH value is not a task result. It flows back
                    # into the parent's live generator through `_join`, so it must stay a
                    # host-language object — the same rule `_run_scoped` follows, and for the
                    # same reason. Branch results are never checkpointed (the gather is pure
                    # structure; the join is reconstructed by replaying each branch), so nothing
                    # here is a serialization boundary and the `_dump` was pure loss:
                    # `gather([scoped(rec:0, body)])` handed the workflow a `dict` durably and
                    # the object in memory.
                    return self._drive(program)
                except _GatherPark as park:
                    # A branch park is a VALUE at the round barrier, not an exception
                    # through the TaskGroup — the generator is abandoned; resume is
                    # whole-task replay (the branch re-runs, committed ops re-bind).
                    return BranchParked(event=park.event, until=park.until, name=park.name)
        match self.run_task(program):
            case Finished(value=value):
                return value
            case Continued(result=result):
                return result
            case unreachable:
                assert_never(unreachable)

    def run_task[T](self, program: Callable[[], Effect[T]]) -> Finished | Continued:
        """``run`` for a task body that must tell a finished walk from a generation boundary: a
        child answers its parent only when its walk finished."""
        spend = (lambda: self._meter) if meters(self._contract, self.domain) else None
        with _walk_run(spend):
            return self._run(program)

    def _respawn(self, op: Respawn) -> Never:
        """The generation boundary: spawn *n+1*, record it, and END this task.

        Three steps, in this order because each depends on the last surviving a crash. The spawn
        is CHECKPOINTED under a deterministic key, so a crash between spawning and completing
        replays to the same spawn rather than a second child (the window `spawn_fork` documents,
        closed the same way). The ledger row goes to the CANONICAL record because a granted
        generation is a governance decision: deriving it from checkpoint rotation would derive
        the canonical bookkeeper from the disposable one. Then the task completes.

        The ledger id carries the generation. Generation stays out of the ledger keys of the
        AUTHOR's domain events, where the same message triaged in generation 0 and again in 3
        should be ONE row (cross-generation idempotency is the feature); this row's subject is
        the boundary itself, so its subject is exactly `(run, generation)`.

        **The chain's result is not on the spawner** (measured): the spawner completes with
        `next_generation`, and the chain's real value lands in the last task's result row, which
        no external handle holds. Follow a chain by its `workflow_run_id` in the ledger rather
        than by the task id you spawned; a chain-done event is the planned repair for the
        operator surface."""
        # `gen:` is `effective.improve`'s tag at another arity; one tag, one shape.
        # BEFORE the spawn — the whole point of the ordering. See the refusal's docstring for
        # the liveness half (a TaskGroup turns `_ChainContinues` into an uncatchable group).
        if self._in_gather:
            refuse_respawn_in_branch(getattr(self.ctx, "prefix", ""))
        # `fork ∘ respawn` is refused LOUD. Handler-side (the combinator's `ContextVar` catches
        # nesting) because being inside a fork child is CTX state, and durable-only for the same
        # reason the absolute-await rename guard is: a fork child exists only under `run_fork`,
        # which runs on this class. Read off `_root_ctx`, so it survives every
        # same-handler frame a `scoped`/`route`/`descend` pushes.
        if rename := getattr(self._root_ctx, "event_rename", ""):
            refuse_respawn_under_a_rename(rename, op.task)
        key = respawn_name(op.run_id, op.generation)
        params = self._spawn_budget.stamp(
            {
                **op.params,
                GENERATION_PARAM: op.generation,
                CARRY_PARAM: op.state,
                # The substrate's own carry, added HERE rather than by the combinator: an author's
                # carry is theirs to declare, the measured accrual is ours to keep.
                ACCRUAL_PARAM: [self._meter.cost, self._grants, self._trips],
            }
        )
        # The PARENT's queue: a chain working off a non-default queue continues there. Read
        # privately, and defaulted for engines that have no queues (SQLite).
        queue = getattr(self.ctx, "_queue_name", "default")

        def enqueue() -> Any:
            # Named inside the step, as a spawn is, so a replay served from the record never asks
            # the ctx for the task that spawned it.
            if (task := getattr(self.ctx, "task_id", None)) is None:
                raise Refused(
                    op, "a respawn is named by the task that spawns it, and this ctx has no task"
                )
            successor = SpawnArgs(
                task_name=op.task,
                params=params,
                queue=queue,
                idempotency_key=idempotency_key_for(
                    Writer(task=str(task), placement=key)
                ).stored(),
            )
            return _dump(self.domain.run(successor.call()))

        raw = self.ctx.step(key, enqueue)
        # A checkpoint round-trips as JSON, so the value comes back a dict — validate it here
        # the way `_step` does for every other step, or `.task_id` is an AttributeError that
        # fails the task AFTER it has already enqueued its successor.
        spawned = _load(SpawnResult, raw)
        self._record_ledger(
            LedgerRow(
                event_id=respawned_name(op.run_id, op.generation),
                kind="respawned",
                generation=op.generation,
                # The other two fields a `respawned` row carries. The GRANT because a granted
                # generation is a governance decision the canonical record exists to hold; the
                # DIGEST because what crossed the boundary is the only thing that survives it,
                # and a row saying a generation happened without saying what it carried cannot
                # be audited.
                granted=op.granted,
                carry_digest=content_digest(op.state),
            ),
            None,  # a respawn is never placed, so its row's writer is unknown
        )
        raise _ChainContinues({"next_generation": op.generation, "task_id": str(spawned.task_id)})

    def _run[T](self, program: Callable[[], Effect[T]]) -> Finished | Continued:
        """Drive a workflow and return its result in DURABLE form — the TASK boundary.

        `_dump` happens **exactly once per task, here**. Everything below this is in-process: a
        structural op's body (`_run_scoped`) and a gather branch (above) both hand their value to a
        live generator, so both use `_drive`. The rule is worth stating as one sentence because it
        was got wrong twice in one day, once per composition."""
        try:
            return Finished(_dump(self._drive(program)))
        except _ChainContinues as boundary:
            # Control flow, not failure — the generation boundary unwinding to the task edge.
            # The generator is abandoned exactly as a branch park abandons one; the next
            # generation replays nothing, which is the entire point of the cut.
            return Continued(boundary.result)

    def _drive[T](self, program: Callable[[], Effect[T]]) -> Any:
        """Drive a workflow to its return value, unserialized. A race branch's generator is
        closed when the drive ends, so a stopped loser's `finally` blocks run."""
        gen = program()
        if not self._stop.racing:
            return self._walk(gen)
        try:
            return self._walk(gen)
        finally:
            gen.close()

    def _walk(self, gen: Any) -> Any:
        """The drive loop over a started workflow generator."""
        send_value: Any = None
        throw: BaseException | None = None
        while True:
            try:
                op = gen.throw(throw) if throw is not None else gen.send(send_value)
            except StopIteration as done:
                return done.value
            throw = None
            if self._stop.racing:
                self._admit(op)
            try:
                if layer_routing(op) == "unlayered":
                    # A STRUCTURAL op orchestrates what is inside it; the op-layers apply *within*
                    # (a gather branch's child handler carries them, a scoped body is driven under
                    # the same stack), never around the structural op itself — it has no `op_key`
                    # and no result of its own to authorize.
                    #
                    # `layer_routing`, not `isinstance(op, UNLAYERED_OPS)`. Both keep the two
                    # handlers agreeing about which ops a layer sees, `Scoped` included, but a
                    # tuple test is a DENYLIST: everything not enumerated falls to the layered
                    # path silently, so a new arm of `WorkflowOp` would be routed through the
                    # stack without anyone deciding it should be. The function is total and asks
                    # the new arm a question instead.
                    send_value = self._handle(op)
                else:
                    # Every non-Gather op — including `AwaitEvent` — routes through the layer
                    # stack here (and re-fires per op on replay), unlike the recording core,
                    # which decides an AwaitEvent before its layers (no-call/cc). A NAMED
                    # divergence (`layers.LAYERED_OPS_DURABLE_ONLY`).
                    #
                    # A positional arm is NAMED HERE, once per op, before any layer runs — so an
                    # authority layer and `_handle` bind the same identity, and a layer that
                    # re-forwards its op (op-seam `retry`) does not advance the ordinal.
                    # `_place` is evaluated INSIDE `placing` — a positional arm has no key
                    # until the walk mints one, and `placed_key` reads it from there.
                    with placing(op, self._position), placement_scope(self._place(op)):
                        send_value = self._through_layers(op)
            except Exception as raised:
                # a refusal, bare or grouped out of a scoped body or a gather, re-derives at the
                # same point on replay
                throw = delivered(raised)

    def _admit(self, op: WorkflowOp) -> None:
        """The WALK CHECK inside a race branch, before any layer sees the op.

        | the op                                  | admitted when                               |
        |-----------------------------------------|---------------------------------------------|
        | an await or a sleep                     | never: a race branch may not park; one a    |
        |                                         | layer injects is refused where it lands     |
        | any other op                            | no stored choice names this thread a loser, |
        |                                         | or an earlier attempt admitted it before    |
        |                                         | its choice (its cursor is within horizon)   |

        A loser that is not admitted raises `Stopping`, and the generator is abandoned at the
        yield it never resumes from."""
        match op:
            case AwaitEvent() | SleepUntil():
                refuse_a_park_in_a_race_branch(op, getattr(self.ctx, "prefix", ""))
            case _ if self._halted:
                raise Stopping
            case _ if self._cursor is not None:
                try:
                    self._free = self._cursor.tick(leaf=isinstance(op, CHECKPOINTED_OPS))
                except Halt:
                    raise Stopping from None
            case _:
                return

    def _new_work(self) -> None:
        """A layer's resume yielded an op: new work, which a stopped loser starts none of."""
        if self._cursor is None:
            return
        try:
            self._cursor.tick(leaf=False)
        except Halt:
            self._halted = True
            raise Stopping from None

    def _through_layers(self, op: WorkflowOp) -> Any:
        """The walk op, once placed, driven through the op layers to `_handle`."""
        outer, self._reforwards = self._reforwards, _Reforwards()
        forwarding = self._reforwards.entering
        try:
            if self._stop.racing:
                return self._race_leaf(op, forwarding)
            return drive_through(
                self.op_layers, op, self._handle, escapes=self._signals, forwarding=forwarding
            )
        finally:
            self._reforwards = outer

    def _race_leaf(self, op: WorkflowOp, forwarding: Callable[[int, WorkflowOp], None]) -> Any:
        """A walk op inside a race branch, once `_place` has counted it, driven through the layers.

        A gate's refusal is recorded at the walk name (`gated_record`); unless this walk is fresh
        that record is read before the layers run, and a refused op that the layers now forward
        or answer is `RefusalDiverged`. What the op hands the workflow joins this thread's inputs,
        which the race's endings carry as a digest."""
        here = self._step_name(op, self._slot(op))
        gate = gated_record(here)
        ctx = _race_capable(self.ctx)
        refused_before = not self._fresh and ctx.peek_step(gate)[0]
        free, outer = self._free, (self._gated, self._domain_refusals)
        self._gated, self._domain_refusals = refused_before, []
        try:
            value = drive_racing(
                self.op_layers,
                op,
                self._handle,
                new_work=self._new_work,
                stopped=(Stopping, RefusalDiverged, *self._signals),
                forwarding=forwarding,
            )
            if refused_before:
                raise RefusalDiverged(
                    f"{here!r} was refused by a layer on an earlier attempt and answered on this "
                    "one: a race branch's gates must decide the same way on every attempt"
                )
            self._inputs.append(observed(["value", value]))
            return value
        except Stopping, RefusalDiverged:
            raise
        except Exception as raised:
            self._inputs.append(observed_error(raised))
            if all_refusals(raised) and self._gate_refused(raised):
                if refused_before:
                    raise
                if not free:
                    raise RefusalDiverged(
                        f"{here!r} was refused by a layer while its loser replayed: a race "
                        "branch's gates must decide the same way on every attempt"
                    ) from raised
                ctx.settle(gate, {"reason": str(next(leaves(raised)))})
            raise
        finally:
            self._gated, self._domain_refusals = outer

    def _gate_refused(self, raised: BaseException) -> bool:
        """Whether a layer, not the domain, made any of `raised`'s refusals."""
        return any(
            all(leaf is not domain for domain in self._domain_refusals) for leaf in leaves(raised)
        )

    def _checkpoint(self, op: WorkflowOp, slot: Key, thunk: Callable[[], Any]) -> Any:
        """`ctx.step` at the op's placed name, and in a race branch the BASE CHECK on a miss.

        | on a miss, in a race branch                  | the guard                               |
        |----------------------------------------------|-----------------------------------------|
        | a walk that is not fresh, and the domain's   | raises it, and the domain is not called |
        | recorded refusal at the step's name          |                                         |
        | a stored choice names this thread a loser    | stops: no new effect after the choice   |
        | otherwise                                    | calls the domain, recording a refusal   |

        A fresh walk reads no refusal record, since no earlier execution reached it. A resume
        after a park runs on the same attempt as the execution that parked, so attempt 1 alone
        does not make a walk fresh: the race's choice on the store says an earlier one ran."""
        name = self._step_name(op, slot)
        try:
            return self._checkpointed(op, name, thunk)
        except ReservedShape as reserved:  # the refusal names the step, as `served` does
            raise ReservedShape(name.stored()) from reserved

    def _checkpointed(self, op: WorkflowOp, name: Key, thunk: Callable[[], Any]) -> Any:
        if not self._stop.racing:
            return self.ctx.step(name, thunk)
        if self._gated:
            raise RefusalDiverged(
                f"{name!r} arrived from an op a layer refused on an earlier attempt: a race "
                "branch's gates must decide the same way on every attempt"
            )
        if self._halted:
            raise Stopping
        return _race_capable(self.ctx).step(name, self._guard(op, name, thunk))

    def _guard(self, op: WorkflowOp, name: Key, thunk: Callable[[], Any]) -> Callable[[], Any]:
        """The thunk `_checkpoint` hands the engine in a race branch, which runs only on a miss."""
        ctx = _race_capable(self.ctx)

        def guard() -> Any:
            recorded = False
            if not self._fresh:
                recorded, record = ctx.peek_step(refusal_record(name))
                if recorded and served_refusal(record):
                    refused = Refused(op, record["reason"])
                    self._domain_refusals.append(refused)
                    raise refused
            if self._stop.now():
                self._halted = True
                raise Stopping
            try:
                return thunk()
            except Refused as refused:
                self._domain_refusals.append(refused)
                served = type(refused) is Refused
                if not recorded:
                    entry = refusal_entry(refused.reason, "serve" if served else "call")
                    ctx.settle(refusal_record(name), entry)
                if not served:
                    raise
                normalized = Refused(op, refused.reason)
                self._domain_refusals.append(normalized)
                raise normalized from None

        return guard

    def _place(self, op: WorkflowOp) -> Key:
        """The op's full address: the ctx's frames, its placed key, and an occurrence.

        A step's occurrence is minted here and nowhere else. `_step_name` hands the engine this
        placement in the ctx's coordinates, and the engine checkpoints at exactly that name, so a
        span, a `Writer` and the checkpoint carry one address. The Absurd SDK still numbers an
        await's or a sleep's name itself; the walk keeps those unique by position and by
        `placing`. A layer that short-circuits an op
        has still had it placed, so the next ask of that name is the next occurrence; a layer
        that re-forwards an op forwards its placement, so a retried step lands where its first
        try would have.

        Deterministic under replay: the handler is rebuilt per attempt and the workflow
        re-executes its ops in yield order, so the n-th execution of a name is the n-th on every
        attempt. A gather branch counts in its own child handler under its gather's frame, whose
        ordinal the handler never hands out twice, so a gather inside a scope entered twice places
        its branches apart. A handler run is one claim's walk: a second run over one ctx restarts
        this counter, places its ops again, and is served the first run's checkpoints.

        Occurrence is byte-preserving at n <= 1 (`Key.occurrence`), so a placement that happens
        once reads exactly as the key it places.

        **It counts only where nothing else did**: `Key.occurrence` refuses a second suffix, and a
        `Scope.SETTLEMENT` await already carries the one `placing` gave it. That suffix counts
        asks of the name, which is the same coordinate under a different producer."""
        placed = placed_key(op).prefixed(getattr(self.ctx, "prefix", ""))
        if split_occurrence(placed.stored())[1] is not None:
            return placed  # the walk already distinguished this one
        reforwards = self._reforwards
        if reforwards is not None and (replayed := reforwards.replayed(placed)) is not None:
            return placed.occurrence(replayed)
        count = self._placements.get(placed, 0) + 1
        self._placements[placed] = count
        if reforwards is not None:
            reforwards.counted(placed, count)
        return placed.occurrence(count)

    def _handle(self, op: WorkflowOp) -> Any:
        match op:
            case Step(op=inner):
                return self._step(op, inner)
            case AwaitEvent():
                return self._await(op)
            case AppendLedgerRow(row=row):
                slot = self._slot(op)
                return self._checkpoint(
                    op, slot, lambda row=row: self._record_ledger(row, self._writer(slot))
                )
            case StoreArtifact(value=value):
                # Content-addressed: the checkpoint (the durable store, absent a CAS)
                # persists the value under the injective key
                # `op_key(op)` = `artifact:{kind}/{subtype},{digest}`, so two distinct artifacts
                # never share a checkpoint (or, when gated, an approval event). Resolve
                # to the content id; the step runs the persist exactly once and replay
                # returns the recorded value. Digest computed once (artifacts can be
                # large) — the checkpoint key is `op_key(op)`, the id its tail.
                aid = artifact_id(op)
                self._checkpoint(op, self._slot(op), lambda v=value: _dump(v))
                return aid
            case SleepUntil(when=when):
                if self._stop.racing:  # a sleep a layer injected, which admission never saw
                    refuse_a_park_in_a_race_branch(op, getattr(self.ctx, "prefix", ""))
                # `current_op_name()` reads the name `_drive` already placed rather than minting
                # one: `_handle` is the BASE of the layer stack, so it runs once per layer yield
                # and a mint here would advance the ordinal per layer instead of per op.
                return (
                    self._branch_sleep(when, self._sleep_name(current_op_name()))
                    if self._in_gather
                    else self.ctx.sleep_until(when, name=self._sleep_name(current_op_name()))
                )
            case Scoped() | Gather() | Race():
                return self._structural(op)
            case Respawn():
                return self._respawn(op)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _structural(self, op: Scoped[Any] | Gather | Race) -> Any:
        """A structural op: the ops inside it are placed and layered one by one, and it has no
        checkpoint of its own beyond a race's choice and endings."""
        if self._stop.racing and self._gated:
            raise RefusalDiverged(
                f"{op!r} arrived from an op a layer refused on an earlier attempt: a race "
                "branch's gates must decide the same way on every attempt"
            )
        match op:
            case Scoped(scope=scope, body=body):
                return self._run_scoped(scope, body)
            case Gather(branches=branches):
                return self._run_gather(branches)
            case Race():
                return self._run_race(op)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _step(self, op: Step[Any], inner: DomainOp[Any]) -> Any:
        """Interpret a `Step` onto a durable checkpoint. The guarded arm is the v1
        usage-in-checkpoint path; the unguarded arm covers every other case (v0, CallTool,
        or a domain with no `run_metered`) and names the model-call marker and `CallTool`, so an
        op outside `ModelCall` is a type error here."""
        schema = _schema_of(inner)
        slot = self._slot(op)
        keyed_call(op, inner)  # a key nobody but the handler may write refuses before the step
        match inner:
            case AsksModel() if (
                run_metered := getattr(self.domain, "run_metered", None)
            ) is not None and metered_call(inner, self._contract, self.domain):
                # `metered_call` is the single placement predicate (shared with `measured_drive`);
                # the walrus binds the narrowed `run_metered` callable for the fold below.
                # Pre-check the measured ceiling BEFORE forwarding — refuse/park the *next*
                # AskLLM once spend crosses the limit (the old `metered` semantics, now
                # replay-derived and above the checkpoint).
                self._enforce_measured(op)
                # Envelope `{result, usage}` so usage becomes recorded state. The fold is
                # ABOVE `ctx.step` — on replay the thunk is skipped and the recorded
                # envelope returned, so the usage re-derives here and the meter re-trips at
                # the same op. The workflow still sees the bare result, so
                # its byte-identity holds.
                raw = self._checkpoint(
                    op,
                    slot,
                    lambda fn=run_metered, inner=inner: _encode_usage_envelope(*fn(inner)),
                )
                result_raw, usage = _decode_usage_envelope(raw)
                self._meter = self._meter + usage
                return _load_step(schema, result_raw)
            case CallTool(name=name) if name == SPAWN_TOOL:
                # The decision runs inside the thunk, and a refusal is its recorded value, so a
                # spawn replayed from a checkpoint or a fork's seed is served as recorded and never
                # decided again.
                writer = self._writer(slot)

                def spawn() -> Any:
                    try:
                        request, done_event = self._spawn_request(op, inner, writer)
                    except Refused as refused:
                        return {"refused": True, "reason": refused.reason}
                    enqueued = _dump(self.domain.run(request))
                    return {**enqueued, "done_event": done_event.stored()}

                # The checkpoint holds the done event as text, and this is where it is read back.
                match self._checkpoint(op, slot, spawn):
                    case {"done_event": str(stored), **enqueued}:
                        return _load(schema, {**enqueued, "done_event": Key.parse(stored)})
                    case {"refused": True, "reason": str(reason)}:
                        raise Refused(op, reason)
                    case recorded:
                        raise LookupError(
                            f"this spawn's checkpoint carries no done event ({recorded!r}): it "
                            "was recorded before the handler named done events, so its join "
                            "cannot be re-bound. Drain runs parked on a child-done or fork-done "
                            "join."
                        )
            case AsksModel() | CallTool():
                # The key is minted on a miss, so a step served from the store asks nothing of
                # the ctx, which may have no task.
                writer = self._writer(slot)
                raw = self._checkpoint(
                    op,
                    slot,
                    lambda inner=inner: _dump(self.domain.run(self._keyed(op, inner, writer))),
                )
                return _load_step(schema, raw)
            case unreachable:
                assert_never(unreachable)

    def _keyed(self, op: Step[Any], inner: DomainOp[Any], writer: Writer | None) -> DomainOp[Any]:
        """`inner`, with the idempotency key the step asks for minted into its args."""
        match keyed_call(op, inner):
            case None:
                return inner
            case CallTool(args=args) as call:
                if writer is None:
                    raise Refused(
                        op,
                        "an idempotency key is minted from the task and the step's placement, "
                        "and this ctx has no task",
                    )
                return replace(
                    call, args={**args, "idempotency_key": idempotency_key_for(writer).stored()}
                )
            case unreachable:
                assert_never(unreachable)

    def _spawn_request(
        self, op: Step[Any], spawn: CallTool[Any], writer: Writer | None
    ) -> tuple[CallTool[Spawned], Key]:
        """The spawn this task may make (refused before enqueue), and the event its child
        answers on.

        Refused when its depth is spent, its args are malformed, or it names itself. Otherwise the
        child gets no more depth than one level below this task and none of the substrate's own
        params but the done event, and the spawn and its done event are both named by `writer`.
        Every spawn a workflow yields passes here, from authoring sugar or a model's tool request
        alike."""
        if self._spawn_budget.depth_exhausted():
            raise Refused(
                op, f"spawn depth exhausted: {BUDGET_DEPTH_PARAM}={self._spawn_budget.depth}"
            )
        try:
            args = SpawnArgs.model_validate(spawn.args)
        except ValidationError as malformed:
            raise Refused(
                op, f"malformed spawn args: {malformed.errors(include_url=False)}"
            ) from malformed
        match args.idempotency_key, writer:
            case str() as supplied, _:
                raise Refused(
                    op, f"a spawn is named by the handler, not by its args: {supplied!r}"
                )
            case None, None:
                # The drive loop places every step, so a spawn without a writer lacks its task.
                raise Refused(
                    op, "a spawn is named by its task and placement, and this ctx has no task"
                )
            case None, Writer() as placed:
                name, done_event = idempotency_key_for(placed), spawn_done_name(placed)
        authored = {k: v for k, v in args.params.items() if k not in SUBSTRATE_PARAMS}
        stamped = {**authored, DONE_EVENT_PARAM: done_event.stored()}
        request = args.model_copy(
            update={
                "params": self._spawn_budget.descend_one().stamp(stamped),
                "idempotency_key": name.stored(),
            }
        )
        return request.call(), done_event

    def _enforce_measured(self, op: Step[Any]) -> None:
        """The measured-spend trip, checked before a sequential v1 model call: a
        driver of `budget.enforce_measured`. `Cleared` advances the accrual, `Exceeded` raises
        `BudgetRefused` at `op`, and `Parked` parks on the named grant, which binds by name on
        resume, then re-drives with the grant folded. A branch handler holds no budget, so this
        does nothing inside a gather."""
        b = self._budget
        if b is None:
            return
        # `self._grants`/`self._trips` stay FIXED across the parking loop; the local `grants`
        # dict grows and the transition RE-FOLDS from entry each park — which is exactly what
        # makes a resume/replay idempotent (the local dict is rebuilt purely from `await_event`
        # replay, never a smuggled bookkeeper). Accrual is written back ONLY on `Cleared`
        # (below). A park/refuse terminal therefore leaves `_grants`/`_trips` at their entry
        # values — harmless because the handler is DISCARDED at both: a park unwinds via
        # suspend and rebuilds fresh on replay (no-call/cc). A refusal the workflow catches leaves
        # them too, so the next metered ask re-folds the same delivered grants and refuses again.
        grants: dict[Key, Grant] = {}  # grants awaited within this trip, folded by the transition
        while True:
            match enforce_measured(self._meter.cost, b, self._grants, self._trips, grants):
                case Cleared(granted=granted, trips=trips):
                    self._grants, self._trips = granted, trips
                    return
                case Exceeded() as exceeded:
                    raise BudgetRefused(op, exceeded)
                case Parked(name=name):
                    # `_root_ctx`, not `self.ctx`: a measured grant is a RUN-level question, so
                    # its park takes no scope frames — see `_root_ctx` for the discriminator and
                    # why an `approve:` park is the opposite case.
                    grants[name] = _load(Grant, self._await_absolute(name))
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _branch_await(self, name: Key, schema: type) -> Any:
        """A gather branch's await PEEKS: it must not touch the engine's
        suspend machinery mid-round (an unsatisfied engine await flips the run
        state as a SQL side effect and forbids a second await per run).
        Satisfied → the payload binds here; unsatisfied → park as a value; the
        single post-barrier re-arm is the run's only real await."""
        if not _supports_peek(self.ctx):
            raise NotImplementedError(
                "await_event inside a gather branch needs a ctx with the peek_event "
                "capability (SqliteTaskContext / ConcurrentAbsurdCtx); this ctx has "
                "none, so the await-in-gather wall stays up here"
            )
        inner: Any = self.ctx
        found, payload = inner.peek_event(name)
        if found:
            return _load(schema, payload)
        # Carry the scope frames: `_join` re-adds only the branch coordinate.
        raise _GatherPark(event=name.prefixed(self._scope_path))

    def _await_absolute(self, name: Key) -> Any:
        """Resolve an ABSOLUTE name at the ctx this handler was CONSTRUCTED at.

        The one reroute, with both callers going through it — `_enforce_measured`'s
        `budget-grant` park and `_await`'s declared-absolute arm. One target behind one function
        is the cheapest guarantee the two cannot diverge.

        `_root_ctx` retains what the handler was BORN with (a gather coordinate, a fork rename)
        and drops only what `_run_scoped` pushed afterwards. A fork child's rename therefore
        survives this reroute — which is *required* (bypassing it would let a base's answer
        resolve a child's park) and is exactly why the rename cannot be rerouted around the way a
        scope can. `_await`'s declared-absolute arm refuses that case before reaching here
        (`ops.refuse_absolute_await_under_a_rename`); a `budget-grant` still passes through,
        because its emitter READS the parked name rather than composing one."""
        return self._root_ctx.await_event(name)

    def _await(self, op: AwaitEvent[Any]) -> Any:
        """Interpret an `AwaitEvent`: a real engine await at a sequential point, a
        non-suspending PEEK inside a gather branch (an unsatisfied engine await flips the run
        state and forbids a second await per run, so a branch must not touch the suspend
        machinery mid-round).

        An ABSOLUTE await resolves through `_await_absolute`, the same function `_enforce_measured`
        reroutes a `budget-grant` through. A
        scope completes RELATIVE names; an absolute one is already whole, and its emitter is a
        different task that has never seen this scope. Inside a gather branch there is nothing to
        reroute to (the coordinate is a concurrency slot, and the barrier re-arms the park with
        it re-added), so that case refuses instead."""
        if self._stop.racing:  # an await a layer injected, which admission never saw
            refuse_a_park_in_a_race_branch(op, getattr(self.ctx, "prefix", ""))
        # The WALK's name for this await — occurrence-qualified where the namespace declared
        # SETTLEMENT (`placing`), `op.name` verbatim otherwise. Read once here so every arm
        # below — the absolute reroute, the branch peek, and the sequential engine await —
        # binds the same name a human is told to emit against.
        name, schema = placed_await_name(op), op.schema
        if awaits_an_absolute_name(op):
            if self._in_gather:
                refuse_absolute_await_in_branch(op, getattr(self.ctx, "prefix", ""))
            # The second frame that cannot be rerouted around, and the one the branch check
            # cannot see: a fork child's event world. Read off `_root_ctx` because that is
            # the ctx `_await_absolute` will actually await at; checking a frame the reroute
            # then discards would refuse the wrong programs.
            #
            # Two deferrals, each to a more specific answer:
            #
            # * `transplanted` is the substrate's own word for "the caller HAS delivered this
            #   under the child's names", so the emitter IS rename-aware and the name is
            #   producible: this shape completes end to end.
            # * a `Seeding`-phase await belongs to `SeedingCtx`, which refuses it as
            #   `ForkedPrefixAwait`, a typed fork refusal that is RELAYED to the parent and says
            #   *which* await and to use `transplanted`. No phase at all (a bare rename, no fork
            #   prefix machinery) means nothing better is coming, so refuse.
            rename = getattr(self._root_ctx, "event_rename", "")
            transplanted = getattr(self._root_ctx, "transplanted", frozenset())
            if (
                rename
                and name not in transplanted
                and not isinstance(getattr(self._root_ctx, "phase", None), Seeding)
            ):
                refuse_absolute_await_under_a_rename(op, rename)
            if op.deadline is not None:
                # The reroute is a NAMING decision — who completes the name — and the clock is a
                # different axis, so an absolute wait keeps its deadline and waits at the root.
                return self._await_bounded(op, name, op.deadline, self._root_ctx)
            return _load(schema, self._await_absolute(name))
        if op.deadline is not None:
            if self._in_gather:
                refuse_a_bounded_wait_in_a_branch(op, getattr(self.ctx, "prefix", ""))
            return self._await_bounded(op, name, op.deadline, self.ctx)
        if self._in_gather:
            return self._branch_await(name, schema)
        return _load(schema, self.ctx.await_event(name))

    def _await_bounded(
        self, op: AwaitEvent[Any], name: Key, deadline: datetime, at: Any
    ) -> WaitOutcome[Any]:
        """A wait the clock can end, answered once at `at` and validated on arrival.

        **The SETTLE SLOT is the op's placement, not its address.** Two waits on one name are one
        address and two questions — the shape a receive loop has — so a slot keyed by the name
        alone hands the second ask the first ask's answer. `current_placement` is the walk's
        occurrence-qualified name for the op in flight, which is the coordinate every interpreter
        already agrees on, and it is the only place that knows WHICH ask is asking.

        `await_until` is an optional ctx capability, so a ctx without one is told what it lacks
        here rather than meeting an `AttributeError` several frames down. The deadline crosses as
        epoch seconds, which is what both engines compare against their own clock."""
        if not _supports_await_until(at):
            raise NotImplementedError(
                f"this ctx ({type(at).__name__}) waits on an event alone, so it cannot "
                f"answer the deadline {deadline.isoformat()} that {name.display()!r} named. "
                "Await the event without a deadline, or run on an engine whose ctx offers "
                "`await_until` (the embedded SQLite engine and the Absurd SDK both do)."
            )
        inner: Any = at
        outcome: WaitOutcome[Any] = inner.await_until(name, deadline.timestamp(), self._slot(op))
        match outcome:
            case Arrived(payload=payload):
                return Arrived(_load(op.schema, payload))
            case Expired():
                return Expired()
            case unreachable:
                assert_never(unreachable)

    def _step_name(self, op: WorkflowOp, slot: Key) -> Key:
        """The name `ctx.step` checkpoints `op` at: its key in the ctx's coordinates, at the
        occurrence of its slot."""
        return op_key(op).occurrence(split_occurrence(slot.stored())[1] or 1)

    def _slot(self, op: WorkflowOp) -> Key:
        """The op's placement in this handler's frames: the walk's, or one counted here.

        | the op                                     | its slot                                  |
        |--------------------------------------------|-------------------------------------------|
        | the walk op, forwarded once or again       | `current_placement`, minted by `_place`   |
        | an op a layer yields: `permission`'s and   | `_place(op)`, counted as the walk counts  |
        | `govern`'s awaits, `govern`'s announce     | its own                                   |

        The ambient placement is this op's when its key, occurrence stripped, is this op's key.
        Under any other op's placement, an injected wait would settle into the wrapped op's
        checkpoint and serve the wait's record back as that op's result. An op a layer yields
        under the key of the op it wraps takes that op's slot, and is served its checkpoint. An
        op injected under a re-forward with the key its first forward asked takes the slot that
        forward counted (`_Reforwards`); one whose layer numbers its key per invocation places a
        new key."""
        mine = placed_key(op).prefixed(getattr(self.ctx, "prefix", "")).stored()
        ambient = current_placement()
        if ambient is not None and split_occurrence(ambient.stored())[0] == mine:
            return ambient
        return self._place(op)

    def _sleep_name(self, name: Key | None) -> Key:
        """The walk's name for a sleep, required.

        The ONE place the walk's `Key | None` becomes the seam's `Key`. `current_op_name()` and
        a parked branch's carried name are both optional in the type because not every op has a
        position, but a sleep always does — so every path to `ctx.sleep_until` narrows here, and
        `None` means the sleep was driven by something other than a handler's walk."""
        if name is None:
            raise ValueError(
                "a sleep reached the engine with no name: a sleep is identified by its "
                "position in its thread of control, which only a handler's drive loop assigns. "
                "Drive the sleep through a handler rather than calling the ctx directly."
            )
        return name

    def _branch_sleep(self, when: datetime, name: Key) -> None:
        """The sleep analog of the await peek is a pure clock compare (a sleep's
        only durable payload is its wake time — nothing to re-bind); undue →
        park as a value, re-armed at the barrier under `name`."""
        if time.time() >= when.timestamp():
            return None
        raise _GatherPark(until=when, name=name)

    def _branch_handler(self, g: int, i: int) -> DurableHandler:
        """A child handler for branch *i* of gather *g*, over a ``gather:{g},{i};`` ctx."""
        return self._child(gather_prefix(g, i), self._stop)

    def _child(self, prefix: str, stop: Stop) -> DurableHandler:
        """A child handler for a gather or race branch over `prefix`. It carries its OWN
        per-branch meter subtotal, folded into this parent at the barrier in
        index order, inherits the parent's contract, and stops as `stop` says."""
        child = DurableHandler(
            _PrefixedCtx(self.ctx, prefix),
            self.domain,
            ledger=self.ledger,
            op_layers=self.op_layers,
            in_gather=True,
            contract=self._contract,
            params=None
            if self._spawn_budget.depth is None
            else {BUDGET_DEPTH_PARAM: self._spawn_budget.depth},
            stop=stop,
        )
        child._fresh = self._fresh
        if self._cursor is not None:
            tree = self._stop.tree_lock()
            with tree:
                child._cursor = self._cursor.inherit(getattr(child.ctx, "prefix", ""))
                for bound in child._cursor.bounds:
                    bound.race.register(child._cursor)
            child._stop = Stop((child._cursor.stopped,), tree)
            self._children.append(child)
        return child

    def _retire(self, children: list[DurableHandler]) -> None:
        """At a barrier: children whose threads have ended leave their races' live sets, and
        their counts and inputs fold into this thread's."""
        mine = [child for child in children if child._cursor is not None]
        if not mine or self._cursor is None:
            return
        with self._stop.tree_lock():
            finished = 0
            for child in mine:
                assert child._cursor is not None
                with child._cursor.lock:
                    finished += child._cursor.count
                    bounds = child._cursor.bounds
                for bound in bounds:
                    bound.race.retire(child._cursor)
                handed = digest(child._inputs)
                self._inputs.append(None if handed is None else ["child", handed])
        self._cursor.fold(finished)
        self._children = [child for child in self._children if child not in mine]

    def _run_scoped(self, scope: Key, body: Callable[[], Any]) -> Any:
        """Durable `scoped(...)`: run `body` over a ctx that namespaces its checkpoints and
        events under `{scope};`: `_PrefixedCtx`, the same wrapper a gather branch rides.

        **The same handler, not a child.** A gather needs children because its branches run
        concurrently and each must carry its own meter subtotal to be folded confluently at the
        barrier. A scope is sequential (one body, no barrier, nothing to fold), so the meter,
        the budget and the layer stack simply continue, which is what a caller means by "run
        this part of my workflow under a namespace".

        Two pieces of state are saved and restored: the **ctx**, which is the scope itself, and
        the scope path. The **ordinals** are not: they belong to the handler, so a gather inside
        the scope takes the handler's next `gather:{g}` and a scope entered twice cannot name its
        gather the same both times. All three interpreters count this way, or a recorded key
        stops matching the replayed one.

        Nothing special is needed for a park: an await inside the body raises `SuspendTask`
        through this frame, the task replays from the top, and re-execution re-enters the scope
        and re-applies the prefix. That is the no-`call/cc` rule paying for itself: the scope
        is re-derived, never captured.
        """
        saved_ctx, saved_path = self.ctx, self._scope_path
        self.ctx = _PrefixedCtx(saved_ctx, scope_prefix(scope))
        self._scope_path = frame_path(saved_path, scope)
        try:
            # `_drive`, NOT `_run`: a scoped body's value flows back into a live generator in
            # this process, so it must stay a host-language object. `_run` serializes, because
            # it is the TASK-result boundary — and a `Deeper(...)` returned through it came back
            # as a dict, matching no arm of the drill's decision table and spinning the
            # trampoline forever. Caught only because a scoped body returned a dataclass; the
            # first conformance workflow returned ints, where the dump is invisible.
            return self._drive(body)
        finally:
            self.ctx, self._scope_path = saved_ctx, saved_path

    def _run_gather(self, branches: tuple[Callable[[], Any], ...]) -> list[Any]:
        """Durable gather: each branch runs through a child handler whose ctx namespaces its
        durable keys ``gather:{g},{i};`` (``{g}`` the gather's position in this workflow,
        ``{i}`` the branch), so checkpoints never collide with a sibling
        branch *or* with another same-shaped gather, and on replay each branch re-binds its
        own committed checkpoints by name (order-independent, per-branch crash-resume: a
        crash mid-gather keeps committed branch-steps and re-runs only the rest). Results
        join in branch index order.

        If the ctx advertises ``concurrent_safe`` (the SQLite engine, or an Absurd ctx
        wrapped in ``ConcurrentAbsurdCtx`` — whose write-lock serializes commits while tool
        work runs lock-free), the branches run under structured concurrency — **tools
        overlap, writes serialize** (sequential consistency → a partial order on the
        ledger). Otherwise they run sequentially; the structural keying carries the
        durability either way.
        """
        g = self._position.next_gather()
        children = [self._branch_handler(g, i) for i in range(len(branches))]
        try:
            if getattr(self.ctx, "concurrent_safe", False):
                slots = asyncio.run(self._gather_concurrent(children, branches))
            else:
                # Sequential round: parks and errors are values here too, so later branches
                # still run to their end before the barrier.
                slots = [branch_slot(children[i].run, thunk) for i, thunk in enumerate(branches)]
        finally:
            self._retire(children)
        parked = any(isinstance(slot, BranchParked) for slot in slots)
        if (raised := barrier_errors(slots, parked=parked)) is not None:
            raise raised
        if any(isinstance(slot, BranchStopped) for slot in slots):
            # A race loser holding this gather stopped inside it. The round completed, so its
            # siblings' spend is folded before the loser stops.
            for child in children:
                self._meter = self._meter + child._meter
            raise Stopping
        joined = self._join(g, slots)
        # The confluent barrier fold: each branch accrued into its OWN subtotal
        # (single-threaded within the branch), now summed into the parent in branch-INDEX
        # order, a pure function of the per-branch results, so the meter is
        # schedule-independent. Only a fully-completed round reaches here (`_join`
        # parks/raises on any parked branch), so a parked branch's partial subtotal is never
        # folded; it re-runs whole on resume and re-derives from the record.
        for child in children:
            self._meter = self._meter + child._meter
        return joined

    def _join(self, g: int, slots: list[Any]) -> list[Any]:
        """The round barrier's decision side: all-complete → the joined
        results; any park → the whole task parks on the LOWEST parked branch's
        wake condition, deterministically.

        A NESTED gather re-raises with the path-composed relative event name —
        only the top level re-arms. The top-level re-arm walks parked branches
        in ascending index and issues the run's ONE real engine await (or
        sleep): the first still-unsatisfied condition raises the engine's own
        park signal and the task parks; a condition already satisfied (the
        payload arrived mid-round or during an earlier branch's park) resolves
        and is frozen for replay, and the walk moves on. All satisfied → the
        wake race: the branches' values are recoverable only by replay (a
        parked branch must never re-run in-process; the occurrence counters
        would shift its committed checkpoint names), so ``repark`` re-queues
        the task with no attempt burned, falling back to ``GatherWakeRace``
        (the engine's retry) when the ctx lacks the capability or the repark
        returns without parking. Repark fires ONLY here, at the full-race
        fall-through: every parked branch has resolved, so each re-binds at
        its peek on every later replay and can never race on the SAME wake
        condition again. A branch's NEXT await can race the same gather again
        (a race is per await, not per branch), which is why the
        deterministic step name (`wake_race_on_event` / `wake_race_at_time`) carries
        the condition: each race is at most once per name, and the occurrence
        counters stay clean."""
        parked = [(i, s) for i, s in enumerate(slots) if isinstance(s, BranchParked)]
        if not parked:
            return slots
        if self._in_gather:
            i, bp = parked[0]
            if bp.event is not None:
                # `_scope_path` for the same reason `_branch_await` carries it: a park travels
                # outward as a VALUE, and each frame it crosses must re-add its own prefix or the
                # name arrives SHORT — and a short name is not a wrong wake, it is no wake at all,
                # because no emitter following the contract composes it. This arm is the
                # NESTED-gather sibling of that one, so it needs the frame path for the same
                # reason and is the arm most easily forgotten, having two frames to re-add.
                # One `prefixed` over both fragments: `_scope_path` is a frame path and
                # `gather:{g},{i};` is the gather grammar's segment, and neither is an identity
                # — building the PREFIX with an f-string is the render-backend position, while
                # applying it to `bp.event` is the composition, so that half takes the named
                # exit. Same bytes as the concatenation this replaced.
                raise _GatherPark(event=bp.event.prefixed(self._scope_path + gather_prefix(g, i)))
            raise _GatherPark(
                until=bp.until,
                name=None
                if bp.name is None
                else bp.name.prefixed(self._scope_path + gather_prefix(g, i)),
            )
        for i, bp in parked:
            branch_ctx = _PrefixedCtx(self.ctx, gather_prefix(g, i))
            if bp.event is not None:
                branch_ctx.await_event(bp.event)  # unsatisfied → the engine parks the task
            else:
                assert bp.until is not None
                branch_ctx.sleep_until(bp.until, name=self._sleep_name(bp.name))  # undue → parks
        # The wake race. _join re-arms only at top level, so self.ctx is the raw
        # (unprefixed) ctx here; repark is optional (getattr, peek_event's pattern).
        # The name puts `wake-race` in the SECOND segment — unforgeable: a top-level
        # author step may not start with `gather:` (op_key) and a branch step always
        # interposes an integer there (the prefix chain) — and carries the lowest
        # raced branch's wake CONDITION, so a later race of the same gather (a
        # branch's NEXT await) mints a fresh name instead of finding this one stale.
        repark = getattr(self.ctx, "repark", None)
        if callable(repark):
            i, bp = parked[0]
            # COMPOSED, not f-stringed. This was
            # `repark(f"{self._scope_path}gather:{g};wake-race:{i},{cond}")` — an identity built
            # by hand, which cost two things. It was invisible to `--key-registry` (a scan of
            # `compose_key` call sites), so the gate that would have caught a mis-translated
            # expectation could not see the key it was about; and the sleep arm spliced an
            # `isoformat()`, composing a name the grammar refuses. Both close here.
            if bp.event is not None:
                name = wake_race_on_event(g, i, bp.event)
            else:
                assert bp.until is not None
                name = wake_race_at_time(g, i, bp.until)
            # `_scope_path` for the same reason every other name on this path carries it: two
            # same-ordinal gathers in DIFFERENT scopes would otherwise mint one race name, and the
            # second would find the first's step stale. Degrades to the documented
            # `GatherWakeRace` fallback rather than corrupting anything, which is why this was a
            # nit rather than a blocker, but it is the last name on the park
            # path that was still frame-blind.
            repark(name.prefixed(self._scope_path).stored())  # raises the park signal…
        raise GatherWakeRace(  # …and returns only on the stale-checkpoint edge: still raise
            f"gather:{g}: every parked branch's wake condition was already satisfied "
            "at the re-arm; re-queueing so replay resolves the branches from the "
            "durable record (see GatherWakeRace)"
        )

    async def _gather_concurrent(
        self, children: list[DurableHandler], branches: tuple[Callable[[], Any], ...]
    ) -> list[Any]:
        async def run_branch(child: DurableHandler, thunk: Callable[[], Any]) -> Any:
            # Each branch drives in its own thread (its own child handler → its own meter
            # subtotal), so the lock-free tool work overlaps; the shared ctx's write-lock
            # serializes the commits. A branch's exception is its slot, so every branch finishes
            # and the barrier reports each branch's error.
            return await asyncio.to_thread(branch_slot, child.run, thunk)

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(run_branch(c, b)) for c, b in zip(children, branches, strict=True)
            ]
        return [t.result() for t in tasks]  # index order, never completion order

    def _run_race(self, op: Race) -> Answer[Any]:
        """Durable race. The choice is read from the store first, with its horizon;
        absent, the batches decide and `RaceState.publish` saves it, and every loser's cursor is
        bounded by what the store returned. Branches run concurrently on a `concurrent_safe` ctx
        and in index order otherwise. At the barrier every branch's meter folds in index order, a
        stopped loser's included, and the endings are settled with a digest of each value ending's
        inputs: an incarnation whose branch was handed different inputs fails with `EndingLost`."""
        ctx = _race_capable(self.ctx)
        r = self._position.next_race()
        if self._fresh is None:
            self._fresh = _attempt_of(ctx) == 1
        found, stored = ctx.peek_step(race_choice(r))
        state = RaceState(
            Choice.from_stored(stored) if found else None,
            dict(stored.get("horizon", {})) if found else {},
        )
        tree = self._stop.tree_lock()
        children = [self._racer(state, tree, r, i) for i in range(len(op.branches))]
        for child in children:
            child._fresh = self._fresh and not found

        def choose(proposal: Choice) -> None:
            state.publish(proposal, lambda value: ctx.settle(race_choice(r), value))

        racing = Racing(
            want=op.want,
            branches=len(op.branches),
            run=lambda i: branch_slot(children[i].run, op.branches[i]),
            choose=choose,
            decided=lambda: state.box.current() is not None,
            enclosed=self._stop.now,
            tree=tree,
            deadline=deadline_of(op),
        )
        try:
            slots = (
                asyncio.run(racing.concurrently())
                if getattr(self.ctx, "concurrent_safe", False)
                else racing.in_order()
            )
        finally:
            self._retire_racers(children, tree)
        for spent in children:
            self._meter = self._meter + spent._meter
        if (diverged := _diverged(slots)) is not None:
            raise diverged
        if (choice := state.box.current()) is None:
            if (raised := race_errors(slots)) is not None:
                raise raised
            raise Stopping  # an enclosing race's choice stopped this race before it chose
        if (transient := transient_errors(slots)) is not None:
            raise transient
        values = {i: slot for i, slot in enumerate(slots) if settled(slot) == "won"}
        saved = self._settle_endings(ctx, r, choice, slots, children)
        self._inputs.append(observed(["race", choice.stored(), saved]))
        return answer(choice, endings_from_stored(saved, values))

    def _settle_endings(
        self, ctx: Any, r: int, choice: Choice, slots: list[Any], children: list[DurableHandler]
    ) -> list[dict[str, Any]]:
        """Race `r`'s endings as the store holds them, each value ending with a digest of its
        branch's inputs. A branch that ended with a value on an earlier attempt must reproduce
        it here from the same inputs, or the race fails with `EndingLost`."""
        inputs = [digest(child._inputs) for child in children]
        replayed, saved = ctx.peek_step(race_endings(r))
        if not replayed:
            endings = [ending_of(i, slot, choice) for i, slot in enumerate(slots)]
            saved = ctx.settle(race_endings(r), stored_endings(endings, inputs))
        for i, raw in enumerate(saved):
            witness = raw.get("inputs_sha256")
            if raw["ending"] in ("won", "unchosen") and (
                settled(slots[i]) != "won"
                or witness != inputs[i]
                or (replayed and witness is None)
            ):
                raise EndingLost(
                    f"branch {i} ended with a value on an earlier attempt, and this attempt "
                    "handed it different inputs, or none it can verify"
                )
        return saved

    def _retire_racers(self, children: list[DurableHandler], tree: threading.Lock) -> None:
        """A race's branches leave their races' live sets at its barrier. Inside a race branch they
        also fold into this thread's cursor, as a gather's do."""
        if self._cursor is not None:
            self._retire(children)
            return
        with tree:
            for child in children:
                assert child._cursor is not None
                for bound in child._cursor.bounds:
                    bound.race.retire(child._cursor)

    def _racer(self, state: RaceState, tree: threading.Lock, r: int, i: int) -> DurableHandler:
        """Branch `i` of race `r`'s handler, its cursor enrolled in `state`."""
        child = self._child(race_prefix(r, i), NO_RACE)
        frame = getattr(child.ctx, "prefix", "")
        with tree:
            if child._cursor is None:
                child._cursor = Cursor(frame)
            state.register(child._cursor, i)
        child._stop = Stop((child._cursor.stopped,), tree)
        return child

    def _record_ledger(self, row: LedgerRow, writer: Writer | None) -> Any:
        # Append to the dedicated append-only ledger (idempotent by event_id);
        # ctx.step makes it skip on replay. Without a ledger writer the row is
        # just checkpointed.
        #
        # `ledger=None` is THE DURABLE NO-COMMIT MODE, not merely a test affordance — a
        # counterfactual runs this way, and it is why no `no_commit` op-layer exists. A layer
        # that answered this op without forwarding would skip the `ctx.step` entirely, collapsing
        # the checkpoint sequence and breaking the parity a fork's diff depends on; the same
        # applies to
        # `StoreArtifact`, whose checkpoint IS the artifact store pre-CAS. Here the checkpoint is
        # written and only the canonical append is suppressed — the guarantee a fork needs,
        # without the damage. The other half of counterfactual safety (stopping a world-mutating
        # TOOL) is `effective.sandbox.DryRun`; this seam does not cover it.
        if self.ledger is not None:
            self.ledger.append(row, writer=writer)
        return _dump(row)

    def _writer(self, placement: Key) -> Writer | None:
        """WHO is writing: the task, and the op's slot inside it (`_slot`), which for an op a
        layer yields is its own and not the wrapped op's.

        `None` when the task is missing: `task_id` is not on the `TaskContext`
        protocol — it is reachable by `getattr` on both engines (a `UUID` on SQLite, a `str` on
        the SDK) the same way `prefix` is, but a test double need not carry it. `None` reads as
        *unknown*, which the store's check allows; a guess would read as *known* and refuse a
        legitimate re-append.

        Attempt is deliberately absent even though the engines track it. Per TASK is the
        predicate precisely so a retry, which re-executes the same placement, stays permitted;
        including attempt would refuse the crash window this exists to allow."""
        task = getattr(self.ctx, "task_id", None)
        return None if task is None else Writer(task=str(task), placement=placement)
