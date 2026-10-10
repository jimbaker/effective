"""Harness layers: cross-cutting effects as a composable stack.

A *harness factor* (retry, cost, compaction, permission, …) is a cross-cutting
concern wrapped around the op stream. This module gives those factors one
writing idiom and one composition idiom, at two seams:

| seam   | decorator       | alphabet     | scope                                                |
|--------|-----------------|--------------|------------------------------------------------------|
| op     | `@op_layer`     | `WorkflowOp` | a drive loop; may suspend or inject control-flow ops |
| domain | `@domain_layer` | `DomainOp`   | one `DomainInterpreter.run` call; no control flow    |

The seams stay separate because authority confinement, replay grain and
lifetime all differ. The shared idiom is a generator middleware,
`result = yield op`, the `contextlib.contextmanager` shape with one divergence:
a layer **may yield more than once**, and a `while`/`for` around `yield op` *is*
retry.

**Which seam? Ask three questions in order** (whether it can park alone
under-decides: `retry` is an op layer that cannot park):
  1. **Which alphabet must it see**: `WorkflowOp` (it reacts to `Step`/`AwaitEvent`
     names, e.g. compaction, op-stream telemetry) or just a `DomainOp` call? Op
     alphabet ⇒ op seam.
  2. **May it park** (suspend and resume for a grant or approval)? Park ⇒ op seam
     (`govern`): parking needs a durable `AwaitEvent`, the op seam's exclusive
     authority. Raising `Refused` is a refusal and a domain layer may raise it.
  3. **May it block** (a long sleep)? Blocking a durable worker slot wants a
     `SleepUntil` op (op alphabet), never a domain-seam `time.sleep`.
Otherwise (it transforms, re-invokes or observes one call) it is a **domain**
seam service (`serve`). The domain `retry_domain` is the twin of the op `retry`.

`drive_through` is the trampoline that pumps a per-op layer stack down to a base.
It supports rewrite (yield a different op), multi-yield (loop → retry), and
exception propagation back *into* a layer's `yield` (via `.throw`, so a layer's
`try/except` around `yield op` catches a downstream failure).
"""

import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, assert_never, runtime_checkable

from effective.domain import DomainOp
from effective.keys import Key
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    Race,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WorkflowOp,
)

if TYPE_CHECKING:
    # `cost` imports this module, so the annotation is the only place `Usage` appears here.
    from effective.cost import Usage

# --- The two layer types (the alias conveys the authority the bare Generator does not) ---

type OpLayer[T] = Callable[[WorkflowOp], Generator[WorkflowOp, Any, T]]
type DomainLayer[T] = Callable[[DomainOp[Any]], Generator[DomainOp[Any], Any, T]]

# --- Which ops an op-layer observes, per handler: a NAMED divergence ----------------------
#
# Op-layers wrap the op stream, but WHICH ops reach them is a per-handler *placement* decision,
# and it diverges by construction. The recording core cannot
# suspend inside a layer (the no-call/cc invariant forbids parking mid-layer), so it decides an
# `AwaitEvent`'s park/resume BEFORE entering the layer stack; the durable handler routes the
# `AwaitEvent` through layers (and re-fires them per op on replay). Alignment is impossible, so
# this is the shared *statement* of the contract, cited from `RecordingHandler._drive` and
# `DurableHandler._run` and pinned by `test_layers.py::test_L3_layer_op_stream_named_divergence`,
# which asserts this contract rather than stream equality.
LAYERED_OPS: tuple[type, ...] = (Step, SleepUntil, AppendLedgerRow, StoreArtifact)
"""Ops BOTH handlers route through the op-layer stack."""

CHECKPOINTED_OPS: tuple[type, ...] = (Step, AppendLedgerRow, StoreArtifact)
"""Ops that reach the engine as a checkpoint write: in a race, what a stopped loser may not
start."""

LAYERED_OPS_DURABLE_ONLY: tuple[type, ...] = (AwaitEvent,)
"""Durable routes these through layers; the recording core decides park/resume first
(no-call/cc forbids parking mid-layer), so they never reach a layer there."""

UNLAYERED_OPS: tuple[type, ...] = (Gather, Race, Scoped, Respawn)
"""Neither routes these through the top-level stack: a `Gather`'s or a `Race`'s layers apply
WITHIN each branch's child handler, and a `Scoped`'s within its BODY, for the same reason.

`Respawn` orchestrates a TASK BOUNDARY, has no `op_key` and no result of its own to authorize
(it does not return at all), and the spawn it performs is a checkpointed step in its own right.
Gating whether a chain may continue belongs to the budget axis; no layer decides it.

`Scoped` is in this tuple because without it the durable handler routes it through the stack and
the recording core does not (measured `['Scoped', 'Step']` against `['Step']`), so a layer that
calls `op_key(op)` raises on one path only. A structural op has no `op_key` and no result of its
own to authorize, so there is nothing for a layer to decide about it; the ops it *contains* are
each layered individually."""


type LayerRouting = Literal["layered", "layered-durable-only", "unlayered"]


def layer_routing(op: WorkflowOp) -> LayerRouting:
    """Which ops reach the op-layer stack — the THREE tuples above as one total decision.

    **The tuples partition `WorkflowOp` exactly.** A denylist (`isinstance(op, UNLAYERED_OPS)`,
    or a `match` whose wildcard means "layered") is bounded by what it enumerates, so a new arm
    of `WorkflowOp` would reach the layer stack silently. The dispatch is total instead, and a
    new arm is a named question. `tests/test_layers.py` pins that the tuples and this function
    agree and that between them they cover every arm.

    The `layered-durable-only` arm is a **named** divergence: the recording core decides an
    `AwaitEvent`'s park or resume *before* the layer stack, because no-call/cc forbids parking
    mid-layer, so a layer never sees one there."""
    match op:
        case Step() | SleepUntil() | AppendLedgerRow() | StoreArtifact():
            return "layered"
        case AwaitEvent():
            return "layered-durable-only"
        case Gather() | Race() | Scoped() | Respawn():
            return "unlayered"
        case unreachable:
            # A new arm of `WorkflowOp` must state whether the op-layer stack sees it, on each
            # handler; defaulting is what this function exists to prevent. The refusal is
            # static: `ty` fails the call above rather than a run failing later.
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def op_layer[F: Callable[..., Generator[Any, Any, Any]]](fn: F) -> F:
    """Mark a generator-middleware as an op-seam layer (alphabet: `WorkflowOp`).

    An op-layer may yield any `WorkflowOp` — including `AwaitEvent` (suspend) and
    injected control-flow ops. An introspection marker; the layer-authority lint
    keys on the *decorator name* (`@op_layer`), not this attribute — see `effective.lint`.
    """
    fn.__effective_layer__ = "op"  # ty: ignore[unresolved-attribute]
    return fn


def domain_layer[F: Callable[..., Generator[Any, Any, Any]]](fn: F) -> F:
    """Mark a generator-middleware as a domain-seam layer (alphabet: `DomainOp`).

    A domain-layer may only forward `yield op` or yield a `DomainOp`; it
    *cannot* reach a control-flow op — enforced by the `domain-layer-no-control-flow`
    ast-grep rule (`effective.lint`), the static form of capability confinement.
    """
    fn.__effective_layer__ = "domain"  # ty: ignore[unresolved-attribute]
    return fn


# --- The trampoline ---------------------------------------------------------


# --- the per-RUN layer scope -------------------------------------------------------------
#
# An op-layer sometimes needs state that spans the ops of ONE run — a gate's occurrence counter,
# its accrued answers. Where that state lives decides whether it is replay-DERIVED or
# process-CAPTURED, and getting it wrong is an authorization bug: a `govern()` closure,
# constructed once at registration, would let task B's gated op proceed on task A's grant with
# ZERO events in B's own durable record.
#
# So the scope's lifetime is tied to `Handler.run()` — **per attempt by construction**, because
# `run()` is called exactly once per attempt and re-called on every replay. A layer never holds the
# state itself, so there is no closure to outlive its task, and nothing to remember to reset.
#
# A `ContextVar` rather than a parameter: it keeps the op-layer signature
# `(op) -> Generator` untouched (a layer that does not need run state is unchanged), and it gives
# gather branches the right semantics for free — a branch child handler calls its OWN `run()`, so
# it gets its OWN scope. Sibling branches therefore cannot lose each other's absorbed answers to a
# last-writer-wins race, and a branch accrues nothing to the parent: the same stance the
# handler's meter already takes ("a branch carries no budget").
_RUN_SCOPE: ContextVar[dict[Key, Any] | None] = ContextVar(
    "effective_layer_run_scope", default=None
)


@contextmanager
def run_scope() -> Iterator[dict[Key, Any]]:
    """Enter a fresh per-run layer scope — called by a handler's `run`, once per attempt.

    Nested entry (a gather branch handler inside a parent's `run`) deliberately gets its OWN dict:
    branch-local accrual, no cross-branch interference, nothing folded into the parent."""
    scope: dict[Key, Any] = {}
    token = _RUN_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _RUN_SCOPE.reset(token)


_METER: ContextVar[Callable[[], Usage] | None] = ContextVar("effective_meter", default=None)


@contextmanager
def _metering(read: Callable[[], Usage] | None) -> Iterator[None]:
    """Publish the reader of this run's replay-derived spend, or `None` for a run that does not
    meter. `handlers.base._walk_run` is its one caller, so a run's spend comes from the handler
    that owns the run.

    The root handler of a task attempt publishes, and its gather branches inherit the root's
    reader, so a gate in a branch reads the task's spend. A branch's own subtotal reaches that
    spend at the barrier."""
    token = _METER.set(read)
    try:
        yield
    finally:
        _METER.reset(token)


def current_meter() -> Usage | None:
    """This run's replay-derived spend, or `None` when the run does not meter."""
    match _METER.get():
        case None:
            return None
        case read:
            return read()


_OP_NAME: ContextVar[Key | None] = ContextVar("effective_layer_op_name", default=None)


@contextmanager
def op_name_scope(name: Key) -> Iterator[None]:
    """Publish the walk's minted name for the op currently going through the layer stack.

    Set by each drive loop around its `drive_through` call, so a layer and the base
    interpretation see the SAME identity. That sameness is the point: a counter inside
    `drive_through` would be a second ordinal that has to agree with the handler's. One mint,
    two consumers.

    `drive_through` is also not the shared walk: `ReplayHandler` runs no layers, neither fork
    driver does, and it is re-entrant per layer, so a counter there would advance once per layer
    rather than once per op.

    Minted ABOVE the stack, deliberately: an ordinal minted here does not advance on a re-forward,
    because the workflow yielded once."""
    token = _OP_NAME.set(name)
    try:
        yield
    finally:
        _OP_NAME.reset(token)


def current_op_name() -> Key | None:
    """The name the walk minted for the op in flight, or `None` outside a drive loop."""
    return _OP_NAME.get()


_PLACEMENT: ContextVar[Key | None] = ContextVar("effective_placement", default=None)


@contextmanager
def placement_scope(placement: Key) -> Iterator[None]:
    """Publish WHERE the op in flight is running — its fully placed, occurrence-qualified name.

    Sibling to `op_name_scope`, and the distinction between them is the whole reason there are
    two. `current_op_name` answers *what name does the walk assign this op* and is a name
    SUBSTITUTION: a `SleepUntil` has no key until the walk mints one, so consumers use it INSTEAD
    of `op_key`. This answers *which placed op is executing right now* — the frames, the op's own
    key, and the occurrence, together — and nothing substitutes it for anything.

    Every consumer that needs an address *per node* rather than *per op* reads it here: the
    canonical ledger's writer (two branches, one `event_id`), a tool's idempotency key, and value
    attribution, where a tree search has to say which node earned a score.

    One publication, several readers, because the alternative is several spellings of one address
    that drift apart; the fork scope token is the worked example of what that costs."""
    token = _PLACEMENT.set(placement)
    try:
        yield
    finally:
        _PLACEMENT.reset(token)


def current_placement() -> Key | None:
    """The fully placed name of the op in flight, or `None` outside a drive loop.

    `None` is a real answer and readers must handle it: the recording/replay core publishes no
    placement, and a layer or domain exercised directly in a unit test has no walk above it."""
    return _PLACEMENT.get()


_REUSED: ContextVar[list[tuple[object, Usage]] | None] = ContextVar(
    "effective_reused", default=None
)


@contextmanager
def reuse_watch() -> Iterator[list[tuple[object, Usage]]]:
    """Watch the ops forwarded inside this block: each op answered from a store below, with what
    its answer cost when it was first asked. A watcher reads the entry for its own op object."""
    seen: list[tuple[object, Usage]] = []
    token = _REUSED.set(seen)
    try:
        yield seen
    finally:
        _REUSED.reset(token)


def mark_reused(op: object, bought: Usage) -> None:
    """Say that `op` was answered from a store, and what that answer cost when it was asked; a
    no-op when nothing watches."""
    if (seen := _REUSED.get()) is not None:
        seen.append((op, bought))


def layer_run_state(key: Key) -> dict[str, Any]:
    """The calling layer's slice of the current run scope, created on first use.

    **`key` is a `Key`, not a `str`**, so an f-string here is a TYPE ERROR rather
    than a silent aliasing bug. It replaced a real defect: this
    function was called with `f"govern:{gate}:{run_id}"`, where `gate='a', run_id='b:c'` and
    `gate='a:b', run_id='c'` addressed the SAME cell, so two gates shared one occurrence
    counter and one set of accrued answers — the per-occurrence authority discipline, silently
    defeated. `compose_key` refuses an interior interpolation carrying the delimiter, so the
    composed form cannot alias.

    **What this does NOT stop, stated plainly because an earlier version of this docstring
    overclaimed it.** A DELIBERATE cast still reaches this parameter ty-green and can still alias:
    `Key` is a frozen dataclass whose `__init__` takes the value, so `Key("govern:a:b:c")`
    constructs, and neither `ty` nor `--key-composition` looks inside it. The guard here is against
    the *accident* — a bare f-string, which is a type error — and not against a caller who means
    it. Do not read a `Key`-typed parameter as proof the value was composed.

    Outside a handler's `run` (a unit test driving a layer directly) this returns a **fresh** dict
    each call — no ambient accumulation, so a layer under test cannot silently carry state between
    cases. A layer that needs cross-op state must therefore be exercised through a handler, which
    is the only place the lifetime is real."""
    scope = _RUN_SCOPE.get()
    if scope is None:
        return {}
    return scope.setdefault(key, {})


def close_under(gen: Generator[Any, Any, Any], unwinding: BaseException) -> None:
    """Close a layer generator while `unwinding` passes it, which stays in flight.

    | the layer's cleanup | then                                                         |
    |---------------------|--------------------------------------------------------------|
    | finishes            | the generator is closed                                      |
    | raises              | its exception becomes a note on `unwinding`                  |
    | yields an op        | the yield's `RuntimeError` becomes a note, and a second      |
    |                     | close finishes the generator                                 |
    """
    for _ in range(2):
        try:
            gen.close()
        except BaseException as cleanup:
            unwinding.add_note(repr(cleanup))


@contextmanager
def closed_after(gen: Generator[Any, Any, Any]) -> Iterator[None]:
    """Close a layer generator when the block ends, and under `close_under` when an exception
    ends it."""
    try:
        yield
    except BaseException as unwinding:
        close_under(gen, unwinding)
        raise
    gen.close()


def drive_through(
    layers: Sequence[Callable[[Any], Generator[Any, Any, Any]]],
    op: Any,
    base: Callable[[Any], Any],
    *,
    escapes: tuple[type[BaseException], ...] = (),
    forwarding: Callable[[int, Any], None] | None = None,
) -> Any:
    """Run one `op` through the layer stack down to `base`, threading the result back.

    Each layer is advanced to its `yield op`, and the op it yields (possibly rewritten) is
    forwarded inward. A layer that returns before yielding (a refusal) answers the op itself.
    What comes back from the layers below decides what this layer sees:

    ```
    the layers below
    ├── return a result ────────────── sent into this layer's `yield`
    ├── raise a member of `escapes` ── this layer is closed under it, and it goes on out
    ├── raise another `Exception` ──── thrown into this layer's `yield`, which decides (`retry`)
    └── raise a `BaseException` ────── this layer is closed under it, and it goes on out
    ```

    `escapes` holds the engine's signals: a park, a cancel, a run already failed.
    `forwarding(level, op)` hears each op as it enters a level, `level` being the layers below it.
    """
    if forwarding is not None:
        forwarding(len(layers), op)
    if not layers:
        return base(op)
    head, *rest = layers
    gen = head(op)
    with closed_after(gen):
        try:
            inner = gen.send(None)  # advance to the first `yield op`
        except StopIteration as done:  # the layer returned without yielding (e.g. a refusal)
            return done.value
        while True:
            try:
                result = drive_through(rest, inner, base, escapes=escapes, forwarding=forwarding)
            except escapes:
                raise
            except Exception as exc:  # propagate into the layer's yield; the layer decides
                try:
                    inner = gen.throw(exc)
                except StopIteration as done:
                    return done.value
            else:
                try:
                    inner = gen.send(result)
                except StopIteration as done:
                    return done.value


# --- The two combinators ----------------------------------------------------


@runtime_checkable
class Interpreter(Protocol):
    """Anything with `run(op) -> Any` — the domain-seam base contract (structural,
    so `effective.layers` needn't import the handler package)."""

    def run(self, op: Any) -> Any: ...


@dataclass
class _DomainStack:
    layers: tuple[DomainLayer[Any], ...]
    base: Interpreter

    def run(self, op: DomainOp[Any]) -> Any:
        return drive_through(self.layers, op, self.base.run)


def check_seam(seam: str, layers: Sequence[Any]) -> None:
    """Refuse a layer built for the OTHER seam, at assembly time.

    **Installing a layer at the wrong seam is silent, and silent in the worst direction.** An
    op-layer composed into the domain stack is accepted, runs, and observes everything on a LIVE
    pass — so the run you would sanity-check it with looks perfect. On REPLAY it collects nothing:
    the domain call sits inside the checkpoint thunk and `ctx.step` returns the committed row
    without running it, while the op seam re-fires per op. Measured: op seam 2 entries, domain
    seam 0, same script.

    The marker has been stamped by `@op_layer`/`@domain_layer` since they were written and read in
    exactly one place in the whole tree. `effective.tape.collecting_layer`'s docstring says "it
    MUST be an OP layer" and explains why at length, which is a sentence rather than a guard —
    and this repo's own rule is that documentation is not mitigation.

    Unmarked callables pass. A layer is an ordinary generator function and plenty are built
    without the decorator; this refuses a POSITIVE mismatch, not the absence of a claim."""
    for layer in layers:
        role = getattr(layer, "__effective_layer__", None)
        if role is not None and role != seam:
            name = getattr(layer, "__qualname__", repr(layer))
            raise TypeError(
                f"{name} is marked `@{role}_layer` and was installed at the {seam} seam. A layer "
                f"at the wrong seam is not an error at run time — it observes the live pass and "
                f"silently sees nothing on replay — so it is refused here instead."
            )


def compose_domain(layers: Sequence[DomainLayer[Any]], base: Interpreter) -> Interpreter:
    """Build the per-`DomainOp` onion. Returns a `run(op)` — drops into
    `DurableHandler(domain=…)` unchanged."""
    check_seam("domain", layers)
    return _DomainStack(tuple(layers), base)


@runtime_checkable
class _LayeredHandler(Protocol):
    op_layers: tuple[OpLayer[Any], ...]


def compose_ops[H: _LayeredHandler](layers: Sequence[OpLayer[Any]], base: H) -> H:
    """Configure an existing handler's op-layer stack and return it.

    The op seam's combination point is the handler's drive loop (it owns
    generator-driving, parking, and the trace), so a handler reads `self.op_layers`
    there rather than being wrapped from outside. This is sugar over
    `Handler(..., op_layers=[...])`.
    """
    base.op_layers = tuple(layers)
    return base


# --- The canonical op-seam layer: retry (multi-yield is the proof) ----------


class TransientError(Exception):
    """A failure a layer is willing to retry (the LangGraph 'transient' class)."""


class RateLimited(TransientError):
    """A provider rate limit (HTTP 429) — transient, but **only retryable with a backoff**.

    A subclass rather than a peer because it *is* transient; what differs is the safe response.
    Re-firing immediately on a 429 makes the limit worse, so `retry_domain` **re-raises it
    unless a `backoff` is configured**, so the dangerous combination is unrepresentable."""


def retry(attempts: int = 2, on: type[Exception] = TransientError) -> OpLayer[Any]:
    """An `@op_layer` that re-forwards its op on a transient failure.

    `attempts` retries → `attempts + 1` total tries. The `for` around `yield` is the
    second yield `@contextmanager` forbids and the harness wants.

    A re-forward carries the op's placement, so on the durable engines a retried success is
    checkpointed where its first try would have been, and a crash replay is served it. A
    re-forward is the same op object yielded again; a layer that rebuilds the op before
    re-yielding forwards it afresh.
    `retry_domain` retries below the seam instead, inside one op interpretation, where the trace
    and the trip see one call.
    """

    @op_layer
    def run(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
        last: Exception | None = None
        for _ in range(attempts + 1):
            try:
                return (yield op)
            except on as exc:
                last = exc
        assert last is not None  # range(attempts + 1) ran at least once
        raise last

    return run


def wait_before_retry(
    exc: Exception, attempt: int, attempts: int, backoff: Callable[[int], float] | None
) -> float | None:
    """The retry POLICY as one pure function: seconds to wait before re-firing, or `None` to stop.

    Every reason to give up collapses to one answer, so the layer driving it stays mechanical — it
    never asks *why* a retry is refused, only *whether* and *how long*. Lifted out of the closure
    so the rule can be read and tested without driving a generator (`test_transient_mapping.py`),
    which is the point of naming a rule rather than inlining it into an `except`."""
    if attempt >= attempts:
        return None  # attempts exhausted
    match exc, backoff:
        case RateLimited(), None:
            return None  # a 429 with nothing to wait with: re-firing at once worsens it
        case _, None:
            return 0.0  # immediate re-fire (the default)
        case _, wait:
            return wait(attempt)


def retry_domain(
    attempts: int = 2,
    on: type[Exception] = TransientError,
    *,
    backoff: Callable[[int], float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> DomainLayer[Any]:
    """`retry`'s domain-seam twin and `serve`'s first Service. Re-invokes the DOMAIN CALL on a
    transient failure, *inside one op interpretation*: invisible to checkpoints, the trace, the
    trip and the ledger. This is retry below the seam: because a replayed op never
    reaches the domain, it fires only on live calls, and because it lives under one `ctx.step`
    one checkpoint holds the call however many tries it took.

    Deliberately a separate `@domain_layer`-decorated def, NOT a factored shared body: the
    layer-authority lint keys on the decorator at the def site, and a domain-seam retry must
    never gain op authority. The 6-line twin is guarded by a cross-seam conformance test.
    `attempts` retries → `attempts + 1` total tries.

    **Backoff and 429.** By default there is no delay between attempts, and a `RateLimited`
    (429) is therefore **re-raised unretried**: an immediate re-fire on a rate limit makes the
    limit worse. Pass `backoff`, an `attempt_number -> seconds` function such as
    `lambda n: 0.5 * 2**n`, to enable both: the
    layer sleeps between attempts and 429s become retryable. The two are one switch precisely
    so "retry a 429 immediately" cannot be spelled.

    **Why sleeping here is legal, and its boundary.** A domain layer runs *below* the
    checkpoint, live-only — never on replay — so a delay cannot affect determinism. But it
    does hold a durable worker slot, so keep backoffs short (seconds). A wait long enough to
    matter belongs on the op seam as a `SleepUntil`, which releases the worker (the `layers`
    header's third question)."""

    @domain_layer
    def run(op: DomainOp[Any]) -> Generator[DomainOp[Any], Any, Any]:
        for attempt in range(attempts + 1):
            try:
                return (yield op)
            except on as exc:
                if (delay := wait_before_retry(exc, attempt, attempts, backoff)) is None:
                    raise
                if delay:
                    sleep(delay)

    return run
