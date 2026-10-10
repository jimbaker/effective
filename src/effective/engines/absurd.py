"""The Absurd engine: the durable ctx `DurableHandler` drives on Absurd and Postgres.

`SdkCtx` adapts the Absurd SDK's `TaskContext` to the handler's ctx protocol, and
`ConcurrentAbsurdCtx` lets gather branches run concurrently over one task connection. This is the
one module that reaches past the SDK's public surface. The public surface cannot read or write a
checkpoint without advancing the SDK's per-name step counter, so the sites below use the private
read and write, the claimed task row, and the queue's tables, at the version pinned in
`infra/absurd/PIN.txt`. Each private name the module reads is pinned against the installed SDK, so
an upgrade that moves one fails by name.
"""

import json
import math
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import psycopg

from effective.handlers.base import Attempt
from effective.keys import Key
from effective.ops import WaitOutcome, settled_wait


def queue_table(kind: str, queue: str) -> str:
    """Absurd's table for one kind of row in one queue, for ``{…:i}`` to quote as an identifier.

    | kind | holds              |
    |------|--------------------|
    | `c`  | checkpoints        |
    | `e`  | events             |
    | `r`  | runs               |
    | `t`  | tasks              |
    | `w`  | waits              |

    The vendored schema spells every one of them `{kind}_{queue}` (`infra/absurd/PIN.txt`), so
    the convention is written once here rather than at each site that reaches past the SDK.
    """
    return f"{kind}_{queue}"


def sdk_peek_step(sdk: Any, name: Key) -> tuple[bool, Any]:
    """Read a checkpoint through the SDK's cache and store, without its occurrence counter."""
    return _sdk_lookup(sdk, name.stored())


def _sdk_lookup(sdk: Any, checkpoint_name: str) -> tuple[bool, Any]:
    """The SDK's own checkpoint read, which `begin_step` makes after counting the name."""
    from absurd_sdk import _CHECKPOINT_NOT_FOUND

    raw = sdk._lookup_checkpoint(checkpoint_name)
    return (False, None) if raw is _CHECKPOINT_NOT_FOUND else (True, raw)


def sdk_settle(sdk: Any, name: Key, value: Any) -> Any:
    """Write `value` under `name` unless the store holds one, then return what the store holds.

    The write is `set_task_checkpoint_state`, fenced by the run's attempt: a run another worker's
    claim has failed raises there. The read-back goes to the store, never the SDK's cache, which
    `_persist_checkpoint` fills with the value it was handed."""
    found, stored = sdk_peek_step(sdk, name)
    if found:
        return stored
    sdk._persist_checkpoint(name.stored(), value)
    queue, task_id, checkpoint = sdk._queue_name, sdk._task["task_id"], name.stored()
    row = (
        sdk._conn.cursor()
        .execute(
            t"SELECT state FROM absurd.get_task_checkpoint_state({queue}, {task_id}, {checkpoint})"
        )
        .fetchone()
    )
    if row is None:
        raise LookupError(f"checkpoint {name.stored()!r} was written and then not found")
    return row[0]


def sdk_await_until(sdk: Any, name: Key, deadline: float, decided: Key) -> WaitOutcome[Any]:
    """Wait on the event and the clock together through the SDK, and answer once.

    ====================================  =====================================================
    the wait                              answer
    ====================================  =====================================================
    is settled                            what it settled, whatever has since
    is open, the SDK returns a payload    ``Arrived``, settled now
    is open, the SDK times out            ``Expired``, settled now
    is open, neither yet                  ``SuspendTask``, and the claim that follows decides
    ====================================  =====================================================

    The SDK takes a RELATIVE timeout and checkpoints an arrival; an expiry it records nowhere, so
    the expiry is settled here, under the same name the embedded engine settles, and a replay
    reads row one rather than waiting again.

    **The expiry also SPENDS the run's wake.** The SDK
    clears ``wake_event`` on the task dict it holds and raises before the statement that would
    clear the column, so the column goes on naming an event this run has already been woken for.
    A later ask of the same name then reaches ``absurd.await_event``'s resumed-due-to-timeout arm,
    which is keyed on the RUN's wake and answers `(false, null)`, and a workflow
    is told an arrival happened that nobody emitted. Settling first and clearing second is the
    order that survives a crash between them: the outcome is durable before the fact it rests on
    is discarded.

    The SDK's timeout is whole seconds relative to the claim, so the park it registers names an
    instant up to a second later than the workflow asked for. ``sdk_pin_the_park`` writes the
    deadline over it, **inside the transaction that registers the park**, because between the two
    the wait is live on an instant the workflow never named: an emit inside that interval takes
    the wait, writes the SDK's arrival checkpoint and deletes the row, and no later claim asks
    again. The interval is bounded by a worker's progress, not by a commit, so a pause or a death
    inside it is enough.

    Row two before row three is the reference's order and a park is where this one sits, so the
    SuspendTask is raised AFTER the commit rather than through it, which would roll the park back.
    """
    import absurd_sdk

    found, stored = sdk_peek_step(sdk, decided)
    if found:
        return settled_wait(stored)
    remaining = math.ceil(max(0.0, deadline - time.time()))
    payload, parked = None, False
    try:
        # ONE transaction: the park and the deadline it ends at commit together or not at all.
        with sdk._conn.transaction():
            try:
                payload = sdk.await_event(name.stored(), timeout=remaining)
            except absurd_sdk.SuspendTask:
                sdk_pin_the_park(sdk, name, deadline)
                parked = True
    except absurd_sdk.TimeoutError:
        expired = settled_wait(sdk_settle(sdk, decided, ["expired"]))
        sdk_spend_wake(sdk, name)
        return expired
    if parked:
        raise absurd_sdk.SuspendTask
    return settled_wait(sdk_settle(sdk, decided, ["arrived", payload]))


def sdk_pin_the_park(sdk: Any, name: Key, deadline: float) -> None:
    """Write the workflow's deadline over the one ``absurd.await_event`` just registered.

    The SDK takes whole seconds, relative, so the park it registers ends at
    ``claim + ceil(deadline - claim)``. Two readers take that instant for the deadline and each
    turns a rounding into an answer.

    | the reader           | what it does with the instant                             |
    |----------------------|-----------------------------------------------------------|
    | ``absurd.emit_event``| deletes the waits it has passed, and WAKES the rest        |
    | ``absurd.claim_task``| claims the run when ``available_at`` arrives              |

    So an event landing inside the rounding answers a wait the embedded engine has expired, and
    a wait nothing answers sleeps on past the deadline it named. Both columns carry the same
    instant and both are written here.

    Written as an absolute ``timestamptz`` from the epoch seconds the workflow named, so the
    comparison the engine makes is between two readings of ITS clock; what is left of the old
    window is the gap between that clock and the caller's.

    **The caller runs this inside the park's own transaction**, and both reasons are the same
    interval read two ways. A wait live on the rounded instant takes an emit the deadline has
    passed, and nothing asks again, since `absurd.emit_event` writes the SDK's arrival checkpoint
    as it deletes the row. And `run_id` names a RUN, which outlives the claim that parked it: a
    later claim on the same run can settle that wait and park a second one on the same event, and
    this write would reach that one instead. Atomicity answers both, so the columns are addressed
    by the run alone.
    """
    run_id, event = sdk._task["run_id"], name.stored()
    w_tbl = queue_table("w", sdk._queue_name)
    r_tbl = queue_table("r", sdk._queue_name)
    cursor = sdk._conn.cursor()
    cursor.execute(
        t"UPDATE absurd.{w_tbl:i} SET timeout_at = to_timestamp({deadline}) "
        t"WHERE run_id = {run_id} AND event_name = {event}"
    )
    cursor.execute(
        t"UPDATE absurd.{r_tbl:i} SET available_at = to_timestamp({deadline}) "
        t"WHERE run_id = {run_id} AND wake_event = {event}"
    )


def sdk_spend_wake(sdk: Any, name: Key) -> None:
    """Clear the run's wake where it still names ``name``, which the SDK does only in memory.

    ``absurd.await_event`` reads ``r_{queue}.wake_event`` to decide whether this run resumed on
    its timeout, and that column belongs to the RUN while the question belongs to one ask. Left
    naming a spent event it answers every later ask of the name, with the null payload that arm
    returns.

    The name in the WHERE clause is defensive and no test needs it: the SDK raises its timeout
    only where ``wake_event`` already equals this name, and a run parks at most once per claim,
    so the column cannot be naming anything else by the time this runs. It costs one predicate to
    keep that reasoning out of the write."""
    run_id, event = sdk._task["run_id"], name.stored()
    r_tbl = queue_table("r", sdk._queue_name)
    sdk._conn.cursor().execute(
        t"UPDATE absurd.{r_tbl:i} SET wake_event = NULL "
        t"WHERE run_id = {run_id} AND wake_event = {event}"
    )
    sdk._task["wake_event"] = None


class ClaimSteps:
    """The SDK's step handles for one claim, one per checkpoint name.

    `begin_step` counts the names it is handed and checkpoints a repeat as `name#2`, so each name
    is begun once per claim and its handle reused: a step forwarded again lands at the name it was
    given. A handle is kept as completed because `complete_step` leaves `done` unset and
    completing it again would overwrite the checkpoint.

    A done handle's value is kept as JSON text, written before the value is persisted, and each
    hit decodes it, as a read from the store would. So a value with no JSON form fails before
    anything is written, and a caller that changes a value it was handed changes nothing a later
    forward is served.

    One per SDK ctx, which is one per claim, so every adapter over a claim shares it. It holds no
    reference to the SDK ctx, which keys it weakly, so the entry ends with the claim."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}
        self._values: dict[str, str] = {}

    @classmethod
    def of(cls, sdk: Any) -> ClaimSteps:
        with _CLAIM_STEPS_LOCK:
            if (steps := _CLAIM_STEPS.get(sdk)) is None:
                steps = _CLAIM_STEPS[sdk] = cls()
            return steps

    def begin(self, sdk: Any, name: Key) -> Any:
        stored = name.stored()
        if (handle := self._handles.get(stored)) is None:
            handle = sdk.begin_step(stored)
            if handle.checkpoint_name != stored:
                raise RuntimeError(
                    f"the SDK checkpoints {stored!r} as {handle.checkpoint_name!r}: a step of "
                    "this name began in this claim outside this ctx"
                )
            self._handles[stored] = handle
            if handle.done:
                self._values[stored] = json.dumps(handle.state)
        return handle

    def complete(self, sdk: Any, handle: Any, value: Any) -> Any:
        text = json.dumps(value)
        # The SDK caches what it persists, so it gets its own copy and the caller keeps theirs.
        sdk.complete_step(handle, json.loads(text))
        self._values[handle.checkpoint_name] = text
        self._handles[handle.checkpoint_name] = replace(handle, done=True, state=None)
        return value

    def served(self, handle: Any) -> Any:
        """A done handle's value, decoded afresh."""
        return json.loads(self._values[handle.checkpoint_name])


_CLAIM_STEPS: weakref.WeakKeyDictionary[Any, ClaimSteps] = weakref.WeakKeyDictionary()
_CLAIM_STEPS_LOCK = threading.Lock()


class SdkCtx:
    """The adapter over the Absurd SDK's own ctx: the one place a `Key` becomes a `str` on the
    durable path.

    `TaskContext` is a structural Protocol, and the SDK's ctx satisfies it by duck-typing. But the
    SDK is a FOREIGN implementation that takes `str` and does string work on it: it composes
    `f"{name}#{count}"` for a repeated step name and `$awaitEvent:{name}` for an await, so a `Key`
    handed through renders its repr into a durable name. A psycopg dumper fixes the *binding* and
    cannot fix that, because the corruption happens in Python before any parameter is bound.

    So the boundary is named. This is the per-engine adapter pattern the bridges and
    `spawning.deliver` already use: our protocol says what OUR ctxs promise, and one small class
    translates at the edge of somebody else's library. `DurableHandler` wraps a raw SDK ctx
    automatically, so callers keep passing `ctx` straight through.

    **ARITY is part of the boundary too.** The SDK's `sleep_until(step_name, wake_at)` takes TWO
    arguments, while our `TaskContext` protocol, written to the SQLite shape, takes one; a
    one-argument forward makes every top-level durable sleep on Absurd raise `TypeError`, retry
    to death, and fail the task. A durable sleep *inside a gather* routes to `_branch_sleep`, a
    pure clock compare that calls neither ctx, so only a top-level sleep reaches this method.
    `ConcurrentAbsurdCtx.sleep_until` makes the same call.
    """

    speaks_sdk_text = True
    """Declares that THIS class converts `Key` -> `str` for the vendored SDK.

    Read by `_adapt_ctx`, which stops descending here: a second converter beneath one of these
    receives a `str` and fails on `.stored()`. Declared at the definition site, so a new adapter
    announces itself and the walk stays a decision table over a property of the class."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._steps = ClaimSteps.of(ctx)

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        handle = self._steps.begin(self._ctx, name)
        if handle.done:
            return self._steps.served(handle)
        return self._steps.complete(self._ctx, handle, thunk())

    def await_event(self, name: Key, /) -> Any:
        # `.stored()` at the SDK boundary, as `step` does: below this line is the vendored SDK,
        # which speaks text.
        return self._ctx.await_event(name.stored())

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> WaitOutcome[Any]:
        return sdk_await_until(self._ctx, name, deadline, decided)

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        # Two args: the SDK wants an explicit step name (its wake-time checkpoint) first, and
        # `.stored()` at that boundary, as `step` and `await_event` do. Below this line is the
        # vendored SDK, which f-strings what it is given; a `Key` handed through would render its
        # repr into a durable name.
        return self._ctx.sleep_until(name.stored(), when)

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        return sdk_peek_step(self._ctx, name)

    def settle(self, name: Key, value: Any, /) -> Any:
        return sdk_settle(self._ctx, name, value)

    @property
    def attempt(self) -> Attempt:
        return sdk_attempt(self._ctx)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


@dataclass(frozen=True, slots=True)
class SdkClaim:
    """The run an SDK ctx claimed, and the connection and queue it claimed it on."""

    task_id: str
    run_id: str
    attempt: Attempt
    params: Mapping[str, Any]
    conn: psycopg.Connection[Any]
    queue: str


def sdk_claim(ctx: Any) -> SdkClaim:
    """Read the claim the SDK keeps on its private `_task`, `_conn` and `_queue_name`, which no
    public surface carries."""
    claimed = ctx._task
    return SdkClaim(
        task_id=claimed["task_id"],
        run_id=claimed["run_id"],
        attempt=Attempt(number=claimed["attempt"], limit=claimed["max_attempts"]),
        params=claimed["params"],
        conn=ctx._conn,
        queue=ctx._queue_name,
    )


def sdk_signals() -> tuple[type[Exception], ...]:
    """The SDK's exceptions that end a walk: a park, a cancel, and a run already failed."""
    try:
        import absurd_sdk
    except ImportError:  # pragma: no cover - the SDK is a declared dependency
        return ()

    return (absurd_sdk.SuspendTask, absurd_sdk.CancelledTask, absurd_sdk.FailedTask)


def sdk_attempt(ctx: Any) -> Attempt:
    """The attempt an SDK ctx claimed its run as."""
    return sdk_claim(ctx).attempt


def _adapt_ctx(ctx: Any) -> Any:
    """Wrap a raw Absurd SDK ctx in `SdkCtx`, **at whatever depth it sits**.

    An `isinstance` against the SDK's class rather than a duck-check: the question is "is this
    somebody else's ctx?", and the SDK type asks it directly. Imported lazily, matching the
    module's existing SDK touch, so a SQLite-only process never pays for it.

    **It scans the whole wrapper stack.** A caller may build a wrapper stack over a raw SDK ctx
    and pass the STACK: the outermost object is one of ours, and the raw SDK ctx underneath would
    receive our `Key` step names, which the vendored SDK f-strings into
    `Key(_value='step:extract_ticket', _scope=None)#2`, a REPR committed to a durable name. A
    SQLite ctx needs no adapter, so a SQLite run stays green whether or not the stack is adapted.

    So this is a **decision table over the ctx chain**: four arms, exhausted by a wildcard, so
    "what happens to a node I did not think of" has a written answer instead of a fall-through.

    **The table is bounded by what it scans.** It descends by the `_ctx` attribute, read from the
    INSTANCE dict so a wrapper's `__getattr__` delegation cannot make it look one level too deep,
    and via `getattr(node, "__dict__", {})` rather than `vars()`, because `vars()` RAISES on an
    object without a `__dict__` and a guard on the wildcard arm that can raise is not a total
    table. Every wrapper here uses that attribute name; one that named its inner ctx something
    else would be invisible, so the backstop is a measurement over the STORE
    (`tests/test_durable_name_conformance.py`), which no attribute name bounds."""
    try:
        from absurd_sdk import TaskContext as SdkTaskContext
    except ImportError:  # pragma: no cover - the SDK is a declared dependency
        return ctx

    def adapt(node: Any) -> Any:
        """The decision table, one arm per kind of node, total by its wildcard.

        Four cases exhaust what a ctx can be at any depth, and reading them in order IS the
        argument: stop at anything that already speaks SDK text, adapt the foreign ctx, descend
        through one of ours, and leave anything else alone.

        **The first arm reads a declared property**, `speaks_sdk_text`, which `SdkCtx` and
        `ConcurrentAbsurdCtx` set at their own definitions. Descending past an adapter that
        converts to text itself converts twice and fails with `'str' object has no attribute
        'stored'`. A declaration is bounded by what announces itself, and a new adapter that
        forgets to declare fails loudly on its first duplicate step."""
        match node:
            case _ if getattr(node, "speaks_sdk_text", False):
                return node  # already an SDK adapter: a second one beneath it would convert twice
            case SdkTaskContext():
                return SdkCtx(node)  # the foreign ctx: THIS is the boundary
            case _ if (inner := getattr(node, "__dict__", {}).get("_ctx")) is not None:
                node._ctx = adapt(inner)  # one of ours: rebuild the link beneath it
                return node
            case _:
                return node  # nothing foreign beneath (a SQLite ctx, or a leaf of our own)

    return adapt(ctx)


class ConcurrentAbsurdCtx:
    """Wraps an Absurd ctx so ``gather`` branches run concurrently: **parallel tools,
    serialized writes**.

    Uses the SDK's *public* ``begin_step``/``complete_step`` split: the checkpoint read and
    the checkpoint write are taken under ``write_lock``, but the step's ``thunk`` (the
    tool/domain work) runs **lock-free**, so branch tools overlap while their commits
    serialize on the one task connection. No extra connections. Every private read goes
    through the module's ``sdk_*`` helpers or ``peek_event``, under the write lock wherever
    it touches the connection. The single
    coarse lock is the same simplifying bet as CPython's GIL: it is held only around the
    cheap write and released during the I/O (the LLM/tool call) that dominates the latency,
    so it is near-free for the I/O-bound workloads gather fans out. (For *write*-bound fan-out
    it would serialize, like the GIL on CPU-bound threads; that case wants a connection per
    branch.) ``concurrent_safe`` tells the handler it may fan branches out.
    """

    speaks_sdk_text = True
    """This class converts `Key` -> `str` for the SDK itself (see its `.stored()` calls below), so
    `_adapt_ctx` must NOT insert an `SdkCtx` beneath it: the double conversion fails with
    `'str' object has no attribute 'stored'`."""

    def __init__(self, ctx: Any, write_lock: threading.Lock | None = None) -> None:
        self._ctx = ctx
        self.write_lock = write_lock or threading.Lock()
        self.concurrent_safe = True
        self._steps = ClaimSteps.of(ctx)

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        with self.write_lock:
            handle = self._steps.begin(self._ctx, name)
        if handle.done:
            return self._steps.served(handle)
        rv = thunk()  # lock-free: tool work overlaps across branches
        with self.write_lock:
            return self._steps.complete(self._ctx, handle, rv)

    def await_event(self, name: Key, /) -> Any:
        with self.write_lock:
            return self._ctx.await_event(name.stored())

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> WaitOutcome[Any]:
        with self.write_lock:
            return sdk_await_until(self._ctx, name, deadline, decided)

    def peek_event(self, name: Key, /) -> tuple[bool, Any]:
        """Whether `name` has arrived, without suspending: the gather branch's probe.

        The public `await_event` cannot serve: it counts the name, and an unsatisfied await
        marks the run sleeping and allows one await per run. So this reads the await's
        checkpoint, then the queue's events table, and freezes a delivered payload into the
        checkpoint the SDK's own `await_event` would commit, so replay binds it even if the
        event is emitted again.
        """
        # The SDK's await checkpoint name, in its `$`-prefixed grammar: `SdkCtx.await_event`
        # hands the SDK `name.stored()`, so the name it commits is built from that text.
        step_name = f"$awaitEvent:{name.stored()}"
        with self.write_lock:
            found, raw = _sdk_lookup(self._ctx, step_name)
            if found:
                return True, raw
            # Delivered but never awaited: read the queue's events table (emit_event
            # upserts a non-NULL payload; NULL is the "not emitted" sentinel row).
            e_tbl = queue_table("e", self._ctx._queue_name)
            cursor = self._ctx._conn.cursor()
            cursor.execute(
                t"SELECT payload FROM absurd.{e_tbl:i} "
                t"WHERE event_name = {name} AND payload IS NOT NULL"
            )
            row = cursor.fetchone()
            if row is None:
                return False, None
            payload = row[0]
            self._ctx._persist_checkpoint(step_name, payload)  # freeze for replay
            return True, payload

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        # The SDK's sleep_until takes an explicit step name first (its wake-time checkpoint);
        # `.stored()` at that boundary, as in `SdkCtx.sleep_until`.
        with self.write_lock:
            return self._ctx.sleep_until(name.stored(), when)

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        with self.write_lock:
            return sdk_peek_step(self._ctx, name)

    def settle(self, name: Key, value: Any, /) -> Any:
        with self.write_lock:
            return sdk_settle(self._ctx, name, value)

    @property
    def attempt(self) -> Attempt:
        return sdk_attempt(self._ctx)

    def emit_event(self, name: str, payload: Any, /) -> None:
        """Emit an event from inside a running task, under the write lock like every other
        write on this connection (a child answering its parent while its own gather branches
        commit must not race them).

        No destination argument, on either engine: events are queue-global, keyed by name alone,
        so the name IS the address. Both engines' signatures are `(name, payload)`, so a green
        SQLite test cannot assert an isolation the deployed engine lacks, and
        `effective.spawning.deliver` is a single call rather than a branch."""
        with self.write_lock:
            self._ctx.emit_event(name, payload)

    def repark(self, name: str, /) -> None:
        """Park so the task re-queues ~immediately, burning no attempt (the
        optional capability in ``TaskContext``'s docstring). Public SDK surface
        only: ``sleep_until`` checkpoints the wake time under ``name``,
        schedules the run, and raises ``SuspendTask``, the *sleeping* park, so
        the attempt counter is untouched.

        The epsilon is required: a ``wake_at <= now`` makes the
        SDK return WITHOUT parking (its stale/past-time path), and the caller
        would fall through to ``GatherWakeRace``. The same return-not-park
        happens if ``name``'s checkpoint already exists with a past wake time,
        which is why the caller keeps its loud fallback, and why ``name`` must
        be deterministic, UNFORGEABLE by author step names, and touched at most
        once per name (see ``_join``: the condition-qualified name guarantees
        this, keeping the occurrence counters clean: ``name`` occupies its
        own counter key, never shifting a ``$awaitEvent:`` or step name)."""
        with self.write_lock:
            self._ctx.sleep_until(name, time.time() + _REPARK_EPSILON)

    def __getattr__(self, attr: str) -> Any:
        # Read-only attribute delegation (e.g. concurrent_safe, task_id). Writes go through the
        # locked methods above, and nothing on the gather path reaches the ctx by another route,
        # so this delegation never bypasses the write lock.
        return getattr(self._ctx, attr)


# Strictly positive, and generously so: the SDK checks its clock only AFTER two
# checkpoint round-trips (lookup + persist), so the epsilon must outlive that SQL
# latency or sleep_until returns without parking and the caller falls back to
# GatherWakeRace, which is loud and safe and costs the attempt the repark exists to save.
# 250ms outlives that latency on a localhost and a managed Postgres, at a sub-second delay on a
# rare path.
_REPARK_EPSILON = 0.25
