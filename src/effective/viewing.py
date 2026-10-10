"""`ViewingCtx` — a durable ctx that can be READ through and never written to.

A viewer wants what a run DECIDED, and a decision is not on the tape. `Checkpoint` is `(key,
state)`, so verdicts and prompts are absent by construction — and that is the substrate working
rather than a gap, because both are a **function of the tape**: re-run the deterministic loop with
the recorded results replayed and it rebuilds them itself. Replay is therefore the read, and on a
COMPLETE run it adds nothing: no checkpoint key, no ledger row
(`test_conformance.py::test_a_viewer_replays_a_complete_run_without_touching_it`).

**On an INCOMPLETE run it is not, and the exposure is what this module closes.** A replay driven
past the last recorded op runs the ops that follow and commits their checkpoints, because
`ctx.step`'s contract is to record a thunk it had to run. Bound to a task the process does not own,
that advances somebody else's run: the checkpoints commit, a ledger-less handler suppresses the
canonical appends, and when the real run resumes it finds the checkpoints present, honors
at-most-once, skips the thunks, and the appends never happen. Measured end to end on SQLite; the
run completed with zero ledger rows.

**Both engines are exposed and neither refuses it.** SQLite hands out a public
`SqliteTaskContext(conn, task_id, lock)`. Absurd's SQL guard
(`absurd.set_task_checkpoint_state`) raises on a run id absent from `r_{queue}`, a cancelled task
and a run already failed, and silently skips a stale attempt — **claim ownership is among none of
them**, so a viewer reusing the real `owner_run_id`, which any reader obtains with one `SELECT`,
writes successfully against a completed run.

**So the guard is here, in one wrapper over both engines.** One wrapper serves both because
**both engines are read-first**: `SqliteTaskContext.step` returns a found checkpoint without
writing, and the SDK's `begin_step`/`complete_step` pair does the same; only a MISS persists. A
ctx that refuses the miss therefore cannot write on either engine, needs no engine change, and
leaves the co-versioned `absurd.sql` alone (the pin at `infra/absurd/PIN.txt` is what a SQL
edit would break).

The precondition this makes enforceable:

> **A replay must not outrun the tape's frontier.** A PARKED run satisfies it without being
> complete — it reaches its await and stops. A crashed or mid-flight run does not.

`OutranTheTape` is that sentence as a raised exception, and it is a *viewer* limit rather than a
defect: reaching it means the answer the pane wanted is not derivable from what has been recorded
yet, which is a thing to render, not to fix.

**The sibling module is `effective.steering`**, and the pair is worth reading together: both wrap a
`TaskContext` and intercept `step`, both count occurrences the way `SeedingCtx` documents, and both
leave the engines untouched. `SteeringCtx` substitutes a value the run has not computed;
this one refuses to let a value be computed at all.
"""

from collections.abc import Callable, Collection
from datetime import datetime
from typing import Any, Never

from effective.handlers.base import TaskContext, framed_refusal_record, served_refusal
from effective.keys import Key
from effective.ops import WaitOutcome, settled_wait

DELEGATED: frozenset[str] = frozenset({"peek_event", "concurrent_safe", "event_rename", "task_id"})
"""Ctx members a viewer may reach unchanged — every one a READ.

An ALLOWLIST, and the shape is the argument. The obvious spelling is `__getattr__` delegating
everything and a refusal list for the writers, which is a denylist bounded by what it enumerates:
a capability added to `TaskContext` later would be delegated silently, and the viewer's one
guarantee would be quietly false. Naming what may pass makes a new member arrive as a loud
`AttributeError` naming this set, which is the failure a reader can act on.

`peek_event`'s contract is what makes it safe to include rather than merely convenient: it
promises `(True, payload)` iff an `await_event` would return right now, *with no side effects* —
no wait registration, no run-state change, no occurrence-counter burn. That is a viewer's read,
specified by somebody solving a different problem."""


class ViewingLimit(Exception):
    """A viewer reached the edge of what the record supports. Never a corruption — the base type
    exists so a pane can catch both arms and render, since neither is an error in the run."""


class OutranTheTape(ViewingLimit):
    """The replay reached an op the tape has no record of, and stopping is the whole point.

    Continuing would run the op and commit its checkpoint into a run this process does not own.
    Carries the occurrence-resolved key so a pane can say where the record ends."""

    def __init__(self, key: Key) -> None:
        super().__init__(
            f"the tape has no record of {key.display()!r}: a replay must not outrun the "
            f"tape's frontier, so this viewer stops rather than committing the op"
        )
        self.key = key


class ReachedThePark(ViewingLimit):
    """The run is SUSPENDED here — an unanswered await, or a sleep that has not come due.

    Distinct from `OutranTheTape` because it is the ordinary end of a healthy parked run rather
    than the edge of a partial record: the pane renders *"parked here"*, and a viewer that reaches
    it has read everything there is."""

    def __init__(self, name: Key) -> None:
        super().__init__(f"the run is parked at {name.display()!r}")
        self.name = name


class ViewingCtx:
    """A `TaskContext` that serves what was recorded and refuses to record anything.

    `tape` is the set of committed keys, occurrence-resolved, exactly as a reader returns them
    (`{c.key for c in read_sqlite_task(...)}` or its Absurd twin). `Key` rather than `str` on
    purpose: this is a durability guard, not a projection, so it compares the identity the engines
    bind to rather than the text a renderer draws. `checkpoints.keys()` is the projection exit and
    is the wrong door for this.

    Stack it where a driver would take the real ctx::

        handler = DurableHandler(ViewingCtx(ctx, tape), ledger=None)

    `ledger=None` stays right and stays necessary — it is the durable no-commit mode, not a test
    affordance — but it was never sufficient, because it suppresses the canonical append and says
    nothing about the checkpoints. This is the half that was missing.
    """

    def __init__(self, ctx: TaskContext, tape: Collection[Key]) -> None:
        self._ctx = ctx
        self._tape = frozenset(tape)
        self._reached: list[Key] = []

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        """Serve a recorded step; refuse one the tape does not have.

        | the tape holds                     | this                                          |
        |------------------------------------|-----------------------------------------------|
        | the step's checkpoint              | the inner ctx returns it without the thunk    |
        | a race branch's served refusal     | the inner ctx runs the thunk, which raises the |
        |                                    | refusal, and writes nothing                   |
        | neither                            | raises before the inner ctx is reached        |
        """
        self._reach(name)
        return self._ctx.step(name, thunk)

    def _reach(self, name: Key) -> None:
        """Note what the tape holds for `name`, or raise `OutranTheTape`."""
        if name in self._tape:
            self._reached.append(name)
            return
        if (record := framed_refusal_record(name)) in self._tape and served_refusal(
            self.peek_step(record)[1]
        ):
            # A race branch's recorded refusal: its thunk raises it and writes nothing, and the
            # walk places the next ask of this name as the next occurrence, with its own record.
            self._reached.append(record)
            return
        raise OutranTheTape(name)

    def await_event(self, name: Key, /) -> Any:
        """Return an already-delivered payload; otherwise report the park.

        PEEKED rather than delegated, and the difference is the whole reason this method exists.
        A real `await_event` on an undelivered name suspends by raising the engine's own signal —
        `sqlite._Suspend`, which is private to that module and has no Absurd twin — so a pane
        that let it escape would be catching one engine's internals to render a state both
        engines have. Peeking asks the same question with a public answer and no side effect.
        """
        peek = getattr(self._ctx, "peek_event", None)
        if not callable(peek):
            raise NotImplementedError(
                f"{type(self._ctx).__name__} has no `peek_event`, so this viewer cannot tell a "
                f"delivered event from a park without suspending the run it is reading"
            )
        found, payload = peek(name)
        if not found:
            raise ReachedThePark(name)
        return payload

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> WaitOutcome[Any]:
        """Report what a bounded wait settled, or that the run is parked at it.

        A viewer reads; a wait DECIDES, and both of its endings are writes — the arrival and the
        expiry are each recorded the first time they are reached. So the record is read and
        nothing else: a settled wait renders its outcome, and an open one is a park like any
        other. Deciding here would mean a pane's clock, rather than the run's, chose the answer
        a later reader sees.

        Defined rather than delegated, for the reason `DELEGATED` gives: the allowlist names
        reads, `await_until` is not one, and a member left to `__getattr__` would be an
        `AttributeError` several frames into a pane while `_supports_await_until` reported the
        capability present.
        """
        peek = getattr(self._ctx, "peek_step", None)
        if not callable(peek):
            raise NotImplementedError(
                f"{type(self._ctx).__name__} has no `peek_step`, so this viewer cannot tell a "
                f"settled wait from a park without deciding {name.display()!r} itself"
            )
        settled, stored = peek(decided)
        if not settled:
            raise ReachedThePark(name)
        return settled_wait(stored)

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        """Three arms, because the engines disagree about whether a sleep has an identity.

        A SQLite sleep is nameless — the wake time lands on `tasks.available_at` and no checkpoint
        row is written — while Absurd's timer has a checkpoint of its own. So membership answers
        the question on one engine and is silent on the other, and neither fact alone is a rule:

        - recorded — a read on Absurd, and the ordinary path;
        - already due and unrecorded — SQLite's nameless sleep, which writes nothing and returns.
          Answered HERE rather than delegated, so the arm cannot write on the engine where the
          same shape is keyed;
        - not yet due — the run is suspended, which is a park by another name.
        """
        if name in self._tape:
            self._ctx.sleep_until(when, name=name)
            return
        if datetime.now(tz=when.tzinfo).timestamp() >= when.timestamp():
            return
        raise ReachedThePark(name)

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        """A race's read of its choice or of a loser's op: a read, so it is delegated."""
        inner: Any = self._ctx
        return inner.peek_step(name)

    def settle(self, name: Key, value: Any, /) -> Any:
        """Serve a race's settled checkpoint; refuse one the tape does not have, since settling it
        would write the choice of a run this viewer does not own."""
        if name not in self._tape:
            raise OutranTheTape(name)
        self._reached.append(name)
        return self.peek_step(name)[1]

    def emit_event(self, name: str, payload: Any, /) -> Never:
        """Always refuses. An emission is a durable write and the one a viewer is most likely to
        reach for, since answering a park looks like the natural thing to do from a pane holding a
        ctx. It is not this seam's to make: `effective.parked.answer` takes the park a reader
        READ, and that matters because a wake registration carries coordinates no caller can
        reconstruct — the enclosing frames and `Key.occurrence`'s `#N`. A ctx-side emit would
        compose a name instead of answering one."""
        raise ViewingLimit(
            f"a viewer may not emit {name!r}: answer a park with `effective.parked.answer`, "
            f"which settles the name a reader read rather than one composed here"
        )

    @property
    def reached(self) -> tuple[Key, ...]:
        """The recorded keys this viewer re-bound, in order — how far the read actually got.

        Empty after a replay that raised on its first op, which is the case worth distinguishing:
        a pane showing nothing because the run has no record yet reads identically to one showing
        nothing because the viewer was pointed at the wrong task."""
        return tuple(self._reached)

    def __getattr__(self, item: str) -> Any:
        """Delegate the reads in `DELEGATED`; refuse everything else by name.

        Reached only for attributes this class does not define, so the methods it intercepts never
        arrive here."""
        if item in DELEGATED:
            return getattr(self._ctx, item)
        raise AttributeError(
            f"{item!r} is not delegated by a viewing ctx: only {sorted(DELEGATED)} pass through, "
            f"because a member that is not a read may write to a run this process does not own"
        )
