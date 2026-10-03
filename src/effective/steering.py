"""`SteeringCtx`: a principal's answer, substituted at a live op.

A **steer** is a value a principal supplies for an op the run has not reached yet: *at this key,
answer V rather than calling the domain*. Steerability follows the bookkeeper: a checkpoint is
disposable, so substituting its value is a re-execution the substrate already supports; the ledger
is canonical, so steering it would rewrite history, and the append-only trigger refuses it.

**The mechanism is the one `SeedingCtx` uses**, and it is engine-agnostic because `TaskContext`
is the interface both engines implement: hand the inner ctx a thunk that returns the value, and
the steered answer commits by the ordinary path. Nothing here reaches into an engine's checkpoint
tables, the SDK's private writer, or the co-versioned `absurd.sql`: those tables are another
bookkeeper's *disposable* state, and writing into them from outside would treat disposable state
as an interface.

**The decision table differs from seeding; the mechanism is shared.** `SeedingCtx` matches over
`(phase, key ∈ seed)` and RAISES on `Live()` + seeded, because for a fork a substitution past the
fork point means the caller's `through` reached into the tail. A steer *is* that arm: there is no
prefix to replay and no phase to cross, so the table here is two arms over `key ∈ steers`.

**Occurrence-keyed, for the reason `SeedingCtx` documents**: a repeated op name is suffixed
`name#k` by the engines *below* this seam, so a steer authored from a tape carries the suffix while
the handler passes the bare `name` each time. This ctx counts the same way and passes the bare name
down, so the inner ctx's own counter remains the thing that names the checkpoint.

**Two things this seam cannot check, both for one reason: it sees `(name, thunk)` and never the
op.**

- *The declared `result_schema`.* The walk holds both the key and the op, so it may check a steer
  against the schema; this seam sees no op. Validation belongs where the steer is authored, which
  is also where the tape that names the key is.
- *The checkpoint's encoding.* `Steer.value` is the RAW checkpoint state, undecoded, exactly as a
  seed is. A metered `AskLLM` under `Contract.V1` folds usage from a `{result, usage}` envelope
  AFTER `ctx.step` returns, so a steer carrying a decoded result replays green while the meter
  under-derives.

Only `step` is steered; `await_event`, `sleep_until` and the optional capabilities delegate
unchanged, matching `SeedingCtx`'s scope. An await's answer arrives by `emit_event` and is
broadcast by name, so steering one is a different question with a different mechanism.

`applied` is what a caller appends to the canonical record. The value rides the checkpoint because
it is execution state; **who decided rides the ledger**, and a row saying a principal steered
appends to history without rewriting it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Never, assert_never

from effective.handlers.base import TaskContext, refuse_a_settled_checkpoint_under
from effective.keys import Key, Segment


@dataclass(frozen=True)
class Steer:
    """One principal's answer at one occurrence-resolved op key.

    `by` is a `Segment` rather than a `str` because it is bound for the identity axis — the ledger
    row that records the decision — and a `Segment` cannot carry the key delimiter, so a
    principal's name is a well-formed coordinate rather than one that might forge another. A
    foreign value digests (`handlers.base.digest_atom`) rather than being trusted.

    `value` is the RAW checkpoint state; see the module docstring for what that costs to get
    wrong.
    """

    value: Any
    by: Segment


class SteeringCtx:
    """A `TaskContext` that answers steered keys from a principal instead of from the domain.

    Stacked by a caller around the ctx a `DurableHandler` will drive, the way `run_fork` stacks
    `SeedingCtx`. `steers` is keyed by the OCCURRENCE-RESOLVED key (`name#k`), which is what a
    reader of a completed tape has.
    """

    def __init__(self, ctx: TaskContext, steers: Mapping[Key, Steer]) -> None:
        self._ctx = ctx
        self._steers = steers
        self._occurrences: dict[Key, int] = {}
        self._applied: dict[Key, Segment] = {}

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        """Answer `name` from `steers` when it is steered; otherwise run `thunk` as usual.

        Two arms over `key ∈ steers`, closed with `assert_never` — the substrate's states-as-data
        idiom, and what makes a third arm a type error rather than a silent fall-through.
        """
        count = self._occurrences.get(name, 0) + 1
        self._occurrences[name] = count
        key = name.occurrence(count)
        match self._steers.get(key):
            case None:
                return self._ctx.step(name, thunk)
            case Steer(value=value, by=by):
                self._applied[key] = by
                # The BARE `name`, not `key`: the inner ctx runs its own occurrence counter and is
                # what names the checkpoint. Handing it the suffixed key would suffix it twice, and
                # the resulting checkpoint would be one no replay ever binds to.
                return self._ctx.step(name, lambda: value)
            case unreachable:
                assert_never(
                    unreachable
                )  # pragma: no cover - `Mapping.get` returns `Steer | None`

    @property
    def applied(self) -> Mapping[Key, Segment]:
        """Which steers fired, and who authored each — a caller's input to the ledger append.

        In memory, like `SeedingCtx.consumed`, and for the same reason: a ctx is not a writer. What
        makes a steer *durable* is the row a caller appends from this, which is the half that
        distinguishes a steered answer from a computed one after the run is over.
        """
        return dict(self._applied)

    def unapplied(self) -> frozenset[Key]:
        """Steers the run never reached: the complement of `applied`, and the completeness check
        a caller asserts.

        A leftover means the steer's coordinate missed: a key from a tape the run no longer
        produces (the harness moved), or an occurrence the run did not reach this time. Empty is
        NECESSARY, NOT SUFFICIENT: it cannot see a step that ran live where a steer was intended
        but never authored, the quantifier gap `SeedingCtx` closes by refusing an unseeded prefix
        step.
        """
        return frozenset(self._steers) - frozenset(self._applied)

    def peek_step(self, name: Key, /) -> Never:
        refuse_a_settled_checkpoint_under("SteeringCtx")

    def step_resolved(self, name: Key, thunk: Callable[[Key], Any], /) -> Never:
        refuse_a_settled_checkpoint_under("SteeringCtx")

    def settle(self, name: Key, value: Any, /) -> Never:
        refuse_a_settled_checkpoint_under("SteeringCtx")

    def __getattr__(self, item: str) -> Any:
        """Delegate everything not steered — `await_event`, `sleep_until`, `concurrent_safe`,
        `peek_event`, `repark` — to the wrapped ctx, unchanged."""
        return getattr(self._ctx, item)
