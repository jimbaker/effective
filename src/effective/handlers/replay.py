"""ReplayHandler — re-run a workflow against a recorded trace.

Feeds each op its recorded result without performing any I/O, and verifies that
the workflow yields the *same op sequence*. A control-flow change makes replay
fail loudly with a ReplayMismatch rather than silently diverging.

A ``Gather`` is pure structure, not a recorded op: replay **re-runs the branch generators**
(sequentially, since their leaf results come from the trace), matching each branch leaf against its
``gather:{g},{i};``-path-prefixed key and reconstructing the aggregated result. This is the same
re-execution model the durable handler uses, so a recorded key and a durable checkpoint key for one
op are the same string.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, assert_never

from effective.api import Effect
from effective.choice import Answer, Choice, answer, endings_from_stored
from effective.govern import delivered
from effective.handlers.base import (
    BranchRaised,
    BranchStopped,
    Stopping,
    TraceEntry,
    barrier_errors,
    ending_of,
    placed_key,
    placing,
    race_errors,
    settled,
    transient_errors,
    walk_run,
)
from effective.handlers.recording import Respawned
from effective.keys import (
    FramePosition,
    Key,
    frame_path,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
)
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
    refuse_a_park_in_a_race_branch,
    refuse_respawn_in_branch,
)


class ReplayMismatch(Exception):
    pass


def _named(op: Any, prefix: str) -> str:
    """An op's name for a DIAGNOSTIC, which must never be the thing that raises.

    `placed_key` raises `TypeError` for an op whose identity is positional and unplaced, so
    calling it inside a `ReplayMismatch` message turns a mismatch report into an unrelated
    crash — measured on `Respawn`, whose arm this handler was also missing entirely. An error
    path that can fail is an error path that hides the error it was written to name."""
    try:
        return repr(placed_key(op).prefixed(prefix))
    except TypeError:
        return f"<{type(op).__name__}, no placed key>"


def _settled_by(slots: list[Any], recorded: list[dict[str, Any]] | None) -> list[Any]:
    """The slots, each loser the recorded endings settle as `raised` held as an ending: the
    recording answered past its error, so replaying it must too."""
    if recorded is None:
        return slots
    return [
        replace(slot, before_choice=True)
        if isinstance(slot, BranchRaised) and recorded[i]["ending"] == "raised"
        else slot
        for i, slot in enumerate(slots)
    ]


class ReplayHandler:
    def __init__(self, trace: list[TraceEntry]) -> None:
        self.recorded = list(trace)
        self._i = 0  # position in the flat, path-ordered recorded trace

    def run[T](self, program: Callable[[], Effect[T]]) -> T:
        """Re-execute `program` against the recorded trace.

        **Establishes the same per-run ambient its two siblings do** (`walk_run`). This handler
        runs no layers, and workflow code still reads that ambient:

        | without             | a workflow observes                                           |
        |---------------------|---------------------------------------------------------------|
        | `run_scope()`       | a `layer_run_state` cell bumped across two ops reads `[1, 1]` |
        |                     | here and `[1, 2]` on record: a fresh dict on every call       |
        | `enter_task_run()`  | an earlier run's `ops.CHAIN_GENERATION` in the `descend`      |
        |                     | grant names this run composes                                 |
        """
        with walk_run():
            self._i = 0
            result = self._drive(program(), "")
            if self._i != len(self.recorded):
                raise ReplayMismatch(
                    f"workflow ended after {self._i} ops but {len(self.recorded)} were recorded"
                )
            return result

    def _drive(
        self,
        gen: Any,
        prefix: str,
        *,
        in_gather: bool = False,
        position: FramePosition | None = None,
        racing: bool = False,
        stopping: bool = False,
    ) -> Any:
        """Drive one generator (the workflow, a gather branch, or a scoped body) whose leaf keys
        carry ``prefix``. ``position`` holds the positional ordinals of the thread of control: a
        gather branch starts a fresh one, as a child handler does on the record and durable paths,
        and a scoped body is handed its parent's, so a scope entered twice counts on.

        ``in_gather`` is a SEPARATE flag rather than a property of ``prefix``. A ``Scoped`` body
        also carries a non-empty prefix while being no kind of branch, so ``if prefix:`` would
        refuse ``scoped ∘ respawn``, a LEGAL composition (`ops.py`'s terminal law), with an error
        blaming a "gather branch" that is a scope frame.
        `RecordingHandler` carries `_in_gather` for the same reason; this mirrors it, sticky
        through a `Scoped` because a scope inside a branch is still inside the branch.

        ``racing`` holds inside any race branch, and ``stopping`` inside a race loser, whose next
        leaf stops the branch once the record holds no entry of the branch's own at this
        position. Both stay set through every frame nested inside the branch."""
        send_value: Any = None
        throw: BaseException | None = None
        position = FramePosition() if position is None else position
        while True:
            try:
                # ANNOTATED, and the annotation is the whole proof. `gen.send()` is `Any`, so
                # binding the wildcard below buys nothing on its own: with `op` untyped, deleting
                # an arm from this table raises no `ty` error. Naming the union here is what makes
                # the refusal a static check.
                op: WorkflowOp = gen.throw(throw) if throw is not None else gen.send(send_value)
            except StopIteration as done:
                return done.value
            throw = None
            # THE DECISION TABLE over `ops.WorkflowOp`'s arms. An `isinstance` chain lets a
            # missing arm fall THROUGH to the leaf path, where `placed_key` raises
            # `TypeError: unknown op` from inside an error message. A fall-through gives an op
            # the wrong answer silently; a table makes every arm state one.
            #
            # Four behaviors partition them:
            #
            #   | arms                                        | on this walk                    |
            #   |---------------------------------------------|---------------------------------|
            #   | Gather, Scoped                              | STRUCTURE — no recorded entry,  |
            #   |                                             | recurse, leaves carry the frame |
            #   | Race                                        | STRUCTURE, plus its choice and  |
            #   |                                             | endings entries around it       |
            #   | Respawn                                     | TERMINAL — no entry, ends here  |
            #   | Step, AwaitEvent, AppendLedgerRow,          | LEAF — one entry, matched by    |
            #   | StoreArtifact, SleepUntil                   | POSITION, key checked           |
            #
            # This partition is not `layers.UNLAYERED_OPS`, which answers which ops a layer sees.
            self._stop_before_structure(op, prefix, stopping)
            match op:
                case Gather(branches=branches):
                    throw, send_value = self._gather(branches, prefix, position, racing, stopping)
                case Race():
                    throw, send_value = self._race(op, prefix, position, stopping)
                case Scoped(scope=scope, body=body):
                    # A scope is structure exactly like a gather: no recorded entry of its own,
                    # and its body's leaves carry the extended prefix. Unlike a gather branch it
                    # continues this thread's ordinals, so it is handed `position`.
                    throw, send_value = self._scoped(
                        body, frame_path(prefix, scope), position, in_gather, racing, stopping
                    )
                    if isinstance(send_value, Respawned):
                        # It travels OUT rather than being handed back as the scope's value —
                        # the same escape `RecordingHandler._run_scoped` makes, for the reason
                        # it records: "without this the workflow resumed past the scope holding
                        # a `Respawned` object, while the durable engine ended the task."
                        return send_value
                case Respawn():
                    # THE GENERATION BOUNDARY, mirroring `RecordingHandler._respawn` — refuse it
                    # in a gather branch, else END this run. The recorder writes NO trace entry
                    # (the generator is abandoned as the durable engine abandons it), so the
                    # boundary always arrives at `self._i == len(self.recorded)`. Same refusal
                    # function as both siblings, because two interpreters that decide this
                    # separately drift.
                    if in_gather:
                        refuse_respawn_in_branch(prefix)
                    return Respawned(
                        generation=op.generation,
                        state=op.state,
                        task=op.task,
                        run_id=op.run_id,
                        params=op.params,
                    )
                case Step() | AwaitEvent() | AppendLedgerRow() | StoreArtifact() | SleepUntil():
                    throw, send_value = self._leaf(op, prefix, position, racing, stopping)
                case unreachable:
                    # A new arm of `WorkflowOp` with no answer written here, and BOUND rather
                    # than `_`, so the refusal is a `ty` error at this line before it is ever a
                    # runtime one. Passing `op` instead re-widens the type and loses the proof.
                    assert_never(unreachable)

    def _scoped(
        self,
        body: Callable[[], Any],
        prefix: str,
        position: FramePosition,
        in_gather: bool,
        racing: bool,
        stopping: bool,
    ) -> tuple[BaseException | None, Any]:
        """A scoped body's value, or the refusal to throw into the workflow."""
        try:
            return None, self._drive(
                body(),
                prefix,
                in_gather=in_gather,
                position=position,
                racing=racing,
                stopping=stopping,
            )
        except Exception as raised:
            return delivered(raised), None

    def _gather(
        self,
        branches: Sequence[Callable[[], Any]],
        prefix: str,
        position: FramePosition,
        racing: bool,
        stopping: bool,
    ) -> tuple[BaseException | None, Any]:
        """Walk every branch to its end, then the refusals to throw, the stop of a race loser
        holding this gather, or the joined values."""
        g = position.next_gather()
        slots = [
            self._branch_slot(branch, prefix + gather_prefix(g, i), racing, stopping)
            for i, branch in enumerate(branches)
        ]
        match barrier_errors(slots, parked=False):
            case None if any(isinstance(slot, BranchStopped) for slot in slots):
                raise Stopping
            case None:
                return None, slots
            case raised:
                return delivered(raised), None

    def _branch_slot(
        self, branch: Callable[[], Any], prefix: str, racing: bool, stopping: bool
    ) -> Any:
        """A branch's value, or its exception held as `BranchRaised`, or its stop, so every
        branch walks its recorded entries as the recorder ran them."""
        try:
            return self._drive(branch(), prefix, in_gather=True, racing=racing, stopping=stopping)
        except ReplayMismatch:
            raise
        except Stopping:
            return BranchStopped()
        except Exception as raised:
            return BranchRaised(raised)

    def _stop_before_structure(self, op: WorkflowOp, prefix: str, stopping: bool) -> None:
        """Stop a loser at a structure the record holds nothing of its own for: new work."""
        match op:
            case Gather() | Race() | Scoped() if stopping and not self._holds(prefix):
                raise Stopping
            case _:
                return

    def _holds(self, prefix: str) -> bool:
        """Whether the next recorded entry is one of the branch's own, under `prefix`.

        A frame prefix ends in its term separator, so `race:0,1;` is never a prefix of
        `race:0,10;`, and the text test is exact."""
        return self._i < len(self.recorded) and self.recorded[self._i].key.stored().startswith(
            prefix
        )

    def _ahead(self, key: Key) -> Any | None:
        """The result recorded at `key` anywhere ahead of this position, left unconsumed."""
        return next((entry.result for entry in self.recorded[self._i :] if entry.key == key), None)

    def _entry(self, key: Key) -> Any | None:
        """The result recorded at this position if its key is `key`, consuming it; else `None`."""
        if self._i < len(self.recorded) and self.recorded[self._i].key == key:
            self._i += 1
            return self.recorded[self._i - 1].result
        return None

    def _race(
        self, op: Race, prefix: str, position: FramePosition, stopping: bool
    ) -> tuple[BaseException | None, Answer[Any] | None]:
        """Serve the recorded choice, walk each branch against the record, then serve the
        endings. A loser the endings record as stopped stops at the first op the record holds
        nothing of its own for; so does every branch of a race inside a stopped loser, whose
        choice, if one was saved, stands. A loser that ended otherwise holds every leaf it ran,
        and walks them all."""
        r = position.next_race()
        stored = self._entry(race_choice(r).prefixed(prefix))
        choice = None if stored is None else Choice.from_stored(stored)
        recorded = self._ahead(race_endings(r).prefixed(prefix))

        def stops(i: int) -> bool:
            if stopping or choice is None or not choice.loses(i):
                return stopping
            return recorded is None or recorded[i]["ending"] == "stopped"

        slots = [
            self._branch_slot(branch, prefix + race_prefix(r, i), racing=True, stopping=stops(i))
            for i, branch in enumerate(op.branches)
        ]
        if choice is None:
            if stopping:
                raise Stopping  # whatever its branches raised, as the recorder stopped it
            if (raised := race_errors(slots)) is not None:
                return delivered(raised), None
            raise ReplayMismatch(f"race {r} under {prefix!r} has no recorded choice")
        if (transient := transient_errors(_settled_by(slots, recorded), choice)) is not None:
            return delivered(transient), None
        values = {i: slot for i, slot in enumerate(slots) if settled(slot) == "won"}
        if (endings := self._entry(race_endings(r).prefixed(prefix))) is None:
            computed = tuple(ending_of(i, slot, choice) for i, slot in enumerate(slots))
            return None, answer(choice, computed)
        return None, answer(choice, endings_from_stored(endings, values))

    def _leaf(
        self,
        op: Any,
        prefix: str,
        position: FramePosition,
        racing: bool = False,
        stopping: bool = False,
    ) -> tuple[BaseException | None, Any]:
        """One recorded entry, matched by POSITION with the key as a checksum. Inside a race
        branch an await or a sleep is refused by its kind, and a loser stops here when the record
        holds no entry of its own.

        The key is compared and then discarded — the value comes from `self._i`. That is what
        makes this walk stricter than a keyed store lookup rather than weaker: a lookup can only
        fail to find, a checksum fails when anything moved."""
        if racing and isinstance(op, AwaitEvent | SleepUntil):
            refuse_a_park_in_a_race_branch(op, prefix)
        if stopping and not self._holds(prefix):
            raise Stopping
        with placing(op, position):
            if self._i >= len(self.recorded):
                raise ReplayMismatch(
                    f"workflow yielded an extra op {_named(op, prefix)} at position {self._i}"
                )
            expected: TraceEntry = self.recorded[self._i]
            if (actual := placed_key(op).prefixed(prefix)) != expected.key:
                raise ReplayMismatch(
                    f"replay divergence at position {self._i}: "
                    f"recorded {expected.key!r}, workflow yielded {actual!r}"
                )
        self._i += 1
        # A recorded refusal is re-delivered as a throw, so a workflow that caught it on the
        # record run re-catches it identically (deterministic, no re-running the layer).
        return (expected.error, None) if expected.error is not None else (None, expected.result)
