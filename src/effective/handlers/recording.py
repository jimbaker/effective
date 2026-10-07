"""RecordingHandler — interpret a workflow with no I/O.

Canned responses (by step / event name) stand in for the LLM and tools; ledger
appends and artifact writes go to in-memory lists. Every op is recorded to a
trace that ReplayHandler can later re-run. When the workflow awaits an event
with no canned value, the handler *parks* and returns a ``Suspended`` — the
live generator holds all local state, so resuming needs no serialization.

This handler is surface-agnostic: it drives both an Option-A sync generator and
an Option-C coroutine, since both speak ``.send()`` / yield over the same ops.
"""

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, assert_never

from effective.api import Effect
from effective.choice import Answer, answer, stored_endings
from effective.govern import all_refusals, delivered
from effective.handlers.admission import ChoiceBox, drive_racing
from effective.handlers.base import (
    NO_RACE,
    BranchRaised,
    BranchStopped,
    Racing,
    Stop,
    Stopping,
    TraceEntry,
    artifact_id,
    barrier_errors,
    branch_slot,
    deadline_of,
    ending_of,
    placed_await_name,
    placed_key,
    placing,
    race_errors,
    transient_errors,
    walk_run,
)
from effective.keys import (
    FramePosition,
    Key,
    compose_key,
    frame_path,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
)
from effective.layers import CHECKPOINTED_OPS, OpLayer, drive_through, run_scope
from effective.ops import (
    CARRY_PARAM,
    GENERATION_PARAM,
    Addressing,
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
    WorkflowOp,
    awaits_an_absolute_name,
    refuse_a_bounded_wait_in_a_branch,
    refuse_a_park_in_a_race_branch,
    refuse_absolute_await_in_branch,
    refuse_respawn_in_branch,
)
from effective.permission import Refused


@dataclass
class Suspended[T]:
    """A parked workflow awaiting an event. Resume by delivering it.

    One divergence from the durable engines, by design: the engines treat an
    emitted event as a durable FACT (first write wins), so a workflow awaiting
    the same name twice binds the same payload from one emission; the recorder
    parks per un-canned await, so two resumes may deliver two different
    payloads. Use distinct (run-scoped) names per logical question — the same
    rule the engines' global-event semantics already demand."""

    gen: Any
    awaiting: Key
    schema: type
    handler: RecordingHandler
    # An ABSOLUTE park resolves unframed (see `_qualified`): the task that emits it has never
    # seen this handler's prefix, so prepending one would name something nothing produces.
    # `kw_only` so a defaulted field on the BASE does not force defaults onto the subclasses'
    # required ones (`GatherSuspended.children`, `ScopedSuspended.inner`).
    addressing: Addressing = field(default=Addressing.RELATIVE, kw_only=True)
    deadline: datetime | None = field(default=None, kw_only=True)
    """The instant the parked wait gives up, so a resume answers in that wait's own kind.

    A deadline-free park resumes with the payload; a bounded one resumes with an `Arrived` or an
    `Expired`, and a bare payload delivered to one is refused rather than sent into a workflow
    whose arms would all miss it."""

    def resume(self, event: Any, *, name: Key | None = None) -> T | Suspended[T]:
        # Pass `name` to assert WHICH event this delivers. The recorder binds POSITIONALLY
        # (it wakes THIS slot regardless of the name), so out-of-order emission silently binds
        # the wrong payload; the engines bind by name (an emitted event is a durable fact), so
        # this guard is the cheap parity tooth. Absent `name`, behavior is unchanged.
        if name is not None and name != self.awaiting:
            raise ValueError(
                f"resume delivers to the parked await {self.awaiting.display()!r}, not "
                f"{name.display()!r}: the "
                "recorder binds by POSITION, so naming a different event would silently swap "
                "payloads (the engines bind by name)."
            )
        # The event entry carries the handler's path prefix like every other branch leaf, so a
        # resumed-gather trace replays by re-execution with the same keys the durable checkpoint
        # store would use. An ABSOLUTE await is the exception in both halves: it RESOLVED at the
        # root, so it is recorded there — prefixing it would make the trace disagree with the
        # durable record (Absurd checkpoints the bare `$awaitEvent:spawn-done:...`) and with this
        # comment's own claim. The reconstructed op carries its addressing for the same reason.
        key = compose_key(t"event;{self.awaiting:domain=address}")
        parked = AwaitEvent(
            self.awaiting, self.schema, addressing=self.addressing, deadline=self.deadline
        )
        answer = _canned_answer(parked, event)
        self.handler.trace.append(
            TraceEntry(
                key
                if self.addressing is Addressing.ABSOLUTE
                else key.prefixed(self.handler._prefix),
                parked,
                answer,
            )
        )
        return self.handler._drive(self.gen, answer)


def _canned_answer(op: AwaitEvent[Any], response: Any) -> Any:
    """What the recorder hands a workflow for one delivered answer.

    A deadline-free wait takes the payload as given. A bounded one takes a `WaitOutcome`, which
    is what both engines answer and what the workflow matches on, so a fixture canning one writes
    ``Arrived(payload)`` or ``Expired()``. The `isinstance` reads untyped fixture data rather than
    dispatching over the union: what it asks is whether the fixture is in the language at all.
    """
    if op.deadline is None or isinstance(response, Arrived | Expired):
        return response
    raise TypeError(
        f"the wait on {op.name.display()!r} named a deadline, so it is answered by an "
        f"`Arrived(payload)` or an `Expired()`. This delivered {response!r}."
    )


@dataclass
class Respawned:
    """A run that ended at a GENERATION BOUNDARY — the in-memory analogue of the task cut.

    The recorder deliberately does NOT loop the chain in-process. A generation boundary is a real
    task boundary, and the whole value of respawn is that the next generation replays nothing —
    so a test that wants generation *n+1* re-runs the program with `next_params()`, which is
    exactly what the engine does. Pretending the cut away in memory would hide the one property
    the feature exists for.

    Sibling of `Suspended`: both are "the run stopped, here is what it takes to continue", and
    both make an engine-level fact visible to an infra-free test."""

    generation: int
    """The ordinal of the NEXT generation — 0 is the run that has just ended."""

    state: Any
    task: str
    run_id: str
    params: dict[str, Any]

    def next_params(self) -> dict[str, Any]:
        """The spawn params the engine would hand generation `self.generation`."""
        return {**self.params, GENERATION_PARAM: self.generation, CARRY_PARAM: self.state}


def _qualified(slot: Suspended[Any]) -> Key:
    """A parked slot's fully-path-qualified event name: a ``GatherSuspended`` and a
    ``ScopedSuspended`` already composed it (each qualified while its own prefix was
    installed); a plain branch ``Suspended`` qualifies with its handler's prefix."""
    if isinstance(slot, (GatherSuspended, ScopedSuspended)):
        return slot.awaiting
    if slot.addressing is Addressing.ABSOLUTE:
        return slot.awaiting  # nothing may be prepended to an absolute address
    # `prefixed`, not `+`: applying a frame to a finished identity is composition, and this is
    # the recorder's half of the same seam `_PrefixedCtx` owns on the durable path.
    return slot.awaiting.prefixed(slot.handler._prefix)


@dataclass
class GatherSuspended[T](Suspended[T]):
    """A gather whose round completed with parked branches (in memory).

    Mirrors the durable serialized-wake contract: ``awaiting`` names the LOWEST
    parked branch's fully-qualified event; ``resume`` delivers into that
    branch's LIVE generator (legal in-process only — the recorder was never
    crash-durable; the durable path replays instead), re-parks on the next
    parked branch if any remain, and otherwise merges every child in
    branch-index order (the deferred merge — nothing reaches the parent's
    canonical record while parked) and continues the parent generator with the
    joined results."""

    children: list[tuple[RecordingHandler, Any]]  # slot: result | Suspended

    def _refresh(self) -> GatherSuspended[T] | None:
        """Point ``awaiting``/``schema`` at the lowest still-parked slot."""
        for _, slot in self.children:
            # lint: totality(blocked) — `children` is `list[tuple[RecordingHandler, Any]]`, so the
            # slot's type is `Any` and no `match` over it can be closed. Typing that tuple is its
            # own
            # change, tracked with the parked-slot work rather than smuggled in here.
            if isinstance(slot, Suspended):
                self.awaiting = _qualified(slot)
                self.schema = slot.schema
                return self
        return None

    def resume(self, event: Any, *, name: Key | None = None) -> T | Suspended[T]:
        # `awaiting` names the LOWEST parked branch (serialized-wake); `name` (if given) asserts
        # the caller is delivering to THAT branch — the same positional-vs-by-name guard as the
        # base `Suspended.resume`. The durable engines wake the branch whose name matches;
        # naming a different branch here would deliver to the wrong (lowest) one.
        if name is not None and name != self.awaiting:
            raise ValueError(
                f"gather resume delivers to the lowest parked branch "
                f"{self.awaiting.display()!r}, not {name.display()!r}: the recorder wakes "
                f"branches by POSITION, not name."
            )
        for k, (child, slot) in enumerate(self.children):
            # lint: totality(blocked) — same `Any`-typed slot as the scan above; see there.
            if isinstance(slot, Suspended):
                try:
                    self.children[k] = (child, slot.resume(event))
                except Exception as raised:
                    self.children[k] = (child, BranchRaised(raised))
                break
        else:  # resume-once: a second resume after completion would double-merge
            raise RuntimeError(
                "GatherSuspended.resume called after the gather already completed — "
                "each Suspended resumes exactly once (hold the RETURN value; a "
                "completed resume hands back the result, not this object)"
            )
        parked = self._refresh() is not None
        slots = [slot for _, slot in self.children]
        match barrier_errors(slots, parked=parked):
            case None if parked:
                return self
            case None:
                return self.handler._drive(self.gen, self.handler._merge_children(self.children))
            case raised:
                if all_refusals(raised):  # delivered to the parent, so the round's record is kept
                    self.handler._merge_children(self.children)
                return self.handler._drive(self.gen, None, throw=delivered(raised))


@dataclass
class ScopedSuspended[T](Suspended[T]):
    """A park *inside* a ``scoped(...)`` body — the parent's stack frame, made a value.

    `Scoped` is the one structural op whose body the HANDLER drives rather than the workflow
    ``yield from``-ing it, so when that body parks there is no generator chain left holding the
    parent's frame. This is that frame: ``inner`` is the body's own park, ``gen`` the parent
    generator waiting for the body's return value.

    It exists only on the recording path. The durable engines need no analog — a park there
    raises ``SuspendTask`` and the task replays from the top, re-entering the scope by
    re-execution (the no-``call/cc`` rule). In memory there is no replay, so the frame has to
    be held explicitly, exactly as ``GatherSuspended`` holds a parked round.

    ``prefix`` is the scope's installed namespace, re-installed around each resume so keys minted
    after the wake are namespaced identically to those before it. The ordinals need no carrying:
    they belong to the handler, which a scope shares, so a ``gather`` after the wake keeps counting
    where the handler left off.
    """

    inner: Suspended[Any]
    prefix: str

    def resume(self, event: Any, *, name: Key | None = None) -> T | Suspended[T]:
        # `awaiting` is already scope-qualified, so the by-name guard compares against the
        # same string an engine would have delivered to.
        if name is not None and name != self.awaiting:
            raise ValueError(
                f"resume delivers to the parked await {self.awaiting.display()!r}, not "
                f"{name.display()!r}: the "
                "recorder binds by POSITION, so naming a different event would silently swap "
                "payloads (the engines bind by name)."
            )
        with self.handler._scope(self.prefix):
            try:
                out = self.inner.resume(event)
            except Exception as raised:
                return self.handler._drive(self.gen, None, throw=delivered(raised))
            if isinstance(out, Suspended):
                # The body parked again (a second await, or a gather round inside it). Stay a
                # ScopedSuspended: the parent frame is still owed a return value.
                self.inner = out
                self.awaiting = _qualified(out)
                self.schema = out.schema
                return self
        # The body returned. Its value is what the parent's `yield from scoped(...)` evaluates
        # to — continue the parent OUTSIDE the scope, since the scope ended with the body.
        return self.handler._drive(self.gen, out)


@dataclass(frozen=True)
class Joined:
    """A structural op completed; `value` is what the parent's `yield from` evaluates to."""

    value: Any


@dataclass(frozen=True)
class Parked:
    """Something inside a structural op parked, so the PARENT parks: `slot` holds the parent's
    generator and resumes the whole frame."""

    slot: Suspended[Any]


@dataclass(frozen=True)
class Deliver:
    """A refusal escaped an op uncaught, bare or grouped: throw it into the parent generator, the
    chance a plain `yield from` would have given it."""

    error: Exception


@dataclass(frozen=True)
class Ended:
    """A generation boundary inside a structural op ENDED the run; `value` is the run's result.

    A fourth marker rather than an overload of `Parked`, because it is a fourth thing: `Joined`
    means the op produced a value for the parent, `Parked` means the run suspends and can resume,
    `Deliver` means an error goes into the parent, and `Ended` means the run is over and nothing
    resumes.

    `respawn` inside a `scoped(...)` is legal namespacing, and the durable engine handles it (keys
    `s;tool:…`, `s;respawn:…`) by ending the task. This marker makes the recorder end the run at
    the same point, so the two interpreters agree about `Scoped`: handing the `Respawned` back as
    the scope's value would let the workflow resume past it."""

    value: Any


type Structural = Joined | Parked | Deliver | Ended
"""What interpreting a structural op can produce. A closed marker union rather than a sentinel or
an `isinstance` ladder: the four outcomes are different control flow, and a `match` over them
is where a fifth would have to declare itself."""


class RecordingHandler:
    def __init__(
        self,
        responses: Mapping[str, Any] | None = None,
        op_layers: Sequence[OpLayer[Any]] = (),
        _prefix: str = "",
        _in_gather: bool = False,
        _stop: Stop = NO_RACE,
    ) -> None:
        # A `dict` is COPIED — it is the caller's, and 111 fixtures build one inline, so a later
        # mutation must not leak in. Any other Mapping is used AS GIVEN, because a Mapping that
        # COMPUTES has no items to copy and `dict()` would flatten it to whatever it happened to
        # enumerate. That is the seam `tests/_funnel.Answers` needs: it answers by
        # reading the placement off the key it is asked for, so the fixture never spells one.
        #
        # **`is not None`, never truthiness.** This read `responses if responses else {}` and so
        # discarded any Mapping reporting `len() == 0` — which is the honest self-report of a
        # computing Mapping whose every answer needs a frame to say WHICH execution is asking
        # (`tests/_coding.Answers`: the suite's verdict depends on the turn). The comment above
        # promised the seam and the line below took it away, silently: the handler then raised
        # "no canned response" for an op the responder would have answered.
        self.responses: Mapping[str, Any] = (
            dict(responses)
            # lint: totality(coercion) — a three-way normalizer over an OPTIONAL (`dict` /
            # a mapping
            # / `None`), which is `is None` plus a coercion rather than a union dispatch.
            if isinstance(responses, dict)
            else responses
            if responses is not None
            else {}
        )
        self.op_layers: tuple[OpLayer[Any], ...] = tuple(op_layers)
        self.trace: list[TraceEntry] = []
        self.ledger: list[LedgerRow] = []
        self.artifacts: dict[str, tuple[str, Any]] = {}
        # A gather branch runs in a child handler whose keys are namespaced by the path to
        # it: `gather:{g},{i};` per frame (`_position.gather` is this handler's gather ordinal,
        # `i` the branch). So a leaf op's recorded key is its full path through the execution
        # tree (matching the durable checkpoint key), and the gather itself is pure structure
        # with no entry of its own. Nested gathers compose the prefix.
        self._prefix = _prefix
        # Whether this handler IS a gather branch's child, distinct from a non-empty
        # `_prefix`, which a `Scoped` also extends, and the distinction is the whole policy: a
        # scope frame is REROUTED around an ABSOLUTE await (a scope completes RELATIVE names),
        # while a branch coordinate REFUSES it (a concurrency slot has nothing to reroute to).
        # Mirrors `DurableHandler._in_gather`.
        self._in_gather = _in_gather
        # When this handler, as a race loser or inside one, must stop at its next admission.
        self._stop = _stop
        self._position = FramePosition()

    def run[T](self, program: Callable[[], Effect[T]]) -> T | Suspended[T]:
        # Same per-attempt layer scope the durable handler enters, so a layer needing cross-op
        # state behaves identically in-memory and durably (no scope = no cross-op state at all).
        #
        # A GATHER BRANCH TAKES THE SCOPE AND NOT THE RUN, mirroring `DurableHandler.run`: a
        # branch is one run's STRUCTURE, so it gets its own layer dict (branch-local accrual)
        # while inheriting the run's control state. A parent at
        # `(CHAIN_DEPTH, CHAIN_GENERATION) = (1, 3)` hands its branch `(1, 3)`, as the durable
        # handler does; `walk_run()` here would reset the branch to `(0, None)`.
        if self._in_gather:
            with run_scope():
                return self._drive(program(), None)
        with walk_run():
            return self._drive(program(), None)

    def _dispatch(self, op: WorkflowOp, gen: Any) -> Structural:
        """THE DECISION TABLE over `ops.WorkflowOp`'s arms: every arm produces an outcome.

        `Respawn` yields `Ended` ("the run is over and nothing resumes"), so one place acts on
        every outcome and the dispatch lifts out whole.

        **The leaf arms are ENUMERATED and the wildcard REFUSES.** Routed on to
        `_interpret_layered`, an unknown arm would fail in `op_key` with
        `TypeError: unknown op: <object at 0x…>`, a crash from the wrong place naming a memory
        address. A new arm of `WorkflowOp` is a question for a human.
        """
        # ADMISSION. Once its race's choice is saved, a loser stops at its next op of any kind,
        # a structure included: a recording has no earlier attempt to replay, so every op past
        # the choice is new work. The generator is abandoned at a yield it never resumes from.
        if self._stop.now():
            raise Stopping
        if self._stop.racing and isinstance(op, AwaitEvent | SleepUntil):
            refuse_a_park_in_a_race_branch(op, self._prefix)
        with placing(op, self._position):
            match op:
                case AwaitEvent() as awaited:
                    return self._await_or_park(gen, awaited)
                case Gather() | Race() | Scoped():
                    return self._run_structural(op, gen)
                case Respawn():
                    return Ended(self._respawn(op))
                case Step() | AppendLedgerRow() | StoreArtifact() | SleepUntil():
                    return self._interpret_layered(op)
                case unreachable:
                    # BOUND, not `op` — the binding is what carries `Never` here once every arm
                    # above is written, so `ty` fails this call the day a new arm appears.
                    # Passing `op` re-widens the type and the proof disappears silently.
                    assert_never(unreachable)

    def _drive(self, gen: Any, send_value: Any, throw: BaseException | None = None) -> Any:
        while True:
            try:
                op = gen.throw(throw) if throw is not None else gen.send(send_value)
            except StopIteration as done:
                return done.value
            throw = None
            # ANNOTATED so the `match` below is a proof: `_dispatch` declares `-> Structural`,
            # but a plain assignment carries the type only as far as a checker infers it, and
            # the census reads declarations.
            outcome: Structural = self._dispatch(op, gen)
            # Four outcomes, one place that acts on them. `Ended` returns like `Parked` does but
            # means the opposite: nothing resumes.
            match outcome:
                case Parked(slot=slot) | Ended(value=slot):
                    return slot
                case Deliver(error=error):
                    throw = error
                case Joined(value=value):
                    send_value = value
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _respawn(self, op: Respawn) -> Respawned:
        """The generation boundary, in memory: refuse it in a branch, else END this run.

        The refusal mirrors the durable one exactly — same function, same message — because two
        interpreters that decide this separately drift. The recorder does NOT loop the chain
        in-process: a boundary is a real task boundary, so a test that wants generation *n+1*
        re-runs the program with `Respawned.next_params()`, which is what the engine does."""
        if self._in_gather:
            refuse_respawn_in_branch(self._prefix)
        # No `outcome`, no send back into the generator, which is abandoned exactly as the
        # durable engine abandons it.
        return Respawned(
            generation=op.generation,
            state=op.state,
            task=op.task,
            run_id=op.run_id,
            params=op.params,
        )

    def _await_or_park(self, gen: Any, awaited: AwaitEvent[Any]) -> Structural:
        """An AwaitEvent is decided park/resume HERE, before the op-layer stack — the recording
        core cannot park inside a layer (no-call/cc). So op-layers never observe an AwaitEvent on
        this handler, whereas the durable handler routes it through them: a NAMED divergence
        (`layers.LAYERED_OPS_DURABLE_ONLY`)."""
        # The walk's name (see `placed_await_name`): the canned lookup and the park must both
        # use it, or an in-memory test would answer a question the durable path never asked.
        name, schema = placed_await_name(awaited), awaited.schema
        # Refused by its kind on both interpreters, so they cannot drift: a branch
        # resolves its wait by peeking, and a peek reads the event alone. AFTER the absolute
        # check, in the durable handler's order, so a wait that is both gets one diagnosis.
        if awaits_an_absolute_name(awaited):
            # ONE guard here, TWO on the durable side, and the asymmetry is real rather than an
            # omission: the second is `refuse_absolute_await_under_a_rename` (a fork child's
            # event world), and a fork child exists only under `fork.run_fork`, which runs on
            # `DurableHandler`. This class takes no ctx at all, so there is no rename to see and
            # no state to diverge in — the `LAYERED_OPS_DURABLE_ONLY` shape, a NAMED asymmetry.
            # Said here because this is where a reader comparing the two interpreters looks.
            # A branch coordinate is a concurrency slot, not a naming choice, so there is
            # nothing to reroute to — refuse, and name the loop as the fix.
            if self._in_gather:
                refuse_absolute_await_in_branch(awaited, self._prefix)
            # Reroute: resolve unframed. Canned lookups and the park both use the bare name.
            # `.stored()` is required: `name` is a `Key`, which is opaque, so
            # `name in self.responses` would compare a frozen dataclass against `str` keys and
            # be silently always False: an absolute await would park even where the fixture has
            # canned an answer. The `Mapping[str, Any]` annotation on `responses` makes `ty`
            # flag that comparison.
            if (absolute := name.stored()) in self.responses:
                canned = _canned_answer(awaited, self.responses[absolute])
                return self._record_result(awaited, canned)
            return Parked(
                Suspended(
                    gen=gen,
                    awaiting=name,
                    schema=schema,
                    handler=self,
                    addressing=Addressing.ABSOLUTE,
                    deadline=awaited.deadline,
                )
            )
        if awaited.deadline is not None and self._in_gather:
            refuse_a_bounded_wait_in_a_branch(awaited, self._prefix)
        match self._canned(name.stored()):
            case (True, response):
                answered = AwaitEvent(name, schema, deadline=awaited.deadline)
                return self._record_result(answered, _canned_answer(awaited, response))
            case _:
                return Parked(
                    Suspended(
                        gen=gen,
                        awaiting=name,
                        schema=schema,
                        handler=self,
                        deadline=awaited.deadline,
                    )
                )

    def _interpret_layered(self, op: WorkflowOp) -> Structural:
        """Drive one ordinary op through the layer stack, recording it either way. In a race branch
        a stop escapes the layers, and an op a layer yields after a resume is new work a stopped
        loser does not start (`admission.drive_racing`)."""
        try:
            result = (
                drive_racing(
                    self.op_layers,
                    op,
                    self._interpret,
                    new_work=self._stop_if_chosen,
                    stopped=(Stopping,),
                )
                if self._stop.racing
                else drive_through(self.op_layers, op, self._interpret)
            )
        except Refused as refused:
            # A layer (a permission cascade) blocked the op. Record the refusal and deliver it
            # *into* the workflow so a loop can catch it (e.g. route around a denied tool). If the
            # workflow doesn't catch it, `gen.throw` re-raises and it propagates out.
            self._record(
                TraceEntry(placed_key(op).prefixed(self._prefix), op, None, error=refused)
            )
            return Deliver(refused)
        return self._record_result(op, result)

    def _record_result(self, op: WorkflowOp, result: Any) -> Joined:
        """Record a resolved op at its path-prefixed key and hand the value to the workflow."""
        self._record(TraceEntry(placed_key(op).prefixed(self._prefix), op, result))
        return Joined(result)

    @contextmanager
    def _scope(self, prefix: str) -> Iterator[None]:
        """Install a scope's namespacing on THIS handler for the duration of the block.

        A gather uses child handlers because its branches run concurrently and must not share
        mutable state; a scope is sequential, so it runs on this handler and keeps the body's
        leaves in this trace with no merge step.

        The scope changes only the namespace. The ordinals stay the handler's, so a gather inside
        the scope takes the handler's next `gather:{g}` and a scope entered twice cannot name its
        gather the same both times."""
        saved_prefix = self._prefix
        self._prefix = prefix
        try:
            yield
        finally:
            self._prefix = saved_prefix

    def _run_structural(self, op: Gather | Race | Scoped[Any], gen: Any) -> Structural:
        """Interpret a structural op — one that names the ops inside it and records nothing of
        its own — as one of the three things that can come out of it.

        The two share a contract worth stating once: pure structure, a namespace for what is
        inside, and a park that propagates outward as the parent's own park. They differ in what
        the namespace is (a branch coordinate the substrate assigns vs a scope the author chose),
        and in whether the bodies run concurrently.

        An error out of either whose every leaf is a refusal is DELIVERED to the parent, as a plain
        `yield from` would, on every handler (`govern.delivered`); anything else is task-level. The
        refusal is not re-recorded here: the inner drive recorded it at the refused op's own key,
        and a structural op has no key of its own."""
        match op:
            case Gather(branches=branches):
                try:
                    out = self._run_gather(branches)
                except Exception as raised:
                    return Deliver(delivered(raised))
                if isinstance(out, GatherSuspended):
                    out.gen = gen
                    return Parked(out)
                return Joined(out)
            case Race():
                try:
                    return Joined(self._run_race(op))
                except Exception as raised:
                    return Deliver(delivered(raised))
            case Scoped(scope=scope, body=body):
                try:
                    out = self._run_scoped(gen, scope, body)
                except Exception as raised:
                    return Deliver(delivered(raised))
                if isinstance(out, Respawned):
                    return Ended(out)
                return Parked(out) if isinstance(out, Suspended) else Joined(out)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _run_scoped(self, gen: Any, scope: Key, body: Callable[[], Any]) -> Any:
        """Drive a `scoped(...)` body under its prefix; return its value, or the parent's park.

        The qualified event name is read INSIDE the scope: after `_scope` exits the prefix is the
        enclosing one again, and a park qualified out here would carry the wrong prefix (the bug
        this shape exists to make unwritable)."""
        prefix = frame_path(self._prefix, scope)
        with self._scope(prefix):
            out = self._drive(body(), None)
            if isinstance(out, Respawned):  # travels out as `Ended`, never as the scope's value
                # A generation boundary inside a `scoped(...)` is LEGAL: ordinary namespacing,
                # and the durable engine handles it (keys `s;tool:…`, `s;respawn:…`) by ending
                # the task. So it travels OUT: handed back as the scope's value, it would let the
                # workflow resume past the scope holding a `Respawned` object.
                return out
            if isinstance(out, Suspended):
                return ScopedSuspended(
                    gen=gen,
                    awaiting=_qualified(out),
                    schema=out.schema,
                    handler=self,
                    inner=out,
                    prefix=prefix,
                )
        return out

    def _record(self, entry: TraceEntry) -> None:
        self.trace.append(entry)

    def _run_gather(
        self, branches: tuple[Callable[[], Any], ...]
    ) -> list[Any] | GatherSuspended[Any]:
        """Run independent branches under structured concurrency; join by branch index.

        Each branch *i* of the *g*-th gather in this handler runs in a child handler whose
        keys are namespaced ``{self._prefix}gather:{g},{i};``, in its own thread, so there
        is no shared mutation while they run concurrently. An ``asyncio.TaskGroup`` is the
        structured scope: it barriers on all branches, each branch runs to its end, and the
        branches' exceptions surface together as one ``ExceptionGroup``. Post-barrier the branch
        traces are merged into this handler's trace in **branch-index order** (never completion
        order),
        so a gathered leaf's recorded key is its full path — the same string the durable
        checkpoint uses — and replay re-runs the branch generators to those keys (the gather
        is pure structure; the aggregated result is reconstructed, not stored).
        """
        g = self._position.next_gather()
        return asyncio.run(self._gather_async(g, branches))

    async def _gather_async(
        self, g: int, branches: tuple[Callable[[], Any], ...]
    ) -> list[Any] | GatherSuspended[Any]:
        async def run_branch(i: int, thunk: Callable[[], Any]) -> tuple[RecordingHandler, Any]:
            child = RecordingHandler(
                responses=self.responses,
                op_layers=self.op_layers,
                _prefix=self._prefix + gather_prefix(g, i),
                _in_gather=True,
                _stop=self._stop,
            )
            # A branch park is a VALUE at the barrier: the child returns its
            # Suspended, the round completes, and the merge is DEFERRED — merging
            # completed siblings early would put their entries ahead of a parked
            # branch's later ones and break the branch-index replay order. A branch's exception
            # is a value too, so no branch's error cancels a sibling that would raise its own.
            try:
                return child, await asyncio.to_thread(child.run, thunk)
            except Stopping:
                return child, BranchStopped()
            except Exception as raised:
                return child, BranchRaised(raised)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_branch(i, b)) for i, b in enumerate(branches)]
        children = [task.result() for task in tasks]
        lowest = next((slot for _, slot in children if isinstance(slot, Suspended)), None)
        slots = [slot for _, slot in children]
        if (raised := barrier_errors(slots, parked=lowest is not None)) is not None:
            if all_refusals(raised):  # delivered to the parent, so the round's record is kept
                self._merge_children(children)
            raise raised
        if any(isinstance(slot, BranchStopped) for slot in slots):
            # A race loser holding this gather stopped inside it. The round completed, so its
            # siblings' committed ops are on the record before the loser stops.
            self._merge_children(children)
            raise Stopping
        # The LOWEST parked branch is what `awaiting` names (the serialized-wake rule), so find it
        # first and construct with it. A field whose contract is "always a real key" holds
        # nothing else, however briefly — so no placeholder, and nothing here mints an empty
        # `Key`. `awaiting: Key | None` is the alternative and is worse: it propagates through
        # the BASE `Suspended` to 11 reads here and 35 in `tests/` to encode a state that does
        # not exist.
        if lowest is not None:
            return GatherSuspended(
                gen=None,
                awaiting=_qualified(lowest),
                schema=lowest.schema,
                handler=self,
                children=children,
            )
        return self._merge_children(children)

    def _run_race(self, op: Race) -> Answer[Any]:
        """Run a race's branches in threads, reading their completions in batches.

        The choice is published by putting it in `chosen`, a `ChoiceBox` every loser's `Stop` reads
        under its lock at its next admission. The race returns at the barrier, and its record is
        the choice, each branch's ops in branch-index order, then the endings."""
        r, chosen = self._position.next_race(), ChoiceBox()
        tree = self._stop.tree_lock()

        def child(i: int) -> RecordingHandler:
            return RecordingHandler(
                responses=self.responses,
                op_layers=self.op_layers,
                _prefix=self._prefix + race_prefix(r, i),
                _in_gather=True,
                _stop=self._stop.within(lambda: chosen.loses(i), tree),
            )

        children = [child(i) for i in range(len(op.branches))]
        racing = Racing(
            want=op.want,
            branches=len(op.branches),
            run=lambda i: branch_slot(children[i].run, op.branches[i]),
            choose=chosen.put,
            decided=lambda: chosen.current() is not None,
            enclosed=self._stop.now,
            tree=tree,
            deadline=deadline_of(op),
        )
        slots = asyncio.run(racing.concurrently())
        merged = list(zip(children, slots, strict=True))
        if (choice := chosen.current()) is None:
            if (raised := race_errors(slots)) is not None:
                raise raised
            # An enclosing race's choice stopped this race's branches before it chose.
            self._merge_children(merged)
            raise Stopping
        if (transient := transient_errors(slots)) is not None:
            raise transient
        endings = tuple(ending_of(i, slot, choice) for i, slot in enumerate(slots))
        self._record(TraceEntry(race_choice(r).prefixed(self._prefix), op, choice.stored()))
        self._merge_children(merged)
        self._record(
            TraceEntry(race_endings(r).prefixed(self._prefix), op, stored_endings(endings))
        )
        return answer(choice, endings)

    def _merge_children(self, children: list[tuple[RecordingHandler, Any]]) -> list[Any]:
        """Post-barrier, deterministic: in branch (index) order, merge each branch's
        leaves (already path-prefixed) into this trace, and its
        ledger/artifacts, so the canonical record is independent of completion
        timing — and, for a parked round, independent of wake timing."""
        results: list[Any] = []
        for child, result in children:
            self.ledger.extend(child.ledger)
            # Content-addressed ids make the artifact merge safe: same content dedups,
            # distinct content never clobbers.
            self.artifacts.update(child.artifacts)
            for entry in child.trace:
                self._record(entry)  # path-prefixed leaf → parent trace
            results.append(result)
        return results

    def _canned(self, name: str) -> tuple[bool, Any]:
        """Look up a canned response for `name`, QUALIFIED first, then bare.

        Once a scope is structural, an op's bare name does not distinguish the frame it ran in:
        three identical lanes under `scoped(t"talk:{i}")` all yield `ask_llm("enrich")`,
        and a bare-name-only table would hand all three the same answer with no way to say
        otherwise. Trying `{prefix}{name}` first makes each frame addressable
        (`talk:1;enrich`), and falling back to the bare name keeps every existing table working
        and stays the right default: most tests do not care which branch asked.

        The fallback is a TEST-surface convenience and nothing more — the *recorded* key is
        always the qualified one, so what replay binds to is unaffected by which arm hit.

        **Text in, because the table is text.** It is authored by a test
        (`RecordingHandler(responses={"tool:a": 1})`, 111 sites), so it is a READER keyed by
        author text, and nothing here is persisted. A caller holding a `Key` renders it; a
        caller holding a `Step`'s author name — which is a coordinate value and not a key at
        all — passes it straight through. The concatenation is a lookup, not a composition."""
        if (qualified := f"{self._prefix}{name}") in self.responses:
            return True, self.responses[qualified]
        if name in self.responses:
            return True, self.responses[name]
        return False, None

    def _stop_if_chosen(self) -> None:
        """Stop this loser once its race's choice is saved: new work is past the choice."""
        if self._stop.now():
            raise Stopping

    def _interpret(self, op: WorkflowOp) -> Any:
        if self._stop.racing and isinstance(op, CHECKPOINTED_OPS):
            self._stop_if_chosen()  # an op reaching the base after the choice, injected or not
        match op:
            case Step(name=name):
                # The AUTHOR's name, not `op_key(op)`. The arm term makes a Step's key
                # `step:{name}` / `step;{name}` — substrate bookkeeping the fixture author never
                # wrote and should not have to, so the table stays keyed by what they typed.
                match self._canned(name):
                    case (True, response):
                        return response
                raise KeyError(f"RecordingHandler: no canned response for step {name!r}")
            case AwaitEvent(name=name):
                # An op-layer (e.g. a permission cascade) injected an AwaitEvent the
                # workflow never yielded, so it arrives here at the base rather than the
                # `_drive` parking path. A canned response lets in-memory tests exercise the
                # layer's approve/deny logic; without one we CANNOT park, because the
                # recorder holds the live generator and a mid-layer suspend would mean
                # capturing the trampoline stack (a continuation — forbidden, see CLAUDE.md
                # "No call/cc"). Durable suspend-from-mid-layer is the DurableHandler's job.
                if self._stop.racing:  # refused by its kind, as the durable handler refuses it
                    refuse_a_park_in_a_race_branch(op, self._prefix)
                match self._canned(name.stored()):
                    case (True, response):
                        return _canned_answer(op, response)
                raise NotImplementedError(
                    f"RecordingHandler cannot suspend on a layer-injected event {name!r}: "
                    "the in-memory recorder holds the live generator and cannot park a "
                    "mid-layer stack (no call/cc). Provide a canned response to test the "
                    "layer's logic, or use DurableHandler for real durable suspend/resume."
                )
            case AppendLedgerRow(row=row):
                self.ledger.append(row)
                return None
            case StoreArtifact(value=value, content_type=content_type):
                # Content-addressed: the id is a function of the value, so two
                # gather branches storing the same value get the same id (a
                # dedup, correct) and different values never collide — replacing
                # the per-handler sequence counter that reset to 0 in each branch
                # child and made `artifact-0` collide across branches.
                aid = artifact_id(op)
                self.artifacts[aid] = (content_type, value)
                return aid
            case SleepUntil():
                return None
        raise TypeError(f"unknown op: {op!r}")
