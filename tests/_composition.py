"""The composition table — every ordered pair of combinator HOLES, and what it does.

Composition here is **substitution into an Effect-hole**: there is no `Workflow`
protocol, only `type Effect[T] = Generator[WorkflowOp, Any, T]` (`api.py:30`), and every
combinator is `Effect[T]`-valued with one or more holes of shape
`Callable[..., Effect[T]]`. So ``A ∘ B`` means *B substituted into A's hole*, and the
table is over ordered pairs rather than a wrapper stack.

A member is a HOLE rather than a combinator, which is why `recurse` appears twice: a body
substituted into its ``leaf`` and one substituted into its ``combine`` run under different
frames (``gather:0,{i};rec:{i}`` against ``gather:1,{k};fold:{lv},{k}``), and a table indexed
by combinator cannot state the difference: indexed that way, the fold frames go undeclared
and unexercised.

**Why a table rather than more hand-written cases.** Each hazard below is a pair nobody had
written down, and three of them are SILENT: they compose, produce the wrong thing, and raise
nothing.

===========================  ==================================================
durable sleep                `sleep ∘ gather` routes to `_branch_sleep`, which
                             calls neither ctx, so covering that pair alone
                             bypasses a defect in a ctx's own `sleep`.
`gather ∘ respawn`           can re-execute every sibling once per generation:
                             three emails where the author wrote one.
`Budget ∘ respawn`           `$100` can mean $100 *per generation*, with no
                             chain total.
`Scoped ∘ op-layer`          the two interpreters can disagree about the op
                             stream.
`scoped ∘ measured_drive`    prefixed writes can be compared against bare names.
===========================  ==================================================

**The oracle is `qualified_event_name`** (`api.py:165`), which already composes a
qualified name from a frame path. Each combinator declares what frames a body nested in
its hole will see; the expected key is then *computed*, never spelled. A divergence between
the computed key and the actual one IS the bug, and a new combinator declares its contribution once
and gets a full row and column for free.

The law this encodes is that **the frame map is a monoid homomorphism**:
``frames(A ∘ B) = frames(A) · frames(B)``, into the free monoid on frame atoms.

**Its precondition:** a ``GatherBranch``'s ordinal is a POSITIONAL counter within the enclosing
scope, so a member's frame contribution is not a property of the member alone. Put one sibling
gather beside ``recurse@fold`` and its fold lands at ``gather:2:0`` rather than ``gather:1:0``.
The homomorphism therefore holds over **strictly-nested spines**, which is exactly what
``compose`` builds; the next member that emits a gather BESIDE its hole is where this bites.
`op_key`'s injectivity is the claim that the homomorphism is *injective*.
`formal/lean/Effective/Keys.lean` proves this for the structured encoding and explicitly
defers it for the string serialization, which is the seam this table exercises.

Two axes, one declared property, and no new stratification noun:

* the **interpreter axis** — layers over a handler base (`compose_ops`/`compose_domain`),
  where the handler is literally the `base` argument that ends the onion. Orthogonal;
  layers do not nest inside combinators, the handler drives them.
* the **workflow axis** — the combinators below, nesting by substitution.
* ``Addressing`` (`effective.ops`) — is a NAME completed by the frames (`RELATIVE`) or already
  whole (`ABSOLUTE`)? Exactly a filesystem path's distinction, and it belongs to the name rather
  than to the combinator — attaching it to the op class is what put the first version of the
  guard on the spawn instead of the join.

Distinct atoms per slot (``s:0``/``s:1``, ``sk0``/``sk1``, ``classify0``/``classify1``)
so the grid tests *composition* and not name collision — collision has its own suite
(`test_op_key_injectivity.py`).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from effective.api import (
    Effect,
    GatherBranch,
    await_event,
    gather,
    qualified_event_name,
    scoped,
    step,
)
from effective.combinators import (
    Again,
    Answered,
    Chain,
    Deeper,
    Done,
    Turn,
    descend,
    hoisted,
    recurse,
    respawn,
    route,
)
from effective.domain import CallTool, Spawned
from effective.fork import join_fork, spawn_fork
from effective.handlers.base import step_key
from effective.handlers.durable import spawn_done_name
from effective.handlers.recording import RecordingHandler, Respawned, Suspended
from effective.keys import Key, compose_key
from effective.ops import Writer
from effective.skills import Pin
from effective.spawning import ChildAnswer, Returned

type Body = Callable[[], Effect[str]]
"""A thunk producing one observable op — what gets substituted into a hole."""

type Terminated = Callable[[], Effect[Any]]
"""A TERMINAL's effect. Typed apart from `Body` because it is a different kind of thing: a body
returns a value to the combinator enclosing it (which is why `Body` is `str`-valued and the grid
can assert the leaf's value comes back), while a terminal ends or leaves the task and hands its
enclosing combinator nothing a caller could use — a `ForkOutcome` that never arrives, or no
return at all."""

type Frame = GatherBranch | Key
"""One frame on the path to an op: a gather's branch coordinate or a scope atom."""

LEAF = "leaf"
"""The name of the single op every composed workflow bottoms out in."""


class Disposition(Enum):
    """What we have decided about an ordered pair.

    There is deliberately no fourth value. "Only the outer position is defined" is not a
    disposition — it is a *reason* the inner order is `LOUD`, and it lives in the reason
    text. `SILENT` is likewise absent by construction: a pair that composes, is wrong, and
    says nothing is a defect, not a classification. That absence is the law."""

    OK = "OK"
    """Composes; the key is exactly `frames(A) · frames(B) · name`."""

    LOUD = "LOUD"
    """Raises at the seam, with a message naming the culprit and the fix."""

    COSTLY = "COSTLY"
    """Correct, but pays what the other order does not."""


@dataclass(frozen=True)
class Prologue:
    """One op a combinator mints BEFORE the body, and the frames IT sits under.

    `frames` defaults to nothing because most members mint their prologue at the combinator's own
    level, where a bare name is enough. `recurse@combine`'s prologue is its two *leaves*, each
    under `gather:0,{i};rec:{i}`, so without the field the table could not state that member."""

    name: str
    frames: tuple[Frame, ...] = ()


@dataclass(frozen=True)
class Built:
    """One combinator-hole, instantiated for a slot in a pair.

    A member is a HOLE, not a combinator: `recurse` appears twice because substituting into its
    `leaf` and into its `combine` puts a body under different frames, and a table indexed by
    combinator could not say so.

    `frames` is the contribution a body nested in `hole` will see. `overhead` are the ops the
    combinator mints before running the body — `recurse`'s decompose, `route`'s classifier,
    `hoisted`'s activation — each with the frames it sits under."""

    name: str
    frames: tuple[Frame, ...]
    is_branch: bool
    """Does the body run inside a gather BRANCH? Distinct from `frames`: `descend` contributes
    a frame but runs its judge in a `scoped`, while `recurse`'s leaf IS a gather branch. This
    is what decides a TASK-ADDRESSING member's disposition, not whether frames exist."""
    overhead: tuple[Prologue, ...]
    responses: Mapping[str, Any]
    hole: Callable[[Body], Effect[str]]


def observable(name: str, **args: Any) -> Effect[str]:
    """One canned observational step — the same `_op` shape `test_combinators.py` uses."""
    return step(name, CallTool(name="op", args=args, result_schema=str))


def _leaf_body() -> Effect[str]:
    return (yield from observable(LEAF))


def leaf() -> Body:
    """The innermost body: one op named `LEAF`, under whatever frames enclose it."""
    return _leaf_body


# --- the builders: one per combinator, parameterized by slot -----------------------
#
# `slot` distinguishes the outer (0) from the inner (1) instance so that a pair like
# `route ∘ route` — neither of which contributes a frame — does not mint the same key
# twice and fail the grid for a reason that has nothing to do with composition.


def _build_gather(slot: int) -> Built:
    def hole(body: Body) -> Effect[str]:
        results = yield from gather([body])
        return results[0]

    return Built(
        name="gather",
        is_branch=True,
        frames=(GatherBranch(0, 0),),
        overhead=(),
        responses={},
        hole=hole,
    )


def _build_scoped(slot: int) -> Built:
    # lint: terminal-hole — `slot: int`, and an `int` cannot carry a delimiter, so it needs
    # no wrapper in any position. `Segment` would in fact REFUSE it (it takes a `str`).
    atom = compose_key(t"s:{slot}")

    def hole(body: Body) -> Effect[str]:
        return (yield from scoped(atom, body))

    return Built(
        name="scoped",
        is_branch=False,
        frames=(atom,),
        overhead=(),
        responses={},
        hole=hole,
    )


def _build_recurse(slot: int) -> Built:
    # Exactly ONE chunk, so the tree-fold never runs and `combine` is never called —
    # asserted below rather than assumed. That keeps a recurse cell's trace to
    # decompose + the body, which is what makes the frame law readable.
    decompose_name = f"decompose{slot}"

    def hole(body: Body) -> Effect[str]:
        def decompose(ctx: str) -> Effect[Sequence[str]]:
            raw = yield from observable(decompose_name, ctx=ctx)
            return [raw]

        def one_leaf(_chunk: str) -> Effect[str]:
            return (yield from body())

        def combine(_group: Sequence[str]) -> Effect[str]:
            raise AssertionError("recurse: a single chunk must pass through uncombined")
            yield  # pragma: no cover - unreachable, keeps this a generator function

        return (yield from recurse("ctx", decompose, one_leaf, combine, fanin=2))

    return Built(
        name="recurse",
        is_branch=True,
        frames=(GatherBranch(0, 0), compose_key(t"rec:{0}")),
        overhead=(Prologue(decompose_name),),
        responses={decompose_name: "chunk"},
        hole=hole,
    )


def _build_recurse_fold(slot: int) -> Built:
    """`recurse`'s OTHER hole: a body substituted into `combine` rather than into `leaf`.

    With one chunk the tree-fold never runs, so `combine` is unreachable and `fold:{level},{k}`
    (`combinators.py`) goes unexercised. TWO chunks at `fanin=2` run exactly one fold level,
    which is the smallest fixture that reaches it.

    Note what the frames say and a combinator-indexed table could not: the fold's coordinate is
    ``gather:1:0``, not ``gather:0:0``. `recurse` issues its leaf gather first and its fold gather
    second at the SAME level, so the ordinal discriminates them **relative to this member's own
    position**, not absolutely (see the module docstring's precondition: one sibling gather ahead
    of it and the fold is ``gather:2:0``). That relative ordinal is the `{g}` in `_PrefixedCtx`'s
    prefix. The two leaves become this member's prologue, each under its own
    branch, which is why `Prologue` carries frames at all."""
    decompose_name = f"decompose{slot}"
    leaf_name = f"rleaf{slot}"
    chunks = ("a", "b")

    def hole(body: Body) -> Effect[str]:
        def decompose(ctx: str) -> Effect[Sequence[str]]:
            yield from observable(decompose_name, ctx=ctx)
            return list(chunks)

        def each_leaf(chunk: str) -> Effect[str]:
            return (yield from observable(leaf_name, chunk=chunk))

        def combine(_group: Sequence[str]) -> Effect[str]:
            return (yield from body())

        return (yield from recurse("ctx", decompose, each_leaf, combine, fanin=2))

    return Built(
        name="recurse@fold",
        # The fold's `combine` runs in a gather BRANCH exactly as the leaf does — so a
        # task-addressing member is refused here too, and the disposition stays derived.
        is_branch=True,
        frames=(GatherBranch(1, 0), compose_key(t"fold:{0},{0}")),
        overhead=(
            Prologue(decompose_name),
            *(
                # lint: terminal-hole — `i` is the loop index below, an `int`.
                Prologue(leaf_name, (GatherBranch(0, i), compose_key(t"rec:{i}")))
                for i in range(len(chunks))
            ),
        ),
        responses={decompose_name: "chunk", leaf_name: "L"},
        hole=hole,
    )


def _build_route(slot: int) -> Built:
    classify_name = f"classify{slot}"

    def hole(body: Body) -> Effect[str]:
        def classifier(chunk: str) -> Effect[str]:
            return (yield from observable(classify_name, chunk=chunk))

        return (yield from route("ctx", classifier, {"go": lambda _chunk: body()}))

    return Built(
        name="route",
        is_branch=False,
        frames=(),
        overhead=(Prologue(classify_name),),
        responses={classify_name: "go"},
        hole=hole,
    )


def _build_descend(slot: int) -> Built:
    def hole(body: Body) -> Effect[str]:
        def judge(_ctx: str, _level: Any) -> Effect[Answered[str] | Deeper[str]]:
            value = yield from body()
            return Answered(value)

        return (yield from descend("ctx", judge, budget=1))

    return Built(
        name="descend",
        is_branch=False,
        frames=(compose_key(t"d:{0}"),),
        overhead=(),
        responses={},
        hole=hole,
    )


def _build_hoisted(slot: int) -> Built:
    skill = f"sk{slot}"
    activate = f"skill:{skill},activate"

    def hole(body: Body) -> Effect[str]:
        return (yield from hoisted((skill,), lambda _pins: body()))

    return Built(
        name="hoisted",
        is_branch=False,
        frames=(),
        overhead=(Prologue(activate),),
        responses={activate: Pin(name=skill, content_hash="h", body="b")},
        hole=hole,
    )


BUILDERS: Mapping[str, Callable[[int], Built]] = {
    "gather": _build_gather,
    "scoped": _build_scoped,
    "recurse": _build_recurse,
    "recurse@fold": _build_recurse_fold,
    "route": _build_route,
    "descend": _build_descend,
    "hoisted": _build_hoisted,
}
"""The workflow-axis combinators that take a body and can be built from one uniformly.

Deliberately NOT here, each for a stated reason rather than an oversight:
`run_agent` / `improve` / `run_code` (they need a domain or a sandbox, so they belong in the
promoted durable subset rather than the infra-free grid); and the task-addressing members
(`spawn_fork`, `marginal_sweep`, `spawn_subagent_task`), which have no Effect-hole at all
— they can only ever be the INNER of a pair, and are covered by their own cases."""


CHILD_RUN_ID = "c1"
SPAWN_KEY = "tool:spawn,c1"
DONE_EVENT = spawn_done_name(Writer(task=str(uuid4()), placement=Key.parse(SPAWN_KEY))).stored()
"""What the CHILD task emits, from its own params — unframed, because it has never seen ours."""
FORK_ANSWER = ChildAnswer(answer=Returned(value=None))
"""A fork child's answer, as the child emits it."""


def spawn_at_root() -> Effect[Any]:
    """`spawn_fork` at the task root. Its own `Step` key is a PLACE and may be framed freely;
    what must not be framed is the `done_event` the child will emit."""
    return spawn_fork(
        "child-task",
        child_run_id=CHILD_RUN_ID,
        base_task_id=uuid4(),
        through="k",
        fork_point=Key.parse("p"),
        forked_from="r0",
        forked_at_event="e0",
        delta={},
    )


def terminal_disposition(outer: Built) -> Disposition:
    """LOUD iff `outer` runs its body in a gather BRANCH, a disposition *derived* from the member.

    The policy does NOT key on "has frames". A scope frame is rerouted around: a scope completes
    RELATIVE names, and an absolute one needs no completing, the same rule `budget-grant` follows
    when it roots at `_root_ctx`. A branch coordinate is a concurrency slot
    with nothing to reroute to, so it refuses. Hence `scoped`/`descend` are OK *despite*
    contributing frames, and `gather`/`recurse` are LOUD because their bodies run in branches.

    **One discriminator, two mechanisms**, which is why the name is `terminal_` rather than
    fork-specific. A fork's join is refused
    because a branch coordinate would rescope an ABSOLUTE name away from the task that emits it;
    a `Respawn` is refused because a generation boundary ends the whole task, and a branch that
    respawned would end its siblings' task too (`ops.refuse_respawn_in_branch`). Different
    failure, different op kind, same predicate — measured for both, not assumed from one."""
    return Disposition.LOUD if outer.is_branch else Disposition.OK


def compose(*chain: Built, body: Body | None = None) -> Body:
    """`A ∘ B ∘ …` — each member substituted into the previous one's hole, `body` innermost.

    Variadic because depth buys frame SHAPES a pair cannot make: pairs reach 12 distinct shapes
    (max 4 frames), triples reach 33 (max 6), 21 of them only at depth three. A pair is just the
    two-member case; nothing about the laws is about arity."""
    composed: Body = body if body is not None else leaf()
    for built in reversed(chain):
        composed = partial(built.hole, composed)
    return composed


def enclosing_frames(*chain: Built) -> tuple[Frame, ...]:
    """The frame path a body sees at the bottom of `chain` — the homomorphism's image.

    `frames(A ∘ B ∘ …) = frames(A) · frames(B) · …`, into the free monoid on frame atoms.
    Every expectation in the table is composed from this, so the law is stated ONCE and the
    tests differ only in what they measure at the bottom (a checkpoint key, a park name)."""
    return tuple(frame for built in chain for frame in built.frames)


def expected_keys(*chain: Built, leaf_name: str = LEAF) -> list[str]:
    """The oracle: what the trace MUST be, computed from each member's frame contribution.

    The homomorphism made executable: each member's prologue sits under the frames of
    everything ENCLOSING it plus its own, and the leaf sits under all of them, in nesting
    order. Nothing here spells a key.

    **It ASKS for the step arm rather than spelling it** (`step_key`). The grid computes every
    expectation through the substrate's own minters, so when a key's shape changes the identity
    moves, the law does not, and no verifier needs an edit.

    Deliberately NOT applied to `expected_responses` below: `RecordingHandler` keys canned
    responses by AUTHOR text, not by op key (recording.py), so arming that one would break tests
    that pass correctly."""
    keys: list[str] = []
    enclosing: tuple[Frame, ...] = ()
    for built in chain:
        keys += [
            qualified_event_name(
                *enclosing, *pro.frames, name=step_key(pro.name).stored()
            ).stored()
            for pro in built.overhead
        ]
        enclosing += built.frames
    keys.append(qualified_event_name(*enclosing, name=step_key(leaf_name).stored()).stored())
    return keys


def _prologue_responses(built: Built) -> list[tuple[Prologue, Any]]:
    """Each prologue paired with its canned response.

    A LIST, not a name-keyed dict: `recurse@fold` mints its leaf op TWICE, once per branch, and
    the two occurrences differ only in their frames — so keying by name silently drops one and
    the member fails to run for a reason that has nothing to do with composition. `responses`
    stays keyed by name (one canned value per op), while the frames come from the declaration."""
    return [
        (pro, built.responses[pro.name]) for pro in built.overhead if pro.name in built.responses
    ]


def expected_responses(*chain: Built) -> dict[str, Any]:
    """Canned responses keyed by the same computed names, plus the leaf's."""
    canned: dict[str, Any] = {}
    enclosing: tuple[Frame, ...] = ()
    for built in chain:
        canned |= {
            qualified_event_name(*enclosing, *pro.frames, name=pro.name).stored(): value
            for pro, value in _prologue_responses(built)
        }
        enclosing += built.frames
    canned[qualified_event_name(*enclosing, name=LEAF).stored()] = "L"
    return canned


PARK_EVENT = "ev:1"
"""The name the parking leaf awaits — RELATIVE, so the frames must complete it."""


def parking_leaf() -> Body:
    """The other innermost body: one op that PARKS instead of completing.

    The second law rests on it. The completing leaf exercises the checkpoint-key path
    (`_prefix`/`_PrefixedCtx`); a park travels by a different mechanism entirely (`_scope_path`
    + park-as-value re-arms at the barrier), and that path carries the frame hazards: the
    nested-gather park name, the frame-blind wake-race name, the qualified-emitter contract. A
    grid of completing leaves alone stays green while missing all three."""

    def body() -> Effect[str]:
        return (yield from await_event(PARK_EVENT, str))

    return body


def expected_park(*chain: Built) -> Key:
    """The park name a RELATIVE await at the bottom of `chain` must reach the engine with.

    Same homomorphism, same oracle, different measurement point, which is the argument that
    this is one law about frames rather than two coincidences about two code paths."""
    return qualified_event_name(*enclosing_frames(*chain), name=PARK_EVENT)


def observed_disposition(outer: Built, inner: Built) -> Disposition | None:
    """RUN the pair and report what it actually did — the disposition MEASURED, not declared.

    `None` means **SILENT**: it composed, returned something other than the leaf's value, and
    raised nothing. `Disposition` deliberately has no such member (see its docstring), so `None`
    here is that absence made checkable — and the law over this function is that no cell ever
    observes it.

    The laws that read a disposition read this one, so they read a computed value. An `ok` set
    built from `pairs()`, the full cartesian product, would assert that every pair is in the set
    of all pairs, and the law would stay green when a non-composing member arrives."""
    handler = RecordingHandler(responses=expected_responses(outer, inner))
    try:
        result = handler.run(compose(outer, inner))
    except BaseException:
        return Disposition.LOUD
    return Disposition.OK if result == "L" else None


def pairs() -> list[tuple[Built, Built]]:
    """Every ordered pair, outer built in slot 0 and inner in slot 1."""
    return [
        (outer_builder(0), inner_builder(1))
        for outer_builder in BUILDERS.values()
        for inner_builder in BUILDERS.values()
    ]


def triples() -> list[tuple[Built, Built, Built]]:
    """Every ordered triple — the frame shapes a pair cannot make.

    Measured rather than argued: pairs reach 12 distinct frame shapes (max depth 4), triples
    reach 33 (max depth 6), and the 21 shapes only depth three reaches are the justification.
    `gather(scoped(gather(await)))` is not among them: `recurse ∘ gather` reaches it as a pair.
    Three distinct slots for the same reason pairs use two."""
    return [
        (a(0), b(1), c(2))
        for a in BUILDERS.values()
        for b in BUILDERS.values()
        for c in BUILDERS.values()
    ]


# --- the TERMINALS: members with no Effect-hole ------------------------------------------------


@dataclass(frozen=True)
class Terminal:
    """A member that can only ever be the INNER of a pair, because it has no Effect-hole.

    Two of them: `fork` **leaves** the task (spawn a child, park on its answer) and `respawn`
    **ends** it (the generation boundary). Neither takes a body, so neither can contribute frames,
    and neither can appear in the frames grid above, so that grid stays green whatever their
    diagonal cells do.

    `prologue` runs at the task ROOT and its value feeds `body` — `fork`'s spawn must sit outside
    the frame under test, because the whole question is what happens to the *join*.

    `outcome` is what a non-refused run returns: a `Suspended` parked on the child's event, or a
    `Respawned`. `refusal_names` are the substrings the LOUD message must contain — the culprit
    and the fix, the standard this repo holds refusals to."""

    name: str
    prologue: Callable[[], Effect[Any]]
    body: Callable[[Any], Terminated]
    responses: Mapping[str, Any]
    outcome: type
    refusal_names: tuple[str, ...]


def _nothing() -> Effect[None]:
    """The empty prologue — `respawn` needs no root-level setup, `fork` needs its spawn."""
    return None
    yield  # pragma: no cover - unreachable, keeps this a generator function


class ChainState(BaseModel):
    """The carry across a generation boundary — a model, because it crosses as JSON."""

    n: int = 0


CHAIN_TASK = "chain-task"
CHAIN_RUN = "chain-run"


def respawning_body(_carried: Any = None) -> Terminated:
    """One generation that asks for another — the real boundary, not a degenerate chain.

    `Again`, deliberately. A `respawn` whose step returns `Done` never crosses a boundary and
    would classify every pair as OK, including the two that are refused; a table built on it
    would be green and wrong. The generation-as-OUTER member below is the one that uses `Done`,
    and for the opposite reason — there it is the BODY under test, not the boundary."""

    def body() -> Effect[Any]:
        def step(state: ChainState, _turn: Turn) -> Effect[Again[ChainState]]:
            return Again(ChainState(n=state.n + 1))
            yield  # pragma: no cover - unreachable, keeps this a generator function

        chain = Chain(task=CHAIN_TASK, state=ChainState(), run_id=CHAIN_RUN)
        return (yield from respawn(step, chain))

    return body


TERMINALS: Mapping[str, Terminal] = {
    "fork": Terminal(
        name="fork",
        prologue=spawn_at_root,
        body=lambda handle: lambda: join_fork(handle),
        responses={SPAWN_KEY: Spawned(task_id=uuid4(), done_event=Key.parse(DONE_EVENT))},
        outcome=Suspended,
        refusal_names=(DONE_EVENT, "is an ABSOLUTE event name", "loop", "marginal_sweep"),
    ),
    "respawn": Terminal(
        name="respawn",
        prologue=_nothing,
        body=respawning_body,
        responses={},
        outcome=Respawned,
        refusal_names=("respawn cannot run inside the gather branch", "OUTER loop"),
    ),
}
"""The task-addressing members, registered so they get a COLUMN the way a hole gets a row.

Still deliberately absent, each for a stated reason: `marginal_sweep` and `spawn_subagent_task`
are compositions OF `fork` (a loop of spawns and a packaged spawn+await), so they inherit its
column rather than adding one; `run_agent`/`improve`/`run_code` need a domain or a
sandbox."""


def compose_terminal(outer: Built, terminal: Terminal) -> Body:
    """`outer ∘ terminal` — the prologue at the ROOT, the terminal inside `outer`'s hole."""

    def composed() -> Effect[str]:
        carried = yield from terminal.prologue()
        return (yield from outer.hole(terminal.body(carried)))

    return composed


def generation_outer() -> Built:
    """`respawn` in the OTHER direction: a generation as the OUTER, with a row of its own.

    The table records an asymmetry here, and it is why `respawn` appears in two registries
    instead of one row-and-column: **the two directions are different things.** As the outer it
    is an ordinary frame-contributing hole that contributes NO frames (a generation is a fresh
    task with a fresh checkpoint store, and `Turn` deliberately carries no scope), so everything
    composes inside one. As the inner it
    is a `Terminal` that ends the task and returns no value to its enclosing combinator.

    A single row-and-column entry would have to pick one, and picking the outer's `Done` shape is
    what would make the table green and wrong: `gather ∘ respawn` is refused, and a `Done`-step
    respawn in that cell composes cleanly because it never reaches a boundary."""

    def hole(body: Body) -> Effect[str]:
        def step(_state: ChainState, _turn: Turn) -> Effect[Done[str]]:
            value = yield from body()
            return Done(value)

        chain = Chain(task=CHAIN_TASK, state=ChainState(), run_id=CHAIN_RUN)
        return (yield from respawn(step, chain))

    return Built(
        name="respawn",
        is_branch=False,
        frames=(),
        overhead=(),
        responses={},
        hole=hole,
    )
