"""Shared handler machinery: the op key and the Handler protocol.

An op's identity is minted one of **three** ways, and which one depends on what names
the op:

- **the op itself** — ``op_key``, for an arm that carries a name (``Step``) or whose
  content determines it (``StoreArtifact``, content-addressed);
- **the walk** — ``placing`` / ``placed_key``, for an arm whose identity is POSITIONAL
  (``SleepUntil``, and ``Gather``'s branch path). A sleep has no name of its
  own and its wake time is data, so what distinguishes the n-th sleep in a frame from the
  n+1-th is the ordinal a handler assigns while walking (``keys.FramePosition``);
- **an authority layer** — ``placed_key`` again, reached through the walk's published
  name, so an approval and a checkpoint bind the same identity.

All of them require identity to be **injective within a run**: two distinct
op-occurrences must not share a key, or the second silently reuses the first's
checkpoint or approval. That holds because **every arm opens with its own tag, a
``Step``'s included** — ``step`` / ``event`` / ``ledger`` / ``artifact`` / ``sleep`` /
``gather`` / ``race`` — so the regions are disjoint by SHAPE. An author may write any
name at all: ``step_key("ledger;x")`` is ``step;ledger;x``, which no ``AppendLedgerRow``
can be. Nothing here rejects a name for the namespace it lands in, and nothing needs to.

The tagged arms are composed through ``compose_key`` (``effective.keys``), the **single
op-key producer** — the durable handler (``handlers/absurd``) names its ``ctx.step``
checkpoints by calling ``op_key`` too, so the recording and durable paths cannot drift to
two key schemes.
"""

import asyncio
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Never, Protocol, assert_never, runtime_checkable

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from effective.api import Effect
from effective.choice import Choice, Ending, Raised, Refusal, Stopped, Unchosen, Won, decide
from effective.cost import Usage
from effective.govern import all_refusals
from effective.keys import (
    FramePosition,
    Key,
    Name,
    Ordinal,
    Scope,
    Segment,
    Subject,
    authored_key,
    carries_structure,
    compose_key,
    scope_prefix,
)
from effective.keys.frame import split_frames
from effective.keys.grammar import PATH_SEPARATOR, TERM_SEPARATOR, KeySyntaxError
from effective.layers import _metering, current_op_name, op_name_scope, run_scope
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
    Unretryable,
    WorkflowOp,
    enter_task_run,
    leaves,
    unretryable,
)


@contextmanager
def walk_run() -> Iterator[None]:
    """The per-run ambient every walk establishes at its entry, for a run that does not meter.

    | establishes        | so that                                                          |
    |--------------------|------------------------------------------------------------------|
    | `enter_task_run()` | the chain state a `descend` grant name reads starts at this task |
    | `run_scope()`      | `layers.layer_run_state` hands out one dict for the whole run    |
    | no meter           | a gate reads no spend, and a nested run none of its host's       |

    The ambient is per run, and a drive loop recurses per frame: a gather branch and a scoped body
    are frames of the same run, so they do not enter it again.

        with walk_run():
            return self._drive(program(), "")
    """
    with _walk_run(None):
        yield


@contextmanager
def _walk_run(spend: Callable[[], Usage] | None) -> Iterator[None]:
    """`walk_run` for a handler that meters: `spend` reads its own replay-derived meter, and the
    gates of the run read it. A durable handler's run entry is its caller."""
    enter_task_run()
    with run_scope(), _metering(spend):
        yield


def canonical_form(value: Any) -> Any:
    """The **deterministic normal form** of ``value`` — a pure function of the value
    alone, stable across ``PYTHONHASHSEED`` and process. The one hazard plain
    ``to_jsonable_python`` leaves: it emits a ``set``/``frozenset`` as a list in
    **hash-iteration order**, which is ``PYTHONHASHSEED``-dependent across processes —
    so a value carrying a set would re-serialize differently across a worker-death +
    fresh-worker resume (the exact case the substrate must survive). Sets are therefore
    sorted by each element's own canonical JSON; real lists/tuples keep their
    (meaningful) order; pydantic models are dumped so their set-typed fields are reached
    too.

    This is the **single canonicalizer** behind both durable serialization boundaries:
    ``content_digest`` (the digest basis, this module) and ``code.canonical`` (the
    sandbox checkpoint form, which wraps this with JSON-mode leaf normalization). One
    canonicalizer, so record and replay cannot disagree on a set, and the ``run_code``
    re-bind fingerprint cannot false-fire across processes."""
    match value:
        case BaseModel():
            return canonical_form(value.model_dump(mode="python"))
        case dict():
            return {k: canonical_form(v) for k, v in value.items()}
        case set() | frozenset():
            elems = [canonical_form(v) for v in value]
            return sorted(elems, key=lambda e: json.dumps(e, sort_keys=True, default=str))
        case list() | tuple():
            return [canonical_form(v) for v in value]
        case _:
            return to_jsonable_python(value)


def content_digest(value: Any) -> str:
    """A deterministic content hash of a JSON-able value — the basis for a
    content-addressed artifact id and a collision-free ``StoreArtifact`` key.

    Canonicalizes (``canonical_form`` — sets sorted, so record and replay agree
    **across processes**) + sorted-key JSON, then a truncated sha256. ``default=str``
    keeps it total for leaves JSON cannot encode directly. Two by-design coincidences
    follow from JSON's type set, harmless unless a caller mixes the types under one
    content_type: ``tuple``/``list`` and ``int``/``str`` dict keys collapse (JSON has
    neither tuples nor non-string keys), and ``bytes b'x'`` hashes as ``str 'x'``
    (``to_jsonable_python`` decodes bytes)."""
    canonical = json.dumps(
        canonical_form(value), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def artifact_key(op: StoreArtifact[Any]) -> Key:
    """A ``StoreArtifact``'s key, composed STRUCTURALLY — `artifact:{type}/{subtype},{digest}`.

    The composer sees that a MIME type is a two-atom PATH and that the digest is a separate
    coordinate, so the key holds together by its structure rather than by an exemption for
    whatever the last hole happens to contain.

    **A content type must be `type/subtype`.** The grammar has no splat (variable arity within
    one template), so the path is fixed at two atoms, which is what RFC 2045 says a media type
    is. Anything else is refused here rather than composing a key whose SHAPE depends on its
    data."""
    kind, slash, subtype = op.content_type.partition(PATH_SEPARATOR)
    if not slash or not subtype:
        raise ValueError(
            f"StoreArtifact content_type {op.content_type!r} is not `type/subtype`. An artifact "
            f"key carries the media type as a two-atom PATH (RFC 2045), so the shape of the key "
            f"cannot depend on the value."
        )
    digest = digest_atom(op.value)
    return compose_key(t"artifact:{Name(kind)}/{Name(subtype)},{Subject(digest)}")


def digest_atom(value: Any) -> Segment:
    """The digest as an ATOM: algorithm-prefixed, so its kind reads from its bytes.

    Bare hex is not a kind a lexer can name: `9f2c1a` reads equally as a name. The prefix is what
    makes the kind legible from the bytes. `content_digest` keeps returning plain hex — it is a
    hash function, not a key renderer, and its other caller puts it in a ledger FIELD where the
    algorithm is already implied.

    **Public**, because a caller across a package seam digests an external id whose charset it
    cannot constrain. A `_`-private reached from another package would claim that nobody outside
    owns it."""
    return Segment(f"sha256-{content_digest(value)}")


def digest_id(value: str) -> Segment:
    """A foreign IDENTITY as an atom — `digest_atom` with the one guard an identity needs.

    **The distinction is totality.** `digest_atom` content-addresses, and content-addressing is
    total by nature: the empty artifact is an artifact, and `artifact_key(StoreArtifact(value=""))`
    must keep working. An identity is the opposite: the empty id addresses nothing, and there is
    no run it could name.

    A digest ERASES that difference. `Segment("")` refuses, while the empty id hashed is a
    well-formed atom, and every empty-id run would mint the SAME one, so
    `PostgresLedger.append`'s `on_conflict_do_nothing(event_id)` would silently drop the second
    such row from a record whose contract is append-only.

    So: digest CONTENT with `digest_atom`, digest an IDENTITY with this. A total function that
    normalizes into an atom kind normalizes the absence of a value right along with the value."""
    if not value:
        raise ValueError(
            "the empty id addresses nothing, so it cannot name a durable identity. A digest would "
            "hash it into a well-formed atom and every such run would mint the same key: a park "
            "on an address nobody meant, and an append the ledger's idempotency would silently "
            "drop. Supply the id, or refuse upstream where it went missing."
        )
    return digest_atom(value)


def artifact_id(op: StoreArtifact[Any]) -> str:
    """The content-addressed id a ``StoreArtifact`` resolves to: its ``artifact_key`` with the
    ``artifact:`` prefix stripped."""
    return artifact_key(op).stored().removeprefix("artifact:")


@dataclass(frozen=True)
class TraceEntry:
    key: Key
    op: WorkflowOp
    result: Any
    error: BaseException | None = None  # set when the op was refused/raised, not resolved


def step_key(name: str) -> Key:
    """A `Step`'s key: the `step` arm term, wrapping the author's own identity.

    **Every key begins with an arm term, a Step's included.** Wrapping keeps author names out of
    substrate namespaces without a denylist: `step;approve:x` is not `approve:x`, so the regions
    are disjoint **by construction** rather than by a tuple somebody has to keep complete. A
    denylist is bounded by what it enumerates; an arm term is bounded by nothing.

    Two shapes, chosen by whether the author's name carries structure:

        step("extract_ticket")       ->  step:extract_ticket       one atom, a coordinate
        step("tool:charge-card")     ->  step;tool:charge-card     a key, spliced as terms

    The split is `keys.carries_structure`, shared with `code.run_code` — the other wrapper of an
    author's name — so the two cannot answer it differently. Underscores stay legal exactly
    because a bare name lands in a COORDINATE, where `NAME` admits them.

    No prefix check and no frame-delimiter check are needed here: the wrapping makes the first
    unnecessary, and `;` means the name is structured, so a structured name has to parse and
    `step("talk:a/b/tool:x")` is refused by the grammar rather than by a special case."""
    if not name:
        raise ValueError("a Step's name is its identity, and the empty name addresses nothing")
    if not carries_structure(name):
        try:
            return compose_key(t"step:{Name(name)}")
        except (ValueError, KeySyntaxError) as exc:
            raise ValueError(
                f"Step name {name!r} is not one well-formed atom, so it cannot be a coordinate "
                f"of the `step` arm: {exc}. Give the step a name in an author namespace instead "
                f"(`tool:{name}`), which composes as `step;tool:{name}`."
            ) from exc
    try:
        # `domain=any`, and it has to be: a step's payload is the name its AUTHOR chose, and the
        # arm is what makes any name safe. `step;ledger:x` is disjoint from `ledger:x` by
        # construction, so no name needs fencing here. Declaring `address` instead refuses
        # `step_key("ledger;x")` — a legal author name — and `test_op_key_injective_across_arms`
        # is the guard that says so.
        return compose_key(t"step;{authored_key(name):domain=any}")
    except KeySyntaxError as exc:
        raise ValueError(
            f"Step name {name!r} carries a separator, so it is read as a KEY — and it is not one: "
            f"{exc}. A term is `tag:arg,arg`, and a nested name is its own term after "
            f"{TERM_SEPARATOR!r} — so `a:b:c` is two separators in one term: write `a:b;c` if `c` "
            f"is a nested name, or `a:b,c` if it is a second coordinate."
        ) from exc


def _await_key(name: Key) -> Key:
    """The `event` arm's key. Text arriving here is turned away by name: `event_name` is the door
    an author's name passes through, and it is where an occurrence the author wrote is refused."""
    match name:
        case Key():
            return compose_key(t"event;{name:domain=address}")
        case str() as text:
            raise TypeError(
                f"an await name is a `Key`, not text: mint it with `event_name({text!r})`, the "
                f"door that refuses an occurrence an author wrote."
            )
        case unreachable:
            assert_never(unreachable)


def op_key(op: WorkflowOp) -> Key:
    """The identity an op carries **in itself** — a name the author gave it, or a function
    of its content. **Every arm opens with its own tag**, a ``Step``'s included: ``step``,
    ``event``, ``ledger``, ``artifact``. Each is composed through ``compose_key``, so an
    interpolated value carrying a delimiter cannot forge one.

    **The POSITIONAL arms raise instead.** A ``Gather`` and a ``SleepUntil`` are identified by
    *where the walk found them*, not by anything the op holds: the g-th gather's
    ``gather:{g},{i};`` frame, the n-th sleep. Returning
    any content function for those means two distinct ops share a key, so this refuses and
    ``placing`` assigns the coordinate.

    **Injectivity across arms is structural, and there is no denylist.** ``step_key`` wraps the
    author's name, so ``step;ledger:X`` is not ``ledger:X`` and the two regions are disjoint **by
    construction** rather than by an enumerated set somebody keeps complete. The lookup that would
    make a collision reachable is still there (``RecordingHandler.responses``); it is safe for a
    reason nobody has to maintain.

    **A denylist is bounded by what it enumerates; a tag is bounded by nothing.** That is the
    general rule.

    **An ``AwaitEvent``'s name is a ``Key``**, minted by ``event_name``, which is where author
    text is refused an occurrence. Text reaching the ``event`` arm is turned away by name rather
    than parsed: parsing would take ``depth-grant:r1,0,depth=1#2`` from an author, and that
    suffix is ``Key.occurrence``'s, minted by ``placing`` for a repeated ask."""
    match op:
        case Step(name=name):
            return step_key(name)
        case AwaitEvent(name=name):
            return _await_key(name)
        case AppendLedgerRow(row=row):
            # The row's `event_id` IS a `Key`, so it splices as one — no `str()` in sight, and
            # no `.get` default to stand in for a missing id, because the model requires it.
            return compose_key(t"ledger;{row.event_id:domain=address}")
        case StoreArtifact():
            return artifact_key(op)
        case SleepUntil():
            raise ValueError(
                "a SleepUntil has no standalone op key: its identity is POSITIONAL (the n-th "
                "sleep in its thread of control, the ordinal a handler assigns while walking), "
                "not a content function of one op. Its wake time is DATA, and belongs in a value: "
                "the checkpoint's own state on Absurd, and `available_at` on both engines, "
                "which is what the scheduler reads. Key it at the walk, via `placing`."
            )
        case Gather() | Race():
            raise ValueError(
                f"a {type(op).__name__} has no standalone op key: its identity is positional "
                "(the g-th gather or r-th race, branch i), the `gather:{g},{i};` or "
                "`race:{r},{i};` frame a handler assigns while walking the execution tree, not a "
                "content function of one op. Every handler keys a branch's leaf by that path; "
                "the gather or race itself is pure structure."
            )
        case Scoped():
            raise ValueError(
                "a Scoped has no standalone op key: a scope is a PREFIX the handler applies "
                "to the keys of the ops inside it, never a key of its own. Key the ops in the "
                "body, not the scope."
            )
        case Respawn():
            raise TypeError(
                "a Respawn has no op key, and no PLACED key either — it is terminal: the run "
                "ends and nothing resumes, so there is no checkpoint to name and no trace entry "
                "to key. `TypeError` and NOT `ValueError` deliberately, unlike the two arms "
                "above: `placed_key` catches `ValueError` to fall back to the walk-minted "
                "positional name, which a Scoped has and a Respawn never does. "
                "`replay._named` catches this type for the same reason."
            )
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


@contextmanager
def placing(op: WorkflowOp, position: FramePosition) -> Iterator[None]:
    """Publish the name `position` assigns to `op` — the walk's coordinate, for the arms that
    need one.

    Two kinds of arm reach it, and the distinction is worth keeping straight:

    - **positional identity** — a `SleepUntil` has no key at all until the walk places it
      (`op_key` refuses), so the minted name IS its identity;
    - **a named identity that needs an occurrence** — a `Scope.SETTLEMENT` `AwaitEvent` names
      itself perfectly well, but its name does not say *which ask*. One answer is declared to
      settle ONE occurrence, so the walk qualifies the name with `Key.occurrence`.

    A no-op for everything else, so nothing in the run is perturbed that did not ask to be.

    **Why the occurrence is applied HERE and not by the composer.** Two answerers compose a
    `descend` grant's name independently: the substrate's own arm (`combinators.refill_levels`)
    and an author-supplied `Grantor` (`combinators.human_grant`), agreeing by construction
    because they call the same function. A coordinate added at either site is bypassed by the
    other; only the walk sits above both. That is the measurement that killed the call-site design,
    and it is the same reason
    this function exists for sleeps: an identity every interpreter must agree on has exactly one
    sound home, the one they all pass through.

    **Byte-preserving at the first ask**, because `Key.occurrence` is the identity at n <= 1. A
    name asked once composes what it always composed, so no recorded run is orphaned and the
    change is invisible to every workflow that does not repeat an authority await.

    **`Scope.ACCRUAL` is deliberately untouched** — `budget-grant:`'s first-emit-wins is the
    FEATURE (a grant raises the run's ceiling and must not be re-requested), and `None` is not a
    reach at all. Only a namespace that *declared* it settles one occurrence gets one.

    One function rather than one per interpreter: recording, replay, the durable walk and both
    fork drivers must agree on this numbering exactly as they must agree on `gather:{g}:`, and a
    rule five walks implement separately is a rule five walks can drift on."""
    match op:
        case SleepUntil():
            # `n`, not the call inline: the registry records a hole's SOURCE EXPRESSION as the
            # decoded field name, so `compose_key(t"sleep:{position.next_sleep()}")` would make
            # every sleep key explain itself as `{"position.next_sleep()": "3"}`.
            n = position.next_sleep()
            with op_name_scope(compose_key(t"sleep:{Ordinal(n)}")):
                yield
        case AwaitEvent(name=name) if name.scope is Scope.SETTLEMENT:
            with op_name_scope(name.occurrence(position.next_await(name))):
                yield
        case _:
            yield


def placed_await_name(op: AwaitEvent[Any]) -> Key:
    """The name this await actually waits ON — the walk's occurrence-qualified form where the
    namespace declared `SETTLEMENT`, and `op.name` verbatim everywhere else.

    **The coordinate has to reach the WAIT name, not merely the trace key.** What a human emits
    against, what the engine registers a wait row under, and what a cached event answers are all
    the name — so a coordinate that lived only in the recorded key would leave the aliasing
    exactly where it was and merely relabel it.

    Falls back to `op.name` outside a drive loop, unlike `placed_key`'s refusal one function
    down, and the asymmetry is real: a positional op has NO identity without the walk, so
    inventing one there would alias. An await always has a name; the walk only qualifies it. A
    layer driven directly in a unit test therefore sees the unqualified name, which is the same
    name it saw before this coordinate existed."""
    if (placed := current_op_name()) is not None:
        return placed
    return op.name


def placed_key(op: WorkflowOp) -> Key:
    """An op's identity **as placed** — `op_key` where the arm names itself, the walk's minted
    name where it does not.

    This is what every consumer of an identity wants: the recorded trace key, the replay
    comparison, an approval event, a gate's park. They all have to agree, and a positional arm
    has no key at all until the walk places it — so a caller that sees only the op (an authority
    layer, notably) asks the walk rather than the op.

    **One function rather than a `try/except` at each site.** The obvious local fallback —
    `type(op).__name__` — yields the literal `"SleepUntil"` for every sleep in every run: an
    alias in exactly the namespace where an alias means an answer delivered to the wrong question
    (the $5/$5,000,000 class). A correct fallback has to be the real coordinate, and only the
    walk knows it.

    Outside a drive loop (a layer exercised directly in a unit test) there is no walk, so this
    re-raises `op_key`'s refusal rather than inventing a name — a layer that needs a positional
    op's identity must be driven by a handler, which is the only place that identity exists.

    **An `AwaitEvent` is handled before the `try`, and it must be**, because `op_key` SUCCEEDS
    for one — `event:{name}` — so the `except` arm below would never consult the walk and the
    trace key would keep the unqualified name while the wait used the qualified one. Two names
    for one op is precisely the drift this function exists to prevent, so the key is recomposed
    from `placed_await_name` and the two cannot disagree."""
    if isinstance(op, AwaitEvent):
        return compose_key(t"event;{placed_await_name(op):domain=address}")
    try:
        return op_key(op)
    except ValueError:
        if (minted := current_op_name()) is not None:
            return minted
        raise


@runtime_checkable
class Handler(Protocol):
    def run[T](self, program: Callable[[], Effect[T]]) -> Any: ...


@dataclass(frozen=True)
class Finished:
    """A task's walk returned: ``value`` is its result in durable form."""

    value: Any


@dataclass(frozen=True)
class Continued:
    """A task's walk ended at a generation boundary: a successor carries the chain on, and
    ``result`` is this generation's own task result."""

    result: Any


@dataclass(frozen=True)
class BranchRaised:
    """A gather branch's exception, held as the branch's slot until every branch has finished."""

    error: Exception


@dataclass(frozen=True)
class BranchStopped:
    """A branch that stopped at an op admission because a race's saved choice names it a loser,
    held as the branch's slot like `BranchRaised`."""


class Stopping(Exception):
    """A handler told to stop declines to admit its next op, and the generator is abandoned.

    Raised by a race loser's handler and caught where the branch runs, which makes it the slot
    `BranchStopped`. A gather inside the loser completes its round first, so a sibling's
    committed ops stay on the record, and then stops the branch that holds it."""


@dataclass(frozen=True)
class Stop:
    """When a branch must stop at its next admission: a race enclosing it has saved a choice
    that names it a loser. One test per enclosing race, outermost first.

    `tree` is the lock every race in one tree of nested races decides under: a race checks
    whether an enclosing choice has stopped it and saves its own choice while holding it, and
    publishes its choice before letting it go, so no inner race saves a choice after its
    enclosing loser's flag."""

    tests: tuple[Callable[[], bool], ...] = ()
    tree: threading.Lock | None = None

    def now(self) -> bool:
        return any(test() for test in self.tests)

    def within(self, test: Callable[[], bool], tree: threading.Lock) -> Stop:
        """The stop for a branch one race deeper, in the race tree `tree`."""
        return Stop((*self.tests, test), tree)

    def tree_lock(self) -> threading.Lock:
        """The race tree's lock for a race this stop encloses: the enclosing tree's, or a new
        tree's when no race encloses it."""
        return self.tree if self.tree is not None else threading.Lock()

    @property
    def racing(self) -> bool:
        """Whether a race encloses this branch at all."""
        return bool(self.tests)


NO_RACE = Stop()
"""The stop of a handler no race encloses: it never stops."""


type Settled = Literal["won", "lost", "raised", "stopped"]


def settled(slot: Any) -> Settled:
    """How a race branch's slot counts toward the choice: a value is a success, a refusal a loss,
    any other error a programming error, and a stop none of these."""
    match slot:
        case BranchStopped():
            return "stopped"
        case BranchRaised(error=error) if all_refusals(error):
            return "lost"
        case BranchRaised():
            return "raised"
        case _:
            return "won"


def ending_of(index: int, slot: Any, choice: Choice) -> Ending[Any]:
    """Branch `index`'s ending, read from its slot at the race's barrier."""
    match slot:
        case BranchStopped():
            return Stopped(index)
        case BranchRaised(error=error) if all_refusals(error):
            return Refusal(index, str(next(leaves(error))))
        case BranchRaised(error=error):  # an `Unretryable` one: `transient_errors` took the rest
            return Raised(index, repr(error))
        case value if index in choice.winners:
            return Won(index, value)
        case value:
            return Unchosen(index, value)


def branch_slot(run: Callable[[Callable[[], Any]], Any], thunk: Callable[[], Any]) -> Any:
    """A branch's slot: its value, its stop as `BranchStopped`, or its exception as
    `BranchRaised`, so every branch ends before the barrier reads any of them."""
    try:
        return run(thunk)
    except Stopping:
        return BranchStopped()
    except Exception as raised:
        return BranchRaised(raised)


def race_clock() -> float:
    """The clock a race reads its deadline against, and the only clock a race reads at all.

    A race's branches run where the handler does rather than on the engine, so this is where a
    caller holds a deadline still: replace it and a deadline fires where the test says, which is
    what keeps a deadline out of the business of sleeping. A held clock decides WHETHER the
    deadline has arrived; it does not shorten the wait, so a clock held at or past the deadline
    is what a test uses to fire one."""
    return time.time()


def deadline_of(op: Race) -> float | None:
    """A race's deadline on the clock a handler reads, or `None` where the race waits for its
    branches. The op carries the instant the workflow handed in, so a restart converts the same
    value."""
    return None if op.deadline is None else op.deadline.timestamp()


@dataclass(frozen=True)
class Racing:
    """What a race's batch loop needs from the interpreter running it.

    `run(i)` drives branch `i` to its slot, `choose` saves a decided choice and publishes it,
    `decided` says whether one has been published, and `enclosed` whether an enclosing race's
    choice has stopped this race, which then decides nothing. `tree` is the race tree's lock,
    held from the `enclosed` check through the publish."""

    want: int
    branches: int
    run: Callable[[int], Any]
    choose: Callable[[Choice], None]
    decided: Callable[[], bool]
    enclosed: Callable[[], bool]
    tree: threading.Lock
    deadline: float | None = None

    def expired(self) -> bool:
        """Whether the deadline has arrived. The instant itself counts as arrived."""
        return self.deadline is not None and race_clock() >= self.deadline

    def undecidable(self) -> bool:
        """Whether the race can decide nothing further: it has saved a choice, or an enclosing
        race's choice has stopped it. Its wakes need no bound from then on, and a branch that
        raised before any choice joins this from the loop that saw it."""
        return self.decided() or self.enclosed()

    def in_time(self, ended_at: float) -> bool:
        """Whether a branch that ended at `ended_at` ended in time to win. The instant itself is
        late, so a race whose only success lands there answers `TimedOut`."""
        return self.deadline is None or ended_at < self.deadline

    def read(
        self,
        batch: list[int],
        slots: list[Any],
        ended_at: list[float],
        earlier: list[int],
        running: int,
        failing: bool,
    ) -> bool:
        """Apply one batch to the choice, and whether a programming error has come before any
        choice; from then on nothing is decided, so the race fails at its barrier with no loser
        told to stop.

        A batch can be empty, which is the race waking on its deadline with nothing new to read.

        A branch that ended at or after the deadline is itself evidence the deadline arrived, so
        it counts alongside the clock read: the two are read from different threads and a clock
        that steps back between them would otherwise answer impossible with a success in hand."""
        kinds = {i: settled(slots[i]) for i in batch}
        failing = failing or ("raised" in kinds.values() and not self.decided())
        if failing or self.decided() or self.enclosed():
            return failing
        won = [i for i in batch if kinds[i] == "won" and self.in_time(ended_at[i])]
        late = [i for i in batch if kinds[i] == "won" and i not in won]
        lost = set(batch) - set(won)
        if choice := decide(self.want, earlier, won, lost, running, self.expired() or bool(late)):
            with self.tree:
                if not self.enclosed():
                    self.choose(choice)
        earlier += won
        return False

    def before_any_branch(self, slots: list[Any], ended_at: list[float]) -> None:
        """Read the deadline before a branch runs, so a race whose deadline is already behind it
        decides before it launches one.

        Without this the timeout races each branch's first admission, and which wins is the
        interpreter's business rather than the race's: on free-threaded CPython the wake lands
        first, and under the GIL the branch does. A race restarted after a crash meets this,
        since its deadline can be long gone before the retry starts."""
        self.read([], slots, ended_at, [], self.branches, False)

    async def concurrently(self) -> list[Any]:
        """Every branch in its own thread, read in batches of whatever has ended at each wake.

        A deadline bounds one wake, the one it ends: a wake it cuts short reads an empty batch and
        the race decides on it. Every later wake is the barrier waiting for the branches to end,
        which no deadline shortens."""

        def ended(i: int) -> tuple[Any, float]:
            # Stamped in the branch's own thread: a loaded event loop can reach a finished task
            # well after it finished, and stamping here is what a branch ENDED at either way.
            return self.run(i), race_clock()

        slots: list[Any] = [None] * self.branches
        ended_at = [math.inf] * self.branches
        self.before_any_branch(slots, ended_at)
        tasks = {asyncio.create_task(asyncio.to_thread(ended, i)): i for i in range(self.branches)}
        pending, earlier, failing = set(tasks), list[int](), False
        bounded = self.deadline is not None and not self.undecidable()
        while pending:
            left = None
            if bounded and self.deadline is not None:
                left = max(0.0, self.deadline - race_clock())
            done, pending = await asyncio.wait(
                pending, timeout=left, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                slots[tasks[task]], ended_at[tasks[task]] = task.result()
            batch = sorted(tasks[task] for task in done)
            failing = self.read(batch, slots, ended_at, earlier, len(pending), failing)
            # Disarmed by what the race has become, never by a second look at the clock: an
            # advancing clock crossing the deadline BETWEEN the batch's read and this one would
            # otherwise leave the next wake unbounded with nothing decided, waiting on a
            # completion that may never come.
            bounded = bounded and not (self.undecidable() or failing)
        return slots

    def in_order(self) -> list[Any]:
        """Every branch in index order, each ending a batch of one.

        The deadline is read before the first branch and at each branch boundary after it, which
        is everywhere a sequential interpreter can read it. So a branch already running when the
        deadline arrives finishes, and every branch after it stops at its first admission. This is
        the path a deployed Absurd worker takes, since its SDK ctx is not `concurrent_safe`."""
        slots: list[Any] = [None] * self.branches
        ended_at = [math.inf] * self.branches
        earlier: list[int] = []
        failing = False
        self.before_any_branch(slots, ended_at)
        for i in range(self.branches):
            slots[i] = self.run(i)
            ended_at[i] = race_clock()
            failing = self.read([i], slots, ended_at, earlier, self.branches - i - 1, failing)
        return slots


def race_errors(slots: Iterable[Any]) -> ExceptionGroup | None:
    """The group a race raises at its barrier when a branch's programming error came before any
    choice, or `None`."""
    raised = [slot.error for slot in slots if settled(slot) == "raised"]
    return ExceptionGroup("race branches raised", raised) if raised else None


class RefusalDiverged(Unretryable):
    """A layer forwarded an op in a race branch that it refused on an earlier attempt.

    The earlier attempt's refusal is on record, and a loser may have been admitted to replay past
    the op on the strength of it, so running the op now would answer the race differently."""


def raise_error(error: Exception) -> Never:
    """`raise` as a call, for a thunk that must end in a raise."""
    raise error


def framed_refusal_record(key: Key) -> Key:
    """`refusal_record` for a key that carries its frames, as a reader of the store sees it: the
    record sits inside the same frames, since the branch wrote it through its framed ctx."""
    frames, identity = split_frames(key.stored())
    within = "".join(scope_prefix(Key.parse(frame)) for frame in frames)
    return refusal_record(Key.parse(identity)).prefixed(within)


def refusal_record(resolved: Key) -> Key:
    """Where a race branch records the domain's refusal of its step at `resolved`, the name the
    engine resolved for it.

    A refused call leaves no checkpoint, so without this a retry would call the domain again. The
    record's `next` says what a later attempt does on the same miss:

    | the domain raised       | `next`  | a later attempt                                      |
    |-------------------------|---------|------------------------------------------------------|
    | `Refused`               | `serve` | raises the recorded refusal without the call         |
    | a subclass of `Refused` | `call`  | calls the domain again, since a record cannot        |
    |                         |         | rebuild the subclass; a stopped loser stops there    |"""
    return compose_key(t"refusal;{resolved:domain=any}")


def gated_record(placed: Key) -> Key:
    """Where a race branch records that a layer refused its walk op at `placed`, the name the walk
    gave it. A later attempt whose layers forward or answer the op fails with `RefusalDiverged`."""
    return compose_key(t"gated;{placed:domain=any}")


type RefusalNext = Literal["serve", "call"]


def refusal_entry(reason: str, next_attempt: RefusalNext) -> dict[str, str]:
    """A refusal record's value."""
    return {"reason": reason, "next": next_attempt}


def served_refusal(stored: Any) -> bool:
    """Whether this refusal record is raised in place of its op, with no call made."""
    match stored:
        case {"next": "serve"}:
            return True
        case _:
            return False


def transient_errors(slots: Iterable[Any]) -> ExceptionGroup | None:
    """The group a race raises at its barrier after its choice, or `None`.

    | a loser's error after the choice               | at the barrier                         |
    |------------------------------------------------|----------------------------------------|
    | one a retry could clear                        | fails the attempt; the retry stops the |
    |                                                | loser at the op it never recorded      |
    | a `RefusalDiverged`                            | fails the task: the record and the     |
    |                                                | gates disagree                         |
    | any other `Unretryable`                        | the loser's `Raised` ending            |"""
    errors = [
        slot.error
        for slot in slots
        if settled(slot) == "raised"
        and (
            unretryable(slot.error) is None
            or any(isinstance(leaf, RefusalDiverged) for leaf in leaves(slot.error))
        )
    ]
    return ExceptionGroup("race branches raised", errors) if errors else None


def barrier_errors(slots: Iterable[Any], *, parked: bool) -> ExceptionGroup | None:
    """The group a gather round raises at its barrier, or `None`. A round with a `parked` branch
    is incomplete, so when every error is a refusal the group waits for the resume that completes
    it; any other error is raised now."""
    errors = [slot.error for slot in slots if isinstance(slot, BranchRaised)]
    if not errors or (parked and all(all_refusals(error) for error in errors)):
        return None
    return ExceptionGroup("gather branches raised", errors)


@dataclass(frozen=True)
class Attempt:
    """One execution of a task: its number, counted from 1, and the task's limit (`None` when the
    engine sets none)."""

    number: int
    limit: int | None

    @property
    def final(self) -> bool:
        """No retry follows this execution if it fails."""
        return self.limit is not None and self.number >= self.limit


def failing_leaf(raised: BaseException, attempt: Attempt) -> BaseException | None:
    """The error a task that raised `raised` on `attempt` fails of: a leaf a retry would raise
    again, the first of a failure that is only refusals, or on the last attempt its first leaf.
    `None` while a retry could still succeed.

    | failure                       | fails of                    | when             |
    |-------------------------------|-----------------------------|------------------|
    | holds an `Unretryable` leaf   | that leaf                   | this attempt     |
    | only refusals                 | the first refusal           | this attempt     |
    | anything else                 | its first leaf              | the last attempt |"""
    match unretryable(raised), all_refusals(raised), attempt.final:
        case (Unretryable() as leaf, _, _):
            return leaf
        case (None, True, _) | (None, False, True):
            return next(leaves(raised))
        case _:
            return None


def refuse_a_settled_checkpoint_under(wrapper: str) -> Never:
    """A fork seeds and steers its steps by name, and a settled checkpoint is not a step, so
    neither wrapper can say what a race inside a fork would have chosen."""
    raise NotImplementedError(
        f"a race cannot run under {wrapper}: its choice is a settled checkpoint, which the fork "
        "can neither seed nor steer. Run the race outside the fork."
    )


@runtime_checkable
class TaskContext(Protocol):
    """The durable execution surface a handler interprets ops onto.

    Both ``DurableHandler`` over a real Absurd ``ctx`` (Postgres, 0↔N) and the
    embedded SQLite engine (0↔1) satisfy this. The handler depends on this
    contract alone, so the workflow generator is engine-independent. The
    **durability semantics** below are what the crash-at-every-op /
    suspend-resume conformance suite pins.

    **Optional capability: ``peek_event(name: Key) -> tuple[bool, Any]``** (the
    await-in-gather park's branch-side surface; like ``concurrent_safe``,
    discovered by ``getattr``, deliberately NOT a required member so minimal
    test/script ctxs stay small): a non-suspending probe,
    ``(True, payload)`` iff an ``await_event(name)`` would return without
    suspending right now, else ``(False, None)`` with **no side effects** (no
    wait registration, no run-state change, no replay-bookkeeping perturbation;
    on Absurd, no occurrence-counter burn that would shift the checkpoint
    name a later ``await_event`` computes). A gather branch's await must not
    touch the engine's suspend machinery mid-round (an unsatisfied engine
    await flips the run state and forbids a second await per run), so the
    branch peeks; the single post-barrier re-arm is the only real
    ``await_event`` a run ever issues. A ctx WITHOUT ``peek_event`` keeps the
    legible await-in-gather NotImplementedError wall.

    **Optional capability: ``repark(name) -> None``** (same discovery rules as
    ``peek_event``: ``getattr``, deliberately NOT a required member): park this
    task so it re-queues ~immediately, **burning no attempt**: the no-burn
    resolution of the gather wake race (an emission landing between a branch's
    peek and the barrier leaves every parked branch already satisfied at the
    re-arm; the values are recoverable only by replay). ``name`` must be a
    deterministic step name, unforgeable by author step names, that the caller
    touches at most once per name (the wake CONDITION is part of it: a branch's
    next await may race the same gather again).
    An implementation may legally RETURN without parking (e.g. a stale wake-time
    checkpoint under ``name``), so the caller must keep a loud fallback
    (``GatherWakeRace``) rather than proceed. A ctx without ``repark`` resolves
    the race via that fallback: the engine's ordinary retry, one attempt burned.

    **Optional capabilities: ``peek_step(name: Key) -> tuple[bool, Any]`` and
    ``settle(name: Key, value) -> Any``** (discovered the same way): a race's checkpoint
    surface. ``peek_step`` reads a checkpoint under its full name with no side effects, so no
    occurrence counter moves. ``settle`` writes ``value`` unless the store already holds one
    under ``name``, and returns what the store holds; the write carries the engine's attempt
    fence, so a claim the task has moved past raises rather than writing. Neither applies an
    occurrence suffix: the name a race settles is positional and already unique in its thread.

    **Optional capability: ``await_until(name: Key, deadline: float) -> WaitOutcome``**
    (discovered the same way): ``await_event`` with a clock beside it. ``deadline`` is absolute,
    in epoch seconds, so every attempt waits on the instant the first one chose. It answers
    ``Arrived(payload)`` where an ``await_event`` would have returned, ``Expired`` once the
    deadline has passed, and otherwise parks on both. The outcome is RECORDED when it is first
    reached, so a replay serves it and an event landing after an expiry leaves that expiry
    standing: a wait with two possible answers has to say which one it gave. A ctx without this
    meets a named refusal (``DurableHandler._await_bounded``) rather than a dropped deadline.

    **Optional capability: ``step_resolved(name: Key, thunk: Callable[[Key], Any]) -> Any``**
    (discovered the same way): ``step``, with the thunk handed the name the checkpoint lands
    under, occurrence suffix included, in the caller's coordinates. A race branch's guard reads
    the refusal record at that name before calling the domain.
    """

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        """Run ``thunk`` AT MOST ONCE per ``name``; checkpoint its JSON-able
        result; on replay return the recorded value WITHOUT re-running ``thunk``.
        A thunk that returns is checkpointed, a ``Cancelled`` included; one that raises is
        not. A crash after the thunk runs but before commit re-runs it on resume (thunks must
        tolerate re-run until committed). Idempotent by ``name`` within a task.

        Params are positional-only: the contract is structural, so an
        implementation may name them as it likes (e.g. ``fn`` for ``thunk``).

        **``name`` is a ``Key``.** A checkpoint name is the thing replay BINDS to, so an
        f-string here is a `ty` error rather than a convention. The seam is typed, and the call
        sites need not be, because every checkpoint writer already goes through ``op_key``: the
        write side is typed and the read side parses. A name coming back out of a store goes
        through ``Key.parse``, a cast accepted until the writers close.

        A ``str`` here, unwrapped at each call site with ``op_key(op).stored()``, would leave the
        durable identity seam accepting any string at all."""
        ...

    def await_event(self, name: Key, /) -> Any:
        """Suspend the task durably; resume when an external ``emit_event(name,
        payload)`` arrives, returning ``payload``. The payload is recorded so
        replay re-binds BY NAME (deterministically named; no captured
        continuation). Survives worker death + fresh-worker resume.

        **``name`` is a ``Key``**, as ``step``'s argument above is. A park name is what an *answer*
        is delivered to, so the stakes are `step`'s and then some: Absurd delivers by name
        first-emit-wins across the whole queue, so two parks that collide on one name do not merely
        mis-key a checkpoint, they consume each other's answers. Every await writer already funnels
        through ``ops.event_name`` the way every checkpoint writer funnels through ``op_key``, so
        the guarantee is enforced on the write side here too.

        Typing it closes an injectivity hole: with a ``str`` here, a wrapper applying a frame
        prefix could write ``f"{self._prefix}{name}"`` beside a ``step`` arm that writes
        ``name.prefixed(self._prefix)``, and since ``op_key`` refuses ``Step(name="b/c")`` but
        not the await, a scoped await and a nested-scope await could compose one name.
        ``Key.prefixed`` is the named exit and an f-string here is a ``ty`` error."""
        ...

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        """Durable timer; does not pin a worker while parked; resumes at/after
        ``when`` across crashes.

        ``name`` is the timer's identity, minted above this seam by the walk: a sleep has no name
        of its own, and its wake time is DATA rather than identity, so the walk's ordinal is what
        distinguishes the n-th sleep in a frame from the n+1-th. It is minted above the seam
        because that is where the frame path exists to qualify it with.

        An engine that keeps a named wake-time checkpoint (Absurd) uses it; one that keeps the
        wake time on the task row alone (the SQLite engine) ignores it."""
        ...
