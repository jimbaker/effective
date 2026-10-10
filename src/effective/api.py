"""The authoring surface: typed wrappers over yielded ops.

Workflow authors call these and `yield from` them; they never `yield` a raw
op. Each wrapper carries the result type ``T`` through the ``Any`` send-channel
via a single contained ``cast``, so call sites get full inference. This is the
one place a bare ``yield`` is allowed — the smart-constructor layer; the
ast-grep rule (phase 1) exempts this module and forbids raw yields elsewhere.
"""

from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from string.templatelib import Template
from typing import Any, Never, assert_never, cast

from pydantic import BaseModel

from effective.cancel import served
from effective.choice import Answer, Chosen, Stopped
from effective.domain import Answers, AskLLM, CallTool, ChoiceAnswer, DomainOp, Judge
from effective.judgment import (
    NO_MATCH,
    Choice,
    TemplateError,
    battery,
    checked,
    field_names,
    gathered,
)
from effective.keys import Index, Key, Name, compose_key, scope_prefix
from effective.keys.marker import authored_key
from effective.ops import (
    Addressing,
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    LedgerRow,
    Minted,
    Race,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WaitOutcome,
    WorkflowOp,
    event_name,
)

type Effect[T] = Generator[WorkflowOp, Any, T]


def step[T](name: str, op: DomainOp[T], idempotency_key: Minted | None = None) -> Effect[T]:
    """The `Step` primitive, typed: every domain op reaches its handler through here.

    A cancelled op raises `OpCancelled` here, whichever handler recorded it."""
    raw = yield Step(name=name, op=op, idempotency_key=idempotency_key)
    return cast(T, served(name, raw))


def ask_llm[T](name: str, messages: Any, schema: type[T]) -> Effect[T]:
    """A convenience over `step` carrying an `AskLLM`; `name` is the op name."""
    return (yield from step(name, AskLLM(messages=messages, response_schema=schema)))


def judge[M: BaseModel](name: str, template: Template, output: type[M]) -> Effect[M]:
    """Ask every question in `template` in one `Judge`; `output`'s fields are the question names.

    The op name is minted as `judge:{name}`."""
    asked = battery(template)
    if (fields := field_names(asked.questions)) != set(output.model_fields):
        raise TemplateError(f"questions {sorted(fields)} != fields {sorted(output.model_fields)}")
    op = Judge(state=asked.state, questions=asked.questions, response_schema=Answers)
    answered = yield from step(compose_key(t"judge:{Name(name)}").stored(), op)
    # A durable handler loads the schema; the recording core hands back the value as canned.
    answers = Answers.model_validate(answered).root
    checked(asked.questions, answers)
    return output.model_validate(gathered(answers))


class _Selection(BaseModel):
    pick: ChoiceAnswer


def select(
    name: str,
    template: Template,
    candidates: Sequence[str] | Mapping[str, str | None],
    no_match: str = NO_MATCH,
) -> Effect[ChoiceAnswer]:
    """Choose among code-found `candidates`, or `no_match`; an answer outside them is refused.

    A mapping gives each candidate a description. The literal text ending `template` is the
    question."""
    match candidates:
        case str():
            raise TemplateError("candidates are a sequence of names, and a str is one name")
        case Mapping():
            options = dict(candidates)
        case Sequence():
            options = dict.fromkeys(candidates)
        case unreachable:
            assert_never(unreachable)
    if not options:
        raise TemplateError("select needs at least one candidate")
    if len(options) != len(candidates) or no_match in options:
        raise TemplateError("candidates must be distinct and exclude the no-match option")
    pick = Choice({**options, no_match: None})
    chosen = yield from judge(name, template + t"{pick}", _Selection)
    return chosen.pick


def direct_tool_key(name: str) -> Key:
    """The op key for a tool call, `tool:{name}`; a frame around the call, such as a ReAct turn's
    `d:{i}`, places it.

    `tool:` carries a hierarchical step name inside the hole as author path text rather than
    forging interior segments. A minter, so a test asks for the name rather than spelling it;
    `react.tool_key` refuses a model-chosen name this cannot compose."""
    return compose_key(t"tool:{Name(name)}")


def call_tool[T](tool: str, args: dict[str, Any], schema: type[T]) -> Effect[T]:
    """A convenience over `step` carrying a `CallTool`; the op name is MINTED from `tool`."""
    key = direct_tool_key(tool)
    return (yield from step(key.stored(), CallTool(name=tool, args=args, result_schema=schema)))


def await_event[T](
    name: str | Key, schema: type[T], addressing: Addressing = Addressing.RELATIVE
) -> Effect[T]:
    """Park until `name` is delivered, then resume with its payload decoded as `schema`.

    **`name` must be FORK-STABLE — scope it on the run's SUBJECT, never on its run id.** This is
    the one authoring rule the await surface has, and it is easy to get backwards, because the
    obvious way to make a name unique across runs is the one way that breaks forking:

        await_event(f"review:{message_id}", Decision)   # the subject — forkable
        await_event(f"review:{run_id}", Decision)       # REFUSED on any fork of this workflow

    A fork re-runs *this same generator* over the base's subject under a **different run id**, so
    a run-scoped name is a name the base never parked on. Measured, both arms, on a real fork
    (`tests/test_fork_sweep.py::decision_wf`): a run-scoped await gives `ForkedPrefixAwait: …
    awaited 'review:r-fork', which is not the fork point 'review:r-base'`, and run-scoped ledger
    ids give `SeedBoundaryError` for the same underlying reason — the child's key is not in the
    seed it was handed.

    **Uniqueness is a real requirement; it just is not yours.** Absurd delivers events by name
    across the whole queue, so a bare constant *does* collide between concurrent runs. The subject
    supplies uniqueness and fork-stability at once (a fresh `message_id` per logical unit of
    work), and the substrate supplies the rest:
    `RenamedAwaitCtx` parks a child at `fork:{child_run_id};{name}`, and a gather branch is
    prefixed per branch. **Reaching for the run id is doing the substrate's job with the one
    variable a fork changes.**

    Prefer `compose_key(t"review:{Segment(message_id)}")` to an f-string here: this is an
    *identity*, and the composer is what makes it injective. A composed `Key` is also what the
    substrate passes for its own parks, and the difference is enforced: author text may not name a
    reserved authority namespace or carry the frame delimiter (`ops.event_name`).

    **`addressing` is almost never yours to set, and it is a different axis from the SUBJECT
    rule above.** That rule is about what to EMBED in a name; this is about whether the handler's
    frames COMPLETE it. It defaults to `RELATIVE` — an author's await is completed by its frames,
    so two scopes awaiting the same bare name are two questions, and an emitter that knows the
    structure completes it with `qualified_event_name`. (`review:{message_id}` is subject-scoped
    *and* relative; the two rules do not compete.) Pass `ABSOLUTE` only when the emitter is a
    *different task* running from its own params and so cannot complete anything — the substrate
    does this for `fork.join_fork` and `compose.spawn_subagent_task`, the only two today."""
    raw = yield AwaitEvent(name=event_name(name), schema=schema, addressing=addressing)
    return cast(T, raw)


def await_until[T](
    name: str | Key,
    schema: type[T],
    *,
    deadline: datetime,
    addressing: Addressing = Addressing.RELATIVE,
) -> Effect[WaitOutcome[T]]:
    """Park until `name` is delivered or `deadline` passes, and say which happened.

    `await_event`'s arguments in `await_event`'s order, plus the instant, keyword-only so a
    swapped instant and schema is a type error. Its two rules hold here unchanged: the name is
    fork-stable, scoped on the run's SUBJECT, and `addressing` is almost never the author's to set.

    ``deadline`` is a wall-clock instant the caller already holds, so a workflow reads its clock
    through a `step` and hands the answer in; every attempt then waits on the instant the first
    one chose. The wait answers once — the engine records which ending it reached — so a replay
    serves that answer and an event landing after an expiry leaves it alone.

    ==============================  =========================
    the wait ends at                the answer
    ==============================  =========================
    the event                       ``Arrived(payload)``
    the deadline                    ``Expired()``
    ==============================  =========================

    `match` the two arms: `WaitOutcome` is closed, so a handler for both is the totality check.
    """
    raw = yield AwaitEvent(
        name=event_name(name), schema=schema, addressing=addressing, deadline=deadline
    )
    return cast(WaitOutcome[T], raw)


def append_ledger(row: LedgerRow) -> Effect[None]:
    """Append one row to the canonical, append-only record.

    `row` is a `LedgerRow`, not a dict, so the `event_id` is a `Key` that BOTH `ty` and Pydantic
    refuse to accept as a bare string. That is the point: the ledger is `UNIQUE(event_id)` and
    append-only, so a wrong id is committed silently and permanently — the one identity position
    where a mistake neither raises nor can be repaired. Domain fields ride as extras
    (`LedgerRow(event_id=…, kind="reviewed", decision=decision)`), and the stored payload is
    byte-identical to the dict form it replaces."""
    yield AppendLedgerRow(row=row)
    return None


def store_artifact[T](value: T, content_type: str) -> Effect[str]:
    raw = yield StoreArtifact(value=value, content_type=content_type)
    return cast(str, raw)


def sleep_until(when: datetime) -> Effect[None]:
    yield SleepUntil(when=when)
    return None


def respawn_generation(
    *,
    task: str,
    generation: int,
    state: Any,
    params: dict[str, Any],
    run_id: str,
    granted: int = 0,
) -> Effect[Never]:
    """Mint the generation-boundary op — **not the author surface**.

    `effective.combinators.respawn` is what an author calls; this is the typed wrapper it yields
    from, and it exists here for the reason every wrapper does: the determinism boundary requires
    a workflow-role module to `yield from` a wrapper rather than bare-`yield` an op, and
    `combinators.py` is workflow-role. So the one bare `yield` that mints a `Respawn` lives where
    every other op is minted.

    The name is longer than the house op/api-function rule (`gather`/`Gather`) would give,
    deliberately: `respawn` is already taken by the combinator, which IS the author's word for
    the mechanism (one mechanism, one word). Two different `respawn`s one module apart would be
    a near-neighbor collision.

    Returns `Never`: the handler ends the task on this op, so nothing follows it.
    """
    yield Respawn(
        task=task,
        generation=generation,
        state=state,
        params=params,
        run_id=run_id,
        granted=granted,
    )
    raise AssertionError(  # pragma: no cover - a handler that returns here is broken
        "respawn_generation: the handler returned from a Respawn op. That op ENDS the task; a "
        "handler that resumes the generator has not implemented the generation boundary."
    )


def gather[T](branches: Sequence[Callable[[], Effect[T]]]) -> Effect[list[T]]:
    """Run independent sub-workflows concurrently; join their results in branch order.

    The applicative combinator: ``results = yield from gather([b0, b1, ...])`` where each
    ``bi`` is a thunk returning an ``Effect``. Results come back indexed by branch position,
    which is also what makes replay independent of completion order. An effect, which the
    handler interprets (RecordingHandler runs the branches under structured concurrency;
    ReplayHandler re-binds the recorded result with no concurrency at all).

    Every branch runs to its end, and the branches' exceptions come out as one
    ``ExceptionGroup`` in branch order. When every leaf is a refusal, the group is thrown into
    this workflow, so ``except* Refused`` around the gather catches it; a parked branch leaves
    the round incomplete, and the refusals arrive on the resume that completes it. Any other
    error is a task-level failure, which retry-with-replay converges.
    """
    raw = yield Gather(branches=tuple(branches))
    return cast(list[T], raw)


def quorum[T](
    want: int,
    branches: Sequence[Callable[[], Effect[T]]],
    *,
    deadline: datetime | None = None,
) -> Effect[Answer[T]]:
    """Run branches concurrently and answer with the first `want` to succeed.

        match (yield from quorum(2, [ask_a, ask_b, ask_c], deadline=at)):
            case Chosen(winners=winners):
                ...
            case TimedOut(endings=endings):
                ...
            case Impossible(endings=endings):
                ...

    A refusal is a loss. Once `want` branches have succeeded, or too few can, the choice is saved
    and every other branch stops at its next op admission; an op already admitted runs to its
    end, and the race returns when every branch has ended. Winners come back in branch-index
    order. `want` of 0 answers at once without starting a branch, and a `want` above the number of
    branches is refused as a composition error.

    `deadline` is an instant the caller already holds, read through a `step` as `await_until`
    reads one, and it is a third way for the same choice to be decided rather than a second
    authority: reaching it makes the parent decide, and `TimedOut` names what it decided. A
    branch that succeeds at the instant itself is late, so `Chosen` means every winner landed
    strictly before it. Without a deadline a race waits for its branches, and `TimedOut` cannot
    happen."""
    if want == 0:
        return Chosen((), tuple(Stopped(i) for i in range(len(branches))))
    raw = yield Race(want=want, branches=tuple(branches), deadline=deadline)
    return cast(Answer[T], raw)


def race[T](
    branches: Sequence[Callable[[], Effect[T]]], *, deadline: datetime | None = None
) -> Effect[Answer[T]]:
    """`quorum` with one winner: the first branch to succeed, the lowest index among those that
    finish together."""
    return (yield from quorum(1, branches, deadline=deadline))


def scoped[T](scope: Key, body: Callable[[], Effect[T]]) -> Effect[T]:
    """Run `body` with every key it mints namespaced under `scope`.

        total = yield from scoped(compose_key(t"rec:{i}"), lambda: leaf(chunk))

    The *handler* applies a scope, in the same position and by the same mechanism as a gather's
    `gather:{g},{i};` branch coordinate, so the scope grammar needs no policing: no author threads
    one through a callback or splices it onto the front of a name. The author writes the op's own
    name and nothing else.

    **`scope` is a `Key`.** Every scope atom is a tag plus fields (`rec:0`, `fold:1,2`, `d:3`), so
    it is composed by the one composer and is injective by construction. Nesting is nesting:
    `scoped(a, lambda: scoped(b, body))` namespaces under `{a};{b};`, so a step `s` inside
    `scoped(a:1, scoped(b:2, …))` records as `a:1;b:2;step:s`.

    The body is a thunk (like a gather branch) rather than an already-started generator, so the
    handler owns when it runs, which is what lets replay re-enter the scope by re-execution
    instead of resuming a captured frame."""
    raw = yield Scoped(scope=scope, body=body)
    return cast(T, raw)


@dataclass(frozen=True)
class GatherBranch:
    """One gather-branch frame on the path to a park — the coordinate a handler assigns.

    ``gather`` is the gather's 0-based ordinal among the GATHERS of its thread of control: the
    workflow body or an enclosing gather branch, each of which starts the count, and every
    ``scoped`` body inside it, which continues the count. It counts no other ops. ``index`` is the
    branch position."""

    gather: int
    index: int


def gather_frame(g: int, i: int, qualified: Key) -> Key:
    """A gather branch's frame around an inner key — `gather:{g},{i};{qualified}`.

    The handlers prefix a branch's keys with the same coordinate through `gather_prefix`, and
    this function composes the whole framed key for a caller holding the inner key, so the
    coordinate is spelled in one grammar. A frame that drifts is a checkpoint nobody can find
    again.

    **The terse parameter names are deliberate.** A template's hole EXPRESSIONS become the
    registry's field names, so moving a template renames the source map — `explain()` reported
    `{"gather": …, "index": …, "inner": …}` the moment this function existed. Keeping `g`/`i`/
    `qualified` keeps the map byte-identical, so extracting the minter is a pure refactor; they
    are also the names a dozen docstrings already use for this coordinate. Renaming the published
    fields is a separate decision from moving the code, and should look like one."""
    # `domain=any`, because this mints a FRAME and a frame wraps EITHER. `qualified_event_name`
    # hands it a park's wire address; the composition oracle hands it `step:leaf`, an op's own
    # identity, exactly as the handlers do with their `gather_prefix(g, i)`. A frame is not
    # a position that narrows, so there is nothing here for a domain to say — which is what the
    # escape is for, and the only place in the substrate that earns it.
    #
    # Frames appearing INSIDE a narrowed position are a different question and are handled:
    # `grammar.admits` looks past them, so `parked.py` can declare `address` for a value that
    # arrives as `gather:0,0;ev0:r1`.
    return compose_key(t"gather:{Index(g)},{Index(i)};{qualified:domain=any}")


def qualified_event_name(*path: GatherBranch | Key, name: str) -> Key:
    """The fully-qualified event name an EMITTER uses to wake a park — the importable anchor
    of the qualified-emitter contract, for **every** kind of frame a park can sit under.

        qualified_event_name(GatherBranch(0, 1), compose_key(t"rec:{0}"), name="grant:d1")
        # -> Key('gather:0,1;rec:0;grant:d1')

    A park's name on the wire is the path to it, outermost frame first, then the bare name the
    workflow passed to ``await_event``. Two frame kinds exist and they compose freely, so they
    are one argument list rather than two helpers: a **`GatherBranch`** contributes
    ``gather:{g},{i};`` and a **`Key`** contributes a ``scoped(...)`` atom as a leading term. Both
    are handler-side and invisible at the await's call site, which is exactly why an emitter
    needs this: emitting the bare ``name`` does not resolve the park (the conformance suite
    pins that sharp edge on both engines).

    **This replaces ``gather_event_name``, which covered one frame kind and left the composite
    case to hand-assembly** — the real shape is `gather:{g},{i};{scope};{name}`, and callers
    were concatenating two spellings by hand at an identity boundary. Nesting is now the
    argument list, not nested calls.

    Returns a ``Key``, not text: this is an identity, and the old helper composed a ``Key``
    only to return its string, laundering both ends. Take ``.stored()`` at the wire boundary
    where the emitting API needs text.

    Two orthogonal rules meet here: ``name`` must already carry its own uniqueness — Absurd
    events are global per queue and no prefix supplies it — and the path qualifies per frame.
    **Scope that name on the run's SUBJECT, not its run id**; see ``await_event`` for the rule
    and the measured failure (a run-scoped await makes the workflow unforkable). The
    substrate's own run-scoped parks are not a counterexample: ``budget-grant:{run_id},{trip}``
    is composed by the handler, and a fork that hits one in its prefix is also refused unless
    ``transplanted`` relaxes it (``SeedingCtx``), which grants authored names nothing.

    ``name`` goes through ``authored_key``, so text may not carry an occurrence: the suffix has
    one producer, ``Key.occurrence``, which the walk applies where a ``SETTLEMENT`` namespace is
    asked again. An emitter waking that second ask asks the composed key for it::

        qualified_event_name(GatherBranch(0, 0), name=grant.stored()).occurrence(2)

    which is the registration ``parked`` reports, rather than appending ``#2`` to ``name``.

    Inside a workflow you never need this — awaits are prefixed automatically."""
    qualified = authored_key(name)
    for frame in reversed(path):
        match frame:
            case GatherBranch(gather=g, index=i):
                # The composer, not a hand-built prefix: a `Key` splices in TERMINAL position,
                # which is exactly this shape — and it is the same composition the handler does.
                qualified = gather_frame(g, i, qualified)
            case Key():
                qualified = qualified.prefixed(scope_prefix(frame))
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return qualified
