"""Vector budget for the combinators: the structural ceilings and the measured (dollar) one.

A ``Budget`` bounds a workflow's *structural* consumables: ``depth`` (nesting), which is what
bounds recursive subagents. Structural consumables are **deterministic**: the count is threaded
explicitly through the workflow, exactly like ``descend``'s ``remaining`` (``combinators.py``)
or ``recurse``'s branch list, so the trip is a pure check with no handler-side state. The child
gets a decremented *copy* (``descend_one``) and nothing reads a shared counter, so there is no
read-modify-write race to guard.

On exhaustion the behavior splits by whether the surface can durably suspend:

- a **durable combinator** (``descend`` over ``DurableHandler``) can PARK on a grant
  event and ask a human "grant more depth, or answer with what you have?"
  (``on_exhaust="park"``);
- a **spawn** past its task's depth is refused by the durable handler (``Refused``)
  before any child is enqueued, and a child carries one level less
  (``on_exhaust="fail"``, the default).

``MeasuredBudget`` is the dollar ceiling, enforced by the durable handler against a
replay-derived meter. Tokens, a cumulative breadth count and wall time are not budgeted here.
Only a field that is enforced belongs in a budget: a dead field is a silent lie.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Never, assert_never

from pydantic import BaseModel, ConfigDict, Field

from effective.govern import (
    Ask,
    Exceeded,
    GateState,
    Policy,
    Proceed,
    Refuse,
    park,
    policies_of,
    run_answers_for,
)
from effective.keys import AuthorityTag, Index, Key, Run, Scope, compose_key
from effective.keys.grammar import TAG_SEPARATOR, KeySyntaxError
from effective.layers import OpLayer
from effective.ops import CompositionRefused, WorkflowOp

BUDGET_GRANT = AuthorityTag("budget-grant", scope=Scope.ACCRUAL)
"""The measured trip's grant namespace — its names ARE the authorization. `ACCRUAL` because a
grant RAISES the run's ceiling: one delivered while gating op1 is still in force at op2 and must
not be re-requested. That is the feature, not an alias."""

GENERATION_GRANT = AuthorityTag("generation-grant", scope=Scope.ACCRUAL)
"""The per-generation ceiling's grant family: a separate tag from `budget-grant`. One tag has
one shape, and the key registry refuses a namespace minted at two arities; a reader can also tell
from the name which question they are being asked."""

DEPTH_GRANT = AuthorityTag("depth-grant", scope=Scope.SETTLEMENT)
ROUND_GRANT = AuthorityTag("round-grant", scope=Scope.SETTLEMENT)
"""A search's rounds, settled once per search as a depth grant is once per descent."""
CHAIN_GRANT = AuthorityTag("chain-grant", scope=Scope.SETTLEMENT)
"""The STRUCTURAL grant families: `descend`'s levels and `respawn`'s generations.

`SETTLEMENT`, unlike the two above: a grant of `add_depth=2` authorizes ONE descent to drill
further, and the next descent asks again. Two `descend(...)` calls would otherwise compose one
name, so the first answer would settle the second; the walk (`handlers/base.placing`) applies the
coordinate that separates them, which is why the template here carries no occurrence field.

**`Addressing.RELATIVE`**, both of them: an enclosing frame completes the name, and the emitter
reads the parked name off the engine rather than composing one from its own params
(`combinators.human_grant`; the criterion is stated once at `ops.awaits_an_absolute_name`)."""

type OnExhaust = Literal["park", "fail"]


def budget_grant_name(run_id: str, trip: int) -> Key:
    """The park a measured (dollars) budget asks on at its `trip`-th exhaustion.

    Named for the NAMESPACE it mints, like its two siblings. It was `trip_grant_name`, which named
    the coordinate instead — so the function and the key it composes were two different words, and
    a reader grepping either one missed the other. There is no `trip-grant:` namespace."""
    return compose_key(t"{BUDGET_GRANT}:{Run(run_id)},{Index(trip)}")


BUDGET_GRANT_LIKE = f"{BUDGET_GRANT}{TAG_SEPARATOR}%"
"""A SQL ``LIKE`` PREFILTER for the namespace — the tag and nothing else.

It carries no identity: both VOI bridges narrow the rows to one run by *parsing* them
(`names_a_budget_grant`), never by gluing a `run_id` into the pattern. A glued form
(`f"budget-grant:{run_id}:%"`) is a partial reader of the grammar and breaks silently when the
coordinate separator changes: `grants` comes back empty, `measured_drive` re-parks at trip 0, and
nothing raises."""


def names_a_budget_grant(key: Key, run_id: str) -> bool:
    """Does `key` name a budget grant of `run_id`? — the READ side of `budget_grant_name`.

    Derived from the writer rather than spelled beside it, which is the property the pair exists
    for: a reader that re-implements a name is a reader that drifts out of it.

    TOTAL, because its callers sweep a store: an event row that is not in the language is not a
    grant, which is an answer rather than an error. (Written as plain code, not a `match` — the
    tag is a bare module name, and a bare name in a class pattern's keyword position CAPTURES
    rather than compares, so `case Term(tag=BUDGET_GRANT)` would match every tag.)"""
    try:
        head, *_ = key.terms()
    except KeySyntaxError:
        return False
    return (
        head.tag == BUDGET_GRANT
        and len(head.coordinates) == 2
        and head.coordinates[0].path == run_id
    )


def depth_grant_name(run_id: str, *, generation: int, depth: int) -> Key:
    """The park `descend` asks on when its level budget runs out.

    The axis is the tag: `depth-grant:` and `chain-grant:` are separate namespaces the registry can
    see.

    **`generation` is the cross-task coordinate**, the half no within-frame ordinal reaches.
    A `respawn` chain keeps `run_id` stable across generations and each generation is a fresh
    task, so `(run_id, depth)` alone would compose in generation 1 the name generation 0 parked
    on, and Absurd would answer it at once from the earlier grant
    (`tests/test_grant_aliasing.py::test_two_generations_share_one_grant_across_tasks`). The
    within-task occurrence coordinate (`handlers/base.placing`) cannot see this: two
    generations are two tasks, with two `FramePosition`s and two checkpoint counters.

    The generation is **its own slot**. A two-field variant without it would not separate from
    this one: the named `depth=` coordinate is optional, which widens each variant's arity to a
    range, the two ranges overlap, and the key registry refuses the pair. A `g{n}` suffix glued to
    literal text cannot be told from the value on the wire, and the composer refuses it.

    **0 outside a chain** is a real generation: an unchained run IS generation 0, as
    `Chain.from_params` reads it. `combinators.refill_levels` and `combinators.human_grant`
    read `ops.CHAIN_GENERATION`, which `combinators.respawn` sets. The parameter has no
    default, because a defaulted identity coordinate aliases silently; a caller outside a
    chain passes 0 explicitly.

    **The coordinates are keyword-only, listed in wire order.** Two adjacent `int`s swap into a
    different, equally well-formed key with no error anywhere, and decoding a key into its named
    fields and feeding them back must give the same key. The wire order narrows: a run contains
    generations, which contain depths."""
    return compose_key(
        t"{DEPTH_GRANT}:{Run(run_id)},{Index(generation)},depth={Index(depth):default=0}"
    )


def round_grant_name(run_id: str, *, generation: int, rounds: int) -> Key:
    """The park `tree_search` asks on after `rounds` rounds, when its granted rounds run out."""
    return compose_key(t"{ROUND_GRANT}:{Run(run_id)},{Index(generation)},{Index(rounds)}")


def chain_grant_name(run_id: str, generation: int) -> Key:
    """The park a `respawn` chain asks on when its generation budget runs out."""
    return compose_key(t"{CHAIN_GRANT}:{Run(run_id)},{Index(generation)}")


# The reserved key under which a spawned child's remaining ``depth`` travels in the
# serializable spawn ``params``. Depth crosses a task-spawn boundary as *recorded*
# state, never a shared Python object — a spawn lands on an arbitrary worker, so a
# shared counter is unrepresentable there. ``spawn_subagent_task`` writes
# it; ``Budget.from_spawn_params`` reads it back on the child.
BUDGET_DEPTH_PARAM = "budget_depth"
BUDGET_GENERATIONS_PARAM = "budget_generations"


def _spawned_bound(value: object) -> int | None:
    """A bound read back from spawn params. Absent is unbounded, and anything but an int counts as
    spent: the params were written by whatever yielded the spawn, a model included."""
    match value:
        case None:
            return None
        case bool():
            return 0
        case int():
            return value
        case _:
            return 0


class Grant(BaseModel):
    """A budget-park's resolution: how much more of each bounded axis, or stop.

    Every additive field carries ``ge=0``, because this is where a human's raw JSON reaches
    combinator state and a negative grant — one typo from ``1`` — would un-bound the axis it
    is meant to bound. ``stop`` means "answer with what you have": an explicit best-so-far, kept
    distinct from a malformed or absent grant so the exhaustion arm is unambiguous.
    ``add_dollars`` refills the *measured* ceiling at a park; ``add_depth`` refills the
    *structural* one."""

    model_config = ConfigDict(extra="forbid")
    """An unknown field is a REFUSAL, not something to ignore. Pydantic's default silently drops
    it, which turned a wrong-schema payload into a valid all-defaults grant: `{"extra_levels": 3}`
    validated to `Grant()` and granted zero levels, with nothing in the log. `ge=0` already
    promised this boundary would be loud; extra-ignore defeated that promise for the one mistake
    an emitter actually makes, which is using the other schema's field name."""

    add_depth: int = Field(0, ge=0)
    add_generations: int = Field(0, ge=0)
    """How many more GENERATIONS a respawn chain may run. Same `ge=0` discipline as
    `add_depth`: a negative refill would un-bound the axis it is meant to bound, and `stop`
    is how you say "no more" rather than a negative number."""
    add_dollars: float = Field(0.0, ge=0.0)
    add_rounds: int = Field(0, ge=0)
    """How many more rounds a `tree_search` may run."""
    stop: bool = False
    """Answer with what the run has. A delivered stop stays delivered, so every later trip that
    reaches the same grant name is refused again."""


@dataclass(frozen=True)
class Budget:
    """Structural bounds threaded through the combinators — build-now: ``depth`` only.

    ``None`` on an axis means *unbounded* there (today's behaviour, so an absent
    ``Budget`` and ``Budget()`` both change nothing). ``on_exhaust`` selects
    park-and-ask (durable combinators) versus raise (the default, and the only
    option across the non-durable subagent boundary).

    ``run_id`` scopes the park, exactly as ``MeasuredBudget.run_id`` does: Absurd events are
    global per queue, so the run id is what keeps two runs' grants apart. It is a coordinate
    and never a caller-supplied pre-composed name: a whole name would leave the tag inside an
    interpolation, where ``compose_key`` cannot see it, hiding these authority names from the key
    registry."""

    depth: int | None = None
    generations: int | None = None
    """How many more GENERATIONS a `respawn` chain may run. `None` is unbounded, exactly as
    `depth` is: the two are the same shape on different axes, so this threads through spawn
    params the same way, since a cross-task budget needs a params carrier."""
    on_exhaust: OnExhaust = "fail"
    run_id: str | None = None

    def descend_one(self) -> Budget:
        """The child's budget: one level of ``depth`` consumed, as an immutable copy.

        The copy is immutable: the parent keeps its own value, the child drills on the
        decremented one, and no two frames share a mutable counter, so the structural gate has
        no race to lock (contrast the *measured* meter, which does). Unbounded
        (``depth is None``) passes through unchanged."""
        if self.depth is None:
            return self
        return replace(self, depth=self.depth - 1)

    def next_generation(self) -> Budget:
        """The next generation's budget: one `generations` consumed, as an immutable copy.

        `descend_one`'s twin on the chain axis. Unbounded (`generations is None`) passes through
        unchanged, so an endless poller costs nothing to express."""
        if self.generations is None:
            return self
        return replace(self, generations=self.generations - 1)

    def generations_exhausted(self) -> bool:
        """Whether this chain may go round again without a grant."""
        return self.generations is not None and self.generations <= 0

    def depth_exhausted(self) -> bool:
        """True when a bounded ``depth`` has been fully consumed. Pure check —
        the caller decides whether that means park (durable) or raise."""
        return self.depth is not None and self.depth <= 0

    @classmethod
    def from_spawn_params(
        cls,
        params: Mapping[str, Any],
        *,
        on_exhaust: OnExhaust = "fail",
        run_id: str | None = None,
    ) -> Budget:
        """A spawned task's budget, read from the params it was enqueued with.

        An absent bound is unbounded. ``on_exhaust`` and ``run_id`` are the task's own: only the
        bounds cross the spawn boundary."""
        return cls(
            depth=_spawned_bound(params.get(BUDGET_DEPTH_PARAM)),
            generations=_spawned_bound(params.get(BUDGET_GENERATIONS_PARAM)),
            on_exhaust=on_exhaust,
            run_id=run_id,
        )

    def stamp(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """``params`` carrying no more depth than this budget; a smaller request stands."""
        match _spawned_bound(params.get(BUDGET_DEPTH_PARAM)), self.depth:
            case _, None:
                return dict(params)
            case int() as requested, int() as allowed if requested < allowed:
                return {**params, BUDGET_DEPTH_PARAM: requested}
            case _, allowed:
                return {**params, BUDGET_DEPTH_PARAM: allowed}


@dataclass(frozen=True, kw_only=True)
class MeasuredBudget:
    """The *measured* (dollar) ceiling the durable handler enforces via a replay-derived
    trip; the counterpart to the structural ``Budget`` above.

    Distinct from ``cost.CostBudget``, the telemetry meter's own ceiling, refused *inside* the
    checkpoint thunk: this ceiling is enforced ABOVE ``ctx.step`` against the handler-owned,
    replay-derived meter, so its trip re-derives at the same op on replay.

    On an over-budget model call at a **sequential** program point:

    | ``on_exhaust``       | does                                                          |
    |----------------------|---------------------------------------------------------------|
    | ``"park"`` (default) | parks on ``budget-grant:{run_id},{trip}``; a grantor answers  |
    |                      | with more dollars or a ``stop``                               |
    | ``"fail"``           | raises ``BudgetRefused`` at once, for an unattended batch      |

    ``run_id`` scopes the grant event: Absurd events are global per queue, the same rule
    ``human()`` and ``descend`` follow.

    Measured spend is gated at sequential points only: two branches of a ``gather`` tripping in
    one round would compute the same ``trip_n`` and so the same grant name, so a branch handler
    carries no ``MeasuredBudget``.

    **Two ceilings, because a chain makes them different questions.** Once a run is a `respawn`
    chain, "may it spend more?" splits in two:

    | field              | bounds                                                        |
    |--------------------|---------------------------------------------------------------|
    | ``overall``        | spend **across every generation**: a cap                      |
    | ``per_generation`` | spend by any one generation: an allowance that re-arms; not   |
    |                    | enforced yet                                                  |

    They read as spoken English together, *"$100 overall, $5 per generation"*, and each has its
    own grant family, because they ask different questions of whoever answers: `budget-grant` is
    about the run, `generation-grant` about this generation (see `enforce_generation`). A ceiling
    that silently meant per-generation would give a chain a fresh ceiling every generation, spend
    linear in chain length (`tests/test_carrier_audit.py` measured 5.3x over four generations).

    **No handler enforces ``per_generation`` yet** (`enforce_generation` is not wired), so a
    budget with only ``per_generation`` set limits nothing at runtime: ``enforce_measured``
    clears every charge against it."""

    # `kw_only`: two optional ceilings make a positional build a SILENT mis-assignment — the
    # conformance harness passed `MeasuredBudget(budget_limit, rid, on_exhaust)` and the limit
    # landed in `run_id`. Caught here by the tests; keyword-only means it cannot recur.
    run_id: str
    overall: float | None = None
    per_generation: float | None = None
    on_exhaust: OnExhaust = "park"
    """**PARK by default.**

    The park is proven on both engines (`test_measured_trip_parks_grants_and_resumes`, the scoped
    variant, and `test_enforce_measured_conformance`).

    **What it buys, and why a refusal is the worse answer.** A trip parks on
    `budget-grant:{run_id},{trip}`; `parked.py` is unfiltered so a dashboard sees the park AND
    its name; a grantor answers with more dollars or `stop`, and the run RESUMES from the trip.
    Raising ends the run, and inside a fork it ends it badly: the child crashes, the engine
    retries it to death, and the parent sits on its join forever with no reason recorded
    anywhere. Reporting a refusal would at least answer the parent; parking answers it AND lets
    the work continue, which is the difference between a diagnosis and a recovery.

    `"fail"` stays available and stays right for an unattended batch, where there is nobody to
    answer a park and dying at the ceiling is the point.

    **Its sibling `Budget.on_exhaust` deliberately did NOT flip.** The structural budget is
    enforced across the in-process subagent boundary
    (`effective.interpreters.tools.subagent_runner` over `LocalCtx`), which cannot park at all:
    an `await_event` there raises, so `"fail"` is the only option there, as this module's header
    says."""

    def __post_init__(self) -> None:
        # Assembly-time, loudly — the shape `govern()` uses for a gate with no policies and
        # `descend` for run_id-with-grantor. A budget that cannot bind is a
        # misconfiguration, not a default, and finding out at the first metered call is worse.
        if self.overall is None and self.per_generation is None:
            raise ValueError(
                "MeasuredBudget needs `overall` (the run's cap across every generation) or "
                "`per_generation` (each generation's allowance), or both — one with neither "
                "limits nothing. If you mean 'no ceiling', pass no budget at all."
            )
        if (
            self.overall is not None
            and self.per_generation is not None
            and self.per_generation > self.overall
        ):
            raise ValueError(
                f"MeasuredBudget: per_generation={self.per_generation} exceeds "
                f"overall={self.overall}, so it can never bind — the run's cap trips first, "
                "every time. That reads like a limit and is a no-op; drop it, or raise `overall`."
            )


# --- the measured trip transition --------------------------------------------------------
#
# The classification a driver acts on, in the shared domain module so both interpreters import
# the one definition: the in-process `measured_drive` (`effective.fork`) and the durable
# `DurableHandler._enforce_measured` (`effective.handlers.absurd`). Formalized once in
# `formal/lean/Effective/EnforceMeasured.lean` and conformance-pinned in
# `tests/test_enforce_measured_conformance.py` (both interpreters vs the machine-derived rows).


@dataclass(frozen=True)
class Cleared:
    """Trip outcome: the ceiling is clear (after folding any grants) — proceed."""

    granted: float
    trips: int


@dataclass(frozen=True)
class Parked:
    """Trip outcome: over the ceiling, no grant available — park on this event name."""

    name: Key


type TripOutcome = Cleared | Parked | Exceeded


def _trip(
    meter_cost: float,
    ceiling: float,
    on_exhaust: OnExhaust,
    name_of: Callable[[int], Key],
    granted: float,
    trips: int,
    grants: Mapping[Key, Grant],
) -> TripOutcome:
    """The transition itself, shared by both ceilings — one loop, so they cannot drift.

    While spend is over ``ceiling + granted``: **fail first** (``on_exhaust="fail"``), else
    classify the grant this trip is named by. Pure and total; the driver realizes
    ``Parked``/``Exceeded`` as it must (an in-process return vs. the durable suspend/raise).

    The loop is the TRAMPOLINE and the `match` is the TRANSITION: the loop decides when, the
    decision table decides what. As a table the total set of outcomes is visible and an omission
    is a missing arm. The guard is `g.stop or g.add_dollars <= 0.0` rather than
    `case Grant(add_dollars=0.0)`, which would silently miss `stop=True` and a negative.

    Only the ceiling and the NAME differ between the two callers, which is why this is one
    function taking `name_of` rather than two loops that would have to be kept in step."""
    while meter_cost >= ceiling + granted:
        if on_exhaust == "fail":
            return Exceeded(meter_cost, ceiling + granted)
        match grants.get(name_of(trips)):
            case None:  # nobody has answered this trip's park yet
                return Parked(name_of(trips))
            case Grant() as g if g.stop or g.add_dollars <= 0.0:  # answered: stop here
                return Exceeded(meter_cost, ceiling + granted)
            case Grant() as g:  # answered with more room: refill and re-check
                granted += g.add_dollars
                trips += 1
    return Cleared(granted, trips)


def enforce_measured(
    meter_cost: float,
    budget: MeasuredBudget,
    granted: float,
    trips: int,
    grants: Mapping[Key, Grant],
) -> TripOutcome:
    """The OVERALL measured trip: the run's cap, across every generation.

    Consults ``budget-grant:{run_id},{trip}``. The Lean model and `formal/enforce_vectors.json`
    pin the TRANSITION (limit, meter, fail, grants → outcome) and carry no name.

    Placement, *which op fires this*, is the caller's named decision, outside this function.
    Returns `Cleared` unchanged when no `overall` is set: a budget may carry only a
    per-generation allowance."""
    if budget.overall is None:
        return Cleared(granted, trips)
    return _trip(
        meter_cost,
        budget.overall,
        budget.on_exhaust,
        lambda n: budget_grant_name(budget.run_id, n),
        granted,
        trips,
        grants,
    )


def enforce_generation(
    meter_cost: float,
    budget: MeasuredBudget,
    generation: int,
    granted: float,
    trips: int,
    grants: Mapping[Key, Grant],
) -> TripOutcome:
    """The PER-GENERATION measured trip — this generation's allowance, which re-arms.

    **Not wired: no handler calls this.** The transition below is correct and the naming argument
    holds, but nothing drives it, so a per-generation ceiling is not enforced at runtime today.
    Pinned by `tests/test_respawn_durable.py::test_a_per_generation_ceiling_is_ACTUALLY_ENFORCED`,
    an `xfail(strict=True)` that fails the day it is wired.

    Consults ``generation-grant:{run_id},{generation},{trip}``. The generation is IN the name,
    and that is the whole point: `budget-grant:{run_id},{trip}` is byte-identical across
    generations (stable run id by design, trip counter restarting per task), so generation *n*'s
    human grant answered *n+1* instantly — no park, no human, no record that two generations
    shared one authorization (`tests/test_respawn_hazards_pending.py` pins the engine semantics
    that make it so).

    An emitter can compose this: a generation is a plain ordinal, recoverable by one indexed
    ledger query on the `respawned` event. That is why the objection which kept frames OUT
    of `budget-grant` does not transfer — that objection was that a *spend-dependent* scope is
    uncomputable, and a generation is not spend-dependent.

    `meter_cost` here is THIS generation's spend, not the chain's."""
    if budget.per_generation is None:
        return Cleared(granted, trips)
    return _trip(
        meter_cost,
        budget.per_generation,
        budget.on_exhaust,
        lambda n: compose_key(
            t"{GENERATION_GRANT}:{Run(budget.run_id)},{Index(generation)},{Index(n)}"
        ),
        granted,
        trips,
        grants,
    )


# --- the measured trip as a `govern` policy -----------------------------------------------


def _tripped(budget: MeasuredBudget, name: str) -> str:
    """Which ceiling this park is about, read off the grant name's own tag.

    The two families are distinguishable by construction (`budget-grant` vs
    `generation-grant`), so the prompt does not need to be told — it can look."""
    if name.startswith(f"{GENERATION_GRANT}:"):
        return f"{budget.per_generation:.4f} for this generation"
    return f"{budget.overall:.4f} overall" if budget.overall is not None else "the run's cap"


def as_policy(budget: MeasuredBudget, *, name: str = "budget") -> Policy:
    """The measured trip as a `govern` policy, driving the same `enforce_measured` as
    `DurableHandler` and `fork.measured_drive`.

    The spend is the gate's `GateState.meter`, the task's replay-derived spend.

    The park name is the gate's. The trip's `budget-grant:{run_id},{trip}` names are only the
    grant lookup keys, rebuilt positionally from the answers. Answers accrue across the run
    (`run_answers_for`): a grant raises the run's ceiling and stays in force at the next op, where
    a permission approval settles the one op it answered (`answers_for`)."""
    return BudgetPolicy(budget, name)


@dataclass(frozen=True)
class BudgetPolicy:
    """`as_policy`'s policy, a value so an assembly can tell a gate that drives a budget."""

    budget: MeasuredBudget
    name: str = "budget"

    def __call__(self, op: WorkflowOp, state: GateState) -> Any:
        budget, name = self.budget, self.name
        grants = {
            budget_grant_name(budget.run_id, i): _as_grant(answer)
            for i, answer in enumerate(run_answers_for(state, name))
        }
        spent = self._spent(state)
        match enforce_measured(spent, budget, 0.0, 0, grants):
            case Cleared():
                return Proceed()
            case Parked():
                return park(
                    Ask(
                        name,
                        # The ceiling that tripped, named: with two ceilings, "of $X" alone is a
                        # prompt a human cannot answer well.
                        f"grant more budget? spent ${spent:.4f} of ${_tripped(budget, name)}",
                        {
                            "spent": spent,
                            "overall": budget.overall,
                            "per_generation": budget.per_generation,
                            "trips": len(grants),
                        },
                    )
                )
            case Exceeded() as exceeded:
                return Refuse((exceeded.reason,), exceeded)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _spent(self, state: GateState) -> float:
        match state.meter:
            case None:
                _refuse_unmetered(self.name)
            case usage:
                return usage.cost


def refuse_two_drivers(budget: MeasuredBudget | None, layers: Sequence[OpLayer[Any]]) -> None:
    """Refuse a handler `budget` beside a gate that holds a budget policy.

    Each driver keeps its own grant book, so one overshoot parks twice and a grant answered to one
    never raises the other's ceiling."""
    if budget is None:
        return
    for name in _budget_policies(layers):
        raise ValueError(
            f"a handler `budget=` and the gate policy {name!r} both drive run "
            f"{budget.run_id!r}'s budget, with two grant books. Keep one: pass "
            "`budget=None` to the handler, or drop the budget policy from the gate."
        )


def refuse_an_unmetered_budget_gate(meters: bool, layers: Sequence[OpLayer[Any]]) -> None:
    """Refuse a gate holding a budget policy on a handler that does not meter, before any op runs.

    A wrapped policy is refused at its first gated op instead, by `BudgetPolicy` itself."""
    if meters:
        return
    for name in _budget_policies(layers):
        _refuse_unmetered(name)


def _budget_policies(layers: Sequence[OpLayer[Any]]) -> Iterator[str]:
    """The name of each budget policy a gate in `layers` was handed. A budget policy wrapped
    inside another callable is not among them."""
    for layer in layers:
        for policy in policies_of(layer):
            match policy:
                case BudgetPolicy(name=name):
                    yield name
                case _:
                    pass


def _refuse_unmetered(name: str) -> Never:
    raise CompositionRefused(
        f"budget policy {name!r} gates a run that does not meter. Run it on a durable handler "
        "under `Contract.V1` with a `MeteredDomain`."
    )


def _as_grant(answer: Any) -> Grant:
    """A delivered answer as a `Grant` — the model itself, a posted mapping, or a bare number of
    dollars (the one-field convenience an operator surface will actually send)."""
    match answer:
        case Grant():
            return answer
        case Mapping():
            return Grant.model_validate(dict(answer))
        case int() | float():
            return Grant(add_dollars=float(answer))
        case _:
            raise TypeError(
                f"a budget grant answer must be a Grant, a mapping, or a dollar amount; "
                f"got {type(answer).__name__}"
            )


__all__ = [
    "BUDGET_DEPTH_PARAM",
    "BUDGET_GENERATIONS_PARAM",
    "Budget",
    "Cleared",
    "Exceeded",
    "Grant",
    "MeasuredBudget",
    "OnExhaust",
    "Parked",
    "TripOutcome",
    "as_policy",
    "enforce_measured",
]
