"""`govern`: the control gate, a council of policies with a single merged park.

A governed boundary is an op-seam layer that asks every **policy** the same question about one
op (*may this proceed?*) and combines their answers into **one** verdict:

    Proceed | Park | Refuse

Permission and the measured budget trip are one pattern with two policy bodies: a
**classifier** (fold tiers into a verdict) and an **accumulator** (meter vs ceiling). `govern` is
that shared control shape, built once, so a new gate (a quota, a rate limit) is *enrolled* in it.

**Policies are peers.** Any `Refuse` refuses; else any `Park` parks; else
`Proceed`. There is no `rules → escalate → human` fall-through here: tiered escalation inside a
single policy is that policy's own business (`permission.as_policy`), and `govern` adds no
escalation axis of its own. The verdict is therefore **order-free** (a permutation of the policies
gives the same constructor), while the fused payload keeps argument order so it stays
deterministic and legible.

**One park, merged.** Several policies wanting to block do NOT park in sequence. Their `Ask`s fuse
into one `Park`, the gate yields exactly one `AwaitEvent`, and one `Resolution` carries the
combined payload (a grant *and* an approval) keyed by the policy that asked. This is the single
point where value-of-information prices the whole gate-bundle's counterfactual at once; two
sequential parks would price two unrelated questions and make a human answer twice for one
decision.

**The resolution payload is policy-body, not control-shape.** `GateState.answers` maps a policy
name to the *history* of answers delivered to it, in pass order; each policy folds that history as
its own semantics require (budget accumulates grants positionally — its trip is defined over a
*list* of grants; permission takes the latest approval). `govern` never interprets an answer.

**A policy must be pure** in the same sense a `rules` tier is: a function of the op, the gate
state, and replay-derived handler state, with no clock, random or live I/O. Replay re-runs it and
the verdict must be identical.

The refusal names: a *verdict* names a state and an *exception* names the raise.
`Refuse` raises `Refused`: a gate blocked the op. A `Refuse` carrying `Exceeded` raises
`BudgetRefused`: spend reached its ceiling and no grant raised it. `cost.BudgetExceeded` is the
domain meter's own cap, raised inside a checkpoint thunk with no verdict. `Refused` lives here, in
the gate's own vocabulary, and `permission` re-exports it.
"""

from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any, Protocol, assert_never, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from effective.cost import Usage
from effective.keys import AuthorityTag, Index, Key, Name, Run, Scope, Segment, compose_key
from effective.layers import OpLayer, current_meter, current_placement, layer_run_state, op_layer
from effective.ops import AwaitEvent, CompositionRefused, WorkflowOp, current_generation, leaves
from effective.ruling import ruling_schema, split_ruling

GOVERN = AuthorityTag("govern", scope=Scope.SETTLEMENT)
"""A gate's merged-park namespace — its names ARE the authorization.

A **settlement**, and the reference implementation of one: `GateState.park_name` carries the
occurrence as an interior template FIELD, so the coordinate is visible in the registered shape
rather than below the grammar. That is the other of the two forms `Scope.SETTLEMENT` accepts."""

GATE_STATE = AuthorityTag("gate-state", scope=Scope.ACCRUAL)
"""A gate's per-run STATE cell (`layer_run_state`), distinct from `GOVERN`'s park namespace.

One tag names one shape: the park name and this 2-field cell address are different shapes, and
under one tag they collide (`gate='a', run_id='b:c'` and `gate='a:b', run_id='c'` would address
the same cell).

The cell lives in a `ContextVar` and is re-derived on every replay, so nothing durable carries
this name and no stored row depends on its spelling."""

# --- the gate's vocabulary --------------------------------------------------------------

WRAPPER = "answers"
"""The key a posted mapping wraps its answers under, so no policy asks under it."""


@dataclass(frozen=True)
class Ask:
    """One policy's share of a fused park — what it needs answered to let the op through.

    `policy` is the key the answer comes back under (`Resolution.answers[policy]`), so it is
    identity, not decoration. `detail` is the machine-readable half a pricing surface reads (the
    dollars-marginal, the op being approved); `prompt` is the human-readable half."""

    policy: str
    prompt: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    binds: Segment | None = None
    """The digest of the effect the ask is about, which a ruling on it names under `effect`.
    `None` for an ask whose answer is not about one effect, as a grant of more budget is not."""

    def __post_init__(self) -> None:
        if self.policy == WRAPPER:
            raise CompositionRefused(
                f"no policy asks under {WRAPPER!r}, the key that wraps the answers"
            )


@dataclass(frozen=True)
class Proceed:
    """Verdict: no policy objects, so the op is forwarded."""


@dataclass(frozen=True)
class Park:
    """Verdict: at least one policy wants to suspend. `asks` is the FUSED question: one entry per
    parking policy, in policy order. The gate parks once, whatever the length."""

    asks: tuple[Ask, ...]


@dataclass(frozen=True)
class Exceeded:
    """Spend reached `ceiling`, the limit plus every grant, with no recourse taken: the fail
    budget, a `stop`, or a zero grant."""

    spent: float
    ceiling: float

    @property
    def reason(self) -> str:
        return f"measured budget exceeded: spent ${self.spent:.4f} >= ${self.ceiling:.4f}"


@dataclass(frozen=True)
class Refusal:
    """Why `policy`'s part of a gate refused an op, and who ruled when a ruling caused it."""

    policy: str
    reason: str
    by: Segment | None = None


@dataclass(frozen=True)
class Refuse:
    """Verdict: at least one policy blocks. Every refusing policy's reason is kept, so a gate that
    refuses for three reasons says all three.
    `exceeded` is set when a budget policy refused."""

    refusals: tuple[Refusal, ...]
    exceeded: Exceeded | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(refusal.reason for refusal in self.refusals)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


type Verdict = Proceed | Park | Refuse


def park(*asks: Ask) -> Park:
    """A policy's park verdict (`park(Ask(...))`): the lowercase verb builds the value, and the
    capitalized noun is the type, as with `rules`/`human` in `permission`."""
    return Park(asks)


def refuse(
    reason: str, *, policy: str, by: Segment | None = None, exceeded: Exceeded | None = None
) -> Refuse:
    """A policy's refusal verdict, naming the policy, who ruled when a ruling caused it, and the
    spend when a budget refused."""
    return Refuse((Refusal(policy, reason, by),), exceeded)


class Refused(Exception):
    """A governed boundary blocked this op — recourse existed and was denied (or was never
    offered). Distinct from `cost.BudgetExceeded`, the domain seam's no-recourse hard cap."""

    def __init__(self, op: WorkflowOp, reason: str, refusals: tuple[Refusal, ...] = ()) -> None:
        super().__init__(reason)
        self.op = op
        self.reason = reason
        self.refusals = refusals
        """Each policy's refusal, when a gate's policies refused the op."""


class ChildRefused(Exception):
    """A joined child stopped at a refusal. Raised in its parent, and itself a refusal, so it
    climbs a spawn tree to the root."""


REFUSALS: tuple[type[Exception], ...] = (Refused, ChildRefused)
"""The runtime refusals. Each re-derives on a retry, so every attempt reaches the same one."""


def all_refusals(raised: BaseException) -> bool:
    """Whether every leaf of `raised` is a refusal, which a spawned child answers its parent
    with."""
    return all(isinstance(leaf, REFUSALS) for leaf in leaves(raised))


def delivered(raised: Exception) -> Exception:
    """`raised`, for a handler to throw into the workflow that yielded the op or structure it came
    out of, when every leaf is a refusal. Anything else is raised again, task-level."""
    if not all_refusals(raised):
        raise raised
    return raised


class BudgetRefused(Refused):
    """Spend reached its ceiling at this op. Every later metered op is refused the same way, so a
    loop that turns a `Refused` into an observation re-raises this one."""

    def __init__(
        self,
        op: WorkflowOp,
        exceeded: Exceeded,
        reason: str | None = None,
        refusals: tuple[Refusal, ...] = (),
    ) -> None:
        super().__init__(op, exceeded.reason if reason is None else reason, refusals)
        self.exceeded = exceeded


class Unbound(Refused):
    """A ruling on an ask that binds an effect named another effect, or none: the op changed under
    the ruling, so no policy reads it. Each refusal gives `ANOTHER_EFFECT` as its reason; the
    message names the ruler and both effects."""


ANOTHER_EFFECT = "the ruling is on an effect other than the op's"
"""Why a gate refuses an op whose ruling names an effect other than the one its ask binds."""


def refuse_an_unbound_ruling(
    op: WorkflowOp, parked: Park, resolution: Resolution, by: Segment | None
) -> None:
    """Raise `Unbound` at `op` when the ruling on `parked` names another effect than an ask binds,
    naming each such policy, the ruler, and both effects."""
    unbound = [
        (ask, named)
        for ask in parked.asks
        if ask.binds is not None
        and (named := effect_named(resolution.answers.get(ask.policy))) != ask.binds
    ]
    if unbound:
        raise Unbound(
            op,
            "; ".join(
                f"{ask.policy}: the ruling by {by or 'nobody'} rules on "
                f"{named or 'no effect'}, and the effect is {ask.binds}"
                for ask, named in unbound
            ),
            tuple(Refusal(ask.policy, ANOTHER_EFFECT, by) for ask, _ in unbound),
        )


def effect_named(answer: object) -> str | None:
    """The effect a ruling's answer names under `effect`."""
    match answer:
        case {"effect": str(digest)}:
            return digest
        case _:
            return None


def routable(refusal: Refused) -> Refused:
    """`refusal`, when a loop may route around it. A `BudgetRefused` is raised again instead,
    since every later metered op is refused the same way."""
    match refusal:
        case BudgetRefused():
            raise refusal
        case Refused():
            return refusal
        case unreachable:
            assert_never(unreachable)


def refused(op: WorkflowOp, refusal: Refuse) -> Refused:
    """The exception a `Refuse` verdict raises at `op`."""
    match refusal.exceeded:
        case None:
            return Refused(op, refusal.reason, refusal.refusals)
        case Exceeded() as exceeded:
            return BudgetRefused(op, exceeded, refusal.reason, refusal.refusals)
        case unreachable:
            assert_never(unreachable)


class Resolution(BaseModel):
    """One answer to a merged park: a per-policy payload keyed by the asking policy's name.

    Deliberately an open mapping over the known payload types (a grant, an approval). The
    control shape is "an answer arrived for this gate"; what an answer *means* belongs to the
    policy that asked, so a new policy adds a key and the shared type gains no variant.

    The durable await and `as_resolution` both read a posted mapping here:

    | the mapping holds                                  | it reads as             |
    |----------------------------------------------------|-------------------------|
    | `answers` alone                                    | the answers, wrapped    |
    | no `answers` key                                   | the answers themselves  |
    | `answers` beside another key                       | refused                 |
    """

    answers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _wrapped(cls, payload: Any) -> Any:
        match payload:
            case Mapping() if WRAPPER not in payload:
                return {WRAPPER: dict(payload)}
            case Mapping() if len(payload) > 1:
                beside = sorted(k for k in payload if k != WRAPPER)
                raise ValueError(
                    "a govern resolution holds `answers` alone or the answers themselves; "
                    f"got `answers` beside {beside}"
                )
            case _:
                return payload


@dataclass(frozen=True)
class GateState:
    """What a policy may read: the gate's identity and every answer delivered so far.

    `answers` is a *history* per policy (pass order), not a latest-wins slot: the measured trip
    is defined over a LIST of grants, so collapsing to the latest would lose the accrual. A policy
    that only wants the latest takes the last element."""

    run_id: str
    gate: str
    op_key: str
    """WHICH op this gate is deciding. Required, so it sits above the defaulted fields: Python's
    "no non-default after a default" is the same rule the key grammar adopts for coordinates.

    A default on an identity coordinate would make a park name with an empty terminal
    coordinate (`govern:spend:run-g:0:0:`) mintable through `park_name`."""

    generation: int = 0
    """WHICH generation of a `respawn` chain is asking — the CROSS-TASK coordinate.

    `run_id` is stable across a chain's generations by construction, and each generation is a
    fresh task, so every other coordinate here restarts: `occurrence` counts in the handler's
    per-`run()` scope and `pass_n` starts a new gate. Generation 1 therefore composed the
    byte-identical name generation 0 had parked on, and a broadcast engine answered it instantly
    from the earlier approval — measured on real Absurd as a $5,000,000 charge settled by a $5
    approval, with no second park.

    `budget.depth_grant_name` carries the same coordinate for the same reason. An engine whose
    events table is keyed `(task_id, name)` hides the collision, because the addressee separates
    what the name does not.

    Supplied by `ops.current_generation()` at the single construction site, never by a caller —
    the same "a call site cannot forget it" discipline `combinators.respawn` applies to `descend`.

    **Defaulted, unlike `depth_grant_name`'s, and the difference is the caller set**.
    That one refuses a default because `effective/voi.py` composes it from OUTSIDE the
    substrate and silently got 0; the danger there is real and unowned. This field has exactly one
    non-test construction site, inside `govern` itself, which reads the ambient — so a default
    cannot be silently taken by anyone who matters. It also sits beside `pass_n` and `occurrence`,
    which are identity coordinates on this same record, default to a correct 0, and are likewise
    always supplied here. Requiring one of the three and not the others would be a rule about
    which coordinate was found last rather than about how they behave.

    The requirement lives where it does work instead: `tests/_authority.py`'s census makes
    `generation` mandatory of every minter, so a namespace that cannot accept one is visible in
    its signature."""

    pass_n: int = 0
    answers: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    occurrence: int = 0
    run_answers: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    meter: Usage | None = None
    """The task's replay-derived spend when the gate decides, or `None` when the run does not
    meter. It is the root handler's, read from a gather branch too, and it counts a branch's spend
    once the gather's barrier folds it."""
    rulers: Mapping[str, tuple[Segment | None, ...]] = field(default_factory=dict)
    """Who ruled each of `answers`, position for position; `None` where an answer came with no
    ruling, which only an in-process recorder delivers."""
    run_rulers: Mapping[str, tuple[Segment | None, ...]] = field(default_factory=dict)
    """Who ruled each of `run_answers`, position for position."""
    placement: Key | None = None
    """Where the gated op runs: its frames, its key and its occurrence. Two gather branches gating
    one op share a `park_name`, which the handler frames, and differ here. `None` where no walk
    places the op."""

    @property
    def park_name(self) -> Key:
        """The gate's park event name —
        `govern:{gate},{run_id}[,generation=N][,pass-n=N][,occurrence=N];{op_key}` — composed
        through the single audited key producer so an interpolated value carrying a `:` cannot
        forge a delimiter (`Keys.lean`'s injectivity, by construction). Deterministic in all six
        components, so it re-derives on replay and re-binds by name; run-scoped because Absurd
        events are global per queue.

        **Why `op_key` AND `occurrence`.** The op component alone is not enough, because `op_key`
        is not *occurrence*-injective: a `Step` carries the author's bare name, so an agent that
        calls one tool twice yields two ops with the same key. The walk already knows this and
        places duplicate *checkpoints* `name#2` (`DurableHandler._place`).
        Without the same for **authority** names, one approval settles every later occurrence:
        a `$5` approval authorizes a `$5,000,000` charge of the same tool, on both engines.
        `occurrence` is the same discipline for the authority namespace: a per-`(gate, op_key)`
        counter over the gated ops of this run.

        **And why `generation` on top of both.** `occurrence` counts within one `run()` and
        `pass_n` within one gate, so neither crosses a task — and a `respawn` chain's generations
        are different tasks sharing a `run_id`. The field's own docstring carries the measurement.

        Each defaulted coordinate is OMITTED at its default, so a gate outside a chain, on its
        first pass, at occurrence 0 composes the bytes of a name without the three coordinates.

        `op_key` stays LAST because it contains delimiters, which a key permits only in its
        terminal hole; the three interior coordinates are `int`, delimiter-free by type."""
        gate, run_id = Name(self.gate), Run(self.run_id)
        # Bound to NAMES, because a template's hole expressions become the registry's field
        # names: `explain()` reported `Key.parse(self.op_key)` as a field until this line existed.
        generation, pass_n = Index(self.generation), Index(self.pass_n)
        occurrence, op_key = Index(self.occurrence), Key.parse(self.op_key)
        return compose_key(
            t"{GOVERN}:{gate},{run_id},generation={generation:default=0},"
            t"pass-n={pass_n:default=0},"
            t"occurrence={occurrence:default=0};{op_key:domain=identity}"
        )

    def absorb(self, resolution: Resolution, *, by: Segment | None = None) -> GateState:
        """Fold one resolution, ruled `by`, in: each policy's answer is APPENDED to its history
        with `by` beside it, and the pass advances (so the next park gets a fresh, deterministic
        name). An answer held with no ruler beside it reads as unruled, so the two stay
        position for position."""
        merged, merged_run = dict(self.answers), dict(self.run_answers)
        rulers, run_rulers = dict(self.rulers), dict(self.run_rulers)
        for policy_name, answer in resolution.answers.items():
            merged[policy_name] = (*merged.get(policy_name, ()), answer)
            merged_run[policy_name] = (*merged_run.get(policy_name, ()), answer)
            rulers[policy_name] = _aligned(rulers.get(policy_name, ()), merged[policy_name], by)
            run_rulers[policy_name] = _aligned(
                run_rulers.get(policy_name, ()), merged_run[policy_name], by
            )
        return replace(
            self,
            pass_n=self.pass_n + 1,
            answers=merged,
            run_answers=merged_run,
            rulers=rulers,
            run_rulers=run_rulers,
        )


def _aligned(
    rulers: tuple[Segment | None, ...], answers: tuple[Any, ...], by: Segment | None
) -> tuple[Segment | None, ...]:
    """`rulers` with `by` for the newest of `answers`, each earlier answer without a ruler
    read as unruled."""
    return (*(None,) * (len(answers) - 1 - len(rulers)), *rulers, by)


@runtime_checkable
class Policy(Protocol):
    """A governor: `(op, state) -> Verdict`.

    A real `Protocol` rather than a bare alias (as `Service` is): it has a `Verdict` return to
    constrain, so the type pays rent in annotations and in the diagnostics an agent gets pointed
    at."""

    def __call__(self, op: WorkflowOp, state: GateState) -> Verdict: ...


# --- the transition ---------------------------------------------------------------------
#
# The pure half, lifted out of the driver exactly as `permission.decide` and
# `budget.enforce_measured` are — one definition, formalized in
# `formal/lean/Effective/Govern.lean` and conformance-pinned by
# `tests/test_govern_conformance.py`.


def combine(verdicts: Iterable[Verdict]) -> Verdict:
    """The council's conjunctive fold — `govern`'s decision, as ONE explicit transition.

    Any `Refuse` refuses (carrying every refusing reason); else any `Park` parks (fusing every
    ask into one park); else `Proceed`. Total and deterministic. The VERDICT is order-free: a
    permutation of the input gives the same constructor. The fused payload preserves argument
    order, so a merged prompt reads in the order the gate was assembled.

    Refuse dominating Park is the fail-closed direction: a gate never asks a human to grant past
    a policy that has already said no.

    Classification is by CONSTRUCTOR, not by payload: a `Refuse(())` that forgot to say why still
    refuses. Deciding on "did any reason string arrive?" would let an under-populated verdict fall
    through to a park — a guard must not be silenceable by an empty tuple."""
    collected = list(verdicts)
    if refusals := [v for v in collected if isinstance(v, Refuse)]:
        exceeded = next((v.exceeded for v in refusals if v.exceeded is not None), None)
        return Refuse(tuple(chain.from_iterable(v.refusals for v in refusals)), exceeded)
    if parks := [v for v in collected if isinstance(v, Park)]:
        return Park(tuple(chain.from_iterable(v.asks for v in parks)))
    return Proceed()


# --- the driver -------------------------------------------------------------------------

DEFAULT_MAX_PASSES = 8
"""How many resolutions one gate will absorb before giving up on settling. A policy that parks
forever (a grantor that keeps granting too little) would otherwise park-resume-park without end;
this bounds the protocol so the Quint model has a finite reachability question and an operator
gets a legible refusal instead of a run that never finishes."""


def govern(
    *policies: Policy,
    gate: str,
    run_id: str,
    schema: type = Resolution,
    announce: Callable[[Park, GateState], WorkflowOp] | None = None,
    ruled: Callable[[Park, GateState, Segment | None, Any], WorkflowOp] | None = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> OpLayer[Any]:
    """Compose `policies` into one op-seam gate.

    Per op: ask every policy, combine, then realize the verdict: forward, raise `Refused`, or park
    ONCE on `state.park_name` and re-ask with the answer folded in. `gate` names the boundary (it
    is part of the park name, so it must be stable across replay); `run_id` scopes the event.

    `announce` is the seam an operator surface hangs off: a pure `(park, state) -> WorkflowOp`
    that the gate yields *before* suspending, so the fused ask reaches the recorded stream (a
    ledger row, an artifact) and `just approve` can render an informed prompt rather than a bare
    "approve?". Without it the fused asks would be a field nothing can read.

    `ruled` is its counterpart once the answer arrives: a pure `(park, state, by, answer) ->
    WorkflowOp` the gate yields before folding the answer in, with the state the park was asked
    under, so a record of who ruled pairs with the ask it answers.

    **At least one policy is required.** `serve()` with no services is a defensible bypass; a
    *gate* with no policies is a gate that permits everything, which is a misconfiguration, not a
    default. Fail at assembly, loudly."""
    if not policies:
        raise ValueError(
            f"govern(gate={gate!r}) needs at least one Policy — a gate with no policies "
            "permits every op. Pass a policy (e.g. permission.as_policy(...) or "
            "budget.as_policy(...)), or do not install the gate."
        )

    @op_layer
    def gate_layer(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
        # Imported in-function, not at module scope: `handlers.base` lives under the `handlers`
        # PACKAGE, whose `__init__` pulls in `recording` -> `permission` -> back to this module,
        # so a module-level import cycles. (`compose_key` above needs no such dance — it lives in
        # the leaf `effective.keys`.)
        from effective.handlers.base import placed_key

        # The park name is OP-scoped, so one settlement cannot authorize a later op — which needs
        # the op's real identity, including for the arms that have none of their own until the
        # walk places them. `placed_key` supplies both. A `Gather` still has neither (it is not a
        # gated decision — its BRANCHES are), so it falls back to the type name.
        try:
            key = placed_key(op).stored()
        except TypeError, ValueError:
            key = type(op).__name__
        # Cross-op state lives in the HANDLER's per-attempt run scope, never in this factory's
        # closure: a closure survives as long as `govern()`'s result does, and a gate constructed
        # once at registration would let task B proceed on task A's grant with zero events in B's
        # own durable record. `layer_run_state` is per `run()` (one attempt), so the answers
        # below are re-derived from the durable event store on every replay rather than carried
        # in process memory.
        scope = layer_run_state(compose_key(t"{GATE_STATE}:{Segment(gate)},{Segment(run_id)}"))
        occurrences: dict[str, int] = scope.setdefault("occurrences", {})
        run_answers: dict[str, tuple[Any, ...]] = scope.setdefault("answers", {})
        run_rulers: dict[str, tuple[Segment | None, ...]] = scope.setdefault("rulers", {})
        occurrences[key] = occurrences.get(key, -1) + 1
        state = GateState(
            run_id=run_id,
            gate=gate,
            op_key=key,
            # The one coordinate NOT derived from this run's own bookkeeping: it says which
            # generation of a respawn chain is asking, which no per-run counter can see.
            generation=current_generation(),
            occurrence=occurrences[key],
            run_answers=dict(run_answers),
            run_rulers=dict(run_rulers),
            meter=current_meter(),
            placement=current_placement(),
        )
        for _ in range(max_passes + 1):
            match combine([policy(op, state) for policy in policies]):
                case Proceed():
                    return (yield op)
                case Refuse() as refusal:
                    raise refused(op, refusal)
                case Park() as parked:
                    if announce is not None:
                        yield announce(parked, state)
                    asked = AwaitEvent(name=state.park_name, schema=ruling_schema(schema))
                    by, answer = split_ruling((yield asked))
                    resolution = as_resolution(answer)
                    refuse_an_unbound_ruling(op, parked, resolution, by)
                    if ruled is not None:
                        yield ruled(parked, state, by, answer)
                    state = state.absorb(resolution, by=by)
                    run_answers.update(state.run_answers)  # accrual outlives this op
                    run_rulers.update(state.run_rulers)
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        raise Refused(op, f"gate {gate!r} did not settle after {max_passes} resolutions")

    # A function declares no attributes, so the stamp is invisible to `ty`; `policies_of` reads it.
    gate_layer.__effective_policies__ = policies  # ty: ignore[unresolved-attribute]
    return gate_layer


def policies_of(layer: OpLayer[Any]) -> tuple[Policy, ...]:
    """The policies a `govern` gate was handed; any other layer holds none."""
    return getattr(layer, "__effective_policies__", ())


def as_resolution(answer: Any) -> Resolution:
    """A delivered park answer as a `Resolution`: the instance itself, or a mapping read by
    `Resolution.model_validate`, as the durable await reads it."""
    match answer:
        case Resolution():
            return answer
        case Mapping():
            return Resolution.model_validate(answer)
        case _:
            raise TypeError(
                f"a govern park resolution must be a Resolution or a mapping of "
                f"policy-name -> answer; got {type(answer).__name__}. If a policy needs a "
                f"custom payload type, put it UNDER its own key: {{'<policy>': payload}}."
            )


def fuse_prompt(parked: Park) -> str:
    """The merged park rendered for a human — one numbered line per asking policy.

    The whole point of merged-park at the reading end: an operator sees the *bundle* ("this run
    wants $2 more AND a manager's approval") and answers it once."""
    return "\n".join(f"{i}. [{ask.policy}] {ask.prompt}" for i, ask in enumerate(parked.asks, 1))


def answers_for(state: GateState, policy_name: str) -> Sequence[Any]:
    """This policy's answers **for the op being gated**, oldest first — a SETTLEMENT view.

    Empty at each new op, which is what a per-decision policy wants: `permission` must not treat
    an approval given for one op as settling another. Contrast
    `run_answers_for`."""
    return state.answers.get(policy_name, ())


def rulers_for(state: GateState, policy_name: str) -> Sequence[Segment | None]:
    """Who ruled each of `answers_for(state, policy_name)`, position for position."""
    return state.rulers.get(policy_name, ())


def ruler_of(state: GateState, policy_name: str) -> Segment | None:
    """Who ruled this policy's latest answer **for the op being gated**. `None` before this op's
    first answer, and for an answer delivered without a ruling. An answer from an earlier op
    has its ruler in `run_rulers_for`."""
    match rulers_for(state, policy_name):
        case [*_, latest]:
            return latest
        case _:
            return None


def run_answers_for(state: GateState, policy_name: str) -> Sequence[Any]:
    """This policy's answers **across the whole run**, oldest first — an ACCUMULATOR view.

    The scope an accumulating policy wants: a `budget` grant raises the run's ceiling, so a grant
    delivered while gating op1 is still in force at op2 and must not be re-requested.

    **The two scopes exist because the two policy KINDS differ.** With one per-op answer map and
    a run-scoped *park name*, every gated op would mint the same name and op2 would re-absorb
    op1's delivered event: accrual for `budget`, and an authorization bypass for `permission`,
    since one approval would satisfy every later gated op in the run. Separate scopes let the
    park name be op-scoped (correct for settlement) while accrual rides state (correct for
    accumulation).
    """
    return state.run_answers.get(policy_name, ())


def run_rulers_for(state: GateState, policy_name: str) -> Sequence[Segment | None]:
    """Who ruled each of `run_answers_for(state, policy_name)`, position for position."""
    return state.run_rulers.get(policy_name, ())


__all__ = [
    "DEFAULT_MAX_PASSES",
    "Ask",
    "GateState",
    "Park",
    "Policy",
    "Proceed",
    "Refuse",
    "Refused",
    "Resolution",
    "answers_for",
    "as_resolution",
    "combine",
    "fuse_prompt",
    "govern",
    "park",
    "policies_of",
    "refuse",
    "ruler_of",
    "rulers_for",
    "run_rulers_for",
]
