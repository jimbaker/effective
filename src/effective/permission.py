"""Permission as a composable cascade: the op-seam guard.

A *permission cascade* is an ``@op_layer`` that runs each op past an ordered list
of **tiers**; the first decisive verdict wins, and the cascade **fails closed**.
The three-valued ``Verdict`` is a *guard ruling*. It rhymes with a channel's
``Resolution`` (Done / Repair) and is a **separate type**, because an
``Escalate`` can suspend and a ``Repair`` cannot:

| verdict      | ruling | effect                                       |
|--------------|--------|----------------------------------------------|
| ``Allow``    | settle | forward the op                               |
| ``Deny``     | block  | raise ``Refused``; the op never runs         |
| ``Escalate`` | defer  | fall through to the next tier                |

A **tier** is ``op -> Generator[WorkflowOp, Any, Verdict]``: it MAY yield ops
(the ``human`` tier injects an ``AwaitEvent`` and parks for a reviewer, proven to
survive crash/replay on the durable path) or just return
a verdict (the deterministic ``rules`` tier). The descent re-derives on replay
from recorded state, with no captured continuation.

``human`` takes the approval *schema* as a parameter and duck-types ``.decision`` /
``.rationale``, so a domain's event model stays in the domain.
"""

from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, assert_never

from pydantic import BaseModel

from effective.govern import Ask, GateState, Policy, Proceed, Refused, answers_for, park, refuse
from effective.handlers.base import placed_key
from effective.keys import AuthorityTag, Index, Key, Name, Scope, compose_key
from effective.keys.grammar import KeySyntaxError, parse
from effective.layers import OpLayer, layer_run_state, op_layer
from effective.ops import AwaitEvent, WorkflowOp, current_generation

APPROVE = AuthorityTag("approve", scope=Scope.SETTLEMENT)
"""The human-tier approval namespace — its names ARE the authorization.

A **settlement**: one approval settles one gated op-occurrence, never the next call of the same
tool. The coordinate rides BELOW the grammar — `Key.occurrence`'s `#N` in `tier`, counted in a
per-run cell, rather than as a template field, which is why occurrence 1 is byte-identical to
the bare name."""

HUMAN_SCOPE = Name("human")
"""The `human` tier's slice of the run scope, where its per-op occurrence counts live.

A constant rather than a parameter because the tier has no gate/run identity of its own:
`layer_run_state` is already per-`run()`, so this only has to separate the tier from other
layers sharing that scope."""


def names_an_approval(wake_event: str) -> bool:
    """Does `wake_event` name a sign-off gate's park? — the READ side of `human`'s event name.

    Derived from the writer rather than spelled beside it, which is the property the pair exists
    for: a reader that re-implements a name is a reader that drifts out of it — silently, because
    a filter that matches nothing looks exactly like a queue with nothing in it. Its sibling one
    module over is `budget.names_a_budget_grant`, and it exists against the same failure.

    Takes a `str`, not a `Key`, because its callers sweep a STORE: a wake registration is text out
    of a column, and this is the door it re-enters the typed world through. TOTAL for the same
    reason — a row that is not in the language is not an approval, which is an answer rather than
    an error.

    **A tag comparison, not a prefix.** `approve:` and `approve;` are both well-formed openings and
    only one is a park; a prefix has to guess which delimiter follows the tag, and that guess is
    exactly what moved. Asking `parse` costs one call per row and cannot be wrong about it."""
    try:
        head, *_ = parse(wake_event).terms
    except KeySyntaxError, ValueError:
        return False
    return head.tag == APPROVE


@dataclass(frozen=True)
class Allow:
    """Settle — forward the op."""


@dataclass(frozen=True)
class Deny:
    """Block — refuse the op; it never runs."""

    reason: str = "denied"


@dataclass(frozen=True)
class Escalate:
    """Defer — fall through to the next tier."""

    reason: str = ""


type Verdict = Allow | Deny | Escalate
type Decision = Allow | Deny
"""A *settled* verdict — what `decide` always returns. `Escalate` is a request to keep
looking, never an answer, so the fold's codomain excludes it by construction."""

type Tier = Callable[[WorkflowOp], Generator[WorkflowOp, Any, Verdict]]

# The fail-closed verdict: if every tier escalates, the cascade denies.
FAIL_CLOSED = Deny("no tier allowed the op")

# A `default` that is itself an `Escalate` cannot settle anything — a misconfiguration,
# resolved fail-closed (the same reason string the driver has always raised).
MISCONFIGURED_DEFAULT = Deny("cascade default did not decide")


# `Refused` (imported above, re-exported here) is the GATE's exception, not permission's own:
# permission is one governor among several (budget, quota), and a refusal from any of them is the
# same event to a workflow. It lives in `effective.govern`, and both
# `from effective.permission import Refused` and `from effective import Refused` resolve to it.
# The refusal-name family: one exception per refusing verdict, one spelling per meaning.


def rules(policy: Callable[[WorkflowOp], Verdict]) -> Tier:
    """A deterministic tier from a pure ``op -> Verdict`` policy.

    MUST NOT read the clock / random / live state: replay re-runs it and the
    verdict must be identical (the op-layer determinism rule).
    """

    def tier(op: WorkflowOp) -> Generator[WorkflowOp, Any, Verdict]:
        yield from ()  # an empty generator: yields nothing, returns the verdict
        return policy(op)

    return tier


class PermitPolicy(BaseModel):
    """The **data twin** of a permission rule (the auto-mode allow-table): op-key prefixes
    that auto-settle. It is *data*, not a closure, so it checkpoints, diffs, and is
    `improve`-optimizable: a tunable seam takes data (cf. `rules`, which takes a bare
    `Callable` and is therefore optimizer-invisible)."""

    allow: tuple[str, ...] = ()  # op-key prefixes that auto-settle

    def permits(self, key: Key) -> bool:
        # `.stored()`: prefix matching is over the DURABLE form, because that is what the
        # allow-set is written against and what replay re-derives. A projection form that ever
        # diverged would silently change which ops auto-settle.
        return any(key.stored().startswith(prefix) for prefix in self.allow)


def allow_table(policy: PermitPolicy) -> Tier:
    """A data-driven **Allow | Escalate** tier: auto-settle ops whose key matches ``policy``,
    escalate the rest. The data twin of ``rules`` — the policy is optimizable data, so a
    tuned or learned allow-set drops in here without a code change.

    **Never denies.** Denial stays with an explicit deny-rules tier and the ``human``, so a
    widened (or mis-tuned) allow-set can only auto-settle *more*, never *block* — the
    fail-safe direction. Deterministic (a pure function of the op key), so replay re-derives
    the same verdict (the op-layer rule)."""

    def tier(op: WorkflowOp) -> Generator[WorkflowOp, Any, Verdict]:
        yield from ()  # deterministic: no ops, just the data-driven verdict
        return Allow() if policy.permits(placed_key(op)) else Escalate()

    return tier


def approval_name(op: WorkflowOp) -> Key:
    """The default `approve` park name — `approve[:generation=N];{placed_key(op)}`.

    A named function, for two reasons that are the same reason. A template's hole *expressions*
    become the key registry's field names, so the generation is bound to a local called
    `generation`: `explain()` on the composed key names its fields, and `current_generation()` is
    not a field name. And a default worth reading is a default worth citing: the
    `tests/_authority.py` census calls the minter the substrate calls instead of spelling a
    second shape beside it.

    **`generation` is the CROSS-TASK coordinate**, and this namespace had the least protection of
    any: `approve;{op_key}` carries no run coordinate at all, so a caller keyed by message
    id relies on a run-unique id inside the op key (`processed:{message_id}`): see below. A
    `respawn` chain keeps `run_id` stable and each generation is a fresh task, so every within-run
    coordinate restarts, and without this coordinate generation 1 composes exactly what
    generation 0 parked on (both asking `approve;step;tool:charge_card`); end-to-end on real
    Absurd through `govern`, that is a $5,000,000 charge settled by a $5 approval.
    `budget.depth_grant_name` and `govern:` carry the same coordinate.

    Omitted at generation 0, so an unchained run composes the bytes it always did.

    **A CUSTOM `event_name` does not get this coordinate**, and the substrate cannot supply it:
    the generation is licensed as an identity source only because ordinary workflow code sets it
    from the task's own params, so a handler-applied version would be absent on the replay walk
    (`ops.CHAIN_GENERATION`'s docstring makes the argument). An author who both replaces this name
    and respawns must carry the generation themselves, the way `agent/voi.py` passes an explicit
    `generation=0` and says why."""
    # `generation` is bound to a NAME because a template's hole expressions become the registry's
    # field names, and `current_generation()` is not one. The op hole keeps its `placed_key(op)`
    # spelling, which two tests pin as the field name.
    generation = current_generation()
    return compose_key(
        t"{APPROVE}:generation={Index(generation):default=0};{placed_key(op):domain=identity}"
    )


def human(
    schema: type,
    event_name: Callable[[WorkflowOp], Key] = approval_name,
) -> Tier:
    """A human-approval tier: inject an ``AwaitEvent`` and park for a reviewer.

    ``schema`` is the approval event type; its value must expose
    ``.decision`` (``"approve"`` / ``"reject"``) and an optional ``.rationale``.
    The event name is deterministic (``op_key``-derived by default) so it re-binds
    by name on replay.

    **Run-scope the event name in any multi-run deployment.** Absurd events are global by
    *name*, and the default here is only op-shaped, so a persisted approval from an earlier run
    of a same-shaped workflow resumes the next run's park instantly.

    **A second `approve:` variant does NOT work.** The default variant is ``approval_name``'s,
    whose terminal hole absorbs the rest of the string, so no second `approve:` variant (say,
    one with a run id before the op key) can be separated from it; and reaching for the
    constant from another module leaves the namespace unresolvable to the scanner. One tag has
    one shape, because two variants under one tag means an authorization under one name could
    answer the other. Cite the shape as ``approval_name`` rather than spelling the template.

    So take one of these instead:

    - **put the run-unique component in the OP key**, which is what a caller keyed by message
      id does (``processed:{message_id}``): the default name then carries it for free and no
      second variant exists; or
    - **declare your own namespace**, ``AuthorityTag("approve-<yours>", scope=Scope.SETTLEMENT)``
      at the composition site, reserved in `RESERVED_AUTHORITY_TAGS`, if the run id genuinely has
      no home inside the op key.

    **Run-uniqueness is NECESSARY and insufficient.** A run-unique id separates *runs*, and two
    occurrences of one gated op inside a single run still share it: ``ledger:processed:{id}``
    meets the condition and would alias two appends of the same row. That second axis is closed
    BELOW, by the per-run occurrence counter in ``tier``; the condition here covers the first
    axis only.

    **Compose it with `compose_key`.** An f-string here is wrong twice over: a `Key` has no
    ``__str__``, so it interpolates the *repr* into the park name, and the result is a
    hand-rolled identity outside the composer either way. `compose_key` refuses a
    delimiter-bearing `run_id` and puts the name in the registry the lint can see.

    **Inside a gather branch** the event an emitter must send is additionally
    gather-qualified (``gather:{g},{i};`` from the ctx prefix, invisible here)
    — compose it with ``effective.api.qualified_event_name``; run-scoping applies
    on top, the same two-rule contract as ``descend`` grants.
    """

    def tier(op: WorkflowOp) -> Generator[WorkflowOp, Any, Verdict]:
        # The OCCURRENCE coordinate. `op_key` is not occurrence-injective: a `Step` carries the
        # author's bare name, so an agent that calls one tool twice yields two ops with one key.
        # On the checkpoint axis the engines restore the missing coordinate by suffixing `name#k`;
        # on the AUTHORITY axis this counter does, so one approval cannot settle a later
        # occurrence (a `$5` approval authorizing a `$5,000,000` charge of the same tool).
        # `govern` does the same with `GateState.occurrence`.
        #
        # Counted in the handler's per-attempt run scope, never in this factory's closure, for the
        # same reason `govern` gives: a closure outlives `human()`'s result, so a tier built once
        # at registration would carry task A's count into task B. `layer_run_state` is per `run()`,
        # so the count is re-derived from the replayed op stream on every attempt and the name a
        # replay computes is the name the original parked on.
        scope = layer_run_state(compose_key(t"{APPROVE}:{HUMAN_SCOPE}"))
        occurrences: dict[str, int] = scope.setdefault("occurrences", {})
        base = event_name(op)
        occurrences[base.stored()] = occurrences.get(base.stored(), 0) + 1
        # `Key.occurrence` returns `self` at 1, so a name that occurs once is BYTE-IDENTICAL to
        # the bare name and no recorded run is orphaned; only a repeat gains `#k`.
        name = base.occurrence(occurrences[base.stored()])
        # No `.stored()`: `AwaitEvent.name` is a `Key`, so the composed identity travels as
        # itself instead of unwrapping to satisfy a `str` field.
        approval = yield AwaitEvent(name=name, schema=schema)
        if getattr(approval, "decision", None) == "approve":
            return Allow()
        return Deny(getattr(approval, "rationale", "") or "denied by reviewer")

    return tier


# --- the cascade decision transition ------------------------------------------------------
#
# The cascade's *decision* lifted out of the driver that realizes it, as budget's
# `enforce_measured` is: ONE definition, pure and total, that every driver is
# conformance-tested against. What stays in the driver is the part that
# cannot be pure — running a tier (it may yield ops) and realizing the ruling (forward vs raise).
# Formalized in `formal/lean/Effective/Decide.lean`; conformance-pinned by
# `tests/test_decide_conformance.py`.


def decisive(verdict: Verdict) -> bool:
    """True when `verdict` settles the cascade — the driver's stop condition.

    Split from `decide` because the driver must stop running tiers *eagerly*: a later tier
    may park a human, and a settled cascade must never pay for an approval it does not need.
    `decide` is then a fold over the prefix the driver actually collected — sound precisely
    because a decisive verdict absorbs everything after it (`decide_ignores_suffix`, Lean)."""
    return not isinstance(verdict, Escalate)


def decide(verdicts: Iterable[Verdict], default: Verdict = FAIL_CLOSED) -> Decision:
    """The cascade as ONE explicit transition: an ordered fold — first decisive verdict wins,
    an `Escalate` defers, and an all-escalate run falls to `default` (fail-closed: `Deny`).

    **Total** (always a `Decision`, never an `Escalate` — a misconfigured `Escalate` default
    resolves to `MISCONFIGURED_DEFAULT`), **deterministic** (a pure function of the verdict
    sequence and the default), and **fail-closed** (nothing here can turn silence into an
    `Allow`). Pure: the `human` park is *not* in here — the driver realizes an `Escalate`
    reaching a human tier as an `AwaitEvent`, exactly as `enforce_measured`'s `Parked` is."""
    for verdict in verdicts:
        # lint: totality(guard) — the wildcard REJECTS, it is not an unnamed arm. A tier is
        # called dynamically, so `Iterable[Verdict]` is what the signature promises rather
        # than what arrives: the case this catches is a tier that returned a GENERATOR
        # instead of a verdict, and its message tells the author to use `yield from`.
        # `assert_never` would be statically equivalent and would delete a check that
        # catches a real mistake the type system cannot see.
        match verdict:
            case Allow() | Deny():
                return verdict
            case Escalate():
                continue
            case _:  # not silently a deferral: a non-Verdict would fail closed INVISIBLY
                raise TypeError(
                    f"a cascade tier must return Allow / Deny / Escalate; got "
                    f"{type(verdict).__name__}. A `rules(...)`/`allow_table(...)` tier is a "
                    f"GENERATOR — call it with `yield from`, or pass its bare "
                    f"`op -> Verdict` function to `as_policy`."
                )
    match default:
        case Allow() | Deny():
            return default
        case Escalate():
            return MISCONFIGURED_DEFAULT
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


# --- permission as a `govern` policy ------------------------------------------------------

type RuleTier = Callable[[WorkflowOp], Verdict]
"""A **pure** tier — `op -> Verdict`, no yields. Under `govern` a tier never parks: the gate owns
the single park, so the human tier stops being a special kind of tier. `rules`/`allow_table` bodies
are already this shape; only `human` was not, and it is exactly what `govern` absorbs."""


def _verdict_of(tier: Tier | RuleTier, op: WorkflowOp) -> Verdict:
    """Run one tier for its verdict, accepting either shape a caller has on hand: a bare
    `op -> Verdict`, or an existing `rules(...)` / `allow_table(...)` generator tier (which yields
    nothing and returns the verdict, so it drives to completion immediately).

    A tier that *does* yield — `human` — is rejected by name, because under `govern` the gate owns
    the single park. That rejection is the migration message: escalation IS the park now."""
    match tier(op):
        case Allow() | Deny() | Escalate() as verdict:
            return verdict  # a bare `op -> Verdict` tier
        case Generator() as tier_generator:
            try:
                yielded = next(tier_generator)
            except StopIteration as done:
                return done.value
        case other:
            raise TypeError(
                f"a govern policy tier must return Allow / Deny / Escalate (or be a "
                f"deterministic `rules`/`allow_table` tier); got {type(other).__name__}."
            )
    raise TypeError(
        f"a govern policy tier must not yield ops — the gate owns the single park, so drop the "
        f"`human(...)` tier and let escalation BE the park (it yielded {type(yielded).__name__}). "
        f"Deterministic tiers (`rules`, `allow_table`) are fine as-is."
    )


def as_policy(
    tiers: Sequence[Tier | RuleTier],
    *,
    default: Verdict = FAIL_CLOSED,
    on_escalate: Literal["park", "deny"] = "park",
    name: str = "permission",
    prompt: Callable[[WorkflowOp], str] = lambda op: f"approve {placed_key(op)}?",
) -> Policy:
    """Permission as a `govern` policy — the classifier half of the unified gate.

    **Escalation IS the park.** The tiers fold by `decide`; if every tier defers, the policy asks
    the gate to park (`on_escalate="park"`, the default) instead of falling to `default`. That is
    safe here in a way a fail-open default never would be, because parking cannot let the op
    through: nothing proceeds until an answer arrives and says `approve`. On an unattended surface
    that cannot park, pass `on_escalate="deny"` and the fold's fail-closed `default` rules.

    The shared default is a named parameter: budget's is park, permission's is
    park-when-attended, and neither is baked into `govern`, which only combines.

    On resume, the answer under `name` is read as an approval: duck-typed `.decision` /
    `.rationale`, the same schema-free contract `human` uses. The LATEST answer settles:
    an approval is a settlement, not an accrual (contrast budget, which folds the whole
    history)."""

    def policy(op: WorkflowOp, state: GateState) -> Any:
        if history := answers_for(state, name):
            answer = history[-1]
            if _approved(answer):
                return Proceed()
            return refuse(_rationale(answer) or "denied by reviewer")
        verdicts = [_verdict_of(tier, op) for tier in tiers]
        if on_escalate == "park" and not any(map(decisive, verdicts)):
            return park(Ask(name, prompt(op), {"op_key": placed_key(op).stored()}))
        match decide(verdicts, default):
            case Allow():
                return Proceed()
            case Deny(reason=reason):
                return refuse(reason)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    return policy


def _approved(answer: Any) -> bool:
    """Duck-typed approval — `.decision == "approve"`, or the same key in a posted mapping."""
    match answer:
        case {"decision": decision}:
            return decision == "approve"
        case _:
            return getattr(answer, "decision", None) == "approve"


def _rationale(answer: Any) -> str:
    match answer:
        case {"rationale": str() as rationale}:
            return rationale
        case _:
            return getattr(answer, "rationale", "") or ""


def cascade(tiers: list[Tier], default: Verdict = FAIL_CLOSED) -> OpLayer[Any]:
    """Compose ``tiers`` into an ``@op_layer``.

    Runs each tier in order; the first ``Allow`` / ``Deny`` decides, an
    ``Escalate`` defers to the next tier. If every tier escalates, ``default``
    decides (fail-closed: ``Deny``). ``Allow`` forwards the op; ``Deny`` raises
    ``Refused``.

    The **driver**: it runs tiers (they may yield ops — the impure half), stops at
    the first decisive verdict, and realizes `decide`'s ruling. The *decision* is
    `decide` — one pure transition, shared and conformance-tested.
    """

    @op_layer
    def gate(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
        verdicts: list[Verdict] = []
        for tier in tiers:
            verdicts.append((yield from tier(op)))
            if decisive(verdicts[-1]):
                break  # a later tier may park a human — never pay for an unneeded approval
        match decide(verdicts, default):
            case Allow():
                return (yield op)
            case Deny(reason=reason):
                raise Refused(op, reason)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    return gate
