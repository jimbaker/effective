"""`govern` — the control gate: the council fold, the single merged park, the two adapters.

Infra-free: the gate is driven directly through `drive_through` (the same instrument
`test_permission.py` uses), with a hand-driven generator where a park's suspend/resume has to be
observed op by op. The durable park — worker-death, resume, exactly-once — is the Quint model's
half (`formal/quint/govern_park.qnt`) plus `test_govern_durable.py` on the pg-test container.

The load-bearing case here is **merged-park**: two heterogeneous policies (a dollars accumulator
and an approval classifier) blocking the same op must produce ONE `AwaitEvent` carrying BOTH
asks, and ONE resolution must answer both. Two sequential parks would price two unrelated
questions and make a human answer twice for one decision.
"""

import pytest
from _gate import at_spend

from effective import permission
from effective.budget import Grant, MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.govern import (
    Ask,
    BudgetRefused,
    Exceeded,
    GateState,
    Park,
    Policy,
    Proceed,
    Refuse,
    Refused,
    Resolution,
    as_resolution,
    combine,
    fuse_prompt,
    govern,
    park,
    refuse,
    routable,
)
from effective.handlers.base import step_key
from effective.keys import Key, compose_key
from effective.layers import drive_through
from effective.ops import AppendLedgerRow, AwaitEvent, LedgerRow
from effective.permission import Allow, Deny, Escalate

OP = AppendLedgerRow(row=LedgerRow(event_id=Key.parse("e1"), kind="commitment"))
RUN = "run-g"


def _base(op):
    return "forwarded"


def _always(verdict) -> Policy:
    return lambda op, state: verdict


# --- the transition: `combine` ------------------------------------------------------------


def test_no_objection_proceeds():
    assert combine([Proceed(), Proceed()]) == Proceed()


def test_an_empty_council_proceeds():
    """The fold's unit. (The DRIVER refuses to be built with no policies — see below; this is
    the algebraic identity, not the assembly rule.)"""
    assert combine([]) == Proceed()


def test_any_park_parks_and_the_asks_fuse():
    a, b = Ask("budget", "more?"), Ask("permission", "ok?")
    assert combine([park(a), Proceed(), park(b)]) == Park((a, b))


def test_any_refuse_refuses_and_every_reason_is_kept():
    assert combine([refuse("broke"), park(Ask("p", "?")), refuse("denied")]) == Refuse(
        ("broke", "denied")
    )


def test_refuse_dominates_park():
    """Fail-closed direction: a gate never asks a human to grant past a policy that said no."""
    assert isinstance(combine([park(Ask("b", "?")), refuse("no")]), Refuse)


@pytest.mark.parametrize(
    "verdicts",
    [
        [Proceed(), park(Ask("a", "?")), refuse("no")],
        [refuse("no"), Proceed(), park(Ask("a", "?"))],
        [park(Ask("a", "?")), refuse("no"), Proceed()],
    ],
)
def test_the_ruling_is_order_free(verdicts):
    """`govern` is a council of peers, not a queue: a permutation changes the payload order,
    never which constructor comes out."""
    assert type(combine(verdicts)) is Refuse


def test_the_fused_payload_keeps_argument_order():
    a, b = Ask("first", "?"), Ask("second", "?")
    assert combine([park(a), park(b)]) == Park((a, b))
    assert combine([park(b), park(a)]) == Park((b, a))


# --- gate state: park naming and answer history -------------------------------------------


def test_the_park_name_is_deterministic_gate_run_pass_occurrence_and_OP_scoped():
    state = GateState(run_id=RUN, gate="spend", op_key=step_key("tool:charge_card").stored())
    # OP-scoped: without the op component one approval would settle every later gated op in the
    # run. Five components: `occurrence` separates a repeated op_key, which the op component alone
    # cannot (see `test_govern_op_scope.py`).
    assert state.park_name.stored() == f"govern:spend,{RUN};step;tool:charge_card"
    assert (
        state.absorb(Resolution()).park_name.stored()
        == f"govern:spend,{RUN},pass-n=1;step;tool:charge_card"
    )
    # The SAME op at a later occurrence, the only thing that differs between these two names, so
    # the comparison reads one coordinate at a time.
    keyed = GateState(
        run_id=RUN, gate="spend", op_key=step_key("tool:charge_card").stored(), occurrence=2
    )
    assert keyed.park_name.stored() == f"govern:spend,{RUN},occurrence=2;step;tool:charge_card"


def test_park_names_are_injective_across_gate_run_pass_and_op():
    """Injective in all FOUR components. The op component is what makes an approval settle exactly
    one op."""
    combos = [
        (g, r, n, k)
        for g in ("spend", "approve")
        for r in ("a", "b")
        for n in range(3)
        for k in (step_key("tool:charge_card").stored(), step_key("tool:wire_transfer").stored())
    ]
    names = {GateState(run_id=r, gate=g, pass_n=n, op_key=k).park_name for g, r, n, k in combos}
    assert len(names) == len(combos) == 24


def test_absorb_appends_per_policy_history():
    state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored())
    state = state.absorb(Resolution(answers={"budget": 1, "permission": "yes"}))
    state = state.absorb(Resolution(answers={"budget": 2}))
    assert state.answers["budget"] == (1, 2)  # accrual
    assert state.answers["permission"] == ("yes",)  # untouched by the second pass
    assert state.pass_n == 2


def test_as_resolution_accepts_the_shapes_a_handler_can_deliver():
    assert as_resolution(Resolution(answers={"b": 1})).answers == {"b": 1}
    assert as_resolution({"answers": {"b": 1}}).answers == {"b": 1}
    assert as_resolution({"b": 1}).answers == {"b": 1}
    with pytest.raises(TypeError, match="policy-name -> answer"):
        as_resolution(object())


# --- the driver ---------------------------------------------------------------------------


def test_a_gate_with_no_policies_is_a_misconfiguration_not_a_default():
    """`serve()` with no services is a defensible bypass; a GATE with none permits everything."""
    with pytest.raises(ValueError, match="at least one Policy"):
        govern(gate="g", run_id=RUN)


def test_proceed_forwards_the_op():
    gate = govern(_always(Proceed()), gate="g", run_id=RUN)
    assert drive_through([gate], OP, _base) == "forwarded"


def test_refuse_raises_refused_carrying_every_reason():
    gate = govern(_always(refuse("a")), _always(refuse("b")), gate="g", run_id=RUN)
    with pytest.raises(Refused) as ei:
        drive_through([gate], OP, _base)
    assert ei.value.reason == "a; b"
    assert ei.value.op is OP


def _drive_until_await(gate, op):
    """Advance the gate by hand to its first yielded op (the harness a park needs)."""
    generator = gate(op)
    return generator, generator.send(None)


def test_a_park_yields_exactly_one_await_then_re_asks_with_the_answer():
    seen: list[GateState] = []

    def policy(op, state):
        seen.append(state)
        return Proceed() if state.pass_n else park(Ask("p", "may I?"))

    gate = govern(policy, gate="g", run_id=RUN)
    generator, first = _drive_until_await(gate, OP)
    assert first.name.stored().startswith(f"govern:g,{RUN};")

    forwarded = generator.send(Resolution(answers={"p": "yes"}))
    assert forwarded is OP  # the second pass proceeds -> the op itself is yielded
    assert [s.pass_n for s in seen] == [0, 1]
    assert seen[1].answers["p"] == ("yes",)


def test_a_gate_that_never_settles_refuses_rather_than_parking_forever():
    gate = govern(_always(park(Ask("p", "?"))), gate="g", run_id=RUN, max_passes=3)
    generator = gate(OP)
    parks = [generator.send(None)]

    def _keep_answering():
        while True:
            parks.append(generator.send(Resolution()))

    with pytest.raises(Refused, match="did not settle after 3"):
        _keep_answering()
    assert all(isinstance(op, AwaitEvent) for op in parks)
    assert len(parks) == 4  # max_passes resolutions absorbed, plus the initial park


def test_announce_puts_the_fused_ask_in_the_recorded_stream_before_suspending():
    """The `announce` seam: without it, `Park.asks` would be a field nothing can read."""
    gate = govern(
        _always(park(Ask("p", "may I?"))),
        gate="g",
        run_id=RUN,
        announce=lambda parked, state: AppendLedgerRow(
            row=LedgerRow(
                # lint: terminal-hole — `GovernState.pass_n: int` (`govern.py:159`).
                event_id=compose_key(t"ask:{state.pass_n}"),
                kind="gate_park",
                ask=fuse_prompt(parked),
            )
        ),
    )
    generator = gate(OP)
    announced = generator.send(None)
    assert isinstance(announced, AppendLedgerRow)
    assert announced.row.get("ask") == "1. [p] may I?"
    assert isinstance(generator.send(None), AwaitEvent)  # then, and only then, the park


# --- merged park: the reason `govern` exists ----------------------------------------------


def test_two_heterogeneous_policies_fuse_into_ONE_park_answered_ONCE():
    """A dollars accumulator and an approval classifier both block the same op. One
    `AwaitEvent`, both asks in it, one resolution settling both."""
    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="park")
    gate = govern(
        at_spend(0.010, budget_policy(budget)),  # over the $0.005 ceiling until a grant lands
        permission.as_policy([permission.rules(lambda op: Escalate())]),
        gate="spend",
        run_id=RUN,
    )

    generator = gate(OP)
    awaited = generator.send(None)
    assert isinstance(awaited, AwaitEvent)
    assert awaited.name.stored().startswith(f"govern:spend,{RUN};")

    forwarded = generator.send(
        Resolution(
            answers={
                "budget": Grant(add_dollars=0.010),
                "permission": {"decision": "approve"},
            }
        )
    )
    assert forwarded is OP  # BOTH policies satisfied by the ONE answer


def test_the_fused_prompt_shows_the_operator_the_whole_bundle():
    parked = Park((Ask("budget", "grant $0.01 more?"), Ask("permission", "approve ledger;e1?")))
    assert combine([park(parked.asks[0]), park(parked.asks[1])]) == parked
    assert fuse_prompt(parked) == (
        "1. [budget] grant $0.01 more?\n2. [permission] approve ledger;e1?"
    )


def test_one_policy_refusing_blocks_the_other_policys_park():
    """Merged-park is conjunctive, not a race: an approved op with an exhausted, fail-mode
    budget is still refused — and the human is never asked."""
    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="fail")
    gate = govern(
        at_spend(0.010, budget_policy(budget)),
        permission.as_policy([permission.rules(lambda op: Escalate())]),
        gate="spend",
        run_id=RUN,
    )
    with pytest.raises(BudgetRefused, match="measured budget exceeded") as ei:
        drive_through([gate], OP, _base)
    assert ei.value.exceeded == Exceeded(spent=0.010, ceiling=0.005)


# --- the adapters ---------------------------------------------------------------------------


def test_permission_policy_folds_pure_tiers_by_decide():
    allow = permission.as_policy([permission.rules(lambda op: Allow())])
    deny = permission.as_policy([permission.rules(lambda op: Deny("nope"))])
    state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored())
    assert allow(OP, state) == Proceed()
    assert deny(OP, state) == refuse("nope")


def test_permission_policy_escalation_becomes_the_gates_park():
    """The human tier stops being a special tier: escalation IS the park."""
    policy = permission.as_policy([permission.rules(lambda op: Escalate())])
    ruling = policy(OP, GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored()))
    assert isinstance(ruling, Park)
    assert ruling.asks[0].policy == "permission"


def test_permission_policy_can_fail_closed_where_no_park_is_possible():
    policy = permission.as_policy([permission.rules(lambda op: Escalate())], on_escalate="deny")
    assert policy(
        OP, GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored())
    ) == refuse(permission.FAIL_CLOSED.reason)


def test_permission_policy_reads_the_latest_answer_not_the_history():
    policy = permission.as_policy([permission.rules(lambda op: Escalate())])
    state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored()).absorb(
        Resolution(answers={"permission": {}})
    )
    assert isinstance(policy(OP, state), Refuse)  # a non-approval settles as a refusal
    state = state.absorb(Resolution(answers={"permission": {"decision": "approve"}}))
    assert policy(OP, state) == Proceed()


def test_budget_policy_accrues_the_whole_grant_history():
    """The trip is defined over a LIST of grants — collapsing to the latest would lose it."""
    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="park")
    policy = at_spend(0.010, budget_policy(budget))
    state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored())
    assert isinstance(policy(OP, state), Park)
    state = state.absorb(Resolution(answers={"budget": Grant(add_dollars=0.002)}))
    assert isinstance(policy(OP, state), Park)  # $0.007 ceiling, still under $0.010 spend
    state = state.absorb(Resolution(answers={"budget": Grant(add_dollars=0.004)}))
    assert policy(OP, state) == Proceed()  # $0.011 > $0.010 — the accrual cleared it


def test_budget_policy_accepts_the_shapes_an_operator_surface_sends():
    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="park")
    policy = at_spend(0.010, budget_policy(budget))
    for answer in (Grant(add_dollars=0.01), {"add_dollars": 0.01}, 0.01):
        state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored()).absorb(
            Resolution(answers={"budget": answer})
        )
        assert policy(OP, state) == Proceed()


def test_a_stop_grant_refuses_through_the_policy():
    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="park")
    policy = at_spend(0.010, budget_policy(budget))
    state = GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored()).absorb(
        Resolution(answers={"budget": Grant(stop=True)})
    )
    assert isinstance(policy(OP, state), Refuse)


def test_a_refusal_keeps_the_budget_policys_exceeded_whatever_its_position():
    over = Exceeded(spent=0.010, ceiling=0.005)
    ruling = combine([refuse("denied"), Refuse((over.reason,), over), refuse("also")])
    assert ruling == Refuse(("denied", over.reason, "also"), over)


def test_a_denial_is_routable_and_a_budget_refusal_is_raised_again():
    denial = Refused(OP, "denied")
    assert routable(denial) is denial
    over = BudgetRefused(OP, Exceeded(spent=0.010, ceiling=0.005))
    with pytest.raises(BudgetRefused) as ei:
        routable(over)
    assert ei.value is over


def test_a_yielding_tier_is_rejected_with_the_migration_message():
    """`human(...)` under `govern` is a category error, and the error says why: the gate owns
    the single park, so escalation IS the park now."""

    class _Approval:
        decision = "approve"

    policy = permission.as_policy([permission.human(_Approval)])
    with pytest.raises(TypeError, match="the gate owns the single park"):
        policy(OP, GateState(run_id=RUN, gate="g", op_key=step_key("tool:x").stored()))


def test_a_non_verdict_tier_fails_loudly_rather_than_closed_and_silent():
    """A tier returning the WRONG type is refused by name: read as a deferral, it would fail
    closed invisibly. The error names the culprit and the fix."""
    with pytest.raises(TypeError, match="must return Allow / Deny / Escalate"):
        # deliberately the wrong type — that is the case under test
        permission.decide(["not-a-verdict"])  # ty: ignore[invalid-argument-type]


def test_a_refusal_with_no_stated_reason_still_refuses():
    """Classification is by constructor, not payload: a guard must not be silenceable by an
    empty tuple. (The Lean model characterizes the ruling the same way.)"""
    assert combine([Refuse(()), park(Ask("p", "?"))]) == Refuse(())
    assert combine([Park(()), Proceed()]) == Park(())
