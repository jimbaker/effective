"""The AR firm's settlement saga through the ERP port, crashed at every op, on both engines, on the
scripted Odoo and on a live one.

| control                  | what holds                                                           |
|--------------------------|----------------------------------------------------------------------|
| crash-safe settlement    | one memo posts; it and the payment apply once, whatever crashes      |
| rejection unwinds        | a rejected notice unapplies the memo and payment, reverses the memo  |
| no step without its undo | a saga that cannot be compensated in order is refused before it runs |
| closed-period reversal   | a reversal whose period locked posts in the open one, matched        |

A leg rejected before the pivot undoes only the legs that landed, a retriable leg is asked
`RETRIES` times before the saga stops with it unlanded, and the customer hears each notice and
collection once, however a crash lands. Each control is a row of the table in
`wiki/concepts/agent-firm.md`.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from decimal import Decimal
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from _odoo import books, crashed, durable_sweeps, fresh_round

from effective.api import Effect, claim_ledger, step
from effective.budget import Grant
from effective.contexts import RecordingCtx
from effective.domain import CallTool
from effective.govern import BudgetRefused, Exceeded
from effective.handlers.durable import DurableHandler
from effective.keys import Index, Key, Name, Subject, compose_key
from effective.ops import LedgerRow
from examples.agent_firm.ar import APPLY, REMIND, TERMS, Round, legs_of, settle, settlement
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.erp.port import Apply, Erp, Kind
from examples.agent_firm.erp.scripted import ScriptedOdoo
from examples.agent_firm.firm import Closed, advance
from examples.agent_firm.saga import (
    RETRIES,
    Compensated,
    Completed,
    Landed,
    Leg,
    LegKind,
    Outcome,
    Rejected,
    SagaOrder,
    Unlanded,
    saga,
)
from examples.agent_firm.world import World

CREDIT = Decimal(1000)
INVOICE, PAYMENT = SLA_CAP.invoice.id, SLA_CAP.payment.id
OK = {"ok": True}


def _world(erp: Erp, notify=lambda args: OK, remind=lambda args: OK) -> World:
    return World(erp=erp, tools={"notify": notify, REMIND: remind})


def _run(backend, world: World, fault: Fault, round: Round | None = None) -> Any:
    reason = "clause 4.2 caps service credits at $1,000 a month"
    return _driven(backend, world, fault, settle(SLA_CAP, CREDIT, reason, round or fresh_round()))


def _driven(backend, world: World, fault: Fault, program: Callable[[], Effect[Any]]) -> Any:
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(f"settle-{run_id}", lambda _run_id: program(), world.domain(), fault, ())
    snapshot = backend.run_until_result(backend.spawn(f"settle-{run_id}", run_id))
    assert snapshot.state == "completed", snapshot
    return snapshot.result


def _swept(
    backend, position: FaultPosition, world_of: Callable[[], World], check, arm: str = "scripted"
) -> int:
    """Run unarmed, then crashed at each op the unarmed run reached that `arm` sweeps, checking
    every run's ERP; returns how many crash points there were."""
    unarmed = Fault(None, position=position)
    world = world_of()
    check(_run(backend, world, unarmed), world.erp, None)
    crashes = 0
    for k, fault in crashed(arm, at_every_op(unarmed)):
        world = world_of()
        check(_run(backend, world, fault), world.erp, k)
        crashes += 1
    return crashes


def _names(credit: Decimal) -> list[str]:
    """The settlement's legs at `credit`, by name, in the order they run."""
    return [leg.name for leg in settlement(SLA_CAP, credit, "", fresh_round())]


def _kinds(erp: Erp) -> list[Kind]:
    return [entry.kind for entry in erp.entries(INVOICE)]


def _bounced(args: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": False, "reason": "the customer's AP address bounced"}


def _closed(args: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": False, "reason": "AP is closed today"}


def _ended(round: Round, continued: int = 0) -> Key:
    """Where the ledger holds how `SLA_CAP`'s settlement in `round` ended, or how its
    `continued`th continuation did."""
    books, invoice, payment = Name(round.books), Subject(INVOICE), Subject(PAYMENT)
    return compose_key(
        t"settlement-ended:{books},{invoice},{payment},round={Index(round.number):default=0},"
        t"continued={Index(continued):default=0}"
    )


def _terms(round: Round) -> Key:
    """Where the ledger holds `SLA_CAP`'s claim of `round`."""
    books, invoice, payment = Name(round.books), Subject(INVOICE), Subject(PAYMENT)
    return compose_key(t"settle:{books},{invoice},{payment},round={Index(round.number):default=0}")


ENDINGS = {
    "completed": {},
    "compensated": {"notify": _bounced},
    "unlanded": {"remind": _closed},
}
"""The tools that end a settlement each way."""


@pytest.mark.adversarial
@pytest.mark.parametrize("ending", ENDINGS)
def test_a_settlement_appends_its_ending_to_the_ledger(backend, ending):
    round = fresh_round()
    _run(
        backend,
        _world(ScriptedOdoo.seeded([SLA_CAP]).erp(), **ENDINGS[ending]),
        Fault(None),
        round,
    )
    assert (backend.ledger_read(_ended(round)) or {}).get("ending") == ending


def _claimed(backend, round: Round) -> None:
    """`round` claimed by a settlement that has not ended."""
    row = LedgerRow(
        event_id=_terms(round), kind=TERMS, credit="1000.00", claimant="another settlement"
    )
    name = f"claim-{uuid4().hex[:8]}"
    backend.register(name, lambda _run_id: claim_ledger(row), World().domain(), Fault(None), ())
    assert backend.run_until_result(backend.spawn(name, name)).state == "completed"


@pytest.mark.adversarial
def test_a_round_waits_for_the_earlier_round_to_end(backend):
    earlier = fresh_round()
    _claimed(backend, earlier)
    world = _world(ScriptedOdoo.seeded([SLA_CAP]).erp())
    result = _run(backend, world, Fault(None), Round(earlier.books, 1))
    assert (result["earlier"], result.get("ending"), world.calls["erp"]) == (0, None, 0)


@pytest.mark.adversarial
@pytest.mark.parametrize("ending", ["completed", "unlanded"])
def test_a_round_does_not_open_after_one_that_completed_or_ended_unlanded(backend, ending):
    """A completed case is settled, and an unlanded one continues its own round, so the next
    round of either would credit the case again."""
    earlier, erp = fresh_round(), ScriptedOdoo.seeded([SLA_CAP]).erp()
    _run(backend, _world(erp, **ENDINGS[ending]), Fault(None), earlier)
    world = _world(erp)
    result = _run(backend, world, Fault(None), Round(earlier.books, 1))
    assert (result, world.calls["erp"]) == ({"earlier": 0, "ending": ending}, 0)


@pytest.mark.adversarial
def test_an_unlanded_settlement_continues_at_its_leg_under_fresh_attempts(backend):
    """The continuation reminds the customer once more, under keys no earlier attempt held,
    touches no entry, and appends its own ending."""
    answers = iter([{"ok": False, "reason": "AP is closed today"}] * RETRIES + [OK])
    round, erp = fresh_round(), ScriptedOdoo.seeded([SLA_CAP]).erp()
    world = _world(erp, remind=lambda args: next(answers))
    assert Unlanded.model_validate(_run(backend, world, Fault(None), round))
    calls = Counter(world.calls)
    advanced = partial(advance, SLA_CAP, round.books, None, critic=False)
    closed = Closed.model_validate(_driven(backend, world, Fault(None), advanced))
    assert closed == Closed(case=INVOICE, round=0)
    assert (world.calls["erp"] - calls["erp"], world.calls[REMIND] - calls[REMIND]) == (0, 1)
    assert (backend.ledger_read(_ended(round, 1)) or {}).get("ending") == "completed"
    assert _kinds(erp) == [Kind.INVOICE, Kind.PAYMENT, Kind.CREDIT_MEMO]


@pytest.mark.adversarial
def test_a_round_starts_once_the_earlier_round_ended(backend):
    earlier, erp = fresh_round(), ScriptedOdoo.seeded([SLA_CAP]).erp()
    assert Compensated.model_validate(_run(backend, _world(erp, _bounced), Fault(None), earlier))
    result = _run(backend, _world(erp), Fault(None), Round(earlier.books, 1))
    assert set(Completed.model_validate(result).landed) == {
        "credit-memo",
        APPLY,
        "apply-memo",
        "notify",
        REMIND,
    }


@pytest.mark.adversarial
@pytest.mark.parametrize(("position", "arm"), durable_sweeps(list(FaultPosition)))
def test_a_settlement_posts_once_whatever_a_crash_interrupts(backend, position, arm):
    erp_of = books(arm, [SLA_CAP])

    def check(result, erp: Erp, k):
        landed = Completed.model_validate(result).landed
        assert set(landed) == {"credit-memo", APPLY, "apply-memo", "notify", REMIND}, k
        assert _kinds(erp) == [Kind.INVOICE, Kind.PAYMENT, Kind.CREDIT_MEMO], k
        (memo,) = [e for e in erp.entries(INVOICE) if e.kind is Kind.CREDIT_MEMO]
        assert (memo.amount, erp.keyed(INVOICE)) == (CREDIT, 1), k
        assert (erp.residual(PAYMENT), erp.residual(memo.id), erp.residual(INVOICE)) == (
            0,
            0,
            2000,
        ), k

    assert (
        _swept(backend, position, lambda: _world(erp_of()), check, arm)
        == {
            "scripted": 9,
            "odoo": 0,
        }[arm]
    )


@pytest.mark.adversarial
@pytest.mark.parametrize(("position", "arm"), durable_sweeps(list(FaultPosition)))
def test_a_rejected_notice_unapplies_the_memo_and_the_payment_then_reverses_the_memo(
    backend, position, arm
):
    erp_of = books(arm, [SLA_CAP])

    def check(result, erp: Erp, k):
        compensated = Compensated.model_validate(result)
        undone = (compensated.rejected, compensated.undone)
        assert undone == ("notify", ["apply-memo", APPLY, "credit-memo"]), k
        assert _kinds(erp) == [Kind.INVOICE, Kind.PAYMENT, Kind.CREDIT_MEMO, Kind.REVERSAL], k
        *_, memo, reversal = erp.entries(INVOICE)
        assert (reversal.reverses, reversal.amount, erp.keyed(INVOICE)) == (memo.id, CREDIT, 2), k
        assert (erp.residual(memo.id), erp.residual(reversal.id)) == (0, 0), k
        assert (erp.residual(PAYMENT), erp.residual(INVOICE)) == (97000, 100000), k

    assert (
        _swept(backend, position, lambda: _world(erp_of(), _bounced), check, arm)
        == {
            "scripted": 11,
            "odoo": 0,
        }[arm]
    )


def _spent() -> World:
    """A world whose payment was already applied in full to another of the customer's invoices."""
    odoo = ScriptedOdoo.seeded([SLA_CAP])
    customer = odoo.moves[odoo.documents[INVOICE]].partner
    other = odoo.posted("out_invoice", customer, "2026-08-01", "INV-0999", Decimal(97000))
    odoo.erp().perform(Apply(entry=PAYMENT, invoice=str(other)), "applied before the saga")
    return _world(odoo.erp())


@pytest.mark.adversarial
@pytest.mark.parametrize("position", list(FaultPosition))
def test_a_rejected_leg_undoes_only_the_legs_that_landed(backend, position):
    def check(result, erp: Erp, k):
        compensated = Compensated.model_validate(result)
        assert (compensated.rejected, compensated.undone) == (APPLY, ["credit-memo"]), k
        *_, memo, reversal = erp.entries(INVOICE)
        assert (memo.kind, reversal.kind, reversal.reverses) == (
            Kind.CREDIT_MEMO,
            Kind.REVERSAL,
            memo.id,
        ), k
        assert erp.residual(INVOICE) == 100000, k

    assert _swept(backend, position, _spent, check) == 7


@pytest.mark.adversarial
@pytest.mark.parametrize(("rejections", "landed"), [(RETRIES - 1, True), (RETRIES, False)])
def test_a_retriable_leg_is_asked_until_it_lands_or_runs_out(backend, rejections, landed):
    answers = iter([{"ok": False, "reason": "AP is closed today"}] * rejections + [OK])
    world = _world(ScriptedOdoo.seeded([SLA_CAP]).erp(), remind=lambda args: next(answers))
    result = _run(backend, world, Fault(None))
    assert world.calls[REMIND] == min(rejections + 1, RETRIES)
    if landed:
        assert REMIND in Completed.model_validate(result).landed
    else:
        assert Unlanded.model_validate(result).leg == REMIND
        assert _kinds(world.erp) == [Kind.INVOICE, Kind.PAYMENT, Kind.CREDIT_MEMO]


@pytest.mark.parametrize("granted", [True, False], ids=["granted", "not-granted"])
def test_a_grant_extends_a_retriable_leg_past_its_attempts(backend, granted):
    """The grantor is asked once, when the attempts run out, and not again once the leg lands."""
    asked: list[int] = []

    def one_more(depth: int) -> Effect[Grant]:
        asked.append(depth)
        return Grant(add_depth=1) if granted else Grant(stop=True)
        yield  # a grantor is a generator

    grantor, landed = one_more, granted
    answers = iter([{"ok": False, "reason": "AP is closed today"}] * RETRIES + [OK])
    world = _world(ScriptedOdoo.seeded([SLA_CAP]).erp(), remind=lambda args: next(answers))
    round = fresh_round()
    legs = legs_of(settlement(SLA_CAP, CREDIT, "r", round), SLA_CAP, round, grantor=grantor)
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, lambda _run_id: saga(legs)(), world.domain(), Fault(None), ())
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert snapshot.state == "completed", snapshot
    assert world.calls[REMIND] == RETRIES + landed
    assert (REMIND in snapshot.result.get("landed", {})) is landed
    assert asked == [RETRIES]


class _Refusing:
    """A domain whose budget refuses every op's spend."""

    def run(self, op: Any) -> Any:
        raise BudgetRefused(op, Exceeded(1, 1))


def _reminds(attempt: int, _landed: Mapping[str, Any]) -> Effect[Outcome]:
    yield from step("remind", CallTool(name="remind", args={}, result_schema=dict))
    return Landed()


def test_a_refused_spend_is_raised_out_of_a_retriable_leg():
    """Every later metered op is refused the same way, so the loop does not read it as a
    rejection to retry."""
    leg = Leg("remind", LegKind.RETRIABLE, _reminds)
    with pytest.raises(BudgetRefused):
        DurableHandler(RecordingCtx(), _Refusing()).run(saga([leg]))


@pytest.mark.adversarial
@pytest.mark.parametrize("position", list(FaultPosition))
def test_the_customer_hears_once_whatever_a_crash_interrupts(backend, position):
    heard: Counter[str] = Counter()

    def tool(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        def told(args: Mapping[str, Any]) -> dict[str, Any]:
            heard[name] += 1
            return OK

        return told

    def world() -> World:
        heard.clear()
        return _world(ScriptedOdoo.seeded([SLA_CAP]).erp(), tool("notify"), tool(REMIND))

    def check(result, erp: Erp, k):
        assert heard == {"notify": 1, REMIND: 1}, k

    assert _swept(backend, position, world, check) == 9


def test_a_reversal_after_its_period_locked_posts_in_the_open_period(backend):
    odoo = ScriptedOdoo.seeded([SLA_CAP])

    def locks_then_bounces(args: Mapping[str, Any]) -> dict[str, Any]:
        odoo.locked = "2026-09-30"
        return _bounced(args)

    world = _world(odoo.erp(), locks_then_bounces)
    _run(backend, world, Fault(None))
    *_, memo, reversal = world.erp.entries(INVOICE)
    assert (memo.period, reversal.period, reversal.reverses) == ("2026-09", "2026-10", memo.id)
    assert (world.erp.residual(memo.id), world.erp.residual(reversal.id)) == (0, 0)


def _leg(name: str, kind: LegKind, undo: bool) -> Leg:
    def act(attempt: int = 0, landed: Mapping[str, Any] | None = None):
        yield from ()
        return Landed()

    return Leg(name, kind, act, (lambda _value: act()) if undo else None)


COMPENSABLE = _leg("credit", LegKind.COMPENSABLE, undo=True)
PIVOT = _leg("notify", LegKind.PIVOT, undo=False)
RETRIABLE = _leg("remind", LegKind.RETRIABLE, undo=False)


@pytest.mark.parametrize(
    "legs",
    [
        [PIVOT, COMPENSABLE],
        [COMPENSABLE, RETRIABLE, PIVOT],
        [PIVOT, PIVOT],
        [_leg("credit", LegKind.COMPENSABLE, undo=False), PIVOT],
        [COMPENSABLE, _leg("notify", LegKind.PIVOT, undo=True)],
    ],
    ids=[
        "compensable-after-pivot",
        "pivot-after-retriable",
        "two-pivots",
        "no-undo",
        "pivot-undo",
    ],
)
def test_a_saga_that_cannot_be_compensated_in_order_is_refused(legs):
    with pytest.raises(SagaOrder):
        saga(legs)


def test_a_rejected_undo_fails_the_saga_and_leaves_the_legs_before_it_landed():
    called: list[str] = []

    def answered(name: str, outcome: Outcome) -> Callable[..., Effect[Outcome]]:
        def answer(*_args: Any) -> Effect[Outcome]:
            yield from ()
            called.append(name)
            return outcome

        return answer

    def compensable(name: str, undo: Outcome) -> Leg:
        return Leg(name, LegKind.COMPENSABLE, answered(name, Landed()), answered(name, undo))

    legs = [
        compensable("memo", Landed()),
        compensable("payment", Rejected(reason="locked")),
        compensable("apply", Landed()),
        Leg("notify", LegKind.PIVOT, answered("notify", Rejected(reason="bounced"))),
    ]
    with pytest.raises(RuntimeError, match="the undo of payment was rejected: locked"):
        next(saga(legs)())
    assert called == ["memo", "payment", "apply", "notify", "apply", "payment"]
    assert called.count("memo") == 1


@pytest.mark.parametrize(
    ("credit", "legs"),
    [
        (CREDIT, ["credit-memo", APPLY, "apply-memo", "notify", REMIND]),
        (Decimal(3000), ["credit-memo", APPLY, "apply-memo", "notify"]),
        (Decimal(0), [APPLY, "notify", REMIND]),
    ],
    ids=["capped", "whole-gap", "no-credit"],
)
def test_a_settlement_credits_and_reminds_only_of_what_is_owed(credit, legs):
    assert _names(credit) == legs
