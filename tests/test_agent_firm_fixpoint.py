"""The AR firm's pass over a case, read from the ledger, on both engines.

| the case's ledger, before the pass | what holds                                                |
|------------------------------------|-----------------------------------------------------------|
| its round completed                | the pass closes it, calling nothing                       |
| its round compensated              | the pass resolves it in the next round                    |
| its round ended unlanded           | the pass continues the round's leg, and opens no round    |
| its round claimed and not ended    | the pass leaves it open, in flight, calling nothing       |
| no claim, the invoice paid         | the pass closes it, asking no model                       |
| no claim, no usable proposal       | the pass leaves it open, stopped                          |
| no claim, the credit denied        | the pass leaves it open, denied, and the next asks nobody |
| replayed                           | the pass reads its statement from the record              |

| the firm over its books               | what holds                                              |
|---------------------------------------|---------------------------------------------------------|
| the canonical cases                   | settled in two passes                                   |
| a case stopped once                   | settled on the next pass, in three                      |
| a case stopped every time             | open, stopped: the firm's dissent                       |
| a case denied beside settling ones    | open, denied, and its siblings settle                   |
| a budget of one pass                  | `Unconverged`, holding that pass                        |
| a case compensated every pass         | `Unconverged`: it moves a round each pass               |
| a reminder out of attempts twice      | continued until it lands, and closed                    |
| a compensation, then an unlanded one  | settled in the next round, its reminder continued       |

Two runs of the firm, the first converged with every case closed, equal one in the books and in
what the customer is told, which is what the done pins hold: the canonical cases run twice, two
runs of one case in every schedule within one switch, and one run crashed at every op. The runs
differ in the reads they record, since each read is a step of the run that made it.
"""

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from decimal import Decimal
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from _firm import CAPPED, CONTROLLER, Run, gate, replayed, ruling
from _turns import Turns, gather_branches, schedules

from effective.api import Effect, claim_ledger, gather
from effective.combinators import Converged, Settled, Unconverged
from effective.cost import Usage
from effective.keys import Index, Key, Name, Subject, compose_key
from effective.ops import LedgerRow
from effective.parked import answer
from examples.agent_firm.ar import ENDED, REMIND, TERMS, Round, _ended_of
from examples.agent_firm.cases import CASES, DISCOUNT, SLA_CAP, Case
from examples.agent_firm.critic import AS_CLAIMED
from examples.agent_firm.engage import REPAIRS, CritiqueWire, Proposal
from examples.agent_firm.erp.port import Apply
from examples.agent_firm.erp.scripted import ScriptedOdoo
from examples.agent_firm.firm import Books, Closed, Last, Open, advance, firm
from examples.agent_firm.saga import RETRIES
from examples.agent_firm.world import World

INVOICE, PAYMENT = SLA_CAP.invoice.id, SLA_CAP.payment.id
OK = {"ok": True}


def _answers(*answers: dict[str, Any]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """A tool answering `answers` in turn, then `OK`."""
    queued: Iterator[dict[str, Any]] = iter(answers)
    return lambda args: next(queued, OK)


def _world(odoo: ScriptedOdoo | None = None, proposal: Any = AS_CLAIMED, **tools: Any) -> World:
    return World(
        erp=(odoo or ScriptedOdoo.seeded([SLA_CAP])).erp(),
        llm=lambda op: proposal,
        critic=lambda op: (CritiqueWire.model_validate(CAPPED), Usage()),
        critiques=frozenset({CritiqueWire}),
        tools={"notify": lambda args: OK, REMIND: lambda args: OK} | tools,
    )


type Told = list[tuple[str, list[tuple[str, Any]]]]


def _telling(world: World) -> Told:
    """Each notice and reminder `world` sends from now on, by the tool and its arguments less the
    key, which names the run that sent it. A repeat under one key is answered unsent."""
    told: Told = []

    def recorded(name: str, inner: Callable[[Mapping[str, Any]], Any]) -> Any:
        def tool(args: Mapping[str, Any]) -> Any:
            told.append((name, sorted((k, v) for k, v in args.items() if k != "idempotency_key")))
            return inner(args)

        return tool

    world.tools = {**world.tools} | {
        name: recorded(name, world.tools[name]) for name in ("notify", REMIND)
    }
    return told


def _effects(world: World, told: Told, cases: Sequence[Case]) -> tuple[Any, ...]:
    """What the firm did: each invoice's entries and what it still owes, and what the customer
    was told."""
    books = [
        (
            [(entry.kind, entry.amount) for entry in world.erp.entries(case.invoice.id)],
            world.erp.residual(case.invoice.id),
        )
        for case in cases
    ]
    return books, sorted(told)


def _books() -> str:
    """Books no other run keeps, since the test databases keep every run's ledger."""
    return f"books-{uuid4().hex[:12]}"


def _driven(
    backend,
    world: World,
    program: Callable[[], Effect[Any]],
    decision: str | Callable[[int], str] = "approve",
    *,
    fault: Fault | None = None,
    turns: Turns | None = None,
) -> Run:
    """Run `program` under the credit gate, answering each park with `decision`, or with what
    `decision` says of the park's place in the order the run parked; under `fault`, and taking
    `turns` when given."""
    run_id, domain = f"f-{uuid4().hex[:8]}", world.domain()
    layers = (gate(run_id),) if turns is None else (turns.layer(), gate(run_id))
    fresh = None if turns is None else turns.claiming(domain)
    backend.register(
        run_id, lambda _: program(), domain, fault or Fault(None), layers, fresh=fresh
    )
    task_id = backend.spawn(run_id, run_id)
    snapshot = backend.run_until_result(task_id)
    parks = 0
    while parked := backend.parked(task_id):
        for park in parked:
            decided = decision if isinstance(decision, str) else decision(parks)
            answer(backend.engine, park, ruling(backend, run_id, park, decided), by=CONTROLLER)
            parks += 1
        snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    ((_, params),) = backend.enqueued(run_id)
    return Run(task_id, run_id, snapshot, params, None, parks, [])


def _passed(
    backend, world: World, books: str, last: Open | None = None, decision: str = "approve"
) -> Open | Closed:
    run = _driven(backend, world, lambda: advance(SLA_CAP, books, last, critic=True), decision)
    return _read(run.result)


def _read(result: dict[str, Any]) -> Open | Closed:
    return Open.model_validate(result) if "last" in result else Closed.model_validate(result)


def _opened(last: Last, round: int) -> Open:
    broken = ("paid-in-full",)
    return Open(case=INVOICE, broken=broken, last=last, round=round)


def _terms(books: str, round: int) -> Key:
    held, invoice, payment = Name(books), Subject(INVOICE), Subject(PAYMENT)
    return compose_key(t"settle:{held},{invoice},{payment},round={Index(round):default=0}")


def _calls(world: World, before: Counter[str]) -> Counter[str]:
    """The calls `world` received since it had received `before`."""
    return world.calls - before


@pytest.mark.adversarial
def test_a_completed_case_closes_calling_nothing(backend):
    world, books = _world(), _books()
    assert _passed(backend, world, books) == Closed(case=INVOICE, round=0)
    before = Counter(world.calls)
    assert _passed(backend, world, books) == Closed(case=INVOICE, round=0)
    assert _calls(world, before) == Counter()


@pytest.mark.adversarial
def test_a_compensated_case_is_resolved_in_the_next_round(backend):
    bounced = {"ok": False, "reason": "the address bounced"}
    world, books = _world(notify=_answers(bounced)), _books()
    assert _passed(backend, world, books) == _opened(Last.COMPENSATED, 0)
    assert _passed(backend, world, books) == Closed(case=INVOICE, round=1)
    assert world.erp.residual(INVOICE) == 2000


@pytest.mark.adversarial
def test_an_unlanded_case_continues_its_round_and_opens_no_other(backend):
    closed = {"ok": False, "reason": "AP is closed today"}
    world, books = _world(remind=_answers(*[closed] * RETRIES)), _books()
    assert _passed(backend, world, books) == _opened(Last.UNLANDED, 0)
    before = Counter(world.calls)
    assert _passed(backend, world, books) == Closed(case=INVOICE, round=0)
    assert _calls(world, before) == Counter({REMIND: 1})
    assert backend.ledger_read(_terms(books, 1)) is None


@pytest.mark.adversarial
def test_a_round_another_settlement_holds_leaves_the_case_in_flight(backend):
    world, books = _world(), _books()
    row = LedgerRow(event_id=_terms(books, 0), kind=TERMS, credit="1000.00", claimant="another")
    _driven(backend, world, lambda: claim_ledger(row))
    before = Counter(world.calls)
    assert _passed(backend, world, books) == _opened(Last.IN_FLIGHT, 0)
    assert _calls(world, before) == Counter()


@pytest.mark.adversarial
def test_a_case_whose_invoice_was_paid_closes_asking_no_model(backend):
    odoo = ScriptedOdoo.seeded([SLA_CAP])
    customer = odoo.moves[odoo.documents[INVOICE]].partner
    rest = odoo.posted("entry", customer, SLA_CAP.payment.received, "PAY-5502", Decimal(3000))
    for entry in (PAYMENT, str(rest)):
        odoo.erp().perform(Apply(entry=entry, invoice=INVOICE), f"paid by {entry}")
    world, books = _world(odoo), _books()
    assert _passed(backend, world, books) == Closed(case=INVOICE, round=0)
    assert (world.asked, backend.ledger_read(_terms(books, 0))) == (Counter(), None)


@pytest.mark.adversarial
def test_a_case_with_no_usable_proposal_stays_open_stopped(backend):
    world, books = _world(proposal=AS_CLAIMED.model_copy(update={"credit": 4000})), _books()
    assert _passed(backend, world, books) == _opened(Last.STOPPED, 0)


@pytest.mark.adversarial
def test_a_denied_case_stays_open_and_the_next_pass_asks_nobody(backend):
    world, books = _world(), _books()
    denied = _opened(Last.DENIED, 0)
    assert _passed(backend, world, books, decision="deny") == denied
    before = Counter(world.calls)
    assert _passed(backend, world, books, denied) == denied
    assert _calls(world, before) == Counter()


def test_a_replayed_pass_reads_its_statement_from_the_record(backend):
    world, books = _world(), _books()
    program = partial(advance, SLA_CAP, books, None, critic=True)
    run = _driven(backend, world, program)
    assert _read(replayed(backend, run, program)) == _read(run.result)


PARTIAL = AS_CLAIMED.model_copy(update={"credit": Decimal(1000)})
"""`SLA_CAP` credited $1,000 of its $3,000 shortfall, so its settled invoice still owes."""


def _proposed(*stopping: Decimal) -> Callable[[Any], Proposal]:
    """The firm's model: `DISCOUNT`'s shortfall and `PARTIAL` as the credits, `SLA_CAP` asked
    first at each of `stopping` in turn."""
    queued = iter(stopping)

    def propose(op: Any) -> Proposal:
        if DISCOUNT.invoice.id in str(op.messages):
            return Proposal(credit=Decimal(400), rationale="2/10 net 30", cites=("MSA-FR-2024",))
        return PARTIAL.model_copy(update={"credit": next(queued, PARTIAL.credit)})

    return propose


def _settled(
    backend, world: World, decision: str | Callable[[int], str] = "approve", **firmed: Any
) -> tuple[Settled[Books], Run]:
    books = _books()
    program = partial(firm, books, list(CASES.values()), critic=False, **firmed)
    run = _driven(backend, world, program, decision)
    return _settled_as(run.result), run


def _settled_as(result: Any) -> Settled[Books]:
    match result:
        case {"value": value, "cause": cause}:
            return Unconverged(Books.model_validate(value), cause)
        case {"value": value}:
            return Converged(Books.model_validate(value))
        case unexpected:
            raise AssertionError(unexpected)


def _firm_world(proposer: Callable[[Any], Proposal]) -> World:
    """A world holding the canonical cases' books, with `proposer` as the firm's model."""
    world = _world(ScriptedOdoo.seeded(CASES.values()))
    world.llm = proposer
    return world


def _owed(world: World) -> dict[str, Decimal]:
    return {case.name: world.erp.residual(case.invoice.id) for case in CASES.values()}


@pytest.mark.adversarial
def test_the_firm_settles_the_canonical_cases_in_two_passes(backend):
    world = _firm_world(_proposed())
    settled, _ = _settled(backend, world)
    assert settled == Converged(Books(passes=2, open={}))
    assert _owed(world) == {"sla-cap": 2000, "clean": 0, "discount": 0}


@pytest.mark.adversarial
def test_a_case_stopped_once_settles_on_the_next_pass(backend):
    world = _firm_world(_proposed(*[Decimal(4000)] * (1 + REPAIRS)))
    settled, _ = _settled(backend, world)
    assert settled == Converged(Books(passes=3, open={}))


@pytest.mark.adversarial
def test_a_case_stopped_every_time_is_the_firms_dissent(backend):
    world = _firm_world(_proposed(*[Decimal(4000)] * 3 * (1 + REPAIRS)))
    settled, _ = _settled(backend, world)
    stopped = Open(case=INVOICE, broken=("paid-in-full",), last=Last.STOPPED, round=0)
    assert settled == Converged(Books(passes=2, open={INVOICE: stopped}))
    assert _owed(world) == {"sla-cap": 100000, "clean": 0, "discount": 0}


@pytest.mark.adversarial
def test_a_case_the_firm_left_stopped_closes_on_a_later_pass(backend):
    """The firm stops when two passes agree, and a stopped case is still asked on the next pass."""
    world, books = _firm_world(_proposed(*[Decimal(4000)] * 2 * (1 + REPAIRS))), _books()
    firmed = partial(firm, books, list(CASES.values()), critic=False)
    stopped = Open(case=INVOICE, broken=("paid-in-full",), last=Last.STOPPED, round=0)
    assert _settled_as(_driven(backend, world, firmed).result) == Converged(
        Books(passes=2, open={INVOICE: stopped})
    )
    advanced = _driven(backend, world, partial(advance, SLA_CAP, books, stopped, critic=False))
    assert _read(advanced.result) == Closed(case=INVOICE, round=0)


@pytest.mark.parametrize("critic", [True, False], ids=["critic", "no-critic"])
def test_an_exception_asks_one_approval_with_or_without_the_critic(backend, critic):
    world, books = _world(), _books()
    run = _driven(backend, world, lambda: advance(SLA_CAP, books, None, critic=critic))
    assert (_read(run.result), run.parks) == (Closed(case=INVOICE, round=0), 1)


@pytest.mark.adversarial
def test_a_denied_case_stays_open_beside_its_settling_siblings(backend):
    """The gathered parks wake lowest branch first, so the first ruling is `SLA_CAP`'s."""
    world = _firm_world(_proposed())
    settled, run = _settled(backend, world, lambda parked: "deny" if parked == 0 else "approve")
    denied = Open(case=INVOICE, broken=("paid-in-full",), last=Last.DENIED, round=0)
    assert settled == Converged(Books(passes=2, open={INVOICE: denied}))
    assert (_owed(world), run.parks) == ({"sla-cap": 100000, "clean": 0, "discount": 0}, 2)


@pytest.mark.adversarial
def test_a_budget_of_one_pass_answers_unconverged_holding_it(backend):
    world = _firm_world(_proposed())
    settled, _ = _settled(backend, world, budget=1)
    assert settled == Unconverged(Books(passes=1, open={}), "levels")


@pytest.mark.adversarial
def test_a_case_compensated_every_pass_answers_unconverged(backend):
    """Every notice bounces, so each case is open after every pass, in a later round each time,
    and the firm has not converged."""
    world = _firm_world(_proposed())
    world.tools = {**world.tools, "notify": lambda args: {"ok": False, "reason": "bounced"}}
    settled, _ = _settled(backend, world)
    match settled:
        case Unconverged(value=Books(passes=passes, open={**cases}), cause="levels"):
            moved = {(case.last, case.round) for case in cases.values() if case is not None}
            assert (set(cases), moved) == (
                {case.invoice.id for case in CASES.values()},
                {(Last.COMPENSATED, passes - 1)},
            )
        case unexpected:
            raise AssertionError(unexpected)


BOUNCED = {"ok": False, "reason": "AP is closed today"}


@pytest.mark.adversarial
def test_a_reminder_out_of_attempts_twice_is_continued_until_it_lands(backend):
    world = _world(proposal=PARTIAL, remind=_answers(*[BOUNCED] * (2 * RETRIES)))
    run = _driven(backend, world, _firmed([SLA_CAP], _books()))
    assert _settled_as(run.result) == Converged(Books(passes=4, open={}))
    assert (world.calls[REMIND], world.erp.residual(INVOICE)) == (2 * RETRIES + 1, 2000)


def test_a_continuation_that_runs_out_again_leaves_the_case_unlanded(backend):
    world, books = _world(remind=_answers(*[BOUNCED] * (2 * RETRIES))), _books()
    assert _passed(backend, world, books) == _opened(Last.UNLANDED, 0)
    before = Counter(world.calls)
    assert _passed(backend, world, books, _opened(Last.UNLANDED, 0)) == _opened(Last.UNLANDED, 0)
    assert _calls(world, before)[REMIND] == RETRIES
    assert _passed(backend, world, books, _opened(Last.UNLANDED, 0)) == Closed(
        case=INVOICE, round=0
    )


@pytest.mark.parametrize(
    ("ending", "result", "closed"),
    [
        ("completed", {"landed": {}}, 0),
        ("compensated", {"rejected": "notify", "reason": "r", "undone": []}, 1),
    ],
    ids=["completed", "compensated"],
)
def test_a_case_in_flight_follows_its_holders_ending(backend, ending, result, closed):
    world, books = _world(), _books()
    held = LedgerRow(event_id=_terms(books, 0), kind=TERMS, credit="1000.00", claimant="another")
    _driven(backend, world, lambda: claim_ledger(held))
    assert _passed(backend, world, books) == _opened(Last.IN_FLIGHT, 0)
    row = LedgerRow(
        event_id=_ended_of(SLA_CAP, Round(books, 0)), kind=ENDED, ending=ending, result=result
    )
    _driven(backend, world, lambda: claim_ledger(row))
    assert _passed(backend, world, books, _opened(Last.IN_FLIGHT, 0)) == Closed(
        case=INVOICE, round=closed
    )


@pytest.mark.adversarial
def test_a_compensation_then_an_unlanded_reminder_settles_in_the_next_round(backend):
    world = _world(
        proposal=PARTIAL, notify=_answers(BOUNCED), remind=_answers(*[BOUNCED] * RETRIES)
    )
    books = _books()
    run = _driven(backend, world, _firmed([SLA_CAP], books))
    assert _settled_as(run.result) == Converged(Books(passes=4, open={}))
    assert world.erp.residual(INVOICE) == 2000
    assert backend.ledger_read(_terms(books, 1)) is not None


def _firmed(cases: Sequence[Case], books: str) -> Callable[[], Effect[Settled[Books]]]:
    return lambda: firm(books, cases, critic=False)


@pytest.mark.adversarial
def test_the_canonical_firm_run_twice_equals_it_run_once(backend):
    cases, books = list(CASES.values()), _books()
    world = _firm_world(_proposed())
    told = _telling(world)
    _driven(backend, world, _firmed(cases, books))
    once = _effects(world, told, cases)
    again = _driven(backend, world, _firmed(cases, books))
    assert (again.result, again.parks) == ({"value": {"passes": 2, "open": {}}}, 0)
    assert _effects(world, told, cases) == once


def _once(backend) -> tuple[Any, ...]:
    """What one firm run over `SLA_CAP` does."""
    world = _world(proposal=PARTIAL)
    told = _telling(world)
    _driven(backend, world, _firmed([SLA_CAP], _books()))
    return _effects(world, told, [SLA_CAP])


@pytest.mark.adversarial
def test_two_firm_runs_of_one_case_equal_one_in_every_schedule(backend):
    """Every schedule of the two runs' ops within one switch, across the approval park each can
    take."""
    once, schedules_run = _once(backend), 0
    for turns in schedules(gather_branches(2), switches=1):
        world, books = _world(proposal=PARTIAL), _books()
        told = _telling(world)
        runs = turns.gathered([_firmed([SLA_CAP], books) for _ in range(2)])
        run = _driven(backend, world, partial(gather, runs), turns=turns)
        turns.settle()
        settled = [_settled_as(result) for result in run.result]
        assert all(s == Converged(Books(passes=s.value.passes, open={})) for s in settled), (
            turns.choices
        )
        assert _effects(world, told, [SLA_CAP]) == once, turns.choices
        schedules_run += 1
    assert schedules_run > 1


@pytest.mark.adversarial
@pytest.mark.parametrize("position", list(FaultPosition))
def test_a_firm_run_equals_one_whatever_a_crash_interrupts(backend, position):
    once, unarmed = _once(backend), Fault(None, position=position)
    _driven(backend, _world(proposal=PARTIAL), _firmed([SLA_CAP], _books()), fault=unarmed)
    for k, fault in at_every_op(unarmed):
        world = _world(proposal=PARTIAL)
        told = _telling(world)
        run = _driven(backend, world, _firmed([SLA_CAP], _books()), fault=fault)
        assert run.result == {"value": {"passes": 2, "open": {}}}, k
        assert _effects(world, told, [SLA_CAP]) == once, k
