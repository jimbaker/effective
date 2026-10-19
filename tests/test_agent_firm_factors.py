"""The factors of a resolution, projected from its admission, its ledger rows and its tape, and
its books scored apart from the firm, on both engines.

| property                    | what holds                                                       |
|-----------------------------|------------------------------------------------------------------|
| model calls counted         | uncrashed, model calls and, under v1, spend are the world's      |
| a crash loses only a record | a crash after a step's effect undercounts by the calls it lost   |
| rulers read from the ledger | the factors name each ruler, from the ledger alone               |
| the books score the credit  | a wrong credit scores wrong; a superseded run scores its round's |
"""

from decimal import Decimal

import pytest
from _conformance import Fault, FaultPosition
from _firm import CONTROLLER, FOUNDING, PROPOSER, briefed, manifest, resolved
from _odoo import fresh_round

from effective.graphview import bare_name
from examples.agent_firm.ar import Superseded
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.factors import MODEL_STEPS, factors
from examples.agent_firm.saga import Completed
from examples.agent_firm.score import scored


def _called(firm) -> int:
    """The model calls the world answered, the firm's and its critic's."""
    return firm.world.calls["llm"] + firm.world.calls["critic"]


def _factors(backend, run):
    rows = backend.ledger_payloads(run.run_id)
    return factors(run.params, rows, backend.checkpoint_states(run.task_id))


def test_an_uncrashed_runs_model_calls_are_the_worlds(backend):
    firm = briefed()
    run = resolved(backend, firm, manifest(FOUNDING, fresh_round()))
    recorded = _factors(backend, run)
    assert (recorded.model_calls, _called(firm)) == (2, 2)
    assert (recorded.asks, recorded.rulers, recorded.spend) == (1, (CONTROLLER,), Decimal("0.01"))


@pytest.mark.adversarial
def test_a_crash_after_a_steps_effect_loses_only_a_model_calls_record(backend):
    clean = resolved(backend, briefed(), manifest(FOUNDING, fresh_round()))
    tape = backend.checkpoint_states(clean.task_id)
    losses = []
    for key in tape:
        firm = briefed()
        fault = Fault(named=key.stored(), position=FaultPosition.AFTER_THUNK)
        run = resolved(backend, firm, manifest(FOUNDING, fresh_round()), fault=fault)
        if fault.armed:  # the op at `key` runs no effect a crash could follow
            continue
        lost = int(bare_name(key.stored()) in MODEL_STEPS)
        assert _called(firm) - _factors(backend, run).model_calls == lost, key
        losses.append(lost)
    assert sorted(set(losses)) == [0, 1]
    assert sum(losses) == 2


def test_the_factors_name_each_ruler_from_the_ledger_alone(backend):
    firm = briefed()
    run = resolved(backend, firm, manifest(FOUNDING, fresh_round()), rulers=(PROPOSER, CONTROLLER))
    rows = backend.ledger_payloads(run.run_id)
    assert factors(run.params, rows, {}).rulers == (PROPOSER, CONTROLLER)


@pytest.mark.adversarial
def test_the_books_score_the_rounds_credit_whatever_a_run_ended(backend):
    round = fresh_round()
    capped, uncapped = briefed(), briefed()
    uncapped.world.erp = capped.world.erp
    first = resolved(backend, capped, manifest(FOUNDING, round))
    assert scored(SLA_CAP, capped.world.erp).correct
    second = resolved(backend, uncapped, manifest(FOUNDING, round, critic=False))
    assert Completed.model_validate(first.snapshot.result)
    assert Superseded.model_validate(second.snapshot.result).held == Decimal(1000)
    assert scored(SLA_CAP, capped.world.erp).credit == Decimal(1000)

    wrong = briefed()
    resolved(backend, wrong, manifest(FOUNDING, fresh_round(), critic=False))
    assert scored(SLA_CAP, wrong.world.erp).model_dump() == {
        "credit": Decimal(3000),
        "gold": Decimal(1000),
    }


def test_a_hypothetical_runs_factors_read_its_own_rows(backend):
    run = resolved(backend, briefed(), manifest(FOUNDING, fresh_round()), hypothetical=True)
    tape = backend.checkpoint_states(run.task_id)
    assert backend.ledger_payloads(run.run_id) == []
    recorded = factors(run.params, backend.hypothetical_payloads(run.run_id), tape)
    assert (recorded.asks, recorded.rulers) == (1, (CONTROLLER,))
