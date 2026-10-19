"""The factor account of a resolution, projected from its admission, its ledger rows and its tape,
and its books scored apart from the firm, on both engines.

| question | the pin                                                                              |
|----------|--------------------------------------------------------------------------------------|
| E1       | on an uncrashed run, the account's model calls are the world's, and v1 its spend     |
| E2       | under a crash after any step's effect, the account undercounts by the calls it lost  |
| E3       | the account names each ruler, from the ledger alone                                  |
| E4       | the books score a wrong credit as wrong, and a superseded run as its round's credit  |
"""

from decimal import Decimal

import pytest
from _conformance import Fault, FaultPosition
from _firm import CONTROLLER, GENESIS, PROPOSER, briefed, manifest, resolved
from _odoo import fresh_round

from effective.cost import Contract
from effective.graphview import bare_name
from examples.agent_firm.account import MODEL_STEPS, account
from examples.agent_firm.ar import Superseded
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.saga import Completed
from examples.agent_firm.score import scored


def _called(firm) -> int:
    """The model calls the world answered, the firm's and its critic's."""
    return firm.world.calls["llm"] + firm.world.calls["critic"]


def _account(backend, run):
    rows = backend.ledger_payloads(run.run_id)
    return account(run.params, rows, backend.checkpoint_states(run.task_id))


@pytest.mark.parametrize(
    ("contract", "spend"), [(Contract.V0, None), (Contract.V1, Decimal("0.01"))], ids=str
)
def test_e1_an_uncrashed_runs_model_calls_are_the_worlds(backend, contract, spend):
    firm = briefed()
    run = resolved(backend, firm, manifest(GENESIS, fresh_round()), contract=contract)
    accounted = _account(backend, run)
    assert (accounted.model_calls, _called(firm)) == (2, 2)
    assert (accounted.asks, accounted.rulers, accounted.spend) == (1, (CONTROLLER,), spend)


@pytest.mark.adversarial
def test_e2_a_crash_after_a_steps_effect_loses_only_a_model_calls_record(backend):
    clean = resolved(backend, briefed(), manifest(GENESIS, fresh_round()))
    tape = backend.checkpoint_states(clean.task_id)
    losses = []
    for key in tape:
        firm = briefed()
        fault = Fault(named=key.stored(), position=FaultPosition.AFTER_THUNK)
        run = resolved(backend, firm, manifest(GENESIS, fresh_round()), fault=fault)
        if fault.armed:  # the op at `key` runs no effect a crash could follow
            continue
        lost = int(bare_name(key.stored()) in MODEL_STEPS)
        assert _called(firm) - _account(backend, run).model_calls == lost, key
        losses.append(lost)
    assert sorted(set(losses)) == [0, 1]
    assert sum(losses) == 2


def test_e3_the_account_names_each_ruler_from_the_ledger_alone(backend):
    firm = briefed()
    run = resolved(
        backend, firm, manifest(GENESIS, fresh_round()), approvers=(PROPOSER, CONTROLLER)
    )
    rows = backend.ledger_payloads(run.run_id)
    assert account(run.params, rows, {}).rulers == (PROPOSER, CONTROLLER)


@pytest.mark.adversarial
def test_e4_the_books_score_the_rounds_credit_whatever_a_run_ended(backend):
    round = fresh_round()
    capped, uncapped = briefed(), briefed()
    uncapped.world.erp = capped.world.erp
    first = resolved(backend, capped, manifest(GENESIS, round))
    assert scored(SLA_CAP, capped.world.erp).correct
    second = resolved(backend, uncapped, manifest(GENESIS, round, critic=False))
    assert Completed.model_validate(first.snapshot.result)
    assert Superseded.model_validate(second.snapshot.result).held == Decimal(1000)
    assert scored(SLA_CAP, capped.world.erp).credit == Decimal(1000)

    wrong = briefed()
    resolved(backend, wrong, manifest(GENESIS, fresh_round(), critic=False))
    assert scored(SLA_CAP, wrong.world.erp).model_dump() == {
        "credit": Decimal(3000),
        "gold": Decimal(1000),
    }


def test_a_hypothetical_runs_account_reads_its_own_rows(backend):
    run = resolved(backend, briefed(), manifest(GENESIS, fresh_round()), hypothetical=True)
    tape = backend.checkpoint_states(run.task_id)
    assert backend.ledger_payloads(run.run_id) == []
    accounted = account(run.params, backend.hypothetical_payloads(run.run_id), tape)
    assert (accounted.asks, accounted.rulers) == (1, (CONTROLLER,))
