"""The AR firm's runnable example, end to end on embedded SQLite and the scripted Odoo.

Expectations are computed from each case: the memo is the critic's credit, the customer is told
of it, and the invoice owes the gap less the memo, which the settlement applies.
"""

import json
import re
import sqlite3
import sys
from contextlib import closing
from decimal import Decimal

import pytest

from effective import Contract
from effective.checkpoints import read_sqlite_conn
from effective.ledgerread import sqlite_payloads
from examples.agent_firm.__main__ import (
    CONTROLLER,
    CRITIQUES,
    MAIN,
    admits,
    main,
    run,
    scripted_firm,
)
from examples.agent_firm.account import account
from examples.agent_firm.cases import CASES
from examples.agent_firm.world import scripted

pytestmark = pytest.mark.journey


def _printed(capsys, monkeypatch, case: str) -> list[list[str]]:
    monkeypatch.setattr(sys, "argv", ["examples.agent_firm", case])
    main()
    return [line.split() for line in capsys.readouterr().out.splitlines()]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_example_resolves_each_case(capsys, monkeypatch, name):
    case = CASES[name]
    lines = _printed(capsys, monkeypatch, name)
    entries = [line[1:] for line in lines if line[0] == "entry:"]
    memos = [Decimal(amount) for kind, amount in entries if kind == "credit-memo"]
    approvals = [line[-1] for line in lines if line[0] == "approved:"]
    asked = [line for line in lines if line[0] == "asked:"]
    (owed,) = [Decimal(line[1]) for line in lines if line[0] == "owed:"]
    critique = CRITIQUES.get(name)
    assert memos == ([critique.credit] if critique else [])
    assert approvals == ([str(CONTROLLER)] if critique else [])
    assert len(asked) == len(approvals)
    amounts = [
        Decimal(m) for line in asked for m in re.findall(r"'amount': '([0-9.]+)'", " ".join(line))
    ]
    assert amounts == ([critique.credit] if critique else [])
    assert owed == case.gap - (critique.credit if critique else 0)


@pytest.mark.parametrize(
    ("name", "spend", "calls"),
    [("clean", "0", 0), ("discount", "0.01", 2), ("sla-cap", "0.01", 2)],
)
def test_the_example_accounts_its_own_runs_spend(tmp_path, name, spend, calls):
    """The proposer's call costs $0.01 and the critic's nothing, and a matched payment asks no
    model, so the run's account, projected from its own database, spends what its tape recorded."""
    case, db = CASES[name], tmp_path / "firm.db"
    run(
        admits(case, MAIN),
        scripted_firm(case, scripted(), lambda args: {"ok": True}),
        MAIN.books,
        db,
    )
    with closing(sqlite3.connect(db)) as conn:
        task_id, raw = conn.execute("SELECT task_id, params FROM tasks").fetchone()
        params = json.loads(raw)
        tape = {
            checkpoint.key: checkpoint.state
            for checkpoint in read_sqlite_conn(conn, task_id, exclude=())
        }
        accounted = account(params, sqlite_payloads(conn, params["run_id"]), tape)
    marked = Contract.from_params(params)
    assert (accounted.spend, accounted.model_calls, marked) == (Decimal(spend), calls, Contract.V1)
