"""Resolve one accounts-receivable case, on a scripted Odoo or on Odoo Community.

    uv run python -m examples.agent_firm sla-cap
    uv run python -m examples.agent_firm sla-cap --odoo     # against `just odoo-up`

The firm's model proposes the credit the customer deducted, a critic reads the contract and
revises it, and the settlement parks until a controller approves it; then the saga posts the
memo, applies the payment and the memo, tells the customer, and queues what is still owed. A
script answers every model call and the approval, so the run needs no keys.

| case       | what the run does                                          |
|------------|------------------------------------------------------------|
| `clean`    | applies the payment, and asks nobody                       |
| `discount` | credits the $400 early-payment discount the contract allows |
| `sla-cap`  | credits $1,000 of the $3,000 deducted: clause 4.2's cap     |
"""

import argparse
import tempfile
from collections.abc import Callable, Mapping
from decimal import Decimal
from functools import partial
from pathlib import Path
from random import choices
from string import ascii_lowercase
from typing import Any
from uuid import uuid4

from effective import CONTRACT_PARAM, Contract, Usage
from effective.budget import MeasuredBudget
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.govern import govern
from effective.handlers.durable import DurableHandler
from effective.keys import Segment
from effective.ledgerread import sqlite_payloads
from effective.parked import answer, read_sqlite_parked_conn

from .ar import Round
from .cases import CASES, Case
from .engage import CritiqueWire, Proposal, Verdict
from .erp.odoo import OdooErp, XmlRpc, seed
from .erp.port import Erp
from .knowledge import Pinned
from .manifest import Manifest, head
from .resolve import APPROVAL, ASKED, admitted, announced, approval, ruled
from .world import World, scripted

PROPOSER, CONTROLLER = Segment("ar-clerk"), Segment("controller")
MAIN = Round("main")
"""The first round of the scripted books, which each run opens afresh."""

CODE = "examples.agent_firm"
"""The code a run of this script is admitted under."""

CRITIQUES = {
    "sla-cap": CritiqueWire(
        verdict=Verdict.REVISE,
        credit=Decimal(1000),
        cites=["MSA-NW-2025"],
        reason="clause 4.2 caps service credits at $1,000 a month",
    ),
    "discount": CritiqueWire(
        verdict=Verdict.ACCEPT,
        credit=Decimal(400),
        cites=["MSA-FR-2024"],
        reason="2/10 net 30, and the payment arrived on day 7",
    ),
}
"""What the critic finds in each case's contract."""


def deducted(case: Case) -> Proposal:
    """The firm's proposal: credit what the customer deducted, on the documents the case holds."""
    cites = tuple(document.id for document in case.documents)
    return Proposal(credit=case.gap, rationale="the customer's deduction", cites=cites)


def books(case: Case, odoo: bool) -> tuple[Erp, Round]:
    """The run's books, and the first round of them. A live Odoo keeps every run's keys under one
    case's ids, so each run on it opens books of its own."""
    if not odoo:
        return scripted(), MAIN
    transport, tag = XmlRpc.from_env(), "".join(choices(ascii_lowercase, k=8))
    return OdooErp(transport, seed(transport, [case], tag)), Round(tag)


def run(admit: Callable[[Pinned], Manifest], world: World, kept_in: str, db: Path) -> Any:
    """Resolve what `admit` admits at the head of the firm's knowledge kept in the books
    `kept_in`, durably on SQLite, answering each approval park as the controller. The run is
    spawned under the v1 contract, so its tape records each model call's usage."""
    app = SqliteApp(str(db))
    manifest = admit(head(SqliteLedger(app.conn, None, app.write_lock).read, kept_in))

    @app.register_task("resolve")
    def task(params: dict[str, Any], ctx: Any) -> Any:
        run_id, admission = params["run_id"], Manifest.admitted(params)
        gate = govern(
            approval(PROPOSER), gate="credit", run_id=run_id, announce=announced, ruled=ruled
        )
        ledger = SqliteLedger(app.conn, run_id, app.write_lock)
        budget = None
        if admission.budget is not None:
            budget = MeasuredBudget(overall=float(admission.budget), run_id=run_id)
        handler = DurableHandler(
            ctx,
            world.domain(),
            ledger=ledger,
            op_layers=(gate,),
            contract=Contract.from_params(params),
            budget=budget,
        )
        return handler.run(admitted(admission))

    try:
        run_id = str(uuid4())
        marked = {"run_id": run_id, CONTRACT_PARAM: Contract.V1.value}
        task_id = app.spawn("resolve", marked | manifest.params())
        done = app.run_until_result(task_id)
        while parked := [p for p in read_sqlite_parked_conn(app.conn) if p.task_id == task_id]:
            asked = [row for row in sqlite_payloads(app.conn, run_id) if row["kind"] == ASKED]
            print("asked:    ", asked[-1]["ask"], *asked[-1]["detail"])
            for park in parked:
                print("approved: ", park.wake_event, "by", CONTROLLER)
                answer(app, park, {"answers": {APPROVAL: {"decision": "approve"}}}, by=CONTROLLER)
            done = app.run_until_result(task_id)
        if done is None or done.state != "completed":
            raise RuntimeError(f"the run ended {done}")
        return done.result
    finally:
        app.close()


def scripted_firm(case: Case, erp: Erp, told: Callable[[Mapping[str, Any]], Any]) -> World:
    """The world a run of this script reaches: its scripted proposer and critic, `erp`, and `told`
    answering the customer's notice and collection."""
    return World(
        erp=erp,
        llm=lambda op: deducted(case),
        critic=lambda op: (CRITIQUES[case.name], Usage()),
        critiques=frozenset({CritiqueWire}),
        tools={"notify": told, "collect": told},
    )


def admits(case: Case, round: Round) -> Callable[[Pinned], Manifest]:
    """What a run of this script admits, at the knowledge it pins."""
    admit = partial(
        Manifest,
        case=case,
        round=round,
        critic=True,
        code=CODE,
        models={"proposer": "scripted", "critic": "scripted"},
    )
    return lambda pinned: admit(knowledge=pinned)


def main() -> None:
    parser = argparse.ArgumentParser(prog="examples.agent_firm")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--odoo", action="store_true", help="settle on the Odoo `odoo-up` starts")
    args = parser.parse_args()
    case = CASES[args.case]

    def told(args: Mapping[str, Any]) -> dict[str, Any]:
        print(
            "told:     ",
            {name: value for name, value in args.items() if name != "idempotency_key"},
        )
        return {"ok": True}

    erp, round = books(case, args.odoo)
    world = scripted_firm(case, erp, told)
    with tempfile.TemporaryDirectory() as directory:
        result = run(admits(case, round), world, round.books, Path(directory) / "firm.db")
    print("result:   ", result)
    for entry in world.erp.entries(case.invoice.id):
        print("entry:    ", entry.kind, entry.amount)
    print("owed:     ", world.erp.residual(case.invoice.id), "on", case.invoice.id)
    print("calls:    ", dict(world.calls))


if __name__ == "__main__":
    main()
