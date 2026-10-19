"""The critique program: an outside critic briefed on the `sla-cap` case, run durably on SQLite.

The firm proposes the customer's claimed $3,000 credit; a critic that reads the contract finds its
$1,000 cap. `run` performs one critique and `replay` re-drives a finished run from its checkpoints
with every interpreter raising if asked, so a replayed critique is the recorded one.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any, assert_never
from uuid import UUID

from pydantic import BaseModel

from effective import Contract, MeteredInterpreter
from effective.api import Effect
from effective.channels import Repair
from effective.checkpoints import read_sqlite_task
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.durable import DurableHandler
from effective.viewing import ViewingCtx

from .cases import SLA_CAP
from .engage import Critique, Proposal, Verdict, engage
from .world import World

CLAIMED = Proposal(
    credit=Decimal(3000),
    rationale=(
        "The customer's SLA claim for the September 14 outage: six hours at 0.5% of the monthly "
        "fee, per CRM-2207"
    ),
    cites=("CRM-2207", "TCK-881"),
)


class Heard(BaseModel):
    critique: Critique | None
    refused: str | None


def critiqued() -> Effect[Heard]:
    match (yield from engage(SLA_CAP, CLAIMED)):
        case Repair(reason=reason):
            return Heard(critique=None, refused=reason)
        case Critique() as critique:
            return Heard(critique=critique, refused=None)
        case unreachable:
            assert_never(unreachable)


def finds_the_cap(heard: Heard) -> bool:
    match heard.critique:
        case Critique(verdict=Verdict.REVISE, credit=credit, cites=cites):
            return credit == Decimal(1000) and "MSA-NW-2025" in cites
        case _:
            return False


def run(world: World, db: Path) -> Heard:
    """One critique, as a durable run on the SQLite database `db`."""
    app = SqliteApp(str(db))

    @app.register_task("critique")
    def task(params: dict[str, Any], ctx: Any) -> Any:
        return DurableHandler(ctx, world.domain(), ledger=None, contract=Contract.V1).run(
            critiqued
        )

    try:
        done = app.run_until_result(app.spawn("critique", {}))
        if done is None or done.state != "completed":
            raise RuntimeError(f"the run ended {done}")
        return Heard.model_validate(done.result)
    finally:
        app.close()


def _unasked(op: Any) -> Any:
    raise AssertionError("replay asked a model")


def replay(db: Path) -> Heard:
    """Re-drive the finished run in `db` over its checkpoints, with every interpreter raising if
    called."""
    app = SqliteApp(str(db))
    try:
        (task_id,) = [UUID(str(row[0])) for row in app.conn.execute("SELECT task_id FROM tasks")]
        tape = {checkpoint.key for checkpoint in read_sqlite_task(db, task_id)}
        ctx = ViewingCtx(SqliteTaskContext(app.conn, task_id, app.write_lock), tape)
        interpreter = MeteredInterpreter(llm=_unasked, tools=_unasked, judge=_unasked)
        handler = DurableHandler(ctx, interpreter, ledger=None, contract=Contract.V1)
        return Heard.model_validate(handler.run(critiqued))
    finally:
        app.close()
