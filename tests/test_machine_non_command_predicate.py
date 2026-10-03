"""A third embodiment whose success predicate is NOT a command: the case parameterization is for.

ROLE: conformance. The coding machine shells out to pytest; the docs machine shells out to a link
checker. Both are commands, so both reuse `CommandRun` and neither one needs the record type to
vary, which means neither one can tell whether parameterizing it bought anything.

This one can. A sweep machine's predicate produces an objective VECTOR: a score, a cost, a
latency. Nothing ran as a process, so there is no exit code and nothing "failed" by name. The real
article is `effective.improve.Measurement` (`measures: dict[str, float]` plus the ASI diagnostic),
which `bench_sweep` already produces; `SweepScore` here is the same shape declared locally,
because `effective/` may not import `agent/`.

**What this file falsifies.** Bind `Measured` to `green` AND `exit_code` AND `failures`, and
`ty` refuses this record at the type boundary. That is why the row carries the record whole
instead of naming two of its fields.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, assert_never

import pytest

from effective.api import call_tool
from effective.domain import CallTool, DomainOp
from effective.handlers.absurd import DurableHandler
from effective.keys import Run
from effective.machine.evidence import Predicate
from effective.machine.outcomes import Advance, Exhausted, Finish, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import build_specs
from effective.machine.trampoline import run_machine
from effective.sqlite import SqliteApp, SqliteLedger

pytestmark = pytest.mark.conformance

SWEEP = "sweep_candidate"


@dataclass(frozen=True, slots=True)
class SweepScore:
    """An objective vector. No process ran, so no exit code and no named failures.

    `green` is the ONE thing the substrate reads, and it is this embodiment's own definition of
    good enough — which is the point: the machine does not know what "passing" means here, only
    that this record can say."""

    measures: dict[str, float]
    asi: str = ""

    @property
    def green(self) -> bool:
        return (
            self.measures.get("score", 0.0) >= 0.9 and self.measures.get("cost_usd", 1.0) <= 0.05
        )


SWEEP_PREDICATE = Predicate(SWEEP, SweepScore)
"""Declared once: the tool a sweep deployment serves, and the record it answers with."""


class SweepState(StrEnum):
    PROPOSE = "propose"
    MEASURE = "measure"


class SweepVerdict(StrEnum):
    GOOD_ENOUGH = "good-enough"
    RETUNE = "retune"


def route(state: SweepState, verdict: SweepVerdict | Exhausted[SweepState]) -> Outcome[SweepState]:
    match state, verdict:
        case _, Exhausted():
            return Park(state, ParkReason.EXHAUSTED)
        case SweepState.PROPOSE, _:
            return Advance(SweepState.MEASURE)
        case SweepState.MEASURE, SweepVerdict.GOOD_ENOUGH:
            return Finish()
        case SweepState.MEASURE, SweepVerdict.RETUNE:
            return Advance(SweepState.PROPOSE)
        case _:
            assert_never(verdict)


class SweepDeployment:
    """Serves a sweep, and nothing a coding machine would recognise."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.rounds = 0
        self.proposed_against: list[str] = []

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        self.tools.append(op.name)
        match op.name:
            case "propose":
                self.proposed_against.append(op.args.get("asi", ""))
                return {"config.json": '{"temperature": 0.2}'}
            case _ if op.name == SWEEP:
                self.rounds += 1
                good = self.rounds >= 2
                return SweepScore(
                    measures={"score": 0.94 if good else 0.71, "cost_usd": 0.01},
                    asi="the first candidate over-spent its context budget",
                )
            case unserved:
                raise KeyError(f"a sweep deployment does not serve {unserved!r}")


type SweepCtx = Ctx[SweepState, SweepScore]
"""This embodiment's `Ctx`, spelled ONCE.

`Ctx` is generic over the RECORD as well as the state, because `ctx.incoming` carries the previous
visit's whole report and this embodiment reads `measured.measures` — a field `Measured` does not
have. A bare `Ctx` would default the record to `CommandRun` and then disagree with the
`Evidence[SweepScore]` beside it, which `build_specs` refuses by type.

Note who pays: a WORKER never touches `incoming`, and pays anyway, because the parameter is on the
type it receives rather than on the field it uses. An alias is the whole mitigation, and an
embodiment over `CommandRun` needs none — the default absorbs it."""


def propose(ctx: SweepCtx) -> Any:
    """Propose a candidate — against the last round's diagnostic, once there is one.

    **This is the read `Ctx`'s record parameter exists for, and until now nothing performed it.**
    `asi` is a field `Measured` does not have, so reaching it requires `incoming` to be typed by
    THIS embodiment's record rather than by the protocol. Both halves of that were run, not
    reasoned: spell the alias `Ctx[SweepState]` and the PEP 696 default fires, `measured` comes
    back `CommandRun | None`, and `ty` refuses the assignment below and the `build_specs` call
    beside it — two diagnostics. Widen the field to `Report[Any, Any]` instead and the whole repo
    stays ty-clean: the alternative costs nothing a checker can see. What the alias buys is over
    the read below once its annotation is removed — see `machine/spec.py`'s `incoming`, which
    carries that as a command. An optimizer that proposes without reading the previous
    measurement's gradient is the degenerate one, so the alias is buying something a deployment
    wants anyway."""
    prior: SweepScore | None = ctx.incoming.measured if ctx.incoming is not None else None
    tree: dict[str, str] = yield from call_tool(
        "propose", {"asi": prior.asi if prior is not None else ""}, dict[str, str]
    )
    return Evidence[SweepScore](summary=f"{ctx.state.value}: candidate written", tree=tree)


def measure(ctx: SweepCtx) -> Any:
    score: SweepScore = yield from call_tool(SWEEP, {}, SweepScore)
    return Evidence(summary=f"{ctx.state.value}: {score.measures}", measured=score)


def judge_propose(_ctx: SweepCtx, _evidence: Evidence[SweepScore]) -> Any:
    return SweepVerdict.RETUNE
    yield  # pragma: no cover  -- a `Judge` is a generator


def judge_measure(_ctx: SweepCtx, evidence: Evidence[SweepScore]) -> Any:
    assert evidence.measured is not None
    # The embodiment reads its OWN record's fields, typed. `measures` is not on `Measured`.
    return SweepVerdict.GOOD_ENOUGH if evidence.measured.green else SweepVerdict.RETUNE
    yield  # pragma: no cover


def sweep_specs():
    return build_specs(
        SweepState,
        workers={SweepState.PROPOSE: propose, SweepState.MEASURE: measure},
        judges={SweepState.PROPOSE: judge_propose, SweepState.MEASURE: judge_measure},
        canonical=frozenset({SweepState.MEASURE}),
    )


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def drive(app, deployment: SweepDeployment):
    @app.register_task("sweep")
    def task(params, ctx):
        rid = params["run_id"]
        return DurableHandler(
            ctx, deployment, ledger=SqliteLedger(app.conn, rid, app.write_lock)
        ).run(
            lambda: run_machine(
                Run(rid),
                "find a cheap configuration that scores",
                sweep_specs(),
                route,
                start=SweepState.PROPOSE,
                budget=8,
                tree={},
                predicate=SWEEP_PREDICATE,
            )
        )

    return app.run_until_result(app.spawn("sweep", {"run_id": "sweep-1"}))


def test_a_machine_whose_predicate_is_not_a_command_walks_to_a_finish(app):
    """The whole claim. No exit code exists anywhere in this embodiment."""
    deployment = SweepDeployment()
    snap = drive(app, deployment)
    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert [turn["state"] for turn in snap.result["turns"]] == [
        "propose",
        "measure",
        "propose",
        "measure",
    ]
    assert set(deployment.tools) == {"propose", SWEEP}, deployment.tools


def test_the_retune_round_proposes_against_the_last_measurement(app):
    """What the record parameter on `Ctx` buys, stated as a property of the WALK.

    The first PROPOSE has no predecessor and proposes against nothing. The second is entered by
    the RETUNE edge, so `ctx.incoming` is MEASURE's report and `measured.asi` is the diagnostic
    the failing round produced — a field `Measured` does not declare, reachable only because
    `incoming` is typed `Report[Any, SweepScore]`.

    `spec.py` justifies that parameter by `incoming`, and a field with no reader anywhere in the
    tree is write-only. This is the reader."""
    deployment = SweepDeployment()
    snap = drive(app, deployment)
    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert deployment.proposed_against == [
        "",
        "the first candidate over-spent its context budget",
    ], deployment.proposed_against


def test_the_outcome_row_carries_the_records_OWN_fields(app):
    """The row is self-describing, which is what let `Measured` stay at one member.

    The row does not name `exit_code` and `failures`: naming them would rule this record out and
    claim a vocabulary this domain has no answer for."""
    deployment = SweepDeployment()
    drive(app, deployment)
    rows = list(app.conn.execute("SELECT kind, payload FROM ledger ORDER BY rowid"))
    outcome = [json.loads(payload) for kind, payload in rows if kind == "machine-finished"]
    assert len(outcome) == 1
    measured = outcome[0]["measured"]
    assert measured["measures"] == {"score": 0.94, "cost_usd": 0.01}
    assert "exit_code" not in measured
    assert outcome[0]["passed"] is True


def test_the_predicate_ran_after_the_walk_stopped(app):
    """The unconditional tail, on an embodiment that has never heard of pytest: the last tool call
    is the commitment point reading the predicate, once more, after the machine finished."""
    deployment = SweepDeployment()
    drive(app, deployment)
    assert deployment.tools[-1] == SWEEP
