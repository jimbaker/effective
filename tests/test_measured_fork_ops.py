"""Forking a REAL workflow: op coverage for `measured_drive`, inspect-only.

A measured driver that handled only `Step` could drive only toy programs. The exemplar
`process_refund` yields a `call_tool`, a `store_artifact`, an `ask_llm`, two `append_ledger`s and
an `await_event`, so `measured_drive` covers every one of them. This is the substrate half of
forking a deployed run.

Covering those ops is NOT "do what the durable handler does". A fork is a **counterfactual**: it
asks what a run would have cost or answered, and a counterfactual that writes is not a
counterfactual; it is a second run. The ledger is the canonical record (two bookkeepers), so the
rule under test here is: **observe, return what the workflow needs, write nothing.**

Infra-free: the point is the driver's op coverage, not durability.
"""

from typing import Any

import pytest
from _approval_domain import (
    ApprovalEvent,
    Assessment,
    RefundRequest,
    process_refund,
    review_name,
    sample_request,
)

from effective.budget import MeasuredBudget
from effective.cost import Usage
from effective.domain import AskLLM, CallTool
from effective.fork import measured_drive
from effective.handlers.base import artifact_id
from effective.keys import Key
from effective.ops import AppendLedgerRow, AwaitEvent, StoreArtifact

RUN = "run-a1"
MID = "msg-a1"
STEP_COST = 0.001

_ASSESSMENT = {
    "product": "Kettle",
    "amount": "4.50",
    "confidence": 0.40,  # below the review threshold -> routes to REVIEW (so the fork parks)
    "reasoning": "clear request",
}


def _request() -> RefundRequest:
    return sample_request(MID, body="Kettle arrived broken, refund $4.50")


class RefundDomain:
    """A metered double for the exemplar's two domain calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_metered(self, op: Any) -> tuple[Any, Usage]:
        match op:
            case CallTool(name="fetch_request"):
                self.calls.append("fetch_request")
                return _request(), Usage(cost=STEP_COST)
            case AskLLM():
                self.calls.append("assess_request")
                return Assessment.model_validate(_ASSESSMENT), Usage(cost=STEP_COST)
        raise AssertionError(f"unexpected domain op {op!r}")


class RecordingLedger:
    """A ledger that would notice being written to. It never should be."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []


def _budget(limit: float = 1.0) -> MeasuredBudget:
    return MeasuredBudget(overall=limit, run_id=RUN, on_exhaust="park")


def _drive(grants: dict[Key, object] | None = None, **kwargs):
    return measured_drive(
        lambda: process_refund(MID), _budget(), RefundDomain(), grants or {}, **kwargs
    )


def test_forking_the_production_workflow_drives_every_op_it_yields():
    """The headline: the exemplar yields ledger rows, an artifact, and an await, and the measured
    driver handles each rather than raising at the first non-Step op."""
    tail = _drive()
    kinds = [type(entry.op).__name__ for entry in tail.trace]
    assert "AppendLedgerRow" in kinds
    assert "StoreArtifact" in kinds
    assert tail.trace, "the fork produced no trace at all"


def test_a_fork_parks_at_the_workflows_await_rather_than_inventing_an_answer():
    """`process_refund` routes this request to review, so the fork reaches `await_event` with
    no delivered answer — and reports the park instead of guessing."""
    tail = _drive()
    assert tail.tripped_at == review_name(MID)
    assert tail.result is None


def test_a_delivered_answer_lets_the_fork_run_to_completion():
    approval = ApprovalEvent(decision="approve", actor="approver", rationale="ok")
    tail = _drive(grants={review_name(MID): approval})
    assert tail.tripped_at is None
    assert tail.result is not None


def test_a_fork_NEVER_appends_to_the_ledger():
    """The two-bookkeepers rule: a counterfactual that writes the canonical record would forge
    history for a run that never happened. The ledger rows are observed, not committed —
    `measured_drive` holds no ledger at all, and the ops that would write return as if they had."""
    approval = ApprovalEvent(decision="approve", actor="approver", rationale="ok")
    tail = _drive(grants={review_name(MID): approval})
    appends = [e for e in tail.trace if isinstance(e.op, AppendLedgerRow)]
    assert len(appends) >= 2  # the assessment + the review decision
    assert all(entry.result is None for entry in appends)  # exactly what a real append returns
    assert all(entry.usage == Usage() for entry in appends)  # and it costs nothing


def test_an_artifact_id_is_DERIVED_not_stored():
    """Content addressing pays off here: the fork returns the same id the real run would,
    without writing the blob."""
    tail = _drive()
    stores = [e for e in tail.trace if isinstance(e.op, StoreArtifact)]
    assert len(stores) == 1
    assert stores[0].result == artifact_id(stores[0].op)


def test_only_metered_steps_cost_anything():
    """The inspect-only ops fold zero usage, so a fork's dollars answer is about model calls —
    the number a grantor is actually deciding against."""
    approval = ApprovalEvent(decision="approve", actor="approver", rationale="ok")
    tail = _drive(grants={review_name(MID): approval})
    assert tail.usage.cost == pytest.approx(2 * STEP_COST)  # fetch_request + assess_request
    assert tail.live_usage.cost == pytest.approx(2 * STEP_COST)  # nothing was replayed


def test_prefix_is_step_indexed_like_the_bridge_exports_it():
    """A COUPLING pin, not a preference: `export_measured_prefix` filters non-Step checkpoints
    out of the durable prefix (`effective.checkpoints.NON_STEP`), so the prefix is Step-indexed
    and
    the driver must index it the same way. If either side starts counting the other's ops, a real
    forked run silently misaligns — a ledger row would consume the next Step's recorded result.

    Here: a one-entry prefix must line up with the FIRST STEP, even though a `StoreArtifact`
    is yielded before the second one."""
    from effective.fork import MeteredEntry

    domain = RefundDomain()
    request = _request()
    prefix = [
        MeteredEntry(
            key=Key.parse("step;tool:fetch_request"),
            op=None,
            result=request,
            usage=Usage(cost=STEP_COST),
        )
    ]
    tail = measured_drive(lambda: process_refund(MID), _budget(), domain, {}, recorded=prefix)
    assert domain.calls == ["assess_request"]  # fetch_request replayed FREE, not re-run
    assert tail.live_usage.cost == pytest.approx(STEP_COST)  # only the live step spent
    assert tail.usage.cost == pytest.approx(2 * STEP_COST)  # the meter still re-derives in full


def test_a_gather_in_a_fork_is_refused_loudly():
    """Guessing at concurrency inside a counterfactual would silently change what is being
    measured, so the refusal names the op."""
    from effective.api import gather, step

    def _concurrent():
        yield from gather([lambda: step("a", AskLLM(messages=[], response_schema=int))])

    with pytest.raises(TypeError, match="cannot interpret Gather"):
        measured_drive(_concurrent, _budget(), RefundDomain(), {})


def test_an_undelivered_await_parks_the_fork_without_touching_the_domain():
    domain = RefundDomain()
    tail = measured_drive(lambda: process_refund(MID), _budget(), domain, {})
    assert tail.tripped_at == review_name(MID)
    assert domain.calls == ["fetch_request", "assess_request"]  # and nothing after the park
    assert not isinstance(tail.trace[-1].op, AwaitEvent)  # the park itself is not recorded


def test_a_prefix_sleep_diverges_between_fork_at_and_measured_drive():
    """D7 (documented): an already-ELAPSED sleep in the recorded prefix replays
    fine under `fork_at` (which replays the prefix structurally, position included) but is refused
    by `measured_drive` — whose Step-only prefix dropped the sleep's position, so it re-meets the
    sleep 'live' and trips `ForkedSleep`. Latent (no current workflow sleeps) and the measured
    driver's own Step-only contract: fork a sleeping run with `fork_at`. This pins the documented
    divergence on BOTH drivers rather than silently leaving it uncovered."""
    from datetime import UTC, datetime

    from effective.api import ask_llm, await_event, sleep_until
    from effective.fork import MeteredEntry, OpIndex, fork_at
    from effective.handlers.recording import RecordingHandler, Suspended
    from effective.sandbox import ForkedSleep

    def prog():
        yield from ask_llm("first", [], str)
        yield from sleep_until(datetime(2026, 1, 1, tzinfo=UTC))
        return (yield from await_event("review:x", str))

    base = RecordingHandler(responses={"first": "a1"})
    parked = base.run(prog)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "review:x"
    trace = base.trace
    assert trace[1].key.stored().startswith("sleep:")  # the sleep is IN the recorded prefix

    dom = RefundDomain()  # unused on both paths (no tail Step; ForkedSleep before any live call)

    # fork_at replays the elapsed prefix sleep and completes past it:
    tail = fork_at(
        prog, trace, OpIndex(len(trace)), "reviewed", dom, pending_name=Key.parse("review:x")
    )
    assert tail.result == "reviewed"

    # measured_drive, given the Step-only prefix, refuses the same run:
    prefix = [MeteredEntry(key=Key.parse("step:first"), result="a1", usage=Usage())]
    with pytest.raises(ForkedSleep):
        measured_drive(prog, _budget(), dom, {}, recorded=prefix)


def test_the_trip_and_the_decode_agree_on_which_op_is_metered():
    """6.4. `decode_checkpoint` asked the shared `metered_call`; `trip_at` re-derived
    `isinstance(op.op, AskLLM)`. So the driver honored the placement when DECODING a prefix
    entry and a different one when TRIPPING on it — the F-2 defect one layer down from the one
    `trip_at` was extracted to fix.

    The reachable case is a meter carrying spend the CURRENT domain did not produce: fork a
    metered recording under a plain domain, and the old form tripped where the durable handler
    (which gates on `metered_call`) would not."""
    from effective.budget import MeasuredBudget
    from effective.cost import Contract, Usage
    from effective.domain import AskLLM
    from effective.fork import trip_at
    from effective.govern import BudgetRefused
    from effective.handlers.durable import metered_call
    from effective.ops import Step

    ask = Step(name="ask0", op=AskLLM(messages="m", response_schema=str))
    budget = MeasuredBudget(run_id="r1", overall=0.001, on_exhaust="fail")
    spent = Usage(cost=1.0)  # a prefix recorded by a metered run

    class Plain:  # not a `MeteredDomain`: no `run_metered`
        def run(self, op):
            return "x"

    plain = Plain()
    assert not metered_call(ask.op, Contract.V1, plain), "the shared predicate says: not metered"
    assert trip_at(ask, spent, budget, 0.0, 0, {}, plain) == (0.0, 0), "so the trip must not fire"

    # And it still fires for a domain the predicate DOES accept — the arm order matters.
    class Metered:
        def run(self, op):
            return "x"

        def run_metered(self, op):
            return "x", Usage(cost=0.5)

    metered = Metered()
    assert metered_call(ask.op, Contract.V1, metered)
    with pytest.raises(BudgetRefused):
        trip_at(ask, spent, budget, 0.0, 0, {}, metered)
