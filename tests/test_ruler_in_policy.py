"""A `govern` policy reads who ruled the answer it asked for, on both engines.

The ruler is the principal `parked.answer` names, carried from the stored ruling to the policy.
A policy that refuses an approval by the op's proposer is the separation of duties the ruler
makes possible.
"""

from typing import Any
from uuid import uuid4

import pytest
from _authority import ruled
from _conformance import Fault
from _steps import placed_step

from effective.api import step
from effective.domain import CallTool
from effective.govern import Ask as GateAsk
from effective.govern import (
    GateState,
    Proceed,
    Resolution,
    govern,
    park,
    refuse,
    ruler_of,
    rulers_for,
    run_answers_for,
    run_rulers_for,
)
from effective.keys import Segment
from effective.ops import Step, WorkflowOp
from effective.parked import answer
from effective.ruling import load_answer, ruling_schema, split_ruling

PROPOSER = Segment("proposer")
REVIEWER = Segment("reviewer")
POLICY = "separate"


class _Charges:
    def __init__(self) -> None:
        self.charged: list[int] = []

    def run(self, op: CallTool) -> str:
        self.charged.append(op.args["n"])
        return "charged"


def _charge(_run_id: str):
    return (yield from step("charge", CallTool(name="charge", args={"n": 3}, result_schema=str)))


def separated(op: WorkflowOp, state: GateState) -> Any:
    """Parks a charge for approval, and refuses an approval its proposer ruled."""
    if not (isinstance(op, Step) and op.name == "charge"):
        return Proceed()
    match ruler_of(state, POLICY):
        case None:
            return park(GateAsk(POLICY, "approve the charge?"))
        case ruler if ruler == PROPOSER:
            return refuse("the proposer may not approve its own charge", policy=POLICY)
        case _:
            return Proceed()


def _parked(backend):
    token = "t" + uuid4().hex[:8]
    charges = _Charges()
    gate = govern(separated, gate="spend", run_id=f"run-{token}")
    backend.register(f"separate-{token}", _charge, charges, Fault(), (gate,))
    task_id = backend.spawn(f"separate-{token}", f"run-{token}")
    backend.run_until_result(task_id)
    (parked,) = backend.parked(task_id)
    return task_id, parked, charges


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("by", "charged"), [(PROPOSER, []), (REVIEWER, [3])], ids=["proposer", "reviewer"]
)
def test_a_policy_refuses_an_approval_by_its_proposer(backend, by, charged):
    task_id, parked, charges = _parked(backend)
    answer(backend.engine, parked, {"answers": {POLICY: {"decision": "approve"}}}, by=by)
    snapshot = backend.run_until_result(task_id)
    assert charges.charged == charged
    if charged:
        assert snapshot.state == "completed", snapshot
    else:
        assert "the proposer may not approve" in str(snapshot.failure), snapshot


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("payload", "charged", "state", "failure"),
    [
        ({POLICY: {"decision": "approve"}}, [3], "completed", None),
        ({"answers": {POLICY: {"decision": "approve"}}, "note": "x"}, [], "failed", "['note']"),
    ],
    ids=["bare", "wrapper-beside-a-key"],
)
def test_a_durable_park_reads_a_payload_as_the_in_process_gate_does(
    backend, payload, charged, state, failure
):
    task_id, parked, charges = _parked(backend)
    answer(backend.engine, parked, payload, by=REVIEWER)
    snapshot = backend.run_until_result(task_id)
    assert (charges.charged, snapshot.state) == (charged, state), snapshot
    assert (snapshot.failure is None) if failure is None else failure in str(snapshot.failure)


def test_each_answer_keeps_its_ruler_for_the_op_and_the_run():
    state = GateState(run_id="r", gate="g", op_key=placed_step("charge", "tool"))
    state = state.absorb(Resolution(answers={"a": 1, "b": 2}), by=PROPOSER)
    state = state.absorb(Resolution(answers={"a": 3}), by=REVIEWER)
    state = state.absorb(Resolution(answers={"b": 4}))
    assert rulers_for(state, "a") == run_rulers_for(state, "a") == (PROPOSER, REVIEWER)
    assert rulers_for(state, "b") == (PROPOSER, None)
    assert (ruler_of(state, "a"), ruler_of(state, "b"), ruler_of(state, "c")) == (
        REVIEWER,
        None,
        None,
    )


def test_an_answer_held_with_no_ruler_reads_as_unruled():
    carried = GateState(
        run_id="r", gate="g", op_key=placed_step("x", "tool"), answers={"p": ("a0",)}
    )
    state = carried.absorb(Resolution(answers={"p": "a1"}), by=REVIEWER)
    assert rulers_for(state, "p") == (None, REVIEWER)


def _twice(_run_id: str):
    yield from step("charge-1", CallTool(name="charge", args={"n": 1}, result_schema=str))
    yield from step("charge-2", CallTool(name="charge", args={"n": 2}, result_schema=str))


def approved_once_for_the_run(op: WorkflowOp, state: GateState) -> Any:
    """Parks the run's first charge for one approval, which the second charge rides on unless
    the proposer gave it."""
    if not (isinstance(op, Step) and isinstance(op.name, str) and op.name.startswith("charge")):
        return Proceed()
    if op.name == "charge-2" and PROPOSER in run_rulers_for(state, POLICY):
        return refuse("the proposer may not approve the run's charges", policy=POLICY)
    if run_answers_for(state, POLICY):
        return Proceed()
    return park(GateAsk(POLICY, "approve the run's charges?"))


@pytest.mark.adversarial
@pytest.mark.parametrize(("by", "charged"), [(PROPOSER, [1]), (REVIEWER, [1, 2])], ids=["p", "r"])
def test_an_approval_carried_to_a_later_op_keeps_its_ruler(backend, by, charged):
    token = "t" + uuid4().hex[:8]
    charges = _Charges()
    gate = govern(approved_once_for_the_run, gate="spend", run_id=f"run-{token}")
    backend.register(f"twice-{token}", _twice, charges, Fault(), (gate,))
    task_id = backend.spawn(f"twice-{token}", f"run-{token}")
    backend.run_until_result(task_id)
    (parked,) = backend.parked(task_id)
    answer(backend.engine, parked, {"answers": {POLICY: {"decision": "approve"}}}, by=by)
    snapshot = backend.run_until_result(task_id)
    assert charges.charged == charged
    if by == PROPOSER:
        assert "the proposer may not approve" in str(snapshot.failure), snapshot


def test_a_ruling_schema_on_a_plain_name_is_refused():
    with pytest.raises(ValueError, match="no authority park"):
        load_answer(
            "review:m1", ruling_schema(dict), {"__ruling__": {"by": "ceo"}}, lambda s, v: v
        )


@pytest.mark.parametrize(
    ("delivered", "split"),
    [({POLICY: "yes"}, (None, {POLICY: "yes"})), (ruled("yes", by="r"), (Segment("r"), "yes"))],
    ids=["unruled", "ruled"],
)
def test_a_delivered_answer_splits_into_its_ruler_and_answer(delivered, split):
    assert split_ruling(delivered) == split


def test_a_ruling_reads_an_optional_field_it_does_not_know():
    name = "approve:charge"
    stored = {"__ruling__": {"by": "reviewer", "answer": {"ok": True}, "note": "later field"}}
    loaded = load_answer(name, ruling_schema(dict), stored, lambda _schema, raw: raw)
    assert loaded == ruled({"ok": True}, by="reviewer")
