"""Counterfactual safety: the shared inspect-only policy + the dry-run domain.

No op-layer answers `AppendLedgerRow`/`StoreArtifact` without forwarding: that collapses the
durable checkpoint sequence, destroying the parity a fork's diff needs and un-persisting the
artifact. The durable no-commit seam is `DurableHandler(ledger=None)`, so what is tested here is
the **shared policy** (so the in-process drivers cannot drift) and the **dry-run domain** (the
half nothing else covers).

These tests assert the **observable contract** (canonical ledger untouched, world untouched,
workflow-visible values identical) rather than per-op decisions, which legitimately differ
between an in-process driver and the durable one. Asserting identical decisions across drivers
would pin the in-process behavior onto the durable driver.
"""

from decimal import Decimal

import pytest

from effective.cost import Usage
from effective.domain import AskLLM, CallTool
from effective.handlers.base import artifact_id
from effective.keys import Key
from effective.ops import AppendLedgerRow, AwaitEvent, Gather, LedgerRow, SleepUntil, StoreArtifact
from effective.sandbox import (
    DryRun,
    ForkedSleep,
    Observed,
    Unanswered,
    WorldMutation,
    inspect_only,
)

ROW = LedgerRow(event_id=Key.parse("e1"), kind="commitment")


# --- the shared policy ---------------------------------------------------------------------


def test_a_ledger_row_is_observed_and_never_appended():
    """Two bookkeepers: a fork that appended would forge history for a run that never happened.
    It returns `None` — exactly what a real append returns — so control flow is unchanged."""
    assert inspect_only(AppendLedgerRow(row=ROW), {}) == Observed(None)


def test_an_artifact_id_is_derived_from_content_not_stored():
    op = StoreArtifact(value="raw bytes", content_type="text/plain")
    assert inspect_only(op, {}) == Observed(artifact_id(op))


def test_a_sleep_is_refused_not_skipped():
    """A counterfactual explores a decision, not a clock: a durable sleep inside one is a category
    error, so it is refused loudly rather than silently skipped. Skipping would
    report a marginal for timing that never happened; parking is incoherent under `DryRun`."""
    from datetime import UTC, datetime

    with pytest.raises(ForkedSleep, match="not a clock"):
        inspect_only(SleepUntil(when=datetime(2030, 1, 1, tzinfo=UTC)), {})


def test_a_delivered_answer_is_returned_and_an_undelivered_one_parks():
    op = AwaitEvent(name=Key.parse("review:m1"), schema=dict)
    assert inspect_only(op, {Key.parse("review:m1"): {"decision": "approve"}}) == Observed(
        {"decision": "approve"}
    )
    assert inspect_only(op, {}) == Unanswered(Key.parse("review:m1"))


def test_an_op_a_counterfactual_cannot_interpret_is_refused_loudly():
    """Guessing at concurrency would silently change what is being measured."""
    with pytest.raises(TypeError, match="cannot interpret Gather"):
        inspect_only(Gather(branches=()), {})


def test_the_policy_is_pure_and_total_over_its_op_set():
    """Same op, same answers, same inspection — a fork's safety cannot depend on when it ran."""
    op = StoreArtifact(value="v", content_type="text/plain")
    assert inspect_only(op, {}) == inspect_only(op, {})


# --- the dry-run domain --------------------------------------------------------------------


class _World:
    """A domain that records everything it was actually asked to do."""

    def __init__(self) -> None:
        self.did: list[str] = []

    def run(self, op):
        self.did.append(getattr(op, "name", type(op).__name__))
        return "real"

    def run_metered(self, op):
        self.did.append(getattr(op, "name", type(op).__name__))
        return "real", Usage(cost=0.01)


def _tool(name: str) -> CallTool:
    return CallTool(name=name, args={}, result_schema=str)


def test_an_unlisted_tool_is_REFUSED_not_mocked():
    """A fabricated result would let the fork report a marginal computed from an answer nobody
    produced. The honest outcome is that the counterfactual cannot be run as posed."""
    world = _World()
    with pytest.raises(WorldMutation, match="send_email"):
        DryRun(world).run(_tool("send_email"))
    assert world.did == []  # and the world was not touched on the way to failing


def test_the_refusal_names_the_tool_and_the_fix():
    with pytest.raises(WorldMutation) as ei:
        DryRun(_World(), allow=frozenset({"fetch_email"})).run(_tool("send_email"))
    assert "send_email" in str(ei.value)
    assert "fetch_email" in str(ei.value)  # what IS allowed
    assert "read-only" in str(ei.value)


def test_an_allow_listed_read_only_tool_forwards():
    world = _World()
    assert DryRun(world, allow=frozenset({"fetch_email"})).run(_tool("fetch_email")) == "real"
    assert world.did == ["fetch_email"]


def test_a_canned_tool_answers_WITHOUT_being_called():
    world = _World()
    sandbox = DryRun(world, canned={"fetch_email": "canned"})
    assert sandbox.run(_tool("fetch_email")) == "canned"
    assert world.did == []


def test_a_canned_answer_costs_nothing_on_the_metered_path():
    """The meter must not claim spend for a call that never happened."""
    result, usage = DryRun(_World(), canned={"t": "c"}).run_metered(_tool("t"))
    assert result == "c"
    assert usage == Usage()


def test_an_askllm_always_forwards():
    """A model call has no world effect beyond cost, and a fork that never asks the model
    measures nothing. Cost is the budget's job, not the sandbox's."""
    world = _World()
    result, usage = DryRun(world).run_metered(AskLLM(messages=[], response_schema=str))
    assert result == "real"
    assert usage.cost == 0.01
    assert world.did == ["AskLLM"]


def test_the_metered_arm_is_PRESERVED_not_hidden():
    """The `serve` lesson: a `.run`-only wrapper would silently disable the measured cap, since
    `metered_call` probes for `run_metered` by `getattr`."""
    sandbox = DryRun(_World())
    assert callable(getattr(sandbox, "run_metered", None))
    assert callable(getattr(sandbox, "run", None))


def test_the_sandbox_records_which_tools_were_attempted():
    """An attempted mutation is part of the record — the fork's answer should be able to say
    'this counterfactual would have emailed the vendor'."""
    sandbox = DryRun(_World(), allow=frozenset({"fetch_email"}))
    sandbox.run(_tool("fetch_email"))
    with pytest.raises(WorldMutation):
        sandbox.run(_tool("send_email"))
    assert sandbox.calls == ["fetch_email", "send_email"]


# --- the observable contract, end to end ---------------------------------------------------


def test_the_observable_contract_a_fork_of_a_real_workflow_writes_nothing():
    """The contract the two halves are jointly held to, on the production exemplar: the
    canonical ledger is untouched, the world is untouched, and the workflow still completes with
    the values it would have seen."""
    from _approval_domain import (
        ApprovalEvent,
        Assessment,
        process_refund,
        review_name,
        sample_request,
    )

    from effective.budget import MeasuredBudget
    from effective.fork import measured_drive

    # One message id, asked for by name where the grant has to match the workflow's await.
    MID = "m1"
    ledger_appends: list[dict] = []

    class _Base:
        def run_metered(self, op):
            match op:
                case CallTool(name="fetch_request"):
                    return sample_request("m1", body="Kettle, refund $4.50"), Usage(cost=0.001)
                case _:
                    return (
                        Assessment(product="Kettle", amount=Decimal("4.50"), confidence=0.4),
                        Usage(cost=0.001),
                    )

    sandbox = DryRun(_Base(), allow=frozenset({"fetch_request"}))
    tail = measured_drive(
        lambda: process_refund(MID),
        MeasuredBudget(overall=1.0, run_id="r1", on_exhaust="park"),
        sandbox,
        {review_name(MID): ApprovalEvent(decision="approve", actor="approver", rationale="ok")},
    )

    assert tail.result is not None  # the workflow completed...
    assert ledger_appends == []  # ...the canonical ledger is untouched...
    assert sandbox.calls == ["fetch_request"]  # ...and only the read-only tool was called
