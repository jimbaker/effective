"""Phase-0 acceptance tests: no Absurd, no Postgres, no I/O.

Proves the design claims for the exemplar workflow:
- a recorded production run replays without re-calling the LLM/tool;
- a control-flow change makes replay verification fail loudly;
- a human approval resumes the suspended run and appends exactly one commitment;
- the generator and coroutine surfaces produce an identical op trace.
"""

from decimal import Decimal

import pytest
from _approval_domain import (
    ApprovalEvent,
    Assessment,
    RefundDecision,
    RefundRequest,
    assessed_id,
    committed_id,
    process_refund,
    process_refund_async,
    review_name,
    sample_request,
)

from effective import (
    RecordingHandler,
    ReplayHandler,
    ReplayMismatch,
    Suspended,
    ask_llm,
    call_tool,
)
from effective.handlers.base import content_digest

MID = "m1"


def _request() -> RefundRequest:
    return sample_request("m1", body="Widget, refund $42.00")


def responses_auto() -> dict:
    return {
        "tool:fetch_request": _request(),
        "assess_request": Assessment(product="Widget", amount=Decimal("42.00"), confidence=0.95),
    }


def responses_review() -> dict:
    # over the amount threshold -> routes to review -> awaits an event
    return {
        "tool:fetch_request": _request(),
        "assess_request": Assessment(product="Console", amount=Decimal("500.00"), confidence=0.95),
    }


# StoreArtifact keys are content-addressed (artifact:{kind}/{subtype},{digest}); compute the
# digest from the value the workflow actually stores (request.body) so this stays
# correct if the fixture changes.
# `type/subtype` is a PATH coordinate and the digest is a second coordinate, algorithm-
# prefixed so its kind is legible; the byte table pins both.
_ARTIFACT_KEY = f"artifact:text/plain,sha256-{content_digest(_request().body)}"

AUTO_TRACE = [
    "step;tool:fetch_request",
    _ARTIFACT_KEY,
    "step:assess_request",
    f"ledger;{assessed_id(MID).stored()}",
    f"ledger;{committed_id(MID).stored()}",
]


# --- zero-I/O happy path ----------------------------------------------------


def test_auto_path_runs_with_zero_io():
    h = RecordingHandler(responses_auto())
    result = h.run(lambda: process_refund("m1"))

    assert isinstance(result, RefundDecision)
    assert result.status == "committed"
    assert [r.kind for r in h.ledger] == ["assessment", "commitment"]
    assert len(h.artifacts) == 1
    assert [e.key.stored() for e in h.trace] == AUTO_TRACE


# --- replay -----------------------------------------------------------------


def test_replay_reproduces_without_responses():
    rec = RecordingHandler(responses_auto())
    first = rec.run(lambda: process_refund("m1"))

    # ReplayHandler gets *no* canned responses — only the recorded trace.
    replay = ReplayHandler(rec.trace)
    second = replay.run(lambda: process_refund("m1"))

    assert second == first


def test_replay_detects_control_flow_change():
    rec = RecordingHandler(responses_auto())
    rec.run(lambda: process_refund("m1"))

    def altered(message_id: str):
        # diverges at position 1: skips store_artifact
        yield from call_tool("fetch_request", {"id": message_id}, RefundRequest)
        assessment = yield from ask_llm("assess_request", [], Assessment)
        return RefundDecision(status="committed", assessment=assessment)

    with pytest.raises(ReplayMismatch):
        ReplayHandler(rec.trace).run(lambda: altered("m1"))


# --- HITL suspend / resume --------------------------------------------------


def test_suspend_then_resume_approve():
    h = RecordingHandler(responses_review())  # no review answer delivered -> parks
    parked = h.run(lambda: process_refund(MID))

    assert isinstance(parked, Suspended)
    assert parked.awaiting == review_name(MID)
    assert [r.kind for r in h.ledger] == ["assessment"]  # nothing committed yet

    result = parked.resume(ApprovalEvent(decision="approve", actor="approver"))
    assert isinstance(result, RefundDecision)
    assert result.status == "committed"
    # the assessment made *before* the suspension survived in the generator frame
    assert result.assessment.product == "Console"
    assert [r.kind for r in h.ledger] == ["assessment", "review", "commitment"]


def test_suspend_then_resume_reject():
    h = RecordingHandler(responses_review())
    parked = h.run(lambda: process_refund("m1"))
    assert isinstance(parked, Suspended)
    result = parked.resume(
        ApprovalEvent(decision="reject", actor="approver", rationale="duplicate")
    )
    assert isinstance(result, RefundDecision)

    assert result.status == "rejected"
    assert result.approval is not None
    assert result.approval.decision == "reject"
    assert [r.kind for r in h.ledger] == ["assessment", "review"]  # no commitment


# --- generator vs coroutine surface parity -----------------------------------


def test_surface_parity_auto():
    ha = RecordingHandler(responses_auto())
    ra = ha.run(lambda: process_refund("m1"))
    hc = RecordingHandler(responses_auto())
    # Option-C coroutine surface — run() is surface-agnostic at runtime; typed for Effect.
    rc = hc.run(lambda: process_refund_async("m1"))  # ty: ignore[invalid-argument-type]

    assert ra == rc
    assert [e.key.stored() for e in ha.trace] == [e.key.stored() for e in hc.trace] == AUTO_TRACE


def test_surface_parity_suspend_resume():
    ha = RecordingHandler(responses_review())
    pa = ha.run(lambda: process_refund("m1"))
    hc = RecordingHandler(responses_review())
    # Option-C coroutine surface — run() is surface-agnostic at runtime; typed for Effect.
    pc = hc.run(lambda: process_refund_async("m1"))  # ty: ignore[invalid-argument-type]

    assert isinstance(pa, Suspended)
    assert isinstance(pc, Suspended)
    fa = pa.resume(ApprovalEvent(decision="approve", actor="approver"))
    fc = pc.resume(ApprovalEvent(decision="approve", actor="approver"))

    assert fa == fc
    assert [e.key.stored() for e in ha.trace] == [e.key.stored() for e in hc.trace]
