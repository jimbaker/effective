"""The `at` index convention: `to_step_index` and its boundary invariant.

`at` (the caller-facing fork point) is ALL-OPS; the bridge/meter prefix is STEP-ONLY.
`to_step_index` is the one named conversion between them. The hazard it guards: the two
coordinates silently disagree by the number of non-Step ops in the prefix, so a human `at`
passed where a Step count is wanted misaligns the replay: green, wrong.

A proof pin over an IN-PROCESS trace, because the durable record cannot host it (SQLite leaves
awaits/sleeps unpositioned, Absurd needs the forbidden `updated_at` order): `to_step_index`
(which counts `Step` ops) must AGREE with the bridge's own Step filter (`is_step_checkpoint`)
at every prefix length, two independent projections that must not drift. Sibling of
`test_measured_fork_ops.py::test_prefix_is_step_indexed_like_the_bridge_exports_it`, which pins
the other end of the same boundary (the measured driver indexes Step-only).
"""

from effective.api import append_ledger, ask_llm, await_event, call_tool, scoped, store_artifact
from effective.checkpoints import is_step_checkpoint
from effective.fork import OpIndex, to_step_index
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Key, compose_key
from effective.ops import LedgerRow


def _shaped():
    """The exemplar's op shape: a Step, a non-Step, a Step, a non-Step, the fork await, a non-Step.
    (An approval workflow in miniature.)"""
    yield from call_tool("fetch_email", {}, dict)  # all-ops 0 · Step(CallTool)
    yield from store_artifact("raw", "text/plain")  # all-ops 1 · StoreArtifact  (non-Step)
    yield from ask_llm("extract", [], dict)  # all-ops 2 · Step(AskLLM)
    yield from append_ledger(
        LedgerRow(event_id=Key.parse("extracted:m1"), kind="ex")
    )  # all-ops 3 · ledger
    decision = yield from await_event("review:m1", dict)  # all-ops 4 · AwaitEvent  ← fork here
    yield from append_ledger(
        LedgerRow(event_id=Key.parse("reviewed:m1"), kind="reviewed")
    )  # all-ops 5
    return decision


def _recorded_trace() -> list:
    base = RecordingHandler(responses={"tool:fetch_email": {"f": "a"}, "extract": {"amt": "5"}})
    parked = base.run(_shaped)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "review:m1"
    parked.resume({"decision": "approve"})  # deliver the await -> records event;review:m1 + tail
    return base.trace


def test_the_recorded_trace_has_the_expected_all_ops_shape():
    keys = [e.key.stored() for e in _recorded_trace()]
    assert keys == [
        "step;tool:fetch_email",
        keys[1],  # artifact:<content hash> — id is content-addressed, not fixed here
        "step:extract",
        "ledger;extracted:m1",
        "event;review:m1",
        "ledger;reviewed:m1",
    ]
    assert keys[1].startswith("artifact:")  # the non-Step op between the two Steps


def test_to_step_index_maps_the_all_ops_fork_point_to_the_step_prefix_length():
    trace = _recorded_trace()
    at = OpIndex(4)  # the review await — the human-meaningful fork point
    assert to_step_index(trace, at) == 2  # only fetch_email + extract are Steps before it
    assert to_step_index(trace, at) != at  # the two coordinates genuinely differ (the whole point)


def _scoped_shaped():
    """The same shape with the non-Step ops inside a `scoped(...)`, so their keys carry a frame."""
    yield from call_tool("fetch_email", {}, dict)

    def body():
        yield from store_artifact("raw", "text/plain")
        yield from append_ledger(LedgerRow(event_id=Key.parse("extracted:m1"), kind="ex"))
        return (yield from ask_llm("extract", [], dict))

    yield from scoped(compose_key(t"rec:{0}"), body)
    yield from append_ledger(LedgerRow(event_id=Key.parse("reviewed:m1"), kind="reviewed"))


def _scoped_trace() -> list:
    handler = RecordingHandler(
        responses={"tool:fetch_email": {"f": "a"}, "rec:0;extract": {"amt": "5"}}
    )
    handler.run(_scoped_shaped)
    return handler.trace


def test_to_step_index_agrees_with_the_bridge_step_filter_at_every_boundary():
    """The invariant: `to_step_index` (counts `Step` ops) equals the bridge's Step-only export
    length — for EVERY prefix, not just the fork point. Two independent projections; if either
    started counting the other's ops, a fork would misalign.

    Run over a FRAMED trace too. Without it the pin cannot see a scoped `rec:0;ledger;…` key, and
    that blindness is why a bare `startswith` survived in one bridge for four days."""
    for trace in (_recorded_trace(), _scoped_trace()):
        for at in range(len(trace) + 1):
            bridge_kept = sum(1 for e in trace[:at] if is_step_checkpoint(e.key.stored()))
            assert to_step_index(trace, OpIndex(at)) == bridge_kept, f"disagree at {at}"


def test_a_step_name_that_merely_starts_like_an_arm_tag_is_still_a_step():
    """The attack: name a step `monitoring:daily` and see whether the Step filter believes it.

    Every arm marker is a TAG — it ends in the delimiter — so the prefix test is exact by
    construction. A marker without its colon matches any name that merely begins with the same
    letters, which is a step the bridges would drop from a Step-only projection and a measured
    prefix would index short.

    `monitoring:` is the reachable case: `op_key` reserves `monitor:`, so `monitoring:daily` is a
    name an author may legally choose."""

    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import Step

    legal = "monitoring:daily"
    assert (
        op_key(Step(name=legal, op=CallTool(name="x", result_schema=dict))).stored()
        == f"step;{legal}"
    )
    assert is_step_checkpoint(legal), "a legal step name was read as a substrate arm"

    # …while the arm itself is not a Step. The author MAY take that name, because the arm makes
    # the regions disjoint by CONSTRUCTION rather than by a list somebody keeps complete: an
    # author's `monitor:0` keys as `step;monitor:0` and cannot be the walk's `monitor:0`.
    assert not is_step_checkpoint("monitor:0")
    authored = op_key(Step(name="monitor:0", op=CallTool(name="x", result_schema=dict)))
    assert authored.stored() == "step;monitor:0"
    assert authored.stored() != "monitor:0"
