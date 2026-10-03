"""fork-as-replay value-of-information probe: the driver's pins.

Covers the two `at` conventions (recorded op vs pending park), the fork-point guard, the
free-prefix / poison proof, the `probe ≪ grant` shape, and `live_drive` folding a real
`MeteredInterpreter`. Infra-free (recording core).
"""

import pytest

from effective.api import step
from effective.budget import Grant, depth_grant_name
from effective.combinators import Answered, Deeper, descend
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.fork import OpIndex, fork_at, live_drive, replay_prefix
from effective.handlers.base import TraceEntry, artifact_id
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayMismatch
from effective.keys import Key, Segment, compose_key
from effective.ops import AppendLedgerRow, LedgerRow, Step, StoreArtifact

PER_CALL = Usage(prompt_tokens=10, completion_tokens=20, cost=0.001)


class SequencedDomain:
    """A metered domain returning canned outputs in call order, each priced `PER_CALL`.
    The output list covers the TAIL only — a wrongly re-executed prefix would over-run it
    (the poison proof) — and `i` counts the calls actually made."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.i = 0

    def run_metered(self, op: object) -> tuple[str, Usage]:
        out = self.outputs[self.i]
        self.i += 1
        return out, PER_CALL


def _judge(ctx, level):
    raw = yield from step("judge", CallTool(name="op", args={}, result_schema=str))
    return Answered(raw) if level.final else Deeper(raw)


def _program():
    return (yield from descend("ctx", _judge, budget=1, run_id="r"))


def _parked_base() -> RecordingHandler:
    """A base run parked at grant:d1 (prefix = ['d:0;judge']); the park is NOT recorded."""
    base = RecordingHandler(responses={"d:0;judge": "v0"})
    parked = base.run(_program)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"{depth_grant_name('r', depth=1, generation=0).stored()}"
    assert [e.key.stored() for e in base.trace] == ["d:0;step:judge"]
    return base


def test_fork_at_recorded_park_two_arms_diverge_off_a_free_prefix():
    # A COMPLETED base lineage: parks at the d1 grant, resume records the event + d1/judge.
    base = RecordingHandler(responses={"d:0;judge": "v0", "d:1;judge": "v1"})
    parked = base.run(_program)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"{depth_grant_name('r', depth=1, generation=0).stored()}"
    parked.resume(Grant(add_depth=0))  # base: stop at d1
    trace = base.trace
    assert [e.key.stored() for e in trace] == [
        "d:0;step:judge",
        f"event;{depth_grant_name('r', depth=1, generation=0).stored()}",
        "d:1;step:judge",
    ]
    at = OpIndex(
        [e.key.stored() for e in trace].index(
            f"event;{depth_grant_name('r', depth=1, generation=0).stored()}"
        )
    )

    stop_dom, grant_dom = SequencedDomain(["stop-d1"]), SequencedDomain(["g1", "g2"])
    stop = fork_at(_program, trace, at, Grant(add_depth=0), stop_dom)
    grant = fork_at(
        _program,
        trace,
        at,
        Grant(add_depth=1),
        grant_dom,
        grants={depth_grant_name("r", depth=2, generation=0): Grant(add_depth=0)},
    )

    assert stop.result == "stop-d1"
    assert grant.result == "g2"
    # tail traces are the divergence only; the shared prefix (d0) is in NEITHER
    assert [e.key.stored() for e in stop.trace] == ["d:1;step:judge"]
    assert [e.key.stored() for e in grant.trace] == [
        "d:1;step:judge",
        f"event;{depth_grant_name('r', depth=2, generation=0).stored()}",
        "d:2;step:judge",
    ]
    # poison / free-prefix: each domain ran ONLY its tail's Step calls, never d0
    assert stop_dom.i == 1
    assert grant_dom.i == 2
    assert stop.usage.cost == PER_CALL.cost
    assert grant.usage.cost == 2 * PER_CALL.cost


def test_fork_at_pending_park_the_actual_voi_case():
    # A parked-not-resumed run: at == len(trace); validate against the pending name.
    trace = _parked_base().trace
    grant = fork_at(
        _program,
        trace,
        OpIndex(len(trace)),
        Grant(add_depth=0),
        SequencedDomain(["final-d1"]),
        pending_name=depth_grant_name("r", depth=1, generation=0),
    )
    assert grant.result == "final-d1"
    assert [e.key.stored() for e in grant.trace] == ["d:1;step:judge"]


def test_fork_point_guard_bites_on_a_wrong_pending_name():
    trace = _parked_base().trace
    with pytest.raises(ReplayMismatch):
        fork_at(
            _program,
            trace,
            OpIndex(len(trace)),
            Grant(add_depth=0),
            SequencedDomain(["x"]),
            pending_name=Key.parse("not-the-park"),
        )


def test_pending_park_fork_requires_a_pending_name():
    trace = _parked_base().trace
    with pytest.raises(ValueError, match="pending_name"):
        fork_at(
            _program,
            trace,
            OpIndex(len(trace)),
            Grant(add_depth=0),
            SequencedDomain(["x"]),
        )


def test_prefix_replay_guard_is_non_vacuous():
    bogus = [
        TraceEntry(Key.parse("wrong:key"), Step("x", CallTool(name="op", result_schema=str)), "v0")
    ]
    with pytest.raises(ReplayMismatch, match="prefix divergence"):
        replay_prefix(_program, bogus, at=OpIndex(1))


def test_probe_one_level_is_cheaper_than_the_grant_it_informs():
    # The VOI shape: probe ONE level to inform a grant of N.
    trace = _parked_base().trace
    probe = fork_at(
        _program,
        trace,
        OpIndex(len(trace)),
        Grant(add_depth=1),
        SequencedDomain(["probe-d1"]),
        pending_name=depth_grant_name("r", depth=1, generation=0),
    )
    assert probe.parked_at == depth_grant_name(
        "r", depth=2, generation=0
    )  # re-parked: the cheap probe
    assert probe.result is None
    assert probe.usage.cost == PER_CALL.cost  # exactly one level

    grant = fork_at(
        _program,
        trace,
        OpIndex(len(trace)),
        Grant(add_depth=3),
        SequencedDomain(["g1", "g2", "g3", "g4"]),
        pending_name=depth_grant_name("r", depth=1, generation=0),
        grants={depth_grant_name("r", depth=4, generation=0): Grant(add_depth=0)},
    )
    assert grant.result == "g4"
    assert grant.usage.cost == 4 * PER_CALL.cost
    assert probe.usage.cost < grant.usage.cost  # probe ≪ grant


def test_live_drive_folds_a_real_metered_interpreter():
    # The Option-B seed: the tail runs under the production MeteredInterpreter and its usage
    # folds into the ForkTail — with the free prefix excluded.
    def ask_judge(ctx, level):
        raw = yield from step("judge", AskLLM(messages=[], response_schema=str))
        return Answered(raw) if level.final else Deeper(raw)

    def prog():
        return (yield from descend("ctx", ask_judge, budget=1, run_id="r"))

    base = RecordingHandler(responses={"d:0;judge": "v0"})
    assert isinstance(base.run(prog), Suspended)
    trace = base.trace

    calls: list[AskLLM] = []

    def fake_llm(op: AskLLM) -> tuple[str, Usage]:
        calls.append(op)
        return f"a{len(calls)}", Usage(prompt_tokens=5, completion_tokens=7, cost=0.01)

    domain = MeteredInterpreter(llm=fake_llm, tools=lambda op: None)
    tail = fork_at(
        prog,
        trace,
        OpIndex(len(trace)),
        Grant(add_depth=0),
        domain,
        pending_name=depth_grant_name("r", depth=1, generation=0),
    )
    assert tail.result == "a1"  # one tail call (d1 final); d0 was free
    assert len(calls) == 1
    assert tail.usage.cost == 0.01
    assert tail.usage.prompt_tokens == 5
    assert domain.meter.cost == 0.01  # the interpreter's own meter agrees


def test_live_drive_refuses_an_op_a_counterfactual_cannot_interpret():
    """Every non-Step op goes through the ONE shared inspect-only policy, so `live_drive` and
    `measured_drive` cannot disagree about what a counterfactual may touch. What is refused is
    refused for a reason, and says it."""
    gen = (x for x in [object()])  # a generator yielding a bare non-op
    with pytest.raises(TypeError, match="a counterfactual cannot interpret"):
        live_drive(gen, None, SequencedDomain([]), {})


def test_live_drive_observes_a_ledger_row_and_an_artifact():
    """`live_drive` drives a real workflow, not only Step/AwaitEvent. It observes a ledger row
    (not appended) and an artifact (id derived, not stored): the same rules `measured_drive`
    follows, from the same definition."""
    row = LedgerRow(event_id=Key.parse("e1"), kind="commitment")
    artifact = StoreArtifact(value="raw", content_type="text/plain")
    ops: list[object] = [AppendLedgerRow(row=row), artifact]
    gen = (x for x in ops)
    tail = live_drive(gen, None, SequencedDomain([]), {})

    assert [type(e.op).__name__ for e in tail.trace] == ["AppendLedgerRow", "StoreArtifact"]
    assert tail.trace[0].result is None  # exactly what a real append returns
    assert tail.trace[1].result == artifact_id(artifact)  # derived from content, not stored


def test_forking_at_a_recorded_sleep_is_refused_with_forkedsleep():
    """Forking AT a recorded sleep is refused loudly by the fork-point op-class guard. The tail
    guard alone does not see it, and the sleep would be silently "answered" with the
    substitute."""
    from datetime import UTC, datetime

    from effective.api import ask_llm, sleep_until
    from effective.sandbox import ForkedSleep

    def prog():
        yield from ask_llm("first", [], str)
        yield from sleep_until(datetime(2026, 1, 1, tzinfo=UTC))
        return (yield from ask_llm("second", [], str))

    base = RecordingHandler(responses={"first": "a1", "second": "a2"})
    assert base.run(prog) == "a2"
    trace = base.trace
    assert trace[1].key.stored().startswith("sleep:")  # the fork point IS a recorded sleep
    with pytest.raises(ForkedSleep):
        fork_at(prog, trace, OpIndex(1), "substitute", SequencedDomain([]))


def test_forking_at_a_write_op_is_refused():
    """A fork substitutes a DECISION, not a write. Forking at an `AppendLedgerRow` fork
    point is refused rather than fabricating a counterfactual for an op with no decision."""
    from effective.api import append_ledger, ask_llm
    from effective.sandbox import ForkPointRefused

    def prog():
        yield from ask_llm("first", [], str)
        yield from append_ledger(LedgerRow(event_id=Key.parse("e1"), kind="k"))
        return (yield from ask_llm("second", [], str))

    base = RecordingHandler(responses={"first": "a1", "second": "a2"})
    base.run(prog)
    trace = base.trace
    assert trace[1].key.stored().startswith("ledger;")
    with pytest.raises(ForkPointRefused, match="not a write"):
        fork_at(prog, trace, OpIndex(1), "x", SequencedDomain([]))


def test_forking_at_a_gather_reaches_the_typed_refusal_not_op_keys_valueerror():
    """`op_key(Gather)` raises `ValueError`, so the class guard runs before the key check: a
    `Gather` fork point gets the typed refusal, never a bare `ValueError`."""
    from effective.api import gather
    from effective.domain import AskLLM

    def prog():
        return (yield from gather([lambda: step("a", AskLLM(messages=[], response_schema=int))]))

    with pytest.raises(TypeError, match="cannot fork at a Gather"):
        fork_at(prog, [], OpIndex(0), "x", SequencedDomain([]))


# --- P1: a recorded refusal must survive the prefix replay ---------------------------------


def _blocked_policy(op):
    from effective.ops import Step
    from effective.permission import Allow, Deny

    return (
        Deny("policy says no") if isinstance(op, Step) and op.name == "tool:blocked" else Allow()
    )


def _routes_around_a_refusal():
    """Catches the refusal and routes around it — the documented reason a refusal is DELIVERED
    into the workflow rather than raised out of it (`recording.py`'s `Deliver(refused)`)."""
    from effective.govern import Refused

    try:
        yield from step("tool:blocked", CallTool(name="b", result_schema=str))
        route = "allowed"
    except Refused:
        route = "refused"
    yield from step(f"tool:after-{route}", CallTool(name="a", result_schema=str))
    return route


def test_a_refused_prefix_op_replays_as_a_refusal_not_as_none():
    """The fork's prefix must reproduce the run that happened, refusals included.

    Fed `trace[i].result` with no `error` arm, a refused op would replay as `None`, the
    `except Refused` would never fire, and the fork would explore the ALLOWED branch of a run
    that was refused, with no `ReplayMismatch`, because the divergence lands in the tail. The
    free-prefix guarantee is that the prefix is the recorded one."""
    from effective.permission import cascade, rules

    base = RecordingHandler(
        responses={"tool:blocked": "B", "tool:after-refused": "x", "tool:after-allowed": "y"},
        op_layers=[cascade([rules(_blocked_policy)])],
    )
    assert base.run(_routes_around_a_refusal) == "refused"
    assert base.trace[0].error is not None, "the refusal is what the base recorded"

    gen, delivery = replay_prefix(_routes_around_a_refusal, base.trace, OpIndex(1))
    assert delivery.refusal is base.trace[0].error, "the recorded refusal, not a value"
    assert delivery.into(gen).name == "tool:after-refused", "the branch the base actually took"


def test_a_fork_point_inside_a_scope_is_refused_with_the_REGION_as_the_reason():
    """`_replay_scoped`'s refusal names the REGION. `ForkPointRefused`'s message says a fork
    substitutes "a Step result or an AwaitEvent answer", exactly the kind refused here, so it
    would tell a reader the op they forked at is the legal kind, as the reason for refusing it.

    The op is fine; the REGION is the problem. `replay_prefix` drives a scoped body as a
    sub-generator of its own rather than one the workflow `yield from`-ed, so there is no
    generator to hand back positioned at an op inside it.

    Pinned with the CLOSED-scope case beside it, because that is what makes the message true
    rather than merely narrower: a scope the prefix runs to completion forks fine, so the
    refusal really is about the open one containing the fork point."""
    from effective.api import scoped
    from effective.sandbox import ForkPointInsideScope, ForkPointRefused

    def program():
        def inner():
            yield from step("a", CallTool(name="op", result_schema=str))
            return (yield from step("b", CallTool(name="op", result_schema=str)))

        yield from scoped(compose_key(t"lane:{Segment('0')}"), inner)
        return (yield from step("after", CallTool(name="op", result_schema=str)))

    recorded = RecordingHandler(responses={"lane:0;a": "va", "lane:0;b": "vb", "after": "vc"})
    recorded.run(program)
    trace = recorded.trace
    assert [e.key.stored() for e in trace] == ["lane:0;step:a", "lane:0;step:b", "step:after"], [
        e.key.stored() for e in trace
    ]

    # Forking INSIDE the scope — at `lane:0/b`, a Step, which is the legal KIND of op.
    with pytest.raises(ForkPointInsideScope) as refused:
        replay_prefix(program, trace, OpIndex(1))
    assert "lane:0" in str(refused.value)
    assert "one node in the fork's order" in str(refused.value)
    assert refused.value.scope == "lane:0;"
    # It stays a `ForkPointRefused`, so existing handlers keep catching it.
    assert isinstance(refused.value, ForkPointRefused)
    # And it does NOT repeat the parent's claim, which was the defect.
    assert "not a write" not in str(refused.value)

    # AFTER the closed scope is fine — the same program, one index later.
    generator, delivery = replay_prefix(program, trace, OpIndex(2))
    assert generator is not None
    assert delivery.value == "vb"
