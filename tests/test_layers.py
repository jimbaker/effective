"""Harness layer scaffolding: the trampoline, the two combinators, retry.

Proves the mechanism in isolation (rewrite / transform / multi-yield / exception-
into-yield / refusal short-circuit), then end-to-end at each seam:

- domain seam: a `@domain_layer` observing AskLLM over a toy base interpreter;
- op seam: `retry` wired into `DurableHandler.op_layers`, recovering a flaky domain.

The record/replay interaction of a retried op (op_key disambiguation; modeling a
failed attempt in the trace) is deferred and deliberately not exercised here: retry is
proven against live/no-trace handlers only.
"""

import pytest

from effective import (
    Interpreter,
    TransientError,
    compose_domain,
    compose_ops,
    domain_layer,
    drive_through,
    op_layer,
    retry,
)
from effective.api import ask_llm, await_event, call_tool
from effective.domain import AskLLM, CallTool
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import Key
from effective.layers import (
    LAYERED_OPS,
    LAYERED_OPS_DURABLE_ONLY,
    UNLAYERED_OPS,
    layer_routing,
)
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    Race,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WorkflowOp,
)
from effective.react import AssistantTurn, ToolResult

# --- the trampoline --------------------------------------------------------


def _identity(op):
    return op


def test_empty_stack_calls_base_directly():
    assert drive_through([], 5, lambda op: op + 1) == 6


def test_layer_can_rewrite_the_op_before_forwarding():
    @op_layer
    def double(op):
        return (yield op * 2)

    assert drive_through([double], 5, _identity) == 10


def test_layer_can_transform_the_result():
    @op_layer
    def plus_one(op):
        return (yield op) + 1

    assert drive_through([plus_one], 5, _identity) == 6


def test_layer_order_is_outer_to_inner():
    @op_layer
    def add1(op):
        return (yield op) + 1

    @op_layer
    def times2(op):
        return (yield op) * 2

    # add1 is outer, times2 inner: base(5)=5 -> times2 ->10 -> add1 ->11
    assert drive_through([add1, times2], 5, _identity) == 11


def test_downstream_exception_propagates_into_the_yield():
    @op_layer
    def swallow(op):
        try:
            return (yield op)
        except ValueError:
            return "caught"

    def boom(op):
        raise ValueError("downstream")

    assert drive_through([swallow], "op", boom) == "caught"


def test_layer_may_return_before_yielding():
    @op_layer
    def refuse(op):
        if op == "deny":
            return "refused"
        return (yield op)
        yield  # unreachable; keeps this a generator function

    assert drive_through([refuse], "deny", _identity) == "refused"
    assert drive_through([refuse], "allow", _identity) == "allow"


# --- retry: multi-yield is the proof ---------------------------------------


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky(op):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("flaky")
        return "ok"

    assert drive_through([retry(attempts=2)], "OP", flaky) == "ok"
    assert calls["n"] == 3  # 2 failures + 1 success


def test_retry_exhausts_and_reraises_the_last_error():
    def always(op):
        raise TransientError("always")

    with pytest.raises(TransientError):
        drive_through([retry(attempts=2)], "OP", always)


def test_retry_ignores_non_transient_errors():
    def boom(op):
        raise ValueError("not transient")

    with pytest.raises(ValueError, match="not transient"):
        drive_through([retry(attempts=3)], "OP", boom)


# --- compose_domain (the domain seam) --------------------------------------


class _Base:
    """A toy DomainInterpreter base: returns a typed result per op kind."""

    def run(self, op):
        match op:
            case AskLLM():
                return AssistantTurn(answer="hi")
            case CallTool():
                return ToolResult(content="tool")
        raise TypeError(op)


def test_compose_domain_threads_through_a_domain_layer():
    seen: list[AskLLM] = []

    @domain_layer
    def observe(op):
        result = yield op
        if isinstance(op, AskLLM):
            seen.append(op)
        return result

    stack = compose_domain([observe], base=_Base())
    assert isinstance(stack, Interpreter)

    out = stack.run(AskLLM(messages=[], response_schema=AssistantTurn))
    assert isinstance(out, AssistantTurn)
    assert out.answer == "hi"
    assert len(seen) == 1

    tool_out = stack.run(CallTool(name="x", result_schema=ToolResult))
    assert isinstance(tool_out, ToolResult)
    assert len(seen) == 1  # CallTool is not an AskLLM


# --- compose_ops (the op seam, through a real handler) ---------------------


class _LocalCtx:
    """A non-durable Absurd ctx: a step just runs its thunk inline."""

    def step(self, name, thunk):
        return thunk()

    def await_event(self, name):
        raise NotImplementedError

    def sleep_until(self, when, *, name: Key | None = None):
        raise NotImplementedError


class _FlakyDomain:
    """Fails the first AskLLM with a transient error, then succeeds."""

    def __init__(self):
        self.n = 0

    def run(self, op):
        self.n += 1
        if self.n == 1:
            raise TransientError("first call is flaky")
        return AssistantTurn(thought="recovered", answer="done")


def _one_shot():
    turn = yield from ask_llm("ask:0", [], AssistantTurn)
    return turn


def test_retry_op_layer_recovers_a_flaky_step_through_the_handler():
    domain = _FlakyDomain()
    handler = DurableHandler(ctx=_LocalCtx(), domain=domain, op_layers=[retry(attempts=2)])

    result = handler.run(_one_shot)
    turn = AssistantTurn.model_validate(result)

    assert turn.answer == "done"
    assert domain.n == 2  # the Step was re-forwarded once after the transient failure


def test_compose_ops_configures_the_handler_op_layers():
    handler = DurableHandler(ctx=_LocalCtx(), domain=_Base())
    same = compose_ops([retry(attempts=1)], handler)
    assert same is handler
    assert len(handler.op_layers) == 1


# --- L3: which ops reach an op-layer is a per-handler placement (a NAMED divergence) ------


class _InlineAwaitCtx:
    """A durable ctx whose await returns inline (no real park) — enough to route an AwaitEvent
    through the op-layer stack, which is the L3 point (a real park needs the engine)."""

    def step(self, name, thunk):
        return thunk()

    def await_event(self, name):
        return {"ok": True}

    def sleep_until(self, when, *, name: Key | None = None):
        raise NotImplementedError


class _ToolDomain:
    def run(self, op):
        return "tool-done"


def _tool_then_await():
    yield from call_tool("t", {}, str)
    return (yield from await_event("ev", dict))


def _counting_layer(seen: list[str]):
    @op_layer
    def layer(op):
        seen.append(type(op).__name__)
        result = yield op
        return result

    return layer


def test_L3_layer_op_stream_named_divergence():
    # L3: op-layers see `AwaitEvent` on the DURABLE handler but NEVER on the recording core (which
    # decides park/resume before the layer: no call/cc). Alignment is impossible; the contract is
    # `layers.LAYERED_OPS` / `LAYERED_OPS_DURABLE_ONLY`, and this row asserts the streams PER that
    # contract, not by equality, with the divergence pinned as intended.
    rec_seen: list[str] = []
    dur_seen: list[str] = []
    RecordingHandler(
        responses={"tool:t": "tool-done", "ev": {"ok": True}},
        op_layers=[_counting_layer(rec_seen)],
    ).run(_tool_then_await)
    DurableHandler(
        ctx=_InlineAwaitCtx(), domain=_ToolDomain(), op_layers=[_counting_layer(dur_seen)]
    ).run(_tool_then_await)

    assert "Step" in rec_seen  # BOTH route the tool Step through layers
    assert "Step" in dur_seen
    # everything the recording layer saw is a shared LAYERED_OPS type — no AwaitEvent among them
    layered_names = {t.__name__ for t in LAYERED_OPS}
    assert all(name in layered_names for name in rec_seen)
    assert "AwaitEvent" not in rec_seen
    # the durable handler routes the AwaitEvent through layers — the named durable-only divergence
    assert "AwaitEvent" in dur_seen
    assert [t.__name__ for t in LAYERED_OPS_DURABLE_ONLY] == ["AwaitEvent"]


# --- the layer-routing partition, enforced rather than trusted ------------------------------
#
# Three tuples encode which ops reach the op-layer stack, and until `layer_routing` existed
# nothing said they PARTITION `WorkflowOp`. A tenth arm would appear in none of them and be
# routed to the layer stack silently, by both handlers, without anyone deciding it should be.


def _one_of_each_arm():
    """An instance of every arm of `WorkflowOp`, built from the union itself.

    Enumerated from `typing.get_args` rather than hand-listed, because a hand-listed set is the
    very thing these tests exist to stop drifting from the union."""
    import typing
    from datetime import UTC, datetime

    from effective.domain import CallTool
    from effective.ops import LedgerRow

    made: dict[type, object] = {
        Step: Step(name="s", op=CallTool(name="t", args={}, result_schema=object)),
        AwaitEvent: AwaitEvent(name=Key.parse("ev:x"), schema=object),
        AppendLedgerRow: AppendLedgerRow(row=LedgerRow(event_id=Key.parse("led:x"), kind="k")),
        StoreArtifact: StoreArtifact(value="c", content_type="text/plain"),
        SleepUntil: SleepUntil(when=datetime(2030, 1, 1, tzinfo=UTC)),
        Gather: Gather(branches=()),
        Race: Race(want=1, branches=(lambda: None,)),
        Scoped: Scoped(scope=Key.parse("s:one"), body=lambda: None),
        Respawn: Respawn(task="t", generation=1, state=None, params={}, run_id="r"),
    }
    arms = [typing.get_origin(a) or a for a in typing.get_args(WorkflowOp.__value__)]
    assert set(arms) == set(made), "an arm of WorkflowOp has no instance here — add one"
    return [made[a] for a in arms]


def test_the_three_tuples_partition_WorkflowOp_exactly():
    """Reddens if an arm is added to the union and to no tuple — the silent-default case.

    Disjoint AND covering. Either failure is the denylist hazard: an arm in two tuples is
    ambiguous, an arm in none defaults to the layer stack without a decision."""
    tuples = {
        "layered": LAYERED_OPS,
        "durable-only": LAYERED_OPS_DURABLE_ONLY,
        "unlayered": UNLAYERED_OPS,
    }
    seen: dict[type, str] = {}
    for label, group in tuples.items():
        for arm in group:
            assert arm not in seen, f"{arm.__name__} is in both {seen[arm]} and {label}"
            seen[arm] = label
    import typing

    arms = {typing.get_origin(a) or a for a in typing.get_args(WorkflowOp.__value__)}
    assert set(seen) == arms, f"not partitioned: {arms ^ set(seen)}"


def test_layer_routing_agrees_with_the_tuples_on_every_arm():
    """Reddens if the function and the tuples drift — two statements of one partition.

    The tuples stay because docstrings and tests cite them as vocabulary; the function is what
    the handlers actually call. Neither may quietly disagree with the other."""
    by_tuple = (
        {t: "layered" for t in LAYERED_OPS}
        | {t: "layered-durable-only" for t in LAYERED_OPS_DURABLE_ONLY}
        | {t: "unlayered" for t in UNLAYERED_OPS}
    )
    for op in _one_of_each_arm():
        assert layer_routing(op) == by_tuple[type(op)], type(op).__name__


def test_layer_routing_refuses_an_arm_it_has_no_answer_for():
    """Reddens if the wildcard is softened back to a default.

    The whole point: a new arm must be a loud question, not a silent trip through the stack.

    `AssertionError`, from `assert_never`: the one spelling every closed dispatch in the tree
    uses. The check is static, so `ty` names the missing arm before this test runs."""

    class Tenth:
        pass

    with pytest.raises(AssertionError, match="unreachable, but got"):
        layer_routing(Tenth())  # ty: ignore[invalid-argument-type]
