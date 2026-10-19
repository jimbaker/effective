"""An op's effect: what it would do to the world, in the canonical form an approval is about."""

import subprocess
import sys
from datetime import UTC, datetime
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict, Field

from effective import call_tool, permission
from effective.domain import AskLLM, CallTool, Judge
from effective.govern import GateState, govern
from effective.handlers.base import UnshowableEffect, effect_of
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    LedgerRow,
    Race,
    ReadLedgerRow,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WorkflowOp,
)

WHEN = datetime(2026, 10, 10, tzinfo=UTC)


def _charge(amount: int) -> Step[Any]:
    return Step("charge", CallTool("charge", dict, {"amount": amount}))


# Each arm twice, differing only in what the op would do. A control op's pair differs too, and
# its effect is None for both, since its key already names it.
PAIRS: dict[type, tuple[WorkflowOp, WorkflowOp]] = {
    Step: (_charge(5), _charge(5_000_000)),
    AppendLedgerRow: (
        AppendLedgerRow(LedgerRow(event_id=Key.parse("e1"), kind="charge", amount=5)),
        AppendLedgerRow(LedgerRow(event_id=Key.parse("e1"), kind="charge", amount=6)),
    ),
    ReadLedgerRow: (ReadLedgerRow(Key.parse("e1")), ReadLedgerRow(Key.parse("e2"))),
    StoreArtifact: (StoreArtifact("a", "text/plain"), StoreArtifact("b", "text/plain")),
    SleepUntil: (SleepUntil(WHEN), SleepUntil(WHEN.replace(day=11))),
    Respawn: (
        Respawn(task="t", generation=1, state={"n": 1}, params={}, run_id="r"),
        Respawn(task="t", generation=1, state={"n": 2}, params={}, run_id="r"),
    ),
    AwaitEvent: (AwaitEvent(Key.parse("review:a"), dict), AwaitEvent(Key.parse("review:b"), dict)),
    Gather: (Gather(branches=()), Gather(branches=(lambda: None,))),
    Race: (
        Race(want=1, branches=(lambda: None, lambda: None)),
        Race(want=2, branches=(lambda: None, lambda: None)),
    ),
    Scoped: (Scoped(Key.parse("s:a"), lambda: None), Scoped(Key.parse("s:b"), lambda: None)),
}
CONTROL = {AwaitEvent, Gather, Race, Scoped}


def test_every_arm_of_the_op_union_is_placed():
    arms = {get_origin(arm) or arm for arm in get_args(WorkflowOp.__value__)}
    assert arms == set(PAIRS)


@pytest.mark.parametrize("arm", list(PAIRS), ids=lambda arm: arm.__name__)
def test_an_effect_tells_apart_two_ops_that_would_act_differently(arm):
    first, second = PAIRS[arm]
    if arm in CONTROL:
        assert effect_of(first) is effect_of(second) is None
    else:
        assert effect_of(first) != effect_of(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            Step("ask", AskLLM([{"role": "user", "content": "a"}], dict)),
            Step("ask", AskLLM([{"role": "user", "content": "b"}], dict)),
        ),
        (
            Step("judge", Judge(state={"x": 1}, questions={}, response_schema=dict)),
            Step("judge", Judge(state={"x": 2}, questions={}, response_schema=dict)),
        ),
    ],
    ids=["ask", "judge"],
)
def test_a_model_request_is_a_digest_that_tells_requests_apart(first, second):
    assert set(effect_of(first)) == {"request", "schema"}
    assert effect_of(first)["request"] != effect_of(second)["request"]


def test_an_effect_is_the_same_under_every_hash_seed():
    probe = (
        "from effective.domain import CallTool\n"
        "from effective.ops import Step\n"
        "from effective.handlers.base import effect_of\n"
        "print(effect_of(Step('t', CallTool('t', dict, {'tags': {'c', 'a', 'b', 'd'}}))))\n"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", probe],
            env={"PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(seen) == 1, seen


def test_a_parked_permission_ask_carries_the_effect_of_the_op_it_gates():
    def detail_for(op: WorkflowOp) -> Any:
        parks = []
        gate = govern(
            permission.as_policy([permission.rules(lambda op: permission.Escalate())]),
            gate="g",
            run_id="r",
            announce=lambda parked, state: parks.append(parked) or op,
        )
        gate(op).send(None)
        (ask,) = parks[0].asks
        return ask.detail

    small, large = detail_for(_charge(5)), detail_for(_charge(5_000_000))
    assert small["op_key"] == large["op_key"]
    assert small["effect"] == {"tool": "charge", "args": {"amount": 5}}
    assert large["effect"] == {"tool": "charge", "args": {"amount": 5_000_000}}


def test_equal_ops_give_equal_effects_however_their_keys_were_made():
    composed = GateState(run_id="r1", gate="g", op_key="step;tool:x").park_name
    parsed = Key.parse(composed.stored())
    assert composed == parsed  # composed carries a scope; the parse does not

    def ledgered(key: Key) -> AppendLedgerRow:
        return AppendLedgerRow(LedgerRow(event_id=Key.parse("e1"), kind="k", ref=key))

    assert effect_of(ledgered(composed)) == effect_of(ledgered(parsed))
    assert effect_of(ledgered(composed))["ref"] == composed.stored()


def test_an_op_with_no_canonical_effect_is_refused_by_name():
    with pytest.raises(UnshowableEffect):
        effect_of(Step("t", CallTool("t", dict, {"handle": object()})))


def test_a_response_schema_reads_as_its_source_spells_it():
    def schema_of(schema: Any) -> str:
        return effect_of(Step("ask", AskLLM([], schema)))["schema"]

    assert schema_of(list[int]) != schema_of(list[str])
    assert schema_of(dict) == "dict"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int


class _Guarded(BaseModel):
    shown: int
    secret: str = Field(exclude=True)


@pytest.mark.parametrize(
    ("args", "effect"),
    [
        ({"pairs": {("b", 2), ("a", 1)}}, {"pairs": [["a", 1], ["b", 2]]}),
        ({"models": frozenset({_Frozen(x=2), _Frozen(x=1)})}, {"models": [{"x": 1}, {"x": 2}]}),
        ({"model": _Guarded(shown=1, secret="pw")}, {"model": {"shown": 1}}),
        ({"keys": {Key.parse("tool:b"), Key.parse("tool:a")}}, {"keys": ["tool:a", "tool:b"]}),
    ],
    ids=["set-of-tuples", "frozenset-of-frozen-models", "excluded-field", "set-of-keys"],
)
def test_an_effect_canonicalizes_what_an_argument_holds(args, effect):
    assert effect_of(Step("t", CallTool("t", dict, args))) == {"tool": "t", "args": effect}


def test_an_op_holding_a_set_of_tuples_parks_at_its_gate(sqlite_app):
    app = sqlite_app(":memory:")

    class _Tags:
        def run(self, op):
            return {}

    def tagging():
        return (yield from call_tool("tag", {"pairs": {("a", 1)}}, dict))

    @app.register_task("tag")
    def task(params, ctx):
        policy = permission.as_policy([permission.rules(lambda op: permission.Escalate())])
        gate = govern(policy, gate="g", run_id="r")
        return DurableHandler(ctx, _Tags(), op_layers=(gate,)).run(tagging)

    assert app.run_until_result(app.spawn("tag", {})).state == "waiting"
