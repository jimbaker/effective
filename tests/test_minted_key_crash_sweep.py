"""A minted idempotency key, crashed at every op in both halves of the op, on both engines.

Each key a tool receives names a checkpoint that exists, a crash hands the tool only keys it
has seen, and no key names two placements: so a receiver that remembers keys acts once.
"""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op

from effective.api import gather, race, scoped, step
from effective.choice import Chosen, Won
from effective.domain import CallTool
from effective.handlers.absurd import idempotency_key_for
from effective.keys import Key, compose_key
from effective.layers import TransientError, op_layer, retry
from effective.ops import Minted, Step, Writer

pytestmark = pytest.mark.adversarial


class _Remembering:
    """Records each `(tool, key)` it is handed; the first `fails` calls of the `flaky` tool fail
    transiently."""

    def __init__(self, flaky: str | None = None, fails: int = 1) -> None:
        self.keys: list[tuple[str, str]] = []
        self.flaky, self.fails = flaky, fails

    def run(self, op: Any) -> Any:
        self.keys.append((op.name, op.args["idempotency_key"]))
        if op.name == self.flaky and [name for name, _ in self.keys].count(op.name) <= self.fails:
            raise TransientError(f"call {len(self.keys)} of {op.name}")
        return {"tool": op.name}


def _keyed(name: str):
    call = CallTool(name=name, args={}, result_schema=dict)
    return (yield from step(name, call, idempotency_key=Minted()))


def _nested(run_id: str):
    """`gather[scoped(rec:0, race[a, a]), b]`."""

    def branch():
        return [(yield from _keyed("a")), (yield from _keyed("a"))]

    def raced():
        match (yield from scoped(compose_key(t"rec:{0}"), lambda: race([branch]))):
            case Chosen(endings=[Won(value=value)]):
                return value
            case unexpected:
                raise AssertionError(unexpected)

    return (yield from gather([raced, lambda: _keyed("b")]))


def _one_a(run_id: str):
    return (yield from _keyed("a"))


def _retried(run_id: str):
    return [(yield from _keyed("a")), (yield from _keyed("b"))]


@op_layer
def _audits_first(op):
    """Runs a keyed audit step ahead of the op it wraps."""
    audit = CallTool(name="audit", args={}, result_schema=dict)
    yield Step(name="audit", op=audit, idempotency_key=Minted())
    return (yield op)


@op_layer
def _audits_around(op):
    """Runs a keyed audit step on each side of the op it wraps."""
    before = yield from _keyed("audit")
    value = yield op
    after = yield from _keyed("audit")
    return {"before": before, "value": value, "after": after}


@op_layer
def _announces_each_failure(op):
    """Re-forwards its op on a transient failure, announcing each failure in a keyed step."""
    for _ in range(3):
        try:
            return (yield op)
        except TransientError:
            announce = CallTool(name="announce", args={}, result_schema=dict)
            yield Step(name="announce", op=announce, idempotency_key=Minted())
    raise TransientError("out of tries")


class _FailsByKind(_Remembering):
    """Fails `a` transiently with these kinds, a call each, then succeeds."""

    KINDS = ("rate-limit", "timeout", "rate-limit", "rate-limit", "timeout")

    def run(self, op: Any) -> Any:
        self.keys.append((op.name, op.args["idempotency_key"]))
        calls = [name for name, _ in self.keys].count(op.name)
        if op.name == "a" and calls <= len(self.KINDS):
            raise TransientError(self.KINDS[calls - 1])
        return {"tool": op.name}


ANNOUNCEMENT = {"rate-limit": "announce-rate-limit", "timeout": "announce-timeout"}


@op_layer
def _announces_the_kind(op):
    """Announces a transient failure's kind in a keyed step, then raises it again."""
    try:
        return (yield op)
    except TransientError as failure:
        yield from _keyed(ANNOUNCEMENT[failure.args[0]])
        raise


SHAPES = {
    "nested": (_nested, tuple, _Remembering),
    "retried": (_retried, lambda: (retry(2),), lambda: _Remembering(flaky="a")),
    "retried-above-an-injecting-layer": (
        _retried,
        lambda: (retry(2), _audits_first),
        lambda: _Remembering(flaky="a"),
    ),
    "retried-injected-op": (
        _retried,
        lambda: (_audits_first, retry(2)),
        lambda: _Remembering(flaky="audit"),
    ),
    "nested-retries-around-an-auditing-layer": (
        _retried,
        lambda: (retry(1), retry(1), _audits_around),
        lambda: _Remembering(flaky="a", fails=2),
    ),
}


def _run(backend, shape, fault):
    factory, layers, domain_for = shape
    domain, run_id = domain_for(), f"r-{uuid4().hex[:8]}"
    backend.register(f"minted-{run_id}", factory, domain, fault, layers())
    task_id = backend.spawn(f"minted-{run_id}", run_id)
    snapshot = backend.run_until_result(task_id)
    steps = {k for k in backend.checkpoint_keys(task_id) if "step:" in k}
    minted = {
        idempotency_key_for(Writer(task=str(task_id), placement=Key.parse(s))) for s in steps
    }
    return snapshot, domain, {key.stored() for key in minted}


def _placements(keys: set[str]) -> list[str]:
    return sorted(key.split(";", 1)[1] for key in keys)


@pytest.mark.parametrize("position", list(FaultPosition))
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_a_minted_key_survives_a_crash_at_every_op(backend, shape, position):
    unarmed = Fault(None, position=position)
    snapshot, domain, expected = _run(backend, shape, unarmed)
    assert snapshot.state == "completed", snapshot
    assert {key for _, key in domain.keys} == expected
    placements, calls = _placements(expected), len(domain.keys)

    for k, fault in at_every_op(unarmed):
        snapshot, domain, expected = _run(backend, shape, fault)
        seen = {key for _, key in domain.keys}
        assert snapshot.state == "completed", f"k={k}: {snapshot}"
        assert seen == expected, f"k={k}"
        assert _placements(seen) == placements, f"k={k}"
        # a crash after the thunk repeats one call, with a key the receiver already holds
        extra = 1 if position is FaultPosition.AFTER_THUNK else 0
        assert len(domain.keys) == calls + extra, f"k={k}: {domain.keys}"
        tools_per_key: dict[str, set[str]] = {}
        for tool, key in domain.keys:
            tools_per_key.setdefault(key, set()).add(tool)
        assert all(len(tools) == 1 for tools in tools_per_key.values()), f"k={k}"


def _swaps_its_audits_on_a_re_forward():
    """A layer that runs `x` then `y` on its first invocation and `y` then `x` after, counting its
    own invocations as the shipped gates number theirs."""
    invocations = []

    @op_layer
    def swaps(op):
        invocations.append(op)
        for name in ("x", "y") if len(invocations) == 1 else ("y", "x"):
            yield from _keyed(name)
        return (yield op)

    return swaps


INTENTS = {
    "nested": (SHAPES["nested"], {"a": (2, 2), "b": (1, 1)}),
    "retried": (SHAPES["retried"], {"a": (1, 2), "b": (1, 1)}),
    "a-layer-announcing-each-failure": (
        (_retried, lambda: (_announces_each_failure,), lambda: _Remembering("a", fails=2)),
        {"announce": (2, 2)},
    ),
    "nested-retries-around-an-auditing-layer": (
        SHAPES["nested-retries-around-an-auditing-layer"],
        {"audit": (4, 4)},
    ),
    "a-layer-swapping-its-audits-on-a-re-forward": (
        (
            _one_a,
            lambda: (retry(1), _swaps_its_audits_on_a_re_forward()),
            lambda: _Remembering("a"),
        ),
        {"x": (2, 2), "y": (2, 2), "a": (1, 2)},
    ),
}
"""Each shape, and for each tool how many distinct keys it is handed and how many calls it gets
on an uncrashed run. A step served from its checkpoint calls no tool. The announcing layer's
shape stays out of the sweep: a crash can absorb a live failure, so its count of announcements is
the attempt's."""


@pytest.mark.parametrize(("shape", "handed"), INTENTS.values(), ids=INTENTS.keys())
def test_each_tool_is_handed_its_intended_keys_and_calls(backend, shape, handed):
    snapshot, domain, minted = _run(backend, shape, Fault(None))
    assert snapshot.state == "completed", snapshot
    for tool, (distinct, calls) in handed.items():
        keys = [key for name, key in domain.keys if name == tool]
        assert (len(set(keys)), len(keys)) == (distinct, calls), domain.keys
        assert set(keys) <= minted, (keys, minted)


def _one_key_per_kind(keys: list[tuple[str, str]]) -> bool:
    return all(
        len({key for tool, key in keys if tool == announcement}) == 1
        for announcement in ANNOUNCEMENT.values()
    )


@pytest.mark.parametrize("position", list(FaultPosition))
def test_a_kind_announcing_retry_survives_a_crash_at_every_op_on_a_domain_per_attempt(
    backend, position
):
    """A domain built per attempt fails by kind from its first call on every attempt, so each
    resumed attempt meets the same failures and must be handed the same names."""

    def run(fault):
        run_id, attempts = f"r-{uuid4().hex[:8]}", []

        def fresh():
            attempts.append(_FailsByKind())
            return attempts[-1]

        layers = (retry(1), retry(2), _announces_the_kind)
        backend.register(f"minted-{run_id}", _retried, None, fault, layers, fresh=fresh)
        task_id = backend.spawn(f"minted-{run_id}", run_id)
        snapshot = backend.run_until_result(task_id)
        steps = {k for k in backend.checkpoint_keys(task_id) if "step:" in k}
        minted = {
            idempotency_key_for(Writer(task=str(task_id), placement=Key.parse(s))).stored()
            for s in steps
        }
        return snapshot, [pair for domain in attempts for pair in domain.keys], minted

    unarmed = Fault(None, position=position)
    snapshot, keys, expected = run(unarmed)
    assert snapshot.state == "completed", snapshot
    placements = _placements(expected)
    assert _one_key_per_kind(keys), keys
    for k, fault in at_every_op(unarmed):
        snapshot, keys, expected = run(fault)
        assert snapshot.state == "completed", f"k={k}: {snapshot}"
        assert {key for _, key in keys} == expected, f"k={k}"
        assert _placements(expected) == placements, f"k={k}"
        assert _one_key_per_kind(keys), (k, keys)
