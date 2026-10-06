"""`FaultCtx`, the crash instrument, pinned on the surface a race settles through.

A settled checkpoint runs no thunk, so it sits between the two rules the instrument already had:
reachable by NAME at either position, which is what lets a sweep crash a run where its race is
decided, and uncounted at `AFTER_THUNK` by anything else, which is what keeps that position's
population to step executions. An unarmed fault's count is that population, since a sweep arms one
ordinal fault per touch it counted.

| the fault                          | over: step `a`, settle `choice`, step `b` | crashes at |
|------------------------------------|-------------------------------------------|------------|
| `k=2` at `AFTER_THUNK`             | the settle is not counted                 | `b`        |
| `k=2` at `BEFORE_OP`               | the settle is counted                     | `choice`   |
| `named="choice"` at `AFTER_THUNK`  | aimed by name                             | `choice`   |
| `named="choice"` at `BEFORE_OP`    | aimed by name                             | `choice`   |
| `on_name="cho"` at either position | aimed by a substring of the name          | `choice`   |
| unarmed, at either position        | counts what that position can arm         | nothing    |
"""

from contextlib import suppress
from typing import Any

import pytest
from _conformance import Fault, FaultCtx, FaultInjected, FaultPosition, at_every_op


class _Store:
    """A ctx that runs a step's thunk and keeps a settled value, recording each name it commits."""

    def __init__(self) -> None:
        self.committed: list[str] = []
        self.values: dict[str, Any] = {}

    def step(self, name: str, thunk: Any) -> Any:
        value = thunk()
        self.committed.append(name)
        return value

    def peek_step(self, name: str) -> tuple[bool, Any]:
        return (name in self.values, self.values.get(name))

    def settle(self, name: str, value: Any) -> Any:
        self.committed.append(name)
        return self.values.setdefault(name, value)


def _crashes_at(fault: Fault) -> str | None:
    """Which of the three ops the fault crashed, driving them in order through `FaultCtx`."""
    ctx = FaultCtx(_Store(), fault)
    ops = [
        ("a", lambda: ctx.step("a", lambda: "a")),
        ("choice", lambda: ctx.settle("choice", {"kind": "winners"})),
        ("b", lambda: ctx.step("b", lambda: "b")),
    ]
    for name, op in ops:
        try:
            op()
        except FaultInjected:
            return name
    return None


AFTER, BEFORE = FaultPosition.AFTER_THUNK, FaultPosition.BEFORE_OP


def table() -> list[tuple[Fault, str]]:
    """Built per call: a `Fault` fires once, so a table held at module level answers only its first
    reading.

    The aims are `Fault`'s own: a `k`, a `named` whole name and an `on_name` substring, each at
    both positions, with the unarmed fault in its own test below. Rows chosen from the cases in
    mind missed the unarmed aim for a round; the constructor is where the dimension lives."""
    return [
        (Fault(k=2, position=AFTER), "b"),
        (Fault(k=2, position=BEFORE), "choice"),
        (Fault(named="choice", position=AFTER), "choice"),
        (Fault(named="choice", position=BEFORE), "choice"),
        (Fault(on_name="cho", position=AFTER), "choice"),
        (Fault(on_name="cho", position=BEFORE), "choice"),
    ]


def test_a_settle_is_reachable_by_name_and_uncounted_by_an_after_thunk_k():
    rows = table()
    got = [(fault.k, fault.named, fault.position, _crashes_at(fault)) for fault, _ in rows]
    want = [(fault.k, fault.named, fault.position, where) for fault, where in rows]
    assert got == want


@pytest.mark.parametrize("position", [AFTER, BEFORE])
def test_an_unarmed_count_is_the_population_its_position_can_arm(position):
    """A sweep measures how many `k` faults to arm from an unarmed run's count, so every counted
    touch must be one a `k` of that position fires on, and nothing past them."""
    measured = Fault(position=position)
    ctx = FaultCtx(_Store(), measured)
    ctx.step("a", lambda: "a")
    ctx.settle("choice", {"kind": "winners"})
    ctx.step("b", lambda: "b")
    reached = [_crashes_at(Fault(k=k, position=position)) for k in range(1, measured.count + 1)]
    assert None not in reached, (measured.count, reached)
    assert _crashes_at(Fault(k=measured.count + 1, position=position)) is None


@pytest.mark.parametrize("position", [AFTER, BEFORE])
def test_a_crash_at_a_settle_leaves_nothing_committed_under_its_name(position):
    """The two positions name one moment for a settle: the crash lands before the record, so the
    store never holds the choice whichever position armed the fault."""
    store = _Store()
    ctx = FaultCtx(store, Fault(named="choice", position=position))
    with pytest.raises(FaultInjected):
        ctx.settle("choice", {"kind": "winners"})
    assert store.committed == []


@pytest.mark.parametrize(("position", "crashes"), [(AFTER, False), (BEFORE, True)])
def test_a_settle_the_store_holds_crashes_only_before_the_op(position, crashes):
    """A replayed settle writes nothing, so no record is lost after it: `AFTER_THUNK` passes it
    through, as a replayed step runs no thunk. `BEFORE_OP` still dies at the touch."""
    store = _Store()
    store.settle("choice", {"kind": "winners"})
    fault = Fault(named="choice", position=position)
    with suppress(FaultInjected):
        FaultCtx(store, fault).settle("choice", {"kind": "timeout"})
    assert (fault.fired, store.values["choice"]) == (int(crashes), {"kind": "winners"})


def test_at_every_op_refuses_a_fault_that_never_fired():
    """The crash sweeps share this check, so a run that skipped the crash cannot pass as one."""
    unarmed = Fault()
    unarmed.count = 2
    walk = at_every_op(unarmed)
    k, fault = next(walk)
    assert (k, fault.k, fault.armed) == (1, 1, True)
    with pytest.raises(AssertionError, match="never fired"):
        next(walk)
