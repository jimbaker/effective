"""The schedules `_turns` explores are the orders of the attempts' calls, each once."""

from functools import partial
from itertools import permutations
from math import comb
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, branched, private
from _turns import Turns, gather_branches, schedules

from effective.api import await_event, call_tool, gather
from effective.keys import Key


def _orders(n: int) -> tuple[int, set[tuple[str, ...]]]:
    """How many runs `schedules` makes for two attempts of `n` calls each, and the call orders
    they cover."""
    runs, orders = 0, set()
    for turns in schedules(["a", "b"], switches=2 * n):
        called: list[str] = []

        def calls(name: str, turns: Turns = turns, called: list[str] = called) -> None:
            for _ in range(n):
                turns.wait(name)
                called.append(name)

        turns.run({name: lambda name=name: calls(name) for name in turns.names})
        runs += 1
        orders.add(tuple(called))
    return runs, orders


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_run_is_a_distinct_order_of_the_calls(n):
    runs, orders = _orders(n)
    assert runs == len(orders) == comb(2 * n, n)


def test_a_prefix_choosing_an_attempt_that_is_not_waiting_fails_at_once():
    turns = Turns(["a", "b"], prefix=["c"])
    with pytest.raises(ValueError, match="'c'"):
        turns.run({"a": lambda: turns.wait("a"), "b": lambda: turns.wait("b")})


class Called:
    """A domain that answers every tool and records the order the calls ran in."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op):
        self.calls.append(op.name)
        return 1


def _two_calls(branch: str):
    for i in (1, 2):
        yield from call_tool(f"{branch}{i}", {}, int)


def _gathered(backend, switches: int) -> list[tuple[str, ...]]:
    """The call order of each run of two gathered branches of two tool calls, one run per
    schedule of their ops."""
    orders = []
    for turns in schedules(gather_branches(2), switches):
        domain, name = Called(), private("turns")
        branches = turns.gathered([partial(_two_calls, branch) for branch in "ab"])
        program = partial(gather, branches)
        backend.register(name, lambda _, run=program: run(), domain, Fault(), (turns.layer(),))
        snap = backend.run_until_result(backend.spawn(name, str(uuid4())))
        turns.settle()
        assert snap.state == "completed", snap
        orders.append(tuple(domain.calls))
    return orders


@pytest.mark.parametrize(("switches", "runs"), [(1, 4), (2, 6)])
def test_ops_of_gathered_branches_take_turns_in_every_order(backend, switches, runs):
    """Every schedule is a distinct order of the calls, and two switches reach all of them."""
    orders = _gathered(backend, switches)
    assert len(orders) == len(set(orders)) == runs
    every = {o for o in permutations(["a1", "a2", "b1", "b2"]) if o.index("a1") < o.index("a2")}
    every = {o for o in every if o.index("b1") < o.index("b2")}
    assert set(orders) <= every
    assert switches < 2 or set(orders) == every


def _awaits_then_calls(branch: str, event: Key):
    yield from await_event(event, int)
    yield from call_tool(f"{branch}1", {}, int)


def test_the_choices_go_on_across_a_park(backend):
    """Both branches park in the first claim. In the claim that resumes them, either may call
    first, and a schedule reaches each."""
    orders = set()
    for turns in schedules(gather_branches(2), switches=2):
        domain, name = Called(), private("parked")
        events = {branch: Key.parse(f"ready-{branch}:{uuid4()}") for branch in "ab"}
        bodies = [partial(_awaits_then_calls, branch, events[branch]) for branch in "ab"]
        program = partial(gather, turns.gathered(bodies))

        fresh = turns.claiming(domain)
        backend.register(
            name, lambda _, run=program: run(), domain, Fault(), (turns.layer(),), fresh=fresh
        )
        task = backend.spawn(name, str(uuid4()))
        snap = backend.run_until_result(task)
        assert snap.state not in ("completed", "failed"), snap
        for i, event in enumerate(events.values()):
            backend.emit_event(task, branched(0, i, event).stored(), 1)
        snap = backend.run_until_result(task)
        turns.settle()
        assert snap.state == "completed", snap
        orders.add(tuple(domain.calls))
    assert orders == {("a1", "b1"), ("b1", "a1")}


@pytest.mark.parametrize("position", list(FaultPosition))
def test_a_crash_in_a_turn_ends_its_branch_and_the_resume_completes(backend, position):
    """A crash inside branch 0's first call ends the branch's turn at once, so branch 1 is never
    left waiting on a turn the dead branch holds, and every schedule resumes to completion."""
    for turns in schedules(gather_branches(2), switches=2):
        domain, name = Called(), private("crashed")
        program = partial(gather, turns.gathered([partial(_two_calls, b) for b in "ab"]))
        fault = Fault(named="gather:0,0;step;tool:a1", position=position)
        layers, fresh = (turns.layer(),), turns.claiming(domain)
        backend.register(name, lambda _, run=program: run(), domain, fault, layers, fresh=fresh)
        snap = backend.run_until_result(backend.spawn(name, str(uuid4())))
        turns.settle()
        assert snap.state == "completed", (snap, turns.choices)
        assert fault.fired, turns.choices
