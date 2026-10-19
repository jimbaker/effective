"""The schedules `_turns` explores are the orders of the attempts' calls, each once."""

from math import comb

import pytest
from _turns import Turns, schedules


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
