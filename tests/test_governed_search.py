"""A search under a spend ceiling ends with the state its earlier rounds built, on both engines.

Each round runs one sequential `select` ask, then three branch leaf asks, then a pure join; every
ask costs `COST`. A budget refusal stops the search whether it is raised at the sequential ask or
inside the branches, and whether it arrives at once or as the answer to a park."""

from collections.abc import Callable, Sequence
from uuid import uuid4

import pytest
from _conformance import Fault, private
from _shapes import Outcome, governing, run

from effective import permission
from effective.api import Effect, ask_llm
from effective.budget import Grant, MeasuredBudget, budget_grant_name
from effective.budget import as_policy as budget_policy
from effective.combinators import Answered, Branch, Level, tree_search
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.govern import Policy, Resolution
from effective.permission import Allow, Deny

COST = 0.001
LEAVES = ["a", "b", "c"]
ROUNDS = 3


def join(values: Sequence[int]) -> Effect[int]:
    yield from ()
    return sum(values)


def node(path: str, level: Level):
    if level.depth == 0:
        yield from ask_llm("select", "select", str)
        return Branch(LEAVES, join)
    yield from ask_llm(f"leaf-{path}", f"leaf-{path}", str)
    return Answered(1)


def search(_run_id: str) -> Effect[int]:
    return (
        yield from tree_search(
            "", lambda _s: node, lambda s, v: s + v, initial=0, iterations=ROUNDS, depth=2
        )
    )


def domain() -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda _op: ("ans", Usage(prompt_tokens=1, completion_tokens=1, cost=COST)),
        tools=lambda _op: 0,
    )


def governed(backend, policy_for: Callable[[str], Policy]) -> Outcome:
    """`search` under a gate, on a handler that meters its asks, run to where it stops."""
    return run(
        backend,
        search,
        domain(),
        layers=governing(policy_for),
        contract=Contract.V1,
        max_attempts=1,
    )


def test_a_trip_at_the_sequential_ask_keeps_the_rounds_before_it(backend):
    """Round 0 spends past the ceiling; round 1's `select` trips, so the search answers round 0."""
    name, run_id = private("governed"), str(uuid4())
    backend.register(name, search, domain(), Fault(), (), budget_limit=0.0015, on_exhaust="fail")
    task = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)
    snap = backend.run_until_result(task)
    assert snap.state == "completed", snap
    assert snap.result == len(LEAVES)


def test_a_trip_answered_stop_at_a_park_keeps_the_rounds_before_it(backend):
    name, run_id = private("governed"), str(uuid4())
    backend.register(name, search, domain(), Fault(), (), budget_limit=0.0015, on_exhaust="park")
    task = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)
    assert backend.run_until_result(task).state != "completed"
    backend.emit_event(task, budget_grant_name(run_id, 0).stored(), Grant(stop=True).model_dump())
    snap = backend.run_until_result(task)
    assert snap.state == "completed", snap
    assert snap.result == len(LEAVES)


def test_a_trip_inside_the_branches_keeps_the_rounds_before_it(backend):
    """The gate reads the root meter, which `select` has already pushed past the ceiling, so all
    three leaves of round 0 are refused inside their branches and the search answers `initial`."""
    outcome = governed(
        backend,
        lambda run_id: budget_policy(
            MeasuredBudget(overall=COST / 2, run_id=run_id, on_exhaust="fail")
        ),
    )
    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == 0
    keys = backend.checkpoint_keys(outcome.task)
    assert any(k.endswith("step:select") for k in keys)
    assert not any("step:leaf-" in k for k in keys)


def test_a_denial_beside_a_budget_refusal_is_raised_on(backend):
    def policies(run_id: str) -> Policy:
        over = budget_policy(MeasuredBudget(overall=COST / 2, run_id=run_id, on_exhaust="fail"))
        denies_b = permission.as_policy(
            [permission.rules(lambda op: Deny("no b") if "leaf-b" in str(op) else Allow())]
        )

        def either(op, state):
            return denies_b(op, state) if "leaf-b" in str(op) else over(op, state)

        return either

    snap = governed(backend, policies).snap
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "Refused"


def test_parks_inside_the_branches_answered_stop_keep_the_rounds_before_them(backend):
    """Each leaf parks on its own gate; the task parks at the barrier until every one is answered,
    and three `stop`s end the search with `initial`."""
    outcome = governed(
        backend,
        lambda run_id: budget_policy(
            MeasuredBudget(overall=COST / 2, run_id=run_id, on_exhaust="park")
        ),
    )
    task = outcome.task
    stop = Resolution(answers={"budget": Grant(stop=True).model_dump()}).model_dump()
    answered = 0
    while (snap := backend.run_until_result(task)).state not in ("completed", "failed"):
        (parked,) = backend.parked(task)
        backend.emit_event(task, parked.wake_event, stop)
        answered += 1
        assert answered <= len(LEAVES), "more parks than leaves"
    assert snap.state == "completed", snap
    assert snap.result == 0
    assert answered == len(LEAVES)


ASKS = ROUNDS * (1 + len(LEAVES))
"""Every ask `search` makes: a `select` and a leaf per branch, each round."""


def granted_search(run_id: str) -> Effect[int]:
    """`search`, parking on its round grant when its rounds run out."""
    return (
        yield from tree_search(
            "",
            lambda _s: node,
            lambda s, v: s + v,
            initial=0,
            iterations=ROUNDS,
            depth=2,
            run_id=run_id,
        )
    )


def test_a_ceiling_reached_on_the_last_permitted_round_keeps_every_round(backend):
    """A ceiling half an ask short of what the rounds spend lets the last round through and
    refuses the park for more rounds: the search answers what all the rounds built."""
    outcome = run(
        backend,
        granted_search,
        domain(),
        layers=governing(
            lambda run_id: budget_policy(
                MeasuredBudget(overall=COST * (ASKS - 0.5), run_id=run_id, on_exhaust="fail")
            )
        ),
        contract=Contract.V1,
        max_attempts=1,
    )
    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == ROUNDS * len(LEAVES)


GATES = {
    "budget": lambda budget: budget_policy(budget),
    "permission": lambda _budget: permission.as_policy([permission.rules(lambda _op: Allow())]),
}

ASSEMBLIES = {
    "a handler budget and a budget gate": (True, "budget", "failed"),
    "a handler budget alone": (True, None, "completed"),
    "a budget gate alone": (False, "budget", "completed"),
    "a handler budget and a permission gate": (True, "permission", "completed"),
}
"""Whether the handler holds the budget, which policy the gate holds, and how the task ends."""


@pytest.mark.parametrize("assembly", ASSEMBLIES)
def test_one_budget_takes_one_driver(backend, assembly):
    """Two drivers keep two grant books, so the handler refuses the pair before any op runs."""
    on_handler, policy, ended = ASSEMBLIES[assembly]
    name, run_id = private("governed"), str(uuid4())
    budget = MeasuredBudget(overall=1.0, run_id=run_id, on_exhaust="park")
    layers = () if policy is None else governing(lambda _run_id: GATES[policy](budget))(run_id)
    backend.register(
        name,
        search,
        domain(),
        Fault(),
        layers,
        budget_limit=budget.overall if on_handler else None,
    )
    task = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)
    snap = backend.run_until_result(task)
    assert snap.state == ended, snap
    if ended == "failed":
        assert backend.failure_kind(snap) == "ValueError", snap.failure
        assert "two grant books" in str(snap.failure)
        assert backend.checkpoint_keys(task) == [], "refused before any op"
