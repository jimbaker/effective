"""Which rows are schedule-independent: each spelling under each forced order, compared.

The predictions were written before these ran.

| row                                           | independent |
|-----------------------------------------------|-------------|
| `halving`, `mcts`, `beam`                     | yes         |
| the root meter's bits after a V1 gather       | yes         |
| a `CostBudget` shared by the branches (V0)    | no          |
| a count held in a layer's closure             | no          |
| last write wins over the leaf rows            | no          |
| the telemetry meter's bits                    | no          |
"""

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from _schedules import ledger_path
from _shapes import Outcome, Schedule, Shape, governing, independent, interleave
from test_reader_quotient import FORKED, latest
from test_search_conformance import SEARCHES
from test_shape_conformance import BUDGETS, SPELLINGS, Tree, at, row

from effective.api import Effect, ask_llm, call_tool, gather
from effective.cost import Contract, CostBudget, MeteredInterpreter, Usage
from effective.govern import GateState, Proceed, Refused
from effective.layers import OpLayer, op_layer
from effective.ops import Step, WorkflowOp

GATHERING: dict[str, Shape] = {
    **{f"halving, {budget}": row("halving", BUDGETS[budget]) for budget in BUDGETS},
    **SEARCHES,
}


@pytest.mark.parametrize("name", list(GATHERING))
def test_a_gathering_row_is_schedule_independent(backend, name):
    seen = interleave(backend, GATHERING[name])
    assert {s.state for s in seen.values()} == {"completed"}
    assert independent(seen)


# --- rows that read a cell their siblings mutate -------------------------------------------------

BRANCHES = ("x0", "x1", "x2")


def asks(names: Sequence[str]) -> Callable[[str], Effect[list[str]]]:
    def program(_run_id: str) -> Effect[list[str]]:
        answers = yield from gather([lambda n=n: ask_llm(n, n, str) for n in names])
        yield from ask_llm("after", "after", str)
        return answers

    return program


def priced(costs: dict[str, float], budget: CostBudget | None = None) -> MeteredInterpreter:
    """Prices an ask by its prompt, which `asks` sets to the ask's name."""
    return MeteredInterpreter(
        llm=lambda op: ("ans", Usage(cost=costs[op.messages])),
        tools=lambda _op: 0,
        budget=budget,
    )


def model_s() -> Shape:
    """Each branch asks $1 of a domain whose $1.50 `CostBudget` all three share."""
    costs = dict.fromkeys((*BRANCHES, "after"), 1.0)

    def domain() -> MeteredInterpreter:
        return priced(costs, CostBudget(1.5))

    return Shape(spellings={"gather": asks(BRANCHES)}, domain=domain)


def third_refused(_run_id: str) -> Sequence[OpLayer[Any]]:
    admitted: list[WorkflowOp] = []

    @op_layer
    def count(op: WorkflowOp):
        admitted.append(op)
        if len(admitted) == 3:
            raise Refused(op, "the third op")
        return (yield op)

    return (count,)


class Ones:
    def run(self, _op: Any) -> int:
        return 1


def closure_count() -> Shape:
    def program(_run_id: str) -> Effect[list[int]]:
        return (yield from gather([lambda n=n: call_tool(n, {}, int) for n in BRANCHES]))

    return Shape(spellings={"gather": program}, domain=Ones, layers=third_refused)


def last_write_wins() -> Shape:
    orders = {"left first": ("0", "1"), "right first": ("1", "0")}
    return Shape(
        spellings={s: at(p, FORKED) for s, p in SPELLINGS["halving"].items()},
        domain=Tree,
        schedules={name: Schedule(order, ledger_path) for name, order in orders.items()},
        observe=lambda backend, outcome: latest(backend.ledger_payloads(outcome.run_id)),
    )


COSTS = {"x0": 0.1, "x1": 0.2, "x2": 0.3, "after": 0.0}
"""`(0.1 + 0.2) + 0.3` and `(0.3 + 0.2) + 0.1` differ in the last bit."""


def fold(reading: str) -> Shape:
    """Three branches under V1. `root` reads the meter the handler folds at the barrier, from a
    gate at the ask after it; `telemetry` reads the domain's, which accrues as asks complete."""
    root: dict[str, str] = {}

    def policy_for(run_id: str):
        def read(op: WorkflowOp, state: GateState) -> Proceed:
            if isinstance(op, Step) and op.name == "after" and state.meter is not None:
                root[run_id] = float(state.meter.cost).hex()
            return Proceed()

        return read

    def observe(_backend, outcome: Outcome) -> str:
        match reading:
            case "root":
                return root[outcome.run_id]
            case _:
                return outcome.domain.meter.cost.hex()

    return Shape(
        spellings={"gather": asks(BRANCHES)},
        domain=lambda: priced(COSTS),
        layers=governing(policy_for),
        contract=Contract.V1,
        observe=observe,
    )


ROWS: dict[str, tuple[Callable[[], Shape], bool]] = {
    "the root meter's bits": (lambda: fold("root"), True),
    "a CostBudget the branches share": (model_s, False),
    "a count in a layer's closure": (closure_count, False),
    "last write wins over the leaf rows": (last_write_wins, False),
    "the telemetry meter's bits": (lambda: fold("telemetry"), False),
}


@pytest.mark.parametrize("name", list(ROWS))
def test_a_row_is_independent_as_predicted(backend, name):
    shape, predicted = ROWS[name]
    seen = interleave(backend, shape())
    print(backend.name, name, {key: (s.state, s.answer, s.observed) for key, s in seen.items()})
    assert independent(seen) is predicted, seen
    if name == "the root meter's bits":
        assert {s.observed for s in seen.values()} == {"0x1.3333333333334p-1"}, "index order"
