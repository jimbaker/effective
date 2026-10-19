"""Runs parked under an earlier public release, banked to be resumed under this tree.

`scripts/bank_public_runs.py` records each shape with the release's own source on the path, and
the resume each one reaches there; `tests/test_public_runs_resume.py` resumes the banked record
here. The module imports only what the release also has, so both trees run it.

| shape             | the run                                        | its answer           |
|-------------------|------------------------------------------------|----------------------|
| `approve-bare`    | a v1 run gated on `charge`                     | the answers, bare    |
| `approve-wrapped` | the same                                       | `{"answers": …}`     |
| `step-then-park`  | a step, then the gated `charge`                | `{"answers": …}`     |
| `gather-park`     | `charge-a` and `charge-b` gathered, `a` gated  | `{"answers": …}`     |
| `unmarked-park`   | a run spawned with no contract, gated `charge` | `{"answers": …}`     |
| `model-collision` | a model call, then the gated `charge`          | `{"answers": …}`     |
| `model-gather`    | a model call and the gated `charge-a` gathered | `{"answers": …}`     |

A step named `model:<name>` is a model call, which the unmetered `Calls` answers bare with a
result whose keys an envelope also has.
"""

from dataclasses import dataclass
from typing import Any

from effective import cost
from effective.api import ask_llm, gather, step
from effective.domain import CallTool
from effective.engines.sqlite import SqliteApp
from effective.govern import Ask, Proceed, govern, park
from effective.handlers.durable import DurableHandler
from effective.ops import Step

RELEASE_MARK = "__accrual_contract__"
"""The param the release marks a run's contract under."""

GATED = frozenset({"charge", "charge-a"})
APPROVED = {"approve": "yes"}


@dataclass(frozen=True)
class Shape:
    name: str
    steps: tuple[str, ...]
    gathered: bool
    marked: bool
    answer: Any

    @property
    def run_id(self) -> str:
        return f"r-{self.name}"

    def params(self) -> dict[str, Any]:
        marked = {RELEASE_MARK: "v1"} if self.marked else {}
        return {"run_id": self.run_id} | marked


SHAPES = (
    Shape("approve-bare", ("charge", "tail"), False, True, APPROVED),
    Shape("approve-wrapped", ("charge", "tail"), False, True, {"answers": APPROVED}),
    Shape("step-then-park", ("prefix", "charge", "tail"), False, True, {"answers": APPROVED}),
    Shape("gather-park", ("charge-a", "charge-b"), True, True, {"answers": APPROVED}),
    Shape("unmarked-park", ("charge", "tail"), False, False, {"answers": APPROVED}),
    Shape(
        "model-collision", ("model:prefix", "charge", "tail"), False, True, {"answers": APPROVED}
    ),
    Shape("model-gather", ("model:prefix", "charge-a"), True, True, {"answers": APPROVED}),
)


COLLIDING = {"result": {"amount": 1}, "usage": {"cost": 7}, "business": "keep-me"}


class Calls:
    """A domain that answers each tool with its name and each model call with `COLLIDING`, and
    records the calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: Any) -> Any:
        match op:
            case CallTool(name=name):
                self.calls.append(name)
                return name
            case _:
                self.calls.append("model")
                return COLLIDING


def _approval(op: Any, state: Any) -> Any:
    match op:
        case Step(name=name) if name in GATED and not state.answers.get("approve"):
            return park(Ask("approve", "Approve?"))
        case _:
            return Proceed()


def _call(name: str) -> Any:
    if name.startswith("model:"):
        return ask_llm(name.removeprefix("model:"), "x", dict)
    return step(name, CallTool(name=name, args={}, result_schema=str))


def _flow(shape: Shape) -> Any:
    if shape.gathered:
        return (yield from gather([lambda name=name: _call(name) for name in shape.steps]))
    results = []
    for name in shape.steps:
        results.append((yield from _call(name)))
    return results


def register(app: SqliteApp, shape: Shape, domain: Calls) -> None:
    gate = govern(_approval, gate="review", run_id=shape.run_id)

    @app.register_task(shape.name)
    def task(params: dict[str, Any], ctx: Any) -> Any:
        return _handler(ctx, domain, gate, params).run(lambda: _flow(shape))


def _handler(ctx: Any, domain: Calls, gate: Any, params: dict[str, Any]) -> DurableHandler:
    """The handler each tree builds for a run: the release reads the run's contract from its params
    by hand, through its own `cost.Contract`, and this tree reads its format from them."""
    match getattr(cost, "Contract", None):
        case None:
            return DurableHandler(ctx, domain, ledger=None, op_layers=(gate,), params=params)
        case released:
            contract = released.from_params(params)
            # The release's handler takes `contract`, and its source is what runs here.
            return DurableHandler(ctx, domain, ledger=None, op_layers=(gate,), contract=contract)  # ty: ignore[unknown-argument]
