"""A hook in Effective is a layer: generator middleware around `yield op`.

Run it with `uv run python examples/hooks_as_layers.py`. The code before `yield op` runs before
each call and the code after it runs on the result, so one function holds both hook positions
and the state between them.
"""

from collections.abc import Generator
from typing import Any

from first_workflow import House, run_durably

from effective.layers import op_layer
from effective.ops import Step, WorkflowOp


def describe(op: WorkflowOp) -> str:
    match op:
        case Step(name=name):
            return name
        case _:
            return type(op).__name__


@op_layer
def audit(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
    print("before a call:", describe(op))
    result = yield op
    print("after a call: ", describe(op), "->", result)
    return result


if __name__ == "__main__":
    done, _ = run_durably(House(), "make it warmer", layers=(audit,))
    print("result:", done.result)
