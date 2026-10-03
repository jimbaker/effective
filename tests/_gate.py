"""A gate policy under a fixed spend, for tests that pin what a gate does with a spend.

How a run meters is pinned where an engine runs a metered domain (`test_gate_meter.py`,
`test_governed_search.py`, `test_search_conformance.py`); a test here states the spend instead.
The wrapped policy is also outside what a handler refuses when it is built, so such a test may run
a budget gate on a handler that does not meter."""

from dataclasses import replace

from effective.cost import Usage
from effective.govern import GateState, Policy
from effective.ops import WorkflowOp


def at_spend(cost: float, policy: Policy) -> Policy:
    """`policy`, deciding as if the task had spent `cost`."""

    def fixed(op: WorkflowOp, state: GateState):
        return policy(op, replace(state, meter=Usage(cost=cost)))

    return fixed
