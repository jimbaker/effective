"""Conformance: BOTH live interpreters of the measured trip vs the Lean model's rows.

The measured trip is formalized once in `formal/lean/Effective/EnforceMeasured.lean`. Its
`conformanceVectors` are `decide`-verified against the Lean transition (`conformance_vectors_hold`,
axiom-free) and emitted to `formal/enforce_vectors.json` by `lake exe enforce_vectors`
(`just formal-vectors`). This test runs BOTH live Python interpreters of the trip against every
row:

  - `effective.fork.enforce_measured` — the in-process VOI-probe transition;
  - `DurableHandler._enforce_measured` — the PRODUCTION durable interpreter (`handlers.absurd`).

The two share ONE transition, so there is one copy of the trip *arithmetic*. This pins each
**driver's realization** against the Lean vectors: the in-process return, and the durable
park-await, `BudgetRefused` raise and state write-back. Implicit placement and realization are
where a driver can drift, for both interpreters.

`(A-money)`: the Lean model uses Nat units; Python uses dollars, mapped by `UNIT = 0.001`. The
reals-to-Nat abstraction is discharged HERE by sampling (running the float transition). NOTE: float
addition is non-associative, so a NEW vector whose *accumulated* `granted` lands exactly on the
ceiling with a non-dyadic sum could flip cleared<->parked without the Lean row noticing — keep
vectors on small integer-unit scales away from an exact accumulated `==` at the ceiling (the
current rows all clear/park by a strict margin).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from effective.budget import Grant, MeasuredBudget
from effective.cost import Contract, Usage
from effective.domain import AskLLM, DomainOp
from effective.fork import Cleared, Exceeded, Parked, enforce_measured
from effective.govern import BudgetRefused
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Segment, compose_key
from effective.ops import Step

UNIT = 0.001  # one Lean Nat unit in dollars (the STEP scale of the measured-fork tests)
RUN = "run-c"
_VECTORS_PATH = Path(__file__).resolve().parents[1] / "formal" / "enforce_vectors.json"
VECTORS = json.loads(_VECTORS_PATH.read_text())

# A normalized outcome both interpreters map onto, so one assertion covers both:
#   ("cleared", granted_dollars, trips) | ("parked", name) | ("refused", spent, ceiling)


# One model, one law, many arms: the overlap across arms IS the design, so the unit-role
# minimize-overlap rule does not apply here.
pytestmark = pytest.mark.conformance


ASK = Step("ask", AskLLM(messages="x", response_schema=str))


def _grants(spec: list[dict]) -> dict[Key, Grant]:
    """The model's positional grant list as the production `budget-grant:{run_id},{i}` map."""
    return {
        # lint: terminal-hole — `i` is the `enumerate` index below, an `int`.
        compose_key(t"budget-grant:{Segment(RUN)},{i}"): (
            Grant(stop=True) if g.get("stop") else Grant(add_dollars=g["add"] * UNIT)
        )
        for i, g in enumerate(spec)
    }


def _budget(row: dict) -> MeasuredBudget:
    return MeasuredBudget(
        overall=row["limit"] * UNIT, run_id=RUN, on_exhaust="fail" if row["fail"] else "park"
    )


def _drive_fork(row: dict) -> tuple:
    """Interpreter A: the in-process `enforce_measured` transition."""
    match enforce_measured(row["meter"] * UNIT, _budget(row), 0.0, 0, _grants(row["grants"])):
        case Cleared(granted=g, trips=t):
            return ("cleared", g, t)
        case Parked(name=n):
            return ("parked", n.stored())  # normalize to the wire name both drivers report
        case Exceeded(spent=s, ceiling=c):
            return ("refused", s, c)


class _Park(Exception):
    """The durable park signal, standing in for the engine's suspend (the run awaits a grant)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _GrantCtx:
    """A `TaskContext` double: `await_event` returns a delivered grant's payload or PARKS (raises
    `_Park`) — enough to drive `DurableHandler._enforce_measured` without a live engine."""

    def __init__(self, delivered: dict[str, Any]) -> None:
        self._delivered = delivered

    def await_event(self, name: Key) -> Any:
        # `.stored()`: `_delivered` doubles for the ENGINE's event store, which is keyed by the
        # wire name — the same text `SdkCtx.await_event` hands the SDK.
        if (wire := name.stored()) in self._delivered:
            return self._delivered[wire]
        raise _Park(wire)

    def step(self, name: Key, thunk: Any) -> Any:  # unused by _enforce_measured
        return thunk()

    def sleep_until(self, when: Any, *, name: Key | None = None) -> None:
        raise NotImplementedError


class _NoDomain:
    """A domain that is never called (`_enforce_measured` touches only ctx + budget)."""

    def run(self, op: DomainOp[Any]) -> Any:
        raise NotImplementedError


def _drive_durable(row: dict) -> tuple:
    """Interpreter B: the PRODUCTION `DurableHandler._enforce_measured` (the one that ships)."""
    delivered = {
        f"budget-grant:{RUN},{i}": (
            {"stop": True} if g.get("stop") else {"add_dollars": g["add"] * UNIT}
        )
        for i, g in enumerate(row["grants"])
    }
    handler = DurableHandler(
        ctx=_GrantCtx(delivered), domain=_NoDomain(), contract=Contract.V1, budget=_budget(row)
    )
    handler._meter = Usage(cost=row["meter"] * UNIT)
    try:
        handler._enforce_measured(ASK)
    except _Park as p:
        return ("parked", p.name)
    except BudgetRefused as e:  # the ceiling is the limit plus every grant
        return ("refused", e.exceeded.spent, e.exceeded.ceiling)
    return ("cleared", handler._grants, handler._trips)


DRIVERS = {"fork": _drive_fork, "durable": _drive_durable}


def _rid(r: dict) -> str:
    return f"l{r['limit']}m{r['meter']}f{int(r['fail'])}g{len(r['grants'])}"


@pytest.mark.parametrize("driver", DRIVERS.values(), ids=list(DRIVERS))
@pytest.mark.parametrize("row", VECTORS, ids=_rid)
def test_interpreter_matches_the_lean_model(row: dict, driver):
    got = driver(row)
    want = row["outcome"]
    match want["kind"]:
        case "cleared":
            assert got[0] == "cleared"
            assert got[2] == want["trips"]  # the trip index re-derives exactly
            assert got[1] == pytest.approx(want["granted"] * UNIT)
        case "parked":
            # the park NAME carries the trip index — the durable park's identity, deterministic
            assert got == ("parked", f"budget-grant:{RUN},{want['trip']}")
        case "refused":
            assert got[0] == "refused"
            assert got[1] == pytest.approx(want["spent"] * UNIT)
            assert got[2] == pytest.approx(want["ceiling"] * UNIT)
        case other:  # pragma: no cover - guards the JSON contract
            pytest.fail(f"unknown outcome kind in the vector file: {other!r}")


class _RecordingGrantCtx(_GrantCtx):
    """`_GrantCtx` that records the awaited grant names (in order) — to pin the trip-index
    sequence across multiple `_enforce_measured` calls on one handler."""

    def __init__(self, delivered: dict[str, Any]) -> None:
        super().__init__(delivered)
        self.awaited: list[str] = []

    def await_event(self, name: Key) -> Any:
        self.awaited.append(name.stored())
        return super().await_event(name)


def test_trip_index_carries_forward_across_enforce_calls():
    # Ratchet pin: `_grants`/`_trips` carry across MULTIPLE
    # `_enforce_measured` calls on ONE handler. The 2nd trip must await the NEXT trip name
    # (`:1`), never re-await the consumed `:0` and never reset to `:0`. Asserts the CLASS —
    # a strictly-monotonic awaited-name sequence + final state == summed grants — not one row.
    from effective.handlers.absurd import DurableHandler

    g0, g1 = 5 * UNIT, 5 * UNIT
    ctx = _RecordingGrantCtx(
        {
            f"budget-grant:{RUN},0": {"add_dollars": g0},
            f"budget-grant:{RUN},1": {"add_dollars": g1},
        }
    )
    budget = MeasuredBudget(overall=3 * UNIT, run_id=RUN, on_exhaust="park")
    h = DurableHandler(ctx=ctx, domain=_NoDomain(), contract=Contract.V1, budget=budget)

    h._meter = Usage(cost=4 * UNIT)  # 0.004 >= 0.003 -> trip :0; +g0 (ceiling 0.008) -> clear
    h._enforce_measured(ASK)
    assert (h._trips, h._grants) == (1, pytest.approx(g0))
    assert ctx.awaited == [f"budget-grant:{RUN},0"]

    h._meter = Usage(cost=9 * UNIT)  # 0.009 >= 0.008 -> trip :1 (carry-forward); +g1 -> clear
    h._enforce_measured(ASK)
    assert (h._trips, h._grants) == (2, pytest.approx(g0 + g1))
    # strictly monotonic, no re-await of the consumed :0, no reset:
    assert ctx.awaited == [f"budget-grant:{RUN},0", f"budget-grant:{RUN},1"]
