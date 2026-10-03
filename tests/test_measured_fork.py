"""The MEASURED (dollars) fork.

An in-process refinement loop under a `MeasuredBudget` parks at the $ ceiling (the measured
trip), and the fork probes a $-grant to make the grant prompt informed in dollars. Pins the trip
parity with production (`_enforce_measured`: the grant name, the ceiling, stop/fail), the DOLLARS
free-prefix proof (the probe's live spend excludes the replayed prefix), and the measured VOI
prompt. Infra-free: the measured trip runs on the in-process driver, since the durable handler
needs Postgres; the DEPLOYED park needs a checkpoint→trace bridge.
"""

from typing import Any

import pytest

from agent.voi import measured_prompt, should_grant
from effective.api import step
from effective.budget import Grant, MeasuredBudget
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.fork import (
    Cleared,
    Exceeded,
    MeasuredTail,
    Parked,
    enforce_measured,
    measured_drive,
)
from effective.govern import BudgetRefused
from effective.keys import Key, Segment, compose_key

STEP = 0.001  # dollars per refinement step (the surrogate's per-call cost)
RUN = "run-b"


def _refine(rounds: int):
    # each round asks the model to improve the answer; the "answer" is the refinement count,
    # threaded so replay re-derives it and the grader can score "how much refinement".
    answer = 0
    for i in range(rounds):
        answer = yield from step(
            f"refine:{i}", AskLLM(messages=[{"answer": answer}], response_schema=int)
        )
    return answer


def _program():
    return (yield from _refine(10))


class RefineDomain:
    """A metered surrogate: each step returns `prev + 1` (one more refinement) at `STEP` cost."""

    def run_metered(self, op: Any) -> tuple[int, Usage]:  # a test double, duck-typed
        return op.messages[0]["answer"] + 1, Usage(cost=STEP)


def _park_budget() -> MeasuredBudget:
    return MeasuredBudget(overall=3 * STEP, run_id=RUN, on_exhaust="park")


def _base() -> MeasuredTail:
    """Run to the measured park: 3 steps fit under the $0.003 ceiling, the 4th trips."""
    return measured_drive(_program, _park_budget(), RefineDomain(), grants={})


def test_base_run_parks_at_the_measured_ceiling():
    base = _base()
    assert base.tripped_at == Key.parse(
        f"budget-grant:{RUN},0"
    )  # the production grant name (trip_n=0)
    assert [e.key.stored() for e in base.trace] == [
        "step;refine:0",
        "step;refine:1",
        "step;refine:2",
    ]
    assert base.result is None  # parked, not completed
    assert base.usage.cost == pytest.approx(3 * STEP)  # spend at the ceiling
    assert base.trace[-1].result == 3  # the current best: 3 refinements


def test_fork_replays_the_prefix_free_the_dollars_payoff():
    prefix = _base().trace
    # fork with a $0.001 grant (one more step), then it re-parks: the probe is ONE live step.
    trip = compose_key(t"budget-grant:{Segment(RUN)},0")
    tail = measured_drive(
        _program,
        _park_budget(),
        RefineDomain(),
        {trip: Grant(add_dollars=STEP)},
        recorded=prefix,
    )
    # re-parked after one granted step
    assert tail.tripped_at == Key.parse(f"budget-grant:{RUN},1")
    assert tail.trace[-1].result == 4  # one more refinement than the base's 3
    # the dollars free-prefix proof: total meter includes the prefix, live spend does NOT
    assert tail.usage.cost == pytest.approx(4 * STEP)  # prefix (3) + probe (1)
    assert tail.live_usage.cost == pytest.approx(STEP)  # the probe re-paid ONLY its one step


def test_measured_grant_prompt_prices_the_marginal_in_dollars():
    prefix = _base().trace
    prompt = measured_prompt(
        _program, prefix, _park_budget(), RefineDomain(), grader=float, probe_dollars=STEP
    )
    assert prompt.current_best.cost == 0.0  # the recorded current best is free
    assert prompt.current_best.quality == 3.0  # 3 refinements
    assert prompt.probe.cost == pytest.approx(STEP)  # the probe's live dollars only
    assert prompt.probe.quality == 4.0  # one more refinement
    assert prompt.marginal_quality == pytest.approx(1.0)
    assert prompt.marginal_cost == pytest.approx(STEP)
    assert should_grant(prompt, threshold=0.5) is True  # refinement helps -> grant


def _at_trip_1() -> MeasuredTail:
    """Grant trip 0, re-park at trip 1 — the multi-trip predecessor a multi-hop task reaches."""
    prefix0 = _base().trace
    return measured_drive(
        _program,
        _park_budget(),
        RefineDomain(),
        {compose_key(t"budget-grant:{Segment(RUN)},0"): Grant(add_dollars=STEP)},
        recorded=prefix0,
    )


def test_measured_prompt_at_a_later_trip_carries_prior_grants_and_names_the_pending_trip():
    """A multi-hop task parks repeatedly; the informed prompt at trip K must name `…:{K}` AND
    carry grants 0..K-1 — `measured_drive` re-runs `enforce_measured` on every replayed AskLLM and
    re-parks at trip 0 otherwise. A hardcoded `:0` gives a phantom zero marginal at every park
    past the first (probe re-parks without drilling); `delivered_grants` carries the history."""
    at_trip_1 = _at_trip_1()
    assert at_trip_1.tripped_at == Key.parse(f"budget-grant:{RUN},1")  # the second park
    delivered = {compose_key(t"budget-grant:{Segment(RUN)},0"): Grant(add_dollars=STEP)}
    prompt = measured_prompt(
        _program,
        at_trip_1.trace,
        _park_budget(),
        RefineDomain(),
        grader=float,
        probe_dollars=STEP,
        delivered_grants=delivered,
    )
    assert prompt.current_best.quality == 4.0  # the trip-1 current best (4 refinements)
    assert prompt.probe.quality == 5.0  # the probe drilled ONE more — not a phantom 4
    assert prompt.marginal_quality == pytest.approx(1.0)
    assert prompt.probe.cost == pytest.approx(STEP)  # only the single live probe step


def test_trip_parity_stop_grant_refuses():
    prefix = _base().trace
    trip = compose_key(t"budget-grant:{Segment(RUN)},0")
    with pytest.raises(BudgetRefused):
        measured_drive(
            _program, _park_budget(), RefineDomain(), {trip: Grant(stop=True)}, recorded=prefix
        )


def test_trip_parity_on_exhaust_fail_raises_instead_of_parking():
    fail_budget = MeasuredBudget(overall=3 * STEP, run_id=RUN, on_exhaust="fail")
    with pytest.raises(BudgetRefused):
        measured_drive(_program, fail_budget, RefineDomain(), grants={})


def test_a_generous_grant_runs_multiple_steps_before_re_parking():
    prefix = _base().trace
    trip = compose_key(t"budget-grant:{Segment(RUN)},0")
    tail = measured_drive(
        _program,
        _park_budget(),
        RefineDomain(),
        {trip: Grant(add_dollars=3 * STEP)},  # room for 3 more steps
        recorded=prefix,
    )
    assert tail.tripped_at == Key.parse(f"budget-grant:{RUN},1")
    assert tail.trace[-1].result == 6  # 3 recorded + 3 granted
    assert tail.live_usage.cost == pytest.approx(3 * STEP)  # three live steps
    assert tail.usage.cost == pytest.approx(6 * STEP)  # prefix + granted


# --- trip PLACEMENT (only at a metered AskLLM) + the transition -------------------------


def _mixed():
    a = yield from step("ask:a", AskLLM(messages=[{"answer": 0}], response_schema=int))
    yield from step("tool:t", CallTool(name="t", result_schema=str))
    b = yield from step("ask:b", AskLLM(messages=[{"answer": a}], response_schema=int))
    return b


def test_trip_fires_only_at_askllm_not_at_a_calltool_step():
    # SELF-TEST arm: drives `measured_drive` alone against the placement the durable handler
    # uses (trip only at a metered AskLLM, so a CallTool AFTER the crossing runs and the run parks
    # before the NEXT ask, never before the tool). The CROSS-IMPLEMENTATION authority for this
    # placement (durable run → bridge → fork, so it can catch a real drift) is
    # `test_durable_voi_bridge.py::test_P3b_calltool_interleave_parity` and its adversarial
    # sibling `::test_L1_envelope_shaped_tool_result_is_not_mis_decoded`. This arm is the fast,
    # infra-free pin of the same fact.
    domain = MeteredInterpreter(llm=lambda op: (1, Usage(cost=STEP)), tools=lambda op: "done")
    budget = MeasuredBudget(overall=STEP, run_id=RUN, on_exhaust="park")  # crossed after ask:a
    tail = measured_drive(_mixed, budget, domain, grants={})
    # ran the tool, THEN parked
    assert [e.key.stored() for e in tail.trace] == ["step;ask:a", "step;tool:t"]
    assert tail.tripped_at == Key.parse(f"budget-grant:{RUN},0")


def test_enforce_measured_transition_classifies():
    # The trip as a directly-testable transition: Cleared | Parked | Exceeded.
    park = MeasuredBudget(overall=0.003, run_id=RUN, on_exhaust="park")
    fail = MeasuredBudget(overall=0.003, run_id=RUN, on_exhaust="fail")
    grant0 = {compose_key(t"budget-grant:{Segment(RUN)},0"): Grant(add_dollars=0.005)}

    assert enforce_measured(0.002, park, 0.0, 0, {}) == Cleared(0.0, 0)  # under the ceiling
    assert enforce_measured(0.003, park, 0.0, 0, {}) == Parked(
        compose_key(t"budget-grant:{Segment(RUN)},0")
    )  # no grant
    assert enforce_measured(0.003, park, 0.0, 0, grant0) == Cleared(0.005, 1)  # refill advances
    assert enforce_measured(  # stop grant -> refuse
        0.003, park, 0.0, 0, {compose_key(t"budget-grant:{Segment(RUN)},0"): Grant(stop=True)}
    ) == Exceeded(0.003, 0.003)
    # fail FIRST: a fail budget refuses BEFORE consulting grants
    assert enforce_measured(0.003, fail, 0.0, 0, grant0) == Exceeded(0.003, 0.003)


def test_agent_fork_reexports_the_budget_transition_symbols():
    # Back-compat pin: the trip transition lives in `effective.budget`, and `effective.fork`
    # re-exports the SAME objects, `TripOutcome` included.
    import effective.fork as fork
    from effective import budget

    for name in ("enforce_measured", "Cleared", "Parked", "Exceeded", "TripOutcome"):
        assert getattr(fork, name) is getattr(budget, name), name
