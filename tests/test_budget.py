"""Unit tests for the structural vector budget (build-now cut).

Pure and infra-free (no Postgres, no handler): these pin the *type contract* —
immutable threading, the ``ge=0`` un-bounding guard, and the exhaustion check —
before any combinator consumes it.
"""

import pytest
from pydantic import ValidationError

from effective.budget import (
    Budget,
    Cleared,
    Exceeded,
    Grant,
    MeasuredBudget,
    Parked,
    enforce_generation,
    enforce_measured,
)
from effective.keys import Key


def test_unbounded_budget_is_a_no_op():
    b = Budget()
    assert b.depth is None
    assert not b.depth_exhausted()
    # descend_one on an unbounded axis threads the same value through, unchanged.
    assert b.descend_one() is b or b.descend_one().depth is None


def test_descend_one_is_immutable_and_decrements():
    parent = Budget(depth=3)
    child = parent.descend_one()
    # The parent keeps its own value — no shared counter (the race-free property).
    assert parent.depth == 3
    assert child.depth == 2
    # Frozen dataclass: the copy is a distinct object.
    assert child is not parent


def test_depth_exhaustion_gate():
    assert not Budget(depth=1).depth_exhausted()
    assert Budget(depth=0).depth_exhausted()
    # Over-consumption (<= 0) still reads as exhausted, never "un-exhausts".
    assert Budget(depth=-1).depth_exhausted()


def test_descend_to_exhaustion():
    b = Budget(depth=2)
    b = b.descend_one()  # depth 1
    assert not b.depth_exhausted()
    b = b.descend_one()  # depth 0
    assert b.depth_exhausted()


def test_grant_defaults_are_a_zero_no_grant():
    g = Grant()
    assert g.add_depth == 0
    assert g.stop is False


def test_grant_rejects_negative_additive_field():
    # ge=0 is the un-bounding guard: a negative grant must fail at the schema
    # boundary, not silently skip the exhaustion gate — a negative refill would un-bound the axis.
    with pytest.raises(ValidationError):
        Grant(add_depth=-1)


def test_grant_stop_is_expressible():
    g = Grant(add_depth=0, stop=True)
    assert g.stop is True


# --- two ceilings: overall vs per-generation ----------------------------------------------


def test_a_budget_that_limits_nothing_is_refused_at_assembly():
    """`govern()`-with-no-policies shape: a misconfiguration, not a default. Finding out at the
    first metered call — or never — is worse than finding out at construction."""
    with pytest.raises(ValueError, match="needs `overall`"):
        MeasuredBudget(run_id="r1")


def test_a_per_generation_allowance_above_the_overall_cap_is_refused():
    """It can never bind: the run's cap trips first, every time. So it reads like a limit and
    is a no-op — the class of thing that looks like governance and is not."""
    with pytest.raises(ValueError, match="can never bind"):
        MeasuredBudget(run_id="r1", overall=10.0, per_generation=20.0)


def test_the_budget_is_keyword_only_so_a_positional_build_cannot_mis_assign():
    """Two optional ceilings make positional construction a SILENT mis-assignment — measured:
    the conformance harness passed `MeasuredBudget(budget_limit, rid, on_exhaust)` and the limit
    landed in `run_id`, so every metered task ran with no ceiling at all."""
    # Both diagnostics ARE the assertion: `ty` refuses the call statically and CPython refuses
    # it at runtime, which is the belt-and-braces this class of mis-assignment earns.
    with pytest.raises(TypeError):
        MeasuredBudget(0.5, "r1")  # ty: ignore[too-many-positional-arguments, missing-argument]


def test_each_ceiling_is_enforced_independently():
    """Only the one that is set binds; the other returns `Cleared` untouched, so a budget may
    carry either or both."""
    over_only = MeasuredBudget(run_id="r1", overall=1.0, on_exhaust="fail")
    per_only = MeasuredBudget(run_id="r1", per_generation=1.0, on_exhaust="fail")

    assert isinstance(enforce_measured(2.0, over_only, 0.0, 0, {}), Exceeded)
    assert isinstance(enforce_generation(2.0, over_only, 0, 0.0, 0, {}), Cleared)

    assert isinstance(enforce_generation(2.0, per_only, 0, 0.0, 0, {}), Exceeded)
    assert isinstance(enforce_measured(2.0, per_only, 0.0, 0, {}), Cleared)


def test_the_two_grants_are_different_questions_and_different_names():
    """The park name says which ceiling tripped, so whoever answers knows what they are
    authorizing — and the two families cannot collide."""
    budget = MeasuredBudget(run_id="r1", overall=10.0, per_generation=1.0, on_exhaust="park")

    overall = enforce_measured(20.0, budget, 0.0, 0, {})
    generation = enforce_generation(2.0, budget, 3, 0.0, 0, {})

    assert isinstance(overall, Parked)
    assert isinstance(generation, Parked)
    assert overall.name.stored() == "budget-grant:r1,0"
    assert generation.name.stored() == "generation-grant:r1,3,0"


def test_the_generation_grant_name_DISTINGUISHES_generations():
    """The defect this ceiling exists to fix. `budget-grant:{run_id},{trip}` is byte-identical
    across generations — stable run id by design, trip counter restarting per task — so
    generation n's human grant answered n+1 instantly, with no record they shared it.

    Asserted as an inequality over generations rather than an equality against a literal, so it
    keeps meaning what it says if the shape changes."""
    budget = MeasuredBudget(run_id="r1", per_generation=1.0, on_exhaust="park")
    names = set()
    for generation in range(5):
        parked = enforce_generation(2.0, budget, generation, 0.0, 0, {})
        assert isinstance(parked, Parked)  # narrowed, not suppressed
        names.add(parked.name)
    assert len(names) == 5, names

    # The overall grant deliberately does NOT vary by generation — it addresses the RUN, which
    # is the whole distinction. Its trip counter is what must become chain-wide (build item 4).
    run_wide = enforce_measured(
        20.0, MeasuredBudget(run_id="r1", overall=1.0, on_exhaust="park"), 0.0, 0, {}
    )
    assert isinstance(run_wide, Parked)
    assert run_wide.name.stored() == "budget-grant:r1,0"


def test_a_generation_grant_refills_only_that_generation():
    """A grant answers the question it was asked: topping up generation 3 clears generation 3."""
    budget = MeasuredBudget(run_id="r1", per_generation=1.0, on_exhaust="park")
    grants = {Key.parse("generation-grant:r1,3,0"): Grant(add_dollars=5.0)}

    cleared = enforce_generation(2.0, budget, 3, 0.0, 0, grants)
    assert isinstance(cleared, Cleared)
    assert cleared.granted == 5.0

    # ...and does nothing for generation 4, which must ask for itself.
    assert isinstance(enforce_generation(2.0, budget, 4, 0.0, 0, grants), Parked)


def test_a_stop_grant_ends_the_trip_even_when_it_carries_dollars():
    """A decision-table trap: `case Grant(add_dollars=0.0)` reads like "a zero grant stops" and
    silently misses `stop=True` carrying a positive amount. That grant would fall through to the
    refill arm and hand the run money its answerer said not to spend; this test is the one that
    reddens when the guard is swapped for `case Grant(add_dollars=0.0)`.

    A negative `add_dollars` is unreachable rather than untested (`Field(ge=0.0)` refuses it at
    construction), so the live half is `stop` alone."""
    budget = MeasuredBudget(run_id="r1", overall=1.0, on_exhaust="park")
    park = Key.parse("budget-grant:r1,0")

    stop_with_money = enforce_measured(
        2.0, budget, 0.0, 0, {park: Grant(add_dollars=5.0, stop=True)}
    )
    assert isinstance(stop_with_money, Exceeded), "a `stop` grant must not refill"

    zero = enforce_measured(2.0, budget, 0.0, 0, {park: Grant(add_dollars=0.0)})
    assert isinstance(zero, Exceeded)

    refill = enforce_measured(2.0, budget, 0.0, 0, {park: Grant(add_dollars=5.0)})
    assert isinstance(refill, Cleared), "a real grant still refills — the arm order matters"

    with pytest.raises(ValidationError):
        Grant(add_dollars=-1.0)  # the other half of the guard is unreachable by construction


def test_a_MEASURED_budget_PARKS_by_default():
    """The default is `"park"`, not `"fail"`.

    Pinned because the flip broke NOTHING — the whole suite stayed green — which means no test
    was exercising exhaustion at the default, so the behaviour change had no guard at all. A
    silent default is exactly the kind that flips back.

    Why it is the right default: a trip parks on `budget-grant:{run_id},{trip}`, `parked.py` is
    unfiltered so a dashboard sees the park and its name, and a grantor answers with more dollars
    or `stop` — so the run RESUMES. Raising ends it, and inside a fork it ended it badly (the
    child crashed, was retried to death, and the parent hung on `fork-done:` with no reason
    recorded). `"fail"` stays right for an unattended batch and stays available.

    **`Budget` — the STRUCTURAL sibling — deliberately did not flip**, and that is asserted here
    rather than left to a docstring: it is enforced across the in-process subagent boundary
    (`LocalCtx`), which cannot park at all, so `"fail"` is the only option there.
    """
    from effective.budget import Budget, MeasuredBudget

    assert MeasuredBudget(run_id="r1", overall=1.0).on_exhaust == "park"
    assert Budget().on_exhaust == "fail"
    # still selectable, because an unattended batch wants the ceiling to be fatal
    assert MeasuredBudget(run_id="r1", overall=1.0, on_exhaust="fail").on_exhaust == "fail"
