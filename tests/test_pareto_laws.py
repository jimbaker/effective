"""The frontier's laws, over every small input a two-objective grid generates.

Dominance is a strict partial order on points with finite values, so on any finite input:

| law          | what holds                                                    |
|--------------|---------------------------------------------------------------|
| nonempty     | a nonempty input has a nonempty frontier                      |
| order-free   | whether a point is on the frontier does not depend on order   |
| justified    | every point left off is dominated by a point kept             |
| undominated  | no point kept is dominated by any point of the input          |

The grid has ties on each axis and repeated points, the corners where a strict comparison and a
non-strict one part.
"""

from itertools import combinations_with_replacement, permutations, product

import pytest

from effective.pareto import Objective, dominates, frontier

pytestmark = pytest.mark.property

OBJECTIVES = [Objective("cost", "min"), Objective("quality", "max")]
GRID = [{"cost": c, "quality": q} for c, q in product(range(3), repeat=2)]
INPUTS = [
    [dict(point) for point in chosen]
    for size in range(1, 5)
    for chosen in combinations_with_replacement(GRID, size)
]


def _kept(points: list[dict]) -> list[bool]:
    """For each input position, whether that point is on the frontier."""
    front = frontier(points, OBJECTIVES)
    return [any(point is kept for kept in front) for point in points]


def test_a_nonempty_input_has_a_nonempty_frontier():
    assert [points for points in INPUTS if not frontier(points, OBJECTIVES)] == []


def test_membership_does_not_depend_on_input_order():
    def differs(points: list[dict]) -> bool:
        kept = _kept(points)
        for order in permutations(range(len(points))):
            reordered = [points[i] for i in order]
            if [kept[i] for i in order] != _kept(reordered):
                return True
        return False

    assert [points for points in INPUTS if differs(points)] == []


def test_every_point_left_off_is_dominated_by_a_point_kept():
    def unjustified(points: list[dict]) -> list[dict]:
        front = frontier(points, OBJECTIVES)
        return [
            point
            for point, kept in zip(points, _kept(points), strict=True)
            if not kept and not any(dominates(q, point, OBJECTIVES) for q in front)
        ]

    assert [points for points in INPUTS if unjustified(points)] == []


def test_no_point_kept_is_dominated_by_an_input_point():
    def better(q: dict, p: dict) -> bool:
        """`q` beats `p`: no worse on either objective, strictly better on one."""
        no_worse = q["cost"] <= p["cost"] and q["quality"] >= p["quality"]
        return no_worse and (q["cost"] < p["cost"] or q["quality"] > p["quality"])

    def beaten(points: list[dict]) -> list[dict]:
        return [p for p in frontier(points, OBJECTIVES) if any(better(q, p) for q in points)]

    assert [points for points in INPUTS if beaten(points)] == []
