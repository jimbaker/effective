"""Pareto frontier over agent configs (red-green spec).

Choosing a model/backend is multi-objective: cost (min), quality (max), response
time (min). There's no single scalar winner; we want the non-dominated set (the
frontier) and to discard anything another config beats on every axis — "no low-
hanging fruit." A human's actual choice is then a revealed-preference point on
the frontier.
"""

from effective.pareto import Objective, dominates, frontier

# cost & latency minimized, quality maximized
OBJ = [
    Objective("cost", "min"),
    Objective("quality", "max"),
    Objective("latency", "min"),
]


def test_dominance_requires_ge_all_and_gt_one():
    a = {"cost": 0.0, "quality": 1.0, "latency": 3.0}
    # a beats `worse` on cost+latency, ties quality
    worse = {"cost": 0.01, "quality": 1.0, "latency": 3.5}
    assert dominates(a, worse, OBJ)
    assert not dominates(worse, a, OBJ)


def test_no_domination_on_a_genuine_tradeoff():
    cheap_fast_weaker = {"cost": 0.0, "quality": 0.7, "latency": 3.0}
    pricey_better = {"cost": 0.01, "quality": 1.0, "latency": 3.0}
    # neither dominates: one wins quality, the other wins cost
    assert not dominates(cheap_fast_weaker, pricey_better, OBJ)
    assert not dominates(pricey_better, cheap_fast_weaker, OBJ)


def test_identical_points_do_not_dominate():
    p = {"cost": 0.0, "quality": 1.0, "latency": 3.0}
    q = dict(p)
    assert not dominates(p, q, OBJ)
    assert not dominates(q, p, OBJ)


def test_frontier_drops_dominated_keeps_tradeoffs():
    points = [
        {"name": "local", "cost": 0.0, "quality": 1.0, "latency": 3.07},
        {"name": "cloud", "cost": 0.00012, "quality": 1.0, "latency": 3.46},  # dominated by local
        {"name": "big", "cost": 0.02, "quality": 1.0, "latency": 1.0},  # non-dominated: fastest
        {"name": "weak", "cost": 0.0, "quality": 0.5, "latency": 5.0},  # dominated by local
    ]
    names = {p["name"] for p in frontier(points, OBJ)}
    assert names == {"local", "big"}  # cloud and weak are dominated
