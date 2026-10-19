"""Pareto frontier over multi-objective agent configs.

Picking a model/backend trades off cost, quality, and response time — there is
no single scalar to maximize. The right object is the *non-dominated set*: the
configs that nothing else beats on every axis at once. Anything dominated is
strictly wasteful ("low-hanging fruit" you should never pick); the frontier is
the menu of real choices, and *which* frontier point you take is a preference,
revealed by what you actually run.

A config is a plain ``dict`` of objective key -> value plus any labels; an
``Objective`` says which key to read and whether more is better.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, assert_never, get_args

type Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Objective:
    key: str
    direction: Direction

    def __post_init__(self) -> None:
        if self.direction not in get_args(Direction.__value__):
            raise ValueError(f"objective {self.key!r}: direction {self.direction!r} is min or max")

    def better_or_equal(self, x: float, y: float) -> bool:
        match self.direction:
            case "min":
                return x <= y
            case "max":
                return x >= y
            case unreachable:
                assert_never(unreachable)

    def strictly_better(self, x: float, y: float) -> bool:
        match self.direction:
            case "min":
                return x < y
            case "max":
                return x > y
            case unreachable:
                assert_never(unreachable)


def dominates(a: dict[str, Any], b: dict[str, Any], objectives: Sequence[Objective]) -> bool:
    """True if ``a`` Pareto-dominates ``b``: at least as good on every objective
    and strictly better on at least one."""
    ge_all = all(o.better_or_equal(a[o.key], b[o.key]) for o in objectives)
    gt_any = any(o.strictly_better(a[o.key], b[o.key]) for o in objectives)
    return ge_all and gt_any


def frontier(
    points: Sequence[dict[str, Any]], objectives: Sequence[Objective]
) -> list[dict[str, Any]]:
    """The non-dominated subset of ``points`` (order preserved). Each point carries every
    objective's key with a finite value; dominance is a strict partial order only there."""
    for position, point in enumerate(points):
        for objective in objectives:
            if objective.key not in point:
                raise ValueError(f"objective {objective.key!r} is missing from point {position}")
            if not math.isfinite(point[objective.key]):
                raise ValueError(
                    f"objective {objective.key!r} is {point[objective.key]} at point {position}"
                )
    return [
        p for p in points if not any(dominates(q, p, objectives) for q in points if q is not p)
    ]


def label_frontier(
    points: Sequence[dict[str, Any]], objectives: Sequence[Objective]
) -> list[dict[str, Any]]:
    """Return copies of ``points`` with an ``on_frontier`` bool added."""
    front = [id(p) for p in frontier(points, objectives)]
    return [{**p, "on_frontier": id(p) in front} for p in points]
