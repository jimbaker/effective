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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Objective:
    key: str
    direction: str  # "min" or "max"

    def better_or_equal(self, x: float, y: float) -> bool:
        return x <= y if self.direction == "min" else x >= y

    def strictly_better(self, x: float, y: float) -> bool:
        return x < y if self.direction == "min" else x > y


def dominates(a: dict[str, Any], b: dict[str, Any], objectives: Sequence[Objective]) -> bool:
    """True if ``a`` Pareto-dominates ``b``: at least as good on every objective
    and strictly better on at least one."""
    ge_all = all(o.better_or_equal(a[o.key], b[o.key]) for o in objectives)
    gt_any = any(o.strictly_better(a[o.key], b[o.key]) for o in objectives)
    return ge_all and gt_any


def frontier(
    points: Sequence[dict[str, Any]], objectives: Sequence[Objective]
) -> list[dict[str, Any]]:
    """The non-dominated subset of ``points`` (order preserved)."""
    return [
        p for p in points if not any(dominates(q, p, objectives) for q in points if q is not p)
    ]


def label_frontier(
    points: Sequence[dict[str, Any]], objectives: Sequence[Objective]
) -> list[dict[str, Any]]:
    """Return copies of ``points`` with an ``on_frontier`` bool added."""
    front = [id(p) for p in frontier(points, objectives)]
    return [{**p, "on_frontier": id(p) in front} for p in points]
