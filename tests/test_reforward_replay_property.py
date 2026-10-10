"""A re-forward places what its first forward placed, for generated layer programs.

A program is fixed data: per level, the tries a layer runs, chosen by its parent's try, each try
a run of items that inject a key or forward the op to the level below. Tries differ, so a deeper
re-forward diverges. The program runs once as the first forward of a layer above it and once as
that layer's re-forward of the same op object, which must be handed the same names in order and
count none afresh.

The law reddens under a visit that stops at a diverged forward, a forward never marked diverged,
and a handed name the forwards inside the handing one fail to record.
"""

import random

import pytest

from effective.domain import CallTool
from effective.handlers.absurd import _Reforwards
from effective.keys import Key
from effective.ops import Step

pytestmark = pytest.mark.property

KEYS = [Key.parse("step:a"), Key.parse("step:b"), Key.parse("step:c")]
FORWARD = None


def _op() -> Step:
    """A fresh op object: a re-forward is the same object yielded again."""
    return Step(name="x", op=CallTool(name="x", args={}, result_schema=dict))


def _program(rng: random.Random, depth: int) -> dict[int, list[list[list[Key | None]]]]:
    def tries():
        return [
            [
                rng.choice(KEYS) if rng.random() < 0.55 else FORWARD
                for _ in range(rng.randint(1, 3))
            ]
            for _ in range(rng.randint(1, 3))
        ]

    return {level: [tries() for _ in range(rng.randint(1, 3))] for level in range(depth, 0, -1)}


def _run(reforwards, program, depth, counts, op):
    placed, fresh = [], []

    def place(key):
        if (occurrence := reforwards.replayed(key)) is None:
            occurrence = counts[key] = counts.get(key, 0) + 1
            reforwards.counted(key, occurrence)
            fresh.append((key, occurrence))
        placed.append((key, occurrence))

    def layer(level, parent_try):
        variants = program[level]
        for this_try, items in enumerate(variants[parent_try % len(variants)]):
            for item in items:
                match item:
                    case None if level > 1:
                        reforwards.entering(level - 1, op)
                        layer(level - 1, this_try)
                    case None:
                        reforwards.entering(0, op)
                    case Key() as key:
                        injected = _op()
                        for below in range(level - 1, -1, -1):
                            reforwards.entering(below, injected)
                        place(key)

    reforwards.entering(depth, op)
    layer(depth, 0)
    return placed, fresh


CORNERS = {(2, 1), (3, 1), (3, 2)}
"""Each (program depth, level of a forward that diverged) the seeds must reach."""


def _walks():
    """Each generated program that places a name, run as a first forward and as its re-forward."""
    for depth in (1, 2, 3):
        for seed in range(300):
            program, op, reforwards, counts = (
                _program(random.Random(seed), depth),
                _op(),
                _Reforwards(),
                {},
            )
            reforwards.entering(depth + 1, op)
            first, _ = _run(reforwards, program, depth, counts, op)
            again, fresh = _run(reforwards, program, depth, counts, op)
            if first:
                yield depth, seed, reforwards, first, again, fresh


def test_a_re_forward_is_handed_every_name_its_first_forward_placed():
    for depth, seed, _, first, again, fresh in _walks():
        assert (again, fresh) == (first, []), (depth, seed)


def test_the_seeds_reach_every_level_a_forward_diverges_at():
    reached = {
        (depth, forward.level)
        for depth, _, reforwards, *_ in _walks()
        for forward in reforwards.handed
        if forward.diverged
    }
    assert reached == CORNERS
